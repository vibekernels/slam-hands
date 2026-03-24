/*
 * Full GPU pipeline: NVDEC decode → CUDA tonemap → NVENC encode.
 *
 * Keeps frames on GPU throughout, eliminating CPU zscale tonemapping.
 * Uses libavformat/libavcodec for container handling and codec init,
 * with a custom CUDA kernel for HDR→SDR tonemapping between decode and encode.
 *
 * Usage:
 *   ./gpu_convert input.mov output.ivf [bitrate_kbps]
 *   ffmpeg -i output.ivf -c copy -movflags +faststart output.mp4
 *
 * Build:
 *   nvcc -O3 -o gpu_convert gpu_convert.cu \
 *     $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
 *     -lcuda -lnvcuvid -lnvidia-encode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* msg; cuGetErrorString(err, &msg); \
        fprintf(stderr, "CU error at %s:%d: %s\n", __FILE__, __LINE__, msg); exit(1); \
    } \
} while(0)

/* av_err2str uses compound literals which nvcc (C++ mode) doesn't like */
static char av_err_buf[AV_ERROR_MAX_STRING_SIZE];
#undef av_err2str
#define av_err2str(errnum) av_make_error_string(av_err_buf, AV_ERROR_MAX_STRING_SIZE, errnum)

/*
 * CUDA tonemap kernel: P010/P016 (10-bit, BT.2020/HLG) → NV12 (8-bit, BT.709)
 *
 * P010 layout: Y plane (uint16, 10-bit in upper bits), UV interleaved plane (uint16)
 * NV12 layout: Y plane (uint8), UV interleaved plane (uint8)
 *
 * Each thread processes one luma pixel. Chroma is processed by threads
 * at even (x,y) positions.
 */

/* HLG EOTF: electrical signal → linear light */
__device__ __forceinline__ float hlg_eotf(float x) {
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    if (x <= 0.5f)
        return x * x / 3.0f;
    else
        return (expf((x - c) / a) + b) / 12.0f;
}

/* BT.1886 inverse EOTF: linear light → electrical signal
 * zscale/zimg uses the pure power law V = L^(1/2.4) for BT.709 output,
 * matching the BT.1886 display standard, rather than the BT.709 OETF
 * which has a linear segment for small values. */
__device__ __forceinline__ float bt1886_inverse_eotf(float x) {
    return __powf(fmaxf(x, 0.0f), 1.0f / 2.4f);
}

/*
 * Unified tonemap kernel: one thread per luma pixel.
 * Converts through full RGB pipeline for correct color science:
 *   P010 YCbCr (BT.2020/HLG) → RGB → EOTF → gamut map → OETF → NV12 YCbCr (BT.709)
 *
 * Each thread:
 *   - Reads its Y value and nearest chroma (UV)
 *   - Converts to RGB, tonemaps, converts back
 *   - Writes Y always, writes UV only for top-left of each 2×2 block
 */
__global__ void tonemap_full(
    const uint16_t* __restrict__ src_y,   int src_pitch_y,
    const uint16_t* __restrict__ src_uv,  int src_pitch_uv,
    uint8_t* __restrict__ dst_y,          int dst_pitch_y,
    uint8_t* __restrict__ dst_uv,         int dst_pitch_uv,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    /* Read 10-bit luma (P010: 10-bit value in upper bits of uint16) */
    float Y = (float)(src_y[y * (src_pitch_y / 2) + x] >> 6) / 1023.0f;

    /* Read 10-bit chroma (nearest neighbor, half-res UV plane) */
    int cx = x >> 1, cy = y >> 1;
    int uv_idx = cy * (src_pitch_uv / 2) + cx * 2;
    float Cb = (float)(src_uv[uv_idx] >> 6) / 1023.0f;
    float Cr = (float)(src_uv[uv_idx + 1] >> 6) / 1023.0f;

    /* 10-bit limited range → normalized */
    Y  = (Y  - 64.0f / 1023.0f) * (1023.0f / 876.0f);
    Cb = (Cb - 512.0f / 1023.0f) * (1023.0f / 896.0f);
    Cr = (Cr - 512.0f / 1023.0f) * (1023.0f / 896.0f);

    /* BT.2020 NCL YCbCr → RGB */
    float R = Y + 1.4746f * Cr;
    float G = Y - 0.16455f * Cb - 0.57135f * Cr;
    float B = Y + 1.8814f * Cb;
    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    /* HLG inverse OETF: electrical → scene-referred linear light */
    R = hlg_eotf(R); G = hlg_eotf(G); B = hlg_eotf(B);

    /* HLG OOTF + npl scaling: scene linear → display linear
     * Luma-weighted formula (matches zimg's InverseAribB67OperationC):
     *   Ys = dot(scene, [0.2627, 0.6780, 0.0593])  (BT.2020 luma)
     *   display[i] = scene[i] * pow(Ys, 0.2) * (1000 / npl)
     * For npl=100: display[i] = scene[i] * pow(Ys, 0.2) * 10
     * Preserves chromaticity (same scale for all channels). */
    float Ys = 0.2627f * R + 0.6780f * G + 0.0593f * B;
    float ootf_scale = (Ys > 0.0f) ? __powf(Ys, 0.2f) * 10.0f : 0.0f;
    R *= ootf_scale;
    G *= ootf_scale;
    B *= ootf_scale;

    /* BT.2020 → BT.709 gamut mapping matrix */
    float R709 = fmaxf( 1.6605f * R - 0.5877f * G - 0.0728f * B, 0.0f);
    float G709 = fmaxf(-0.1246f * R + 1.1329f * G - 0.0083f * B, 0.0f);
    float B709 = fmaxf(-0.0182f * R - 0.1006f * G + 1.1187f * B, 0.0f);

    /* BT.1886 inverse EOTF: linear → electrical (pure power law) */
    R709 = bt1886_inverse_eotf(R709);
    G709 = bt1886_inverse_eotf(G709);
    B709 = bt1886_inverse_eotf(B709);

    /* RGB → BT.709 YCbCr */
    float Y709  =  0.2126f * R709 + 0.7152f * G709 + 0.0722f * B709;

    /* Write 8-bit luma (limited range) */
    dst_y[y * dst_pitch_y + x] = (uint8_t)fminf(fmaxf(Y709 * 219.0f + 16.5f, 0.0f), 255.0f);

    /* Write chroma only for top-left pixel of each 2×2 block */
    if ((x & 1) == 0 && (y & 1) == 0) {
        float Cb709 = (B709 - Y709) / 1.8556f;
        float Cr709 = (R709 - Y709) / 1.5748f;
        int out_uv = cy * dst_pitch_uv + cx * 2;
        dst_uv[out_uv]     = (uint8_t)fminf(fmaxf(Cb709 * 224.0f + 128.5f, 0.0f), 255.0f);
        dst_uv[out_uv + 1] = (uint8_t)fminf(fmaxf(Cr709 * 224.0f + 128.5f, 0.0f), 255.0f);
    }
}


/* Find the best CUDA hw device type for decoding */
static enum AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_CUDA;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output.mp4> [bitrate_kbps]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int bitrate_kbps = argc > 3 ? atoi(argv[3]) : 2550;

    int ret;
    AVFormatContext* ifmt_ctx = NULL;
    AVFormatContext* ofmt_ctx = NULL;
    AVCodecContext* dec_ctx = NULL;
    AVCodecContext* enc_ctx = NULL;
    AVBufferRef* hw_device_ctx = NULL;
    AVPacket* pkt = NULL;
    AVFrame* hw_frame = NULL;
    int video_stream_idx = -1;
    int64_t frame_count = 0;

    /* Open input */
    ret = avformat_open_input(&ifmt_ctx, input_path, NULL, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot open input: %s\n", av_err2str(ret)); return 1; }
    ret = avformat_find_stream_info(ifmt_ctx, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot find stream info\n"); return 1; }

    /* Find video stream */
    for (unsigned i = 0; i < ifmt_ctx->nb_streams; i++) {
        if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx < 0) { fprintf(stderr, "No video stream\n"); return 1; }

    AVStream* in_stream = ifmt_ctx->streams[video_stream_idx];
    int width = in_stream->codecpar->width;
    int height = in_stream->codecpar->height;

    fprintf(stderr, "Input: %dx%d, codec=%s\n", width, height,
            avcodec_get_name(in_stream->codecpar->codec_id));

    /* Create CUDA device */
    ret = av_hwdevice_ctx_create(&hw_device_ctx, hw_type, NULL, NULL, 0);
    if (ret < 0) { fprintf(stderr, "Cannot create CUDA device: %s\n", av_err2str(ret)); return 1; }

    /* Setup decoder with CUDA hwaccel */
    const AVCodec* decoder = avcodec_find_decoder(in_stream->codecpar->codec_id);
    if (!decoder) { fprintf(stderr, "Decoder not found\n"); return 1; }

    dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, in_stream->codecpar);
    dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    /* Request CUDA output format */
    dec_ctx->pix_fmt = AV_PIX_FMT_CUDA;

    ret = avcodec_open2(dec_ctx, decoder, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot open decoder: %s\n", av_err2str(ret)); return 1; }

    /* Setup NVENC AV1 encoder */
    const AVCodec* encoder = avcodec_find_encoder_by_name("av1_nvenc");
    if (!encoder) { fprintf(stderr, "av1_nvenc not found\n"); return 1; }

    enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = AV_PIX_FMT_CUDA;
    enc_ctx->time_base = in_stream->time_base;
    enc_ctx->framerate = av_guess_frame_rate(ifmt_ctx, in_stream, NULL);
    enc_ctx->color_range = AVCOL_RANGE_MPEG;  /* limited/TV range */
    enc_ctx->colorspace = AVCOL_SPC_BT709;
    enc_ctx->color_trc = AVCOL_TRC_BT709;
    enc_ctx->color_primaries = AVCOL_PRI_BT709;
    enc_ctx->gop_size = 2;
    enc_ctx->max_b_frames = 0;
    enc_ctx->bit_rate = (int64_t)bitrate_kbps * 1000;
    enc_ctx->rc_max_rate = (int64_t)(bitrate_kbps * 1.5) * 1000;
    enc_ctx->rc_buffer_size = bitrate_kbps * 2000;
    enc_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    /* NVENC speed tuning: fastest preset, low-latency, zero buffering delay */
    av_opt_set(enc_ctx->priv_data, "preset", "p1", 0);
    av_opt_set(enc_ctx->priv_data, "tune", "ll", 0);
    av_opt_set(enc_ctx->priv_data, "delay", "0", 0);
    av_opt_set(enc_ctx->priv_data, "zerolatency", "1", 0);

    /* Create hardware frames context for encoder input */
    #define NUM_BUFS 3
    AVBufferRef* enc_hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    AVHWFramesContext* enc_hw_frames = (AVHWFramesContext*)enc_hw_frames_ref->data;
    enc_hw_frames->format = AV_PIX_FMT_CUDA;
    enc_hw_frames->sw_format = AV_PIX_FMT_NV12;  /* 8-bit output after tonemap */
    enc_hw_frames->width = width;
    enc_hw_frames->height = height;
    enc_hw_frames->initial_pool_size = NUM_BUFS + 4;
    ret = av_hwframe_ctx_init(enc_hw_frames_ref);
    if (ret < 0) { fprintf(stderr, "Cannot init enc hw frames: %s\n", av_err2str(ret)); return 1; }
    enc_ctx->hw_frames_ctx = av_buffer_ref(enc_hw_frames_ref);

    ret = avcodec_open2(enc_ctx, encoder, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot open encoder: %s\n", av_err2str(ret)); return 1; }

    /* Open output */
    ret = avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, output_path);
    if (ret < 0) { fprintf(stderr, "Cannot create output context\n"); return 1; }

    AVStream* out_stream = avformat_new_stream(ofmt_ctx, NULL);
    avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    out_stream->time_base = enc_ctx->time_base;

    if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, output_path, AVIO_FLAG_WRITE);
        if (ret < 0) { fprintf(stderr, "Cannot open output file: %s\n", av_err2str(ret)); return 1; }
    }

    /* Write header with faststart-compatible settings */
    AVDictionary* opts = NULL;
    av_dict_set(&opts, "movflags", "+faststart", 0);
    ret = avformat_write_header(ofmt_ctx, &opts);
    av_dict_free(&opts);
    if (ret < 0) { fprintf(stderr, "Cannot write header: %s\n", av_err2str(ret)); return 1; }

    /* Allocate frames and packet */
    pkt = av_packet_alloc();
    hw_frame = av_frame_alloc();

    /* Pool of NV12 frames for pipelining: tonemap into frame N+1
     * while NVENC is still encoding frame N */
    AVFrame* nv12_frames[NUM_BUFS];
    for (int i = 0; i < NUM_BUFS; i++) {
        nv12_frames[i] = av_frame_alloc();
        nv12_frames[i]->format = AV_PIX_FMT_CUDA;
        nv12_frames[i]->width = width;
        nv12_frames[i]->height = height;
        nv12_frames[i]->hw_frames_ctx = av_buffer_ref(enc_hw_frames_ref);
        ret = av_hwframe_get_buffer(enc_hw_frames_ref, nv12_frames[i], 0);
        if (ret < 0) { fprintf(stderr, "Cannot alloc NV12 frame %d: %s\n", i, av_err2str(ret)); return 1; }
    }
    int buf_idx = 0;

    /* CUDA stream for tonemap kernel (avoids default-stream serialization) */
    cudaStream_t tonemap_stream;
    CHECK_CUDA(cudaStreamCreate(&tonemap_stream));

    /* CUDA kernel launch config */
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    fprintf(stderr, "Processing: NVDEC → CUDA tonemap → NVENC AV1 @ %d kbps, GOP=%d, preset=p1\n",
            bitrate_kbps, enc_ctx->gop_size);

    /* Helper: drain encoded packets from encoder */
    AVPacket* enc_pkt = av_packet_alloc();
    auto drain_encoder = [&]() {
        while (1) {
            ret = avcodec_receive_packet(enc_ctx, enc_pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) { fprintf(stderr, "Encode receive error\n"); break; }
            enc_pkt->stream_index = 0;
            av_packet_rescale_ts(enc_pkt, enc_ctx->time_base, out_stream->time_base);
            av_interleaved_write_frame(ofmt_ctx, enc_pkt);
        }
    };

    /* Helper: tonemap a decoded frame and send to encoder */
    auto tonemap_and_encode = [&](AVFrame* decoded) {
        AVFrame* out = nv12_frames[buf_idx];
        buf_idx = (buf_idx + 1) % NUM_BUFS;

        tonemap_full<<<grid, block, 0, tonemap_stream>>>(
            (const uint16_t*)decoded->data[0], decoded->linesize[0],
            (const uint16_t*)decoded->data[1], decoded->linesize[1],
            (uint8_t*)out->data[0], out->linesize[0],
            (uint8_t*)out->data[1], out->linesize[1],
            width, height);
        CHECK_CUDA(cudaStreamSynchronize(tonemap_stream));

        out->pts = decoded->pts;
        out->pkt_dts = decoded->pkt_dts;
        out->duration = decoded->duration;

        ret = avcodec_send_frame(enc_ctx, out);
        if (ret < 0) fprintf(stderr, "Encode send error: %s\n", av_err2str(ret));

        drain_encoder();
        frame_count++;
        if (frame_count % 200 == 0)
            fprintf(stderr, "  Frame %ld...\n", frame_count);
    };

    /* Main decode-tonemap-encode loop */
    while (av_read_frame(ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_stream_idx) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(dec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0) { fprintf(stderr, "Decode send error: %s\n", av_err2str(ret)); continue; }

        while (1) {
            ret = avcodec_receive_frame(dec_ctx, hw_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) { fprintf(stderr, "Decode receive error: %s\n", av_err2str(ret)); break; }

            tonemap_and_encode(hw_frame);
            av_frame_unref(hw_frame);
        }
    }

    /* Flush decoder */
    avcodec_send_packet(dec_ctx, NULL);
    while (1) {
        ret = avcodec_receive_frame(dec_ctx, hw_frame);
        if (ret < 0) break;
        tonemap_and_encode(hw_frame);
        av_frame_unref(hw_frame);
    }

    /* Flush encoder */
    avcodec_send_frame(enc_ctx, NULL);
    drain_encoder();

    /* Write trailer */
    av_write_trailer(ofmt_ctx);

    fprintf(stderr, "Done: %ld frames\n", frame_count);

    /* Cleanup */
    av_packet_free(&pkt);
    av_packet_free(&enc_pkt);
    av_frame_free(&hw_frame);
    for (int i = 0; i < NUM_BUFS; i++)
        av_frame_free(&nv12_frames[i]);
    cudaStreamDestroy(tonemap_stream);
    av_buffer_unref(&enc_hw_frames_ref);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    avformat_close_input(&ifmt_ctx);
    if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
        avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    av_buffer_unref(&hw_device_ctx);

    return 0;
}
