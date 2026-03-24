/*
 * Full GPU pipeline: NVDEC decode → CUDA format convert → NVENC encode.
 *
 * Keeps frames on GPU throughout. Supports H.264 (default, fastest) and AV1.
 * For HDR input, can optionally tonemap (--tonemap) or just truncate 10→8 bit
 * (default, preserves HLG's backwards-compatible look).
 *
 * Usage:
 *   ./gpu_convert input.mov output.mp4 [bitrate_kbps] [--av1] [--tonemap]
 *
 * Build:
 *   nvcc -O3 -o gpu_convert gpu_convert.cu \
 *     $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
 *     -lcuda
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
 * Simple P010 → NV12 bit truncation kernel.
 * Just shifts 10-bit values to 8-bit. No color space conversion.
 * HLG signal values displayed as-is look natural on SDR displays
 * (HLG was designed for backwards compatibility).
 *
 * Each thread processes 4 luma bytes (uint32) for coalesced access.
 */
__global__ void p010_to_nv12_y(
    const uint16_t* __restrict__ src, int src_pitch,
    uint8_t* __restrict__ dst,        int dst_pitch,
    int width, int height)
{
    /* Each thread handles 4 pixels horizontally */
    int x4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (x4 >= width || y >= height) return;

    int src_row = y * (src_pitch / 2);
    int dst_row = y * dst_pitch;
    int remaining = min(4, width - x4);

    /* Read 4 uint16 values, shift >>6 for 10-bit, >>2 for 8-bit */
    uint8_t out[4];
    for (int i = 0; i < remaining; i++)
        out[i] = (uint8_t)(src[src_row + x4 + i] >> 8);

    /* Write (partial writes at edge) */
    if (remaining == 4 && (x4 & 3) == 0) {
        *(uint32_t*)&dst[dst_row + x4] = *(uint32_t*)out;
    } else {
        for (int i = 0; i < remaining; i++)
            dst[dst_row + x4 + i] = out[i];
    }
}

__global__ void p010_to_nv12_uv(
    const uint16_t* __restrict__ src, int src_pitch,
    uint8_t* __restrict__ dst,        int dst_pitch,
    int width, int height)
{
    /* UV plane: width/2 pairs, height/2 rows. Each thread does 2 pairs (4 bytes). */
    int x2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int uv_w = width;    /* UV plane width in samples (= luma width for NV12 interleaved) */
    int uv_h = height / 2;
    if (x2 >= uv_w || y >= uv_h) return;

    int src_row = y * (src_pitch / 2);
    int dst_row = y * dst_pitch;
    int remaining = min(2, (uv_w - x2));

    for (int i = 0; i < remaining; i++)
        dst[dst_row + x2 + i] = (uint8_t)(src[src_row + x2 + i] >> 8);
}

/*
 * HDR→SDR tonemap kernel: P010 (BT.2020/HLG) → NV12 (BT.709)
 * Full color science: HLG EOTF → OOTF → gamut map → BT.1886 inverse EOTF
 */

__device__ __forceinline__ float hlg_eotf(float x) {
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    if (x <= 0.5f)
        return x * x / 3.0f;
    else
        return (expf((x - c) / a) + b) / 12.0f;
}

__device__ __forceinline__ float bt1886_inverse_eotf(float x) {
    return __powf(fmaxf(x, 0.0f), 1.0f / 2.4f);
}

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

    float Y = (float)(src_y[y * (src_pitch_y / 2) + x] >> 6) / 1023.0f;

    int cx = x >> 1, cy = y >> 1;
    int uv_idx = cy * (src_pitch_uv / 2) + cx * 2;
    float Cb = (float)(src_uv[uv_idx] >> 6) / 1023.0f;
    float Cr = (float)(src_uv[uv_idx + 1] >> 6) / 1023.0f;

    Y  = (Y  - 64.0f / 1023.0f) * (1023.0f / 876.0f);
    Cb = (Cb - 512.0f / 1023.0f) * (1023.0f / 896.0f);
    Cr = (Cr - 512.0f / 1023.0f) * (1023.0f / 896.0f);

    float R = Y + 1.4746f * Cr;
    float G = Y - 0.16455f * Cb - 0.57135f * Cr;
    float B = Y + 1.8814f * Cb;
    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    R = hlg_eotf(R); G = hlg_eotf(G); B = hlg_eotf(B);

    float Ys = 0.2627f * R + 0.6780f * G + 0.0593f * B;
    float ootf_scale = (Ys > 0.0f) ? __powf(Ys, 0.2f) * 10.0f : 0.0f;
    R *= ootf_scale; G *= ootf_scale; B *= ootf_scale;

    float R709 = fmaxf( 1.6605f * R - 0.5877f * G - 0.0728f * B, 0.0f);
    float G709 = fmaxf(-0.1246f * R + 1.1329f * G - 0.0083f * B, 0.0f);
    float B709 = fmaxf(-0.0182f * R - 0.1006f * G + 1.1187f * B, 0.0f);

    R709 = bt1886_inverse_eotf(R709);
    G709 = bt1886_inverse_eotf(G709);
    B709 = bt1886_inverse_eotf(B709);

    float Y709 = 0.2126f * R709 + 0.7152f * G709 + 0.0722f * B709;
    dst_y[y * dst_pitch_y + x] = (uint8_t)fminf(fmaxf(Y709 * 219.0f + 16.5f, 0.0f), 255.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float Cb709 = (B709 - Y709) / 1.8556f;
        float Cr709 = (R709 - Y709) / 1.5748f;
        int out_uv = cy * dst_pitch_uv + cx * 2;
        dst_uv[out_uv]     = (uint8_t)fminf(fmaxf(Cb709 * 224.0f + 128.5f, 0.0f), 255.0f);
        dst_uv[out_uv + 1] = (uint8_t)fminf(fmaxf(Cr709 * 224.0f + 128.5f, 0.0f), 255.0f);
    }
}


static enum AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_CUDA;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input> <output.mp4> [bitrate_kbps] [--av1] [--tonemap]\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int bitrate_kbps = 2550;
    int use_av1 = 0;
    int use_tonemap = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--av1") == 0) use_av1 = 1;
        else if (strcmp(argv[i], "--tonemap") == 0) use_tonemap = 1;
        else bitrate_kbps = atoi(argv[i]);
    }

    const char* enc_name = use_av1 ? "av1_nvenc" : "h264_nvenc";

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
    dec_ctx->pix_fmt = AV_PIX_FMT_CUDA;

    ret = avcodec_open2(dec_ctx, decoder, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot open decoder: %s\n", av_err2str(ret)); return 1; }

    /* Setup NVENC encoder */
    const AVCodec* encoder = avcodec_find_encoder_by_name(enc_name);
    if (!encoder) { fprintf(stderr, "%s not found\n", enc_name); return 1; }

    enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = AV_PIX_FMT_CUDA;
    enc_ctx->time_base = in_stream->time_base;
    enc_ctx->framerate = av_guess_frame_rate(ifmt_ctx, in_stream, NULL);
    enc_ctx->color_range = AVCOL_RANGE_MPEG;
    enc_ctx->gop_size = 2;
    enc_ctx->max_b_frames = 0;
    enc_ctx->bit_rate = (int64_t)bitrate_kbps * 1000;
    enc_ctx->rc_max_rate = (int64_t)(bitrate_kbps * 1.5) * 1000;
    enc_ctx->rc_buffer_size = bitrate_kbps * 2000;
    enc_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    if (use_tonemap) {
        /* Tonemapped output is BT.709 */
        enc_ctx->colorspace = AVCOL_SPC_BT709;
        enc_ctx->color_trc = AVCOL_TRC_BT709;
        enc_ctx->color_primaries = AVCOL_PRI_BT709;
    } else {
        /* Preserve input color metadata */
        enc_ctx->colorspace = in_stream->codecpar->color_space;
        enc_ctx->color_trc = in_stream->codecpar->color_trc;
        enc_ctx->color_primaries = in_stream->codecpar->color_primaries;
    }

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
    enc_hw_frames->sw_format = AV_PIX_FMT_NV12;
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

    /* Write header */
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot write header: %s\n", av_err2str(ret)); return 1; }

    /* Allocate frames and packet */
    pkt = av_packet_alloc();
    hw_frame = av_frame_alloc();

    /* Pool of NV12 frames for pipelining */
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

    /* CUDA stream for conversion kernel */
    cudaStream_t conv_stream;
    CHECK_CUDA(cudaStreamCreate(&conv_stream));

    /* Kernel launch configs */
    dim3 block(32, 8);
    dim3 grid_y(((width / 4) + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 grid_uv(((width / 2) + block.x - 1) / block.x, ((height / 2) + block.y - 1) / block.y);
    dim3 grid_tonemap((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    fprintf(stderr, "Processing: NVDEC → %s → NVENC %s @ %d kbps, GOP=%d, preset=p1\n",
            use_tonemap ? "CUDA tonemap" : "P010→NV12",
            use_av1 ? "AV1" : "H.264",
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

    /* Helper: convert a decoded frame and send to encoder */
    auto convert_and_encode = [&](AVFrame* decoded) {
        AVFrame* out = nv12_frames[buf_idx];
        buf_idx = (buf_idx + 1) % NUM_BUFS;

        if (use_tonemap) {
            tonemap_full<<<grid_tonemap, block, 0, conv_stream>>>(
                (const uint16_t*)decoded->data[0], decoded->linesize[0],
                (const uint16_t*)decoded->data[1], decoded->linesize[1],
                (uint8_t*)out->data[0], out->linesize[0],
                (uint8_t*)out->data[1], out->linesize[1],
                width, height);
        } else {
            /* Simple P010 → NV12 bit truncation */
            p010_to_nv12_y<<<grid_y, block, 0, conv_stream>>>(
                (const uint16_t*)decoded->data[0], decoded->linesize[0],
                (uint8_t*)out->data[0], out->linesize[0],
                width, height);
            p010_to_nv12_uv<<<grid_uv, block, 0, conv_stream>>>(
                (const uint16_t*)decoded->data[1], decoded->linesize[1],
                (uint8_t*)out->data[1], out->linesize[1],
                width, height);
        }
        CHECK_CUDA(cudaStreamSynchronize(conv_stream));

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

    /* Main decode-convert-encode loop */
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

            convert_and_encode(hw_frame);
            av_frame_unref(hw_frame);
        }
    }

    /* Flush decoder */
    avcodec_send_packet(dec_ctx, NULL);
    while (1) {
        ret = avcodec_receive_frame(dec_ctx, hw_frame);
        if (ret < 0) break;
        convert_and_encode(hw_frame);
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
    cudaStreamDestroy(conv_stream);
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
