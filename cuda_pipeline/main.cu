// Combined CUDA SLAM + Hand Pose pipeline
// Single NVDEC decode, per-thread default streams for GPU kernel interleaving
// Compile with: --default-stream per-thread

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cublas_v2.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

// ============ Simple JSON string field parser (flat objects only) ============

static std::string json_get_string(const std::string& json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static int json_get_int(const std::string& json, const char* key, int def) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    // skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return atoi(json.c_str() + pos);
}

static float json_get_float(const std::string& json, const char* key, float def) {
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return (float)atof(json.c_str() + pos);
}

// Pipeline-specific code in namespaces (avoids all symbol conflicts)
#include "slam.cuh"
#include "hand.cuh"

// ============ NVDEC decode kernels (both formats) ============

// NV12 (8-bit) → float32 BGR with bilinear resize (for SLAM)
__global__ void nv12_to_bgr_resize_kernel(
    const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane,
    float* __restrict__ bgr_out, int srcW, int srcH,
    int y_stride, int uv_stride, int dstW, int dstH)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;
    float sx = (dx + 0.5f) * srcW / dstW - 0.5f;
    float sy = (dy + 0.5f) * srcH / dstH - 0.5f;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float Y_val = (1-fy)*((1-fx)*y_plane[y0*y_stride+x0] + fx*y_plane[y0*y_stride+x1])
                + fy*((1-fx)*y_plane[y1*y_stride+x0] + fx*y_plane[y1*y_stride+x1]);
    float uv_sx = sx * 0.5f, uv_sy = sy * 0.5f;
    int uvx = max(0, min((int)floorf(uv_sx), srcW/2-1));
    int uvy = max(0, min((int)floorf(uv_sy), srcH/2-1));
    float U_val = (float)uv_plane[uvy * uv_stride + uvx * 2] - 128.0f;
    float V_val = (float)uv_plane[uvy * uv_stride + uvx * 2 + 1] - 128.0f;
    float R = Y_val + 1.402f * V_val;
    float G = Y_val - 0.344f * U_val - 0.714f * V_val;
    float B = Y_val + 1.772f * U_val;
    int idx_out = dy * dstW + dx;
    bgr_out[0 * dstH * dstW + idx_out] = fminf(fmaxf(B, 0.0f), 255.0f);
    bgr_out[1 * dstH * dstW + idx_out] = fminf(fmaxf(G, 0.0f), 255.0f);
    bgr_out[2 * dstH * dstW + idx_out] = fminf(fmaxf(R, 0.0f), 255.0f);
}

// P010 (10-bit) → float32 BGR with bilinear resize (for SLAM)
__global__ void p010_to_bgr_resize_kernel(
    const uint16_t* __restrict__ y_plane, const uint16_t* __restrict__ uv_plane,
    float* __restrict__ bgr_out, int srcW, int srcH,
    int y_stride_bytes, int uv_stride_bytes, int dstW, int dstH)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;
    int y_stride = y_stride_bytes / 2;
    int uv_stride = uv_stride_bytes / 2;
    float sx = (dx + 0.5f) * srcW / dstW - 0.5f;
    float sy = (dy + 0.5f) * srcH / dstH - 0.5f;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    auto rd = [&](int yy, int xx) { return (float)(y_plane[yy * y_stride + xx] >> 6); };
    float Y_val = (1-fy)*((1-fx)*rd(y0,x0)+fx*rd(y0,x1))
                + fy*((1-fx)*rd(y1,x0)+fx*rd(y1,x1));
    float uv_sx = sx * 0.5f, uv_sy = sy * 0.5f;
    int uvx = max(0, min((int)floorf(uv_sx), srcW/2-1));
    int uvy = max(0, min((int)floorf(uv_sy), srcH/2-1));
    float U_val = (float)(uv_plane[uvy * uv_stride + uvx * 2] >> 6) - 512.0f;
    float V_val = (float)(uv_plane[uvy * uv_stride + uvx * 2 + 1] >> 6) - 512.0f;
    float scale = 255.0f / 1023.0f;
    float R = (Y_val + 1.402f * V_val) * scale;
    float G = (Y_val - 0.344f * U_val - 0.714f * V_val) * scale;
    float B = (Y_val + 1.772f * U_val) * scale;
    int idx_out = dy * dstW + dx;
    bgr_out[0 * dstH * dstW + idx_out] = fminf(fmaxf(B, 0.0f), 255.0f);
    bgr_out[1 * dstH * dstW + idx_out] = fminf(fmaxf(G, 0.0f), 255.0f);
    bgr_out[2 * dstH * dstW + idx_out] = fminf(fmaxf(R, 0.0f), 255.0f);
}

// NV12 (8-bit) → uint8 BGR (for hand pipeline, no resize)
__global__ void nv12_to_bgr_kernel(
    const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane,
    uint8_t* __restrict__ bgr_out, int W, int H,
    int y_stride, int uv_stride, int bgr_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float Y = (float)y_plane[y * y_stride + x];
    int uvx = x / 2, uvy = y / 2;
    float U = (float)uv_plane[uvy * uv_stride + uvx * 2] - 128.0f;
    float V = (float)uv_plane[uvy * uv_stride + uvx * 2 + 1] - 128.0f;
    float R = Y + 1.402f * V;
    float G = Y - 0.344f * U - 0.714f * V;
    float B = Y + 1.772f * U;
    uint8_t* out = bgr_out + y * bgr_stride + x * 3;
    out[0] = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);
    out[1] = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    out[2] = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
}

// P010 (10-bit) → uint8 BGR (for hand pipeline, no resize)
__global__ void p010_to_bgr_kernel(
    const uint16_t* __restrict__ y_plane, const uint16_t* __restrict__ uv_plane,
    uint8_t* __restrict__ bgr_out, int W, int H,
    int y_stride_bytes, int uv_stride_bytes, int bgr_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int y_stride = y_stride_bytes / 2;
    int uv_stride = uv_stride_bytes / 2;
    float Y = (float)(y_plane[y * y_stride + x] >> 6) * (255.0f / 1023.0f);
    int uvx = x / 2, uvy = y / 2;
    float U = (float)(uv_plane[uvy * uv_stride + uvx * 2] >> 6) * (255.0f / 1023.0f) - 128.0f;
    float V = (float)(uv_plane[uvy * uv_stride + uvx * 2 + 1] >> 6) * (255.0f / 1023.0f) - 128.0f;
    float R = Y + 1.402f * V;
    float G = Y - 0.344f * U - 0.714f * V;
    float B = Y + 1.772f * U;
    uint8_t* out = bgr_out + y * bgr_stride + x * 3;
    out[0] = (uint8_t)fminf(fmaxf(B, 0.0f), 255.0f);
    out[1] = (uint8_t)fminf(fmaxf(G, 0.0f), 255.0f);
    out[2] = (uint8_t)fminf(fmaxf(R, 0.0f), 255.0f);
}

// ============ Thread-safe frame queue ============

template<typename T>
struct FrameQueue {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<T> items;
    bool done = false;
    size_t head = 0;

    void push(T item) {
        std::lock_guard<std::mutex> lk(mtx);
        items.push_back(std::move(item));
        cv.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]{ return head < items.size() || done; });
        if (head >= items.size()) return false;
        out = std::move(items[head++]);
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lk(mtx);
        done = true;
        cv.notify_all();
    }
};

struct SlamFrame {
    int frame_idx;
    // GPU pointer to float32 BGR NCHW [3, slamH, slamW] — already converted
    // Points into a ring buffer slot
    int slot;
};

struct HandFrame {
    int frame_idx;
    // GPU pointer to uint8 BGR HWC [H, W, 3] — already converted
    int slot;
};

// ============ Main ============

int main(int argc, char** argv) {
    // Parse arguments
    const char* video_path = nullptr;
    const char* slam_weights = nullptr;
    const char* hand_weights = nullptr;
    const char* calib_file = nullptr;
    const char* pose_output = nullptr;
    const char* hand_output = nullptr;
    int slam_resize_h = 0, slam_resize_w = 0;
    int max_frames = 99999;
    int backend_iters1 = 3, backend_iters2 = 5;
    int backend_radius = 1;
    int frontend_window = 15;
    int update_steps = 2;
    float hand_det_conf = 0.3f;
    int hand_stride = 1;
    int hand_wilor_batch = 48;
    bool skip_slam = false, skip_hands = false;
    bool listen_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--video") == 0 && i+1 < argc) video_path = argv[++i];
        else if (strcmp(argv[i], "--slam-weights") == 0 && i+1 < argc) slam_weights = argv[++i];
        else if (strcmp(argv[i], "--hand-weights") == 0 && i+1 < argc) hand_weights = argv[++i];
        else if (strcmp(argv[i], "--calib") == 0 && i+1 < argc) calib_file = argv[++i];
        else if (strcmp(argv[i], "--resize") == 0 && i+2 < argc) {
            slam_resize_h = atoi(argv[++i]); slam_resize_w = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--pose-output") == 0 && i+1 < argc) pose_output = argv[++i];
        else if (strcmp(argv[i], "--hand-output") == 0 && i+1 < argc) hand_output = argv[++i];
        else if (strcmp(argv[i], "--max-frames") == 0 && i+1 < argc) max_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i+2 < argc) {
            backend_iters1 = atoi(argv[++i]); backend_iters2 = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--backend-radius") == 0 && i+1 < argc) backend_radius = atoi(argv[++i]);
        else if (strcmp(argv[i], "--frontend-window") == 0 && i+1 < argc) frontend_window = atoi(argv[++i]);
        else if (strcmp(argv[i], "--update-steps") == 0 && i+1 < argc) update_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--det-conf") == 0 && i+1 < argc) hand_det_conf = atof(argv[++i]);
        else if (strcmp(argv[i], "--hand-stride") == 0 && i+1 < argc) hand_stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wilor-batch") == 0 && i+1 < argc) hand_wilor_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-slam") == 0) skip_slam = true;
        else if (strcmp(argv[i], "--skip-hands") == 0) skip_hands = true;
        else if (strcmp(argv[i], "--listen") == 0) listen_mode = true;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: cuda_pipeline --video <path> --slam-weights <dir> --hand-weights <dir>\n"
                   "       --calib <file> --resize <h> <w> --pose-output <file> --hand-output <file>\n"
                   "       [--max-frames N] [--backend N N] [--backend-radius N]\n"
                   "       [--frontend-window N] [--update-steps N] [--det-conf F] [--hand-stride N]\n"
                   "       [--listen]  (listen mode: read JSON jobs from stdin)\n");
            return 0;
        }
    }

    // ============ Listen mode ============
    if (listen_mode) {
        if (!slam_weights || !hand_weights) {
            fprintf(stderr, "Listen mode requires: --slam-weights, --hand-weights\n");
            return 1;
        }

        // Initialize WiLoR once (resolution-independent, expensive ~2s)
        hand::WilorModel wilor;
        wilor.init(std::string(hand_weights), hand_wilor_batch);
        fprintf(stderr, "[listen] WiLoR loaded (%d max batch)\n", hand_wilor_batch);

        // WiLoR crop buffer (sized for max batch, resolution-independent)
        const int VIT_IMG_H = 256, VIT_IMG_W = 192;
        float* d_crops_buf = nullptr;
        size_t d_crops_capacity = (size_t)hand_wilor_batch * 3 * VIT_IMG_H * VIT_IMG_W;
        CUDA_CHECK(cudaMalloc(&d_crops_buf, d_crops_capacity * sizeof(float)));

        // Hand frame buffer sized for max expected resolution (1920x1080)
        const int MAX_SRC_W = 1920, MAX_SRC_H = 1080;
        const int YOLO_BATCH = 8;
        hand::GpuBuf hand_frame_buf;
        size_t max_hand_single_frame = (size_t)MAX_SRC_W * MAX_SRC_H * 3;
        CUDA_CHECK(cudaMalloc((void**)&hand_frame_buf.data, max_hand_single_frame * YOLO_BATCH));
        hand_frame_buf.count = max_hand_single_frame * YOLO_BATCH / sizeof(float);

        // SLAM frame buffer sized for max expected resize
        slam::GpuBuf slam_frame_buf;

        // Track previous resolution to know when to reinit YOLO/SLAM
        int prev_srcW = 0, prev_srcH = 0;
        int prev_slamH = 0, prev_slamW = 0;

        // Models that are reinited per-video
        hand::YoloModel yolo;
        bool yolo_inited = false;
        slam::CudaDroid droid;
        bool droid_inited = false;

        // Hand constants
        const float IMAGE_MEAN[3] = {0.485f, 0.456f, 0.406f};
        const float IMAGE_STD[3] = {0.229f, 0.224f, 0.225f};

        printf("READY\n");
        fflush(stdout);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            // Parse JSON job
            std::string job_video = json_get_string(line, "video");
            std::string job_calib = json_get_string(line, "calib");
            int job_resize_h = json_get_int(line, "resize_h", 0);
            int job_resize_w = json_get_int(line, "resize_w", 0);
            std::string job_pose_output = json_get_string(line, "pose_output");
            std::string job_hand_output = json_get_string(line, "hand_output");
            int job_hand_stride = json_get_int(line, "hand_stride", 1);
            float job_det_conf = json_get_float(line, "hand_det_conf", 0.3f);
            int job_max_frames = json_get_int(line, "max_frames", 99999);
            int job_backend_iters1 = json_get_int(line, "backend_iters1", 3);
            int job_backend_iters2 = json_get_int(line, "backend_iters2", 5);
            int job_backend_radius = json_get_int(line, "backend_radius", 1);
            int job_frontend_window = json_get_int(line, "frontend_window", 15);
            int job_update_steps = json_get_int(line, "update_steps", 2);

            if (job_video.empty() || job_calib.empty() || job_resize_h == 0 || job_resize_w == 0) {
                printf("{\"status\":\"error\",\"message\":\"missing required fields: video, calib, resize_h, resize_w\"}\n");
                fflush(stdout);
                continue;
            }

            // Load calibration
            float calib[4];
            {
                FILE* f = fopen(job_calib.c_str(), "rb");
                if (!f) {
                    printf("{\"status\":\"error\",\"message\":\"cannot open calib file\"}\n");
                    fflush(stdout);
                    continue;
                }
                fread(calib, sizeof(float), 4, f);
                fclose(f);
            }

            // Open video with NVDEC
            AVFormatContext* fmt_ctx = nullptr;
            AVCodecContext* dec_ctx = nullptr;
            AVFrame* av_frame = av_frame_alloc();
            AVPacket* pkt = av_packet_alloc();
            AVBufferRef* hw_device_ctx = nullptr;
            bool using_nvdec = false, nvdec_is_p010 = false;
            int video_stream_idx = -1;

            if (avformat_open_input(&fmt_ctx, job_video.c_str(), nullptr, nullptr) < 0) {
                printf("{\"status\":\"error\",\"message\":\"cannot open video\"}\n");
                fflush(stdout);
                av_frame_free(&av_frame);
                av_packet_free(&pkt);
                continue;
            }
            avformat_find_stream_info(fmt_ctx, nullptr);

            for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
                if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                    video_stream_idx = i; break;
                }
            }
            if (video_stream_idx < 0) {
                printf("{\"status\":\"error\",\"message\":\"no video stream\"}\n");
                fflush(stdout);
                avformat_close_input(&fmt_ctx);
                av_frame_free(&av_frame);
                av_packet_free(&pkt);
                continue;
            }

            auto* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
            int srcW = codecpar->width, srcH = codecpar->height;

            // Check resolution fits in our pre-allocated buffers
            if (srcW > MAX_SRC_W || srcH > MAX_SRC_H) {
                printf("{\"status\":\"error\",\"message\":\"video resolution %dx%d exceeds max %dx%d\"}\n",
                       srcW, srcH, MAX_SRC_W, MAX_SRC_H);
                fflush(stdout);
                avformat_close_input(&fmt_ctx);
                av_frame_free(&av_frame);
                av_packet_free(&pkt);
                continue;
            }

            // Try NVDEC hardware decoder
            const char* cuvid_name = nullptr;
            if (codecpar->codec_id == AV_CODEC_ID_HEVC) cuvid_name = "hevc_cuvid";
            else if (codecpar->codec_id == AV_CODEC_ID_H264) cuvid_name = "h264_cuvid";
            else if (codecpar->codec_id == AV_CODEC_ID_AV1) cuvid_name = "av1_cuvid";

            const AVCodec* codec = nullptr;
            bool nvdec_ok = false;
            if (cuvid_name) {
                codec = avcodec_find_decoder_by_name(cuvid_name);
                if (codec) {
                    dec_ctx = avcodec_alloc_context3(codec);
                    avcodec_parameters_to_context(dec_ctx, codecpar);
                    CUcontext cu_ctx;
                    cuCtxGetCurrent(&cu_ctx);
                    if (!cu_ctx) { cudaFree(0); cuCtxGetCurrent(&cu_ctx); }
                    hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
                    if (hw_device_ctx) {
                        AVHWDeviceContext* device_ctx = (AVHWDeviceContext*)hw_device_ctx->data;
                        AVCUDADeviceContext* cuda_ctx = (AVCUDADeviceContext*)device_ctx->hwctx;
                        cuda_ctx->cuda_ctx = cu_ctx;
                        cuda_ctx->stream = nullptr;
                        av_hwdevice_ctx_init(hw_device_ctx);
                        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                        if (avcodec_open2(dec_ctx, codec, nullptr) == 0) {
                            nvdec_ok = true;
                            nvdec_is_p010 = (codecpar->bits_per_raw_sample > 8) ||
                                             (codecpar->format == AV_PIX_FMT_YUV420P10LE);
                        } else {
                            avcodec_free_context(&dec_ctx); dec_ctx = nullptr;
                        }
                    }
                }
            }
            if (!nvdec_ok) {
                printf("{\"status\":\"error\",\"message\":\"NVDEC not available\"}\n");
                fflush(stdout);
                avformat_close_input(&fmt_ctx);
                av_frame_free(&av_frame);
                av_packet_free(&pkt);
                if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
                continue;
            }

            fprintf(stderr, "[listen] Video: %s (%dx%d, %s)\n", job_video.c_str(), srcW, srcH,
                    nvdec_is_p010 ? "P010" : "NV12");

            int slamH = job_resize_h, slamW = job_resize_w;

            // Reinitialize SLAM if resolution changed (or first time)
            if (slamH != prev_slamH || slamW != prev_slamW || !droid_inited) {
                if (droid_inited) droid.destroy();
                droid.init(slamH, slamW, calib[0], calib[1], calib[2], calib[3], slam_weights);
                // Reallocate SLAM frame buffer if needed
                if (slamH != prev_slamH || slamW != prev_slamW) {
                    slam_frame_buf.free();
                    slam_frame_buf.alloc(3 * slamH * slamW);
                }
                prev_slamH = slamH;
                prev_slamW = slamW;
                droid_inited = true;
            } else {
                // Same resolution — just reset per-video state (keeps weights/buffers)
                fprintf(stderr, "[listen] SLAM reset (same resolution %dx%d)\n", slamW, slamH);
                droid.reset(calib[0], calib[1], calib[2], calib[3]);
            }
            droid.frontend_window = job_frontend_window;
            droid.update_steps = job_update_steps;

            // Reinitialize YOLO if video resolution or det_conf changed
            if (srcW != prev_srcW || srcH != prev_srcH || !yolo_inited) {
                yolo.init(std::string(hand_weights), YOLO_BATCH, job_det_conf, srcH, srcW);
                prev_srcW = srcW;
                prev_srcH = srcH;
                yolo_inited = true;
            }

            size_t hand_single_frame = (size_t)srcW * srcH * 3;

            // Hand result storage
            struct FrameResult {
                int frame_idx;
                bool left_detected = false, right_detected = false;
                float left_kp3d[63] = {}, right_kp3d[63] = {};
                float left_kp2d[42] = {}, right_kp2d[42] = {};
            };
            std::vector<FrameResult> hand_results;
            std::vector<float> pending_centers, pending_sizes, pending_img_sizes, pending_rights;
            std::vector<int> pending_result_idx;

            // WiLoR flush helper
            auto flush_wilor = [&]() {
                int n = pending_sizes.size();
                if (n == 0) return;
                std::vector<hand::WilorModel::HandResult> results(n);
                wilor.forward(d_crops_buf, n, pending_centers.data(), pending_sizes.data(),
                              pending_img_sizes.data(), pending_rights.data(), results.data());
                for (int i = 0; i < n; i++) {
                    int ri = pending_result_idx[i];
                    if (pending_rights[i] > 0.5f) {
                        hand_results[ri].right_detected = true;
                        memcpy(hand_results[ri].right_kp3d, results[i].kp3d, 63 * sizeof(float));
                        memcpy(hand_results[ri].right_kp2d, results[i].kp2d, 42 * sizeof(float));
                    } else {
                        hand_results[ri].left_detected = true;
                        memcpy(hand_results[ri].left_kp3d, results[i].kp3d, 63 * sizeof(float));
                        memcpy(hand_results[ri].left_kp2d, results[i].kp2d, 42 * sizeof(float));
                    }
                }
                pending_centers.clear(); pending_sizes.clear();
                pending_img_sizes.clear(); pending_rights.clear();
                pending_result_idx.clear();
            };

            // Decode helpers
            dim3 slam_block(32, 8);
            dim3 slam_grid((slamW + slam_block.x - 1) / slam_block.x,
                           (slamH + slam_block.y - 1) / slam_block.y);
            dim3 hand_block(32, 8);
            dim3 hand_grid((srcW + hand_block.x - 1) / hand_block.x,
                           (srcH + hand_block.y - 1) / hand_block.y);

            int total_frames = 0;
            int hand_n_in_batch = 0;
            int hand_batch_frame_indices[YOLO_BATCH];
            int hand_n_frames = 0;
            int total_dets = 0;

            auto decode_next = [&]() -> bool {
                while (av_read_frame(fmt_ctx, pkt) >= 0) {
                    if (pkt->stream_index != video_stream_idx) { av_packet_unref(pkt); continue; }
                    int ret = avcodec_send_packet(dec_ctx, pkt);
                    av_packet_unref(pkt);
                    if (ret < 0) continue;
                    if (avcodec_receive_frame(dec_ctx, av_frame) == 0) return true;
                }
                avcodec_send_packet(dec_ctx, nullptr);
                return avcodec_receive_frame(dec_ctx, av_frame) == 0;
            };

            // Pass 1: SLAM
            while (total_frames < job_max_frames && decode_next()) {
                if (nvdec_is_p010) {
                    p010_to_bgr_resize_kernel<<<slam_grid, slam_block>>>(
                        (const uint16_t*)av_frame->data[0], (const uint16_t*)av_frame->data[1],
                        slam_frame_buf.data, srcW, srcH,
                        av_frame->linesize[0], av_frame->linesize[1], slamW, slamH);
                } else {
                    nv12_to_bgr_resize_kernel<<<slam_grid, slam_block>>>(
                        av_frame->data[0], av_frame->data[1],
                        slam_frame_buf.data, srcW, srcH,
                        av_frame->linesize[0], av_frame->linesize[1], slamW, slamH);
                }
                cudaDeviceSynchronize();
                droid.process_frame(total_frames, slam_frame_buf.data);
                total_frames++;
            }

            // Pass 2: Hand processing (re-seek)
            avcodec_flush_buffers(dec_ctx);
            av_seek_frame(fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(dec_ctx);

            int hand_frame_count = 0;
            while (hand_frame_count < total_frames && decode_next()) {
                int frame_idx = hand_frame_count;
                bool hand_this_frame = (frame_idx % job_hand_stride == 0);
                if (hand_this_frame) {
                    uint8_t* dst = (uint8_t*)hand_frame_buf.data + hand_n_in_batch * hand_single_frame;
                    int bgr_stride = srcW * 3;
                    if (nvdec_is_p010) {
                        p010_to_bgr_kernel<<<hand_grid, hand_block>>>(
                            (const uint16_t*)av_frame->data[0], (const uint16_t*)av_frame->data[1],
                            dst, srcW, srcH,
                            av_frame->linesize[0], av_frame->linesize[1], bgr_stride);
                    } else {
                        nv12_to_bgr_kernel<<<hand_grid, hand_block>>>(
                            av_frame->data[0], av_frame->data[1],
                            dst, srcW, srcH,
                            av_frame->linesize[0], av_frame->linesize[1], bgr_stride);
                    }
                    cudaDeviceSynchronize();

                    FrameResult fr;
                    fr.frame_idx = frame_idx;
                    hand_results.push_back(fr);

                    hand_batch_frame_indices[hand_n_in_batch] = frame_idx;
                    hand_n_in_batch++;
                    hand_n_frames++;

                    if (hand_n_in_batch == YOLO_BATCH) {
                        auto dets = yolo.forward((const uint8_t*)hand_frame_buf.data,
                                                 hand_n_in_batch, srcH, srcW);
                        total_dets += dets.size();
                        if (!dets.empty()) {
                            float yolo_scale_x = (float)yolo.yolo_w / srcW;
                            float yolo_scale_y = (float)yolo.yolo_h / srcH;
                            int result_base = hand_results.size() - hand_n_in_batch;
                            for (auto& det : dets) {
                                if ((int)pending_sizes.size() >= hand_wilor_batch) flush_wilor();
                                int ri = result_base + det.batch_idx;
                                float x1 = det.x1 / yolo_scale_x, y1 = det.y1 / yolo_scale_y;
                                float x2 = det.x2 / yolo_scale_x, y2 = det.y2 / yolo_scale_y;
                                float cx = (x1+x2)/2, cy = (y1+y2)/2;
                                float bw = (x2-x1)*2, bh = (y2-y1)*2;
                                if (bh/bw < 256.0f/192.0f) bh = bw * 256.0f/192.0f;
                                else bw = bh * 192.0f/256.0f;
                                float bbox_size = std::max(bw, bh);
                                bool is_right = det.cls == 1;
                                pending_centers.push_back(cx); pending_centers.push_back(cy);
                                pending_sizes.push_back(bbox_size);
                                pending_img_sizes.push_back((float)srcW);
                                pending_img_sizes.push_back((float)srcH);
                                pending_rights.push_back(is_right ? 1.0f : 0.0f);
                                pending_result_idx.push_back(ri);
                                int ci = pending_sizes.size() - 1;
                                const uint8_t* fptr = (const uint8_t*)hand_frame_buf.data
                                                      + det.batch_idx * hand_single_frame;
                                int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
                                hand::wilor_preprocess_kernel<<<(total_px+255)/256, 256>>>(
                                    fptr, d_crops_buf, srcH, srcW,
                                    cx, cy, bbox_size, !is_right,
                                    VIT_IMG_H, VIT_IMG_W, ci,
                                    IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                                    IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
                            }
                        }
                        hand_n_in_batch = 0;
                    }
                }
                hand_frame_count++;
            }

            // Flush partial YOLO batch
            if (hand_n_in_batch > 0) {
                auto dets = yolo.forward((const uint8_t*)hand_frame_buf.data,
                                         hand_n_in_batch, srcH, srcW);
                total_dets += dets.size();
                if (!dets.empty()) {
                    float yolo_scale_x = (float)yolo.yolo_w / srcW;
                    float yolo_scale_y = (float)yolo.yolo_h / srcH;
                    int result_base = hand_results.size() - hand_n_in_batch;
                    for (auto& det : dets) {
                        if ((int)pending_sizes.size() >= hand_wilor_batch) flush_wilor();
                        int ri = result_base + det.batch_idx;
                        float x1 = det.x1/yolo_scale_x, y1 = det.y1/yolo_scale_y;
                        float x2 = det.x2/yolo_scale_x, y2 = det.y2/yolo_scale_y;
                        float cx = (x1+x2)/2, cy = (y1+y2)/2;
                        float bw = (x2-x1)*2, bh = (y2-y1)*2;
                        if (bh/bw < 256.0f/192.0f) bh = bw*256.0f/192.0f;
                        else bw = bh*192.0f/256.0f;
                        float bbox_size = std::max(bw, bh);
                        bool is_right = det.cls == 1;
                        pending_centers.push_back(cx); pending_centers.push_back(cy);
                        pending_sizes.push_back(bbox_size);
                        pending_img_sizes.push_back((float)srcW);
                        pending_img_sizes.push_back((float)srcH);
                        pending_rights.push_back(is_right ? 1.0f : 0.0f);
                        pending_result_idx.push_back(ri);
                        int ci = pending_sizes.size() - 1;
                        const uint8_t* fptr = (const uint8_t*)hand_frame_buf.data
                                              + det.batch_idx * hand_single_frame;
                        int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
                        hand::wilor_preprocess_kernel<<<(total_px+255)/256, 256>>>(
                            fptr, d_crops_buf, srcH, srcW,
                            cx, cy, bbox_size, !is_right,
                            VIT_IMG_H, VIT_IMG_W, ci,
                            IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                            IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
                    }
                }
            }
            flush_wilor();

            // Backend optimization
            if (job_backend_iters1 > 0) {
                droid.backend(job_backend_iters1, job_backend_radius);
                if (job_backend_iters2 > 0) droid.backend(job_backend_iters2, job_backend_radius);
            }

            // Output SLAM poses
            int nk = droid.state.num_keyframes;
            if (!job_pose_output.empty()) {
                std::vector<float> poses(nk * 7);
                droid.state.poses.copyTo(poses.data(), nk * 7);
                for (int i = 0; i < nk; i++) {
                    float* p = &poses[i * 7];
                    float qx = p[3], qy = p[4], qz = p[5], qw = p[6];
                    float qi[4] = {-qx, -qy, -qz, qw};
                    float t[3] = {-p[0], -p[1], -p[2]};
                    float uv[3];
                    uv[0] = 2.0f*(qi[1]*t[2]-qi[2]*t[1]);
                    uv[1] = 2.0f*(qi[2]*t[0]-qi[0]*t[2]);
                    uv[2] = 2.0f*(qi[0]*t[1]-qi[1]*t[0]);
                    p[0] = t[0]+qi[3]*uv[0]+(qi[1]*uv[2]-qi[2]*uv[1]);
                    p[1] = t[1]+qi[3]*uv[1]+(qi[2]*uv[0]-qi[0]*uv[2]);
                    p[2] = t[2]+qi[3]*uv[2]+(qi[0]*uv[1]-qi[1]*uv[0]);
                    p[3]=qi[0]; p[4]=qi[1]; p[5]=qi[2]; p[6]=qi[3];
                }
                FILE* f = fopen(job_pose_output.c_str(), "wb");
                if (f) {
                    fwrite(&nk, sizeof(int), 1, f);
                    std::vector<int> ts(droid.state.kf_timestamps.begin(),
                                        droid.state.kf_timestamps.end());
                    fwrite(ts.data(), sizeof(int), nk, f);
                    fwrite(poses.data(), sizeof(float), nk * 7, f);
                    fclose(f);
                }
            }

            // Output hand results
            if (!job_hand_output.empty()) {
                FILE* f = fopen(job_hand_output.c_str(), "wb");
                if (f) {
                    int nhr = hand_results.size();
                    fwrite(&nhr, sizeof(int), 1, f);
                    fwrite(&total_frames, sizeof(int), 1, f);
                    fwrite(&job_hand_stride, sizeof(int), 1, f);
                    for (auto& fr : hand_results) {
                        fwrite(&fr.frame_idx, sizeof(int), 1, f);
                        uint8_t left = fr.left_detected ? 1 : 0;
                        uint8_t right = fr.right_detected ? 1 : 0;
                        fwrite(&left, 1, 1, f);
                        fwrite(&right, 1, 1, f);
                        fwrite(fr.left_kp3d, sizeof(float), 63, f);
                        fwrite(fr.right_kp3d, sizeof(float), 63, f);
                        fwrite(fr.left_kp2d, sizeof(float), 42, f);
                        fwrite(fr.right_kp2d, sizeof(float), 42, f);
                    }
                    fclose(f);
                }
            }

            // Cleanup video resources
            av_frame_free(&av_frame);
            av_packet_free(&pkt);
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&fmt_ctx);
            if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);

            // Print completion JSON to stdout
            printf("{\"status\":\"done\",\"frames\":%d}\n", total_frames);
            fflush(stdout);

            fprintf(stderr, "[listen] Done: %d frames, %d keyframes, %d hand dets\n",
                    total_frames, nk, total_dets);
        }

        // Cleanup persistent resources
        if (droid_inited) droid.destroy();
        slam_frame_buf.free();
        if (d_crops_buf) CUDA_CHECK(cudaFree(d_crops_buf));
        return 0;
    }

    // ============ Single-shot mode (original behavior) ============

    if (!video_path || !slam_weights || !hand_weights || !calib_file ||
        slam_resize_h == 0 || slam_resize_w == 0) {
        fprintf(stderr, "Required: --video, --slam-weights, --hand-weights, --calib, --resize\n");
        return 1;
    }

    // Load calibration
    float calib[4];
    {
        FILE* f = fopen(calib_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", calib_file); return 1; }
        fread(calib, sizeof(float), 4, f);
        fclose(f);
    }

    // ---- Open video with NVDEC ----
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    AVFrame* av_frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    AVBufferRef* hw_device_ctx = nullptr;
    bool using_nvdec = false, nvdec_is_p010 = false;
    int video_stream_idx = -1;

    if (avformat_open_input(&fmt_ctx, video_path, nullptr, nullptr) < 0) {
        fprintf(stderr, "Cannot open video: %s\n", video_path); return 1;
    }
    avformat_find_stream_info(fmt_ctx, nullptr);

    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i; break;
        }
    }
    if (video_stream_idx < 0) { fprintf(stderr, "No video stream\n"); return 1; }

    auto* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    int srcW = codecpar->width, srcH = codecpar->height;

    // Try NVDEC hardware decoder
    const char* cuvid_name = nullptr;
    if (codecpar->codec_id == AV_CODEC_ID_HEVC) cuvid_name = "hevc_cuvid";
    else if (codecpar->codec_id == AV_CODEC_ID_H264) cuvid_name = "h264_cuvid";
    else if (codecpar->codec_id == AV_CODEC_ID_AV1) cuvid_name = "av1_cuvid";

    const AVCodec* codec = nullptr;
    if (cuvid_name) {
        codec = avcodec_find_decoder_by_name(cuvid_name);
        if (codec) {
            dec_ctx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(dec_ctx, codecpar);
            CUcontext cu_ctx;
            cuCtxGetCurrent(&cu_ctx);
            if (!cu_ctx) { cudaFree(0); cuCtxGetCurrent(&cu_ctx); }
            hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
            if (hw_device_ctx) {
                AVHWDeviceContext* device_ctx = (AVHWDeviceContext*)hw_device_ctx->data;
                AVCUDADeviceContext* cuda_ctx = (AVCUDADeviceContext*)device_ctx->hwctx;
                cuda_ctx->cuda_ctx = cu_ctx;
                cuda_ctx->stream = nullptr;
                av_hwdevice_ctx_init(hw_device_ctx);
                dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                if (avcodec_open2(dec_ctx, codec, nullptr) == 0) {
                    using_nvdec = true;
                    nvdec_is_p010 = (codecpar->bits_per_raw_sample > 8) ||
                                     (codecpar->format == AV_PIX_FMT_YUV420P10LE);
                } else {
                    avcodec_free_context(&dec_ctx); dec_ctx = nullptr;
                }
            }
        }
    }
    if (!using_nvdec) {
        fprintf(stderr, "NVDEC not available, this binary requires hardware decode\n");
        return 1;
    }
    printf("Using NVDEC (%s, %s) %dx%d\n", cuvid_name,
           nvdec_is_p010 ? "P010" : "NV12", srcW, srcH);
    printf("SLAM: %dx%d, Hands: %dx%d (stride=%d)\n",
           slam_resize_w, slam_resize_h, srcW, srcH, hand_stride);

    // ---- Initialize pipelines ----
    int slamH = slam_resize_h, slamW = slam_resize_w;

    slam::CudaDroid droid;
    droid.init(slamH, slamW, calib[0], calib[1], calib[2], calib[3], slam_weights);
    droid.frontend_window = frontend_window;
    droid.update_steps = update_steps;

    hand::YoloModel yolo;
    hand::WilorModel wilor;
    const int YOLO_BATCH = 8;
    yolo.init(std::string(hand_weights), YOLO_BATCH, hand_det_conf, srcH, srcW);
    wilor.init(std::string(hand_weights), hand_wilor_batch);

    // ---- Allocate GPU buffers ----
    // SLAM frame buffer (single, processes inline)
    slam::GpuBuf slam_frame_buf;
    slam_frame_buf.alloc(3 * slamH * slamW);

    // Hand frame batch buffer (YOLO_BATCH frames of uint8 BGR)
    hand::GpuBuf hand_frame_buf;
    size_t hand_single_frame = (size_t)srcW * srcH * 3;
    CUDA_CHECK(cudaMalloc((void**)&hand_frame_buf.data, hand_single_frame * YOLO_BATCH));
    hand_frame_buf.count = hand_single_frame * YOLO_BATCH / sizeof(float);

    // WiLoR crop buffer
    const int VIT_IMG_H = 256, VIT_IMG_W = 192;
    float* d_crops_buf = nullptr;
    size_t d_crops_capacity = (size_t)hand_wilor_batch * 3 * VIT_IMG_H * VIT_IMG_W;
    CUDA_CHECK(cudaMalloc(&d_crops_buf, d_crops_capacity * sizeof(float)));

    // ---- Hand result storage ----
    struct FrameResult {
        int frame_idx;
        bool left_detected = false, right_detected = false;
        float left_kp3d[63] = {}, right_kp3d[63] = {};
        float left_kp2d[42] = {}, right_kp2d[42] = {};
    };
    std::vector<FrameResult> hand_results;
    std::vector<float> pending_centers, pending_sizes, pending_img_sizes, pending_rights;
    std::vector<int> pending_result_idx;

    // ---- WiLoR flush helper ----
    auto flush_wilor = [&]() {
        int n = pending_sizes.size();
        if (n == 0) return;
        std::vector<hand::WilorModel::HandResult> results(n);
        wilor.forward(d_crops_buf, n, pending_centers.data(), pending_sizes.data(),
                      pending_img_sizes.data(), pending_rights.data(), results.data());
        for (int i = 0; i < n; i++) {
            int ri = pending_result_idx[i];
            if (pending_rights[i] > 0.5f) {
                hand_results[ri].right_detected = true;
                memcpy(hand_results[ri].right_kp3d, results[i].kp3d, 63 * sizeof(float));
                memcpy(hand_results[ri].right_kp2d, results[i].kp2d, 42 * sizeof(float));
            } else {
                hand_results[ri].left_detected = true;
                memcpy(hand_results[ri].left_kp3d, results[i].kp3d, 63 * sizeof(float));
                memcpy(hand_results[ri].left_kp2d, results[i].kp2d, 42 * sizeof(float));
            }
        }
        pending_centers.clear(); pending_sizes.clear();
        pending_img_sizes.clear(); pending_rights.clear();
        pending_result_idx.clear();
    };

    // Hand constants
    const float IMAGE_MEAN[3] = {0.485f, 0.456f, 0.406f};
    const float IMAGE_STD[3] = {0.229f, 0.224f, 0.225f};

    // ---- Decode + process loop (single-threaded) ----
    // Single NVDEC decode serves both SLAM and hand processing
    // No threading — simpler, avoids CUDA stream sync complexity
    int total_frames = 0;
    // Decode loop
    dim3 slam_block(32, 8);
    dim3 slam_grid((slamW + slam_block.x - 1) / slam_block.x,
                   (slamH + slam_block.y - 1) / slam_block.y);
    dim3 hand_block(32, 8);
    dim3 hand_grid((srcW + hand_block.x - 1) / hand_block.x,
                   (srcH + hand_block.y - 1) / hand_block.y);

    int hand_n_in_batch = 0;
    int hand_batch_frame_indices[YOLO_BATCH];
    int hand_n_frames = 0;
    int total_dets = 0;
    int slam_slot = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    auto decode_next = [&]() -> bool {
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index != video_stream_idx) { av_packet_unref(pkt); continue; }
            int ret = avcodec_send_packet(dec_ctx, pkt);
            av_packet_unref(pkt);
            if (ret < 0) continue;
            if (avcodec_receive_frame(dec_ctx, av_frame) == 0) return true;
        }
        avcodec_send_packet(dec_ctx, nullptr);
        return avcodec_receive_frame(dec_ctx, av_frame) == 0;
    };

    // ---- Pass 1: SLAM only ----
    auto t_slam_start = std::chrono::high_resolution_clock::now();
    printf("Pass 1: SLAM (%d max frames)...\n", max_frames);
    while (total_frames < max_frames && decode_next()) {
        int frame_idx = total_frames;

        // Convert for SLAM: NV12/P010 → float32 BGR with resize
        if (nvdec_is_p010) {
            p010_to_bgr_resize_kernel<<<slam_grid, slam_block>>>(
                (const uint16_t*)av_frame->data[0], (const uint16_t*)av_frame->data[1],
                slam_frame_buf.data, srcW, srcH,
                av_frame->linesize[0], av_frame->linesize[1], slamW, slamH);
        } else {
            nv12_to_bgr_resize_kernel<<<slam_grid, slam_block>>>(
                av_frame->data[0], av_frame->data[1],
                slam_frame_buf.data, srcW, srcH,
                av_frame->linesize[0], av_frame->linesize[1], slamW, slamH);
        }
        cudaDeviceSynchronize();
        droid.process_frame(frame_idx, slam_frame_buf.data);
        total_frames++;
    }
    auto t_slam_end = std::chrono::high_resolution_clock::now();
    float slam_ms = std::chrono::duration<float, std::milli>(t_slam_end - t_slam_start).count();
    printf("SLAM done: %d frames, %d keyframes (%.1f ms)\n", total_frames, droid.state.num_keyframes, slam_ms);

    // ---- Pass 2: Hand processing ----
    // Re-open video for second decode pass
    avcodec_flush_buffers(dec_ctx);
    av_seek_frame(fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(dec_ctx);

    auto t_hand_start = std::chrono::high_resolution_clock::now();
    printf("Pass 2: Hand processing...\n");
    int hand_frame_count = 0;
    while (hand_frame_count < total_frames && decode_next()) {
        int frame_idx = hand_frame_count;
        bool hand_this_frame = (frame_idx % hand_stride == 0);
        if (hand_this_frame) {
            uint8_t* dst = (uint8_t*)hand_frame_buf.data + hand_n_in_batch * hand_single_frame;
            int bgr_stride = srcW * 3;
            if (nvdec_is_p010) {
                p010_to_bgr_kernel<<<hand_grid, hand_block>>>(
                    (const uint16_t*)av_frame->data[0], (const uint16_t*)av_frame->data[1],
                    dst, srcW, srcH,
                    av_frame->linesize[0], av_frame->linesize[1], bgr_stride);
            } else {
                nv12_to_bgr_kernel<<<hand_grid, hand_block>>>(
                    av_frame->data[0], av_frame->data[1],
                    dst, srcW, srcH,
                    av_frame->linesize[0], av_frame->linesize[1], bgr_stride);
            }
            cudaDeviceSynchronize();

            FrameResult fr;
            fr.frame_idx = frame_idx;
            hand_results.push_back(fr);

            hand_batch_frame_indices[hand_n_in_batch] = frame_idx;
            hand_n_in_batch++;
            hand_n_frames++;

            if (hand_n_in_batch == YOLO_BATCH) {
                auto dets = yolo.forward((const uint8_t*)hand_frame_buf.data,
                                         hand_n_in_batch, srcH, srcW);
                total_dets += dets.size();

                if (!dets.empty()) {
                    float yolo_scale_x = (float)yolo.yolo_w / srcW;
                    float yolo_scale_y = (float)yolo.yolo_h / srcH;
                    int result_base = hand_results.size() - hand_n_in_batch;

                    for (auto& det : dets) {
                        if ((int)pending_sizes.size() >= hand_wilor_batch) flush_wilor();

                        int ri = result_base + det.batch_idx;
                        float x1 = det.x1 / yolo_scale_x, y1 = det.y1 / yolo_scale_y;
                        float x2 = det.x2 / yolo_scale_x, y2 = det.y2 / yolo_scale_y;
                        float cx = (x1+x2)/2, cy = (y1+y2)/2;
                        float bw = (x2-x1)*2, bh = (y2-y1)*2;
                        if (bh/bw < 256.0f/192.0f) bh = bw * 256.0f/192.0f;
                        else bw = bh * 192.0f/256.0f;
                        float bbox_size = std::max(bw, bh);
                        bool is_right = det.cls == 1;

                        pending_centers.push_back(cx); pending_centers.push_back(cy);
                        pending_sizes.push_back(bbox_size);
                        pending_img_sizes.push_back((float)srcW);
                        pending_img_sizes.push_back((float)srcH);
                        pending_rights.push_back(is_right ? 1.0f : 0.0f);
                        pending_result_idx.push_back(ri);

                        int ci = pending_sizes.size() - 1;
                        const uint8_t* fptr = (const uint8_t*)hand_frame_buf.data
                                              + det.batch_idx * hand_single_frame;
                        int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
                        hand::wilor_preprocess_kernel<<<(total_px+255)/256, 256>>>(
                            fptr, d_crops_buf, srcH, srcW,
                            cx, cy, bbox_size, !is_right,
                            VIT_IMG_H, VIT_IMG_W, ci,
                            IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                            IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
                    }
                }
                hand_n_in_batch = 0;
            }
        }
        hand_frame_count++;
    }

    // Flush partial YOLO batch
    if (hand_n_in_batch > 0) {
        auto dets = yolo.forward((const uint8_t*)hand_frame_buf.data,
                                 hand_n_in_batch, srcH, srcW);
        total_dets += dets.size();
        if (!dets.empty()) {
            float yolo_scale_x = (float)yolo.yolo_w / srcW;
            float yolo_scale_y = (float)yolo.yolo_h / srcH;
            int result_base = hand_results.size() - hand_n_in_batch;
            for (auto& det : dets) {
                if ((int)pending_sizes.size() >= hand_wilor_batch) flush_wilor();

                int ri = result_base + det.batch_idx;
                float x1 = det.x1/yolo_scale_x, y1 = det.y1/yolo_scale_y;
                float x2 = det.x2/yolo_scale_x, y2 = det.y2/yolo_scale_y;
                float cx = (x1+x2)/2, cy = (y1+y2)/2;
                float bw = (x2-x1)*2, bh = (y2-y1)*2;
                if (bh/bw < 256.0f/192.0f) bh = bw*256.0f/192.0f;
                else bw = bh*192.0f/256.0f;
                float bbox_size = std::max(bw, bh);
                bool is_right = det.cls == 1;
                pending_centers.push_back(cx); pending_centers.push_back(cy);
                pending_sizes.push_back(bbox_size);
                pending_img_sizes.push_back((float)srcW);
                pending_img_sizes.push_back((float)srcH);
                pending_rights.push_back(is_right ? 1.0f : 0.0f);
                pending_result_idx.push_back(ri);
                int ci = pending_sizes.size() - 1;
                const uint8_t* fptr = (const uint8_t*)hand_frame_buf.data
                                      + det.batch_idx * hand_single_frame;
                int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
                hand::wilor_preprocess_kernel<<<(total_px+255)/256, 256>>>(
                    fptr, d_crops_buf, srcH, srcW,
                    cx, cy, bbox_size, !is_right,
                    VIT_IMG_H, VIT_IMG_W, ci,
                    IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                    IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
            }
        }
    }
    flush_wilor();

    auto t_hand_end = std::chrono::high_resolution_clock::now();
    float hand_ms = std::chrono::duration<float, std::milli>(t_hand_end - t_hand_start).count();
    printf("Hand done: %d frames, %d detections (%.1f ms)\n", hand_n_frames, total_dets, hand_ms);

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Backend optimization
    if (backend_iters1 > 0) {
        droid.backend(backend_iters1, backend_radius);
        if (backend_iters2 > 0) droid.backend(backend_iters2, backend_radius);
    }

    // ---- Output ----
    int nk = droid.state.num_keyframes;
    printf("\n=== Results ===\n");
    printf("Frames: %d, SLAM keyframes: %d, Hand detections: %d\n",
           total_frames, nk, total_dets);
    printf("Total: %.1f ms (%.1f fps)\n", total_ms, total_frames * 1000.0f / total_ms);

    // Output SLAM poses (same binary format as cuda_droid)
    if (pose_output) {
        std::vector<float> poses(nk * 7);
        droid.state.poses.copyTo(poses.data(), nk * 7);

        // Invert to camera-to-world
        for (int i = 0; i < nk; i++) {
            float* p = &poses[i * 7];
            float qx = p[3], qy = p[4], qz = p[5], qw = p[6];
            float qi[4] = {-qx, -qy, -qz, qw};
            float t[3] = {-p[0], -p[1], -p[2]};
            float uv[3];
            uv[0] = 2.0f*(qi[1]*t[2]-qi[2]*t[1]);
            uv[1] = 2.0f*(qi[2]*t[0]-qi[0]*t[2]);
            uv[2] = 2.0f*(qi[0]*t[1]-qi[1]*t[0]);
            p[0] = t[0]+qi[3]*uv[0]+(qi[1]*uv[2]-qi[2]*uv[1]);
            p[1] = t[1]+qi[3]*uv[1]+(qi[2]*uv[0]-qi[0]*uv[2]);
            p[2] = t[2]+qi[3]*uv[2]+(qi[0]*uv[1]-qi[1]*uv[0]);
            p[3]=qi[0]; p[4]=qi[1]; p[5]=qi[2]; p[6]=qi[3];
        }

        FILE* f = fopen(pose_output, "wb");
        if (f) {
            fwrite(&nk, sizeof(int), 1, f);
            std::vector<int> ts(droid.state.kf_timestamps.begin(),
                                droid.state.kf_timestamps.end());
            fwrite(ts.data(), sizeof(int), nk, f);
            fwrite(poses.data(), sizeof(float), nk * 7, f);
            fclose(f);
        }
        printf("=== Final Poses (%d keyframes from %d frames) ===\n", nk, total_frames);
    }

    // Output hand results (same binary format as cuda_hand)
    if (hand_output) {
        FILE* f = fopen(hand_output, "wb");
        if (f) {
            int nhr = hand_results.size();
            fwrite(&nhr, sizeof(int), 1, f);
            fwrite(&total_frames, sizeof(int), 1, f);
            fwrite(&hand_stride, sizeof(int), 1, f);
            for (auto& fr : hand_results) {
                fwrite(&fr.frame_idx, sizeof(int), 1, f);
                uint8_t left = fr.left_detected ? 1 : 0;
                uint8_t right = fr.right_detected ? 1 : 0;
                fwrite(&left, 1, 1, f);
                fwrite(&right, 1, 1, f);
                fwrite(fr.left_kp3d, sizeof(float), 63, f);
                fwrite(fr.right_kp3d, sizeof(float), 63, f);
                fwrite(fr.left_kp2d, sizeof(float), 42, f);
                fwrite(fr.right_kp2d, sizeof(float), 42, f);
            }
            fclose(f);
        }
        fprintf(stderr, "[cuda_pipeline] Hand results: %d frames, %d detections\n",
                hand_n_frames, total_dets);
    }

    // Cleanup
    av_frame_free(&av_frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    if (d_crops_buf) CUDA_CHECK(cudaFree(d_crops_buf));
    droid.destroy();

    return 0;
}
