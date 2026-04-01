// CUDA Hand Pose Pipeline: YOLO detection + WiLoR ViT + MANO hand model
// Pure CUDA implementation — no PyTorch dependency at inference
//
// Usage: cuda_hand --weights-dir data/weights --video input.mov [--det-conf 0.3] [--stride 1]
//   or:  cuda_hand --weights-dir data/weights [--det-conf 0.3]  (reads raw BGR from stdin)
// Writes binary results to --output file

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cublas_v2.h>

// FFmpeg headers for video decoding
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "nn.cuh"

// ============================================================================
// Constants
// ============================================================================

static const int YOLO_BATCH = 8;      // YOLO frame batch size (video mode)
static const int YOLO_IMGSZ = 512;    // YOLO max side (long side)
static const int YOLO_NC = 2;        // left=0, right=1
static const int YOLO_REG_MAX = 16;
static const int YOLO_NL = 3;        // detection scales
static const int YOLO_NKP = 21;      // keypoints per detection
static const int YOLO_MAX_DETS = 128; // max detections before NMS

static const int VIT_IMG_H = 256;    // WiLoR input height
static const int VIT_IMG_W = 192;    // WiLoR input width
static const int VIT_EMBED = 1280;
static const int VIT_DEPTH = 32;
static const int VIT_HEADS = 16;
static const int VIT_HEAD_DIM = 80;  // 1280/16
static const int VIT_MLP = 5120;     // 4*1280
static const int VIT_PATCHES = 192;  // 16 * 12 (from 256x192 input / 16x16 patches, padding=2)
// No crop: upstream WiLoR x[:,:,:,32:-32] is buggy (crashes with pos_embed shape mismatch).
// We feed full 256x192 to the ViT: Conv2d(3,1280,k16,s16,p2) -> [B,1280,16,12] -> 192 patches.

static const int VIT_PATCH_H = 16;
static const int VIT_PATCH_W = 12;
static const int VIT_NUM_PATCHES = 192;  // 16*12
static const int VIT_POSE_TOKENS = 16;   // NUM_HAND_JOINTS + 1
static const int VIT_TOTAL_TOKENS = 210; // 16 + 1 + 1 + 192
static const int VIT_SKIP_BLOCKS[] = {25, 27, 26, 23, 24};  // first 5 to skip

static const int MANO_N_VERTS = 778;
static const int MANO_N_JOINTS = 16;
static const int MANO_N_OPENPOSE = 21;

static const float IMAGE_MEAN[] = {0.485f, 0.456f, 0.406f};
static const float IMAGE_STD[]  = {0.229f, 0.224f, 0.225f};
static const float FOCAL_LENGTH = 5000.0f;

// ============================================================================
// GPU Kernels for ViT helpers
// ============================================================================

__global__ void transpose_nchw_to_nhwc_kernel(const float* in, float* out, int B, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * HW * C) return;
    int c = idx % C;
    int hw = (idx / C) % HW;
    int b = idx / (C * HW);
    out[idx] = in[b * C * HW + c * HW + hw];
}

__global__ void transpose_nhwc_to_nchw_kernel(const float* in, float* out, int B, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * HW) return;
    int hw = idx % HW;
    int c = (idx / HW) % C;
    int b = idx / (C * HW);
    out[idx] = in[b * HW * C + hw * C + c];
}

__global__ void add_bias_half_kernel(__half* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols;
    data[idx] = __float2half(__half2float(data[idx]) + bias[c]);
}

__global__ void build_vit_tokens_kernel(
    float* output, const float* img_patches,
    const float* init_pose, const float* init_betas, const float* init_cam,
    const float* pose_w, const float* pose_b,
    const float* shape_w, const float* shape_b,
    const float* cam_w, const float* cam_b,
    int B, int T, int D, int num_patches)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;

    int d = idx % D;
    int t = (idx / D) % T;
    int b = idx / (D * T);

    float val;
    if (t < 16) {
        // Pose token: linear(init_pose[t*6:(t+1)*6]) using pose_w[D,6] and pose_b[D]
        val = pose_b[d];
        for (int k = 0; k < 6; k++) {
            val += init_pose[t * 6 + k] * pose_w[d * 6 + k];
        }
    } else if (t == 16) {
        // Shape token: linear(init_betas) using shape_w[D,10]
        val = shape_b[d];
        for (int k = 0; k < 10; k++) {
            val += init_betas[k] * shape_w[d * 10 + k];
        }
    } else if (t == 17) {
        // Cam token: linear(init_cam) using cam_w[D,3]
        val = cam_b[d];
        for (int k = 0; k < 3; k++) {
            val += init_cam[k] * cam_w[d * 3 + k];
        }
    } else {
        // Image token
        int patch_idx = t - 18;
        val = img_patches[(b * num_patches + patch_idx) * D + d];
    }

    output[idx] = val;
}

__global__ void add_pos_embed_kernel(float* tokens, const float* pos_embed,
                                      int B, int T, int D, int img_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * D) return;

    int d = idx % D;
    int t = (idx / D) % T;

    // pos_embed: [193, D] (193 = 1 cls + 192 patches)
    // Usage: x = x + pos_embed[:, 1:] + pos_embed[:, :1]
    // All tokens get pos_embed[0] added
    float pe = pos_embed[0 * D + d];  // cls/global positional

    // Image tokens (t >= img_start) also get pos_embed[1 + (t - img_start)]
    if (t >= img_start) {
        int patch_idx = t - img_start;
        if (patch_idx < 192) {
            pe += pos_embed[(1 + patch_idx) * D + d];
        }
    }

    tokens[idx] += pe;
}

// Reshape QKV: [B*T, 3D] -> Q[B*H, T, d], K[B*H, T, d], V[B*H, T, d]
__global__ void reshape_qkv_kernel(const __half* qkv, __half* Q, __half* K, __half* V,
                                    int B, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D = H * d;
    int total = B * T * 3 * D;
    if (idx >= B * H * T * d) return;

    // idx maps to output Q/K/V position [bh, t, di]
    int di = idx % d;
    int t = (idx / d) % T;
    int bh = idx / (T * d);
    int b = bh / H;
    int h = bh % H;

    // Source: qkv[(b*T + t) * 3*D + qkv_idx * D + h * d + di]
    int src_base = (b * T + t) * 3 * D;
    Q[idx] = qkv[src_base + 0 * D + h * d + di];
    K[idx] = qkv[src_base + 1 * D + h * d + di];
    V[idx] = qkv[src_base + 2 * D + h * d + di];
}

void reshape_qkv_to_heads(__half* qkv, __half* Q, __half* K, __half* V,
                            int B, int T, int H, int d) {
    int total = B * H * T * d;
    reshape_qkv_kernel<<<(total + 255) / 256, 256>>>(qkv, Q, K, V, B, T, H, d);
}

// Reshape heads back: [B*H, T, d] -> [B*T, D]
__global__ void reshape_heads_kernel(const __half* heads, __half* out,
                                      int B, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D = H * d;
    if (idx >= B * T * D) return;

    int di_full = idx % D;
    int t = (idx / D) % T;
    int b = idx / (T * D);

    int h = di_full / d;
    int di = di_full % d;

    // Source: heads[(b*H + h) * T * d + t * d + di]
    out[idx] = heads[((b * H + h) * T + t) * d + di];
}

void reshape_heads_to_seq(__half* heads, __half* out, int B, int T, int H, int d) {
    int total = B * T * H * d;
    reshape_heads_kernel<<<(total + 255) / 256, 256>>>(heads, out, B, T, H, d);
}

// ============================================================================
// WiLoR crop preprocessing
// ============================================================================

__global__ void wilor_preprocess_kernel(
    const uint8_t* frame, float* output,
    int frameH, int frameW,
    float cx, float cy, float bbox_size, bool flip,
    int outH, int outW, int crop_idx,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 3 * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = idx / (outH * outW);

    // Python pipeline:
    // 1. Creates 256x256 crop from bbox_size square centered at (cx,cy)
    // 2. ViT model does x[:,:,:,32:-32] to get 256x192
    // We directly produce the 256x192 output by accounting for the column offset.
    // In the 256x256 intermediate, pixel (ox, oy) maps to source:
    //   src_x = cx - bbox_size/2 + ox * bbox_size/256
    //   src_y = cy - bbox_size/2 + oy * bbox_size/256
    // After x[:,:,:,32:-32], output col ow maps to intermediate col (ow + 32):
    //   src_x = cx - bbox_size/2 + (ow + 32) * bbox_size/256
    // For flipped left hands, in Python the image is flipped first, then cropped.
    // Flip reverses x in the source frame: src_x = img_width - 1 - original_src_x
    // But center is also mirrored: cx_flip = img_width - 1 - cx
    // Combined: src_x = (img_width - 1 - cx) - bbox_size/2 + (ow + 32) * bbox_size/256
    //                  when flipped for output

    float px_scale = bbox_size / 256.0f;  // same in x and y (square crop)
    float ix, iy;
    if (flip) {
        // For left hands: flip source image, mirror center
        // Equivalent to sampling from original at mirrored x coordinate
        float cx_flip = (float)(frameW - 1) - cx;
        ix = cx_flip - bbox_size * 0.5f + (float)(outW - 1 - ow + 32) * px_scale;
    } else {
        ix = cx - bbox_size * 0.5f + (float)(ow + 32) * px_scale;
    }
    iy = cy - bbox_size * 0.5f + (float)oh * px_scale;

    // Bilinear sample from input frame (BGR)
    float val = 0;
    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    float wx = ix - ix0;
    float wy = iy - iy0;

    // BGR->RGB channel mapping
    int src_c = 2 - c;

    auto sample = [&](int sy, int sx) -> float {
        if (sy >= 0 && sy < frameH && sx >= 0 && sx < frameW)
            return (float)frame[(sy * frameW + sx) * 3 + src_c];
        return 0;
    };

    val = (1-wy)*(1-wx)*sample(iy0, ix0) + (1-wy)*wx*sample(iy0, ix0+1)
        + wy*(1-wx)*sample(iy0+1, ix0) + wy*wx*sample(iy0+1, ix0+1);

    // Normalize
    float means[] = {mean_r, mean_g, mean_b};
    float stds[] = {std_r, std_g, std_b};
    val = (val - means[c] * 255.0f) / (stds[c] * 255.0f);

    output[(crop_idx * 3 + c) * outH * outW + oh * outW + ow] = val;
}


// ============================================================================
// YOLO preprocessing: resize + normalize + HWC->CHW
// ============================================================================

__global__ void yolo_preprocess_kernel(const uint8_t* input, __half* output,
                                        int inH, int inW, int outH, int outW, int N) {
    // Letterbox resize to outH x outW, normalize to [0,1], BGR->RGB, HWC->CHW
    // Output is FP16 for tensor-core accelerated YOLO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 3 * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = (idx / (outH * outW)) % 3;
    int n = idx / (3 * outH * outW);

    float ix = (float)ow * inW / outW;
    float iy = (float)oh * inH / outH;
    int src_c = 2 - c;  // BGR to RGB

    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = min(ix0 + 1, inW - 1);
    int iy1 = min(iy0 + 1, inH - 1);
    ix0 = min(ix0, inW - 1);
    iy0 = min(iy0, inH - 1);
    float wx = ix - floorf(ix);
    float wy = iy - floorf(iy);

    const uint8_t* base = input + (size_t)n * inH * inW * 3;
    float v00 = base[iy0 * inW * 3 + ix0 * 3 + src_c];
    float v01 = base[iy0 * inW * 3 + ix1 * 3 + src_c];
    float v10 = base[iy1 * inW * 3 + ix0 * 3 + src_c];
    float v11 = base[iy1 * inW * 3 + ix1 * 3 + src_c];
    float val = ((1-wy)*((1-wx)*v00 + wx*v01) + wy*((1-wx)*v10 + wx*v11)) / 255.0f;

    output[idx] = __float2half(val);
}

// ============================================================================
// YOLO Model
// ============================================================================

struct YoloModel {
    cudnnHandle_t cudnn;
    GpuBuf workspace;  // raw bytes, shared across all convolutions
    size_t max_workspace = 0;

    // All convolution layers use FP16 (tensor-core accelerated)
    HalfConvLayer l0, l1, l3, l5, l7, l16, l19;

    struct C2fBlock {
        HalfConvLayer cv1, cv2;
        struct { HalfConvLayer cv1, cv2; } m[4];
        int n_bottleneck;
        bool shortcut;
    };
    C2fBlock c2f_2, c2f_4, c2f_6, c2f_8, c2f_12, c2f_15, c2f_18, c2f_21;

    HalfConvLayer sppf_cv1, sppf_cv2;

    struct HeadScale {
        HalfConvLayer cv2[3];
        HalfConvLayer cv3[3];
        HalfConvLayer cv4[3];
    };
    HeadScale head[3];
    GpuBuf dfl_weight, dfl_bias;  // FP32, small (16 values)

    // FP16 feature map buffers (pre-allocated)
    GpuHalfBuf bufs[30];
    GpuHalfBuf tmp1, tmp2, tmp3;
    // Pre-allocated C2f scratch buffers (avoid per-call cudaMalloc)
    GpuHalfBuf c2f_chunk0, c2f_chunk1, c2f_bn[4], c2f_concat;

    int batch_size;
    float det_conf;
    int yolo_h, yolo_w;

    void load_conv(HalfConvLayer& layer, const std::string& dir, const std::string& prefix,
                   int batch, int ci, int co, int kh, int kw, int s, int p, int inH, int inW) {
        std::vector<float> w_data, b_data;
        std::vector<int> w_shape, b_shape;
        load_tensor((dir + "/yolo_" + prefix + "_weight.bin").c_str(), w_data, w_shape);
        load_tensor((dir + "/yolo_" + prefix + "_bias.bin").c_str(), b_data, b_shape);
        layer.init(cudnn, batch, ci, co, kh, kw, s, p, inH, inW, w_data.data(), b_data.data());
        max_workspace = std::max(max_workspace, layer.workspaceSize);
    }

    void load_c2f(C2fBlock& block, const std::string& dir, const std::string& prefix,
                  int batch, int ci, int co, int n_bn, int bn_c, int H, int W, bool shortcut = true) {
        block.n_bottleneck = n_bn;
        block.shortcut = shortcut;
        load_conv(block.cv1, dir, prefix + "_cv1", batch, ci, co, 1, 1, 1, 0, H, W);
        for (int i = 0; i < n_bn; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s_m%d_cv1", prefix.c_str(), i);
            load_conv(block.m[i].cv1, dir, buf, batch, bn_c, bn_c, 3, 3, 1, 1, H, W);
            snprintf(buf, sizeof(buf), "%s_m%d_cv2", prefix.c_str(), i);
            load_conv(block.m[i].cv2, dir, buf, batch, bn_c, bn_c, 3, 3, 1, 1, H, W);
        }
        int cv2_ci = co + n_bn * bn_c;
        load_conv(block.cv2, dir, prefix + "_cv2", batch, cv2_ci, co, 1, 1, 1, 0, H, W);
    }

    void c2f_forward(C2fBlock& block, __half* input, __half* output,
                     int N, int ci, int co, int H, int W) {
        int HW = H * W;
        int half_c = co / 2;
        int bn_c = block.m[0].cv1.Co;

        block.cv1.forward(cudnn, input, tmp1.data, workspace.data, N, H, W);
        silu_half_inplace(tmp1.data, N * co * HW);

        slice_channels_half(c2f_chunk0.data, tmp1.data, N, co, 0, half_c, HW);
        slice_channels_half(c2f_chunk1.data, tmp1.data, N, co, half_c, half_c, HW);

        __half* prev = c2f_chunk1.data;

        __half* concat_parts[8];
        int concat_cs[8];
        concat_parts[0] = c2f_chunk0.data;
        concat_cs[0] = half_c;
        concat_parts[1] = c2f_chunk1.data;
        concat_cs[1] = half_c;

        for (int i = 0; i < block.n_bottleneck; i++) {
            // cv1
            block.m[i].cv1.forward(cudnn, prev, c2f_bn[i].data, workspace.data, N, H, W);
            silu_half_inplace(c2f_bn[i].data, N * bn_c * HW);
            // cv2 -> tmp2
            block.m[i].cv2.forward(cudnn, c2f_bn[i].data, tmp2.data, workspace.data, N, H, W);
            silu_half_inplace(tmp2.data, N * bn_c * HW);
            if (block.shortcut) {
                add_half_kernel<<<(N*bn_c*HW+255)/256, 256>>>(tmp2.data, prev, N * bn_c * HW);
            }
            // Store result in c2f_bn[i]
            CUDA_CHECK(cudaMemcpy(c2f_bn[i].data, tmp2.data, (size_t)N * bn_c * HW * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
            prev = c2f_bn[i].data;
            concat_parts[i + 2] = c2f_bn[i].data;
            concat_cs[i + 2] = bn_c;
        }

        // Concat all parts
        int n_parts = 2 + block.n_bottleneck;
        int total_c = co;
        for (int i = 0; i < block.n_bottleneck; i++) total_c += bn_c;

        for (int n = 0; n < N; n++) {
            size_t dst_base = (size_t)n * total_c * HW;
            size_t c_off = 0;
            for (int p = 0; p < n_parts; p++) {
                size_t src_base = (size_t)n * concat_cs[p] * HW;
                CUDA_CHECK(cudaMemcpy(c2f_concat.data + dst_base + c_off * HW,
                                      concat_parts[p] + src_base,
                                      (size_t)concat_cs[p] * HW * sizeof(__half),
                                      cudaMemcpyDeviceToDevice));
                c_off += concat_cs[p];
            }
        }

        block.cv2.forward(cudnn, c2f_concat.data, output, workspace.data, N, H, W);
        silu_half_inplace(output, N * co * HW);
    }

    void init(const std::string& weights_dir, int batch, float conf, int vidH, int vidW) {
        batch_size = batch;
        det_conf = conf;
        CUDNN_CHECK(cudnnCreate(&cudnn));

        float scale = (float)YOLO_IMGSZ / std::max(vidH, vidW);
        yolo_h = ((int)(vidH * scale) + 31) / 32 * 32;
        yolo_w = ((int)(vidW * scale) + 31) / 32 * 32;
        fprintf(stderr, "[YOLO] Input dimensions: %dx%d (video %dx%d)\n", yolo_h, yolo_w, vidH, vidW);

        int H = yolo_h, W = yolo_w;
        int B = batch;

        // Load backbone (all FP16 with tensor cores)
        load_conv(l0, weights_dir, "l0", B, 3, 48, 3, 3, 2, 1, H, W);
        load_conv(l1, weights_dir, "l1", B, 48, 96, 3, 3, 2, 1, H/2, W/2);
        load_c2f(c2f_2, weights_dir, "l2", B, 96, 96, 2, 48, H/4, W/4);
        load_conv(l3, weights_dir, "l3", B, 96, 192, 3, 3, 2, 1, H/4, W/4);
        load_c2f(c2f_4, weights_dir, "l4", B, 192, 192, 4, 96, H/8, W/8);
        load_conv(l5, weights_dir, "l5", B, 192, 384, 3, 3, 2, 1, H/8, W/8);
        load_c2f(c2f_6, weights_dir, "l6", B, 384, 384, 4, 192, H/16, W/16);
        load_conv(l7, weights_dir, "l7", B, 384, 576, 3, 3, 2, 1, H/16, W/16);
        load_c2f(c2f_8, weights_dir, "l8", B, 576, 576, 2, 288, H/32, W/32);

        load_conv(sppf_cv1, weights_dir, "l9_cv1", B, 576, 288, 1, 1, 1, 0, H/32, W/32);
        load_conv(sppf_cv2, weights_dir, "l9_cv2", B, 1152, 576, 1, 1, 1, 0, H/32, W/32);

        load_c2f(c2f_12, weights_dir, "l12", B, 960, 384, 2, 192, H/16, W/16, false);
        load_c2f(c2f_15, weights_dir, "l15", B, 576, 192, 2, 96, H/8, W/8, false);
        load_conv(l16, weights_dir, "l16", B, 192, 192, 3, 3, 2, 1, H/8, W/8);
        load_c2f(c2f_18, weights_dir, "l18", B, 576, 384, 2, 192, H/16, W/16, false);
        load_conv(l19, weights_dir, "l19", B, 384, 384, 3, 3, 2, 1, H/16, W/16);
        load_c2f(c2f_21, weights_dir, "l21", B, 960, 576, 2, 288, H/32, W/32, false);

        int head_channels[] = {192, 384, 576};
        int head_h[] = {H/8, H/16, H/32};
        int head_w[] = {W/8, W/16, W/32};
        for (int si = 0; si < 3; si++) {
            int ch = head_channels[si];
            int hh = head_h[si], hw = head_w[si];
            char buf[64];

            snprintf(buf, sizeof(buf), "head_cv2_s%d_c0", si);
            load_conv(head[si].cv2[0], weights_dir, buf, B, ch, 64, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv2_s%d_c1", si);
            load_conv(head[si].cv2[1], weights_dir, buf, B, 64, 64, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv2_s%d_c2", si);
            load_conv(head[si].cv2[2], weights_dir, buf, B, 64, 64, 1, 1, 1, 0, hh, hw);

            int cls_mid = 192;
            snprintf(buf, sizeof(buf), "head_cv3_s%d_c0", si);
            load_conv(head[si].cv3[0], weights_dir, buf, B, ch, cls_mid, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv3_s%d_c1", si);
            load_conv(head[si].cv3[1], weights_dir, buf, B, cls_mid, cls_mid, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv3_s%d_c2", si);
            load_conv(head[si].cv3[2], weights_dir, buf, B, cls_mid, YOLO_NC, 1, 1, 1, 0, hh, hw);

            int kp_ch = YOLO_NKP * 3;
            snprintf(buf, sizeof(buf), "head_cv4_s%d_c0", si);
            load_conv(head[si].cv4[0], weights_dir, buf, B, ch, kp_ch, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv4_s%d_c1", si);
            load_conv(head[si].cv4[1], weights_dir, buf, B, kp_ch, kp_ch, 3, 3, 1, 1, hh, hw);
            snprintf(buf, sizeof(buf), "head_cv4_s%d_c2", si);
            load_conv(head[si].cv4[2], weights_dir, buf, B, kp_ch, kp_ch, 1, 1, 1, 0, hh, hw);
        }

        // DFL weight (keep FP32 for CPU decode)
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((weights_dir + "/yolo_dfl_weight.bin").c_str(), w, ws);
            load_tensor((weights_dir + "/yolo_dfl_bias.bin").c_str(), b, bs);
            dfl_weight.alloc(w.size());
            dfl_weight.copyFrom(w.data(), w.size());
            dfl_bias.alloc(b.size());
            dfl_bias.copyFrom(b.data(), b.size());
        }

        // Workspace
        max_workspace = std::max(max_workspace, (size_t)(128 * 1024 * 1024));
        workspace.alloc(max_workspace / sizeof(float) + 1);

        // Allocate FP16 feature map buffers
        size_t max_buf = (size_t)B * std::max({
            (size_t)(3 * H * W),
            (size_t)(48 * (H/2) * (W/2)),
            (size_t)(96 * (H/4) * (W/4)),
            (size_t)(192 * (H/8) * (W/8)),
            (size_t)(384 * (H/16) * (W/16)),
            (size_t)(576 * (H/32) * (W/32)),
            (size_t)(1152 * (H/32) * (W/32)),
            (size_t)(960 * (H/16) * (W/16)),
            (size_t)(576 * (H/8) * (W/8)),
        });
        for (int i = 0; i < 30; i++) bufs[i].alloc(max_buf);
        tmp1.alloc(max_buf);
        tmp2.alloc(max_buf);
        tmp3.alloc(max_buf);
        // C2f scratch buffers (pre-allocated to avoid per-call cudaMalloc)
        c2f_chunk0.alloc(max_buf);
        c2f_chunk1.alloc(max_buf);
        for (int i = 0; i < 4; i++) c2f_bn[i].alloc(max_buf);
        c2f_concat.alloc(max_buf);
    }

    struct Detection {
        float x1, y1, x2, y2;
        float score;
        int cls;
        float kp[21 * 3];
        int batch_idx;  // which frame in the batch
    };

    std::vector<Detection> forward(const uint8_t* input_gpu, int N, int inH, int inW) {
        int H = yolo_h, W = yolo_w;

        // Preprocess: resize + normalize to FP16
        {
            int total = N * 3 * H * W;
            yolo_preprocess_kernel<<<(total+255)/256, 256>>>(input_gpu, bufs[0].data, inH, inW, H, W, N);
        }

        // Backbone (all FP16)
        l0.forward(cudnn, bufs[0].data, bufs[1].data, workspace.data, N, H, W);
        silu_half_inplace(bufs[1].data, N * 48 * (H/2) * (W/2));

        l1.forward(cudnn, bufs[1].data, bufs[2].data, workspace.data, N, H/2, W/2);
        silu_half_inplace(bufs[2].data, N * 96 * (H/4) * (W/4));

        c2f_forward(c2f_2, bufs[2].data, bufs[3].data, N, 96, 96, H/4, W/4);

        l3.forward(cudnn, bufs[3].data, bufs[4].data, workspace.data, N, H/4, W/4);
        silu_half_inplace(bufs[4].data, N * 192 * (H/8) * (W/8));

        c2f_forward(c2f_4, bufs[4].data, bufs[5].data, N, 192, 192, H/8, W/8);

        l5.forward(cudnn, bufs[5].data, bufs[6].data, workspace.data, N, H/8, W/8);
        silu_half_inplace(bufs[6].data, N * 384 * (H/16) * (W/16));

        c2f_forward(c2f_6, bufs[6].data, bufs[7].data, N, 384, 384, H/16, W/16);

        l7.forward(cudnn, bufs[7].data, bufs[8].data, workspace.data, N, H/16, W/16);
        silu_half_inplace(bufs[8].data, N * 576 * (H/32) * (W/32));

        c2f_forward(c2f_8, bufs[8].data, bufs[9].data, N, 576, 576, H/32, W/32);

        // SPPF
        {
            int sh = H/32, sw = W/32, hw = sh * sw;
            sppf_cv1.forward(cudnn, bufs[9].data, bufs[10].data, workspace.data, N, sh, sw);
            silu_half_inplace(bufs[10].data, N * 288 * hw);
            maxpool2d_half(bufs[10].data, bufs[11].data, N, 288, sh, sw, 5, 1, 2);
            maxpool2d_half(bufs[11].data, bufs[12].data, N, 288, sh, sw, 5, 1, 2);
            maxpool2d_half(bufs[12].data, bufs[13].data, N, 288, sh, sw, 5, 1, 2);
            int total = N * 1152 * hw;
            concat4_half_kernel<<<(total+255)/256, 256>>>(bufs[14].data,
                bufs[10].data, bufs[11].data, bufs[12].data, bufs[13].data,
                N, 288, 288, 288, 288, hw);
            sppf_cv2.forward(cudnn, bufs[14].data, bufs[15].data, workspace.data, N, sh, sw);
            silu_half_inplace(bufs[15].data, N * 576 * hw);
        }

        // FPN
        upsample_nearest_2x_half(bufs[15].data, bufs[16].data, N, 576, H/32, W/32);
        {
            int hw = (H/16) * (W/16);
            concat2_half_kernel<<<(N*960*hw+255)/256, 256>>>(bufs[17].data, bufs[16].data, bufs[7].data,
                N, 576, 384, hw);
        }
        c2f_forward(c2f_12, bufs[17].data, bufs[18].data, N, 960, 384, H/16, W/16);

        upsample_nearest_2x_half(bufs[18].data, bufs[19].data, N, 384, H/16, W/16);
        {
            int hw = (H/8) * (W/8);
            concat2_half_kernel<<<(N*576*hw+255)/256, 256>>>(bufs[20].data, bufs[19].data, bufs[5].data,
                N, 384, 192, hw);
        }
        c2f_forward(c2f_15, bufs[20].data, bufs[21].data, N, 576, 192, H/8, W/8);

        // PAN
        l16.forward(cudnn, bufs[21].data, bufs[22].data, workspace.data, N, H/8, W/8);
        silu_half_inplace(bufs[22].data, N * 192 * (H/16) * (W/16));
        {
            int hw = (H/16) * (W/16);
            concat2_half_kernel<<<(N*576*hw+255)/256, 256>>>(bufs[23].data, bufs[22].data, bufs[18].data,
                N, 192, 384, hw);
        }
        c2f_forward(c2f_18, bufs[23].data, bufs[24].data, N, 576, 384, H/16, W/16);

        l19.forward(cudnn, bufs[24].data, bufs[25].data, workspace.data, N, H/16, W/16);
        silu_half_inplace(bufs[25].data, N * 384 * (H/32) * (W/32));
        {
            int hw = (H/32) * (W/32);
            concat2_half_kernel<<<(N*960*hw+255)/256, 256>>>(bufs[26].data, bufs[25].data, bufs[15].data,
                N, 384, 576, hw);
        }
        c2f_forward(c2f_21, bufs[26].data, bufs[27].data, N, 960, 576, H/32, W/32);

        // Detection head: 3 scales, convert FP16→FP32 for CPU post-processing
        __half* feat_maps[] = {bufs[21].data, bufs[24].data, bufs[27].data};
        int feat_h[] = {H/8, H/16, H/32};
        int feat_w[] = {W/8, W/16, W/32};
        float host_strides[] = {8.0f, 16.0f, 32.0f};

        std::vector<Detection> all_dets;

        // DFL weights (small, cached on CPU)
        std::vector<float> h_dfl_w(16), h_dfl_b(1);
        dfl_weight.copyTo(h_dfl_w.data(), 16);
        dfl_bias.copyTo(h_dfl_b.data(), 1);

        for (int si = 0; si < 3; si++) {
            int fh = feat_h[si], fw = feat_w[si];
            int hw = fh * fw;

            // Head convolutions (FP16)
            GpuHalfBuf bbox_feat, cls_feat, kp_feat;
            bbox_feat.alloc(N * 64 * hw);
            cls_feat.alloc(N * YOLO_NC * hw);
            kp_feat.alloc(N * 63 * hw);

            head[si].cv2[0].forward(cudnn, feat_maps[si], tmp1.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp1.data, N * 64 * hw);
            head[si].cv2[1].forward(cudnn, tmp1.data, tmp2.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp2.data, N * 64 * hw);
            head[si].cv2[2].forward(cudnn, tmp2.data, bbox_feat.data, workspace.data, N, fh, fw);

            int cls_mid = 192;
            head[si].cv3[0].forward(cudnn, feat_maps[si], tmp1.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp1.data, N * cls_mid * hw);
            head[si].cv3[1].forward(cudnn, tmp1.data, tmp2.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp2.data, N * cls_mid * hw);
            head[si].cv3[2].forward(cudnn, tmp2.data, cls_feat.data, workspace.data, N, fh, fw);
            sigmoid_half_inplace(cls_feat.data, N * YOLO_NC * hw);

            int kp_mid = YOLO_NKP * 3;
            head[si].cv4[0].forward(cudnn, feat_maps[si], tmp1.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp1.data, N * kp_mid * hw);
            head[si].cv4[1].forward(cudnn, tmp1.data, tmp2.data, workspace.data, N, fh, fw);
            silu_half_inplace(tmp2.data, N * kp_mid * hw);
            head[si].cv4[2].forward(cudnn, tmp2.data, kp_feat.data, workspace.data, N, fh, fw);

            // Convert FP16→FP32 on GPU, then download to CPU
            GpuBuf bbox_f, cls_f, kp_f;
            bbox_f.alloc(N * 64 * hw);
            cls_f.alloc(N * YOLO_NC * hw);
            kp_f.alloc(N * 63 * hw);
            half_to_float(bbox_feat.data, bbox_f.data, N * 64 * hw);
            half_to_float(cls_feat.data, cls_f.data, N * YOLO_NC * hw);
            half_to_float(kp_feat.data, kp_f.data, N * 63 * hw);

            std::vector<float> h_bbox(N * 64 * hw), h_cls(N * YOLO_NC * hw), h_kp(N * 63 * hw);
            bbox_f.copyTo(h_bbox.data(), h_bbox.size());
            cls_f.copyTo(h_cls.data(), h_cls.size());
            kp_f.copyTo(h_kp.data(), h_kp.size());

            float stride = host_strides[si];

            for (int n = 0; n < N; n++) {
                for (int y = 0; y < fh; y++) {
                    for (int x = 0; x < fw; x++) {
                        int pos = y * fw + x;

                        float max_score = 0;
                        int max_cls = 0;
                        for (int c = 0; c < YOLO_NC; c++) {
                            float s = h_cls[(n * YOLO_NC + c) * hw + pos];
                            if (s > max_score) { max_score = s; max_cls = c; }
                        }
                        if (max_score < det_conf) continue;

                        float dist[4];
                        for (int d = 0; d < 4; d++) {
                            float vals[16];
                            float maxv = -1e30f;
                            for (int k = 0; k < 16; k++) {
                                vals[k] = h_bbox[(n * 64 + d * 16 + k) * hw + pos];
                                maxv = std::max(maxv, vals[k]);
                            }
                            float sum = 0;
                            for (int k = 0; k < 16; k++) {
                                vals[k] = expf(vals[k] - maxv);
                                sum += vals[k];
                            }
                            float dfl_val = h_dfl_b[0];
                            for (int k = 0; k < 16; k++) {
                                dfl_val += (vals[k] / sum) * h_dfl_w[k];
                            }
                            dist[d] = dfl_val;
                        }

                        float cx = (x + 0.5f) * stride;
                        float cy = (y + 0.5f) * stride;
                        Detection det;
                        det.x1 = cx - dist[0] * stride;
                        det.y1 = cy - dist[1] * stride;
                        det.x2 = cx + dist[2] * stride;
                        det.y2 = cy + dist[3] * stride;
                        det.score = max_score;
                        det.cls = max_cls;
                        det.batch_idx = n;

                        for (int k = 0; k < 21; k++) {
                            float kx = h_kp[(n * 63 + k * 3 + 0) * hw + pos];
                            float ky = h_kp[(n * 63 + k * 3 + 1) * hw + pos];
                            float kc = h_kp[(n * 63 + k * 3 + 2) * hw + pos];
                            det.kp[k * 3 + 0] = (kx * 2.0f + (x - 0.5f)) * stride;
                            det.kp[k * 3 + 1] = (ky * 2.0f + (y - 0.5f)) * stride;
                            det.kp[k * 3 + 2] = 1.0f / (1.0f + expf(-kc));
                        }

                        all_dets.push_back(det);
                    }
                }
            }
        }

        // Per-frame NMS (needed for batched inference)
        std::vector<Detection> result;
        for (int n = 0; n < N; n++) {
            std::vector<Detection> frame_dets;
            for (auto& d : all_dets) if (d.batch_idx == n) frame_dets.push_back(d);
            auto frame_nms = nms(frame_dets, 0.45f);
            result.insert(result.end(), frame_nms.begin(), frame_nms.end());
        }
        return result;
    }

    static float iou(const Detection& a, const Detection& b) {
        float x1 = std::max(a.x1, b.x1);
        float y1 = std::max(a.y1, b.y1);
        float x2 = std::min(a.x2, b.x2);
        float y2 = std::min(a.y2, b.y2);
        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (area_a + area_b - inter + 1e-6f);
    }

    static std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh) {
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

        std::vector<bool> suppressed(dets.size(), false);
        std::vector<Detection> result;

        bool has_class[YOLO_NC] = {};
        for (size_t i = 0; i < dets.size(); i++) {
            if (suppressed[i]) continue;
            result.push_back(dets[i]);
            has_class[dets[i].cls] = true;
            for (size_t j = i + 1; j < dets.size(); j++) {
                if (!suppressed[j] && dets[i].cls == dets[j].cls && iou(dets[i], dets[j]) > iou_thresh)
                    suppressed[j] = true;
            }
            bool all_found = true;
            for (int c = 0; c < YOLO_NC; c++) if (!has_class[c]) all_found = false;
            if (all_found) break;
        }
        if (result.size() > 2) result.resize(2);
        return result;
    }
};

// ============================================================================
// WiLoR ViT Model
// ============================================================================

struct WilorModel {
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;

    // Patch embed: Conv2d(3, 1280, k=16, s=16, p=2)
    ConvLayer patch_embed;

    // Positional embedding [1, 193, 1280] (193 = 192 patches + 1 cls/pos token)
    GpuBuf pos_embed;  // [193, 1280] stored as float32, converted to half

    // Token embeddings
    GpuBuf pose_emb_weight, pose_emb_bias;  // Linear(6, 1280)
    GpuBuf shape_emb_weight, shape_emb_bias; // Linear(10, 1280)
    GpuBuf cam_emb_weight, cam_emb_bias;     // Linear(3, 1280)

    // Init buffers
    GpuBuf init_hand_pose;  // [96]
    GpuBuf init_betas;      // [10]
    GpuBuf init_cam;        // [3]

    // Transformer blocks
    struct TransformerBlock {
        GpuBuf norm1_weight, norm1_bias;  // [1280]
        GpuBuf qkv_weight, qkv_bias;     // [3840, 1280] and [3840]
        GpuBuf proj_weight, proj_bias;    // [1280, 1280] and [1280]
        GpuBuf norm2_weight, norm2_bias;  // [1280]
        GpuBuf fc1_weight, fc1_bias;      // [5120, 1280] and [5120]
        GpuBuf fc2_weight, fc2_bias;      // [1280, 5120] and [1280]

        // FP16 versions of weights for GEMM
        GpuHalfBuf qkv_weight_h, proj_weight_h, fc1_weight_h, fc2_weight_h;
    };
    TransformerBlock blocks[32];

    // Last norm
    GpuBuf last_norm_weight, last_norm_bias;

    // Decode heads
    GpuBuf decpose_weight, decpose_bias;   // [6, 1280] and [6]
    GpuBuf decshape_weight, decshape_bias; // [10, 1280] and [10]
    GpuBuf deccam_weight, deccam_bias;     // [3, 1280] and [3]

    // RefineNet
    ConvLayer refine_first_conv;  // Conv2d(1280, 640, k=1, s=1, p=0)
    ConvTransposeLayer refine_branch0_0; // ConvT(640->320, k4s2p1)
    ConvTransposeLayer refine_branch1_0; // ConvT(640->320, k4s2p1)
    ConvTransposeLayer refine_branch1_1; // ConvT(320->160, k4s2p1)
    GpuBuf refine_dec_pose_w, refine_dec_pose_b;   // [96, 1120]
    GpuBuf refine_dec_shape_w, refine_dec_shape_b;  // [10, 1120]
    GpuBuf refine_dec_cam_w, refine_dec_cam_b;      // [3, 1120]

    // MANO model
    GpuBuf mano_J_regressor;     // [16, 778]
    GpuBuf mano_v_template;      // [778, 3]
    GpuBuf mano_shapedirs;       // [778, 3, 10]
    GpuBuf mano_posedirs;        // [135, 2334]
    GpuBuf mano_lbs_weights;     // [778, 16]
    int mano_parents_host[16];
    int mano_joint_map_host[21];
    int mano_extra_joints_host[5];

    // Workspace
    GpuBuf workspace;
    size_t max_workspace = 0;

    // Temp buffers for ViT forward
    GpuHalfBuf vit_tokens;     // [B, 210, 1280] half
    GpuHalfBuf vit_tmp1, vit_tmp2, vit_tmp3, vit_qkv, vit_attn;
    GpuBuf fp32_tmp1, fp32_tmp2, fp32_tmp3, fp32_tmp4;

    int max_batch;

    void load_buf(GpuBuf& buf, const std::string& path) {
        std::vector<float> data;
        std::vector<int> shape;
        if (!load_tensor(path.c_str(), data, shape)) {
            fprintf(stderr, "Failed to load %s\n", path.c_str());
            exit(1);
        }
        buf.alloc(data.size());
        buf.copyFrom(data.data(), data.size());
    }

    void load_half(GpuHalfBuf& hbuf, const GpuBuf& fbuf, size_t n) {
        hbuf.alloc(n);
        float_to_half(fbuf.data, hbuf.data, n);
    }

    void init(const std::string& dir, int batch) {
        max_batch = batch;
        CUDNN_CHECK(cudnnCreate(&cudnn));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

        std::string d = dir;

        // Patch embed: Conv2d(3, 1280, k=16, s=16, p=2)
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((d + "/wilor_backbone_patch_embed_proj_weight.bin").c_str(), w, ws);
            load_tensor((d + "/wilor_backbone_patch_embed_proj_bias.bin").c_str(), b, bs);
            patch_embed.init(cudnn, batch, 3, 1280, 16, 16, 16, 2, VIT_IMG_H, VIT_IMG_W,
                             w.data(), b.data());
            max_workspace = std::max(max_workspace, patch_embed.workspaceSize);
        }

        // Pos embed [1, 193, 1280]
        load_buf(pos_embed, d + "/wilor_backbone_pos_embed.bin");

        // Token embeddings
        load_buf(pose_emb_weight, d + "/wilor_backbone_pose_emb_weight.bin");
        load_buf(pose_emb_bias, d + "/wilor_backbone_pose_emb_bias.bin");
        load_buf(shape_emb_weight, d + "/wilor_backbone_shape_emb_weight.bin");
        load_buf(shape_emb_bias, d + "/wilor_backbone_shape_emb_bias.bin");
        load_buf(cam_emb_weight, d + "/wilor_backbone_cam_emb_weight.bin");
        load_buf(cam_emb_bias, d + "/wilor_backbone_cam_emb_bias.bin");

        // Init buffers
        load_buf(init_hand_pose, d + "/wilor_backbone_init_hand_pose.bin");
        load_buf(init_betas, d + "/wilor_backbone_init_betas.bin");
        load_buf(init_cam, d + "/wilor_backbone_init_cam.bin");

        // Transformer blocks
        for (int i = 0; i < 32; i++) {
            char prefix[128];
            snprintf(prefix, sizeof(prefix), "%s/wilor_backbone_blocks_%d_", d.c_str(), i);
            std::string p(prefix);

            load_buf(blocks[i].norm1_weight, p + "norm1_weight.bin");
            load_buf(blocks[i].norm1_bias, p + "norm1_bias.bin");
            load_buf(blocks[i].qkv_weight, p + "attn_qkv_weight.bin");
            load_buf(blocks[i].qkv_bias, p + "attn_qkv_bias.bin");
            load_buf(blocks[i].proj_weight, p + "attn_proj_weight.bin");
            load_buf(blocks[i].proj_bias, p + "attn_proj_bias.bin");
            load_buf(blocks[i].norm2_weight, p + "norm2_weight.bin");
            load_buf(blocks[i].norm2_bias, p + "norm2_bias.bin");
            load_buf(blocks[i].fc1_weight, p + "mlp_fc1_weight.bin");
            load_buf(blocks[i].fc1_bias, p + "mlp_fc1_bias.bin");
            load_buf(blocks[i].fc2_weight, p + "mlp_fc2_weight.bin");
            load_buf(blocks[i].fc2_bias, p + "mlp_fc2_bias.bin");

            // Convert weights to FP16
            load_half(blocks[i].qkv_weight_h, blocks[i].qkv_weight, 3840 * 1280);
            load_half(blocks[i].proj_weight_h, blocks[i].proj_weight, 1280 * 1280);
            load_half(blocks[i].fc1_weight_h, blocks[i].fc1_weight, 5120 * 1280);
            load_half(blocks[i].fc2_weight_h, blocks[i].fc2_weight, 1280 * 5120);
        }

        // Last norm
        load_buf(last_norm_weight, d + "/wilor_backbone_last_norm_weight.bin");
        load_buf(last_norm_bias, d + "/wilor_backbone_last_norm_bias.bin");

        // Decode heads
        load_buf(decpose_weight, d + "/wilor_backbone_decpose_weight.bin");
        load_buf(decpose_bias, d + "/wilor_backbone_decpose_bias.bin");
        load_buf(decshape_weight, d + "/wilor_backbone_decshape_weight.bin");
        load_buf(decshape_bias, d + "/wilor_backbone_decshape_bias.bin");
        load_buf(deccam_weight, d + "/wilor_backbone_deccam_weight.bin");
        load_buf(deccam_bias, d + "/wilor_backbone_deccam_bias.bin");

        // RefineNet
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((d + "/wilor_refine_net_deconv_first_conv_0_weight.bin").c_str(), w, ws);
            load_tensor((d + "/wilor_refine_net_deconv_first_conv_0_bias.bin").c_str(), b, bs);
            refine_first_conv.init(cudnn, batch, 1280, 640, 1, 1, 1, 0, VIT_PATCH_H, VIT_PATCH_W,
                                   w.data(), b.data());
            max_workspace = std::max(max_workspace, refine_first_conv.workspaceSize);
        }
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((d + "/wilor_refine_net_deconv_branch0_0_weight.bin").c_str(), w, ws);
            load_tensor((d + "/wilor_refine_net_deconv_branch0_0_bias.bin").c_str(), b, bs);
            refine_branch0_0.init(cudnn, batch, 640, 320, 4, 4, 2, 1, VIT_PATCH_H, VIT_PATCH_W,
                                  w.data(), b.data());
            max_workspace = std::max(max_workspace, refine_branch0_0.workspaceSize);
        }
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((d + "/wilor_refine_net_deconv_branch1_0_weight.bin").c_str(), w, ws);
            load_tensor((d + "/wilor_refine_net_deconv_branch1_0_bias.bin").c_str(), b, bs);
            refine_branch1_0.init(cudnn, batch, 640, 320, 4, 4, 2, 1, VIT_PATCH_H, VIT_PATCH_W,
                                  w.data(), b.data());
            max_workspace = std::max(max_workspace, refine_branch1_0.workspaceSize);
        }
        {
            std::vector<float> w, b;
            std::vector<int> ws, bs;
            load_tensor((d + "/wilor_refine_net_deconv_branch1_1_weight.bin").c_str(), w, ws);
            load_tensor((d + "/wilor_refine_net_deconv_branch1_1_bias.bin").c_str(), b, bs);
            refine_branch1_1.init(cudnn, batch, 320, 160, 4, 4, 2, 1, VIT_PATCH_H*2, VIT_PATCH_W*2,
                                  w.data(), b.data());
            max_workspace = std::max(max_workspace, refine_branch1_1.workspaceSize);
        }

        load_buf(refine_dec_pose_w, d + "/wilor_refine_net_dec_pose_weight.bin");
        load_buf(refine_dec_pose_b, d + "/wilor_refine_net_dec_pose_bias.bin");
        load_buf(refine_dec_shape_w, d + "/wilor_refine_net_dec_shape_weight.bin");
        load_buf(refine_dec_shape_b, d + "/wilor_refine_net_dec_shape_bias.bin");
        load_buf(refine_dec_cam_w, d + "/wilor_refine_net_dec_cam_weight.bin");
        load_buf(refine_dec_cam_b, d + "/wilor_refine_net_dec_cam_bias.bin");

        // MANO
        load_buf(mano_J_regressor, d + "/wilor_mano_J_regressor.bin");
        load_buf(mano_v_template, d + "/wilor_mano_v_template.bin");
        load_buf(mano_shapedirs, d + "/wilor_mano_shapedirs.bin");
        load_buf(mano_posedirs, d + "/wilor_mano_posedirs.bin");
        load_buf(mano_lbs_weights, d + "/wilor_mano_lbs_weights.bin");

        // Load int tensors to host
        {
            std::vector<int> data;
            std::vector<int> shape;
            load_int_tensor((d + "/wilor_mano_parents.bin").c_str(), data, shape);
            for (int i = 0; i < 16; i++) mano_parents_host[i] = data[i];
            load_int_tensor((d + "/wilor_mano_joint_map.bin").c_str(), data, shape);
            for (int i = 0; i < 21; i++) mano_joint_map_host[i] = data[i];
            load_int_tensor((d + "/wilor_mano_extra_joints_idxs.bin").c_str(), data, shape);
            for (int i = 0; i < 5; i++) mano_extra_joints_host[i] = data[i];
        }

        // Workspace
        max_workspace = std::max(max_workspace, (size_t)(256 * 1024 * 1024));
        workspace.alloc(max_workspace / sizeof(float) + 1);

        // Allocate ViT buffers
        int B = batch;
        int T = VIT_TOTAL_TOKENS;  // 210
        int D = VIT_EMBED;         // 1280

        vit_tokens.alloc(B * T * D);
        vit_tmp1.alloc(B * T * D * 4); // for QKV (3x wider)
        vit_tmp2.alloc(B * T * D * 4);
        vit_tmp3.alloc(B * T * D * 4);
        vit_qkv.alloc(B * T * 3 * D);
        vit_attn.alloc(B * VIT_HEADS * T * T);

        fp32_tmp1.alloc(B * std::max({T * D, MANO_N_VERTS * 16, 1120}));
        fp32_tmp2.alloc(B * std::max({T * D, MANO_N_VERTS * 3 * 4, 1120}));
        fp32_tmp3.alloc(B * std::max({T * D, MANO_N_VERTS * 3, 1120}));
        fp32_tmp4.alloc(B * std::max({96, MANO_N_VERTS * 3, T * D}));
    }

    // Run ViT + RefineNet + MANO forward pass on preprocessed crops
    // input_crops: [B, 3, 256, 192] float32 (preprocessed, normalized)
    // Output: per-hand keypoints
    struct HandResult {
        float kp3d[21 * 3];
        float kp2d[21 * 2];
        float cam_t[3];
    };

    void forward(float* input_crops, int B,
                 float* box_centers, float* box_sizes, float* img_sizes, float* rights,
                 HandResult* results) {
        if (B == 0) return;

        int T = VIT_TOTAL_TOKENS;
        int D = VIT_EMBED;

        // ── Step 1: Patch embed ──
        // Input to ViT: x[:,:,:,32:-32] — but we skip the crop for compatibility
        // (see comment at top about pos_embed size mismatch)
        // Patch embed: [B, 3, 256, 192] -> [B, 1280, 16, 12]
        patch_embed.forward(cudnn, input_crops, fp32_tmp1.data, workspace.data, B, VIT_IMG_H, VIT_IMG_W);

        // Transpose NCHW -> N(HW)C: [B, 1280, 16, 12] -> [B, 192, 1280]
        int Hp = VIT_PATCH_H, Wp = VIT_PATCH_W;
        int num_patches = Hp * Wp;  // 192

        // Convert patch embeddings to FP16 and arrange as [B, 192, 1280]
        // Currently fp32_tmp1 is [B, 1280, 16, 12] in NCHW
        // Need [B, 192, 1280] = [B, H*W, C]

        // Transpose NCHW -> NHW,C using a kernel
        transpose_nchw_to_nhwc(fp32_tmp1.data, fp32_tmp2.data, B, D, num_patches);

        // ── Step 2: Build token sequence ──
        // tokens = [pose_tokens(16), shape_token(1), cam_token(1), image_tokens(192)]
        // pose_tokens = pose_emb(init_hand_pose.reshape(1, 16, 6)) -> [B, 16, 1280]
        // shape_token = shape_emb(init_betas) -> [B, 1, 1280]
        // cam_token = cam_emb(init_cam) -> [B, 1, 1280]

        // Build all tokens in fp32_tmp3 [B, 210, 1280], then convert to half

        // Pose tokens: init_hand_pose [96] reshaped to [16, 6], then Linear(6, 1280)
        // pose_emb: weight [1280, 6], bias [1280]
        // For each of 16 tokens: out = init_pose[t*6:(t+1)*6] @ weight.T + bias
        build_vit_tokens(fp32_tmp3.data, fp32_tmp2.data, B, num_patches);

        // Add positional embedding
        // pos_embed is [1, 193, 1280]. Usage: x = x + pos_embed[:, 1:] + pos_embed[:, :1]
        // So we add pos_embed[1:193] (192 values) to image tokens, and pos_embed[0] to all
        add_pos_embed(fp32_tmp3.data, B, T, D);

        // Convert to FP16
        float_to_half(fp32_tmp3.data, vit_tokens.data, B * T * D);

        // ── Step 3: Transformer blocks ──
        int skip_set[] = {25, 27, 26, 23, 24};  // blocks to skip
        for (int bi = 0; bi < 32; bi++) {
            bool skip = false;
            for (int s = 0; s < 5; s++) {
                if (skip_set[s] == bi) { skip = true; break; }
            }
            if (skip) continue;
            transformer_block(bi, B, T, D);
        }

        // ── Step 4: Last norm + decode ──
        layernorm_half(vit_tokens.data, vit_tmp1.data,
                       last_norm_weight.data, last_norm_bias.data, B * T, D);

        // Convert output to FP32 for decode
        half_to_float(vit_tmp1.data, fp32_tmp1.data, B * T * D);

        // Decode pose: tokens 0..15 -> decpose [1280, 6] per token
        // Shape: tokens 16 -> decshape [1280, 10]
        // Cam: tokens 17 -> deccam [1280, 3]

        // Allocate output arrays
        std::vector<float> h_pred_pose(B * 96);
        std::vector<float> h_pred_betas(B * 10);
        std::vector<float> h_pred_cam(B * 3);
        std::vector<float> h_img_feat(B * num_patches * D);

        decode_vit_output(fp32_tmp1.data, B, T, D,
                          h_pred_pose.data(), h_pred_betas.data(), h_pred_cam.data(),
                          h_img_feat.data(), num_patches);

        // ── Step 5: First MANO pass (temp) ──
        std::vector<float> h_temp_verts(B * 778 * 3);
        run_mano(h_pred_pose.data(), h_pred_betas.data(), B, h_temp_verts.data(), nullptr);

        // ── Step 6: RefineNet ──
        // img_feat [B, 192, 1280] -> [B, 1280, 16, 12]
        // first_conv -> deconv branches -> sample vertices -> decode deltas
        std::vector<float> h_delta_pose(B * 96);
        std::vector<float> h_delta_betas(B * 10);
        std::vector<float> h_delta_cam(B * 3);

        run_refine_net(h_img_feat.data(), h_temp_verts.data(),
                       h_pred_cam.data(), B,
                       h_delta_pose.data(), h_delta_betas.data(), h_delta_cam.data());

        // Add deltas to predictions
        // Read back init values
        std::vector<float> h_init_pose(96), h_init_betas(10), h_init_cam(3);
        init_hand_pose.copyTo(h_init_pose.data(), 96);
        init_betas.copyTo(h_init_betas.data(), 10);
        init_cam.copyTo(h_init_cam.data(), 3);

        for (int b = 0; b < B; b++) {
            for (int i = 0; i < 96; i++)
                h_pred_pose[b * 96 + i] += h_delta_pose[b * 96 + i];
            for (int i = 0; i < 10; i++)
                h_pred_betas[b * 10 + i] += h_delta_betas[b * 10 + i];
            for (int i = 0; i < 3; i++)
                h_pred_cam[b * 3 + i] += h_delta_cam[b * 3 + i];
        }

        // ── Step 7: Final MANO pass ──
        std::vector<float> h_final_verts(B * 778 * 3);
        std::vector<float> h_final_joints(B * 21 * 3);
        run_mano(h_pred_pose.data(), h_pred_betas.data(), B,
                 h_final_verts.data(), h_final_joints.data());

        // ── Step 8: Perspective projection + post-processing ──
        for (int b = 0; b < B; b++) {
            float scale = h_pred_cam[b * 3 + 0];
            float tx_crop = h_pred_cam[b * 3 + 1];
            float ty_crop = h_pred_cam[b * 3 + 2];
            float tz = 2.0f * FOCAL_LENGTH / ((float)VIT_IMG_H * scale + 1e-9f);
            float cam_t[3] = {tx_crop, ty_crop, tz};

            float is_right = rights[b];
            float multiplier = 2.0f * is_right - 1.0f;

            // Flip cam x for handedness
            cam_t[0] = multiplier * cam_t[0];

            // Project 3D joints to 2D
            float fl = FOCAL_LENGTH / (float)VIT_IMG_H;

            for (int j = 0; j < 21; j++) {
                float jx = h_final_joints[b * 63 + j * 3 + 0];
                float jy = h_final_joints[b * 63 + j * 3 + 1];
                float jz = h_final_joints[b * 63 + j * 3 + 2];

                // Flip x for handedness
                jx = multiplier * jx;

                // Translate
                float px = jx + cam_t[0];
                float py = jy + cam_t[1];
                float pz = jz + cam_t[2];

                // 3D keypoints in camera frame
                results[b].kp3d[j * 3 + 0] = px;
                results[b].kp3d[j * 3 + 1] = py;
                results[b].kp3d[j * 3 + 2] = pz;

                // Perspective projection
                float u = fl * px / pz;
                float v = fl * py / pz;

                // Flip x for handedness on 2D
                u = multiplier * u;

                results[b].kp2d[j * 2 + 0] = u;
                results[b].kp2d[j * 2 + 1] = v;
            }

            // cam_crop_to_full
            float img_w = img_sizes[b * 2 + 0];
            float img_h = img_sizes[b * 2 + 1];
            float cx = box_centers[b * 2 + 0];
            float cy = box_centers[b * 2 + 1];
            float bs = box_sizes[b];
            float w2 = img_w / 2.0f;
            float h2 = img_h / 2.0f;

            float cam_scale = h_pred_cam[b * 3 + 0];
            float cam_tx = multiplier * h_pred_cam[b * 3 + 1];
            float cam_ty = h_pred_cam[b * 3 + 2];

            float bs_scaled = bs * cam_scale + 1e-9f;
            float fl_full = FOCAL_LENGTH / (float)VIT_IMG_H * std::max(img_w, img_h);
            float full_tz = 2.0f * fl_full / bs_scaled;
            float full_tx = (2.0f * (cx - w2) / bs_scaled) + cam_tx;
            float full_ty = (2.0f * (cy - h2) / bs_scaled) + cam_ty;

            results[b].cam_t[0] = full_tx;
            results[b].cam_t[1] = full_ty;
            results[b].cam_t[2] = full_tz;

            // Transform 2D keypoints: kp2d = kp2d * box_size + box_center
            for (int j = 0; j < 21; j++) {
                results[b].kp2d[j * 2 + 0] = results[b].kp2d[j * 2 + 0] * bs + cx;
                results[b].kp2d[j * 2 + 1] = results[b].kp2d[j * 2 + 1] * bs + cy;
            }

            // 3D keypoints: joints + cam_t in camera frame
            // Already computed above as kp3d
        }
    }

    // Helper: transpose NCHW -> N(HW)C
    static void transpose_nchw_to_nhwc(const float* in, float* out, int B, int C, int HW) {
        // Simple CPU-side memcpy approach won't work for GPU data.
        // Need a GPU kernel.
        transpose_nchw_to_nhwc_kernel<<<(B * HW * C + 255) / 256, 256>>>(in, out, B, C, HW);
    }

    // Transformer block forward
    void transformer_block(int bi, int B, int T, int D) {
        auto& blk = blocks[bi];

        // LayerNorm1
        layernorm_half(vit_tokens.data, vit_tmp1.data,
                       blk.norm1_weight.data, blk.norm1_bias.data, B * T, D);

        // QKV GEMM: [B*T, D] x [D, 3D] -> [B*T, 3D]
        // Using cublasGemmEx with FP16 inputs, FP32 accumulation
        {
            int M = B * T;
            int N_out = 3 * D;
            int K = D;
            float alpha_f = 1.0f;
            float beta_f = 0.0f;

            // C = A * B^T where A=[M,K], B=[N,K] -> C=[M,N]
            // cuBLAS is column-major, so we compute C^T = B * A^T
            // Result stored in vit_qkv [B*T, 3*D]
            CUBLAS_CHECK(cublasGemmEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N_out, M, K,
                &alpha_f,
                blk.qkv_weight_h.data, CUDA_R_16F, K,
                vit_tmp1.data, CUDA_R_16F, K,
                &beta_f,
                vit_qkv.data, CUDA_R_16F, N_out,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));

            // Add bias: broadcast [3*D] over [B*T, 3*D]
            add_bias_half(vit_qkv.data, blk.qkv_bias.data, B * T, 3 * D);
        }

        // Multi-head attention
        // qkv: [B*T, 3*D] -> Q[B,H,T,d], K[B,H,T,d], V[B,H,T,d]
        // where H=16, d=80
        // Reshape: [B, T, 3, H, d] -> [3, B, H, T, d]
        // Q*K^T / sqrt(d) -> softmax -> * V
        {
            int H = VIT_HEADS;
            int d = VIT_HEAD_DIM;

            // Reshape QKV: vit_qkv is [B*T, 3*D] in row-major
            // Need: Q=[B*H, T, d], K=[B*H, T, d], V=[B*H, T, d]
            // Use scaled_dot_product_attention via cuBLAS batched GEMM

            // Q = qkv[:, 0:D], K = qkv[:, D:2D], V = qkv[:, 2D:3D]
            // Then reshape each to [B, H, T, d]

            // For cuBLAS batched GEMM on attention:
            // attn = Q @ K^T -> [B*H, T, T], scale by 1/sqrt(d)
            // Then softmax, then attn @ V -> [B*H, T, d]

            // First reshape qkv from [B*T, 3D] to Q/K/V as [B*H, T, d]
            // This requires a permute kernel
            reshape_qkv_to_heads(vit_qkv.data, vit_tmp1.data, vit_tmp2.data, vit_tmp3.data,
                                 B, T, H, d);
            // vit_tmp1 = Q [B*H, T, d]
            // vit_tmp2 = K [B*H, T, d]
            // vit_tmp3 = V [B*H, T, d]

            // Q @ K^T -> attn [B*H, T, T]
            {
                float alpha_f = 1.0f / sqrtf((float)d);
                float beta_f = 0.0f;

                // A=Q [B*H*T, d], B=K [B*H*T, d]
                // C = A * B^T -> [B*H, T, T]
                // cuBLAS col-major: C^T = B * A^T
                // strided batched GEMM
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    T, T, d,
                    &alpha_f,
                    vit_tmp2.data, CUDA_R_16F, d, T * d,  // K: [T, d] per batch
                    vit_tmp1.data, CUDA_R_16F, d, T * d,  // Q: [T, d] per batch
                    &beta_f,
                    vit_attn.data, CUDA_R_16F, T, T * T,
                    B * H,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
            }


            // Softmax over last dim
            softmax_half(vit_attn.data, B * H * T, T);

            // attn @ V -> [B*H, T, d]
            {
                float alpha_f = 1.0f;
                float beta_f = 0.0f;

                CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d, T, T,
                    &alpha_f,
                    vit_tmp3.data, CUDA_R_16F, d, T * d,  // V
                    vit_attn.data, CUDA_R_16F, T, T * T,  // attn
                    &beta_f,
                    vit_tmp1.data, CUDA_R_16F, d, T * d,  // output
                    B * H,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
            }

            // Reshape back: [B*H, T, d] -> [B*T, D]
            reshape_heads_to_seq(vit_tmp1.data, vit_tmp2.data, B, T, H, d);
            // vit_tmp2 = [B*T, D]

            // Output projection: [B*T, D] x [D, D] -> [B*T, D]
            {
                float alpha_f = 1.0f;
                float beta_f = 0.0f;
                int M = B * T;

                CUBLAS_CHECK(cublasGemmEx(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    D, M, D,
                    &alpha_f,
                    blk.proj_weight_h.data, CUDA_R_16F, D,
                    vit_tmp2.data, CUDA_R_16F, D,
                    &beta_f,
                    vit_tmp1.data, CUDA_R_16F, D,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));

                add_bias_half(vit_tmp1.data, blk.proj_bias.data, B * T, D);
            }
        }

        // Residual add
        add_inplace_half(vit_tokens.data, vit_tmp1.data, B * T * D);

        // LayerNorm2
        layernorm_half(vit_tokens.data, vit_tmp1.data,
                       blk.norm2_weight.data, blk.norm2_bias.data, B * T, D);

        // MLP: fc1 [D -> 4D] + GELU + fc2 [4D -> D]
        {
            int M = B * T;
            float alpha_f = 1.0f;
            float beta_f = 0.0f;

            // fc1
            CUBLAS_CHECK(cublasGemmEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                VIT_MLP, M, D,
                &alpha_f,
                blk.fc1_weight_h.data, CUDA_R_16F, D,
                vit_tmp1.data, CUDA_R_16F, D,
                &beta_f,
                vit_tmp2.data, CUDA_R_16F, VIT_MLP,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
            add_bias_half(vit_tmp2.data, blk.fc1_bias.data, M, VIT_MLP);
            gelu_half_inplace(vit_tmp2.data, M * VIT_MLP);

            // fc2
            CUBLAS_CHECK(cublasGemmEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                D, M, VIT_MLP,
                &alpha_f,
                blk.fc2_weight_h.data, CUDA_R_16F, VIT_MLP,
                vit_tmp2.data, CUDA_R_16F, VIT_MLP,
                &beta_f,
                vit_tmp1.data, CUDA_R_16F, D,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT));
            add_bias_half(vit_tmp1.data, blk.fc2_bias.data, M, D);
        }

        // Residual add
        add_inplace_half(vit_tokens.data, vit_tmp1.data, B * T * D);
    }

    // Add bias (FP32 bias broadcast to FP16 data)
    static void add_bias_half(__half* data, const float* bias, int rows, int cols) {
        add_bias_half_kernel<<<(rows * cols + 255) / 256, 256>>>(data, bias, rows, cols);
    }

    // Build initial VIT token sequence
    void build_vit_tokens(float* output, const float* img_patches, int B, int num_patches) {
        build_vit_tokens_kernel<<<(B * VIT_TOTAL_TOKENS * VIT_EMBED + 255) / 256, 256>>>(
            output, img_patches,
            init_hand_pose.data, init_betas.data, init_cam.data,
            pose_emb_weight.data, pose_emb_bias.data,
            shape_emb_weight.data, shape_emb_bias.data,
            cam_emb_weight.data, cam_emb_bias.data,
            B, VIT_TOTAL_TOKENS, VIT_EMBED, num_patches);
    }

    // Add positional embedding
    void add_pos_embed(float* tokens, int B, int T, int D) {
        // pos_embed: [1, 193, 1280]
        // x = x + pos_embed[:, 1:] + pos_embed[:, :1]
        // For image tokens (indices 18..209): add pos_embed[1+i] where i=0..191
        // For all tokens: add pos_embed[0]
        add_pos_embed_kernel<<<(B * T * D + 255) / 256, 256>>>(
            tokens, pos_embed.data, B, T, D, VIT_POSE_TOKENS + 2);
    }

    // Decode ViT output to pose/shape/cam
    void decode_vit_output(const float* vit_out, int B, int T, int D,
                           float* pred_pose, float* pred_betas, float* pred_cam,
                           float* img_feat, int num_patches) {
        // pose: tokens 0..15 -> decpose [1280, 6] -> 16*6=96
        // shape: token 16 -> decshape [1280, 10] -> 10
        // cam: token 17 -> deccam [1280, 3] -> 3
        // img: tokens 18..209 -> img_feat [B, 192, 1280]

        // Download to CPU for decode (small tensors)
        std::vector<float> h_vit(B * T * D);
        CUDA_CHECK(cudaMemcpy(h_vit.data(), vit_out, B * T * D * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_decpose_w(6 * D), h_decpose_b(6);
        std::vector<float> h_decshape_w(10 * D), h_decshape_b(10);
        std::vector<float> h_deccam_w(3 * D), h_deccam_b(3);
        decpose_weight.copyTo(h_decpose_w.data(), 6 * D);
        decpose_bias.copyTo(h_decpose_b.data(), 6);
        decshape_weight.copyTo(h_decshape_w.data(), 10 * D);
        decshape_bias.copyTo(h_decshape_b.data(), 10);
        deccam_weight.copyTo(h_deccam_w.data(), 3 * D);
        deccam_bias.copyTo(h_deccam_b.data(), 3);

        std::vector<float> h_init_pose(96), h_init_betas(10), h_init_cam(3);
        init_hand_pose.copyTo(h_init_pose.data(), 96);
        init_betas.copyTo(h_init_betas.data(), 10);
        init_cam.copyTo(h_init_cam.data(), 3);

        for (int b = 0; b < B; b++) {
            // Pose tokens: 0..15, each decoded with decpose [6, 1280]
            for (int t = 0; t < 16; t++) {
                const float* tok = &h_vit[(b * T + t) * D];
                for (int o = 0; o < 6; o++) {
                    float val = h_decpose_b[o];
                    for (int d = 0; d < D; d++) val += tok[d] * h_decpose_w[o * D + d];
                    pred_pose[b * 96 + t * 6 + o] = val + h_init_pose[t * 6 + o];
                }
            }
            // Shape token: index 16
            {
                const float* tok = &h_vit[(b * T + 16) * D];
                for (int o = 0; o < 10; o++) {
                    float val = h_decshape_b[o];
                    for (int d = 0; d < D; d++) val += tok[d] * h_decshape_w[o * D + d];
                    pred_betas[b * 10 + o] = val + h_init_betas[o];
                }
            }
            // Cam token: index 17
            {
                const float* tok = &h_vit[(b * T + 17) * D];
                for (int o = 0; o < 3; o++) {
                    float val = h_deccam_b[o];
                    for (int d = 0; d < D; d++) val += tok[d] * h_deccam_w[o * D + d];
                    pred_cam[b * 3 + o] = val + h_init_cam[o];
                }
            }
            // Image features: tokens 18..209
            for (int t = 0; t < num_patches; t++) {
                memcpy(&img_feat[(b * num_patches + t) * D],
                       &h_vit[(b * T + 18 + t) * D], D * sizeof(float));
            }
        }
    }

    // Run MANO forward pass
    void run_mano(const float* pred_pose_6d, const float* pred_betas, int B,
                  float* out_verts, float* out_joints) {
        // 1. rot6d -> rotmat [B, 16, 3, 3]
        std::vector<float> rotmats(B * 16 * 9);
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < 16; j++) {
                const float* r6 = &pred_pose_6d[b * 96 + j * 6];
                float* R = &rotmats[(b * 16 + j) * 9];
                rot6d_to_rotmat(r6, R);
            }
        }

        // Download MANO constants
        std::vector<float> h_v_template(778 * 3), h_shapedirs(778 * 3 * 10);
        std::vector<float> h_posedirs(135 * 2334), h_J_reg(16 * 778), h_lbs_w(778 * 16);
        mano_v_template.copyTo(h_v_template.data(), 778 * 3);
        mano_shapedirs.copyTo(h_shapedirs.data(), 778 * 3 * 10);
        mano_posedirs.copyTo(h_posedirs.data(), 135 * 2334);
        mano_J_regressor.copyTo(h_J_reg.data(), 16 * 778);
        mano_lbs_weights.copyTo(h_lbs_w.data(), 778 * 16);

        for (int b = 0; b < B; b++) {
            const float* betas = &pred_betas[b * 10];
            const float* rots = &rotmats[b * 16 * 9];

            // v_shaped = v_template + shapedirs @ betas
            // shapedirs: [778, 3, 10], betas: [10]
            float v_shaped[778 * 3];
            for (int v = 0; v < 778; v++) {
                for (int d = 0; d < 3; d++) {
                    float val = h_v_template[v * 3 + d];
                    for (int s = 0; s < 10; s++) {
                        val += h_shapedirs[(v * 3 + d) * 10 + s] * betas[s];
                    }
                    v_shaped[v * 3 + d] = val;
                }
            }

            // J = J_regressor @ v_shaped
            float J[16 * 3];
            for (int j = 0; j < 16; j++) {
                for (int d = 0; d < 3; d++) {
                    float val = 0;
                    for (int v = 0; v < 778; v++) {
                        val += h_J_reg[j * 778 + v] * v_shaped[v * 3 + d];
                    }
                    J[j * 3 + d] = val;
                }
            }

            // Pose offsets: posedirs @ pose_feature
            // pose_feature = (rotmats[:, 1:] - I).flatten() -> [135]
            // posedirs: [135, 2334] -> [2334] = [778*3]
            float pose_feat[135];
            for (int j = 1; j < 16; j++) {
                for (int r = 0; r < 9; r++) {
                    float I_val = (r % 4 == 0) ? 1.0f : 0.0f;  // identity diagonal
                    pose_feat[(j - 1) * 9 + r] = rots[j * 9 + r] - I_val;
                }
            }

            float v_posed[778 * 3];
            for (int v = 0; v < 778; v++) {
                for (int d = 0; d < 3; d++) {
                    float offset = 0;
                    for (int p = 0; p < 135; p++) {
                        offset += h_posedirs[p * 2334 + v * 3 + d] * pose_feat[p];
                    }
                    v_posed[v * 3 + d] = v_shaped[v * 3 + d] + offset;
                }
            }

            // Forward kinematics
            float transforms[16 * 16];  // 16 joints x 4x4 transform
            batch_rigid_transform(rots, J, transforms);

            // Skinning: T = W @ A, vertices = T @ v_posed_homo
            for (int v = 0; v < 778; v++) {
                float T[16] = {0};
                for (int j = 0; j < 16; j++) {
                    float w = h_lbs_w[v * 16 + j];
                    for (int k = 0; k < 16; k++) {
                        T[k] += w * transforms[j * 16 + k];
                    }
                }
                // Apply transform to v_posed
                float vx = v_posed[v * 3 + 0];
                float vy = v_posed[v * 3 + 1];
                float vz = v_posed[v * 3 + 2];
                out_verts[b * 778 * 3 + v * 3 + 0] = T[0]*vx + T[1]*vy + T[2]*vz + T[3];
                out_verts[b * 778 * 3 + v * 3 + 1] = T[4]*vx + T[5]*vy + T[6]*vz + T[7];
                out_verts[b * 778 * 3 + v * 3 + 2] = T[8]*vx + T[9]*vy + T[10]*vz + T[11];
            }

            // Extract joints: J_regressor @ vertices + extra finger tips
            if (out_joints) {
                float base_joints[16 * 3];
                for (int j = 0; j < 16; j++) {
                    for (int d = 0; d < 3; d++) {
                        float val = 0;
                        for (int v = 0; v < 778; v++) {
                            val += h_J_reg[j * 778 + v] * out_verts[b * 778 * 3 + v * 3 + d];
                        }
                        base_joints[j * 3 + d] = val;
                    }
                }

                // Extra joints from vertices (fingertips)
                float extra_joints[5 * 3];
                for (int i = 0; i < 5; i++) {
                    int vi = mano_extra_joints_host[i];
                    for (int d = 0; d < 3; d++) {
                        extra_joints[i * 3 + d] = out_verts[b * 778 * 3 + vi * 3 + d];
                    }
                }

                // Concatenate and reorder: [base_joints(16), extra_joints(5)] -> joint_map -> 21 joints
                float all_joints[21 * 3];
                for (int i = 0; i < 16; i++) {
                    for (int d = 0; d < 3; d++) all_joints[i * 3 + d] = base_joints[i * 3 + d];
                }
                for (int i = 0; i < 5; i++) {
                    for (int d = 0; d < 3; d++) all_joints[(16 + i) * 3 + d] = extra_joints[i * 3 + d];
                }

                // Apply joint_map reordering
                for (int i = 0; i < 21; i++) {
                    int src = mano_joint_map_host[i];
                    for (int d = 0; d < 3; d++) {
                        out_joints[b * 63 + i * 3 + d] = all_joints[src * 3 + d];
                    }
                }
            }
        }
    }

    // rot6d -> rotation matrix (Gram-Schmidt)
    static void rot6d_to_rotmat(const float* r6, float* R) {
        // r6: [6] = [a1(3), a2(3)] in column-major (reshape(-1,2,3).permute(0,2,1))
        // PyTorch: x.reshape(-1,2,3).permute(0,2,1) -> [B,3,2]
        // a1 = x[:,:,0], a2 = x[:,:,1]
        // So a1 = [r6[0], r6[2], r6[4]], a2 = [r6[1], r6[3], r6[5]]
        float a1[3] = {r6[0], r6[2], r6[4]};
        float a2[3] = {r6[1], r6[3], r6[5]};

        // b1 = normalize(a1)
        float norm1 = sqrtf(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2] + 1e-8f);
        float b1[3] = {a1[0]/norm1, a1[1]/norm1, a1[2]/norm1};

        // b2 = normalize(a2 - dot(b1,a2)*b1)
        float dot = b1[0]*a2[0] + b1[1]*a2[1] + b1[2]*a2[2];
        float a2p[3] = {a2[0] - dot*b1[0], a2[1] - dot*b1[1], a2[2] - dot*b1[2]};
        float norm2 = sqrtf(a2p[0]*a2p[0] + a2p[1]*a2p[1] + a2p[2]*a2p[2] + 1e-8f);
        float b2[3] = {a2p[0]/norm2, a2p[1]/norm2, a2p[2]/norm2};

        // b3 = cross(b1, b2)
        float b3[3] = {
            b1[1]*b2[2] - b1[2]*b2[1],
            b1[2]*b2[0] - b1[0]*b2[2],
            b1[0]*b2[1] - b1[1]*b2[0]
        };

        // R = stack([b1, b2, b3], dim=-1) -> column major
        // R[row][col]: R[:,0]=b1, R[:,1]=b2, R[:,2]=b3
        R[0] = b1[0]; R[1] = b2[0]; R[2] = b3[0];
        R[3] = b1[1]; R[4] = b2[1]; R[5] = b3[1];
        R[6] = b1[2]; R[7] = b2[2]; R[8] = b3[2];
    }

    // Forward kinematics
    void batch_rigid_transform(const float* rot_mats, const float* J, float* out_transforms) {
        // rot_mats: [16, 3, 3] (row-major: R[0..8])
        // J: [16, 3]
        // out_transforms: [16, 4, 4] (rel_transforms for skinning)

        // Compute relative joints
        float rel_J[16 * 3];
        for (int d = 0; d < 3; d++) rel_J[d] = J[d];  // root
        for (int j = 1; j < 16; j++) {
            int p = mano_parents_host[j];
            for (int d = 0; d < 3; d++) rel_J[j * 3 + d] = J[j * 3 + d] - J[p * 3 + d];
        }

        // Build local transforms: [R | t; 0 0 0 1]
        float local_T[16 * 16];
        for (int j = 0; j < 16; j++) {
            const float* R = &rot_mats[j * 9];
            float* T = &local_T[j * 16];
            T[0] = R[0]; T[1] = R[1]; T[2] = R[2]; T[3] = rel_J[j*3+0];
            T[4] = R[3]; T[5] = R[4]; T[6] = R[5]; T[7] = rel_J[j*3+1];
            T[8] = R[6]; T[9] = R[7]; T[10]= R[8]; T[11]= rel_J[j*3+2];
            T[12]= 0;    T[13]= 0;    T[14]= 0;    T[15]= 1;
        }

        // Chain transforms through kinematic tree
        float global_T[16 * 16];
        memcpy(global_T, local_T, 16 * sizeof(float));  // root
        for (int j = 1; j < 16; j++) {
            int p = mano_parents_host[j];
            mat4_mul(&global_T[p * 16], &local_T[j * 16], &global_T[j * 16]);
        }

        // Compute rel_transforms = global_T - global_T @ J_homo
        // This subtracts the rest-pose joint positions
        for (int j = 0; j < 16; j++) {
            float* G = &global_T[j * 16];
            float jx = J[j * 3 + 0], jy = J[j * 3 + 1], jz = J[j * 3 + 2];
            // global_T @ [jx, jy, jz, 1]^T
            float tx = G[0]*jx + G[1]*jy + G[2]*jz + G[3];
            float ty = G[4]*jx + G[5]*jy + G[6]*jz + G[7];
            float tz = G[8]*jx + G[9]*jy + G[10]*jz + G[11];

            float* T = &out_transforms[j * 16];
            for (int k = 0; k < 16; k++) T[k] = G[k];
            T[3] -= tx;
            T[7] -= ty;
            T[11] -= tz;
        }
    }

    static void mat4_mul(const float* A, const float* B, float* C) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float val = 0;
                for (int k = 0; k < 4; k++) {
                    val += A[i * 4 + k] * B[k * 4 + j];
                }
                C[i * 4 + j] = val;
            }
        }
    }

    // RefineNet forward
    void run_refine_net(const float* h_img_feat, const float* h_temp_verts,
                        const float* h_pred_cam, int B,
                        float* h_delta_pose, float* h_delta_betas, float* h_delta_cam) {
        int H = VIT_PATCH_H, W = VIT_PATCH_W, D = VIT_EMBED;

        // Upload img_feat to GPU: [B, 192, 1280] -> transpose to [B, 1280, 16, 12]
        GpuBuf d_img_feat, d_img_feat_nchw;
        d_img_feat.alloc(B * 192 * D);
        d_img_feat.copyFrom(h_img_feat, B * 192 * D);
        d_img_feat_nchw.alloc(B * D * H * W);
        // Transpose [B, HW, C] -> [B, C, H, W]
        transpose_nhwc_to_nchw_kernel<<<(B * D * H * W + 255) / 256, 256>>>(
            d_img_feat.data, d_img_feat_nchw.data, B, D, H * W);

        // first_conv: [B, 1280, 16, 12] -> [B, 640, 16, 12]
        GpuBuf d_fc_out;
        d_fc_out.alloc(B * 640 * H * W);
        refine_first_conv.forward(cudnn, d_img_feat_nchw.data, d_fc_out.data, workspace.data, B, H, W);
        // No activation (bnrelu_final=False in first_conv)

        // Branch 0: ConvTranspose(640->320) + ReLU -> [B, 320, 32, 24]
        GpuBuf d_br0;
        d_br0.alloc(B * 320 * H * 2 * W * 2);
        refine_branch0_0.forward(cudnn, d_fc_out.data, d_br0.data, workspace.data, B, H, W);
        relu_inplace(d_br0.data, B * 320 * H * 2 * W * 2);

        // Branch 1: ConvTranspose(640->320) + ReLU -> ConvTranspose(320->160) + ReLU -> [B, 160, 64, 48]
        GpuBuf d_br1a, d_br1b;
        d_br1a.alloc(B * 320 * H * 2 * W * 2);
        refine_branch1_0.forward(cudnn, d_fc_out.data, d_br1a.data, workspace.data, B, H, W);
        relu_inplace(d_br1a.data, B * 320 * H * 2 * W * 2);

        d_br1b.alloc(B * 160 * H * 4 * W * 4);
        refine_branch1_1.forward(cudnn, d_br1a.data, d_br1b.data, workspace.data, B, H * 2, W * 2);
        relu_inplace(d_br1b.data, B * 160 * H * 4 * W * 4);

        // Feature maps (high-res to low-res as in DeConvNet output[::-1]):
        // feat0: d_br1b [B, 160, 64, 48]
        // feat1: d_br0  [B, 320, 32, 24]
        // feat2: d_fc_out [B, 640, 16, 12]
        struct FeatInfo { float* data; int C, H, W; };
        FeatInfo feats[] = {
            {d_br1b.data, 160, H * 4, W * 4},
            {d_br0.data, 320, H * 2, W * 2},
            {d_fc_out.data, 640, H, W},
        };

        // Sample vertex features at projected MANO vertices
        // For each scale: project verts -> grid_sample -> max_pool -> concat
        GpuBuf d_verts, d_verts_2d, d_vert_feat, d_vert_max;
        d_verts.alloc(B * 778 * 3);
        d_verts.copyFrom(h_temp_verts, B * 778 * 3);

        std::vector<float> h_vert_feats_all; // [B, 1120]

        // Collect all vertex features on CPU
        int total_feat_dim = 160 + 320 + 640;  // 1120
        std::vector<float> h_all_feat(B * total_feat_dim, 0);

        int feat_offset = 0;
        for (int fi = 0; fi < 3; fi++) {
            int fC = feats[fi].C, fH = feats[fi].H, fW = feats[fi].W;

            // Compute camera for this scale
            // cam_t = [pred_cam[:, 1], pred_cam[:, 2], 2*focal/(fH * pred_cam[:, 0] + 1e-9)]
            GpuBuf d_grid_xy;
            d_grid_xy.alloc(B * 778 * 2);

            // Project vertices to 2D for this feature map scale
            project_verts_to_feat(d_verts.data, h_pred_cam, B, fH, fW, d_grid_xy.data);

            // grid_sample
            d_vert_feat.alloc(B * fC * 778);
            grid_sample_kernel<<<(B * 778 + 255) / 256, 256>>>(
                feats[fi].data, d_grid_xy.data, d_vert_feat.data, B, fC, fH, fW, 778);

            // max_pool over 778 vertices -> [B, C]
            d_vert_max.alloc(B * fC);
            max_pool_points_kernel<<<(B * fC + 255) / 256, 256>>>(
                d_vert_feat.data, d_vert_max.data, B, fC, 778);

            // Download
            std::vector<float> h_feat(B * fC);
            d_vert_max.copyTo(h_feat.data(), B * fC);

            for (int b = 0; b < B; b++) {
                for (int c = 0; c < fC; c++) {
                    h_all_feat[b * total_feat_dim + feat_offset + c] = h_feat[b * fC + c];
                }
            }
            feat_offset += fC;

            d_grid_xy.free(); d_vert_feat.free(); d_vert_max.free();
        }

        d_verts.free(); d_fc_out.free(); d_br0.free(); d_br1a.free(); d_br1b.free();
        d_img_feat.free(); d_img_feat_nchw.free();

        // Decode deltas
        std::vector<float> h_dec_pose_w(96 * 1120), h_dec_pose_b(96);
        std::vector<float> h_dec_shape_w(10 * 1120), h_dec_shape_b(10);
        std::vector<float> h_dec_cam_w(3 * 1120), h_dec_cam_b(3);
        refine_dec_pose_w.copyTo(h_dec_pose_w.data(), 96 * 1120);
        refine_dec_pose_b.copyTo(h_dec_pose_b.data(), 96);
        refine_dec_shape_w.copyTo(h_dec_shape_w.data(), 10 * 1120);
        refine_dec_shape_b.copyTo(h_dec_shape_b.data(), 10);
        refine_dec_cam_w.copyTo(h_dec_cam_w.data(), 3 * 1120);
        refine_dec_cam_b.copyTo(h_dec_cam_b.data(), 3);

        for (int b = 0; b < B; b++) {
            const float* feat = &h_all_feat[b * 1120];
            for (int o = 0; o < 96; o++) {
                float val = h_dec_pose_b[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_dec_pose_w[o * 1120 + d];
                h_delta_pose[b * 96 + o] = val;
            }
            for (int o = 0; o < 10; o++) {
                float val = h_dec_shape_b[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_dec_shape_w[o * 1120 + d];
                h_delta_betas[b * 10 + o] = val;
            }
            for (int o = 0; o < 3; o++) {
                float val = h_dec_cam_b[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_dec_cam_w[o * 1120 + d];
                h_delta_cam[b * 3 + o] = val;
            }
        }
    }

    void project_verts_to_feat(float* d_verts, const float* h_pred_cam, int B,
                               int featH, int featW, float* d_grid_xy) {
        // Perspective projection of 3D vertices onto feature map
        // cam_t = [cam_x, cam_y, 2*focal/(featH * cam_s + 1e-9)]
        std::vector<float> h_grid(B * 778 * 2);
        std::vector<float> h_verts(B * 778 * 3);
        CUDA_CHECK(cudaMemcpy(h_verts.data(), d_verts, B * 778 * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        for (int b = 0; b < B; b++) {
            float cam_s = h_pred_cam[b * 3 + 0];
            float cam_x = h_pred_cam[b * 3 + 1];
            float cam_y = h_pred_cam[b * 3 + 2];
            float fl = FOCAL_LENGTH;
            float tz = 2.0f * fl / ((float)featH * cam_s + 1e-9f);
            float cam_t[3] = {cam_x, cam_y, tz};
            float fl_scaled = fl / (float)featH;

            for (int v = 0; v < 778; v++) {
                float vx = h_verts[b * 778 * 3 + v * 3 + 0] + cam_t[0];
                float vy = h_verts[b * 778 * 3 + v * 3 + 1] + cam_t[1];
                float vz = h_verts[b * 778 * 3 + v * 3 + 2] + cam_t[2];

                float px = fl_scaled * vx / vz;
                float py = fl_scaled * vy / vz;

                h_grid[(b * 778 + v) * 2 + 0] = px;
                h_grid[(b * 778 + v) * 2 + 1] = py;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_grid_xy, h_grid.data(), B * 778 * 2 * sizeof(float), cudaMemcpyHostToDevice));
    }
};

// ============================================================================
// Main entry point
// ============================================================================

struct FrameHeader {
    uint32_t width, height, frame_idx;
};

// Process a single video file. Models must already be loaded.
// YOLO is lazy-initialized on first call (needs video dimensions).
static void process_video(
    const std::string& video_path, float det_conf, int stride,
    const std::string& output_path, int wilor_batch,
    YoloModel& yolo, bool& yolo_initialized, WilorModel& wilor,
    GpuBuf& d_frame_buf, size_t& max_frame_bytes,
    const std::string& weights_dir)
{
    auto t_program_start = std::chrono::high_resolution_clock::now();

    // Results storage
    struct FrameResult {
        int frame_idx;
        bool left_detected, right_detected;
        float left_kp3d[21 * 3], right_kp3d[21 * 3];
        float left_kp2d[21 * 2], right_kp2d[21 * 2];
    };
    std::vector<FrameResult> all_results;

    // Video decode mode (FFmpeg) or stdin mode
    bool video_mode = !video_path.empty();
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    SwsContext* sws_ctx = nullptr;
    int video_stream_idx = -1;
    AVFrame* av_frame = nullptr;
    AVFrame* bgr_frame = nullptr;
    AVPacket* pkt = nullptr;

    if (video_mode) {
        if (avformat_open_input(&fmt_ctx, video_path.c_str(), nullptr, nullptr) < 0) {
            fprintf(stderr, "[cuda_hand] Cannot open video: %s\n", video_path.c_str());
            return;
        }
        avformat_find_stream_info(fmt_ctx, nullptr);

        for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
            if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_idx = i;
                break;
            }
        }
        if (video_stream_idx < 0) {
            fprintf(stderr, "[cuda_hand] No video stream found\n");
            return;
        }

        auto* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        dec_ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(dec_ctx, codecpar);
        // Use 4 decode threads for speed
        dec_ctx->thread_count = 4;
        avcodec_open2(dec_ctx, codec, nullptr);

        int vW = dec_ctx->width, vH = dec_ctx->height;
        int total_video_frames = (int)fmt_ctx->streams[video_stream_idx]->nb_frames;
        fprintf(stderr, "[cuda_hand] Video: %dx%d, ~%d frames, stride=%d\n",
                vW, vH, total_video_frames, stride);

        // Setup BGR converter
        sws_ctx = sws_getContext(vW, vH, dec_ctx->pix_fmt,
                                 vW, vH, AV_PIX_FMT_BGR24,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);

        av_frame = av_frame_alloc();
        bgr_frame = av_frame_alloc();
        bgr_frame->format = AV_PIX_FMT_BGR24;
        bgr_frame->width = vW;
        bgr_frame->height = vH;
        av_frame_get_buffer(bgr_frame, 32);
        pkt = av_packet_alloc();

        // Pre-init YOLO with batch size for better GPU utilization
        yolo.init(weights_dir, YOLO_BATCH, det_conf, vH, vW);
        yolo_initialized = true;
        fprintf(stderr, "[cuda_hand] YOLO loaded\n");

        // Pre-alloc batch frame buffer (holds YOLO_BATCH frames)
        size_t single_frame_bytes = (size_t)vW * vH * 3;
        max_frame_bytes = single_frame_bytes * YOLO_BATCH;
        d_frame_buf.free();
        CUDA_CHECK(cudaMalloc((void**)&d_frame_buf.data, max_frame_bytes));
    }

    // Read frames
    std::vector<uint8_t> frame_data;
    int n_frames = 0;
    int total_dets = 0;
    int frame_idx = 0;
    double t_decode_total = 0, t_yolo_total = 0, t_wilor_total = 0, t_upload_total = 0;

    // Batched WiLoR: accumulate crops across frames
    std::vector<float> pending_centers, pending_sizes, pending_img_sizes, pending_rights;
    std::vector<int> pending_result_idx;
    float* d_crops_buf = nullptr;
    size_t d_crops_capacity = 0;
    // Pre-alloc for wilor_batch crops
    d_crops_capacity = (size_t)wilor_batch * 3 * VIT_IMG_H * VIT_IMG_W;
    CUDA_CHECK(cudaMalloc(&d_crops_buf, d_crops_capacity * sizeof(float)));

    // Lambda to get next decoded frame (video mode)
    auto decode_next_frame = [&]() -> bool {
        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index != video_stream_idx) {
                av_packet_unref(pkt);
                continue;
            }
            int ret = avcodec_send_packet(dec_ctx, pkt);
            av_packet_unref(pkt);
            if (ret < 0) continue;
            ret = avcodec_receive_frame(dec_ctx, av_frame);
            if (ret == 0) return true;
        }
        // Flush decoder
        avcodec_send_packet(dec_ctx, nullptr);
        return avcodec_receive_frame(dec_ctx, av_frame) == 0;
    };

    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    auto secs = [](auto a, auto b){ return std::chrono::duration<double>(b - a).count(); };

    // Helper: process detections from a YOLO batch, preprocess WiLoR crops
    auto process_yolo_dets = [&](const std::vector<YoloModel::Detection>& dets,
                                  int batch_frame_indices[], int n_in_batch, int W, int H) {
        if (dets.empty()) return;
        float yolo_scale_x = (float)yolo.yolo_w / W;
        float yolo_scale_y = (float)yolo.yolo_h / H;
        size_t single_frame_bytes = (size_t)W * H * 3;

        for (size_t i = 0; i < dets.size(); i++) {
            auto& det = dets[i];
            int bi = det.batch_idx;
            // Find the result_idx for this frame
            // batch_frame_indices[bi] is the frame index in the video
            // We stored result for each frame in order, starting at all_results.size() - n_in_batch
            int result_base = all_results.size() - n_in_batch;
            int result_idx = result_base + bi;

            float x1 = det.x1 / yolo_scale_x;
            float y1 = det.y1 / yolo_scale_y;
            float x2 = det.x2 / yolo_scale_x;
            float y2 = det.y2 / yolo_scale_y;

            float cx = (x1 + x2) / 2.0f;
            float cy = (y1 + y2) / 2.0f;
            float bw = (x2 - x1) * 2.0f;
            float bh = (y2 - y1) * 2.0f;

            float w_exp = bw, h_exp = bh;
            if (bh / bw < 256.0f / 192.0f) h_exp = bw * 256.0f / 192.0f;
            else w_exp = bh * 192.0f / 256.0f;
            float bbox_size = std::max(w_exp, h_exp);

            bool is_right = det.cls == 1;

            pending_centers.push_back(cx);
            pending_centers.push_back(cy);
            pending_sizes.push_back(bbox_size);
            pending_img_sizes.push_back((float)W);
            pending_img_sizes.push_back((float)H);
            pending_rights.push_back(is_right ? 1.0f : 0.0f);
            pending_result_idx.push_back(result_idx);

            int crop_idx_in_batch = pending_sizes.size() - 1;
            size_t needed = (crop_idx_in_batch + 1) * 3 * VIT_IMG_H * VIT_IMG_W;
            if (needed > d_crops_capacity) {
                float* new_buf;
                size_t new_cap = std::max(needed, d_crops_capacity * 2);
                CUDA_CHECK(cudaMalloc(&new_buf, new_cap * sizeof(float)));
                if (d_crops_buf && crop_idx_in_batch > 0) {
                    CUDA_CHECK(cudaMemcpy(new_buf, d_crops_buf,
                        crop_idx_in_batch * 3 * VIT_IMG_H * VIT_IMG_W * sizeof(float),
                        cudaMemcpyDeviceToDevice));
                }
                if (d_crops_buf) CUDA_CHECK(cudaFree(d_crops_buf));
                d_crops_buf = new_buf;
                d_crops_capacity = new_cap;
            }

            // Preprocess crop from the correct frame in the batch buffer
            const uint8_t* frame_ptr = (const uint8_t*)d_frame_buf.data + bi * single_frame_bytes;
            bool flip = !is_right;
            int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
            wilor_preprocess_kernel<<<(total_px + 255) / 256, 256>>>(
                frame_ptr, d_crops_buf,
                H, W, cx, cy, bbox_size, flip,
                VIT_IMG_H, VIT_IMG_W, crop_idx_in_batch,
                IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
        }
    };

    // Helper: flush WiLoR batch
    auto flush_wilor = [&]() {
        int n_pending = pending_sizes.size();
        if (n_pending == 0) return;
        if (n_pending > wilor_batch) n_pending = wilor_batch;
        auto tw0 = now();
        std::vector<WilorModel::HandResult> results(n_pending);
        wilor.forward(d_crops_buf, n_pending,
                      pending_centers.data(), pending_sizes.data(),
                      pending_img_sizes.data(), pending_rights.data(),
                      results.data());
        t_wilor_total += secs(tw0, now());

        for (int i = 0; i < n_pending; i++) {
            int ri = pending_result_idx[i];
            bool is_right = pending_rights[i] > 0.5f;
            if (is_right) {
                all_results[ri].right_detected = true;
                memcpy(all_results[ri].right_kp3d, results[i].kp3d, 63 * sizeof(float));
                memcpy(all_results[ri].right_kp2d, results[i].kp2d, 42 * sizeof(float));
            } else {
                all_results[ri].left_detected = true;
                memcpy(all_results[ri].left_kp3d, results[i].kp3d, 63 * sizeof(float));
                memcpy(all_results[ri].left_kp2d, results[i].kp2d, 42 * sizeof(float));
            }
        }

        pending_centers.clear();
        pending_sizes.clear();
        pending_img_sizes.clear();
        pending_rights.clear();
        pending_result_idx.clear();
    };

    if (video_mode) {
        // Batched YOLO: decode YOLO_BATCH frames, run YOLO, process detections
        int W = dec_ctx->width, H = dec_ctx->height;
        size_t single_frame_bytes = (size_t)W * H * 3;
        int batch_frame_indices[YOLO_BATCH];
        int n_in_batch = 0;

        while (true) {
            auto t0 = now();
            if (!decode_next_frame()) {
                // Flush partial YOLO batch
                if (n_in_batch > 0) {
                    auto ty0 = now();
                    auto dets = yolo.forward((const uint8_t*)d_frame_buf.data, n_in_batch, H, W);
                    t_yolo_total += secs(ty0, now());
                    total_dets += dets.size();
                    process_yolo_dets(dets, batch_frame_indices, n_in_batch, W, H);
                }
                break;
            }

            if (frame_idx % stride != 0) { frame_idx++; continue; }
            int cur_frame_idx = frame_idx;
            frame_idx++;

            sws_scale(sws_ctx, av_frame->data, av_frame->linesize, 0, H,
                       bgr_frame->data, bgr_frame->linesize);
            auto t1 = now();
            t_decode_total += secs(t0, t1);

            // Upload to batch buffer slot
            uint8_t* dst = (uint8_t*)d_frame_buf.data + n_in_batch * single_frame_bytes;
            if (bgr_frame->linesize[0] == W * 3) {
                CUDA_CHECK(cudaMemcpy(dst, bgr_frame->data[0], single_frame_bytes, cudaMemcpyHostToDevice));
            } else {
                for (int row = 0; row < H; row++) {
                    CUDA_CHECK(cudaMemcpy(dst + row * W * 3,
                                          bgr_frame->data[0] + row * bgr_frame->linesize[0],
                                          W * 3, cudaMemcpyHostToDevice));
                }
            }
            t_upload_total += secs(t1, now());

            // Initialize frame result
            FrameResult fr;
            fr.frame_idx = cur_frame_idx;
            fr.left_detected = false;
            fr.right_detected = false;
            memset(fr.left_kp3d, 0, sizeof(fr.left_kp3d));
            memset(fr.right_kp3d, 0, sizeof(fr.right_kp3d));
            memset(fr.left_kp2d, 0, sizeof(fr.left_kp2d));
            memset(fr.right_kp2d, 0, sizeof(fr.right_kp2d));
            all_results.push_back(fr);

            batch_frame_indices[n_in_batch] = cur_frame_idx;
            n_in_batch++;
            n_frames++;

            if (n_in_batch == YOLO_BATCH) {
                // Run batched YOLO
                auto ty0 = now();
                auto dets = yolo.forward((const uint8_t*)d_frame_buf.data, n_in_batch, H, W);
                t_yolo_total += secs(ty0, now());
                total_dets += dets.size();

                process_yolo_dets(dets, batch_frame_indices, n_in_batch, W, H);
                n_in_batch = 0;

                // Flush WiLoR batch when we have enough crops
                if ((int)pending_sizes.size() >= wilor_batch) flush_wilor();
            }
        }
    } else {
        // Stdin mode: single frame at a time
        while (true) {
            FrameHeader hdr;
            if (fread(&hdr, sizeof(hdr), 1, stdin) != 1) break;
            int W = hdr.width, H = hdr.height;
            int cur_frame_idx = hdr.frame_idx;

            size_t frame_bytes = (size_t)W * H * 3;
            frame_data.resize(frame_bytes);
            if (fread(frame_data.data(), 1, frame_bytes, stdin) != frame_bytes) break;

            if (!yolo_initialized) {
                yolo.init(weights_dir, 1, det_conf, H, W);
                yolo_initialized = true;
                fprintf(stderr, "[cuda_hand] YOLO loaded\n");
            }

            if (frame_bytes > max_frame_bytes) {
                max_frame_bytes = frame_bytes;
                d_frame_buf.free();
                CUDA_CHECK(cudaMalloc((void**)&d_frame_buf.data, max_frame_bytes));
            }
            CUDA_CHECK(cudaMemcpy(d_frame_buf.data, frame_data.data(), frame_bytes, cudaMemcpyHostToDevice));

            auto ty0 = now();
            auto dets = yolo.forward((const uint8_t*)d_frame_buf.data, 1, H, W);
            t_yolo_total += secs(ty0, now());
            total_dets += dets.size();

            FrameResult fr;
            fr.frame_idx = cur_frame_idx;
            fr.left_detected = false;
            fr.right_detected = false;
            memset(fr.left_kp3d, 0, sizeof(fr.left_kp3d));
            memset(fr.right_kp3d, 0, sizeof(fr.right_kp3d));
            memset(fr.left_kp2d, 0, sizeof(fr.left_kp2d));
            memset(fr.right_kp2d, 0, sizeof(fr.right_kp2d));
            all_results.push_back(fr);

            int result_idx = all_results.size() - 1;
            int batch_indices[1] = {cur_frame_idx};
            // For single-frame, manually set n_in_batch=1
            if (!dets.empty()) {
                float yolo_scale_x = (float)yolo.yolo_w / W;
                float yolo_scale_y = (float)yolo.yolo_h / H;

                for (size_t i = 0; i < dets.size(); i++) {
                    auto& det = dets[i];
                    float x1 = det.x1 / yolo_scale_x, y1 = det.y1 / yolo_scale_y;
                    float x2 = det.x2 / yolo_scale_x, y2 = det.y2 / yolo_scale_y;
                    float cx = (x1 + x2) / 2.0f, cy = (y1 + y2) / 2.0f;
                    float bw = (x2 - x1) * 2.0f, bh = (y2 - y1) * 2.0f;
                    float w_exp = bw, h_exp = bh;
                    if (bh / bw < 256.0f / 192.0f) h_exp = bw * 256.0f / 192.0f;
                    else w_exp = bh * 192.0f / 256.0f;
                    float bbox_size = std::max(w_exp, h_exp);
                    bool is_right = det.cls == 1;

                    pending_centers.push_back(cx); pending_centers.push_back(cy);
                    pending_sizes.push_back(bbox_size);
                    pending_img_sizes.push_back((float)W); pending_img_sizes.push_back((float)H);
                    pending_rights.push_back(is_right ? 1.0f : 0.0f);
                    pending_result_idx.push_back(result_idx);

                    int crop_idx_in_batch = pending_sizes.size() - 1;
                    size_t needed = (crop_idx_in_batch + 1) * 3 * VIT_IMG_H * VIT_IMG_W;
                    if (needed > d_crops_capacity) {
                        float* new_buf;
                        size_t new_cap = std::max(needed, d_crops_capacity * 2);
                        CUDA_CHECK(cudaMalloc(&new_buf, new_cap * sizeof(float)));
                        if (d_crops_buf && crop_idx_in_batch > 0)
                            CUDA_CHECK(cudaMemcpy(new_buf, d_crops_buf,
                                crop_idx_in_batch * 3 * VIT_IMG_H * VIT_IMG_W * sizeof(float),
                                cudaMemcpyDeviceToDevice));
                        if (d_crops_buf) CUDA_CHECK(cudaFree(d_crops_buf));
                        d_crops_buf = new_buf;
                        d_crops_capacity = new_cap;
                    }

                    bool flip = !is_right;
                    int total_px = 3 * VIT_IMG_H * VIT_IMG_W;
                    wilor_preprocess_kernel<<<(total_px + 255) / 256, 256>>>(
                        (const uint8_t*)d_frame_buf.data, d_crops_buf,
                        H, W, cx, cy, bbox_size, flip,
                        VIT_IMG_H, VIT_IMG_W, crop_idx_in_batch,
                        IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2],
                        IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]);
                }
            }
            n_frames++;
            if ((int)pending_sizes.size() >= wilor_batch) flush_wilor();
        }
    }

    // Flush remaining WiLoR crops
    flush_wilor();
    if (d_crops_buf) { CUDA_CHECK(cudaFree(d_crops_buf)); d_crops_buf = nullptr; }

    if (video_mode) {
        av_frame_free(&av_frame);
        av_frame_free(&bgr_frame);
        av_packet_free(&pkt);
        sws_freeContext(sws_ctx);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
    }
    auto t_program_end = std::chrono::high_resolution_clock::now();
    double t_total_prog = std::chrono::duration<double>(t_program_end - t_program_start).count();
    double t_accounted = t_decode_total + t_upload_total + t_yolo_total + t_wilor_total;
    fprintf(stderr, "[cuda_hand] Processed %d frames, %d total detections\n", n_frames, total_dets);
    fprintf(stderr, "[cuda_hand] Timing: total=%.1fs decode=%.1fs upload=%.1fs yolo=%.1fs wilor=%.1fs other=%.1fs\n",
            t_total_prog, t_decode_total, t_upload_total, t_yolo_total, t_wilor_total, t_total_prog - t_accounted);

    // Write binary output
    // Header: [num_results(i32), total_frames(i32), stride(i32)]
    // Per result: [frame_idx(i32), left_det(u8), right_det(u8), left_kp3d(63f), right_kp3d(63f), left_kp2d(42f), right_kp2d(42f)]
    int total_frames_out = video_mode ? frame_idx : n_frames;
    if (!output_path.empty()) {
        FILE* f = fopen(output_path.c_str(), "wb");
        if (f) {
            int nf = all_results.size();
            fwrite(&nf, sizeof(int), 1, f);
            fwrite(&total_frames_out, sizeof(int), 1, f);
            fwrite(&stride, sizeof(int), 1, f);
            for (auto& fr : all_results) {
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
            fprintf(stderr, "[cuda_hand] Results written to %s\n", output_path.c_str());
        }
    }

}

int main(int argc, char** argv) {
    std::string weights_dir = "data/weights";
    float det_conf = 0.3f;
    int wilor_batch = 48;
    std::string output_path;
    std::string video_path;
    int stride = 1;
    bool listen_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights-dir") == 0 && i + 1 < argc) weights_dir = argv[++i];
        else if (strcmp(argv[i], "--det-conf") == 0 && i + 1 < argc) det_conf = atof(argv[++i]);
        else if (strcmp(argv[i], "--wilor-batch") == 0 && i + 1 < argc) wilor_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) video_path = argv[++i];
        else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "--listen") == 0) listen_mode = true;
        else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: cuda_hand --weights-dir DIR [--video FILE] [--stride N] [--output FILE]\n"
                    "       cuda_hand --weights-dir DIR --listen  (persistent mode, reads jobs from stdin)\n");
            return 0;
        }
    }

    // Load models once
    auto t_load_start = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[cuda_hand] Loading models from %s...\n", weights_dir.c_str());

    YoloModel yolo;
    bool yolo_initialized = false;

    WilorModel wilor;
    wilor.init(weights_dir, wilor_batch);
    fprintf(stderr, "[cuda_hand] WiLoR loaded\n");

    GpuBuf d_frame_buf;
    size_t max_frame_bytes = 1920 * 1080 * 3;
    CUDA_CHECK(cudaMalloc((void**)&d_frame_buf.data, max_frame_bytes));

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double t_load = std::chrono::duration<double>(t_load_end - t_load_start).count();
    fprintf(stderr, "[cuda_hand] Models loaded in %.1fs\n", t_load);

    if (listen_mode) {
        // Persistent mode: read JSON jobs from stdin
        // Format: {"video":"path","output":"path","det_conf":0.3,"stride":1}
        fprintf(stderr, "[cuda_hand] Listening for jobs on stdin...\n");
        printf("READY\n");
        fflush(stdout);

        char line[4096];
        while (fgets(line, sizeof(line), stdin)) {
            // Simple JSON parsing (no library needed for this format)
            std::string s(line);
            if (s.find("SHUTDOWN") != std::string::npos) break;

            // Parse fields — handles optional whitespace after colon
            auto find_key = [&](const char* key) -> size_t {
                std::string pattern = std::string("\"") + key + "\"";
                auto pos = s.find(pattern);
                if (pos == std::string::npos) return std::string::npos;
                pos += pattern.size();
                // skip optional whitespace and colon
                while (pos < s.size() && (s[pos] == ' ' || s[pos] == ':')) pos++;
                return pos;
            };
            auto get_str = [&](const char* key) -> std::string {
                auto pos = find_key(key);
                if (pos == std::string::npos || pos >= s.size() || s[pos] != '"') return "";
                pos++; // skip opening quote
                auto end = s.find('"', pos);
                if (end == std::string::npos) return "";
                return s.substr(pos, end - pos);
            };
            auto get_float = [&](const char* key, float def) -> float {
                auto pos = find_key(key);
                if (pos == std::string::npos) return def;
                return atof(s.c_str() + pos);
            };
            auto get_int = [&](const char* key, int def) -> int {
                auto pos = find_key(key);
                if (pos == std::string::npos) return def;
                return atoi(s.c_str() + pos);
            };

            std::string j_video = get_str("video");
            std::string j_output = get_str("output");
            float j_conf = get_float("det_conf", det_conf);
            int j_stride = get_int("stride", stride);

            if (j_video.empty() || j_output.empty()) {
                printf("{\"status\":\"error\",\"message\":\"need video and output\"}\n");
                fflush(stdout);
                continue;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            process_video(j_video, j_conf, j_stride, j_output, wilor_batch,
                          yolo, yolo_initialized, wilor, d_frame_buf, max_frame_bytes,
                          weights_dir);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            printf("{\"status\":\"done\",\"time\":%.1f}\n", elapsed);
            fflush(stdout);
        }
    } else if (!video_path.empty()) {
        process_video(video_path, det_conf, stride, output_path, wilor_batch,
                      yolo, yolo_initialized, wilor, d_frame_buf, max_frame_bytes,
                      weights_dir);
    } else {
        fprintf(stderr, "Error: specify --video or --listen\n");
        return 1;
    }

    return 0;
}
