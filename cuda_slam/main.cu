// CUDA-only DROID-SLAM implementation for bare-metal profiling
// Uses cuDNN for convolutions, cuBLAS for correlation, cuSOLVER for BA
//
// Build: make
// Run: ./cuda_droid --weights data/weights --frames data/frames --calib data/calib.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <set>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

// FFmpeg headers for NVDEC video decoding
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include "nn.cuh"
#include "se3.cuh"
#include "ba.cuh"
// Note: corr.cuh has issues, inline the needed parts here

// ============ Timing utility ============

struct CudaTimer {
    cudaEvent_t start, stop;
    const char* name;
    float elapsed_ms;

    CudaTimer(const char* n) : name(n), elapsed_ms(0) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void begin() { cudaEventRecord(start); }
    void end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        elapsed_ms += ms;
    }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};

// ============ Weight Store ============

struct WeightStore {
    std::string base_dir;
    std::vector<std::pair<std::string, GpuBuf*>> loaded;

    void set_dir(const char* dir) { base_dir = dir; }

    bool load(const char* name, GpuBuf& buf) {
        std::string path = base_dir + "/" + std::string(name) + ".bin";
        std::vector<float> data;
        std::vector<int> shape;
        if (!load_tensor(path.c_str(), data, shape)) return false;
        buf.alloc(data.size());
        buf.copyFrom(data.data(), data.size());
        loaded.push_back({name, &buf});
        return true;
    }

    // Load weight+bias pair, return total elements in weight
    bool load_conv(const char* prefix, GpuBuf& w, GpuBuf& b) {
        char wname[256], bname[256];
        snprintf(wname, sizeof(wname), "%s_weight", prefix);
        snprintf(bname, sizeof(bname), "%s_bias", prefix);
        return load(wname, w) && load(bname, b);
    }
};

// ============ Residual Block ============

struct ResBlock {
    ConvLayer conv1, conv2;
    ConvLayer downsample_conv;
    bool has_downsample;
    bool has_norm; // InstanceNorm (for fnet)
    int out_channels;

    void init(cudnnHandle_t cudnn, WeightStore& ws, const char* prefix,
              int batch, int ci, int co, int stride, int inH, int inW, bool use_norm) {
        has_norm = use_norm;
        has_downsample = (stride != 1 || ci != co);
        out_channels = co;

        char name[256];
        GpuBuf w1, b1, w2, b2;

        snprintf(name, sizeof(name), "%s_conv1", prefix);
        ws.load_conv(name, w1, b1);
        conv1.init(cudnn, batch, ci, co, 3, 3, stride, 1, inH, inW, w1.data, b1.data);

        int outH = (inH + 2 - 3) / stride + 1;
        int outW = (inW + 2 - 3) / stride + 1;

        snprintf(name, sizeof(name), "%s_conv2", prefix);
        ws.load_conv(name, w2, b2);
        conv2.init(cudnn, batch, co, co, 3, 3, 1, 1, outH, outW, w2.data, b2.data);

        if (has_downsample) {
            GpuBuf dw, db;
            snprintf(name, sizeof(name), "%s_downsample_0", prefix);
            ws.load_conv(name, dw, db);
            downsample_conv.init(cudnn, batch, ci, co, 1, 1, stride, 0, inH, inW, dw.data, db.data);
        }
    }

    // Forward: needs temp buffers
    void forward(cudnnHandle_t cudnn, float* input, float* output, float* temp,
                 float* workspace, int batch, int ci, int inH, int inW) {
        int outH = (inH + 2 - 3) / conv1.stride + 1;
        int outW = (inW + 2 - 3) / conv1.stride + 1;
        int outN = batch * out_channels * outH * outW;

        // temp = conv1(input)
        conv1.forward(cudnn, input, temp, workspace, batch, inH, inW);
        if (has_norm) instance_norm(temp, batch, out_channels, outH, outW);
        relu_inplace(temp, outN);

        // output = relu(norm(conv2(temp)))
        conv2.forward(cudnn, temp, output, workspace, batch, outH, outW);
        if (has_norm) instance_norm(output, batch, out_channels, outH, outW);
        relu_inplace(output, outN);

        // Residual connection
        if (has_downsample) {
            // temp = downsample(input)
            downsample_conv.forward(cudnn, input, temp, workspace, batch, inH, inW);
            if (has_norm) instance_norm(temp, batch, out_channels, outH, outW);
            // output += temp
            add_kernel<<<(outN+255)/256, 256>>>(output, temp, outN);
        } else {
            // output += input
            add_kernel<<<(outN+255)/256, 256>>>(output, input, outN);
        }

        relu_inplace(output, outN);
    }
};

// ============ Basic Encoder ============

struct BasicEncoder {
    ConvLayer conv1, conv2;
    bool has_norm;  // fnet has instance norm, cnet doesn't
    ResBlock layer1_0, layer1_1;
    ResBlock layer2_0, layer2_1;
    ResBlock layer3_0, layer3_1;
    int output_dim;

    void init(cudnnHandle_t cudnn, WeightStore& ws, const char* prefix,
              int batch, int inH, int inW, int out_dim, bool use_norm) {
        output_dim = out_dim;
        has_norm = use_norm;

        char name[256];
        GpuBuf w, b;

        // conv1: 3 -> 32, 7x7, stride=2, pad=3
        snprintf(name, sizeof(name), "%s_conv1", prefix);
        ws.load_conv(name, w, b);
        conv1.init(cudnn, batch, 3, 32, 7, 7, 2, 3, inH, inW, w.data, b.data);

        int h2 = inH / 2, w2 = inW / 2;

        // layer1: 32->32, stride=1
        snprintf(name, sizeof(name), "%s_layer1_0", prefix);
        layer1_0.init(cudnn, ws, name, batch, 32, 32, 1, h2, w2, use_norm);
        snprintf(name, sizeof(name), "%s_layer1_1", prefix);
        layer1_1.init(cudnn, ws, name, batch, 32, 32, 1, h2, w2, use_norm);

        // layer2: 32->64, stride=2
        snprintf(name, sizeof(name), "%s_layer2_0", prefix);
        layer2_0.init(cudnn, ws, name, batch, 32, 64, 2, h2, w2, use_norm);
        int h4 = h2 / 2, w4 = w2 / 2;
        snprintf(name, sizeof(name), "%s_layer2_1", prefix);
        layer2_1.init(cudnn, ws, name, batch, 64, 64, 1, h4, w4, use_norm);

        // layer3: 64->128, stride=2
        snprintf(name, sizeof(name), "%s_layer3_0", prefix);
        layer3_0.init(cudnn, ws, name, batch, 64, 128, 2, h4, w4, use_norm);
        int h8 = h4 / 2, w8 = w4 / 2;
        snprintf(name, sizeof(name), "%s_layer3_1", prefix);
        layer3_1.init(cudnn, ws, name, batch, 128, 128, 1, h8, w8, use_norm);

        // conv2: 128 -> output_dim, 1x1
        snprintf(name, sizeof(name), "%s_conv2", prefix);
        ws.load_conv(name, w, b);
        conv2.init(cudnn, batch, 128, out_dim, 1, 1, 1, 0, h8, w8, w.data, b.data);
    }

    // Forward: input [batch, 3, H, W] -> output [batch, output_dim, H/8, W/8]
    // Needs 3 separate buffers: buf_a, buf_b, buf_c (no aliasing allowed for cuDNN)
    void forward(cudnnHandle_t cudnn, float* input, float* output,
                 float* buf_a, float* buf_b, float* buf_c, float* workspace,
                 int batch, int inH, int inW) {
        int h2 = inH / 2, w2 = inW / 2;
        int h4 = h2 / 2, w4 = w2 / 2;
        int h8 = h4 / 2, w8 = w4 / 2;

        // conv1 + norm + relu: input -> buf_a
        conv1.forward(cudnn, input, buf_a, workspace, batch, inH, inW);
        if (has_norm) instance_norm(buf_a, batch, 32, h2, w2);
        relu_inplace(buf_a, batch * 32 * h2 * w2);

        // layer1: buf_a -> buf_b (temp=buf_c), then buf_b -> buf_a (temp=buf_c)
        layer1_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 32, h2, w2);
        layer1_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 32, h2, w2);

        // layer2: buf_a -> buf_b (temp=buf_c), then buf_b -> buf_a (temp=buf_c)
        layer2_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 32, h2, w2);
        layer2_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 64, h4, w4);

        // layer3: buf_a -> buf_b (temp=buf_c), then buf_b -> buf_a (temp=buf_c)
        layer3_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 64, h4, w4);
        layer3_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 128, h8, w8);

        // conv2 (1x1): buf_a -> output
        conv2.forward(cudnn, buf_a, output, workspace, batch, h8, w8);
    }
};

// ============ FP16 Residual Block ============

struct HalfResBlock {
    HalfConvLayer conv1, conv2;
    HalfConvLayer downsample_conv;
    bool has_downsample;
    bool has_norm;
    int out_channels;

    void init(cudnnHandle_t cudnn, WeightStore& ws, const char* prefix,
              int batch, int ci, int co, int stride, int inH, int inW, bool use_norm) {
        has_norm = use_norm;
        has_downsample = (stride != 1 || ci != co);
        out_channels = co;

        char name[256];
        GpuBuf w1, b1, w2, b2;

        snprintf(name, sizeof(name), "%s_conv1", prefix);
        ws.load_conv(name, w1, b1);
        conv1.init(cudnn, batch, ci, co, 3, 3, stride, 1, inH, inW, w1.data, b1.data);

        int outH = (inH + 2 - 3) / stride + 1;
        int outW = (inW + 2 - 3) / stride + 1;

        snprintf(name, sizeof(name), "%s_conv2", prefix);
        ws.load_conv(name, w2, b2);
        conv2.init(cudnn, batch, co, co, 3, 3, 1, 1, outH, outW, w2.data, b2.data);

        if (has_downsample) {
            GpuBuf dw, db;
            snprintf(name, sizeof(name), "%s_downsample_0", prefix);
            ws.load_conv(name, dw, db);
            downsample_conv.init(cudnn, batch, ci, co, 1, 1, stride, 0, inH, inW, dw.data, db.data);
        }
    }

    void forward(cudnnHandle_t cudnn, __half* input, __half* output, __half* temp,
                 float* workspace, int batch, int ci, int inH, int inW) {
        int outH = (inH + 2 - 3) / conv1.stride + 1;
        int outW = (inW + 2 - 3) / conv1.stride + 1;
        int outN = batch * out_channels * outH * outW;

        conv1.forward(cudnn, input, temp, workspace, batch, inH, inW);
        if (has_norm) instance_norm_half(temp, batch, out_channels, outH, outW);
        relu_half_inplace(temp, outN);

        conv2.forward(cudnn, temp, output, workspace, batch, outH, outW);
        if (has_norm) instance_norm_half(output, batch, out_channels, outH, outW);
        relu_half_inplace(output, outN);

        if (has_downsample) {
            downsample_conv.forward(cudnn, input, temp, workspace, batch, inH, inW);
            if (has_norm) instance_norm_half(temp, batch, out_channels, outH, outW);
            add_half_kernel<<<(outN+255)/256, 256>>>(output, temp, outN);
        } else {
            add_half_kernel<<<(outN+255)/256, 256>>>(output, input, outN);
        }

        relu_half_inplace(output, outN);
    }
};

// ============ FP16 Basic Encoder ============

struct HalfBasicEncoder {
    HalfConvLayer conv1, conv2;
    bool has_norm;
    HalfResBlock layer1_0, layer1_1;
    HalfResBlock layer2_0, layer2_1;
    HalfResBlock layer3_0, layer3_1;
    int output_dim;

    GpuHalfBuf fp16_input;  // FP32 image → FP16

    void init(cudnnHandle_t cudnn, WeightStore& ws, const char* prefix,
              int batch, int inH, int inW, int out_dim, bool use_norm) {
        output_dim = out_dim;
        has_norm = use_norm;

        char name[256];
        GpuBuf w, b;

        snprintf(name, sizeof(name), "%s_conv1", prefix);
        ws.load_conv(name, w, b);
        conv1.init(cudnn, batch, 3, 32, 7, 7, 2, 3, inH, inW, w.data, b.data);

        int h2 = inH / 2, w2 = inW / 2;

        snprintf(name, sizeof(name), "%s_layer1_0", prefix);
        layer1_0.init(cudnn, ws, name, batch, 32, 32, 1, h2, w2, use_norm);
        snprintf(name, sizeof(name), "%s_layer1_1", prefix);
        layer1_1.init(cudnn, ws, name, batch, 32, 32, 1, h2, w2, use_norm);

        snprintf(name, sizeof(name), "%s_layer2_0", prefix);
        layer2_0.init(cudnn, ws, name, batch, 32, 64, 2, h2, w2, use_norm);
        int h4 = h2 / 2, w4 = w2 / 2;
        snprintf(name, sizeof(name), "%s_layer2_1", prefix);
        layer2_1.init(cudnn, ws, name, batch, 64, 64, 1, h4, w4, use_norm);

        snprintf(name, sizeof(name), "%s_layer3_0", prefix);
        layer3_0.init(cudnn, ws, name, batch, 64, 128, 2, h4, w4, use_norm);
        int h8 = h4 / 2, w8 = w4 / 2;
        snprintf(name, sizeof(name), "%s_layer3_1", prefix);
        layer3_1.init(cudnn, ws, name, batch, 128, 128, 1, h8, w8, use_norm);

        snprintf(name, sizeof(name), "%s_conv2", prefix);
        ws.load_conv(name, w, b);
        conv2.init(cudnn, batch, 128, out_dim, 1, 1, 1, 0, h8, w8, w.data, b.data);

        fp16_input.alloc(batch * 3 * inH * inW);
    }

    // Forward: FP32 input → FP16 output
    void forward(cudnnHandle_t cudnn, float* input, __half* output,
                 __half* buf_a, __half* buf_b, __half* buf_c, float* workspace,
                 int batch, int inH, int inW) {
        int h2 = inH / 2, w2 = inW / 2;
        int h4 = h2 / 2, w4 = w2 / 2;
        int h8 = h4 / 2, w8 = w4 / 2;

        // Convert FP32 input to FP16
        float_to_half(input, fp16_input.data, batch * 3 * inH * inW);

        // conv1 + norm + relu
        conv1.forward(cudnn, fp16_input.data, buf_a, workspace, batch, inH, inW);
        if (has_norm) instance_norm_half(buf_a, batch, 32, h2, w2);
        relu_half_inplace(buf_a, batch * 32 * h2 * w2);

        layer1_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 32, h2, w2);
        layer1_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 32, h2, w2);

        layer2_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 32, h2, w2);
        layer2_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 64, h4, w4);

        layer3_0.forward(cudnn, buf_a, buf_b, buf_c, workspace, batch, 64, h4, w4);
        layer3_1.forward(cudnn, buf_b, buf_a, buf_c, workspace, batch, 128, h8, w8);

        conv2.forward(cudnn, buf_a, output, workspace, batch, h8, w8);
    }
};

// ============ ConvGRU ============

struct ConvGRU {
    ConvLayer convz, convr, convq;         // 3x3, input: [h+inp]=448, output: 128
    ConvLayer convz_glo, convr_glo, convq_glo;  // 1x1, 128->128
    ConvLayer w_conv;                       // 1x1, 128->128 (attention gate)

    GpuBuf glo_buf, gate_buf, cat_buf, rnet_buf, gru_inp_buf;
    GpuBuf z_buf, r_buf, q_buf, z_glo, r_glo, q_glo;
    int max_batch_hw;  // pre-allocated capacity

    void init(cudnnHandle_t cudnn, WeightStore& ws, const char* prefix,
              int batch, int h, int w) {
        char name[256];
        GpuBuf wt, bi;

        // Main convolutions (3x3, 448->128)
        snprintf(name, sizeof(name), "%s_convz", prefix);
        ws.load_conv(name, wt, bi);
        convz.init(cudnn, batch, 448, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);

        snprintf(name, sizeof(name), "%s_convr", prefix);
        ws.load_conv(name, wt, bi);
        convr.init(cudnn, batch, 448, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);

        snprintf(name, sizeof(name), "%s_convq", prefix);
        ws.load_conv(name, wt, bi);
        convq.init(cudnn, batch, 448, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);

        // Global convolutions (1x1, 128->128)
        snprintf(name, sizeof(name), "%s_convz_glo", prefix);
        ws.load_conv(name, wt, bi);
        convz_glo.init(cudnn, batch, 128, 128, 1, 1, 1, 0, 1, 1, wt.data, bi.data);

        snprintf(name, sizeof(name), "%s_convr_glo", prefix);
        ws.load_conv(name, wt, bi);
        convr_glo.init(cudnn, batch, 128, 128, 1, 1, 1, 0, 1, 1, wt.data, bi.data);

        snprintf(name, sizeof(name), "%s_convq_glo", prefix);
        ws.load_conv(name, wt, bi);
        convq_glo.init(cudnn, batch, 128, 128, 1, 1, 1, 0, 1, 1, wt.data, bi.data);

        // Attention gate (1x1, 128->128)
        snprintf(name, sizeof(name), "%s_w", prefix);
        ws.load_conv(name, wt, bi);
        w_conv.init(cudnn, batch, 128, 128, 1, 1, 1, 0, h, w, wt.data, bi.data);

        // Allocate working buffers for max batch size
        int NC = batch * 128;
        int HW = h * w;
        max_batch_hw = batch * HW;  // remember capacity
        glo_buf.alloc(NC);        // [batch, 128, 1, 1]
        gate_buf.alloc(NC * HW);  // [batch, 128, h, w]
        cat_buf.alloc(batch * 448 * HW);  // [batch, 448, h, w]
        rnet_buf.alloc(NC * HW);  // r * net
        gru_inp_buf.alloc(batch * 448 * HW);

        // Pre-allocate z/r/q buffers (avoids cudaMalloc/cudaFree per forward call)
        z_buf.alloc(NC * HW);
        r_buf.alloc(NC * HW);
        q_buf.alloc(NC * HW);
        z_glo.alloc(NC);
        r_glo.alloc(NC);
        q_glo.alloc(NC);
    }

    // Forward: updates net in-place
    // net: [batch, 128, h, w] - hidden state (modified in place)
    // inp: [batch, 128, h, w] - context features
    // corr: [batch, 128, h, w] - correlation features (after encoding)
    // flow: [batch, 64, h, w] - flow features (after encoding)
    void forward(cudnnHandle_t cudnn, float* net, float* inp, float* corr, float* flow,
                 float* workspace, int batch, int h, int w) {
        int HW = h * w;
        int NC = batch * 128;
        int total_inp = batch * (128 + 64) * HW;  // inp+corr+flow
        int total_cat = batch * 448 * HW;          // net+inp+corr+flow

        // Build cat_buf = [net(128), inp(128), corr(128), flow(64)] = 448 channels, NCHW
        concat4_kernel<<<(batch*448*HW+255)/256, 256>>>(
            cat_buf.data,
            net, inp, corr, flow,
            batch, 128, 128, 128, 64, HW);

        // Compute global context: sigmoid(w(net)) * net, then spatial mean
        w_conv.forward(cudnn, net, gate_buf.data, workspace, batch, h, w);
        sigmoid_inplace(gate_buf.data, NC * HW);
        // gate = sigmoid(w(net)) * net
        mul_kernel<<<(NC*HW+255)/256, 256>>>(gate_buf.data, net, NC * HW);
        // Global average pool: [batch, 128, h, w] -> [batch, 128, 1, 1]
        int threads = std::min(256, HW);
        int t = 1; while (t < threads) t <<= 1; threads = t;
        global_avg_pool_kernel<<<NC, threads, threads*sizeof(float)>>>(
            gate_buf.data, glo_buf.data, 128, HW);

        // z = sigmoid(convz(cat) + convz_glo(glo))
        convz.forward(cudnn, cat_buf.data, z_buf.data, workspace, batch, h, w);
        convz_glo.forward(cudnn, glo_buf.data, z_glo.data, workspace, batch, 1, 1);
        broadcast_add_kernel<<<NC, 256>>>(z_buf.data, z_glo.data, NC, HW);
        sigmoid_inplace(z_buf.data, NC * HW);

        // r = sigmoid(convr(cat) + convr_glo(glo))
        convr.forward(cudnn, cat_buf.data, r_buf.data, workspace, batch, h, w);
        convr_glo.forward(cudnn, glo_buf.data, r_glo.data, workspace, batch, 1, 1);
        broadcast_add_kernel<<<NC, 256>>>(r_buf.data, r_glo.data, NC, HW);
        sigmoid_inplace(r_buf.data, NC * HW);

        // gru_input = cat(r*net, inp, corr, flow) = 448 channels
        // r*net -> rnet_buf
        CUDA_CHECK(cudaMemcpy(rnet_buf.data, net, NC * HW * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        mul_kernel<<<(NC*HW+255)/256, 256>>>(rnet_buf.data, r_buf.data, NC * HW);
        // Build gru_input: [r*net(128), inp(128), corr(128), flow(64)] = 448
        concat4_kernel<<<(batch*448*HW+255)/256, 256>>>(
            gru_inp_buf.data,
            rnet_buf.data, inp, corr, flow,
            batch, 128, 128, 128, 64, HW);

        // q = tanh(convq(gru_input) + convq_glo(glo))
        convq.forward(cudnn, gru_inp_buf.data, q_buf.data, workspace, batch, h, w);
        convq_glo.forward(cudnn, glo_buf.data, q_glo.data, workspace, batch, 1, 1);
        broadcast_add_kernel<<<NC, 256>>>(q_buf.data, q_glo.data, NC, HW);
        tanh_inplace(q_buf.data, NC * HW);

        // net = (1-z)*net + z*q
        gru_update_kernel<<<(NC*HW+255)/256, 256>>>(net, z_buf.data, q_buf.data, NC * HW);
    }
};

// ============ Update Module ============

struct UpdateModule {
    ConvLayer corr_enc_0, corr_enc_2;  // 196->128, 128->128
    ConvLayer flow_enc_0, flow_enc_2;  // 4->128, 128->64
    ConvGRU gru;
    ConvLayer weight_0, weight_2;  // 128->128, 128->2
    ConvLayer delta_0, delta_2;    // 128->128, 128->2

    // GraphAgg
    ConvLayer agg_conv1, agg_conv2;
    ConvLayer agg_eta_0;  // 128->1
    ConvLayer agg_upmask; // 128->576

    GpuBuf flow_enc_buf;  // pre-allocated: [batch, 64, h, w]

    void init(cudnnHandle_t cudnn, WeightStore& ws, int batch, int h, int w) {
        GpuBuf wt, bi;

        // Correlation encoder
        ws.load_conv("update_corr_encoder_0", wt, bi);
        corr_enc_0.init(cudnn, batch, 196, 128, 1, 1, 1, 0, h, w, wt.data, bi.data);
        ws.load_conv("update_corr_encoder_2", wt, bi);
        corr_enc_2.init(cudnn, batch, 128, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);

        // Flow encoder
        ws.load_conv("update_flow_encoder_0", wt, bi);
        flow_enc_0.init(cudnn, batch, 4, 128, 7, 7, 1, 3, h, w, wt.data, bi.data);
        ws.load_conv("update_flow_encoder_2", wt, bi);
        flow_enc_2.init(cudnn, batch, 128, 64, 3, 3, 1, 1, h, w, wt.data, bi.data);

        // GRU
        gru.init(cudnn, ws, "update_gru", batch, h, w);

        // Weight head (3 output channels: 2 flow weights + 1 depth weight)
        ws.load_conv("update_weight_0", wt, bi);
        weight_0.init(cudnn, batch, 128, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);
        ws.load_conv("update_weight_2", wt, bi);
        weight_2.init(cudnn, batch, 128, 3, 3, 3, 1, 1, h, w, wt.data, bi.data);

        // Delta head (3 output channels: 2 flow + 1 depth)
        ws.load_conv("update_delta_0", wt, bi);
        delta_0.init(cudnn, batch, 128, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);
        ws.load_conv("update_delta_2", wt, bi);
        delta_2.init(cudnn, batch, 128, 3, 3, 3, 1, 1, h, w, wt.data, bi.data);

        // GraphAgg
        ws.load_conv("update_agg_conv1", wt, bi);
        agg_conv1.init(cudnn, batch, 128, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);
        ws.load_conv("update_agg_conv2", wt, bi);
        agg_conv2.init(cudnn, batch, 128, 128, 3, 3, 1, 1, h, w, wt.data, bi.data);
        ws.load_conv("update_agg_eta_0", wt, bi);
        agg_eta_0.init(cudnn, batch, 128, 1, 3, 3, 1, 1, h, w, wt.data, bi.data);
        ws.load_conv("update_agg_upmask_0", wt, bi);
        agg_upmask.init(cudnn, batch, 128, 576, 1, 1, 1, 0, h, w, wt.data, bi.data);

        // Pre-allocate flow encoder output buffer
        flow_enc_buf.alloc(batch * 64 * h * w);
    }

    // Forward pass of the update module
    // corr_features: [batch, 196, h, w] - raw correlation features
    // motion: [batch, 4, h, w] - motion features (flow residuals)
    // net: [batch, 128, h, w] - GRU hidden state (modified in place)
    // inp: [batch, 128, h, w] - context features
    // delta_out: [batch, 3, h, w] - flow correction output (2 flow + 1 depth)
    // weight_out: [batch, 3, h, w] - weight output (2 flow + 1 depth)
    // eta_out: [batch, 1, h, w] - damping output
    void forward(cudnnHandle_t cudnn, float* corr_features, float* motion,
                 float* net, float* inp,
                 float* delta_out, float* weight_out, float* eta_out,
                 float* temp1, float* temp2, float* workspace,
                 int batch, int h, int w) {
        int HW = h * w;

        // Encode correlation features: 196 -> 128
        corr_enc_0.forward(cudnn, corr_features, temp1, workspace, batch, h, w);
        relu_inplace(temp1, batch * 128 * HW);
        corr_enc_2.forward(cudnn, temp1, temp2, workspace, batch, h, w);
        relu_inplace(temp2, batch * 128 * HW);
        // temp2 now has encoded correlation: [batch, 128, h, w]

        // Encode flow features: 4 -> 64
        flow_enc_0.forward(cudnn, motion, temp1, workspace, batch, h, w);
        relu_inplace(temp1, batch * 128 * HW);
        flow_enc_2.forward(cudnn, temp1, flow_enc_buf.data, workspace, batch, h, w);
        relu_inplace(flow_enc_buf.data, batch * 64 * HW);
        // flow_enc_buf now has encoded flow: [batch, 64, h, w]

        // GRU update
        gru.forward(cudnn, net, inp, temp2, flow_enc_buf.data, workspace, batch, h, w);

        // Delta head
        delta_0.forward(cudnn, net, temp1, workspace, batch, h, w);
        relu_inplace(temp1, batch * 128 * HW);
        delta_2.forward(cudnn, temp1, delta_out, workspace, batch, h, w);

        // (debug output removed)

        // Weight head
        weight_0.forward(cudnn, net, temp1, workspace, batch, h, w);
        relu_inplace(temp1, batch * 128 * HW);
        weight_2.forward(cudnn, temp1, weight_out, workspace, batch, h, w);
        sigmoid_inplace(weight_out, batch * 3 * HW);

        // GraphAgg: compute eta (damping)
        agg_conv1.forward(cudnn, net, temp1, workspace, batch, h, w);
        relu_inplace(temp1, batch * 128 * HW);
        // TODO: scatter_mean aggregation by keyframe index
        // For now, just pass through (works for single-keyframe edges)
        agg_conv2.forward(cudnn, temp1, temp2, workspace, batch, h, w);
        relu_inplace(temp2, batch * 128 * HW);
        agg_eta_0.forward(cudnn, temp2, eta_out, workspace, batch, h, w);
        softplus_inplace(eta_out, batch * 1 * HW, 0.01f);
    }
};

// ============ DROID-SLAM State ============

struct DroidState {
    static const int MAX_KEYFRAMES = 512;

    int H, W;        // Full resolution
    int h, w;         // 1/8 resolution
    int num_keyframes; // Number of keyframes stored

    // Keyframe tracking: maps keyframe index -> original frame timestamp
    std::vector<int> kf_timestamps;

    // Per-keyframe state (on GPU), indexed by keyframe index
    GpuBuf poses;      // [MAX_KF, 7]
    GpuBuf disps;      // [MAX_KF, h, w]
    GpuHalfBuf fmaps;  // [MAX_KF, 128, h, w] FP16 for tensor-core correlation
    GpuBuf nets;       // [MAX_KF, 128, h, w]  (initial hidden state)
    GpuBuf inps;       // [MAX_KF, 128, h, w]
    GpuBuf intrinsics; // [4]

    // Edge state (persistent across frames)
    std::vector<int> ii_host, jj_host;
    GpuIntBuf ii_gpu, jj_gpu;

    // Per-edge target, weight, and hidden state (persistent)
    GpuBuf target;     // [MAX_EDGES, 2, h, w]
    GpuBuf weight;     // [MAX_EDGES, 2, h, w]
    GpuBuf edge_nets;  // [MAX_EDGES, 128, h, w]
    static const int MAX_EDGES = 2048;

    void init(int fullH, int fullW, float fx, float fy, float cx, float cy) {
        H = fullH; W = fullW;
        h = fullH / 8; w = fullW / 8;
        num_keyframes = 0;

        int hw = h * w;
        poses.alloc(MAX_KEYFRAMES * 7);
        disps.alloc(MAX_KEYFRAMES * hw);
        fmaps.alloc((size_t)MAX_KEYFRAMES * 128 * hw);
        nets.alloc(MAX_KEYFRAMES * 128 * hw);
        inps.alloc(MAX_KEYFRAMES * 128 * hw);
        intrinsics.alloc(4);

        // Set intrinsics at 1/8 resolution
        float intr[4] = {fx/8.0f, fy/8.0f, cx/8.0f, cy/8.0f};
        intrinsics.copyFrom(intr, 4);

        // Initialize first pose to identity
        poses.zero();
        disps.zero();

        // Set identity quaternion for all poses
        std::vector<float> id_poses(MAX_KEYFRAMES * 7, 0.0f);
        for (int i = 0; i < MAX_KEYFRAMES; i++)
            id_poses[i*7 + 6] = 1.0f;  // qw = 1
        poses.copyFrom(id_poses.data(), MAX_KEYFRAMES * 7);

        // Initialize disparities to 1.0 (matching PyTorch DROID-SLAM)
        std::vector<float> init_disps(MAX_KEYFRAMES * hw, 1.0f);
        disps.copyFrom(init_disps.data(), MAX_KEYFRAMES * hw);

        // Edge buffers
        target.alloc(MAX_EDGES * 2 * hw);
        weight.alloc(MAX_EDGES * 2 * hw);
        edge_nets.alloc(MAX_EDGES * 128 * hw);
        ii_gpu.alloc(MAX_EDGES);
        jj_gpu.alloc(MAX_EDGES);
    }

    // Add a keyframe: store features at keyframe index, record timestamp
    void add_keyframe(int frame_timestamp, __half* fmap, float* net, float* inp) {
        int hw = h * w;
        int idx = num_keyframes;
        CUDA_CHECK(cudaMemcpy(fmaps.data + idx * 128 * hw, fmap,
                              128 * hw * sizeof(__half), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(nets.data + idx * 128 * hw, net,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(inps.data + idx * 128 * hw, inp,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));

        // Copy pose from previous keyframe
        if (idx > 0) {
            CUDA_CHECK(cudaMemcpy(poses.data + idx * 7, poses.data + (idx-1) * 7,
                                  7 * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        kf_timestamps.push_back(frame_timestamp);
        num_keyframes++;
    }

    // Add edges connecting new keyframe to nearby ones
    void add_edges_for_keyframe(int kf_idx, int radius) {
        int start = std::max(0, kf_idx - radius);
        for (int j = start; j < kf_idx; j++) {
            // Bidirectional edges
            ii_host.push_back(kf_idx);
            jj_host.push_back(j);
            ii_host.push_back(j);
            jj_host.push_back(kf_idx);
        }
    }

    int num_edges() const { return (int)ii_host.size(); }

    // Remove edges where BOTH endpoints are older than min_kf.
    // Keep any edge that has at least one endpoint >= min_kf.
    void remove_old_edges(int min_kf, int hw) {
        int n = (int)ii_host.size();
        if (n == 0) return;
        std::vector<int> new_ii, new_jj;
        new_ii.reserve(n); new_jj.reserve(n);
        for (int e = 0; e < n; e++) {
            if (ii_host[e] >= min_kf || jj_host[e] >= min_kf) {
                // Compact edge hidden state
                int new_idx = (int)new_ii.size();
                if (new_idx != e) {
                    CUDA_CHECK(cudaMemcpy(edge_nets.data + new_idx * 128 * hw,
                                          edge_nets.data + e * 128 * hw,
                                          128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(target.data + new_idx * 2 * hw,
                                          target.data + e * 2 * hw,
                                          2 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(weight.data + new_idx * 2 * hw,
                                          weight.data + e * 2 * hw,
                                          2 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                }
                new_ii.push_back(ii_host[e]);
                new_jj.push_back(jj_host[e]);
            }
        }
        ii_host = std::move(new_ii);
        jj_host = std::move(new_jj);
    }

    void sync_edges_to_gpu() {
        int n = ii_host.size();
        if (n == 0) return;
        ii_gpu.alloc(n);
        jj_gpu.alloc(n);
        ii_gpu.copyFrom(ii_host.data(), n);
        jj_gpu.copyFrom(jj_host.data(), n);
    }
};

// ============ NVDEC NV12/P010 → float32 BGR with bilinear resize ============

// NV12 (8-bit) → float32 BGR HWC with bilinear resize
__global__ void nv12_to_bgr_resize_kernel(
    const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane,
    float* __restrict__ bgr_out, int src_w, int src_h,
    int y_stride, int uv_stride, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    // Bilinear sampling coordinates in source
    float sx = ((float)dx + 0.5f) * src_w / dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * src_h / dst_h - 0.5f;
    int sx0 = (int)floorf(sx), sy0 = (int)floorf(sy);
    float wx = sx - sx0, wy = sy - sy0;
    int sx1 = min(sx0 + 1, src_w - 1), sy1 = min(sy0 + 1, src_h - 1);
    sx0 = max(sx0, 0); sy0 = max(sy0, 0);

    // Bilinear Y
    float Y = (1-wy)*((1-wx)*y_plane[sy0*y_stride+sx0] + wx*y_plane[sy0*y_stride+sx1])
            + wy*((1-wx)*y_plane[sy1*y_stride+sx0] + wx*y_plane[sy1*y_stride+sx1]);

    // Nearest UV (chroma is half-res)
    int uvx = ((int)(sx + 0.5f) / 2) * 2;
    int uvy = (int)(sy + 0.5f) / 2;
    uvx = max(0, min(uvx, src_w - 2));
    uvy = max(0, min(uvy, src_h / 2 - 1));
    float U = (float)uv_plane[uvy * uv_stride + uvx] - 128.0f;
    float V = (float)uv_plane[uvy * uv_stride + uvx + 1] - 128.0f;

    // BT.709 YUV to RGB
    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    int idx = (dy * dst_w + dx) * 3;
    bgr_out[idx + 0] = fminf(fmaxf(B, 0.0f), 255.0f);
    bgr_out[idx + 1] = fminf(fmaxf(G, 0.0f), 255.0f);
    bgr_out[idx + 2] = fminf(fmaxf(R, 0.0f), 255.0f);
}

// P010 (10-bit) → float32 BGR HWC with bilinear resize
__global__ void p010_to_bgr_resize_kernel(
    const uint16_t* __restrict__ y_plane, const uint16_t* __restrict__ uv_plane,
    float* __restrict__ bgr_out, int src_w, int src_h,
    int y_stride_bytes, int uv_stride_bytes, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    int y_stride = y_stride_bytes / 2;
    int uv_stride = uv_stride_bytes / 2;

    float sx = ((float)dx + 0.5f) * src_w / dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * src_h / dst_h - 0.5f;
    int sx0 = (int)floorf(sx), sy0 = (int)floorf(sy);
    float wx = sx - sx0, wy = sy - sy0;
    int sx1 = min(sx0 + 1, src_w - 1), sy1 = min(sy0 + 1, src_h - 1);
    sx0 = max(sx0, 0); sy0 = max(sy0, 0);

    // Bilinear Y (10-bit in MSB of uint16, scale to 8-bit range)
    auto y_val = [&](int r, int c) { return (float)(y_plane[r*y_stride+c] >> 6) / 4.0f; };
    float Y = (1-wy)*((1-wx)*y_val(sy0,sx0) + wx*y_val(sy0,sx1))
            + wy*((1-wx)*y_val(sy1,sx0) + wx*y_val(sy1,sx1));

    int uvx = ((int)(sx + 0.5f) / 2) * 2;
    int uvy = (int)(sy + 0.5f) / 2;
    uvx = max(0, min(uvx, src_w - 2));
    uvy = max(0, min(uvy, src_h / 2 - 1));
    float U = (float)(uv_plane[uvy * uv_stride + uvx] >> 6) / 4.0f - 128.0f;
    float V = (float)(uv_plane[uvy * uv_stride + uvx + 1] >> 6) / 4.0f - 128.0f;

    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    int idx = (dy * dst_w + dx) * 3;
    bgr_out[idx + 0] = fminf(fmaxf(B, 0.0f), 255.0f);
    bgr_out[idx + 1] = fminf(fmaxf(G, 0.0f), 255.0f);
    bgr_out[idx + 2] = fminf(fmaxf(R, 0.0f), 255.0f);
}

// CPU-decoded BGR uint8 → float32 BGR HWC with resize
__global__ void bgr8_to_bgrf_resize_kernel(
    const uint8_t* __restrict__ bgr_in, float* __restrict__ bgr_out,
    int src_w, int src_h, int src_stride, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float sx = ((float)dx + 0.5f) * src_w / dst_w - 0.5f;
    float sy = ((float)dy + 0.5f) * src_h / dst_h - 0.5f;
    int sx0 = (int)floorf(sx), sy0 = (int)floorf(sy);
    float wx = sx - sx0, wy = sy - sy0;
    int sx1 = min(sx0 + 1, src_w - 1), sy1 = min(sy0 + 1, src_h - 1);
    sx0 = max(sx0, 0); sy0 = max(sy0, 0);

    int idx = (dy * dst_w + dx) * 3;
    for (int c = 0; c < 3; c++) {
        float v = (1-wy)*((1-wx)*bgr_in[sy0*src_stride+sx0*3+c] + wx*bgr_in[sy0*src_stride+sx1*3+c])
                + wy*((1-wx)*bgr_in[sy1*src_stride+sx0*3+c] + wx*bgr_in[sy1*src_stride+sx1*3+c]);
        bgr_out[idx + c] = v;
    }
}

// ============ Image preprocessing kernel ============

__global__ void preprocess_image_kernel(
    const float* __restrict__ bgr_hwc,  // [H, W, 3] BGR uint8-range
    float* __restrict__ rgb_nchw,       // [1, 3, H, W] normalized
    int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int i = idx / W, j = idx % W;

    // BGR -> RGB and normalize
    float mean[3] = {0.406f, 0.456f, 0.485f};  // BGR order
    float std_[3] = {0.225f, 0.224f, 0.229f};

    for (int c = 0; c < 3; c++) {
        float val = bgr_hwc[i * W * 3 + j * 3 + c] / 255.0f;
        // BGR channel c maps to RGB channel (2-c)
        int rgb_c = 2 - c;
        rgb_nchw[rgb_c * H * W + i * W + j] = (val - mean[c]) / std_[c];
    }
}

// ============ Correlation volume build and sample ============

__global__ void corr_index_kernel(
    const float* __restrict__ volume,  // [H1*W1, H2*W2]
    const float* __restrict__ coords,  // [2, H1, W1]
    float* __restrict__ corr,          // [D*D, H1, W1]
    int r, int H1, int W1, int H2, int W2,
    float coord_scale = 1.0f)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= H1 || x >= W1) return;

    int D = 2 * r + 1;
    float x0 = coords[0 * H1 * W1 + y * W1 + x] * coord_scale;
    float y0 = coords[1 * H1 * W1 + y * W1 + x] * coord_scale;
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);

    int rd = 2 * r + 1;
    for (int i = 0; i < rd + 1; i++) {
        for (int j = 0; j < rd + 1; j++) {
            int x1 = (int)floorf(x0) - r + i;
            int y1 = (int)floorf(y0) - r + j;
            if (y1 >= 0 && y1 < H2 && x1 >= 0 && x1 < W2) {
                float s = volume[(y * W1 + x) * H2 * W2 + y1 * W2 + x1];
                if (i > 0 && j > 0)
                    atomicAdd(&corr[((i-1)*D + (j-1)) * H1 * W1 + y * W1 + x], s * dx * dy);
                if (i > 0 && j < rd)
                    atomicAdd(&corr[((i-1)*D + j) * H1 * W1 + y * W1 + x], s * dx * (1.0f-dy));
                if (i < rd && j > 0)
                    atomicAdd(&corr[(i*D + (j-1)) * H1 * W1 + y * W1 + x], s * (1.0f-dx) * dy);
                if (i < rd && j < rd)
                    atomicAdd(&corr[(i*D + j) * H1 * W1 + y * W1 + x], s * (1.0f-dx) * (1.0f-dy));
            }
        }
    }
}

// Scale coordinates in-place
__global__ void scale_coords(float* coords, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) coords[idx] *= scale;
}

// Convert coords from [H,W,2] (projmap output) to [2,H,W] (channel-first for sampling)
__global__ void coords_hw2_to_2hw(
    const float* __restrict__ hw2,  // [H*W*2] interleaved
    float* __restrict__ chw,        // [2*H*W] channel-first
    int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HW) return;
    chw[0 * HW + idx] = hw2[idx * 2 + 0];  // x channel
    chw[1 * HW + idx] = hw2[idx * 2 + 1];  // y channel
}

// Batched version: [M, H*W, 2] -> [M, 2, H*W]
__global__ void batch_coords_hw2_to_2hw(
    const float* __restrict__ hw2,  // [M*H*W*2]
    float* __restrict__ chw,        // [M*2*H*W]
    int M, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * HW) return;
    int e = idx / HW;
    int k = idx % HW;
    chw[e * 2 * HW + 0 * HW + k] = hw2[e * HW * 2 + k * 2 + 0];
    chw[e * 2 * HW + 1 * HW + k] = hw2[e * HW * 2 + k * 2 + 1];
}

// Simple 2x2 average pooling kernel for batched correlation pyramid building
// Input: [N, H, W], Output: [N, H/2, W/2]
__global__ void avg_pool_2x2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oH = H / 2, oW = W / 2;
    int total = N * oH * oW;
    if (idx >= total) return;

    int n = idx / (oH * oW);
    int rem = idx % (oH * oW);
    int oy = rem / oW;
    int ox = rem % oW;

    int iy = oy * 2, ix = ox * 2;
    const float* in_n = input + (size_t)n * H * W;
    float val = 0.25f * (in_n[iy * W + ix] + in_n[iy * W + ix + 1] +
                         in_n[(iy + 1) * W + ix] + in_n[(iy + 1) * W + ix + 1]);
    output[(size_t)n * oH * oW + oy * oW + ox] = val;
}

// Batched correlation sampling: sample from batch of correlation volumes+pyramids
// Output layout: [M, 196, hw] where 196 = 4_levels * 7*7
// This kernel handles one level; level_idx offsets into the 196 channels
__global__ void batch_corr_sample_kernel(
    const float* __restrict__ volumes,  // [M, hw, H2, W2] for this level
    const float* __restrict__ coords,   // [M, 2, H1, W1]
    float* __restrict__ corr_out,       // [M, 196, H1*W1] base pointer
    int M, int r, int H1, int W1, int H2, int W2, int hw,
    float coord_scale, int level_idx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int e = blockIdx.z;
    if (y >= H1 || x >= W1 || e >= M) return;

    int D = 2 * r + 1;
    int hw1 = H1 * W1;
    float x0 = coords[e * 2 * hw1 + 0 * hw1 + y * W1 + x] * coord_scale;
    float y0 = coords[e * 2 * hw1 + 1 * hw1 + y * W1 + x] * coord_scale;
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);

    int rd = 2 * r + 1;
    const float* vol = volumes + (size_t)e * hw * H2 * W2;
    // Output offset: edge e starts at e*196*hw1, this level starts at level_idx*D*D*hw1
    float* out = corr_out + (size_t)e * 196 * hw1 + level_idx * D * D * hw1;
    int src = y * W1 + x;

    for (int i = 0; i < rd + 1; i++) {
        for (int j = 0; j < rd + 1; j++) {
            int x1 = (int)floorf(x0) - r + i;
            int y1 = (int)floorf(y0) - r + j;
            if (y1 >= 0 && y1 < H2 && x1 >= 0 && x1 < W2) {
                float s = vol[src * H2 * W2 + y1 * W2 + x1];
                if (i > 0 && j > 0)
                    out[((i-1)*D + (j-1)) * hw1 + src] += s * dx * dy;
                if (i > 0 && j < rd)
                    out[((i-1)*D + j) * hw1 + src] += s * dx * (1.0f-dy);
                if (i < rd && j > 0)
                    out[(i*D + (j-1)) * hw1 + src] += s * (1.0f-dx) * dy;
                if (i < rd && j < rd)
                    out[(i*D + j) * hw1 + src] += s * (1.0f-dx) * (1.0f-dy);
            }
        }
    }
}

// ============ Batched edge gather/scatter kernels ============

// Gather per-keyframe features into contiguous batch buffer using edge indices
// out[e, :, :, :] = src[indices[e], :, :, :]
__global__ void gather_features_kernel(
    float* __restrict__ out,
    const float* __restrict__ src,
    const int* __restrict__ indices,
    int num_edges, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_edges * C * HW;
    if (idx >= total) return;

    int hw = idx % HW;
    int c = (idx / HW) % C;
    int e = idx / (HW * C);

    int src_idx = indices[e];
    out[e * C * HW + c * HW + hw] = src[src_idx * C * HW + c * HW + hw];
}

// Scatter batch results back to per-edge storage (just a copy, edges are already indexed)
// dst[e, :, :, :] = src[e, :, :, :]  (identity, used when edge_nets is the batch buffer itself)

// Compute motion features: [flow_x, flow_y, resd_x, resd_y]
// flow = coords1 - coords0, residual = target - coords1
__global__ void compute_motion_kernel(
    float* __restrict__ motion,       // [batch, 4, h, w] output
    const float* __restrict__ coords1,  // [batch, 2, h, w]
    const float* __restrict__ coords0,  // [batch, 2, h, w] identity grid
    const float* __restrict__ target,   // [batch, 2, h, w] previous target (or coords1 for step 0)
    int batch, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * HW) return;
    int e = idx / HW, k = idx % HW;

    float c1x = coords1[e * 2 * HW + 0 * HW + k];
    float c1y = coords1[e * 2 * HW + 1 * HW + k];
    float c0x = coords0[e * 2 * HW + 0 * HW + k];
    float c0y = coords0[e * 2 * HW + 1 * HW + k];
    float tx = target[e * 2 * HW + 0 * HW + k];
    float ty = target[e * 2 * HW + 1 * HW + k];

    // Clamp to [-64, 64] like PyTorch
    motion[e * 4 * HW + 0 * HW + k] = fminf(fmaxf(c1x - c0x, -64.0f), 64.0f);
    motion[e * 4 * HW + 1 * HW + k] = fminf(fmaxf(c1y - c0y, -64.0f), 64.0f);
    motion[e * 4 * HW + 2 * HW + k] = fminf(fmaxf(tx - c1x, -64.0f), 64.0f);
    motion[e * 4 * HW + 3 * HW + k] = fminf(fmaxf(ty - c1y, -64.0f), 64.0f);
}

// Compute target = coords + delta (first 2 channels of delta)
// Also extract weight (first 2 channels of weight output)
__global__ void compute_target_kernel(
    float* __restrict__ target,     // [num_edges, 2, h, w] output
    float* __restrict__ weight_out, // [num_edges, 2, h, w] output
    const float* __restrict__ coords,  // [num_edges, 2, h, w]
    const float* __restrict__ delta,   // [num_edges, 3, h, w]
    const float* __restrict__ wt_raw,  // [num_edges, 3, h, w]
    int num_edges, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_edges * 2 * HW;
    if (idx >= total) return;

    int hw = idx % HW;
    int c = (idx / HW) % 2;
    int e = idx / (HW * 2);

    // target = coords + delta (first 2 channels)
    target[e * 2 * HW + c * HW + hw] =
        coords[e * 2 * HW + c * HW + hw] + delta[e * 3 * HW + c * HW + hw];

    // weight = sigmoid output (already applied), first 2 channels
    weight_out[e * 2 * HW + c * HW + hw] = wt_raw[e * 3 * HW + c * HW + hw];
}

// ============ Main DROID-SLAM class ============

struct CudaDroid {
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    HalfBasicEncoder fnet, cnet;  // FP16 encoders with tensor cores
    UpdateModule update;           // stays FP32
    BundleAdjustment ba;
    DroidState state;

    static const int MAX_BATCH_EDGES = 48;

    // Working buffers — FP32 for update module
    GpuBuf buf_a, buf_b, buf_c, workspace;
    GpuBuf preproc_buf;  // preprocessed image (FP32)

    // FP16 working buffers for encoders
    GpuHalfBuf half_buf_a, half_buf_b, half_buf_c;

    // Correlation buffers (per-edge, reused in loop)
    GpuBuf corr_volume;  // [H*W, H*W] for one edge (motion filter)
    GpuBuf corr_pyramid[4];
    GpuBuf coords_scaled; // [2, h, w]

    // Batched correlation buffers
    GpuBuf batch_corr_vols;  // [CORR_BATCH, hw, hw]
    GpuBuf batch_corr_pyr[4]; // batched pyramid levels
    __half **d_Aptr_h, **d_Bptr_h;  // FP16 fmap pointer arrays (device)
    __half **h_Aptr_h, **h_Bptr_h;  // FP16 fmap pointer arrays (host)
    float **d_Cptr, **h_Cptr;       // FP32 correlation output pointer arrays

    // Batched buffers for edge processing
    GpuBuf batch_corr;    // [MAX_BATCH, 196, h, w]
    GpuBuf batch_coords;      // [MAX_BATCH, 2, h, w] - channel-first format
    GpuBuf batch_coords_hw2;  // [MAX_BATCH, h*w, 2] - projmap output (interleaved)
    GpuBuf coords_hw2;        // [h*w*2] temp for projmap output (single edge)
    GpuBuf batch_motion;  // [MAX_BATCH, 4, h, w]
    GpuBuf batch_nets;    // [MAX_BATCH, 128, h, w] - gathered edge nets
    GpuBuf batch_inps;    // [MAX_BATCH, 128, h, w] - gathered inp features
    GpuBuf batch_delta;   // [MAX_BATCH, 3, h, w]
    GpuBuf batch_weight;  // [MAX_BATCH, 3, h, w]
    GpuBuf batch_eta;     // [MAX_BATCH, 1, h, w]

    // Timers
    CudaTimer t_encode{"encode"};
    CudaTimer t_corr{"correlation"};
    CudaTimer t_update{"update"};
    CudaTimer t_ba{"ba"};
    CudaTimer t_total{"total"};

    int H, W, h, w;
    bool verbose = false;

    void init(int fullH, int fullW, float fx, float fy, float cx, float cy,
              const char* weight_dir) {
        H = fullH; W = fullW;
        h = H / 8; w = W / 8;
        int hw = h * w;

        CUDNN_CHECK(cudnnCreate(&cudnn));
        CUBLAS_CHECK(cublasCreate(&cublas));

        // Load weights
        WeightStore ws;
        ws.set_dir(weight_dir);

        printf("Initializing encoders (FP16)...\n");
        fnet.init(cudnn, ws, "fnet", 1, H, W, 128, true);   // FP16 with instance norm
        cnet.init(cudnn, ws, "cnet", 1, H, W, 256, false);  // FP16 without instance norm

        printf("Initializing update module...\n");
        update.init(cudnn, ws, MAX_BATCH_EDGES, h, w);

        printf("Initializing BA...\n");
        ba.init(h, w);

        printf("Initializing state...\n");
        state.init(H, W, fx, fy, cx, cy);

        // Allocate working buffers (large enough for any intermediate)
        // Must handle both encoder (batch=1) and update (batch=MAX_BATCH_EDGES)
        size_t max_buf = std::max({
            (size_t)(1 * 256 * H/2 * W/2),              // encoder after conv1
            (size_t)(MAX_BATCH_EDGES * 128 * hw),        // update temp buffers
            (size_t)(MAX_BATCH_EDGES * 448 * hw),        // GRU input (largest)
        });
        buf_a.alloc(max_buf);
        buf_b.alloc(max_buf);
        buf_c.alloc(max_buf);
        workspace.alloc(64 * 1024 * 1024 / sizeof(float));  // 64MB workspace
        preproc_buf.alloc(3 * H * W);

        // FP16 encoder buffers (must handle largest intermediate: 256 * H/2 * W/2)
        size_t half_buf_size = std::max((size_t)(256 * H/2 * W/2), (size_t)(128 * hw));
        half_buf_a.alloc(half_buf_size);
        half_buf_b.alloc(half_buf_size);
        half_buf_c.alloc(half_buf_size);

        // Correlation buffers — single-edge (for motion filter)
        corr_volume.alloc(hw * hw);
        for (int l = 1; l < 4; l++) {
            int ph = h >> l, pw = w >> l;
            corr_pyramid[l].alloc(hw * ph * pw);
        }
        coords_scaled.alloc(2 * hw);

        // Batched correlation buffers (for run_update_pass)
        static const int CORR_BATCH = MAX_BATCH_EDGES;  // batch all edges at once
        batch_corr_vols.alloc((size_t)CORR_BATCH * hw * hw);
        for (int l = 1; l < 4; l++) {
            int ph = h >> l, pw = w >> l;
            batch_corr_pyr[l].alloc((size_t)CORR_BATCH * hw * ph * pw);
        }
        // Device pointer arrays for batched FP16 correlation GEMM
        CUDA_CHECK(cudaMalloc(&d_Aptr_h, CORR_BATCH * sizeof(__half*)));
        CUDA_CHECK(cudaMalloc(&d_Bptr_h, CORR_BATCH * sizeof(__half*)));
        CUDA_CHECK(cudaMalloc(&d_Cptr, CORR_BATCH * sizeof(float*)));
        h_Aptr_h = new __half*[CORR_BATCH];
        h_Bptr_h = new __half*[CORR_BATCH];
        h_Cptr = new float*[CORR_BATCH];

        // Batched edge buffers
        batch_corr.alloc(MAX_BATCH_EDGES * 196 * hw);
        batch_coords.alloc(MAX_BATCH_EDGES * 2 * hw);
        batch_coords_hw2.alloc(MAX_BATCH_EDGES * hw * 2);  // projmap output [M,HW,2]
        batch_motion.alloc(MAX_BATCH_EDGES * 4 * hw);
        batch_nets.alloc(MAX_BATCH_EDGES * 128 * hw);
        batch_inps.alloc(MAX_BATCH_EDGES * 128 * hw);
        batch_delta.alloc(MAX_BATCH_EDGES * 3 * hw);
        batch_weight.alloc(MAX_BATCH_EDGES * 3 * hw);
        batch_eta.alloc(MAX_BATCH_EDGES * 1 * hw);

        alloc_encoder_bufs();
        alloc_motion_filter_bufs();
        alloc_coords0();
        printf("CudaDroid initialized: %dx%d -> %dx%d\n", H, W, h, w);
    }

    // Pre-allocated encoder output buffers
    GpuHalfBuf enc_fmap;  // FP16 [128, h, w] — stored directly in state.fmaps
    GpuHalfBuf enc_cmap;  // FP16 [256, h, w] — converted to FP32 for update module
    GpuBuf enc_cmap_f32;  // FP32 conversion of cnet output
    GpuBuf enc_net, enc_inp;

    void alloc_encoder_bufs() {
        int hw = h * w;
        enc_fmap.alloc(128 * hw);
        enc_cmap.alloc(256 * hw);
        enc_cmap_f32.alloc(256 * hw);
        enc_net.alloc(128 * hw);
        enc_inp.alloc(128 * hw);
    }

    // Run feature encoder on one frame (FP16 encoders, FP32 update state)
    void encode_frame(float* bgr_hwc) {
        t_encode.begin();

        // Preprocess: BGR HWC -> RGB NCHW normalized (FP32)
        preprocess_image_kernel<<<(H*W+255)/256, 256>>>(
            bgr_hwc, preproc_buf.data, H, W);

        int hw = h * w;

        // Run fnet (FP16): [1, 3, H, W] -> [1, 128, h, w] FP16
        fnet.forward(cudnn, preproc_buf.data, enc_fmap.data,
                     half_buf_a.data, half_buf_b.data, half_buf_c.data,
                     workspace.data, 1, H, W);

        // Run cnet (FP16): [1, 3, H, W] -> [1, 256, h, w] FP16
        cnet.forward(cudnn, preproc_buf.data, enc_cmap.data,
                     half_buf_a.data, half_buf_b.data, half_buf_c.data,
                     workspace.data, 1, H, W);

        // Convert cnet output to FP32 for update module
        half_to_float(enc_cmap.data, enc_cmap_f32.data, 256 * hw);

        // Split cnet output (FP32): [256, h, w] -> net[128, h, w] + inp[128, h, w]
        slice_channels(enc_net.data, enc_cmap_f32.data, 1, 256, 0, 128, hw);
        slice_channels(enc_inp.data, enc_cmap_f32.data, 1, 256, 128, 128, hw);

        // Apply activations: tanh(net), relu(inp)
        tanh_inplace(enc_net.data, 128 * hw);
        relu_inplace(enc_inp.data, 128 * hw);

        // Note: enc_fmap is FP16, enc_net/enc_inp are FP32
        t_encode.end();
    }

    // Build correlation volume between frames i and j (FP16 fmaps → FP32 output)
    void build_correlation(int i, int j) {
        int hw = h * w;
        __half* f1 = state.fmaps.data + i * 128 * hw;
        __half* f2 = state.fmaps.data + j * 128 * hw;

        // FP16 × FP16 → FP32 correlation with tensor cores
        float alpha = 1.0f / 16.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            hw, hw, 128,
            &alpha,
            f2, CUDA_R_16F, hw,
            f1, CUDA_R_16F, hw,
            &beta,
            corr_volume.data, CUDA_R_32F, hw,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Build pyramid via avg_pool2d
        // Level 0 IS corr_volume; build levels 1-3 via avg_pool
        avg_pool2d(cudnn, corr_volume.data, corr_pyramid[1].data,
                   hw, 1, h, w, 2, 2, 2, 2);
        for (int l = 2; l < 4; l++) {
            int ph = h >> (l-1), pw = w >> (l-1);
            avg_pool2d(cudnn, corr_pyramid[l-1].data, corr_pyramid[l].data,
                       hw, 1, ph, pw, 2, 2, 2, 2);
        }
    }

    // Sample correlation at given coordinates, write to output buffer
    void sample_correlation(float* coords_2hw, float* corr_out) {
        int hw = h * w;
        int D = 7;  // 2*3+1

        dim3 block(8, 8);
        dim3 grid((w + 7) / 8, (h + 7) / 8);

        for (int l = 0; l < 4; l++) {
            int ph = h >> l, pw = w >> l;
            float scale = 1.0f / (float)(1 << l);

            float* level_out = corr_out + l * D * D * hw;
            CUDA_CHECK(cudaMemset(level_out, 0, D * D * hw * sizeof(float)));

            // Level 0 uses corr_volume directly, levels 1-3 use pyramid
            float* level_data = (l == 0) ? corr_volume.data : corr_pyramid[l].data;
            // Pass coord_scale to avoid separate memcpy + scale kernel
            corr_index_kernel<<<grid, block>>>(
                level_data,
                coords_2hw,
                level_out,
                3, h, w, ph, pw, scale);
        }
    }

    // Batched correlation: build + pyramid + sample for multiple edges at once
    void batch_build_and_sample_correlation(int batch_size, int edge_start,
                                            float* coords_2hw, float* corr_out_196hw) {
        int hw = h * w;

        // 1. Setup pointer arrays for batched FP16 GEMM
        for (int e = 0; e < batch_size; e++) {
            int ii_val = state.ii_host[edge_start + e];
            int jj_val = state.jj_host[edge_start + e];
            h_Aptr_h[e] = state.fmaps.data + jj_val * 128 * hw;
            h_Bptr_h[e] = state.fmaps.data + ii_val * 128 * hw;
            h_Cptr[e] = batch_corr_vols.data + (size_t)e * hw * hw;
        }
        CUDA_CHECK(cudaMemcpy(d_Aptr_h, h_Aptr_h, batch_size * sizeof(__half*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Bptr_h, h_Bptr_h, batch_size * sizeof(__half*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Cptr, h_Cptr, batch_size * sizeof(float*), cudaMemcpyHostToDevice));

        // 2. Batched GEMM: FP16 × FP16 → FP32 with tensor cores
        float alpha = 1.0f / 16.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            hw, hw, 128,
            &alpha,
            (const void**)d_Aptr_h, CUDA_R_16F, hw,
            (const void**)d_Bptr_h, CUDA_R_16F, hw,
            &beta,
            (void**)d_Cptr, CUDA_R_32F, hw,
            batch_size,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // 3. Build pyramids via custom 2x2 avg_pool kernel
        {
            int N1 = batch_size * hw;
            int total1 = N1 * (h / 2) * (w / 2);
            avg_pool_2x2_kernel<<<(total1 + 255) / 256, 256>>>(
                batch_corr_vols.data, batch_corr_pyr[1].data, N1, h, w);
            for (int l = 2; l < 4; l++) {
                int ph = h >> (l-1), pw = w >> (l-1);
                int Nl = batch_size * hw;
                int totalL = Nl * (ph / 2) * (pw / 2);
                avg_pool_2x2_kernel<<<(totalL + 255) / 256, 256>>>(
                    batch_corr_pyr[l-1].data, batch_corr_pyr[l].data, Nl, ph, pw);
            }
        }

        // 4. Batched sampling into [M, 196, hw] layout
        CUDA_CHECK(cudaMemset(corr_out_196hw, 0, (size_t)batch_size * 196 * hw * sizeof(float)));

        dim3 block(8, 8);
        dim3 grid((w + 7) / 8, (h + 7) / 8, batch_size);

        for (int l = 0; l < 4; l++) {
            int ph = h >> l, pw = w >> l;
            float scale = 1.0f / (float)(1 << l);
            float* level_data = (l == 0) ? batch_corr_vols.data : batch_corr_pyr[l].data;

            batch_corr_sample_kernel<<<grid, block>>>(
                level_data, coords_2hw, corr_out_196hw,
                batch_size, 3, h, w, ph, pw, hw, scale, l);
        }
    }

    // Motion filter parameters (matching DROID-SLAM defaults)
    float filter_thresh = 2.4f;
    int warmup = 8;
    int edge_radius = 2;
    int frontend_window = 25;  // sliding window size (matches PyTorch DROID-SLAM)
    int update_steps = 3;     // GRU update steps per keyframe

    // Buffers for GRU-based motion filter (single edge)
    GpuBuf mf_corr, mf_coords, mf_motion, mf_nets, mf_inps;
    GpuBuf mf_delta, mf_weight, mf_eta;

    void alloc_motion_filter_bufs() {
        int hw = h * w;
        mf_corr.alloc(196 * hw);
        mf_coords.alloc(2 * hw);
        mf_motion.alloc(4 * hw);
        mf_nets.alloc(128 * hw);
        mf_inps.alloc(128 * hw);
        mf_delta.alloc(3 * hw);
        mf_weight.alloc(3 * hw);
        mf_eta.alloc(hw);
    }

    // Compute GRU-predicted optical flow between last keyframe and current frame
    // Returns mean flow magnitude
    float compute_motion_gru(int last_kf_idx) {
        int hw = h * w;

        // Build correlation between last keyframe and current frame (in temp slot)
        int temp_slot = state.num_keyframes;
        build_correlation(last_kf_idx, temp_slot);

        // Initialize coordinates as identity grid (pixel coordinates at 1/8 res)
        std::vector<float> grid(2 * hw);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                grid[0 * h * w + y * w + x] = (float)x;  // x channel
                grid[1 * h * w + y * w + x] = (float)y;  // y channel
            }
        mf_coords.copyFrom(grid.data(), 2 * hw);

        // Sample correlation at identity coords
        sample_correlation(mf_coords.data, mf_corr.data);

        // Initialize nets from last keyframe, inps from last keyframe
        CUDA_CHECK(cudaMemcpy(mf_nets.data, state.nets.data + last_kf_idx * 128 * hw,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(mf_inps.data, state.inps.data + last_kf_idx * 128 * hw,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));

        // Zero motion
        CUDA_CHECK(cudaMemset(mf_motion.data, 0, 4 * hw * sizeof(float)));

        // Run single GRU iteration with batch=1
        update.forward(cudnn,
            mf_corr.data, mf_motion.data,
            mf_nets.data, mf_inps.data,
            mf_delta.data, mf_weight.data, mf_eta.data,
            buf_a.data, buf_b.data, workspace.data,
            1, h, w);

        // Compute mean flow magnitude from delta [3, h, w] (first 2 channels are dx, dy)
        std::vector<float> delta_host(3 * hw);
        CUDA_CHECK(cudaMemcpy(delta_host.data(), mf_delta.data,
                              3 * hw * sizeof(float), cudaMemcpyDeviceToHost));

        float sum_mag = 0;
        for (int k = 0; k < hw; k++) {
            float dx = delta_host[0 * hw + k];
            float dy = delta_host[1 * hw + k];
            sum_mag += sqrtf(dx * dx + dy * dy);
        }
        return sum_mag / hw;
    }

    // Process one frame with motion filtering and persistent edges
    void process_frame(int frame_t, float* bgr_hwc_gpu) {
        t_total.begin();

        // 1. Always encode features
        encode_frame(bgr_hwc_gpu);

        int hw = h * w;
        int nk = state.num_keyframes;

        // 2. First frame: always a keyframe
        if (nk == 0) {
            state.add_keyframe(frame_t, enc_fmap.data, enc_net.data, enc_inp.data);
            t_total.end();
            return;
        }

        // 3. Store fmap temporarily at slot num_keyframes for correlation computation
        CUDA_CHECK(cudaMemcpy(state.fmaps.data + nk * 128 * hw, enc_fmap.data,
                              128 * hw * sizeof(__half), cudaMemcpyDeviceToDevice));

        // 4. Motion filter: use GRU-predicted flow relative to last keyframe
        float flow = compute_motion_gru(nk - 1);

        // During warmup or if flow exceeds threshold, add as keyframe
        if (flow < filter_thresh && nk >= warmup) {
            t_total.end();
            return;  // Not enough motion, skip
        }

        // 5. Accept as keyframe
        state.add_keyframe(frame_t, enc_fmap.data, enc_net.data, enc_inp.data);
        int new_kf = state.num_keyframes - 1;

        // 6. Add edges connecting new keyframe to recent keyframes
        int prev_num_edges = state.num_edges();
        state.add_edges_for_keyframe(new_kf, edge_radius);

        // Prune edges outside the frontend window
        if (new_kf >= frontend_window) {
            state.remove_old_edges(new_kf - frontend_window + 1, hw);
        }

        state.sync_edges_to_gpu();
        int total_edges = state.num_edges();

        // Initialize hidden states for new edges from keyframe nets
        for (int e = prev_num_edges; e < total_edges; e++) {
            int ii_val = state.ii_host[e];
            CUDA_CHECK(cudaMemcpy(state.edge_nets.data + e * 128 * hw,
                                  state.nets.data + ii_val * 128 * hw,
                                  128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        if (total_edges == 0) { t_total.end(); return; }

        // 7. Run full DROID-SLAM update: N steps of [reproject + corr + GRU + BA]
        run_update_pass(total_edges, update_steps);

        t_total.end();
    }

    // Persistent coords0 buffer (identity grid) for computing flow
    GpuBuf coords0_buf;  // [MAX_BATCH, 2, h, w]

    void alloc_coords0() {
        int hw = h * w;
        coords0_buf.alloc(MAX_BATCH_EDGES * 2 * hw);
        // Fill with identity grid (pixel coordinates)
        std::vector<float> grid(MAX_BATCH_EDGES * 2 * hw);
        for (int e = 0; e < MAX_BATCH_EDGES; e++) {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    grid[e * 2 * hw + 0 * hw + y * w + x] = (float)x;
                    grid[e * 2 * hw + 1 * hw + y * w + x] = (float)y;
                }
        }
        coords0_buf.copyFrom(grid.data(), MAX_BATCH_EDGES * 2 * hw);
    }

    // Run full DROID-SLAM update: 3 steps of [reproject + corr + GRU + BA]
    void run_update_pass(int total_edges, int num_steps = 3) {
        int hw = h * w;

        // Load persistent edge hidden states into batch buffers (for all edges, batched)
        // First, load inps (constant across steps)
        for (int bs = 0; bs < total_edges; bs += MAX_BATCH_EDGES) {
            int batch_size = std::min(MAX_BATCH_EDGES, total_edges - bs);
            for (int e = 0; e < batch_size; e++) {
                int ii_val = state.ii_host[bs + e];
                CUDA_CHECK(cudaMemcpy(batch_inps.data + e * 128 * hw,
                                      state.inps.data + ii_val * 128 * hw,
                                      128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            // Store inp batch back to a persistent location (we'll re-load it each step)
            // Actually, let's just re-gather inps each batch. They don't change.
        }

        for (int step = 0; step < num_steps; step++) {
            t_corr.begin();
            t_update.begin();

            // Process all edges in batches: reproject, correlate, GRU
            for (int bs = 0; bs < total_edges; bs += MAX_BATCH_EDGES) {
                int batch_size = std::min(MAX_BATCH_EDGES, total_edges - bs);

                // 1. Batched reproject using current poses -> coords1
                projmap_kernel<<<batch_size, 256>>>(
                    state.poses.data, state.disps.data, state.intrinsics.data,
                    (const int*)state.ii_gpu.data + bs,
                    (const int*)state.jj_gpu.data + bs,
                    batch_coords_hw2.data, buf_a.data, batch_size, h, w);

                // 2. Batched coords layout conversion [M,HW,2] -> [M,2,HW]
                batch_coords_hw2_to_2hw<<<(batch_size * hw + 255) / 256, 256>>>(
                    batch_coords_hw2.data, batch_coords.data, batch_size, hw);

                // 3. Batched correlation: GEMM + pyramid + sampling
                batch_build_and_sample_correlation(batch_size, bs,
                    batch_coords.data, batch_corr.data);

                // 3. Compute motion: [flow, residual] = [coords1 - coords0, target - coords1]
                // For step 0: target = coords1, so residual = 0, flow = coords1 - identity
                // For step > 0: target was set in previous step, residual = target - coords1
                {
                    // flow channels 0,1: coords1 - coords0
                    // flow channels 2,3: target - coords1 (zero for first step)
                    int total = batch_size * hw;
                    // Compute on CPU for clarity (small data transfer)
                    CUDA_CHECK(cudaMemset(batch_motion.data, 0, batch_size * 4 * hw * sizeof(float)));

                    // flow_x = coords1_x - pixel_x, flow_y = coords1_y - pixel_y
                    // These are batched, and we already have batch_coords (coords1) and coords0_buf
                    // Let's launch a kernel for this
                    compute_motion_kernel<<<(total+255)/256, 256>>>(
                        batch_motion.data,
                        batch_coords.data,
                        coords0_buf.data,
                        (step > 0) ? (state.target.data + bs * 2 * hw) : batch_coords.data,
                        batch_size, hw);
                }

                // 4. Load hidden states and inps
                for (int e = 0; e < batch_size; e++) {
                    CUDA_CHECK(cudaMemcpy(batch_nets.data + e * 128 * hw,
                                          state.edge_nets.data + (bs + e) * 128 * hw,
                                          128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                    int ii_val = state.ii_host[bs + e];
                    CUDA_CHECK(cudaMemcpy(batch_inps.data + e * 128 * hw,
                                          state.inps.data + ii_val * 128 * hw,
                                          128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                }

                // 5. GRU update
                update.forward(cudnn,
                    batch_corr.data, batch_motion.data,
                    batch_nets.data, batch_inps.data,
                    batch_delta.data, batch_weight.data, batch_eta.data,
                    buf_a.data, buf_b.data, workspace.data,
                    batch_size, h, w);

                // 6. target = coords1 + delta
                {
                    int total = batch_size * 2 * hw;
                    compute_target_kernel<<<(total+255)/256, 256>>>(
                        state.target.data + bs * 2 * hw,
                        state.weight.data + bs * 2 * hw,
                        batch_coords.data, batch_delta.data, batch_weight.data,
                        batch_size, hw);
                }


                // 7. Save hidden states
                for (int e = 0; e < batch_size; e++) {
                    CUDA_CHECK(cudaMemcpy(state.edge_nets.data + (bs + e) * 128 * hw,
                                          batch_nets.data + e * 128 * hw,
                                          128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
                }
            }

            t_corr.end();
            t_update.end();


            // 8. BA with all edges (2 iterations like PyTorch)
            t_ba.begin();
            int t0 = 1;
            int t1 = state.num_keyframes;
            for (int ba_iter = 0; ba_iter < 2; ba_iter++) {
                ba.iterate(state.poses.data, state.disps.data, state.intrinsics.data,
                           state.target.data, state.weight.data,
                           state.disps.data,
                           state.ii_gpu.data, state.jj_gpu.data,
                           state.ii_host.data(), state.jj_host.data(),
                           total_edges, t0, t1,
                           1e-4f, 0.1f, true);
            }
            t_ba.end();

            // (verbose pose output removed)
        }
    }

    // Backend optimization: rebuild dense edges and run multiple update+BA iterations
    void backend(int iters, int radius = 2) {
        int N = state.num_keyframes;
        int hw = h * w;
        if (N < 3) return;

        printf("Backend: %d iterations, radius %d, %d keyframes\n", iters, radius, N);

        // Build dense proximity edges (replacing frontend edges)
        state.ii_host.clear();
        state.jj_host.clear();
        for (int i = 0; i < N; i++) {
            for (int j = std::max(0, i - radius); j < std::min(N, i + radius + 1); j++) {
                if (i != j) {
                    state.ii_host.push_back(i);
                    state.jj_host.push_back(j);
                }
            }
        }
        state.sync_edges_to_gpu();
        int total_edges = state.num_edges();

        // Initialize edge hidden states from keyframe nets
        for (int e = 0; e < total_edges; e++) {
            int ii_val = state.ii_host[e];
            CUDA_CHECK(cudaMemcpy(state.edge_nets.data + e * 128 * hw,
                                  state.nets.data + ii_val * 128 * hw,
                                  128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        for (int iter = 0; iter < iters; iter++) {
            // Run update pass: corr + GRU on all edges (batched)
            run_update_pass(total_edges);

            // BA with ALL edges
            ba.iterate(state.poses.data, state.disps.data, state.intrinsics.data,
                       state.target.data, state.weight.data,
                       state.disps.data,
                       state.ii_gpu.data, state.jj_gpu.data,
                       state.ii_host.data(), state.jj_host.data(),
                       total_edges, 1, N,
                       1e-4f, 0.1f, true);
        }
        printf("Backend done\n");
    }

    void print_timing(int num_frames) {
        printf("\n=== Timing Summary (%d frames) ===\n", num_frames);
        printf("  Encode:      %8.2f ms total, %6.2f ms/frame\n",
               t_encode.elapsed_ms, t_encode.elapsed_ms / num_frames);
        printf("  Correlation: %8.2f ms total, %6.2f ms/frame\n",
               t_corr.elapsed_ms, t_corr.elapsed_ms / num_frames);
        printf("  Update:      %8.2f ms total, %6.2f ms/frame\n",
               t_update.elapsed_ms, t_update.elapsed_ms / num_frames);
        printf("  BA:          %8.2f ms total, %6.2f ms/frame\n",
               t_ba.elapsed_ms, t_ba.elapsed_ms / num_frames);
        printf("  Total:       %8.2f ms total, %6.2f ms/frame (%5.1f fps)\n",
               t_total.elapsed_ms, t_total.elapsed_ms / num_frames,
               1000.0f * num_frames / t_total.elapsed_ms);
    }

    void destroy() {
        cudnnDestroy(cudnn);
        cublasDestroy(cublas);
        ba.destroy();
    }
};

// ============ Main ============

int main(int argc, char** argv) {
    // Parse arguments
    const char* weight_dir = "data/weights";
    const char* frame_dir = "data/frames";
    const char* calib_file = "data/calib.bin";
    const char* pose_output = nullptr;
    const char* video_path = nullptr;
    int max_frames = 100;
    int backend_iters1 = 0, backend_iters2 = 0;
    int backend_radius = 2;
    bool cam_to_world = false;
    bool stdin_mode = false;
    int stdin_h = 0, stdin_w = 0;
    int resize_h = 0, resize_w = 0;
    int frontend_window = 25;
    int update_steps = 3;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) weight_dir = argv[++i];
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc) frame_dir = argv[++i];
        else if (strcmp(argv[i], "--calib") == 0 && i+1 < argc) calib_file = argv[++i];
        else if (strcmp(argv[i], "--max-frames") == 0 && i+1 < argc) max_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pose-output") == 0 && i+1 < argc) pose_output = argv[++i];
        else if (strcmp(argv[i], "--video") == 0 && i+1 < argc) video_path = argv[++i];
        else if (strcmp(argv[i], "--resize") == 0 && i+2 < argc) {
            resize_h = atoi(argv[++i]);
            resize_w = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--backend") == 0 && i+2 < argc) {
            backend_iters1 = atoi(argv[++i]);
            backend_iters2 = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--backend-radius") == 0 && i+1 < argc) backend_radius = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cam-to-world") == 0) cam_to_world = true;
        else if (strcmp(argv[i], "--stdin") == 0 && i+2 < argc) {
            stdin_mode = true;
            stdin_h = atoi(argv[++i]);
            stdin_w = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--frontend-window") == 0 && i+1 < argc) frontend_window = atoi(argv[++i]);
        else if (strcmp(argv[i], "--update-steps") == 0 && i+1 < argc) update_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--debug-dump") == 0) {
            // After processing 2 frames, dump intermediate values
            // This is handled after the main loop
        }
    }
    bool debug_dump = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--debug-dump") == 0) debug_dump = true;

    // Load calibration
    float calib[4];
    {
        FILE* f = fopen(calib_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", calib_file); return 1; }
        fread(calib, sizeof(float), 4, f);
        fclose(f);
        printf("Calibration: fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
               calib[0], calib[1], calib[2], calib[3]);
    }

    bool video_mode = (video_path != nullptr);
    int frameH, frameW;
    if (video_mode) {
        if (resize_h == 0 || resize_w == 0) {
            fprintf(stderr, "Error: --video requires --resize <h> <w>\n");
            return 1;
        }
        frameH = resize_h;
        frameW = resize_w;
        printf("Video mode: %s → %dx%d\n", video_path, frameW, frameH);
    } else if (stdin_mode) {
        frameH = stdin_h;
        frameW = stdin_w;
        printf("Stdin mode: %dx%d\n", frameH, frameW);
    } else {
        int totalFrames, stride;
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s/meta.txt", frame_dir);
        FILE* f = fopen(meta_path, "r");
        if (!f) { fprintf(stderr, "Cannot open %s\n", meta_path); return 1; }
        fscanf(f, "%d %d %d %d", &frameH, &frameW, &totalFrames, &stride);
        fclose(f);
        printf("Frames: %dx%d, %d total, stride=%d\n", frameH, frameW, totalFrames, stride);
    }

    // Initialize CUDA DROID
    CudaDroid droid;
    droid.init(frameH, frameW, calib[0], calib[1], calib[2], calib[3], weight_dir);
    droid.frontend_window = frontend_window;
    droid.update_steps = update_steps;
    if (debug_dump) droid.verbose = true;

    // Allocate frame buffer on GPU
    GpuBuf frame_gpu;
    int frame_floats = frameH * frameW * 3;
    frame_gpu.alloc(frame_floats);

    // Process frames
    int frames_processed = 0;
    if (video_mode) {
        // NVDEC video decode mode: decode directly to GPU, resize on GPU
        AVFormatContext* fmt_ctx = nullptr;
        AVCodecContext* dec_ctx = nullptr;
        SwsContext* sws_ctx = nullptr;
        AVFrame* av_frame = av_frame_alloc();
        AVFrame* bgr_frame = nullptr;
        AVPacket* pkt = av_packet_alloc();
        AVBufferRef* hw_device_ctx = nullptr;
        bool using_nvdec = false;
        bool nvdec_is_p010 = false;
        int video_stream_idx = -1;

        if (avformat_open_input(&fmt_ctx, video_path, nullptr, nullptr) < 0) {
            fprintf(stderr, "Cannot open video: %s\n", video_path);
            return 1;
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

        // Try NVDEC (CUVID) hardware decoder
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
                        printf("Using NVDEC (%s, %s) %dx%d → %dx%d\n",
                               cuvid_name, nvdec_is_p010 ? "P010" : "NV12",
                               srcW, srcH, frameW, frameH);
                    } else {
                        avcodec_free_context(&dec_ctx); dec_ctx = nullptr;
                    }
                }
            }
        }

        // GPU buffer for CPU decode upload (only if needed)
        GpuBuf d_src_bgr;

        // Fallback to CPU decoder
        if (!using_nvdec) {
            if (dec_ctx) avcodec_free_context(&dec_ctx);
            codec = avcodec_find_decoder(codecpar->codec_id);
            dec_ctx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(dec_ctx, codecpar);
            dec_ctx->thread_count = 4;
            avcodec_open2(dec_ctx, codec, nullptr);
            sws_ctx = sws_getContext(srcW, srcH, dec_ctx->pix_fmt,
                                     srcW, srcH, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);
            bgr_frame = av_frame_alloc();
            bgr_frame->format = AV_PIX_FMT_BGR24;
            bgr_frame->width = srcW;
            bgr_frame->height = srcH;
            av_frame_get_buffer(bgr_frame, 32);
            // Allocate GPU buffer for source BGR upload
            CUDA_CHECK(cudaMalloc((void**)&d_src_bgr.data, (size_t)srcW * srcH * 3));
            printf("Using CPU decoder %dx%d → %dx%d\n", srcW, srcH, frameW, frameH);
        }

        dim3 block(32, 8);
        dim3 grid((frameW + block.x - 1) / block.x, (frameH + block.y - 1) / block.y);

        // Decode loop
        auto decode_next = [&]() -> bool {
            while (av_read_frame(fmt_ctx, pkt) >= 0) {
                if (pkt->stream_index != video_stream_idx) { av_packet_unref(pkt); continue; }
                int ret = avcodec_send_packet(dec_ctx, pkt);
                av_packet_unref(pkt);
                if (ret < 0) continue;
                ret = avcodec_receive_frame(dec_ctx, av_frame);
                if (ret == 0) return true;
            }
            avcodec_send_packet(dec_ctx, nullptr);
            return avcodec_receive_frame(dec_ctx, av_frame) == 0;
        };

        while (frames_processed < max_frames && decode_next()) {
            if (using_nvdec) {
                // NVDEC: convert NV12/P010 → float32 BGR with resize, directly on GPU
                if (nvdec_is_p010) {
                    p010_to_bgr_resize_kernel<<<grid, block>>>(
                        (const uint16_t*)av_frame->data[0],
                        (const uint16_t*)av_frame->data[1],
                        frame_gpu.data, srcW, srcH,
                        av_frame->linesize[0], av_frame->linesize[1],
                        frameW, frameH);
                } else {
                    nv12_to_bgr_resize_kernel<<<grid, block>>>(
                        av_frame->data[0], av_frame->data[1],
                        frame_gpu.data, srcW, srcH,
                        av_frame->linesize[0], av_frame->linesize[1],
                        frameW, frameH);
                }
            } else {
                // CPU decode: sws_scale → upload → GPU resize
                sws_scale(sws_ctx, av_frame->data, av_frame->linesize, 0, srcH,
                           bgr_frame->data, bgr_frame->linesize);
                CUDA_CHECK(cudaMemcpy(d_src_bgr.data, bgr_frame->data[0],
                                      (size_t)srcW * srcH * 3, cudaMemcpyHostToDevice));
                bgr8_to_bgrf_resize_kernel<<<grid, block>>>(
                    (const uint8_t*)d_src_bgr.data, frame_gpu.data,
                    srcW, srcH, srcW * 3, frameW, frameH);
            }
            droid.process_frame(frames_processed, frame_gpu.data);
            frames_processed++;
        }

        // Cleanup
        av_frame_free(&av_frame);
        if (bgr_frame) av_frame_free(&bgr_frame);
        av_packet_free(&pkt);
        if (sws_ctx) sws_freeContext(sws_ctx);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
        d_src_bgr.free();
    } else if (stdin_mode) {
        // Read frames from stdin: raw float32 HWC BGR data, no per-frame header
        float* pinned_buf;
        CUDA_CHECK(cudaMallocHost(&pinned_buf, frame_floats * sizeof(float)));
        size_t frame_bytes = (size_t)frame_floats * sizeof(float);

        while (frames_processed < max_frames) {
            size_t bytes_read = 0;
            while (bytes_read < frame_bytes) {
                size_t n = fread((char*)pinned_buf + bytes_read, 1,
                                 frame_bytes - bytes_read, stdin);
                if (n == 0) goto done_stdin;
                bytes_read += n;
            }
            CUDA_CHECK(cudaMemcpy(frame_gpu.data, pinned_buf,
                                  frame_bytes, cudaMemcpyHostToDevice));
            droid.process_frame(frames_processed, frame_gpu.data);
            frames_processed++;
        }
        done_stdin:
        CUDA_CHECK(cudaFreeHost(pinned_buf));
    } else {
        std::vector<float> frame_data(frame_floats);
        for (int i = 0; i < max_frames; i++) {
            char frame_path[512];
            snprintf(frame_path, sizeof(frame_path), "%s/frame_%05d.bin", frame_dir, i);

            FILE* f = fopen(frame_path, "rb");
            if (!f) break;

            int fh, fw;
            fread(&fh, sizeof(int), 1, f);
            fread(&fw, sizeof(int), 1, f);
            fread(frame_data.data(), sizeof(float), frame_floats, f);
            fclose(f);

            frame_gpu.copyFrom(frame_data.data(), frame_floats);
            droid.process_frame(i, frame_gpu.data);
            frames_processed++;
        }
    }

    printf("\n");
    droid.print_timing(frames_processed);

    // Debug dump intermediate values
    if (debug_dump && droid.state.num_keyframes >= 2) {
        int hw = droid.h * droid.w;
        int nk = droid.state.num_keyframes;
        printf("\n=== Debug Dump ===\n");

        // Dump fmap stats for first 2 keyframes (FP16 → FP32 for display)
        for (int kf = 0; kf < std::min(2, nk); kf++) {
            // Convert FP16 fmaps to FP32 on GPU, then copy to host
            GpuBuf fmap_f32_tmp;
            fmap_f32_tmp.alloc(128 * hw);
            half_to_float(droid.state.fmaps.data + kf * 128 * hw, fmap_f32_tmp.data, 128 * hw);
            std::vector<float> fmap(128 * hw);
            CUDA_CHECK(cudaMemcpy(fmap.data(), fmap_f32_tmp.data,
                                  128 * hw * sizeof(float), cudaMemcpyDeviceToHost));
            fmap_f32_tmp.free();
            float fmin = 1e9, fmax = -1e9, fsum = 0;
            for (auto v : fmap) { fmin = std::min(fmin, v); fmax = std::max(fmax, v); fsum += v; }
            printf("KF%d fmap: mean=%.6f range=[%.4f, %.4f]\n", kf, fsum/(128*hw), fmin, fmax);

            // Save to file
            char path[256];
            snprintf(path, sizeof(path), "/tmp/cuda_fmap%d.bin", kf);
            FILE* f = fopen(path, "wb");
            int shape[3] = {128, droid.h, droid.w};
            fwrite(shape, sizeof(int), 3, f);
            fwrite(fmap.data(), sizeof(float), 128 * hw, f);
            fclose(f);
        }

        // Dump net and inp for kf 0
        {
            std::vector<float> net(128 * hw), inp(128 * hw);
            CUDA_CHECK(cudaMemcpy(net.data(), droid.state.nets.data,
                                  128 * hw * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(inp.data(), droid.state.inps.data,
                                  128 * hw * sizeof(float), cudaMemcpyDeviceToHost));
            float nmin=1e9, nmax=-1e9, nsum=0;
            for (auto v : net) { nmin=std::min(nmin,v); nmax=std::max(nmax,v); nsum+=v; }
            printf("KF0 net: mean=%.6f range=[%.4f, %.4f]\n", nsum/(128*hw), nmin, nmax);
            float imin=1e9, imax=-1e9, isum=0;
            for (auto v : inp) { imin=std::min(imin,v); imax=std::max(imax,v); isum+=v; }
            printf("KF0 inp: mean=%.6f range=[%.4f, %.4f]\n", isum/(128*hw), imin, imax);
        }

        // Build correlation between kf 0 and kf 1, dump stats
        droid.build_correlation(0, 1);
        std::vector<float> corr_vol(hw * hw);
        CUDA_CHECK(cudaMemcpy(corr_vol.data(), droid.corr_volume.data,
                              hw * hw * sizeof(float), cudaMemcpyDeviceToHost));
        float cmin=1e9, cmax=-1e9, csum=0;
        for (auto v : corr_vol) { cmin=std::min(cmin,v); cmax=std::max(cmax,v); csum+=v; }
        printf("Corr(0,1): mean=%.6f range=[%.4f, %.4f]\n", csum/(hw*hw), cmin, cmax);
        printf("  corr[0,0]=%.6f corr[0,1]=%.6f corr[1,0]=%.6f corr[100,100]=%.6f\n",
               corr_vol[0*hw+0], corr_vol[0*hw+1], corr_vol[1*hw+0], corr_vol[100*hw+100]);
        // Diagonal
        float diag_sum = 0;
        for (int i = 0; i < hw; i++) diag_sum += corr_vol[i*hw+i];
        printf("  diagonal mean=%.6f\n", diag_sum / hw);
    }

    // Backend optimization
    if (backend_iters1 > 0) {
        auto t_be = std::chrono::high_resolution_clock::now();
        droid.backend(backend_iters1, backend_radius);
        if (backend_iters2 > 0)
            droid.backend(backend_iters2, backend_radius);
        auto t_be_end = std::chrono::high_resolution_clock::now();
        float be_ms = std::chrono::duration<float, std::milli>(t_be_end - t_be).count();
        printf("Backend total: %.1f ms\n", be_ms);
    }

    // Output final poses
    int nk = droid.state.num_keyframes;
    printf("\n=== Final Poses (%d keyframes from %d frames) ===\n", nk, frames_processed);
    printf("Keyframe timestamps:");
    for (int i = 0; i < nk; i++) printf(" %d", droid.state.kf_timestamps[i]);
    printf("\n");

    int nf = nk;
    std::vector<float> poses(nf * 7);
    droid.state.poses.copyTo(poses.data(), nf * 7);

    // Convert world-to-camera -> camera-to-world if requested
    if (cam_to_world) {
        for (int i = 0; i < nf; i++) {
            float* p = &poses[i * 7];
            // SE3 inverse: q_inv = conjugate, t_inv = -R_inv * t
            float qx = p[3], qy = p[4], qz = p[5], qw = p[6];
            // q_inv = [-qx, -qy, -qz, qw]
            float qi[4] = {-qx, -qy, -qz, qw};
            // t_inv = -rotate(t, q_inv)
            float t[3] = {-p[0], -p[1], -p[2]};
            // Apply rotation: actSO3(qi, t, t_inv) on CPU
            float uv[3];
            uv[0] = 2.0f * (qi[1]*t[2] - qi[2]*t[1]);
            uv[1] = 2.0f * (qi[2]*t[0] - qi[0]*t[2]);
            uv[2] = 2.0f * (qi[0]*t[1] - qi[1]*t[0]);
            p[0] = t[0] + qi[3]*uv[0] + (qi[1]*uv[2] - qi[2]*uv[1]);
            p[1] = t[1] + qi[3]*uv[1] + (qi[2]*uv[0] - qi[0]*uv[2]);
            p[2] = t[2] + qi[3]*uv[2] + (qi[0]*uv[1] - qi[1]*uv[0]);
            p[3] = qi[0]; p[4] = qi[1]; p[5] = qi[2]; p[6] = qi[3];
        }
    }

    for (int i = 0; i < std::min(10, nf); i++) {
        printf("Frame %3d: t=[%7.3f %7.3f %7.3f] q=[%6.3f %6.3f %6.3f %6.3f]\n",
               i, poses[i*7], poses[i*7+1], poses[i*7+2],
               poses[i*7+3], poses[i*7+4], poses[i*7+5], poses[i*7+6]);
    }

    // Save poses and keyframe timestamps to binary file if requested
    // Format: [num_keyframes:i32] [timestamps:i32*nk] [poses:f32*nk*7]
    if (pose_output) {
        FILE* f = fopen(pose_output, "wb");
        if (f) {
            fwrite(&nk, sizeof(int), 1, f);
            // Write keyframe timestamps
            std::vector<int> ts(droid.state.kf_timestamps.begin(), droid.state.kf_timestamps.end());
            fwrite(ts.data(), sizeof(int), nk, f);
            fwrite(poses.data(), sizeof(float), nk * 7, f);
            fclose(f);
            printf("Saved %d keyframe poses to %s\n", nk, pose_output);
        }
    }

    droid.destroy();
    return 0;
}
