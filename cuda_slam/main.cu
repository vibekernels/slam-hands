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

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

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

        // output = conv2(temp)
        conv2.forward(cudnn, temp, output, workspace, batch, outH, outW);
        if (has_norm) instance_norm(output, batch, out_channels, outH, outW);

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

        // Concatenate inputs: [inp(128), corr(128), flow(64)] = 320 channels
        // Then concatenate with net: [net(128), inp_cat(320)] = 448 channels
        // For simplicity, build the full 448-channel tensor directly
        concat3_kernel<<<(batch*320*HW+255)/256, 256>>>(
            cat_buf.data + NC*HW,  // offset past first 128 channels
            inp, corr, flow,
            batch, 128, 128, 64, HW);

        // Copy net to first 128 channels of cat_buf
        CUDA_CHECK(cudaMemcpy(cat_buf.data, net, NC * HW * sizeof(float),
                              cudaMemcpyDeviceToDevice));

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

        // gru_input = cat(r*net, inp_cat)
        // r*net -> rnet_buf
        CUDA_CHECK(cudaMemcpy(rnet_buf.data, net, NC * HW * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        mul_kernel<<<(NC*HW+255)/256, 256>>>(rnet_buf.data, r_buf.data, NC * HW);
        // Build gru_input: [r*net(128), inp(128), corr(128), flow(64)] = 448
        CUDA_CHECK(cudaMemcpy(gru_inp_buf.data, rnet_buf.data, NC * HW * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gru_inp_buf.data + NC*HW, cat_buf.data + NC*HW,
                              batch * 320 * HW * sizeof(float), cudaMemcpyDeviceToDevice));

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
    int num_frames;   // Total frames added

    // Per-keyframe state (on GPU)
    GpuBuf poses;      // [MAX_KF, 7]
    GpuBuf disps;      // [MAX_KF, h, w]
    GpuBuf fmaps;      // [MAX_KF, 128, h, w]
    GpuBuf nets;       // [MAX_KF, 128, h, w]
    GpuBuf inps;       // [MAX_KF, 128, h, w]
    GpuBuf intrinsics; // [4]

    // Edge state
    std::vector<int> ii_host, jj_host;
    GpuIntBuf ii_gpu, jj_gpu;

    // Per-edge target and weight
    GpuBuf target;     // [MAX_EDGES, 2, h, w]
    GpuBuf weight;     // [MAX_EDGES, 2, h, w]
    GpuBuf edge_nets;  // [MAX_EDGES, 128, h, w]
    static const int MAX_EDGES = 2048;

    void init(int fullH, int fullW, float fx, float fy, float cx, float cy) {
        H = fullH; W = fullW;
        h = fullH / 8; w = fullW / 8;
        num_frames = 0;

        int hw = h * w;
        poses.alloc(MAX_KEYFRAMES * 7);
        disps.alloc(MAX_KEYFRAMES * hw);
        fmaps.alloc(MAX_KEYFRAMES * 128 * hw);
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

        // Initialize disparities to 0.5
        std::vector<float> init_disps(MAX_KEYFRAMES * hw, 0.5f);
        disps.copyFrom(init_disps.data(), MAX_KEYFRAMES * hw);

        // Edge buffers
        target.alloc(MAX_EDGES * 2 * hw);
        weight.alloc(MAX_EDGES * 2 * hw);
        edge_nets.alloc(MAX_EDGES * 128 * hw);
        ii_gpu.alloc(MAX_EDGES);
        jj_gpu.alloc(MAX_EDGES);
    }

    void add_frame(float* fmap, float* net, float* inp) {
        int hw = h * w;
        int idx = num_frames;
        CUDA_CHECK(cudaMemcpy(fmaps.data + idx * 128 * hw, fmap,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(nets.data + idx * 128 * hw, net,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(inps.data + idx * 128 * hw, inp,
                              128 * hw * sizeof(float), cudaMemcpyDeviceToDevice));

        // Copy pose from previous frame
        if (idx > 0) {
            CUDA_CHECK(cudaMemcpy(poses.data + idx * 7, poses.data + (idx-1) * 7,
                                  7 * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        num_frames++;
    }

    // Add edges between nearby keyframes
    void add_neighborhood_edges(int start, int end, int radius = 3) {
        for (int i = start; i < end; i++) {
            for (int j = std::max(start, i - radius); j < std::min(end, i + radius + 1); j++) {
                if (i != j) {
                    ii_host.push_back(i);
                    jj_host.push_back(j);
                }
            }
        }
        sync_edges_to_gpu();
    }

    void sync_edges_to_gpu() {
        int n = ii_host.size();
        if (n == 0) return;
        ii_gpu.alloc(n);
        jj_gpu.alloc(n);
        ii_gpu.copyFrom(ii_host.data(), n);
        jj_gpu.copyFrom(jj_host.data(), n);
    }

    int num_edges() const { return ii_host.size(); }
};

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

    BasicEncoder fnet, cnet;
    UpdateModule update;
    BundleAdjustment ba;
    DroidState state;

    static const int MAX_BATCH_EDGES = 48;

    // Working buffers (3 buffers needed: input, output, temp)
    GpuBuf buf_a, buf_b, buf_c, workspace;
    GpuBuf preproc_buf;  // preprocessed image

    // Correlation buffers (per-edge, reused in loop)
    GpuBuf corr_volume;  // [H*W, H*W] for one edge
    GpuBuf corr_pyramid[4];
    GpuBuf coords_scaled; // [2, h, w]

    // Batched buffers for edge processing
    GpuBuf batch_corr;    // [MAX_BATCH, 196, h, w]
    GpuBuf batch_coords;  // [MAX_BATCH, 2, h, w] - channel-first format
    GpuBuf coords_hw2;    // [h*w*2] temp for projmap output (interleaved)
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

        printf("Initializing encoders...\n");
        fnet.init(cudnn, ws, "fnet", 1, H, W, 128, true);
        cnet.init(cudnn, ws, "cnet", 1, H, W, 256, false);

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

        // Correlation buffers (per-edge, reused in loop)
        corr_volume.alloc(hw * hw);
        // Pyramid levels 1-3 (level 0 uses corr_volume directly)
        for (int l = 1; l < 4; l++) {
            int ph = h >> l, pw = w >> l;
            corr_pyramid[l].alloc(hw * ph * pw);
        }
        coords_scaled.alloc(2 * hw);

        // Batched edge buffers
        batch_corr.alloc(MAX_BATCH_EDGES * 196 * hw);
        batch_coords.alloc(MAX_BATCH_EDGES * 2 * hw);
        coords_hw2.alloc(hw * 2);  // temp for projmap HW2 output
        batch_motion.alloc(MAX_BATCH_EDGES * 4 * hw);
        batch_nets.alloc(MAX_BATCH_EDGES * 128 * hw);
        batch_inps.alloc(MAX_BATCH_EDGES * 128 * hw);
        batch_delta.alloc(MAX_BATCH_EDGES * 3 * hw);
        batch_weight.alloc(MAX_BATCH_EDGES * 3 * hw);
        batch_eta.alloc(MAX_BATCH_EDGES * 1 * hw);

        alloc_encoder_bufs();
        printf("CudaDroid initialized: %dx%d -> %dx%d\n", H, W, h, w);
    }

    // Pre-allocated encoder output buffers
    GpuBuf enc_fmap, enc_cmap, enc_net, enc_inp;

    void alloc_encoder_bufs() {
        int hw = h * w;
        enc_fmap.alloc(128 * hw);
        enc_cmap.alloc(256 * hw);
        enc_net.alloc(128 * hw);
        enc_inp.alloc(128 * hw);
    }

    // Run feature encoder on one frame
    void encode_frame(float* bgr_hwc) {
        t_encode.begin();

        // Preprocess: BGR HWC -> RGB NCHW normalized
        preprocess_image_kernel<<<(H*W+255)/256, 256>>>(
            bgr_hwc, preproc_buf.data, H, W);

        int hw = h * w;

        // Run fnet: [1, 3, H, W] -> [1, 128, h, w]
        fnet.forward(cudnn, preproc_buf.data, enc_fmap.data,
                     buf_a.data, buf_b.data, buf_c.data, workspace.data, 1, H, W);

        // Run cnet: [1, 3, H, W] -> [1, 256, h, w]
        cnet.forward(cudnn, preproc_buf.data, enc_cmap.data,
                     buf_a.data, buf_b.data, buf_c.data, workspace.data, 1, H, W);

        // Split cnet output: [256, h, w] -> net[128, h, w] + inp[128, h, w]
        slice_channels(enc_net.data, enc_cmap.data, 1, 256, 0, 128, hw);
        slice_channels(enc_inp.data, enc_cmap.data, 1, 256, 128, 128, hw);

        // Apply activations: tanh(net), relu(inp)
        tanh_inplace(enc_net.data, 128 * hw);
        relu_inplace(enc_inp.data, 128 * hw);

        // Store in state
        state.add_frame(enc_fmap.data, enc_net.data, enc_inp.data);

        t_encode.end();
    }

    // Build correlation volume between frames i and j
    void build_correlation(int i, int j) {
        int hw = h * w;
        float* f1 = state.fmaps.data + i * 128 * hw;
        float* f2 = state.fmaps.data + j * 128 * hw;

        // All-pairs correlation: fmap1^T @ fmap2 / C
        // fmap1: [128, hw], fmap2: [128, hw]
        // result: [hw, hw]
        float alpha = 1.0f / 128.0f;
        float beta = 0.0f;

        // In col-major (cuBLAS): f1 is [hw, 128], f2 is [hw, 128]
        // result = f1 @ f2^T = [hw, hw]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            hw, hw, 128,
            &alpha,
            f1, hw,
            f2, hw,
            &beta,
            corr_volume.data, hw));

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

    // Process one frame - BATCHED edge processing
    void process_frame(float* bgr_hwc_gpu) {
        t_total.begin();

        // 1. Encode features
        encode_frame(bgr_hwc_gpu);

        int n = state.num_frames;
        if (n < 2) { t_total.end(); return; }

        int hw = h * w;

        // 2. Add edges to nearby frames (limited window like DROID-SLAM motion filter)
        // Only add edges for the newest frame to recent frames
        int start = std::max(0, n - 3);
        state.ii_host.clear();
        state.jj_host.clear();
        for (int j = start; j < n - 1; j++) {
            state.ii_host.push_back(n - 1);
            state.jj_host.push_back(j);
            state.jj_host.push_back(n - 1);
            state.ii_host.push_back(j);
        }
        state.sync_edges_to_gpu();

        int num_edges = std::min(state.num_edges(), MAX_BATCH_EDGES);
        if (num_edges == 0) { t_total.end(); return; }

        // 3. Build correlations ONCE for all edges (correlation volumes don't change between GRU iterations)
        t_corr.begin();
        for (int e = 0; e < num_edges; e++) {
            int ii_val = state.ii_host[e];
            int jj_val = state.jj_host[e];

            // Reproject: get initial coordinates for this edge (outputs [H,W,2] interleaved)
            projmap_kernel<<<1, 256>>>(
                state.poses.data, state.disps.data, state.intrinsics.data,
                (const int*)state.ii_gpu.data + e, (const int*)state.jj_gpu.data + e,
                coords_hw2.data, buf_a.data, 1, h, w);

            // Convert from [H,W,2] to [2,H,W] channel-first format
            float* edge_coords = batch_coords.data + e * 2 * hw;
            coords_hw2_to_2hw<<<(hw+255)/256, 256>>>(coords_hw2.data, edge_coords, hw);

            // Build correlation volume and sample
            build_correlation(ii_val, jj_val);
            float* edge_corr = batch_corr.data + e * 196 * hw;
            sample_correlation(edge_coords, edge_corr);
        }
        t_corr.end();

        // 4. GRU update iterations (BATCHED)
        t_update.begin();

        // Initialize edge nets from keyframe nets (gather by ii index)
        {
            int total = num_edges * 128 * hw;
            gather_features_kernel<<<(total+255)/256, 256>>>(
                batch_nets.data, state.nets.data,
                state.ii_gpu.data, num_edges, 128, hw);
        }

        // Gather inp features for all edges (constant across iterations)
        {
            int total = num_edges * 128 * hw;
            gather_features_kernel<<<(total+255)/256, 256>>>(
                batch_inps.data, state.inps.data,
                state.ii_gpu.data, num_edges, 128, hw);
        }

        // Zero motion for first iteration
        CUDA_CHECK(cudaMemset(batch_motion.data, 0, num_edges * 4 * hw * sizeof(float)));

        for (int itr = 0; itr < 2; itr++) {

            // Run UpdateModule ONCE with batch=num_edges
            update.forward(cudnn,
                batch_corr.data, batch_motion.data,
                batch_nets.data, batch_inps.data,
                batch_delta.data, batch_weight.data, batch_eta.data,
                buf_a.data, buf_b.data, workspace.data,
                num_edges, h, w);

            // Compute targets: target = coords + delta, extract weights
            {
                int total = num_edges * 2 * hw;
                compute_target_kernel<<<(total+255)/256, 256>>>(
                    state.target.data, state.weight.data,
                    batch_coords.data, batch_delta.data, batch_weight.data,
                    num_edges, hw);
            }
        }
        t_update.end();

        // 5. Bundle adjustment with computed targets and weights
        t_ba.begin();
        if (num_edges > 0) {
            int t0 = 1;  // Don't optimize first pose
            int t1 = n;
            ba.iterate(state.poses.data, state.disps.data, state.intrinsics.data,
                       state.target.data, state.weight.data,
                       state.disps.data,  // eta placeholder
                       state.ii_gpu.data, state.jj_gpu.data,
                       state.ii_host.data(), state.jj_host.data(),
                       num_edges, t0, t1,
                       1e-4f, 0.1f, true);
        }
        t_ba.end();

        t_total.end();
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
    int max_frames = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) weight_dir = argv[++i];
        else if (strcmp(argv[i], "--frames") == 0 && i+1 < argc) frame_dir = argv[++i];
        else if (strcmp(argv[i], "--calib") == 0 && i+1 < argc) calib_file = argv[++i];
        else if (strcmp(argv[i], "--max-frames") == 0 && i+1 < argc) max_frames = atoi(argv[++i]);
    }

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

    // Read frame metadata
    int frameH, frameW, totalFrames, stride;
    {
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

    // Allocate frame buffer on GPU
    GpuBuf frame_gpu;
    frame_gpu.alloc(frameH * frameW * 3);

    // Process frames
    int frames_processed = 0;
    for (int i = 0; i < std::min(max_frames, totalFrames); i++) {
        char frame_path[512];
        snprintf(frame_path, sizeof(frame_path), "%s/frame_%05d.bin", frame_dir, i);

        FILE* f = fopen(frame_path, "rb");
        if (!f) { printf("Cannot open frame %d, stopping.\n", i); break; }

        int fh, fw;
        fread(&fh, sizeof(int), 1, f);
        fread(&fw, sizeof(int), 1, f);

        std::vector<float> frame_data(fh * fw * 3);
        fread(frame_data.data(), sizeof(float), fh * fw * 3, f);
        fclose(f);

        frame_gpu.copyFrom(frame_data.data(), fh * fw * 3);
        droid.process_frame(frame_gpu.data);
        frames_processed++;

        if (frames_processed % 10 == 0) {
            printf("Processed %d/%d frames\r", frames_processed, max_frames);
            fflush(stdout);
        }
    }

    printf("\n");
    droid.print_timing(frames_processed);

    // Output final poses
    printf("\n=== Final Poses ===\n");
    std::vector<float> poses(droid.state.num_frames * 7);
    droid.state.poses.copyTo(poses.data(), droid.state.num_frames * 7);
    for (int i = 0; i < std::min(10, droid.state.num_frames); i++) {
        printf("Frame %3d: t=[%7.3f %7.3f %7.3f] q=[%6.3f %6.3f %6.3f %6.3f]\n",
               i, poses[i*7], poses[i*7+1], poses[i*7+2],
               poses[i*7+3], poses[i*7+4], poses[i*7+5], poses[i*7+6]);
    }

    droid.destroy();
    return 0;
}
