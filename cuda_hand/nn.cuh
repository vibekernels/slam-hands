#pragma once
// Neural network primitives for CUDA hand pose pipeline
// cuDNN for convolutions, cuBLAS for GEMM, custom kernels for activations/norms

#include <cuda_fp16.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// ============ Error checking macros ============

#define CUDNN_CHECK(expr) do { \
    cudnnStatus_t status = (expr); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(status)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t status = (expr); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============ GPU Buffers ============

struct GpuBuf {
    float* data = nullptr;
    size_t count = 0;

    void alloc(size_t n) {
        if (data && count >= n) return;
        free();
        count = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(float)));
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; count = 0; }
    }

    void zero() { if (data) CUDA_CHECK(cudaMemset(data, 0, count * sizeof(float))); }

    void copyFrom(const float* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(data, src, n * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copyTo(float* dst, size_t n) const {
        CUDA_CHECK(cudaMemcpy(dst, data, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    ~GpuBuf() { free(); }
};

struct GpuHalfBuf {
    __half* data = nullptr;
    size_t count = 0;

    void alloc(size_t n) {
        if (data && count >= n) return;
        free();
        count = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(__half)));
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; count = 0; }
    }

    void zero() { if (data) CUDA_CHECK(cudaMemset(data, 0, count * sizeof(__half))); }

    ~GpuHalfBuf() { free(); }
};

struct GpuIntBuf {
    int* data = nullptr;
    size_t count = 0;

    void alloc(size_t n) {
        if (data && count >= n) return;
        free();
        count = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(int)));
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; count = 0; }
    }

    void copyFrom(const int* src, size_t n) {
        CUDA_CHECK(cudaMemcpy(data, src, n * sizeof(int), cudaMemcpyHostToDevice));
    }

    ~GpuIntBuf() { free(); }
};

// ============ Weight loading ============

inline bool load_tensor(const char* path, std::vector<float>& data, std::vector<int>& shape) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    int ndim;
    if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    shape.resize(ndim);
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (fread(&shape[i], sizeof(int), 1, f) != 1) { fclose(f); return false; }
        total *= shape[i];
    }
    data.resize(total);
    if (fread(data.data(), sizeof(float), total, f) != total) { fclose(f); return false; }
    fclose(f);
    return true;
}

inline bool load_int_tensor(const char* path, std::vector<int>& data, std::vector<int>& shape) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    int ndim;
    if (fread(&ndim, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    shape.resize(ndim);
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (fread(&shape[i], sizeof(int), 1, f) != 1) { fclose(f); return false; }
        total *= shape[i];
    }
    data.resize(total);
    if (fread(data.data(), sizeof(int), total, f) != (size_t)total) { fclose(f); return false; }
    fclose(f);
    return true;
}

// ============ cuDNN Convolution Layer ============

struct ConvLayer {
    cudnnFilterDescriptor_t filterDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize = 0;

    GpuBuf weight, bias;
    int Ci, Co, kH, kW, stride, pad;

    void init(cudnnHandle_t cudnn, int batch,
              int ci, int co, int kh, int kw, int s, int p,
              int inH, int inW,
              const float* w_data, const float* b_data) {
        Ci = ci; Co = co; kH = kh; kW = kw; stride = s; pad = p;

        weight.alloc(co * ci * kh * kw);
        weight.copyFrom(w_data, co * ci * kh * kw);
        bias.alloc(co);
        bias.copyFrom(b_data, co);

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW, co, ci, kh, kw));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, p, p, s, s, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

        cudnnTensorDescriptor_t inDesc, outDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, ci, inH, inW));
        int outH, outW, outN, outC;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inDesc, filterDesc,
            &outN, &outC, &outH, &outW));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, co, outH, outW));

        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            cudnn, inDesc, filterDesc, convDesc, outDesc,
            8, &returnedAlgoCount, perf));
        algo = perf[0].algo;
        workspaceSize = perf[0].memory;

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }

    void forward(cudnnHandle_t cudnn, float* input, float* output, float* workspace,
                 int batch, int inH, int inW) {
        cudnnTensorDescriptor_t inDesc, outDesc, biasDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, Ci, inH, inW));
        int outH = (inH + 2*pad - kH) / stride + 1;
        int outW = (inW + 2*pad - kW) / stride + 1;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, Co, outH, outW));

        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inDesc, input,
            filterDesc, weight.data, convDesc, algo, workspace, workspaceSize,
            &beta, outDesc, output));

        // Add bias
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, Co, 1, 1));
        alpha = 1.0f; beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, biasDesc, bias.data,
            &beta, outDesc, output));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }
};

// ============ cuDNN ConvTranspose Layer (uses backward-data for transposed conv) ============

struct ConvTransposeLayer {
    cudnnFilterDescriptor_t filterDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspaceSize = 0;

    GpuBuf weight, bias;
    int Ci, Co, kH, kW, stride, pad;

    void init(cudnnHandle_t cudnn, int batch,
              int ci, int co, int kh, int kw, int s, int p,
              int inH, int inW,
              const float* w_data, const float* b_data) {
        Ci = ci; Co = co; kH = kh; kW = kw; stride = s; pad = p;

        // ConvTranspose2d weight: [Ci, Co, kH, kW]
        weight.alloc(ci * co * kh * kw);
        weight.copyFrom(w_data, ci * co * kh * kw);
        bias.alloc(co);
        bias.copyFrom(b_data, co);

        // For cudnnConvolutionBackwardData, filter is [Ci, Co, kH, kW]
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW, ci, co, kh, kw));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, p, p, s, s, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

        // Input (dy) descriptor: [batch, ci, inH, inW]
        // Output (dx) descriptor: [batch, co, outH, outW] where outH = (inH-1)*s - 2*p + kh
        int outH = (inH - 1) * s - 2 * p + kh;
        int outW = (inW - 1) * s - 2 * p + kw;

        cudnnTensorDescriptor_t dyDesc, dxDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, ci, inH, inW));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, co, outH, outW));

        int returnedAlgoCount;
        cudnnConvolutionBwdDataAlgoPerf_t perf[8];
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            cudnn, filterDesc, dyDesc, convDesc, dxDesc,
            8, &returnedAlgoCount, perf));
        algo = perf[0].algo;
        workspaceSize = perf[0].memory;

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    }

    void forward(cudnnHandle_t cudnn, float* input, float* output, float* workspace,
                 int batch, int inH, int inW) {
        int outH = (inH - 1) * stride - 2 * pad + kH;
        int outW = (inW - 1) * stride - 2 * pad + kW;

        cudnnTensorDescriptor_t dyDesc, dxDesc, biasDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, Ci, inH, inW));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, batch, Co, outH, outW));

        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn, &alpha,
            filterDesc, weight.data, dyDesc, input, convDesc,
            algo, workspace, workspaceSize, &beta, dxDesc, output));

        // Add bias
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, Co, 1, 1));
        alpha = 1.0f; beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, biasDesc, bias.data,
            &beta, dxDesc, output));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    }
};

// ============ Activation Kernels ============

__global__ void silu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = fmaxf(data[idx], 0.0f);
}

__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

__global__ void gelu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // But PyTorch uses exact GELU: x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        data[idx] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

inline void silu_inplace(float* data, int n) {
    silu_kernel<<<(n+255)/256, 256>>>(data, n);
}
inline void relu_inplace(float* data, int n) {
    relu_kernel<<<(n+255)/256, 256>>>(data, n);
}
inline void sigmoid_inplace(float* data, int n) {
    sigmoid_kernel<<<(n+255)/256, 256>>>(data, n);
}
inline void gelu_inplace(float* data, int n) {
    gelu_kernel<<<(n+255)/256, 256>>>(data, n);
}

// ============ FP16 Activation Kernels ============

__global__ void gelu_half_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)));
    }
}

inline void gelu_half_inplace(__half* data, int n) {
    gelu_half_kernel<<<(n+255)/256, 256>>>(data, n);
}

// ============ LayerNorm (FP16 in/out, FP32 compute) ============

__global__ void layernorm_half_kernel(const __half* input, __half* output,
                                       const float* gamma, const float* beta,
                                       int N, int C) {
    // One block per row (token)
    int row = blockIdx.x;
    if (row >= N) return;

    const __half* in_row = input + (size_t)row * C;
    __half* out_row = output + (size_t)row * C;

    extern __shared__ float sdata[];

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        sum += __half2float(in_row[i]);
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / C;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = __half2float(in_row[i]) - mean;
        var_sum += v * v;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / C + 1e-6f);
    __syncthreads();

    // Normalize with affine
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = (__half2float(in_row[i]) - mean) * inv_std;
        out_row[i] = __float2half(val * gamma[i] + beta[i]);
    }
}

inline void layernorm_half(const __half* input, __half* output,
                            const float* gamma, const float* beta,
                            int N, int C) {
    int threads = min(256, C);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    layernorm_half_kernel<<<N, threads, threads * sizeof(float)>>>(
        input, output, gamma, beta, N, C);
}

// ============ Element-wise operations ============

__global__ void add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

__global__ void add_half_kernel(__half* a, const __half* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

inline void add_inplace_half(__half* a, const __half* b, int n) {
    add_half_kernel<<<(n+255)/256, 256>>>(a, b, n);
}

// ============ MaxPool2d ============

__global__ void maxpool2d_kernel(const float* input, float* output,
                                  int N, int C, int H, int W, int kH, int sH, int pH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pH - kH) / sH + 1;
    int total = N * C * oH * oW;
    if (idx >= total) return;

    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int c = (idx / (oH * oW)) % C;
    int n = idx / (C * oH * oW);

    float maxval = -1e30f;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kH; kw++) {
            int ih = oh * sH - pH + kh;
            int iw = ow * sH - pH + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float val = input[((n * C + c) * H + ih) * W + iw];
                maxval = fmaxf(maxval, val);
            }
        }
    }
    output[idx] = maxval;
}

inline void maxpool2d(float* input, float* output, int N, int C, int H, int W,
                      int kH, int sH, int pH) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pH - kH) / sH + 1;
    int total = N * C * oH * oW;
    maxpool2d_kernel<<<(total+255)/256, 256>>>(input, output, N, C, H, W, kH, sH, pH);
}

// ============ Upsample nearest 2x ============

__global__ void upsample2x_kernel(const float* input, float* output,
                                    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oH = H * 2, oW = W * 2;
    int total = N * C * oH * oW;
    if (idx >= total) return;

    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int c = (idx / (oH * oW)) % C;
    int n = idx / (C * oH * oW);

    int ih = oh / 2, iw = ow / 2;
    output[idx] = input[((n * C + c) * H + ih) * W + iw];
}

inline void upsample_nearest_2x(const float* input, float* output, int N, int C, int H, int W) {
    int total = N * C * (H*2) * (W*2);
    upsample2x_kernel<<<(total+255)/256, 256>>>(input, output, N, C, H, W);
}

// ============ Concatenation along channel dim ============

__global__ void concat2_kernel(float* output, const float* a, const float* b,
                               int N, int C1, int C2, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (C1 + C2) * HW;
    if (idx >= total) return;
    int hw = idx % HW;
    int c = (idx / HW) % (C1 + C2);
    int n = idx / (HW * (C1 + C2));
    if (c < C1) output[idx] = a[n * C1 * HW + c * HW + hw];
    else output[idx] = b[n * C2 * HW + (c - C1) * HW + hw];
}

__global__ void concat4_kernel(float* output,
                               const float* a, const float* b, const float* c, const float* d,
                               int N, int C1, int C2, int C3, int C4, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int CT = C1 + C2 + C3 + C4;
    int total = N * CT * HW;
    if (idx >= total) return;
    int hw = idx % HW;
    int ch = (idx / HW) % CT;
    int n = idx / (HW * CT);
    if (ch < C1) output[idx] = a[n * C1 * HW + ch * HW + hw];
    else if (ch < C1 + C2) output[idx] = b[n * C2 * HW + (ch - C1) * HW + hw];
    else if (ch < C1 + C2 + C3) output[idx] = c[n * C3 * HW + (ch - C1 - C2) * HW + hw];
    else output[idx] = d[n * C4 * HW + (ch - C1 - C2 - C3) * HW + hw];
}

// ============ Channel slice ============

__global__ void slice_channels_kernel(float* output, const float* input,
                                      int N, int C_total, int C_start, int C_out, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * HW;
    if (idx >= total) return;
    int hw = idx % HW;
    int c = (idx / HW) % C_out;
    int n = idx / (HW * C_out);
    output[idx] = input[n * C_total * HW + (C_start + c) * HW + hw];
}

inline void slice_channels(float* output, const float* input,
                           int N, int C_total, int C_start, int C_out, int HW) {
    int total = N * C_out * HW;
    slice_channels_kernel<<<(total+255)/256, 256>>>(output, input, N, C_total, C_start, C_out, HW);
}

// ============ FP32 <-> FP16 conversion ============

__global__ void float_to_half_kernel(const float* input, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __float2half(input[idx]);
}

__global__ void half_to_float_kernel(const __half* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __half2float(input[idx]);
}

inline void float_to_half(const float* input, __half* output, int n) {
    float_to_half_kernel<<<(n+255)/256, 256>>>(input, output, n);
}

inline void half_to_float(const __half* input, float* output, int n) {
    half_to_float_kernel<<<(n+255)/256, 256>>>(input, output, n);
}

// ============ Bilinear grid sample (for RefineNet vertex feature sampling) ============

__global__ void grid_sample_kernel(const float* feat, const float* grid_xy,
                                    float* output,
                                    int B, int C, int H, int W, int N_pts) {
    // feat: [B, C, H, W], grid_xy: [B, N_pts, 2] (x,y in pixel coords)
    // output: [B, C, N_pts]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N_pts;
    if (idx >= total) return;

    int pt = idx % N_pts;
    int b = idx / N_pts;

    // Normalize pixel coords to [-1,1] for grid_sample align_corners=True
    float px = grid_xy[(b * N_pts + pt) * 2 + 0];
    float py = grid_xy[(b * N_pts + pt) * 2 + 1];
    float nx = px / (W - 1) * 2.0f - 1.0f;
    float ny = py / (H - 1) * 2.0f - 1.0f;

    // Unnormalize back to pixel space for bilinear interpolation
    float ix = (nx + 1.0f) * 0.5f * (W - 1);
    float iy = (ny + 1.0f) * 0.5f * (H - 1);

    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    float wx = ix - ix0;
    float wy = iy - iy0;

    for (int c = 0; c < C; c++) {
        float v00 = 0, v01 = 0, v10 = 0, v11 = 0;
        const float* feat_bc = feat + (b * C + c) * H * W;
        if (ix0 >= 0 && ix0 < W && iy0 >= 0 && iy0 < H) v00 = feat_bc[iy0 * W + ix0];
        if (ix1 >= 0 && ix1 < W && iy0 >= 0 && iy0 < H) v01 = feat_bc[iy0 * W + ix1];
        if (ix0 >= 0 && ix0 < W && iy1 >= 0 && iy1 < H) v10 = feat_bc[iy1 * W + ix0];
        if (ix1 >= 0 && ix1 < W && iy1 >= 0 && iy1 < H) v11 = feat_bc[iy1 * W + ix1];

        float val = (1-wy)*(1-wx)*v00 + (1-wy)*wx*v01 + wy*(1-wx)*v10 + wy*wx*v11;
        output[(b * C + c) * N_pts + pt] = val;
    }
}

// Max-pool over the points dimension: [B, C, N_pts] -> [B, C]
__global__ void max_pool_points_kernel(const float* input, float* output,
                                        int B, int C, int N_pts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;
    int c = idx % C;
    int b = idx / C;

    float maxval = -1e30f;
    const float* ptr = input + (b * C + c) * N_pts;
    for (int i = 0; i < N_pts; i++) {
        maxval = fmaxf(maxval, ptr[i]);
    }
    output[b * C + c] = maxval;
}

// ============ Softmax for attention (FP16 in/out, FP32 accumulate) ============

__global__ void softmax_half_kernel(__half* data, int rows, int cols) {
    // One block per row
    int row = blockIdx.x;
    if (row >= rows) return;

    __half* row_data = data + (size_t)row * cols;
    extern __shared__ float sdata[];

    // Find max
    float maxval = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        maxval = fmaxf(maxval, __half2float(row_data[i]));
    sdata[threadIdx.x] = maxval;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(__half2float(row_data[i]) - row_max);
        sum += v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // Write normalized values
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(__half2float(row_data[i]) - row_max) * inv_sum;
        row_data[i] = __float2half(v);
    }
}

inline void softmax_half(__half* data, int rows, int cols) {
    int threads = min(256, cols);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    softmax_half_kernel<<<rows, threads, threads * sizeof(float)>>>(data, rows, cols);
}

// ============ Scale kernel for attention ============

__global__ void scale_half_kernel(__half* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = __float2half(__half2float(data[idx]) * scale);
}

inline void scale_half(__half* data, float scale, int n) {
    scale_half_kernel<<<(n+255)/256, 256>>>(data, scale, n);
}
