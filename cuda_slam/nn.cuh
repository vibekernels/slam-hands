#pragma once
// Neural network primitives for CUDA DROID-SLAM
// Uses cuDNN for convolutions, custom kernels for InstanceNorm and activations

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

// ============ GPU Buffer ============

struct GpuBuf {
    float* data = nullptr;
    size_t count = 0;  // number of floats

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

// Integer buffer for indices
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
    fread(&ndim, sizeof(int), 1, f);
    shape.resize(ndim);
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        fread(&shape[i], sizeof(int), 1, f);
        total *= shape[i];
    }
    data.resize(total);
    fread(data.data(), sizeof(float), total, f);
    fclose(f);
    return true;
}

// ============ cuDNN Convolution Layer ============

struct ConvLayer {
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize;

    GpuBuf weight, bias;
    int Ci, Co, kH, kW, stride, pad;
    bool has_bias;

    void init(cudnnHandle_t cudnn, int batch,
              int ci, int co, int kh, int kw, int s, int p,
              int inH, int inW,
              const float* w_data, const float* b_data) {
        Ci = ci; Co = co; kH = kh; kW = kw; stride = s; pad = p;
        has_bias = (b_data != nullptr);

        weight.alloc(co * ci * kh * kw);
        weight.copyFrom(w_data, co * ci * kh * kw);
        if (has_bias) {
            bias.alloc(co);
            bias.copyFrom(b_data, co);
        }

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW, co, ci, kh, kw));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, p, p, s, s, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // Find best supported algorithm
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

        // Try algorithms in order of preference
        cudnnConvolutionFwdAlgo_t algos[] = {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        };
        bool found = false;
        for (auto a : algos) {
            cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, inDesc, filterDesc, convDesc, outDesc, a, &workspaceSize);
            if (st == CUDNN_STATUS_SUCCESS) {
                algo = a;
                found = true;
                break;
            }
        }
        if (!found) {
            // Last resort: use the get algorithm heuristic
            int returnedAlgoCount;
            cudnnConvolutionFwdAlgoPerf_t perf[1];
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn, inDesc, filterDesc, convDesc, outDesc,
                1, &returnedAlgoCount, perf));
            algo = perf[0].algo;
            workspaceSize = perf[0].memory;
        }

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }

    // Forward: output = conv(input) + bias
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
        // Try the selected algorithm first, fall back to others
        cudnnStatus_t fwd_status = cudnnConvolutionForward(cudnn, &alpha, inDesc, input,
            filterDesc, weight.data, convDesc, algo, workspace, workspaceSize,
            &beta, outDesc, output);
        if (fwd_status != CUDNN_STATUS_SUCCESS) {
            // Try all algorithms
            cudnnConvolutionFwdAlgo_t try_algos[] = {
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            };
            bool ok = false;
            for (auto a : try_algos) {
                size_t ws = 0;
                if (cudnnGetConvolutionForwardWorkspaceSize(cudnn, inDesc, filterDesc,
                        convDesc, outDesc, a, &ws) != CUDNN_STATUS_SUCCESS) continue;
                if (ws > 64*1024*1024) continue;  // skip if needs too much workspace
                fwd_status = cudnnConvolutionForward(cudnn, &alpha, inDesc, input,
                    filterDesc, weight.data, convDesc, a, workspace, ws,
                    &beta, outDesc, output);
                if (fwd_status == CUDNN_STATUS_SUCCESS) {
                    algo = a; workspaceSize = ws; ok = true; break;
                }
            }
            if (!ok) {
                fprintf(stderr, "Conv forward FAILED (all algos): in=[%d,%d,%d,%d] k=[%d,%d,%d,%d] s=%d p=%d\n",
                        batch, Ci, inH, inW, Co, Ci, kH, kW, stride, pad);
                exit(1);
            }
        }

        if (has_bias) {
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT, 1, Co, 1, 1));
            alpha = 1.0f; beta = 1.0f;
            CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, biasDesc, bias.data,
                &beta, outDesc, output));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
        }

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }

    void destroy() {
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        weight.free();
        bias.free();
    }
};

// ============ Activation Kernels ============

__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = fmaxf(data[idx], 0.0f);
}

__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

__global__ void tanh_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = tanhf(data[idx]);
}

__global__ void softplus_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = scale * logf(1.0f + expf(data[idx]));
}

inline void relu_inplace(float* data, int n) {
    relu_kernel<<<(n+255)/256, 256>>>(data, n);
}

inline void sigmoid_inplace(float* data, int n) {
    sigmoid_kernel<<<(n+255)/256, 256>>>(data, n);
}

inline void tanh_inplace(float* data, int n) {
    tanh_kernel<<<(n+255)/256, 256>>>(data, n);
}

inline void softplus_inplace(float* data, int n, float scale = 1.0f) {
    softplus_kernel<<<(n+255)/256, 256>>>(data, n, scale);
}

// ============ Instance Normalization (no learnable params) ============
// For each (batch, channel) pair: normalize over (H, W)

__global__ void instance_norm_kernel(float* data, int N, int C, int HW) {
    // One block per (n, c) pair
    int nc = blockIdx.x;
    if (nc >= N * C) return;

    float* slice = data + nc * HW;

    // Compute mean using parallel reduction
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        sum += slice[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / HW;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float diff = slice[i] - mean;
        var_sum += diff * diff;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / HW + 1e-5f);
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        slice[i] = (slice[i] - mean) * inv_std;
}

inline void instance_norm(float* data, int N, int C, int H, int W) {
    int HW = H * W;
    int threads = min(256, HW);
    // Pad threads to power of 2
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    instance_norm_kernel<<<N*C, threads, threads*sizeof(float)>>>(data, N, C, HW);
}

// ============ Element-wise operations ============

__global__ void add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

__global__ void gru_update_kernel(float* net, const float* z, const float* q, int n) {
    // net = (1-z)*net + z*q
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float zi = z[idx];
        net[idx] = (1.0f - zi) * net[idx] + zi * q[idx];
    }
}

__global__ void mul_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] *= b[idx];
}

// Scale and add: dst = dst + scale * src
__global__ void scale_add_kernel(float* dst, const float* src, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += scale * src[idx];
}

// ============ Pooling ============

inline void avg_pool2d(cudnnHandle_t cudnn, float* input, float* output,
                       int N, int C, int H, int W, int kH, int kW, int sH, int sW) {
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t inDesc, outDesc;

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN, kH, kW, 0, 0, sH, sW));

    int outH = H / sH, outW = W / sW;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, outH, outW));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn, poolDesc, &alpha, inDesc, input, &beta, outDesc, output));

    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
}

// ============ Global average pool over spatial dims ============

__global__ void global_avg_pool_kernel(const float* input, float* output, int C, int HW) {
    // input: [N, C, H, W], output: [N, C, 1, 1]
    // One block per (n, c)
    int nc = blockIdx.x;
    const float* in_slice = input + nc * HW;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        sum += in_slice[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[nc] = sdata[0] / HW;
}

// Broadcast multiply: output[n,c,h,w] = a[n,c,h,w] * b[n,c,1,1]
__global__ void broadcast_mul_kernel(float* output, const float* a, const float* b,
                                     int NC, int HW) {
    int nc = blockIdx.x;
    if (nc >= NC) return;
    float scale = b[nc];
    float* out = output + nc * HW;
    const float* in = a + nc * HW;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        out[i] = in[i] * scale;
}

// Broadcast add: a[n,c,h,w] += b[n,c,1,1] (in-place on a)
__global__ void broadcast_add_kernel(float* a, const float* b, int NC, int HW) {
    int nc = blockIdx.x;
    if (nc >= NC) return;
    float val = b[nc];
    float* slice = a + nc * HW;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        slice[i] += val;
}

// ============ Concatenation ============

// Concatenate along channel dimension: [N, C1, H, W] + [N, C2, H, W] -> [N, C1+C2, H, W]
__global__ void concat2_kernel(float* output, const float* a, const float* b,
                               int N, int C1, int C2, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (C1 + C2) * HW;
    if (idx >= total) return;

    int hw = idx % HW;
    int c = (idx / HW) % (C1 + C2);
    int n = idx / (HW * (C1 + C2));

    if (c < C1) {
        output[idx] = a[n * C1 * HW + c * HW + hw];
    } else {
        output[idx] = b[n * C2 * HW + (c - C1) * HW + hw];
    }
}

// Concatenate 3 tensors along channel dim
__global__ void concat3_kernel(float* output, const float* a, const float* b, const float* c,
                               int N, int C1, int C2, int C3, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int CT = C1 + C2 + C3;
    int total = N * CT * HW;
    if (idx >= total) return;

    int hw = idx % HW;
    int ch = (idx / HW) % CT;
    int n = idx / (HW * CT);

    if (ch < C1) {
        output[idx] = a[n * C1 * HW + ch * HW + hw];
    } else if (ch < C1 + C2) {
        output[idx] = b[n * C2 * HW + (ch - C1) * HW + hw];
    } else {
        output[idx] = c[n * C3 * HW + (ch - C1 - C2) * HW + hw];
    }
}

// ============ Copy / Slice ============

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
