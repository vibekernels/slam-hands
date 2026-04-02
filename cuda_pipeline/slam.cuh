#pragma once

// Forward declarations needed
#include <cusolverDn.h>

namespace slam {

// === nn.cuh content ===
// Neural network primitives for CUDA DROID-SLAM
// Uses cuDNN for convolutions, custom kernels for InstanceNorm and activations


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
        CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

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

// Concatenate 4 tensors along channel dim: [N,C1,HW] + [N,C2,HW] + [N,C3,HW] + [N,C4,HW] -> [N,C1+C2+C3+C4,HW]
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

    if (ch < C1) {
        output[idx] = a[n * C1 * HW + ch * HW + hw];
    } else if (ch < C1 + C2) {
        output[idx] = b[n * C2 * HW + (ch - C1) * HW + hw];
    } else if (ch < C1 + C2 + C3) {
        output[idx] = c[n * C3 * HW + (ch - C1 - C2) * HW + hw];
    } else {
        output[idx] = d[n * C4 * HW + (ch - C1 - C2 - C3) * HW + hw];
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

// ============ FP16 Infrastructure ============


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

// FP32 ↔ FP16 conversion kernels
__global__ void float_to_half_kernel(const float* __restrict__ in, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
__global__ void half_to_float_kernel(const __half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
inline void float_to_half(const float* in, __half* out, int n) {
    float_to_half_kernel<<<(n+255)/256, 256>>>(in, out, n);
}
inline void half_to_float(const __half* in, float* out, int n) {
    half_to_float_kernel<<<(n+255)/256, 256>>>(in, out, n);
}

// FP16 activation kernels
__global__ void relu_half_kernel(__half* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(data[i]);
        data[i] = __float2half(fmaxf(v, 0.0f));
    }
}
inline void relu_half_inplace(__half* data, int n) {
    relu_half_kernel<<<(n+255)/256, 256>>>(data, n);
}

__global__ void add_half_kernel(__half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        a[i] = __float2half(va + vb);
    }
}

// Instance norm on FP16 data (accumulates in FP32 for precision)
__global__ void instance_norm_half_kernel(__half* data, int N, int C, int HW) {
    int nc = blockIdx.x;
    if (nc >= N * C) return;

    __half* slice = data + nc * HW;
    extern __shared__ float sdata[];

    // Compute mean in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        sum += __half2float(slice[i]);
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / HW;
    __syncthreads();

    // Compute variance in FP32
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float diff = __half2float(slice[i]) - mean;
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

    // Normalize and write back as FP16
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        slice[i] = __float2half((__half2float(slice[i]) - mean) * inv_std);
}

inline void instance_norm_half(__half* data, int N, int C, int H, int W) {
    int HW = H * W;
    int threads = min(256, HW);
    int t = 1; while (t < threads) t <<= 1; threads = t;
    instance_norm_half_kernel<<<N*C, threads, threads*sizeof(float)>>>(data, N, C, HW);
}

// Bias add for FP16 output with FP32 bias: out[n,c,h,w] += bias[c]
__global__ void bias_add_half_kernel(__half* output, const float* bias, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx >= total) return;
    int c = (idx / HW) % C;
    float val = __half2float(output[idx]) + bias[c];
    output[idx] = __float2half(val);
}

// ============ FP16 Convolution Layer ============

struct HalfConvLayer {
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize;

    GpuHalfBuf weight;
    GpuBuf bias;  // bias stays FP32 for cuDNN addTensor
    int Ci, Co, kH, kW, stride, pad;
    bool has_bias;

    void init(cudnnHandle_t cudnn, int batch,
              int ci, int co, int kh, int kw, int s, int p,
              int inH, int inW,
              const float* w_data, const float* b_data) {
        Ci = ci; Co = co; kH = kh; kW = kw; stride = s; pad = p;
        has_bias = (b_data != nullptr);

        // Convert weights FP32 → FP16 on GPU
        int w_count = co * ci * kh * kw;
        weight.alloc(w_count);
        {
            GpuBuf tmp; tmp.alloc(w_count);
            tmp.copyFrom(w_data, w_count);
            float_to_half(tmp.data, weight.data, w_count);
        }
        if (has_bias) {
            bias.alloc(co);
            bias.copyFrom(b_data, co);
        }

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_HALF,
            CUDNN_TENSOR_NCHW, co, ci, kh, kw));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, p, p, s, s, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));  // FP32 accumulation
        CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

        // Find best algorithm with HALF tensor descriptors
        cudnnTensorDescriptor_t inDesc, outDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, ci, inH, inW));
        int outH = (inH + 2*p - kh) / s + 1;
        int outW = (inW + 2*p - kw) / s + 1;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, co, outH, outW));

        cudnnConvolutionFwdAlgo_t algos[] = {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        };
        bool found = false;
        for (auto a : algos) {
            cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, inDesc, filterDesc, convDesc, outDesc, a, &workspaceSize);
            if (st == CUDNN_STATUS_SUCCESS) {
                algo = a; found = true; break;
            }
        }
        if (!found) {
            int cnt;
            cudnnConvolutionFwdAlgoPerf_t perf[1];
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn, inDesc, filterDesc, convDesc, outDesc, 1, &cnt, perf));
            algo = perf[0].algo;
            workspaceSize = perf[0].memory;
        }

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }

    // Forward: FP16 input → FP16 output, FP32 accumulation internally
    void forward(cudnnHandle_t cudnn, __half* input, __half* output, float* workspace,
                 int batch, int inH, int inW) {
        cudnnTensorDescriptor_t inDesc, outDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, Ci, inH, inW));
        int outH = (inH + 2*pad - kH) / stride + 1;
        int outW = (inW + 2*pad - kW) / stride + 1;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, Co, outH, outW));

        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inDesc, input,
            filterDesc, weight.data, convDesc, algo, workspace, workspaceSize,
            &beta, outDesc, output));

        if (has_bias) {
            // Add bias: need FP32 bias descriptor, FP16 output descriptor
            // cuDNN handles the mixed precision bias add
            cudnnTensorDescriptor_t biasDesc;
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT, 1, Co, 1, 1));
            // Convert output to FP32, add bias, convert back - but cuDNN can handle
            // mixed types with CUDNN_DATA_HALF output + CUDNN_DATA_FLOAT bias
            // Actually, cudnnAddTensor requires matching types. Use a FP16 bias instead.
            // Simpler: broadcast-add bias in a custom kernel
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
            // Custom bias add for FP16 output + FP32 bias
            int total = batch * Co * outH * outW;
            int hw = outH * outW;
            bias_add_half_kernel<<<(total+255)/256, 256>>>(output, bias.data, batch, Co, hw);
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

// === se3.cuh content ===
#define MIN_DEPTH 0.25f
// SE3 Lie group operations - ported directly from DROID-SLAM's droid_kernels.cu



__device__ __forceinline__ void
actSO3(const float *q, const float *X, float *Y) {
    float uv[3];
    uv[0] = 2.0f * (q[1]*X[2] - q[2]*X[1]);
    uv[1] = 2.0f * (q[2]*X[0] - q[0]*X[2]);
    uv[2] = 2.0f * (q[0]*X[1] - q[1]*X[0]);
    Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
    Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
    Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__ __forceinline__ void
actSE3(const float *t, const float *q, const float *X, float *Y) {
    actSO3(q, X, Y);
    Y[3] = X[3];
    Y[0] += X[3] * t[0];
    Y[1] += X[3] * t[1];
    Y[2] += X[3] * t[2];
}

__device__ __forceinline__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
    float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
    actSO3(qinv, &X[0], &Y[0]);
    actSO3(qinv, &X[3], &Y[3]);
    float u[3], v[3];
    u[0] = t[2]*X[1] - t[1]*X[2];
    u[1] = t[0]*X[2] - t[2]*X[0];
    u[2] = t[1]*X[0] - t[0]*X[1];
    actSO3(qinv, u, v);
    Y[3] += v[0];
    Y[4] += v[1];
    Y[5] += v[2];
}

__device__ __forceinline__ void
relSE3(const float *ti, const float *qi, const float *tj, const float *qj,
       float *tij, float *qij) {
    qij[0] = -qj[3]*qi[0] + qj[0]*qi[3] - qj[1]*qi[2] + qj[2]*qi[1];
    qij[1] = -qj[3]*qi[1] + qj[1]*qi[3] - qj[2]*qi[0] + qj[0]*qi[2];
    qij[2] = -qj[3]*qi[2] + qj[2]*qi[3] - qj[0]*qi[1] + qj[1]*qi[0];
    qij[3] =  qj[3]*qi[3] + qj[0]*qi[0] + qj[1]*qi[1] + qj[2]*qi[2];
    actSO3(qij, ti, tij);
    tij[0] = tj[0] - tij[0];
    tij[1] = tj[1] - tij[1];
    tij[2] = tj[2] - tij[2];
}

__device__ __forceinline__ void
expSO3(const float *phi, float *q) {
    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float theta_p4 = theta_sq * theta_sq;
    float theta = sqrtf(theta_sq);
    float imag, real;
    if (theta_sq < 1e-8f) {
        imag = 0.5f - (1.0f/48.0f)*theta_sq + (1.0f/3840.0f)*theta_p4;
        real = 1.0f - (1.0f/8.0f)*theta_sq + (1.0f/384.0f)*theta_p4;
    } else {
        imag = sinf(0.5f * theta) / theta;
        real = cosf(0.5f * theta);
    }
    q[0] = imag * phi[0];
    q[1] = imag * phi[1];
    q[2] = imag * phi[2];
    q[3] = real;
}

__device__ __forceinline__ void
crossInplace(const float *a, float *b) {
    float x[3] = {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
    b[0] = x[0]; b[1] = x[1]; b[2] = x[2];
}

__device__ __forceinline__ void
expSE3(const float *xi, float *t, float *q) {
    expSO3(xi + 3, q);
    float tau[3] = {xi[0], xi[1], xi[2]};
    float phi[3] = {xi[3], xi[4], xi[5]};
    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float theta = sqrtf(theta_sq);
    t[0] = tau[0]; t[1] = tau[1]; t[2] = tau[2];
    if (theta > 1e-4f) {
        float a = (1.0f - cosf(theta)) / theta_sq;
        crossInplace(phi, tau);
        t[0] += a * tau[0]; t[1] += a * tau[1]; t[2] += a * tau[2];
        float b = (theta - sinf(theta)) / (theta * theta_sq);
        crossInplace(phi, tau);
        t[0] += b * tau[0]; t[1] += b * tau[1]; t[2] += b * tau[2];
    }
}

__device__ __forceinline__ void
retrSE3(const float *xi, const float *t, const float *q, float *t1, float *q1) {
    float dt[3] = {0,0,0};
    float dq[4] = {0,0,0,1};
    expSE3(xi, dt, dq);
    q1[0] = dq[3]*q[0] + dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1];
    q1[1] = dq[3]*q[1] + dq[1]*q[3] + dq[2]*q[0] - dq[0]*q[2];
    q1[2] = dq[3]*q[2] + dq[2]*q[3] + dq[0]*q[1] - dq[1]*q[0];
    q1[3] = dq[3]*q[3] - dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2];
    actSO3(dq, t, t1);
    t1[0] += dt[0]; t1[1] += dt[1]; t1[2] += dt[2];
}

// ============ Global kernels ============

__global__ void projmap_kernel(
    const float* __restrict__ poses,   // [N, 7]
    const float* __restrict__ disps,   // [N, H, W]
    const float* __restrict__ intrinsics, // [4]
    const int* __restrict__ ii,        // [M]
    const int* __restrict__ jj,        // [M]
    float* __restrict__ coords,        // [M, H, W, 2]
    float* __restrict__ valid,         // [M, H, W, 1]
    int M, int H, int W)
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (block_id >= M) return;

    __shared__ int ix, jx;
    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

    if (thread_id == 0) {
        ix = ii[block_id];
        jx = jj[block_id];
        fx = intrinsics[0]; fy = intrinsics[1];
        cx = intrinsics[2]; cy = intrinsics[3];
    }
    __syncthreads();

    if (thread_id < 3) {
        ti[thread_id] = poses[ix*7 + thread_id];
        tj[thread_id] = poses[jx*7 + thread_id];
    }
    if (thread_id < 4) {
        qi[thread_id] = poses[ix*7 + 3 + thread_id];
        qj[thread_id] = poses[jx*7 + 3 + thread_id];
    }
    __syncthreads();
    if (thread_id == 0) relSE3(ti, qi, tj, qj, tij, qij);
    __syncthreads();

    float Xi[4], Xj[4];
    for (int k = thread_id; k < H*W; k += blockDim.x) {
        int i = k / W, j = k % W;
        float u = (float)j, v = (float)i;
        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1.0f;
        Xi[3] = disps[ix*H*W + k];
        actSE3(tij, qij, Xi, Xj);

        int base = (block_id*H*W + k) * 2;
        if (Xj[2] > 0.01f) {
            coords[base + 0] = fx * (Xj[0] / Xj[2]) + cx;
            coords[base + 1] = fy * (Xj[1] / Xj[2]) + cy;
        } else {
            coords[base + 0] = u;
            coords[base + 1] = v;
        }
        valid[block_id*H*W + k] = (Xj[2] > MIN_DEPTH) ? 1.0f : 0.0f;
    }
}

__global__ void pose_retr_kernel(
    float* __restrict__ poses,         // [N, 7]
    const float* __restrict__ dx,      // [num, 6]
    int t0, int t1)
{
    for (int k = t0 + threadIdx.x; k < t1; k += blockDim.x) {
        float xi[6], q[4], q1_[4], t_[3], t1_[3];
        t_[0] = poses[k*7+0]; t_[1] = poses[k*7+1]; t_[2] = poses[k*7+2];
        q[0] = poses[k*7+3]; q[1] = poses[k*7+4]; q[2] = poses[k*7+5]; q[3] = poses[k*7+6];
        for (int n = 0; n < 6; n++) xi[n] = dx[(k-t0)*6 + n];
        retrSE3(xi, t_, q, t1_, q1_);
        poses[k*7+0] = t1_[0]; poses[k*7+1] = t1_[1]; poses[k*7+2] = t1_[2];
        poses[k*7+3] = q1_[0]; poses[k*7+4] = q1_[1]; poses[k*7+5] = q1_[2]; poses[k*7+6] = q1_[3];
    }
}

__global__ void disp_retr_kernel(
    float* __restrict__ disps,         // [N, H, W]
    const float* __restrict__ dz,      // [num_inds, H*W]
    const int* __restrict__ inds,      // [num_inds]
    int HW)
{
    int idx = inds[blockIdx.x];
    for (int k = threadIdx.x; k < HW; k += blockDim.x) {
        float d = disps[idx*HW + k] + dz[blockIdx.x*HW + k];
        disps[idx*HW + k] = fmaxf(d, 0.001f);
    }
}

// Frame distance kernel for keyframe selection
__global__ void frame_distance_kernel(
    const float* __restrict__ poses,
    const float* __restrict__ disps,
    const float* __restrict__ intrinsics,
    const int* __restrict__ ii,
    const int* __restrict__ jj,
    float* __restrict__ dist,
    int M, int H, int W, float beta)
{
    const int block_id = blockIdx.x;
    if (block_id >= M) return;

    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];
    __shared__ int ix, jx;

    if (threadIdx.x == 0) {
        ix = ii[block_id]; jx = jj[block_id];
        fx = intrinsics[0]; fy = intrinsics[1];
        cx = intrinsics[2]; cy = intrinsics[3];
    }
    __syncthreads();

    if (threadIdx.x < 3) { ti[threadIdx.x] = poses[ix*7+threadIdx.x]; tj[threadIdx.x] = poses[jx*7+threadIdx.x]; }
    if (threadIdx.x < 4) { qi[threadIdx.x] = poses[ix*7+3+threadIdx.x]; qj[threadIdx.x] = poses[jx*7+3+threadIdx.x]; }
    __syncthreads();
    if (threadIdx.x == 0) relSE3(ti, qi, tj, qj, tij, qij);
    __syncthreads();

    float accum_val = 0, valid_val = 0, total_val = 0;
    float Xi[4], Xj[4];

    for (int k = threadIdx.x; k < H*W; k += blockDim.x) {
        int i = k / W, j = k % W;
        float u = (float)j, v = (float)i;
        Xi[0] = (u-cx)/fx; Xi[1] = (v-cy)/fy; Xi[2] = 1; Xi[3] = disps[ix*H*W+k];
        actSE3(tij, qij, Xi, Xj);
        float du = fx*(Xj[0]/Xj[2])+cx-u, dv = fy*(Xj[1]/Xj[2])+cy-v;
        float d = sqrtf(du*du + dv*dv);
        total_val += beta;
        if (Xj[2] > MIN_DEPTH) { accum_val += beta*d; valid_val += beta; }

        // Translation-only version
        Xj[0] = Xi[0]+Xi[3]*tij[0]; Xj[1] = Xi[1]+Xi[3]*tij[1]; Xj[2] = Xi[2]+Xi[3]*tij[2];
        du = fx*(Xj[0]/Xj[2])+cx-u; dv = fy*(Xj[1]/Xj[2])+cy-v;
        d = sqrtf(du*du + dv*dv);
        total_val += (1-beta);
        if (Xj[2] > MIN_DEPTH) { accum_val += (1-beta)*d; valid_val += (1-beta); }
    }

    // Block reduce
    __shared__ float s_accum[256], s_valid[256], s_total[256];
    s_accum[threadIdx.x] = accum_val;
    s_valid[threadIdx.x] = valid_val;
    s_total[threadIdx.x] = total_val;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_accum[threadIdx.x] += s_accum[threadIdx.x+s];
            s_valid[threadIdx.x] += s_valid[threadIdx.x+s];
            s_total[threadIdx.x] += s_total[threadIdx.x+s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dist[block_id] = (s_valid[0]/(s_total[0]+1e-8f) < 0.75f) ? 1000.0f : s_accum[0]/s_valid[0];
    }
}

// === ba.cuh content ===
// Bundle Adjustment for CUDA DROID-SLAM
// Ported from DROID-SLAM's droid_kernels.cu


#define BA_THREADS 256

// ============ Reduction helpers ============

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
    unsigned int tid = threadIdx.x;
    __syncthreads();
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads();
    if (tid <  32) warpReduce(sdata, tid);
    __syncthreads();
}

// ============ Projective Transform Kernel ============
// One block per edge. Computes Jacobians and accumulates Hessian blocks.

__global__ void projective_transform_kernel(
    const float* __restrict__ target,    // [M, 2, H, W]
    const float* __restrict__ weight,    // [M, 2, H, W]
    const float* __restrict__ poses,     // [N, 7]
    const float* __restrict__ disps,     // [N, H, W]
    const float* __restrict__ intrinsics,// [4]
    const int* __restrict__ ii,          // [M]
    const int* __restrict__ jj,          // [M]
    float* __restrict__ Hs,              // [4, M, 6, 6]
    float* __restrict__ vs,              // [2, M, 6]
    float* __restrict__ Eii,             // [M, 6, H*W]
    float* __restrict__ Eij,             // [M, 6, H*W]
    float* __restrict__ Cii,             // [M, H*W]
    float* __restrict__ bz,              // [M, H*W]
    int M, int H, int W)
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (block_id >= M) return;

    const int HW = H * W;
    int ix = ii[block_id];
    int jx = jj[block_id];

    __shared__ float fx, fy, cx, cy;
    __shared__ float ti[3], tj[3], tij[3];
    __shared__ float qi[4], qj[4], qij[4];

    if (thread_id == 0) {
        fx = intrinsics[0]; fy = intrinsics[1];
        cx = intrinsics[2]; cy = intrinsics[3];
    }
    __syncthreads();

    if (ix == jx) {
        // Stereo frame (baseline = -0.1)
        if (thread_id == 0) {
            tij[0] = -0.1f; tij[1] = 0; tij[2] = 0;
            qij[0] = 0; qij[1] = 0; qij[2] = 0; qij[3] = 1;
        }
    } else {
        if (thread_id < 3) {
            ti[thread_id] = poses[ix*7 + thread_id];
            tj[thread_id] = poses[jx*7 + thread_id];
        }
        if (thread_id < 4) {
            qi[thread_id] = poses[ix*7 + 3 + thread_id];
            qj[thread_id] = poses[jx*7 + 3 + thread_id];
        }
        __syncthreads();
        if (thread_id == 0) relSE3(ti, qi, tj, qj, tij, qij);
    }
    __syncthreads();

    float Xi[4], Xj[4];
    float Jx[12], Jz;
    float* Ji = &Jx[0];
    float* Jj = &Jx[6];

    // Upper triangular of 12x12 Hessian
    float hij[12*13/2];
    float vi_[6], vj_[6];

    for (int l = 0; l < 78; l++) hij[l] = 0;
    for (int n = 0; n < 6; n++) { vi_[n] = 0; vj_[n] = 0; }

    __syncthreads();

    for (int k = thread_id; k < HW; k += blockDim.x) {
        int i = k / W, j = k % W;
        float u = (float)j, v = (float)i;

        Xi[0] = (u - cx) / fx;
        Xi[1] = (v - cy) / fy;
        Xi[2] = 1.0f;
        Xi[3] = disps[ix*HW + k];

        actSE3(tij, qij, Xi, Xj);
        float x = Xj[0], y = Xj[1], h = Xj[3];
        float d = (Xj[2] < MIN_DEPTH) ? 0.0f : 1.0f / Xj[2];
        float d2 = d * d;

        float wu = (Xj[2] < MIN_DEPTH) ? 0.0f : 0.001f * weight[block_id*2*HW + 0*HW + k];
        float wv = (Xj[2] < MIN_DEPTH) ? 0.0f : 0.001f * weight[block_id*2*HW + 1*HW + k];
        float ru = target[block_id*2*HW + 0*HW + k] - (fx*d*x + cx);
        float rv = target[block_id*2*HW + 1*HW + k] - (fy*d*y + cy);

        // x-coordinate Jacobians
        Jj[0] = fx*(h*d);
        Jj[1] = 0;
        Jj[2] = fx*(-x*h*d2);
        Jj[3] = fx*(-x*y*d2);
        Jj[4] = fx*(1 + x*x*d2);
        Jj[5] = fx*(-y*d);

        Jz = fx*(tij[0]*d - tij[2]*(x*d2));
        Cii[block_id*HW + k] = wu*Jz*Jz;
        bz[block_id*HW + k] = wu*ru*Jz;

        float wu_eff = (ix == jx) ? 0.0f : wu;
        adjSE3(tij, qij, Jj, Ji);
        for (int n = 0; n < 6; n++) Ji[n] *= -1;

        int l = 0;
        for (int n = 0; n < 12; n++)
            for (int m = 0; m <= n; m++)
                hij[l++] += wu_eff * Jx[n] * Jx[m];

        for (int n = 0; n < 6; n++) {
            vi_[n] += wu_eff * ru * Ji[n];
            vj_[n] += wu_eff * ru * Jj[n];
            Eii[block_id*6*HW + n*HW + k] = wu_eff * Jz * Ji[n];
            Eij[block_id*6*HW + n*HW + k] = wu_eff * Jz * Jj[n];
        }

        // y-coordinate Jacobians
        Jj[0] = 0;
        Jj[1] = fy*(h*d);
        Jj[2] = fy*(-y*h*d2);
        Jj[3] = fy*(-1 - y*y*d2);
        Jj[4] = fy*(x*y*d2);
        Jj[5] = fy*(x*d);

        Jz = fy*(tij[1]*d - tij[2]*(y*d2));
        Cii[block_id*HW + k] += wv*Jz*Jz;
        bz[block_id*HW + k] += wv*rv*Jz;

        float wv_eff = (ix == jx) ? 0.0f : wv;
        adjSE3(tij, qij, Jj, Ji);
        for (int n = 0; n < 6; n++) Ji[n] *= -1;

        l = 0;
        for (int n = 0; n < 12; n++)
            for (int m = 0; m <= n; m++)
                hij[l++] += wv_eff * Jx[n] * Jx[m];

        for (int n = 0; n < 6; n++) {
            vi_[n] += wv_eff * rv * Ji[n];
            vj_[n] += wv_eff * rv * Jj[n];
            Eii[block_id*6*HW + n*HW + k] += wv_eff * Jz * Ji[n];
            Eij[block_id*6*HW + n*HW + k] += wv_eff * Jz * Jj[n];
        }
    }
    __syncthreads();

    // Reduce vi, vj across threads
    __shared__ float sdata[BA_THREADS];
    for (int n = 0; n < 6; n++) {
        sdata[threadIdx.x] = vi_[n];
        blockReduce(sdata);
        if (threadIdx.x == 0) vs[0*M*6 + block_id*6 + n] = sdata[0];
        __syncthreads();
        sdata[threadIdx.x] = vj_[n];
        blockReduce(sdata);
        if (threadIdx.x == 0) vs[1*M*6 + block_id*6 + n] = sdata[0];
        __syncthreads();
    }

    // Reduce Hessian blocks
    int l = 0;
    for (int n = 0; n < 12; n++) {
        for (int m = 0; m <= n; m++) {
            sdata[threadIdx.x] = hij[l];
            blockReduce(sdata);
            if (threadIdx.x == 0) {
                if (n < 6 && m < 6) {
                    // Hs[0]: ii-ii block
                    Hs[0*M*36 + block_id*36 + n*6 + m] = sdata[0];
                    Hs[0*M*36 + block_id*36 + m*6 + n] = sdata[0];
                } else if (n >= 6 && m < 6) {
                    // Hs[1]: ii-jj, Hs[2]: jj-ii
                    Hs[1*M*36 + block_id*36 + m*6 + (n-6)] = sdata[0];
                    Hs[2*M*36 + block_id*36 + (n-6)*6 + m] = sdata[0];
                } else {
                    // Hs[3]: jj-jj
                    Hs[3*M*36 + block_id*36 + (n-6)*6 + (m-6)] = sdata[0];
                    Hs[3*M*36 + block_id*36 + (m-6)*6 + (n-6)] = sdata[0];
                }
            }
            l++;
        }
    }
}

// ============ Accumulate kernel ============
// Scatter-sum data[src_indices] into out[dst_indices]

__global__ void accum_kernel(
    const float* __restrict__ inps, // [N_in, D]
    const int* __restrict__ ptrs,   // [N_out + 1]
    const int* __restrict__ idxs,   // [nnz]
    float* __restrict__ outs,       // [N_out, D]
    int D)
{
    const int block_id = blockIdx.x;
    int start = ptrs[block_id];
    int end = ptrs[block_id + 1];

    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float x = 0;
        for (int i = start; i < end; i++)
            x += inps[idxs[i] * D + k];
        outs[block_id * D + k] = x;
    }
}

// ============ EEt kernel: S -= E * Q * E^T ============

__global__ void EEt6x6_kernel(
    const float* __restrict__ E,  // [N_E, 6, D]
    const float* __restrict__ Q,  // [N_kf, D]
    const int* __restrict__ idx,  // [N_pairs, 3] - (E_row_i, E_row_j, Q_row)
    float* __restrict__ S,        // [N_pairs, 6, 6]
    int D, int N_pairs)
{
    if (blockIdx.x >= N_pairs) return;
    int ix = idx[blockIdx.x * 3 + 0];
    int jx = idx[blockIdx.x * 3 + 1];
    int kx = idx[blockIdx.x * 3 + 2];

    float dS[6][6];
    float ei[6], ej[6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            dS[i][j] = 0;

    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float q = Q[kx * D + k];
        for (int n = 0; n < 6; n++) {
            ei[n] = E[ix * 6 * D + n * D + k] * q;
            ej[n] = E[jx * 6 * D + n * D + k];
        }
        for (int n = 0; n < 6; n++)
            for (int m = 0; m < 6; m++)
                dS[n][m] += ei[n] * ej[m];
    }
    __syncthreads();

    __shared__ float sdata[BA_THREADS];
    for (int n = 0; n < 6; n++) {
        for (int m = 0; m < 6; m++) {
            sdata[threadIdx.x] = dS[n][m];
            blockReduce(sdata);
            if (threadIdx.x == 0)
                S[blockIdx.x * 36 + n * 6 + m] = sdata[0];
        }
    }
}

// ============ Ev kernel: v -= E * Q * w ============

__global__ void Ev6x1_kernel(
    const float* __restrict__ E,  // [N_E, 6, D]
    const float* __restrict__ Q,  // [N_kf, D]
    const float* __restrict__ w,  // [N_kf, D]
    const int* __restrict__ idx,  // [N_v, 1] - which Q/w row to use
    float* __restrict__ v,        // [N_v, 6]
    int D, int N_v)
{
    if (blockIdx.x >= N_v) return;
    int kx = idx[blockIdx.x];

    float b[6] = {0,0,0,0,0,0};
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float q_w = Q[kx * D + k] * w[kx * D + k];
        for (int n = 0; n < 6; n++)
            b[n] += q_w * E[blockIdx.x * 6 * D + n * D + k];
    }
    __syncthreads();

    __shared__ float sdata[BA_THREADS];
    for (int n = 0; n < 6; n++) {
        sdata[threadIdx.x] = b[n];
        blockReduce(sdata);
        if (threadIdx.x == 0)
            v[blockIdx.x * 6 + n] += sdata[0];
    }
}

// ============ EvT kernel: w = E^T * x (back-substitution) ============

__global__ void EvT6x1_kernel(
    const float* __restrict__ E,  // [N_E, 6, D]
    const float* __restrict__ x,  // [P, 6]
    const int* __restrict__ idx,  // [N_E] - maps E row to x row
    float* __restrict__ w,        // [N_E, D]
    int D, int N_E, int P)
{
    if (blockIdx.x >= N_E) return;
    int ix = idx[blockIdx.x];
    if (ix < 0 || ix >= P) return;

    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        float dw = 0;
        for (int n = 0; n < 6; n++)
            dw += E[blockIdx.x * 6 * D + n * D + k] * x[ix * 6 + n];
        w[blockIdx.x * D + k] = dw;
    }
}

// ============ Host-side BA orchestration ============

struct BundleAdjustment {
    cusolverDnHandle_t cusolver;
    int H, W, HW;

    // Pre-allocated GPU buffers
    GpuBuf Hs_buf, vs_buf, Eii_buf, Eij_buf, Cii_buf, wi_buf;
    GpuBuf S_buf, b_buf, dx_buf, dz_buf;
    GpuBuf Q_buf, E_buf, w_buf, C_buf;
    GpuBuf workspace_cusolver;
    int* devInfo = nullptr;

    void init(int h, int w) {
        H = h; W = w; HW = h * w;
        cusolverDnCreate(&cusolver);
        CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
    }

    // Simple scatter-accumulate on CPU (small arrays)
    // Accumulates data[src] into out[dst] where src maps to dst via (ix, kx)
    void accum_cpu(const float* data_gpu, int data_rows, int D,
                   const int* ii_host, int num_edges,
                   const int* kx_host, int num_kf,
                   float* out_gpu) {
        // Download data
        std::vector<float> data(data_rows * D);
        CUDA_CHECK(cudaMemcpy(data.data(), data_gpu, data_rows * D * sizeof(float),
                              cudaMemcpyDeviceToHost));

        std::vector<float> out(num_kf * D, 0.0f);

        // For each keyframe k, accumulate all edges where ii[edge] maps to k
        for (int e = 0; e < num_edges; e++) {
            int src = e;
            // Find which output row this edge maps to
            for (int k = 0; k < num_kf; k++) {
                if (ii_host[e] == kx_host[k]) {
                    for (int d = 0; d < D; d++)
                        out[k * D + d] += data[src * D + d];
                    break;
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(out_gpu, out.data(), num_kf * D * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Solve dense linear system S * x = b via Cholesky on GPU
    // S: [N, N], b: [N, 1], result in b
    void cholesky_solve(float* S_gpu, float* b_gpu, int N) {
        int lwork = 0;
        cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER,
                                    N, S_gpu, N, &lwork);
        workspace_cusolver.alloc(lwork);
        cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_LOWER,
                         N, S_gpu, N,
                         workspace_cusolver.data, lwork, devInfo);
        cusolverDnSpotrs(cusolver, CUBLAS_FILL_MODE_LOWER,
                         N, 1, S_gpu, N, b_gpu, N, devInfo);
    }

    // Run one BA iteration
    // Returns dx [P, 6] and optionally dz [num_kf, HW]
    void iterate(
        float* poses, float* disps, float* intrinsics,
        float* target, float* weight,
        float* eta,  // [num_kf, HW] damping
        int* ii_gpu, int* jj_gpu,
        int* ii_host, int* jj_host,
        int num_edges, int t0, int t1,
        float lm, float ep, bool motion_only)
    {
        int P = t1 - t0;
        int M = num_edges;

        // Allocate per-edge buffers
        Hs_buf.alloc(4 * M * 36); Hs_buf.zero();
        vs_buf.alloc(2 * M * 6); vs_buf.zero();
        Eii_buf.alloc(M * 6 * HW); Eii_buf.zero();
        Eij_buf.alloc(M * 6 * HW); Eij_buf.zero();
        Cii_buf.alloc(M * HW); Cii_buf.zero();
        wi_buf.alloc(M * HW); wi_buf.zero();

        // Compute Jacobians
        projective_transform_kernel<<<M, BA_THREADS>>>(
            target, weight, poses, disps, intrinsics,
            ii_gpu, jj_gpu,
            Hs_buf.data, vs_buf.data,
            Eii_buf.data, Eij_buf.data,
            Cii_buf.data, wi_buf.data,
            M, H, W);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Assemble global Hessian on CPU (simpler for first version)
        // H is [P, P, 6, 6] dense
        int S_size = P * 6;
        std::vector<float> S_host(S_size * S_size, 0.0f);
        std::vector<float> b_host(S_size, 0.0f);

        // Download Hs and vs
        std::vector<float> Hs_host(4 * M * 36);
        std::vector<float> vs_host(2 * M * 6);
        Hs_buf.copyTo(Hs_host.data(), 4 * M * 36);
        vs_buf.copyTo(vs_host.data(), 2 * M * 6);

        // Scatter Hessian blocks into global matrix
        // Edges may reference fixed poses (index < t0), which are out of range.
        // Like PyTorch, we skip contributions where the index is out-of-range,
        // but keep contributions where only one endpoint is out-of-range.
        for (int e = 0; e < M; e++) {
            int i_idx = ii_host[e] - t0;
            int j_idx = jj_host[e] - t0;
            bool i_valid = (i_idx >= 0 && i_idx < P);
            bool j_valid = (j_idx >= 0 && j_idx < P);
            if (!i_valid && !j_valid) continue;

            // H[ii,ii] += Hs[0, e]
            if (i_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(i_idx*6+r)*S_size + (i_idx*6+c)] += Hs_host[0*M*36 + e*36 + r*6 + c];

            // H[ii,jj] += Hs[1, e]
            if (i_valid && j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(i_idx*6+r)*S_size + (j_idx*6+c)] += Hs_host[1*M*36 + e*36 + r*6 + c];

            // H[jj,ii] += Hs[2, e]
            if (i_valid && j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(j_idx*6+r)*S_size + (i_idx*6+c)] += Hs_host[2*M*36 + e*36 + r*6 + c];

            // H[jj,jj] += Hs[3, e]
            if (j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(j_idx*6+r)*S_size + (j_idx*6+c)] += Hs_host[3*M*36 + e*36 + r*6 + c];

            // b[ii] += vs[0, e]
            if (i_valid)
                for (int n = 0; n < 6; n++)
                    b_host[i_idx*6+n] += vs_host[0*M*6 + e*6 + n];
            // b[jj] += vs[1, e]
            if (j_valid)
                for (int n = 0; n < 6; n++)
                    b_host[j_idx*6+n] += vs_host[1*M*6 + e*6 + n];
        }

        if (!motion_only) {
            // TODO: Schur complement for depth optimization
            // For now, just do motion-only solve
            // The full implementation would:
            // 1. Accumulate Eii into per-keyframe E
            // 2. Accumulate Cii into per-keyframe C
            // 3. Compute Q = 1/C
            // 4. Subtract Schur complement: S -= E*Q*E^T, b -= E*Q*w
            // 5. Solve for dx
            // 6. Back-substitute for dz
        }

        // Add damping
        for (int i = 0; i < S_size; i++)
            S_host[i*S_size + i] += ep + lm * S_host[i*S_size + i];

        // Upload and solve
        S_buf.alloc(S_size * S_size);
        b_buf.alloc(S_size);
        S_buf.copyFrom(S_host.data(), S_size * S_size);
        b_buf.copyFrom(b_host.data(), S_size);

        cholesky_solve(S_buf.data, b_buf.data, S_size);

        // Apply pose retraction
        dx_buf.alloc(P * 6);
        CUDA_CHECK(cudaMemcpy(dx_buf.data, b_buf.data, P * 6 * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        pose_retr_kernel<<<1, BA_THREADS>>>(poses, dx_buf.data, t0, t1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void destroy() {
        cusolverDnDestroy(cusolver);
        if (devInfo) cudaFree(devInfo);
    }
};

// === SLAM structs and kernels from main.cu ===
// ============ Timing utility ============

struct CudaTimer {
    cudaEvent_t start, stop;
    const char* name;
    float elapsed_ms;

    CudaTimer(const char* n) : name(n), elapsed_ms(0) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void reset() { elapsed_ms = 0; }
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

    // Reset per-video state without reallocating buffers.
    // Keeps all GPU allocations and just resets poses/disps/edges/keyframes.
    void reset(float fx, float fy, float cx, float cy) {
        int hw = h * w;
        num_keyframes = 0;
        kf_timestamps.clear();
        ii_host.clear();
        jj_host.clear();

        // Set intrinsics (may differ per video)
        float intr[4] = {fx/8.0f, fy/8.0f, cx/8.0f, cy/8.0f};
        intrinsics.copyFrom(intr, 4);

        // Reset poses to identity quaternion
        std::vector<float> id_poses(MAX_KEYFRAMES * 7, 0.0f);
        for (int i = 0; i < MAX_KEYFRAMES; i++)
            id_poses[i*7 + 6] = 1.0f;  // qw = 1
        poses.copyFrom(id_poses.data(), MAX_KEYFRAMES * 7);

        // Reset disparities to 1.0
        std::vector<float> init_disps(MAX_KEYFRAMES * hw, 1.0f);
        disps.copyFrom(init_disps.data(), MAX_KEYFRAMES * hw);
    }
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

        fprintf(stderr, "Initializing encoders (FP16)...\n");
        fnet.init(cudnn, ws, "fnet", 1, H, W, 128, true);   // FP16 with instance norm
        cnet.init(cudnn, ws, "cnet", 1, H, W, 256, false);  // FP16 without instance norm

        fprintf(stderr, "Initializing update module...\n");
        update.init(cudnn, ws, MAX_BATCH_EDGES, h, w);

        fprintf(stderr, "Initializing BA...\n");
        ba.init(h, w);

        fprintf(stderr, "Initializing state...\n");
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
        fprintf(stderr, "CudaDroid initialized: %dx%d -> %dx%d\n", H, W, h, w);
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

        fprintf(stderr, "Backend: %d iterations, radius %d, %d keyframes\n", iters, radius, N);

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
        fprintf(stderr, "Backend done\n");
    }

    // Reset per-video state without reloading weights or reallocating buffers.
    // Call between videos when resolution hasn't changed.
    void reset(float fx, float fy, float cx, float cy) {
        state.reset(fx, fy, cx, cy);
        t_encode.reset();
        t_corr.reset();
        t_update.reset();
        t_ba.reset();
        t_total.reset();
    }

    void print_timing(int num_frames) {
        fprintf(stderr, "\n=== Timing Summary (%d frames) ===\n", num_frames);
        fprintf(stderr, "  Encode:      %8.2f ms total, %6.2f ms/frame\n",
               t_encode.elapsed_ms, t_encode.elapsed_ms / num_frames);
        fprintf(stderr, "  Correlation: %8.2f ms total, %6.2f ms/frame\n",
               t_corr.elapsed_ms, t_corr.elapsed_ms / num_frames);
        fprintf(stderr, "  Update:      %8.2f ms total, %6.2f ms/frame\n",
               t_update.elapsed_ms, t_update.elapsed_ms / num_frames);
        fprintf(stderr, "  BA:          %8.2f ms total, %6.2f ms/frame\n",
               t_ba.elapsed_ms, t_ba.elapsed_ms / num_frames);
        fprintf(stderr, "  Total:       %8.2f ms total, %6.2f ms/frame (%5.1f fps)\n",
               t_total.elapsed_ms, t_total.elapsed_ms / num_frames,
               1000.0f * num_frames / t_total.elapsed_ms);
    }

    void destroy() {
        cudnnDestroy(cudnn);
        cublasDestroy(cublas);
        ba.destroy();
    }
};

} // namespace slam
