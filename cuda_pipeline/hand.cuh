#pragma once

namespace hand {

// === nn.cuh content ===

// Neural network primitives for CUDA hand pose pipeline
// cuDNN for convolutions, cuBLAS for GEMM, custom kernels for activations/norms


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

// ============ FP16 Convolution Layer (tensor-core accelerated) ============

struct HalfConvLayer {
    cudnnFilterDescriptor_t filterDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize = 0;

    GpuHalfBuf weight, bias;
    int Ci, Co, kH, kW, stride, pad;

    void init(cudnnHandle_t cudnn, int batch,
              int ci, int co, int kh, int kw, int s, int p,
              int inH, int inW,
              const float* w_data, const float* b_data) {
        Ci = ci; Co = co; kH = kh; kW = kw; stride = s; pad = p;

        // Convert FP32 weights to FP16 on GPU
        int w_count = co * ci * kh * kw;
        weight.alloc(w_count);
        { GpuBuf tmp; tmp.alloc(w_count); tmp.copyFrom(w_data, w_count);
          float_to_half(tmp.data, weight.data, w_count); }
        bias.alloc(co);
        { GpuBuf tmp; tmp.alloc(co); tmp.copyFrom(b_data, co);
          float_to_half(tmp.data, bias.data, co); }

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_HALF,
            CUDNN_TENSOR_NCHW, co, ci, kh, kw));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, p, p, s, s, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

        cudnnTensorDescriptor_t inDesc, outDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, ci, inH, inW));
        int outH, outW, outN, outC;
        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inDesc, filterDesc,
            &outN, &outC, &outH, &outW));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, batch, co, outH, outW));

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

    void forward(cudnnHandle_t cudnn, __half* input, __half* output, void* workspace,
                 int batch, int inH, int inW) {
        cudnnTensorDescriptor_t inDesc, outDesc, biasDesc;
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

        // Add bias
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&biasDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF, 1, Co, 1, 1));
        alpha = 1.0f; beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, biasDesc, bias.data,
            &beta, outDesc, output));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(biasDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    }
};

// ============ FP16 Activation Kernels ============

__global__ void silu_half_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

__global__ void sigmoid_half_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(data[idx]);
        data[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

inline void silu_half_inplace(__half* data, int n) {
    silu_half_kernel<<<(n+255)/256, 256>>>(data, n);
}
inline void sigmoid_half_inplace(__half* data, int n) {
    sigmoid_half_kernel<<<(n+255)/256, 256>>>(data, n);
}

// ============ FP16 MaxPool2d ============

__global__ void maxpool2d_half_kernel(const __half* input, __half* output,
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
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                maxval = fmaxf(maxval, __half2float(input[((n * C + c) * H + ih) * W + iw]));
        }
    }
    output[idx] = __float2half(maxval);
}

inline void maxpool2d_half(__half* input, __half* output, int N, int C, int H, int W,
                           int kH, int sH, int pH) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pH - kH) / sH + 1;
    int total = N * C * oH * oW;
    maxpool2d_half_kernel<<<(total+255)/256, 256>>>(input, output, N, C, H, W, kH, sH, pH);
}

// ============ FP16 Upsample nearest 2x ============

__global__ void upsample2x_half_kernel(const __half* input, __half* output,
                                        int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oH = H * 2, oW = W * 2;
    int total = N * C * oH * oW;
    if (idx >= total) return;

    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int c = (idx / (oH * oW)) % C;
    int n = idx / (C * oH * oW);

    output[idx] = input[((n * C + c) * H + oh / 2) * W + ow / 2];
}

inline void upsample_nearest_2x_half(const __half* input, __half* output,
                                      int N, int C, int H, int W) {
    int total = N * C * (H*2) * (W*2);
    upsample2x_half_kernel<<<(total+255)/256, 256>>>(input, output, N, C, H, W);
}

// ============ FP16 Concatenation along channel dim ============

__global__ void concat2_half_kernel(__half* output, const __half* a, const __half* b,
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

__global__ void concat4_half_kernel(__half* output,
                                    const __half* a, const __half* b,
                                    const __half* c, const __half* d,
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

// ============ FP16 Channel slice ============

__global__ void slice_channels_half_kernel(__half* output, const __half* input,
                                           int N, int C_total, int C_start, int C_out, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * HW;
    if (idx >= total) return;
    int hw = idx % HW;
    int c = (idx / HW) % C_out;
    int n = idx / (HW * C_out);
    output[idx] = input[n * C_total * HW + (C_start + c) * HW + hw];
}

inline void slice_channels_half(__half* output, const __half* input,
                                int N, int C_total, int C_start, int C_out, int HW) {
    int total = N * C_out * HW;
    slice_channels_half_kernel<<<(total+255)/256, 256>>>(output, input, N, C_total, C_start, C_out, HW);
}

// === Helper kernels and constants from main.cu ===

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
// NVDEC NV12/P010 to BGR conversion kernels

// === YOLO and WiLoR models from main.cu ===

// ============================================================================
// GPU-side decode bias+init kernel for WiLoR ViT output
// ============================================================================
__global__ void decode_add_bias_init_kernel(
    float* pose, float* betas, float* cam,
    const float* pose_bias, const float* betas_bias, const float* cam_bias,
    const float* init_pose, const float* init_betas, const float* init_cam,
    int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Layout: first B*96 for pose, then B*10 for betas, then B*3 for cam
    if (idx < B * 96) {
        pose[idx] += pose_bias[idx % 6] + init_pose[idx % 96];
    } else if (idx < B * 96 + B * 10) {
        int i = idx - B * 96;
        betas[i] += betas_bias[i % 10] + init_betas[i % 10];
    } else if (idx < B * 96 + B * 10 + B * 3) {
        int i = idx - B * 96 - B * 10;
        cam[i] += cam_bias[i % 3] + init_cam[i % 3];
    }
}

// Extract image tokens from ViT output [B, T, D] and transpose to NCHW [B, D, H, W]
__global__ void extract_img_to_nchw_kernel(
    const float* __restrict__ vit_out,  // [B, T, D]
    float* __restrict__ nchw,           // [B, D, H, W]
    int B, int T, int D, int img_start, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D * HW) return;
    int b = idx / (D * HW);
    int rem = idx % (D * HW);
    int c = rem / HW;
    int hw = rem % HW;
    nchw[idx] = vit_out[b * T * D + (img_start + hw) * D + c];
}

// Write max-pooled vertex features into a slice of a concatenated feature buffer
__global__ void max_pool_points_to_slice_kernel(
    const float* __restrict__ feat,  // [B, C, N_pts]
    float* __restrict__ out,         // [B, total_dim] (write at feat_offset)
    int B, int C, int N_pts, int total_dim, int feat_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C) return;
    int b = idx / C;
    int c = idx % C;
    const float* f = feat + ((size_t)b * C + c) * N_pts;
    float mx = f[0];
    for (int i = 1; i < N_pts; i++) mx = fmaxf(mx, f[i]);
    out[b * total_dim + feat_offset + c] = mx;
}

// Project MANO vertices to 2D feature map coordinates (GPU version)
__global__ void project_verts_gpu_kernel(
    const float* __restrict__ verts,    // [B, 778, 3]
    const float* __restrict__ pred_cam, // [B, 3]
    float* __restrict__ grid_xy,        // [B, 778, 2]
    int B, int N_verts, int featH, int featW, float focal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N_verts) return;
    int b = idx / N_verts;
    int v = idx % N_verts;

    float s = pred_cam[b * 3 + 0];
    float tx = pred_cam[b * 3 + 1];
    float ty = pred_cam[b * 3 + 2];
    float tz = 2.0f * focal / (featH * s + 1e-9f);

    float vx = verts[b * N_verts * 3 + v * 3 + 0];
    float vy = verts[b * N_verts * 3 + v * 3 + 1];
    float vz = verts[b * N_verts * 3 + v * 3 + 2];

    float px = focal * (vx + tx) / (vz + tz + 1e-9f);
    float py = focal * (vy + ty) / (vz + tz + 1e-9f);

    // Normalize to [-1, 1] for grid_sample
    grid_xy[idx * 2 + 0] = px / (featW * 0.5f);
    grid_xy[idx * 2 + 1] = py / (featH * 0.5f);
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

    // GPU decode output buffers
    GpuBuf d_decode_pred_pose;   // [B, 96]
    GpuBuf d_decode_pred_betas;  // [B, 10]
    GpuBuf d_decode_pred_cam;    // [B, 3]

    // Pre-allocated RefineNet buffers (avoid per-forward cudaMalloc)
    GpuBuf d_refine_img_nchw;   // [B, 1280, 16, 12]
    GpuBuf d_refine_fc_out;     // [B, 640, 16, 12]
    GpuBuf d_refine_br0;        // [B, 320, 32, 24]
    GpuBuf d_refine_br1a;       // [B, 320, 32, 24]
    GpuBuf d_refine_br1b;       // [B, 160, 64, 48]
    GpuBuf d_refine_verts;      // [B, 778, 3]
    GpuBuf d_refine_grid_xy;    // [B, 778, 2]
    GpuBuf d_refine_vert_feat;  // [B, max_C, 778]
    GpuBuf d_refine_vert_max;   // [B, max_C]
    GpuBuf d_refine_all_feat;   // [B, 1120]

    // Host-cached constant data (loaded once in init, not per-forward)
    std::vector<float> h_decpose_w_cache, h_decpose_b_cache;
    std::vector<float> h_decshape_w_cache, h_decshape_b_cache;
    std::vector<float> h_deccam_w_cache, h_deccam_b_cache;
    std::vector<float> h_init_pose_cache, h_init_betas_cache, h_init_cam_cache;
    std::vector<float> h_mano_v_template_cache, h_mano_shapedirs_cache;
    std::vector<float> h_mano_posedirs_cache, h_mano_J_reg_cache, h_mano_lbs_w_cache;
    std::vector<float> h_refine_dec_pose_w_cache, h_refine_dec_pose_b_cache;
    std::vector<float> h_refine_dec_shape_w_cache, h_refine_dec_shape_b_cache;
    std::vector<float> h_refine_dec_cam_w_cache, h_refine_dec_cam_b_cache;

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

        // GPU decode output buffers
        d_decode_pred_pose.alloc(B * 96);
        d_decode_pred_betas.alloc(B * 10);
        d_decode_pred_cam.alloc(B * 3);

        // Pre-allocate RefineNet buffers
        int pH = VIT_PATCH_H, pW = VIT_PATCH_W;
        d_refine_img_nchw.alloc(B * D * pH * pW);
        d_refine_fc_out.alloc(B * 640 * pH * pW);
        d_refine_br0.alloc(B * 320 * pH * 2 * pW * 2);
        d_refine_br1a.alloc(B * 320 * pH * 2 * pW * 2);
        d_refine_br1b.alloc(B * 160 * pH * 4 * pW * 4);
        d_refine_verts.alloc(B * MANO_N_VERTS * 3);
        d_refine_grid_xy.alloc(B * MANO_N_VERTS * 2);
        d_refine_vert_feat.alloc(B * 640 * MANO_N_VERTS);  // max C = 640
        d_refine_vert_max.alloc(B * 640);
        d_refine_all_feat.alloc(B * 1120);

        // Cache constant data on host
        h_decpose_w_cache.resize(6 * D); decpose_weight.copyTo(h_decpose_w_cache.data(), 6 * D);
        h_decpose_b_cache.resize(6); decpose_bias.copyTo(h_decpose_b_cache.data(), 6);
        h_decshape_w_cache.resize(10 * D); decshape_weight.copyTo(h_decshape_w_cache.data(), 10 * D);
        h_decshape_b_cache.resize(10); decshape_bias.copyTo(h_decshape_b_cache.data(), 10);
        h_deccam_w_cache.resize(3 * D); deccam_weight.copyTo(h_deccam_w_cache.data(), 3 * D);
        h_deccam_b_cache.resize(3); deccam_bias.copyTo(h_deccam_b_cache.data(), 3);
        h_init_pose_cache.resize(96); init_hand_pose.copyTo(h_init_pose_cache.data(), 96);
        h_init_betas_cache.resize(10); init_betas.copyTo(h_init_betas_cache.data(), 10);
        h_init_cam_cache.resize(3); init_cam.copyTo(h_init_cam_cache.data(), 3);

        h_mano_v_template_cache.resize(778 * 3); mano_v_template.copyTo(h_mano_v_template_cache.data(), 778 * 3);
        h_mano_shapedirs_cache.resize(778 * 3 * 10); mano_shapedirs.copyTo(h_mano_shapedirs_cache.data(), 778 * 3 * 10);
        h_mano_posedirs_cache.resize(135 * 2334); mano_posedirs.copyTo(h_mano_posedirs_cache.data(), 135 * 2334);
        h_mano_J_reg_cache.resize(16 * 778); mano_J_regressor.copyTo(h_mano_J_reg_cache.data(), 16 * 778);
        h_mano_lbs_w_cache.resize(778 * 16); mano_lbs_weights.copyTo(h_mano_lbs_w_cache.data(), 778 * 16);

        h_refine_dec_pose_w_cache.resize(96 * 1120); refine_dec_pose_w.copyTo(h_refine_dec_pose_w_cache.data(), 96 * 1120);
        h_refine_dec_pose_b_cache.resize(96); refine_dec_pose_b.copyTo(h_refine_dec_pose_b_cache.data(), 96);
        h_refine_dec_shape_w_cache.resize(10 * 1120); refine_dec_shape_w.copyTo(h_refine_dec_shape_w_cache.data(), 10 * 1120);
        h_refine_dec_shape_b_cache.resize(10); refine_dec_shape_b.copyTo(h_refine_dec_shape_b_cache.data(), 10);
        h_refine_dec_cam_w_cache.resize(3 * 1120); refine_dec_cam_w.copyTo(h_refine_dec_cam_w_cache.data(), 3 * 1120);
        h_refine_dec_cam_b_cache.resize(3); refine_dec_cam_b.copyTo(h_refine_dec_cam_b_cache.data(), 3);
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
        if (B > max_batch) {
            fprintf(stderr, "[WiLoR] FATAL: B=%d exceeds max_batch=%d\n", B, max_batch);
            exit(1);
        }

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

        // ── Step 4: Last norm + GPU decode ──
        layernorm_half(vit_tokens.data, vit_tmp1.data,
                       last_norm_weight.data, last_norm_bias.data, B * T, D);

        // Convert output to FP32 for decode
        half_to_float(vit_tmp1.data, fp32_tmp1.data, B * T * D);

        // GPU-side decode: pose/shape/cam via cuBLAS strided batched GEMMs
        // Pose: [B×16, D] @ decpose_weight^T → [B×16, 6], strided across batches
        {
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                6, 16, D, &alpha,
                decpose_weight.data, CUDA_R_32F, D, 0,        // same weight all batches
                fp32_tmp1.data, CUDA_R_32F, D, (long long)T*D, // stride between batches
                &beta,
                d_decode_pred_pose.data, CUDA_R_32F, 6, 96,    // stride = 16*6 = 96
                B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            // Shape: token 16 → [B, D] @ decshape_weight^T → [B, 10]
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                10, 1, D, &alpha,
                decshape_weight.data, CUDA_R_32F, D, 0,
                fp32_tmp1.data + 16*D, CUDA_R_32F, D, (long long)T*D,
                &beta,
                d_decode_pred_betas.data, CUDA_R_32F, 10, 10,
                B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            // Cam: token 17 → [B, D] @ deccam_weight^T → [B, 3]
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                3, 1, D, &alpha,
                deccam_weight.data, CUDA_R_32F, D, 0,
                fp32_tmp1.data + 17*D, CUDA_R_32F, D, (long long)T*D,
                &beta,
                d_decode_pred_cam.data, CUDA_R_32F, 3, 3,
                B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            // Add bias + init values
            int total_elems = B * 96 + B * 10 + B * 3;
            decode_add_bias_init_kernel<<<(total_elems+255)/256, 256>>>(
                d_decode_pred_pose.data, d_decode_pred_betas.data, d_decode_pred_cam.data,
                decpose_bias.data, decshape_bias.data, deccam_bias.data,
                init_hand_pose.data, init_betas.data, init_cam.data, B);
        }

        // Download decoded values (tiny: ~42KB for B=96 vs 103MB before)
        std::vector<float> h_pred_pose(B * 96);
        std::vector<float> h_pred_betas(B * 10);
        std::vector<float> h_pred_cam(B * 3);
        d_decode_pred_pose.copyTo(h_pred_pose.data(), B * 96);
        d_decode_pred_betas.copyTo(h_pred_betas.data(), B * 10);
        d_decode_pred_cam.copyTo(h_pred_cam.data(), B * 3);

        // ── Step 5: First MANO pass (temp, cached constants) ──
        std::vector<float> h_temp_verts(B * 778 * 3);
        run_mano(h_pred_pose.data(), h_pred_betas.data(), B, h_temp_verts.data(), nullptr);

        // ── Step 6: RefineNet (img_feat stays on GPU) ──
        std::vector<float> h_delta_pose(B * 96);
        std::vector<float> h_delta_betas(B * 10);
        std::vector<float> h_delta_cam(B * 3);

        run_refine_net_gpu(fp32_tmp1.data, h_temp_verts.data(),
                           h_pred_cam.data(), B,
                           h_delta_pose.data(), h_delta_betas.data(), h_delta_cam.data());

        // Add deltas to predictions
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < 96; i++)
                h_pred_pose[b * 96 + i] += h_delta_pose[b * 96 + i];
            for (int i = 0; i < 10; i++)
                h_pred_betas[b * 10 + i] += h_delta_betas[b * 10 + i];
            for (int i = 0; i < 3; i++)
                h_pred_cam[b * 3 + i] += h_delta_cam[b * 3 + i];
        }

        // ── Step 7: Final MANO pass (cached constants) ──
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

        // Use cached MANO constants (no GPU download per call)
        const float* h_v_template = h_mano_v_template_cache.data();
        const float* h_shapedirs = h_mano_shapedirs_cache.data();
        const float* h_posedirs = h_mano_posedirs_cache.data();
        const float* h_J_reg = h_mano_J_reg_cache.data();
        const float* h_lbs_w = h_mano_lbs_w_cache.data();

        for (int b = 0; b < B; b++) {
            const float* betas = &pred_betas[b * 10];
            const float* rots = &rotmats[b * 16 * 9];

            // v_shaped = v_template + shapedirs @ betas
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
            float pose_feat[135];
            for (int j = 1; j < 16; j++) {
                for (int r = 0; r < 9; r++) {
                    float I_val = (r % 4 == 0) ? 1.0f : 0.0f;
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
            float transforms[16 * 16];
            batch_rigid_transform(rots, J, transforms);

            // Skinning
            for (int v = 0; v < 778; v++) {
                float T[16] = {0};
                for (int j = 0; j < 16; j++) {
                    float w = h_lbs_w[v * 16 + j];
                    for (int k = 0; k < 16; k++) {
                        T[k] += w * transforms[j * 16 + k];
                    }
                }
                float vx = v_posed[v * 3 + 0];
                float vy = v_posed[v * 3 + 1];
                float vz = v_posed[v * 3 + 2];
                out_verts[b * 778 * 3 + v * 3 + 0] = T[0]*vx + T[1]*vy + T[2]*vz + T[3];
                out_verts[b * 778 * 3 + v * 3 + 1] = T[4]*vx + T[5]*vy + T[6]*vz + T[7];
                out_verts[b * 778 * 3 + v * 3 + 2] = T[8]*vx + T[9]*vy + T[10]*vz + T[11];
            }

            // Extract joints
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

    // RefineNet forward (GPU-optimized: pre-allocated buffers, GPU vertex projection, cached weights)
    void run_refine_net_gpu(const float* d_vit_out, const float* h_temp_verts,
                            const float* h_pred_cam, int B,
                            float* h_delta_pose, float* h_delta_betas, float* h_delta_cam) {
        int H = VIT_PATCH_H, W = VIT_PATCH_W, D = VIT_EMBED;
        int T = VIT_TOTAL_TOKENS;
        int img_start = 18;  // image tokens start at index 18

        // Extract image tokens from GPU-resident ViT output [B, T, D] -> [B, D, H, W]
        int n_img = B * D * H * W;
        extract_img_to_nchw_kernel<<<(n_img + 255) / 256, 256>>>(
            d_vit_out, d_refine_img_nchw.data, B, T, D, img_start, H * W);

        // first_conv: [B, 1280, 16, 12] -> [B, 640, 16, 12]
        refine_first_conv.forward(cudnn, d_refine_img_nchw.data, d_refine_fc_out.data,
                                  workspace.data, B, H, W);

        // Branch 0: ConvTranspose(640->320) + ReLU -> [B, 320, 32, 24]
        refine_branch0_0.forward(cudnn, d_refine_fc_out.data, d_refine_br0.data,
                                 workspace.data, B, H, W);
        relu_inplace(d_refine_br0.data, B * 320 * H * 2 * W * 2);

        // Branch 1: ConvTranspose(640->320) + ReLU -> ConvTranspose(320->160) + ReLU
        refine_branch1_0.forward(cudnn, d_refine_fc_out.data, d_refine_br1a.data,
                                 workspace.data, B, H, W);
        relu_inplace(d_refine_br1a.data, B * 320 * H * 2 * W * 2);

        refine_branch1_1.forward(cudnn, d_refine_br1a.data, d_refine_br1b.data,
                                 workspace.data, B, H * 2, W * 2);
        relu_inplace(d_refine_br1b.data, B * 160 * H * 4 * W * 4);

        // Upload vertices to GPU for projection
        d_refine_verts.copyFrom(h_temp_verts, B * 778 * 3);

        // Upload pred_cam to GPU for projection kernel
        // Reuse d_decode_pred_cam as temp (it's already allocated for B*3)
        d_decode_pred_cam.copyFrom(h_pred_cam, B * 3);

        // Zero out the feature accumulation buffer
        d_refine_all_feat.zero();

        // Feature maps (high-res to low-res, as DeConvNet output[::-1])
        struct FeatInfo { float* data; int C, fH, fW; };
        FeatInfo feats[] = {
            {d_refine_br1b.data, 160, H * 4, W * 4},
            {d_refine_br0.data, 320, H * 2, W * 2},
            {d_refine_fc_out.data, 640, H, W},
        };

        int total_feat_dim = 1120;
        int feat_offset = 0;
        for (int fi = 0; fi < 3; fi++) {
            int fC = feats[fi].C, fH = feats[fi].fH, fW = feats[fi].fW;

            // Project vertices to 2D on GPU
            int n_proj = B * MANO_N_VERTS;
            project_verts_gpu_kernel<<<(n_proj + 255) / 256, 256>>>(
                d_refine_verts.data, d_decode_pred_cam.data, d_refine_grid_xy.data,
                B, MANO_N_VERTS, fH, fW, FOCAL_LENGTH);

            // grid_sample: [B, C, fH, fW] sampled at [B, 778, 2] -> [B, C, 778]
            grid_sample_kernel<<<(B * 778 + 255) / 256, 256>>>(
                feats[fi].data, d_refine_grid_xy.data, d_refine_vert_feat.data,
                B, fC, fH, fW, 778);

            // Max-pool over 778 vertices and write directly into concatenated buffer
            max_pool_points_to_slice_kernel<<<(B * fC + 255) / 256, 256>>>(
                d_refine_vert_feat.data, d_refine_all_feat.data,
                B, fC, 778, total_feat_dim, feat_offset);

            feat_offset += fC;
        }

        // Download concatenated features [B, 1120]
        std::vector<float> h_all_feat(B * total_feat_dim);
        d_refine_all_feat.copyTo(h_all_feat.data(), B * total_feat_dim);

        // Decode deltas using cached weights (no GPU download needed)
        for (int b = 0; b < B; b++) {
            const float* feat = &h_all_feat[b * 1120];
            for (int o = 0; o < 96; o++) {
                float val = h_refine_dec_pose_b_cache[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_refine_dec_pose_w_cache[o * 1120 + d];
                h_delta_pose[b * 96 + o] = val;
            }
            for (int o = 0; o < 10; o++) {
                float val = h_refine_dec_shape_b_cache[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_refine_dec_shape_w_cache[o * 1120 + d];
                h_delta_betas[b * 10 + o] = val;
            }
            for (int o = 0; o < 3; o++) {
                float val = h_refine_dec_cam_b_cache[o];
                for (int d = 0; d < 1120; d++) val += feat[d] * h_refine_dec_cam_w_cache[o * 1120 + d];
                h_delta_cam[b * 3 + o] = val;
            }
        }
    }
};

// ============================================================================
// Main entry point

} // namespace hand
