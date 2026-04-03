#pragma once
// Bundle Adjustment for CUDA DROID-SLAM
// Ported from DROID-SLAM's droid_kernels.cu

#include "se3.cuh"
#include <cusolverDn.h>
#include <vector>
#include <algorithm>
#include <numeric>

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

    // Solve dense linear system S * x = b via Cholesky on CPU in double precision
    // (matching PyTorch's Eigen float64 solver)
    void cholesky_solve_double(std::vector<double>& S, std::vector<double>& b, int N,
                               std::vector<float>& dx_out) {
        // Simple Cholesky LLT factorization
        // L is stored in lower triangle of S (in-place)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += S[j*N+k] * S[j*N+k];
            S[j*N+j] = sqrt(S[j*N+j] - sum);
            for (int i = j+1; i < N; i++) {
                double sum2 = 0;
                for (int k = 0; k < j; k++)
                    sum2 += S[i*N+k] * S[j*N+k];
                S[i*N+j] = (S[i*N+j] - sum2) / S[j*N+j];
            }
        }
        // Forward substitution: L*y = b
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++)
                sum += S[i*N+k] * b[k];
            b[i] = (b[i] - sum) / S[i*N+i];
        }
        // Back substitution: L^T*x = y
        for (int i = N-1; i >= 0; i--) {
            double sum = 0;
            for (int k = i+1; k < N; k++)
                sum += S[k*N+i] * b[k];
            b[i] = (b[i] - sum) / S[i*N+i];
        }
        dx_out.resize(N);
        for (int i = 0; i < N; i++)
            dx_out[i] = (float)b[i];
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

        // Assemble global Hessian on CPU in DOUBLE precision (matching PyTorch's Eigen solver)
        int S_size = P * 6;
        std::vector<double> S_host(S_size * S_size, 0.0);
        std::vector<double> b_host(S_size, 0.0);

        // Download Hs and vs
        std::vector<float> Hs_host(4 * M * 36);
        std::vector<float> vs_host(2 * M * 6);
        Hs_buf.copyTo(Hs_host.data(), 4 * M * 36);
        vs_buf.copyTo(vs_host.data(), 2 * M * 6);

        // Scatter Hessian blocks into global matrix
        for (int e = 0; e < M; e++) {
            int i_idx = ii_host[e] - t0;
            int j_idx = jj_host[e] - t0;
            bool i_valid = (i_idx >= 0 && i_idx < P);
            bool j_valid = (j_idx >= 0 && j_idx < P);
            if (!i_valid && !j_valid) continue;

            if (i_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(i_idx*6+r)*S_size + (i_idx*6+c)] += (double)Hs_host[0*M*36 + e*36 + r*6 + c];

            if (i_valid && j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(i_idx*6+r)*S_size + (j_idx*6+c)] += (double)Hs_host[1*M*36 + e*36 + r*6 + c];

            if (i_valid && j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(j_idx*6+r)*S_size + (i_idx*6+c)] += (double)Hs_host[2*M*36 + e*36 + r*6 + c];

            if (j_valid)
                for (int r = 0; r < 6; r++)
                    for (int c = 0; c < 6; c++)
                        S_host[(j_idx*6+r)*S_size + (j_idx*6+c)] += (double)Hs_host[3*M*36 + e*36 + r*6 + c];

            if (i_valid)
                for (int n = 0; n < 6; n++)
                    b_host[i_idx*6+n] += (double)vs_host[0*M*6 + e*6 + n];
            if (j_valid)
                for (int n = 0; n < 6; n++)
                    b_host[j_idx*6+n] += (double)vs_host[1*M*6 + e*6 + n];
        }

        if (!motion_only) {
            // Download per-edge Jacobian coupling terms
            std::vector<float> Eii_h(M * 6 * HW), Eij_h(M * 6 * HW);
            std::vector<float> Cii_h(M * HW), bz_h(M * HW);
            Eii_buf.copyTo(Eii_h.data(), M * 6 * HW);
            Eij_buf.copyTo(Eij_h.data(), M * 6 * HW);
            Cii_buf.copyTo(Cii_h.data(), M * HW);
            wi_buf.copyTo(bz_h.data(), M * HW);

            // Download eta (GRU damping per keyframe)
            std::vector<float> eta_h(t1 * HW);
            CUDA_CHECK(cudaMemcpy(eta_h.data(), eta, t1 * HW * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // For each depth keyframe k referenced by edges, compute Schur complement
            for (int k = 0; k < t1; k++) {
                // Find all edges with ii[e] == k
                std::vector<int> k_edges;
                for (int e = 0; e < M; e++)
                    if (ii_host[e] == k) k_edges.push_back(e);
                if (k_edges.empty()) continue;

                // Accumulate C_k and w_k for this depth keyframe
                std::vector<float> C_k(HW, 0), w_k(HW, 0);
                for (int e : k_edges) {
                    for (int d = 0; d < HW; d++) {
                        C_k[d] += Cii_h[e * HW + d];
                        w_k[d] += bz_h[e * HW + d];
                    }
                }
                // Add eta damping (matching PyTorch: C += 0.2*damping + EP)
                for (int d = 0; d < HW; d++)
                    C_k[d] += 0.2f * eta_h[k * HW + d] + 1e-7f;

                // Q_k = 1/C_k
                std::vector<float> Q_k(HW);
                for (int d = 0; d < HW; d++)
                    Q_k[d] = 1.0f / C_k[d];

                // Build E coupling vectors per connected pose (only optimizable poses)
                std::vector<std::pair<int, std::vector<float>>> pose_E_list;
                auto get_or_add = [&](int p_idx) -> std::vector<float>& {
                    for (auto& [idx, vec] : pose_E_list)
                        if (idx == p_idx) return vec;
                    pose_E_list.push_back({p_idx, std::vector<float>(6 * HW, 0)});
                    return pose_E_list.back().second;
                };

                // Self-coupling (pose k, depth k): accumulate Eii
                int k_idx = k - t0;
                if (k_idx >= 0 && k_idx < P) {
                    auto& Ek = get_or_add(k_idx);
                    for (int e : k_edges)
                        for (int n = 0; n < 6; n++)
                            for (int d = 0; d < HW; d++)
                                Ek[n * HW + d] += Eii_h[e * 6 * HW + n * HW + d];
                }

                // Cross-coupling (pose jj[e], depth k): accumulate Eij
                for (int e : k_edges) {
                    int p = jj_host[e] - t0;
                    if (p < 0 || p >= P) continue;
                    auto& Ep = get_or_add(p);
                    for (int n = 0; n < 6; n++)
                        for (int d = 0; d < HW; d++)
                            Ep[n * HW + d] += Eij_h[e * 6 * HW + n * HW + d];
                }

                // Subtract Schur complement: S -= E * Q * E^T, b -= E * Q * w
                for (auto& [p1, E1] : pose_E_list) {
                    for (int n = 0; n < 6; n++) {
                        double sum = 0;
                        for (int d = 0; d < HW; d++)
                            sum += (double)E1[n * HW + d] * (double)Q_k[d] * (double)w_k[d];
                        b_host[p1 * 6 + n] -= sum;
                    }
                    for (auto& [p2, E2] : pose_E_list) {
                        for (int n = 0; n < 6; n++) {
                            for (int m = 0; m < 6; m++) {
                                double sum = 0;
                                for (int d = 0; d < HW; d++)
                                    sum += (double)E1[n * HW + d] * (double)Q_k[d] * (double)E2[m * HW + d];
                                S_host[(p1*6+n) * S_size + (p2*6+m)] -= sum;
                            }
                        }
                    }
                }
            }
        }

        // Add damping (in double)
        for (int i = 0; i < S_size; i++)
            S_host[i*S_size + i] += (double)ep + (double)lm * S_host[i*S_size + i];

        // Solve in double precision on CPU
        std::vector<float> dx_f32;
        cholesky_solve_double(S_host, b_host, S_size, dx_f32);

        // Apply pose retraction
        dx_buf.alloc(P * 6);
        CUDA_CHECK(cudaMemcpy(dx_buf.data, dx_f32.data(), P * 6 * sizeof(float),
                              cudaMemcpyHostToDevice));
        pose_retr_kernel<<<1, BA_THREADS>>>(poses, dx_buf.data, t0, t1);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (!motion_only) {
            // Back-substitute for depth updates: dz = Q * (w - E^T * dx)
            std::vector<float>& dx_h = dx_f32;

            // Re-download per-edge Eii, Eij, Cii, bz for back-substitution
            std::vector<float> Eii_h(M * 6 * HW), Eij_h(M * 6 * HW);
            std::vector<float> Cii_h(M * HW), bz_h(M * HW);
            Eii_buf.copyTo(Eii_h.data(), M * 6 * HW);
            Eij_buf.copyTo(Eij_h.data(), M * 6 * HW);
            Cii_buf.copyTo(Cii_h.data(), M * HW);
            wi_buf.copyTo(bz_h.data(), M * HW);

            std::vector<float> eta_h(t1 * HW);
            CUDA_CHECK(cudaMemcpy(eta_h.data(), eta, t1 * HW * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // For each depth keyframe k, compute dz[k]
            // Include ALL keyframes referenced by edges (matching PyTorch's kx)
            std::vector<float> all_dz(t1 * HW, 0);
            for (int k = 0; k < t1; k++) {
                int k_idx = k - t0;

                std::vector<int> k_edges;
                for (int e = 0; e < M; e++)
                    if (ii_host[e] == k) k_edges.push_back(e);
                if (k_edges.empty()) continue;

                // Accumulate C_k, w_k
                std::vector<float> C_k(HW, 0), w_k(HW, 0);
                for (int e : k_edges) {
                    for (int d = 0; d < HW; d++) {
                        C_k[d] += Cii_h[e * HW + d];
                        w_k[d] += bz_h[e * HW + d];
                    }
                }
                for (int d = 0; d < HW; d++)
                    C_k[d] += 0.2f * eta_h[k * HW + d] + 1e-7f;

                // Compute E^T * dx contribution
                // dw[d] = sum_p sum_n E[p,k,n,d] * dx[p,n]
                std::vector<float> dw(HW, 0);

                // From Eii: E[k_idx, k, n, d] * dx[k_idx, n]
                if (k_idx >= 0 && k_idx < P) {
                    // Accumulate Eii per keyframe
                    std::vector<float> Ek(6 * HW, 0);
                    for (int e : k_edges)
                        for (int n = 0; n < 6; n++)
                            for (int d = 0; d < HW; d++)
                                Ek[n * HW + d] += Eii_h[e * 6 * HW + n * HW + d];
                    for (int d = 0; d < HW; d++)
                        for (int n = 0; n < 6; n++)
                            dw[d] += Ek[n * HW + d] * dx_h[k_idx * 6 + n];
                }

                // From Eij: E[jj[e]-t0, k, n, d] * dx[jj[e]-t0, n]
                for (int e : k_edges) {
                    int p = jj_host[e] - t0;
                    if (p < 0 || p >= P) continue;
                    for (int d = 0; d < HW; d++)
                        for (int n = 0; n < 6; n++)
                            dw[d] += Eij_h[e * 6 * HW + n * HW + d] * dx_h[p * 6 + n];
                }

                // dz = Q * (w - dw), dampened
                for (int d = 0; d < HW; d++) {
                    float Q_kd = 1.0f / C_k[d];
                    all_dz[k * HW + d] = Q_kd * (w_k[d] - dw[d]);
                }
            }

            // Apply depth update: disps[k] += dz[k], clamp
            std::vector<float> disps_h(t1 * HW);
            CUDA_CHECK(cudaMemcpy(disps_h.data(), disps, t1 * HW * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Update ALL keyframes that had dz computed (matching PyTorch)
            for (int k = 0; k < t1; k++) {
                // Only update keyframes that had edges
                bool has_edges = false;
                for (int e = 0; e < M; e++) {
                    if (ii_host[e] == k) { has_edges = true; break; }
                }
                if (!has_edges) continue;

                for (int d = 0; d < HW; d++) {
                    float val = disps_h[k * HW + d] + all_dz[k * HW + d];
                    disps_h[k * HW + d] = fmaxf(0.001f, val);
                }
            }

            CUDA_CHECK(cudaMemcpy(disps, disps_h.data(), t1 * HW * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }
    }

    void destroy() {
        cusolverDnDestroy(cusolver);
        if (devInfo) cudaFree(devInfo);
    }
};
