#pragma once
// SE3 Lie group operations - ported directly from DROID-SLAM's droid_kernels.cu

#include <cmath>

#define MIN_DEPTH 0.25f

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
