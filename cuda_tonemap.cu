/*
 * CUDA HDR→SDR tonemapper for BT.2020/HLG → BT.709 conversion.
 *
 * Reads yuv420p10le frames from stdin, tonemaps on GPU, writes yuv420p to stdout.
 * Designed to sit between two ffmpeg processes:
 *
 *   ffmpeg -hwaccel auto -i input.mov -f rawvideo -pix_fmt yuv420p10le pipe:1 |
 *   ./cuda_tonemap 1920 1080 |
 *   ffmpeg -f rawvideo -pixel_format yuv420p -video_size 1920x1080 ... -i pipe:0 ...
 *
 * Uses double-buffered CUDA streams to overlap transfers with compute.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* HLG EOTF: electrical signal → linear light */
__device__ float hlg_eotf(float x) {
    const float a = 0.17883277f;
    const float b = 1.0f - 4.0f * a;  /* 0.28466892 */
    const float c = 0.5f - a * logf(4.0f * a);  /* ~0.55991073 */
    if (x <= 0.5f)
        return x * x / 3.0f;
    else
        return (expf((x - c) / a) + b) / 12.0f;
}

/* BT.709 OETF: linear light → electrical signal */
__device__ float bt709_oetf(float x) {
    if (x < 0.018f)
        return 4.5f * x;
    else
        return 1.099f * powf(x, 0.45f) - 0.099f;
}

/*
 * Tonemap kernel: processes Y, Cb, Cr planes.
 * Each thread handles one LUMA pixel. Chroma is subsampled (4:2:0).
 *
 * Steps:
 *   1. Read 10-bit YCbCr (BT.2020, limited range)
 *   2. Convert to RGB (BT.2020)
 *   3. HLG EOTF (linearize)
 *   4. BT.2020 → BT.709 gamut mapping (3×3 matrix)
 *   5. BT.709 OETF (gamma)
 *   6. Convert back to YCbCr (BT.709, limited range)
 *   7. Write 8-bit output
 */
__global__ void tonemap_kernel(
    const uint16_t* __restrict__ y_in,
    const uint16_t* __restrict__ u_in,
    const uint16_t* __restrict__ v_in,
    uint8_t* __restrict__ y_out,
    uint8_t* __restrict__ u_out,
    uint8_t* __restrict__ v_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    /* Read 10-bit luma, normalize to [0,1] */
    float Y = (float)y_in[idx] / 1023.0f;

    /* Read 10-bit chroma (subsampled — use nearest for luma pixel) */
    int cx = x >> 1, cy = y >> 1;
    int cw = width >> 1;
    int cidx = cy * cw + cx;
    float Cb = (float)u_in[cidx] / 1023.0f;
    float Cr = (float)v_in[cidx] / 1023.0f;

    /* Limited range → full range */
    Y  = (Y  - 64.0f/1023.0f) * (1023.0f / (940.0f - 64.0f));
    Cb = (Cb - 512.0f/1023.0f) * (1023.0f / (960.0f - 64.0f));
    Cr = (Cr - 512.0f/1023.0f) * (1023.0f / (960.0f - 64.0f));

    /* BT.2020 NCL YCbCr → RGB */
    float R = Y + 1.4746f * Cr;
    float G = Y - 0.16455f * Cb - 0.57135f * Cr;
    float B = Y + 1.8814f * Cb;

    /* Clamp before EOTF */
    R = fminf(fmaxf(R, 0.0f), 1.0f);
    G = fminf(fmaxf(G, 0.0f), 1.0f);
    B = fminf(fmaxf(B, 0.0f), 1.0f);

    /* HLG EOTF: electrical → linear light */
    R = hlg_eotf(R);
    G = hlg_eotf(G);
    B = hlg_eotf(B);

    /* BT.2020 → BT.709 color gamut matrix */
    float R709 =  1.6605f * R - 0.5877f * G - 0.0728f * B;
    float G709 = -0.1246f * R + 1.1329f * G - 0.0083f * B;
    float B709 = -0.0182f * R - 0.1006f * G + 1.1187f * B;

    /* Clamp to [0,1] (gamut clip) */
    R709 = fminf(fmaxf(R709, 0.0f), 1.0f);
    G709 = fminf(fmaxf(G709, 0.0f), 1.0f);
    B709 = fminf(fmaxf(B709, 0.0f), 1.0f);

    /* BT.709 OETF: linear → electrical */
    R709 = bt709_oetf(R709);
    G709 = bt709_oetf(G709);
    B709 = bt709_oetf(B709);

    /* RGB → BT.709 YCbCr */
    float Y709  =  0.2126f * R709 + 0.7152f * G709 + 0.0722f * B709;
    float Cb709 = (B709 - Y709) / 1.8556f;
    float Cr709 = (R709 - Y709) / 1.5748f;

    /* 8-bit limited range */
    float y8f = Y709 * (235.0f - 16.0f) + 16.0f;
    y_out[idx] = (uint8_t)fminf(fmaxf(y8f + 0.5f, 0.0f), 255.0f);

    /* Chroma: only write if this is the top-left pixel of a 2×2 block */
    if ((x & 1) == 0 && (y & 1) == 0) {
        float u8f = (Cb709 + 0.5f) * (240.0f - 16.0f) + 16.0f;
        float v8f = (Cr709 + 0.5f) * (240.0f - 16.0f) + 16.0f;
        u_out[cidx] = (uint8_t)fminf(fmaxf(u8f + 0.5f, 0.0f), 255.0f);
        v_out[cidx] = (uint8_t)fminf(fmaxf(v8f + 0.5f, 0.0f), 255.0f);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <width> <height>\n", argv[0]);
        return 1;
    }

    int W = atoi(argv[1]);
    int H = atoi(argv[2]);
    int y_pixels = W * H;
    int uv_pixels = (W/2) * (H/2);

    /* Input: yuv420p10le (2 bytes per sample) */
    size_t in_y_bytes  = y_pixels * 2;
    size_t in_uv_bytes = uv_pixels * 2;
    size_t in_frame    = in_y_bytes + 2 * in_uv_bytes;

    /* Output: yuv420p (1 byte per sample) */
    size_t out_y_bytes  = y_pixels;
    size_t out_uv_bytes = uv_pixels;
    size_t out_frame    = out_y_bytes + 2 * out_uv_bytes;

    /* Host pinned memory for async transfers (double-buffered) */
    uint8_t *h_in[2], *h_out[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMallocHost(&h_in[i], in_frame));
        CHECK_CUDA(cudaMallocHost(&h_out[i], out_frame));
    }

    /* Device memory (double-buffered) */
    uint16_t *d_y_in[2], *d_u_in[2], *d_v_in[2];
    uint8_t  *d_y_out[2], *d_u_out[2], *d_v_out[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_y_in[i], in_y_bytes));
        CHECK_CUDA(cudaMalloc(&d_u_in[i], in_uv_bytes));
        CHECK_CUDA(cudaMalloc(&d_v_in[i], in_uv_bytes));
        CHECK_CUDA(cudaMalloc(&d_y_out[i], out_y_bytes));
        CHECK_CUDA(cudaMalloc(&d_u_out[i], out_uv_bytes));
        CHECK_CUDA(cudaMalloc(&d_v_out[i], out_uv_bytes));
    }

    /* CUDA streams for pipelining */
    cudaStream_t streams[2];
    CHECK_CUDA(cudaStreamCreate(&streams[0]));
    CHECK_CUDA(cudaStreamCreate(&streams[1]));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    int buf = 0;
    int frames = 0;

    /* Set stdin/stdout to binary mode (for Windows compat, noop on Linux) */
    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);

    while (1) {
        /* Read one frame from stdin */
        size_t nread = fread(h_in[buf], 1, in_frame, stdin);
        if (nread < in_frame) break;

        cudaStream_t s = streams[buf];

        /* Upload planes to GPU */
        uint8_t* src = h_in[buf];
        CHECK_CUDA(cudaMemcpyAsync(d_y_in[buf], src, in_y_bytes,
                                   cudaMemcpyHostToDevice, s));
        CHECK_CUDA(cudaMemcpyAsync(d_u_in[buf], src + in_y_bytes, in_uv_bytes,
                                   cudaMemcpyHostToDevice, s));
        CHECK_CUDA(cudaMemcpyAsync(d_v_in[buf], src + in_y_bytes + in_uv_bytes,
                                   in_uv_bytes, cudaMemcpyHostToDevice, s));

        /* Launch tonemap kernel */
        tonemap_kernel<<<grid, block, 0, s>>>(
            d_y_in[buf], d_u_in[buf], d_v_in[buf],
            d_y_out[buf], d_u_out[buf], d_v_out[buf],
            W, H);

        /* Download results */
        uint8_t* dst = h_out[buf];
        CHECK_CUDA(cudaMemcpyAsync(dst, d_y_out[buf], out_y_bytes,
                                   cudaMemcpyDeviceToHost, s));
        CHECK_CUDA(cudaMemcpyAsync(dst + out_y_bytes, d_u_out[buf], out_uv_bytes,
                                   cudaMemcpyDeviceToHost, s));
        CHECK_CUDA(cudaMemcpyAsync(dst + out_y_bytes + out_uv_bytes,
                                   d_v_out[buf], out_uv_bytes,
                                   cudaMemcpyDeviceToHost, s));

        /* Wait for previous frame's stream to finish before writing */
        int prev = 1 - buf;
        if (frames > 0) {
            CHECK_CUDA(cudaStreamSynchronize(streams[prev]));
            fwrite(h_out[prev], 1, out_frame, stdout);
        }

        buf = 1 - buf;
        frames++;
    }

    /* Flush last frame */
    if (frames > 0) {
        int last = 1 - buf;
        CHECK_CUDA(cudaStreamSynchronize(streams[last]));
        fwrite(h_out[last], 1, out_frame, stdout);
    }

    fflush(stdout);

    /* Cleanup */
    for (int i = 0; i < 2; i++) {
        cudaFreeHost(h_in[i]);
        cudaFreeHost(h_out[i]);
        cudaFree(d_y_in[i]); cudaFree(d_u_in[i]); cudaFree(d_v_in[i]);
        cudaFree(d_y_out[i]); cudaFree(d_u_out[i]); cudaFree(d_v_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
