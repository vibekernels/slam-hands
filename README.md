# robot-video

Optimized video processing for robot training data generation.

Converts iPhone videos (HEVC 10-bit HDR / Dolby Vision) to [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) compatible format.

## Quick start

```bash
# Fastest (requires NVIDIA GPU with NVDEC + NVENC):
python3 convert_video.py /path/to/iphone_video.mov --fast

# Default (CPU libsvtav1, works everywhere):
python3 convert_video.py /path/to/iphone_video.mov
```

Output: `<input>_lerobot.mp4` in the same directory, or specify `-o /path/to/output.mp4`.

### Options

```
-o, --output     Output file path (default: <input>_lerobot.mp4)
--fast           Fastest: NVDEC→scale_cuda→h264_nvenc, all on GPU (10x speedup)
--no-gpu         Disable NVDEC hardware-accelerated decoding
--quality N      CRF value (default: 30, lower = better quality)
--gop N          GOP size (default: 2)
--preset N       libsvtav1 speed preset (default: 12, fastest)
--nvenc          Use NVENC AV1 hardware encoder with zscale tonemapping
--gpu-pipeline   NVDEC→CUDA tonemap→NVENC (requires gpu_convert binary)
--bitrate N      Target bitrate in kbps for NVENC modes (auto-estimated if not set)
```

## LeRobotDataset v3.0 compatibility

The following are **required** for LeRobotDataset v3.0 compatibility (lerobot ≥ 0.4.0):

| Requirement     | Value       | Why |
|----------------|-------------|-----|
| Container      | MP4         | Required by lerobot's video loading |
| Pixel format   | yuv420p     | Required (yuv444p auto-downgraded) |
| GOP size       | 2           | Every other frame is a keyframe, for fast random frame access during training |
| Audio          | None        | Not needed for robot datasets |

The following are **configurable** — lerobot supports multiple codecs and quality settings:

| Parameter      | lerobot default | Valid options |
|----------------|----------------|--------------|
| Codec          | libsvtav1 (AV1) | h264, hevc, libsvtav1, or hardware variants (h264_nvenc, hevc_nvenc, h264_videotoolbox, etc.) |
| Quality (CRF)  | 30             | Any CRF value |
| movflags       | +faststart     | Recommended for streaming |

Output must be decodable by PyAV and/or torchcodec (LeRobotDataset's video backends).

## iPhone-specific handling

iPhone videos are typically HEVC Main 10 with HDR (BT.2020 primaries, HLG transfer, Dolby Vision profile 8). The script performs:

- **HDR to SDR tonemapping** via zscale (BT.2020/HLG → BT.709)
- **10-bit to 8-bit** conversion (yuv420p10le → yuv420p)
- **Color space conversion** (BT.2020 → BT.709 primaries, matrix, and transfer)

Non-HDR inputs skip tonemapping and just do pixel format conversion.

## Performance

Benchmarked on a 60-second 1080p/30fps iPhone 16 Pro video (HEVC 10-bit HDR, 51 MB input) with a Ryzen 9 9950X (32 threads) and RTX 5090:

| Version | Wall time | Output size | Speedup |
|---------|-----------|-------------|---------|
| Naive (CPU decode, two-pass tonemap) | 20.5s | 19.3 MB | 1.0x |
| + NVDEC hardware decode | 16.8s | 19.3 MB | 1.2x |
| + Parallel filter threads | 12.3s | 19.2 MB | 1.7x |
| + Single-pass zscale | 9.3s | 19.2 MB | 2.2x |
| + SVT-AV1 AVX-512 | 8.5s | 19.2 MB | 2.4x |
| + NVENC VBR (`--nvenc`) | 4.5s | 19.3 MB | 4.6x |
| + Full GPU pipeline (`gpu_convert --tonemap`) | 2.5s | 19.3 MB | 8.2x |
| + H.264 NVENC, no tonemap (`--fast`) | **2.0s** | **18.8 MB** | **10.2x** |

### Optimization details

Seven techniques stack to achieve a 10.2x speedup over the naive approach:

**1. NVDEC hardware-accelerated decoding** (`-hwaccel auto`)

Offloads HEVC 10-bit decoding to the GPU, freeing CPU cores for tonemapping and encoding.

**2. Single-pass zscale tonemapping**

The standard HDR→SDR filter chain uses an expensive intermediate 32-bit float RGB conversion:

```
zscale=t=linear:npl=100 → format=gbrpf32le → zscale=p=bt709:t=bt709:m=bt709 → format=yuv420p
```

Specifying both input and output color parameters in a single zscale invocation lets zscale handle the conversion internally, avoiding the float32 intermediate entirely:

```
zscale=tin=arib-std-b67:t=bt709:min=bt2020nc:m=bt709:pin=bt2020:p=bt709:r=tv:npl=100 → format=yuv420p
```

This cut tonemapping time roughly in half.

**3. Tuned thread allocation** (`-filter_threads` + SVT-AV1 `lp`)

The zscale filter and libsvtav1 encoder both compete for CPU cores. Allocating ~20% of cores to filter threads and ~80% to the encoder (via SVT-AV1's `lp` parameter) minimizes contention. The script auto-tunes this based on `os.cpu_count()`.

**4. SVT-AV1 with AVX-512** (setup.sh)

The default Ubuntu/Debian `libsvtav1` package is compiled with AVX2 only. Rebuilding SVT-AV1 1.7.0 from source with `-DENABLE_AVX512=ON` enables AVX-512 SIMD paths on supported CPUs (Zen 4/5, Ice Lake+), giving ~10% faster encoding. The included `setup.sh` script automates this — it builds and installs a drop-in replacement library with the same SONAME (`libSvtAv1Enc.so.1`), so no ffmpeg rebuild is needed.

```bash
sudo bash setup.sh
```

Note: SVT-AV1 at preset 12 with GOP=2 is pipeline-bound, not compute-bound — it peaks at ~16 cores and actually gets slower with more. The AVX-512 gain comes from wider SIMD on the critical path, not from using more cores.

**5. NVENC hardware encoding** (`--nvenc`)

Replaces the CPU-based libsvtav1 encoder with NVIDIA's AV1 hardware encoder (av1_nvenc). Since NVENC runs on the GPU's dedicated encoding ASIC, all CPU cores are freed for zscale tonemapping (which becomes the bottleneck at ~4.3s with 16 threads).

NVENC uses VBR (variable bitrate) mode targeting the same bitrate that libsvtav1 would produce at the given CRF. Quality comparison at matched bitrate: **PSNR 49 dB / SSIM 0.998** vs libsvtav1 — visually identical.

```bash
python3 convert_video.py /path/to/video.mov --nvenc
```

**6. Full GPU pipeline** (`gpu_convert`)

Custom C/CUDA program that keeps frames on the GPU throughout: NVDEC decode → CUDA tonemap kernel → NVENC encode. Eliminates the CPU zscale bottleneck entirely (zscale was ~4.3s of the 4.5s total). Uses NVENC preset p1 with low-latency tuning and zero-delay buffering, plus a pool of NV12 frames to pipeline tonemap and encode across frames.

The CUDA kernel performs the full HDR→SDR conversion: P010 (10-bit BT.2020/HLG) → NV12 (8-bit BT.709), matching zscale's color science (HLG inverse OETF → luma-weighted OOTF → BT.2020→BT.709 gamut mapping → BT.1886 inverse EOTF). Quality: **SSIM 0.984** vs libsvtav1 reference at matched file size.

```bash
# Build
nvcc -O3 -o gpu_convert gpu_convert.cu \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil) -lcuda
# Run via convert_video.py (auto-fallback if binary missing)
python3 convert_video.py /path/to/video.mov --gpu-pipeline
# Or run directly
./gpu_convert input.mov output.mp4 [bitrate_kbps] [--tonemap]
```

**7. H.264 + skip tonemapping** (`--fast`)

Uses ffmpeg's `scale_cuda` filter for GPU-side P010→NV12 format conversion and `h264_nvenc` for encoding. The entire pipeline (NVDEC decode → scale_cuda → h264_nvenc) stays on the GPU with zero CPU involvement.

Tonemapping is skipped: HLG signal values are truncated from 10-bit to 8-bit, which looks natural on SDR displays because HLG was designed for backwards compatibility. H.264 is faster than AV1 on NVENC's encoding ASIC. Combined, this brings the wall time from 2.5s to 2.0s.

```bash
python3 convert_video.py /path/to/video.mov --fast
```

### Approaches that did not help

| Approach | Result | Why |
|----------|--------|-----|
| Parallel segment encoding | 2x slower | Per-segment ffmpeg startup + seek overhead dominates for a 60s video |
| Pipe architecture (two ffmpeg processes) | ~Same speed | Intermediate codec overhead + pipe bandwidth bottleneck |
| CPU pinning with taskset | ~Same speed | OS scheduler already handles this reasonably |
| SVT-AV1 2.3.0 (version upgrade) | ~Same speed | ABI break (different SONAME), API changes, no measurable gain at preset 12 |
| av1_nvenc constqp (CRF-like) | 2x faster but 2.5x larger files | NVENC's constqp mode can't match libsvtav1's compression at GOP=2 (solved by using VBR with bitrate targeting instead) |
| GPU tonemapping (libplacebo/Vulkan) | Failed to initialize | Vulkan loader 1.3.275 too old for NVIDIA driver 580 |
| OpenCL tonemapping (tonemap_opencl) | Slower, wrong colors | 30s, 2x larger output — kernel not optimized for HLG transfer function |
| PyTorch CUDA tonemapping | 4x slower | 35s at 52 fps — per-frame Python/tensor overhead dominates vs zscale's 420 fps |
| Custom CUDA tonemapping kernel | Slower via pipes | GPU compute is instant but pipe I/O for 17GB of raw frames (11GB in + 6GB out) takes 3s — comparable to zscale in-process |
| Reduced SVT-AV1 lookahead | ~Same speed | No measurable difference at preset 12 with GOP=2 |
| GOP-level parallel encoding | Slower | SVT-AV1 instances contend heavily on shared resources; 2×16 cores = 8.8s vs single 26 cores = 7.7s. Loses ffmpeg's internal pipeline overlap. |
| PGO-optimized SVT-AV1 | ~Same speed | Built with -fprofile-generate/-fprofile-use, no measurable improvement at preset 12 |

### Ideas not yet tried

| Idea | Expected gain | Description |
|------|--------------|-------------|
| Fix Vulkan for libplacebo | ~1s | Build vulkan-loader from source to fix version mismatch, enabling GPU-accelerated tonemapping as an ffmpeg filter. |
| SVT-AV1 GOP=2 fast path | Unknown | Patch SVT-AV1 to skip unnecessary analysis stages (temporal prediction, lookahead) for keyframes, which are half of all frames at GOP=2. |

## Requirements

- ffmpeg with libsvtav1 and libzimg (zscale filter)
- NVIDIA GPU + drivers for NVDEC hardware decoding (optional, falls back to CPU)
- Python 3.10+

### Optional: AVX-512 SVT-AV1

On CPUs with AVX-512 support (Zen 4/5, Ice Lake+), run `sudo bash setup.sh` to rebuild SVT-AV1 with AVX-512 enabled for ~10% faster encoding. Requires cmake, nasm, and build-essential.
