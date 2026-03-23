# robot-video

Optimized video processing for robot training data generation.

Converts iPhone videos (HEVC 10-bit HDR / Dolby Vision) to [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) compatible format.

## Quick start

```bash
python3 convert_video.py /path/to/iphone_video.mov
```

Output: `<input>_lerobot.mp4` in the same directory, or specify `-o /path/to/output.mp4`.

### Options

```
-o, --output     Output file path (default: <input>_lerobot.mp4)
--no-gpu         Disable NVDEC hardware-accelerated decoding
--quality N      CRF value (default: 30, lower = better quality)
--gop N          GOP size (default: 2)
--preset N       libsvtav1 speed preset (default: 12, fastest)
```

## Output format

Matches LeRobotDataset v3.0 defaults (lerobot 0.5.0):

| Parameter    | Value       |
|-------------|-------------|
| Codec       | AV1 (libsvtav1) |
| Container   | MP4         |
| Pixel format| yuv420p     |
| CRF         | 30          |
| GOP size    | 2 (every other frame is a keyframe, for fast random access) |
| Audio       | Stripped    |
| movflags    | +faststart  |

Output is decodable by both PyAV and torchcodec (LeRobotDataset's video backends).

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
| + SVT-AV1 AVX-512 | **8.5s** | **19.2 MB** | **2.4x** |

### Optimization details

Four techniques stack to achieve a 2.4x speedup over the naive approach, with no change to output codec, quality, or file size:

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

### Approaches that did not help

| Approach | Result | Why |
|----------|--------|-----|
| Parallel segment encoding | 2x slower | Per-segment ffmpeg startup + seek overhead dominates for a 60s video |
| Pipe architecture (two ffmpeg processes) | ~Same speed | Intermediate codec overhead + pipe bandwidth bottleneck |
| CPU pinning with taskset | ~Same speed | OS scheduler already handles this reasonably |
| SVT-AV1 2.3.0 (version upgrade) | ~Same speed | ABI break (different SONAME), API changes, no measurable gain at preset 12 |
| av1_nvenc GPU encoding | 2x faster but 2.5x larger files | NVENC can't match libsvtav1's compression efficiency at GOP=2 |
| GPU tonemapping (libplacebo/Vulkan) | Failed to initialize | Vulkan loader 1.3.275 too old for NVIDIA driver 580 |
| OpenCL tonemapping | Failed to initialize | No OpenCL ICD installed on the system |
| PyTorch CUDA tonemapping | ~Same speed | CPU↔GPU transfer overhead negated compute gains |
| Reduced SVT-AV1 lookahead | ~Same speed | No measurable difference at preset 12 with GOP=2 |

## Requirements

- ffmpeg with libsvtav1 and libzimg (zscale filter)
- NVIDIA GPU + drivers for NVDEC hardware decoding (optional, falls back to CPU)
- Python 3.10+

### Optional: AVX-512 SVT-AV1

On CPUs with AVX-512 support (Zen 4/5, Ice Lake+), run `sudo bash setup.sh` to rebuild SVT-AV1 with AVX-512 enabled for ~10% faster encoding. Requires cmake, nasm, and build-essential.
