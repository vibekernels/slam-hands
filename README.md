# robot-video

Annotation pipeline for egocentric robot learning videos. Produces [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets following the [EgoVerse](https://arxiv.org/abs/2501.14394) format with:

- **SLAM camera poses** (7-DoF: translation + quaternion) via CUDA DROID-SLAM
- **3D hand poses** (21 MANO keypoints per hand) via WiLoR
- **Audio transcription** (word-level timestamps) via NVIDIA Parakeet
- **Video conversion** (iPhone HEVC HDR to SDR MP4)

## Quick start

```bash
# Recommended: uses optimized settings (~21s for 62s of 1080p30 on RTX 5090)
./annotate.sh /path/to/video.mov

# Or with explicit output directory:
./annotate.sh /path/to/video.mov /path/to/output

# Visualize results
python3 visualizer.py /path/to/output --port 8888
# Open http://localhost:8888 in a browser

# Or upload a video through the browser:
python3 visualizer.py --port 8888
# Open http://localhost:8888 and drag-and-drop a video

# Service mode: keep models warm for fast repeated processing
python3 visualizer.py --service --port 8888
# First video: ~23s, subsequent: ~21s (vs ~39s cold start each time)

# Or use the service programmatically:
python3 pipeline_service.py --listen
# Send JSON jobs on stdin: {"video_path": "/path/to/video.mov", "output_dir": "/path/to/output"}
```

### Pipeline options

```
positional arguments:
  input                    Input video file (e.g., IMG_1443.MOV)

options:
  -o, --output-dir         Output dataset directory (default: <input>_lerobot/)
  --skip-video-convert     Skip video format conversion
  --skip-slam              Skip SLAM camera pose estimation
  --skip-hands             Skip hand pose estimation
  --skip-audio             Skip audio transcription
  --asr-model MODEL        NeMo ASR model name (default: parakeet-tdt_ctc-110m)
  --hand-stride N          Process every Nth frame for hands (default: 2)
  --hand-det-conf F        YOLO hand detection confidence threshold (default: 0.3)
  --fast-traj              Use fewer SLAM backend steps for faster trajectory
  --slam-backend-steps N N Backend optimization steps (default: 3 5)
  --device DEVICE          Torch device (default: cuda)
```

### Output format

```
output_dir/
  meta/
    info.json                                    # LeRobot v3.0 metadata
    audio.json                                   # Full transcript + word/segment timestamps
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  data/
    chunk-000/file-000.parquet                   # Per-frame annotations
  videos/
    observation.video/chunk-000/file-000.mp4     # Converted video
```

The parquet contains per-frame columns:

| Column | Shape | Description |
|--------|-------|-------------|
| `observation.slam.pose` | 7 | Camera pose [tx, ty, tz, qx, qy, qz, qw] |
| `observation.hand.{left,right}.keypoints_2d` | 42 | 21 joints x 2 (pixel coords) |
| `observation.hand.{left,right}.keypoints_3d` | 63 | 21 joints x 3 (camera frame, meters) |
| `observation.hand.{left,right}.detected` | 1 | Detection flag (0 or 1) |
| `observation.audio.transcript` | 1 | Active transcript segment (string) |

## Installation

### System requirements

- Linux (tested on Ubuntu 22.04+)
- NVIDIA GPU with 11+ GB VRAM (sm_80+: Ampere, Ada Lovelace, Blackwell)
- CUDA 12.1+ with cuDNN, cuBLAS, cuSOLVER
- Python 3.10+
- FFmpeg 6.0+ with ffprobe (built with `--enable-cuvid` for NVDEC hardware decode)

### System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake nasm git \
  ffmpeg \
  libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
  libsuitesparse-dev
```

### Python dependencies

```bash
pip install --upgrade pip

# PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# or for CUDA 12.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Pipeline dependencies
pip install pyarrow numpy opencv-python scipy tqdm pybind11
```

### CUDA DROID-SLAM

The SLAM module is a standalone CUDA binary with no PyTorch dependency:

```bash
cd robot-video/cuda_slam

# Export weights from a DROID-SLAM checkpoint (one-time)
python3 export_weights.py --checkpoint /path/to/droid.pth --out data

# Build the binary (requires CUDA 12.1+, cuDNN, cuBLAS, cuSOLVER, FFmpeg dev headers)
make
```

The DROID-SLAM checkpoint (`droid.pth`, ~16 MB) can be obtained from the [original repo](https://github.com/princeton-vl/DROID-SLAM). Weights are exported to `cuda_slam/data/weights/` as raw float32 tensors.

The binary supports NVDEC hardware video decode (`--video <path> --resize <h> <w>`) which decodes and resizes frames entirely on the GPU, or stdin mode (`--stdin <h> <w>`) for piped input.

### CUDA Hand Pose

The hand pose module is a standalone CUDA binary (YOLO detection + WiLoR ViT + MANO) with no PyTorch dependency at inference:

```bash
cd robot-video/cuda_hand

# Export weights from pretrained models (one-time, requires PyTorch)
python3 export_weights.py

# Build the binary (requires CUDA 12.1+, cuDNN, cuBLAS, FFmpeg dev headers)
make
```

The export script requires the original model weights:
```bash
# YOLO hand detector + WiLoR checkpoint
mkdir -p pretrained_models
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P pretrained_models/

# MANO hand model (requires registration at https://mano.is.tue.mpg.de)
mkdir -p mano_data
cp /path/to/MANO_RIGHT.pkl mano_data/
```

Weights are exported to `cuda_hand/data/weights/` as raw binary tensors. After export, PyTorch is only needed for SLAM weight export — the hand pipeline runs entirely in CUDA.

### Combined Pipeline (optional, faster)

A single CUDA binary that runs both SLAM and hand processing with shared NVDEC decode, eliminating GPU context switching and process overhead:

```bash
cd robot-video/cuda_pipeline
make
```

Requires the same weights as the individual SLAM and hand binaries above. The combined pipeline is automatically detected and used by `pipeline_service.py` when present. Falls back to separate binaries if not built.

### NeMo (Parakeet ASR)

```bash
pip install "nemo_toolkit[asr]"
```

The Parakeet model (`parakeet-tdt_ctc-110m`) is downloaded automatically on first use (~110M parameters, ~440 MB). No audio stream in the video? Transcription is skipped automatically.

### Native video decoder (optional, faster)

Builds a GIL-free C++ video decoder for concurrent frame extraction:

```bash
cd robot-video/native_decode
pip install -e .
```

Requires the FFmpeg dev headers installed above. Falls back to OpenCV if not available.

### Verify installation

```bash
./cuda_slam/cuda_droid --help 2>&1 | head -1 && echo "CUDA DROID-SLAM OK"
./cuda_hand/cuda_hand --help 2>&1 | head -1 && echo "CUDA Hand Pose OK"
./cuda_pipeline/cuda_pipeline --help 2>&1 | head -1 && echo "Combined Pipeline OK"
python3 -c "import nemo.collections.asr; print('NeMo ASR OK')"
python3 -c "import native_decode; print('native_decode OK')"
```

## Visualizer

Browser-based tool for inspecting output datasets. Shows the video with 2D hand keypoint overlay and a Three.js 3D scene with camera trajectory and hand skeletons.

```bash
# View an existing dataset
python3 visualizer.py /path/to/output_dataset --port 8888

# Upload mode: drag-and-drop a video in the browser to process + visualize
python3 visualizer.py --port 8888
```

Controls:
- **Space**: play/pause
- **Arrow keys**: step 1 frame
- **Shift + arrows**: step 10 frames
- **Mouse drag** on 3D panel: orbit camera
- **Scroll** on 3D panel: zoom
- Checkboxes: toggle left/right hands, bones, 3D hands, camera follow

## Performance

Benchmarked on 1829 frames (61s) of 1920x1080 30fps iPhone video, RTX 5090:

| Phase | Time | Notes |
|-------|------|-------|
| SLAM + hand pose (combined) | ~17.3s | Single CUDA binary, single-pass decode, interleaved SLAM + hands |
| &nbsp;&nbsp; YOLO hand detection | ~1.6s | YOLOv8m FP16, 8-frame batches |
| &nbsp;&nbsp; WiLoR hand pose | ~7.0s | ViT-H FP16 + MANO + RefineNet |
| SLAM backend optimization | ~0.5s | Distance-based edge selection + bundle adjustment |
| Audio transcription | ~8s | Parakeet on CPU, fully overlapped |
| Video conversion (h264_nvenc) | ~3s | NVENC runs concurrently (dedicated HW encoder) |
| Dataset assembly + Python overhead | ~2.5s | Parquet + info.json + result parsing |
| **Total** | **~21s** | Service mode (warm), combined binary, all outputs included |

Key optimizations:

1. **Combined CUDA binary** (`cuda_pipeline`) — SLAM and hand processing run in a single process with shared NVDEC video decode. Single-pass architecture: each frame is decoded once, downscaled for SLAM and converted at full resolution for hand processing in the same iteration. Eliminates redundant video decode and GPU context switching. In service mode, SLAM state resets between videos without reloading weights (~2s savings per video).
2. **CUDA DROID-SLAM** — Pure CUDA reimplementation of DROID-SLAM (cuDNN convolutions, cuBLAS correlation, cuSOLVER bundle adjustment). No PyTorch dependency at inference. Matches PyTorch DROID-SLAM accuracy on TUM RGB-D benchmarks (ATE RMSE 0.021m on fr1_desk). FP16 tensor core encoders (fnet/cnet) and FP16 correlation via `cublasGemmEx`/`cublasGemmBatchedEx` with FP32 accumulation. Backend uses distance-based co-visibility edges for loop closure, matching the original implementation. NVDEC hardware video decode with GPU resize eliminates CPU decode entirely. A sliding frontend window (default 15 keyframes) prevents quadratic BA cost scaling on long videos.
3. **CUDA hand pose** — Pure CUDA reimplementation of the full hand pose pipeline: YOLOv8m detection, WiLoR ViT-H, MANO skinning, and RefineNet. No PyTorch dependency at inference. YOLO runs in FP16 with tensor cores and 8-frame batching. WiLoR ViT uses FP16 GEMMs with FP32 accumulation via `cublasGemmEx`. GPU-side ViT output decode (cuBLAS strided batched GEMMs) eliminates 103MB D2H transfer per batch. All GPU buffers pre-allocated at startup; MANO and RefineNet weights cached on host.
4. **NVDEC hardware video decode** — Single NVDEC decode shared between SLAM and hand processing. Decoded frames arrive directly in GPU memory (P010/NV12 format), converted to BGR by custom CUDA kernels at both SLAM resolution and full resolution in the same iteration. Eliminates CPU decode and host-to-device transfer entirely.
5. **Concurrent NVENC video conversion** — NVENC uses a dedicated hardware encoder separate from CUDA cores, so it runs concurrently with SLAM and hand detection. Started at the beginning of the pipeline rather than waiting for inference to complete.
6. **Service mode** — `pipeline_service.py` keeps models loaded across videos via a persistent `cuda_pipeline --listen` worker. One-time ~9s startup (WiLoR weight loading), then each video reuses warm GPU buffers. SLAM state resets in-place without reallocating memory or reloading weights. Use `--service` flag with the visualizer or `--listen` for batch processing.

---

## Video conversion (standalone)

The video converter can also be used independently:

```bash
# Fastest (requires NVIDIA GPU with NVDEC + NVENC + scale_cuda):
python3 convert_video.py /path/to/iphone_video.mov --fast

# Default (auto-selects: h264_nvenc if available, else libx264 ultrafast):
python3 convert_video.py /path/to/iphone_video.mov
```

Output: `<input>_lerobot.mp4` in the same directory, or specify `-o /path/to/output.mp4`.

### Video conversion options

```
-o, --output     Output file path (default: <input>_lerobot.mp4)
--fast           Fastest: NVDEC->scale_cuda->h264_nvenc, all on GPU (10x speedup)
--no-gpu         Disable NVDEC hardware-accelerated decoding
--quality N      CRF value (default: 30, lower = better quality)
--gop N          GOP size (default: 2)
--preset N       libsvtav1 speed preset (default: 12, fastest)
--nvenc          Use NVENC AV1 hardware encoder with zscale tonemapping
--gpu-pipeline   NVDEC->CUDA tonemap->NVENC (requires gpu_convert binary)
--bitrate N      Target bitrate in kbps for NVENC modes (auto-estimated if not set)
```

### LeRobotDataset v3.0 video compatibility

The following are **required** for LeRobotDataset v3.0 compatibility (lerobot >= 0.4.0):

| Requirement     | Value       | Why |
|----------------|-------------|-----|
| Container      | MP4         | Required by lerobot's video loading |
| Pixel format   | yuv420p     | Required (yuv444p auto-downgraded) |
| GOP size       | 2           | Every other frame is a keyframe, for fast random frame access during training |
| Audio          | None        | Not needed for robot datasets |

The following are **configurable** -- lerobot supports multiple codecs and quality settings:

| Parameter      | lerobot default | Valid options |
|----------------|----------------|--------------|
| Codec          | libsvtav1 (AV1) | h264, hevc, libsvtav1, or hardware variants (h264_nvenc, hevc_nvenc, h264_videotoolbox, etc.) |
| Quality (CRF)  | 30             | Any CRF value |
| movflags       | +faststart     | Recommended for streaming |

Output must be decodable by PyAV and/or torchcodec (LeRobotDataset's video backends).

### iPhone-specific handling

iPhone videos are typically HEVC Main 10 with HDR (BT.2020 primaries, HLG transfer, Dolby Vision profile 8). Conversion behavior depends on the codec path:

- **NVENC (default on GPU systems)**: Skips tonemapping — HLG's backwards-compatible design means the 10-bit→8-bit truncation looks natural on SDR displays. Converts to yuv420p + h264_nvenc.
- **libx264 fallback (no NVENC)**: Full HDR→SDR tonemapping via zscale (BT.2020/HLG → BT.709), 10-bit→8-bit, color space conversion.
- **`--nvenc` flag**: NVENC AV1 with explicit zscale tonemapping.
- **`--fast` flag**: Full GPU pipeline (NVDEC → scale_cuda → h264_nvenc), no tonemapping.

Non-HDR inputs skip tonemapping in all paths and just do pixel format conversion.

### Video conversion performance

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

See [OPTIMIZATION.md](OPTIMIZATION.md) for detailed optimization notes.

### Optional: AVX-512 SVT-AV1

On CPUs with AVX-512 support (Zen 4/5, Ice Lake+), run `sudo bash setup.sh` to rebuild SVT-AV1 with AVX-512 enabled for ~10% faster encoding. Requires cmake, nasm, and build-essential.
