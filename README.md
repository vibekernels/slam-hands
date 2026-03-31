# robot-video

Annotation pipeline for egocentric robot learning videos. Produces [LeRobot v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) datasets following the [EgoVerse](https://arxiv.org/abs/2501.14394) format with:

- **SLAM camera poses** (7-DoF: translation + quaternion) via DROID-SLAM
- **3D hand poses** (21 MANO keypoints per hand) via WiLoR
- **Audio transcription** (word-level timestamps) via NVIDIA Parakeet
- **Video conversion** (iPhone HEVC HDR to SDR MP4)

## Quick start

```bash
# Recommended: uses optimized settings (~33s for 62s of 1080p30 on RTX 5090)
./annotate.sh /path/to/video.mov

# Or with explicit output directory:
./annotate.sh /path/to/video.mov /path/to/output

# Visualize results
python3 visualizer.py /path/to/output --port 8888
# Open http://localhost:8888 in a browser

# Or upload a video through the browser:
python3 visualizer.py --port 8888
# Open http://localhost:8888 and drag-and-drop a video
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
  --hand-stride N          Process every Nth frame for hands (default: 1)
  --hand-det-conf F        YOLO hand detection confidence threshold (default: 0.3)
  --fast-traj              Use fewer SLAM backend steps for faster trajectory
  --slam-backend-steps N N Backend optimization steps (default: 7 12)
  --droid-weights PATH     Path to DROID-SLAM checkpoint (default: /workspace/DROID-SLAM/checkpoints/droid.pth)
  --wilor-dir PATH         Path to WiLoR repo (default: /workspace/WiLoR)
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
- NVIDIA GPU with 11+ GB VRAM
- CUDA 12.1+ with matching drivers
- Python 3.10+
- FFmpeg 6.0+ with ffprobe

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

### DROID-SLAM

```bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM

pip install -r requirements.txt

# Build third-party CUDA extensions
pip install thirdparty/lietorch
pip install thirdparty/pytorch_scatter

# Build DROID backend
python setup.py build_ext --inplace
# or: pip install -e .
```

Download the checkpoint:
```bash
mkdir -p checkpoints
# Via the provided script:
./tools/download_model.sh
# Or manually (~16 MB):
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O checkpoints/droid.pth
```

### WiLoR

```bash
git clone https://github.com/rolpotamias/WiLoR.git
cd WiLoR

pip install -r requirements.txt
```

Key packages this installs: `ultralytics==8.1.34` (YOLO hand detector), `smplx==0.1.28`, `timm`, `pytorch-lightning`, `scikit-image`.

Download model weights:
```bash
mkdir -p pretrained_models
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P pretrained_models/
```

Download the MANO hand model (requires registration):
```bash
# Register at https://mano.is.tue.mpg.de and download MANO_RIGHT.pkl
mkdir -p mano_data
cp /path/to/MANO_RIGHT.pkl mano_data/
```

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
python3 -c "from droid import Droid; print('DROID-SLAM OK')"
python3 -c "from wilor.models import load_wilor; print('WiLoR OK')"
python3 -c "from ultralytics import YOLO; print('YOLO OK')"
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

Benchmarked on 1858 frames (62s) of 1920x1080 30fps iPhone video, RTX 5090:

| Phase | Time | Notes |
|-------|------|-------|
| SLAM + hands (split decode) | 23s | Separate processes, each decodes independently |
| Video conversion | 31s | ffmpeg libx264, fully overlapped |
| Audio transcription | 14s | Parakeet on CPU, fully overlapped |
| Dataset assembly | 1.6s | Parquet + info.json |
| **Total** | **~33s** | `./annotate.sh` (recommended settings) |

Key optimizations (34x faster than naive sequential baseline):

1. **Split decode architecture** — Parent runs SLAM with native C++ decoder (SLAM-res only). Forked child decodes video independently at full resolution for hand pose. No shared-memory transfer needed.
2. **Native GIL-free decoder** — pybind11 C++ extension using FFmpeg C API. Decode + resize runs in a native thread that never acquires the Python GIL, so DROID-SLAM's CUDA kernels run concurrently.
3. **pytorch_lightning stub** — WiLoR inherits from LightningModule but only needs nn.Module at inference. A 10-line stub replaces the full import, saving 3.4s.
4. **mmap + assign loading** — `torch.load(mmap=True)` + `load_state_dict(assign=True)` reduces WiLoR checkpoint loading from ~25s to ~0.4s.
5. **Fused Triton kernels** — Custom Triton kernels fuse SwiGLU and residual+LayerScale+LayerNorm in the ViT-H backbone, reducing elementwise overhead by 44%.
6. **Full overlap** — Video conversion (ffmpeg subprocess) and audio transcription (Parakeet on CPU) run entirely in the background, adding zero time to the critical path.

---

## Video conversion (standalone)

The video converter can also be used independently:

```bash
# Fastest (requires NVIDIA GPU with NVDEC + NVENC):
python3 convert_video.py /path/to/iphone_video.mov --fast

# Default (CPU libsvtav1, works everywhere):
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

iPhone videos are typically HEVC Main 10 with HDR (BT.2020 primaries, HLG transfer, Dolby Vision profile 8). The script performs:

- **HDR to SDR tonemapping** via zscale (BT.2020/HLG -> BT.709)
- **10-bit to 8-bit** conversion (yuv420p10le -> yuv420p)
- **Color space conversion** (BT.2020 -> BT.709 primaries, matrix, and transfer)

Non-HDR inputs skip tonemapping and just do pixel format conversion.

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
