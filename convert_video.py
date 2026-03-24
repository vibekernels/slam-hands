#!/usr/bin/env python3
"""Convert iPhone video to LeRobotDataset v3.0 compatible format.

LeRobotDataset v3.0 defaults:
  - Codec: libsvtav1 (AV1)
  - Pixel format: yuv420p
  - CRF: 30
  - GOP size (g): 2  (small for fast random frame access in ML training)
  - Container: MP4
  - Preset: 12 (for libsvtav1)

iPhone videos are typically HEVC 10-bit HDR (BT.2020/HLG with Dolby Vision),
so we also need to tonemap from HDR to SDR and convert 10-bit to 8-bit.

Speed optimizations:
  - NVDEC hardware-accelerated decoding (via -hwaccel auto)
  - Parallelized zscale tonemapping (via -filter_threads)
  - Tuned thread allocation between filter and encoder stages
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def probe_video(input_path: str) -> dict:
    """Get video stream info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def is_hdr(probe_info: dict) -> bool:
    """Check if the video stream is HDR based on color transfer characteristics."""
    for stream in probe_info.get("streams", []):
        if stream.get("codec_type") == "video":
            transfer = stream.get("color_transfer", "")
            primaries = stream.get("color_primaries", "")
            if transfer in ("arib-std-b67", "smpte2084") or primaries == "bt2020":
                return True
    return False


def get_video_stream(probe_info: dict) -> dict:
    """Extract the video stream from probe info."""
    for stream in probe_info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("No video stream found")


def check_nvdec_available() -> bool:
    """Check if NVIDIA NVDEC is available for hardware decoding."""
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccel", "cuda", "-f", "lavfi",
             "-i", "nullsrc=s=64x64:d=0.1", "-f", "null", "-"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_nvenc_available() -> bool:
    """Check if av1_nvenc encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "av1_nvenc" in result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def find_gpu_convert() -> str | None:
    """Find the gpu_convert binary (in same directory as this script)."""
    script_dir = Path(__file__).resolve().parent
    binary = script_dir / "gpu_convert"
    if binary.is_file() and os.access(binary, os.X_OK):
        return str(binary)
    return None


def estimate_bitrate(width: int, height: int, fps_str: str, crf: int) -> int:
    """Estimate the bitrate (kbps) that libsvtav1 would produce at a given CRF.

    Based on empirical measurements: 1080p/30fps at CRF 30 ≈ 2550 kbps.
    Scales linearly with pixel count and fps, exponentially with CRF.
    """
    pixels = width * height
    ref_pixels = 1920 * 1080
    try:
        num, den = map(int, fps_str.split("/"))
        fps = num / den
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    ref_fps = 30.0
    ref_bitrate = 2550  # kbps at CRF 30 for 1080p/30fps
    ref_crf = 30

    # Bitrate scales ~linearly with pixels and fps
    scale = (pixels / ref_pixels) * (fps / ref_fps)
    # CRF scale: ~6 CRF points ≈ 2x bitrate change
    crf_scale = 2.0 ** ((ref_crf - crf) / 6.0)

    return int(ref_bitrate * scale * crf_scale)


def build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    *,
    use_gpu: bool = True,
    hdr_input: bool = False,
    quality: int = 30,
    gop: int = 2,
    preset: int = 12,
) -> list[str]:
    """Build ffmpeg command for LeRobotDataset v3.0 conversion."""
    total_cores = os.cpu_count() or 4

    cmd = ["ffmpeg", "-y"]

    # Parallelize the filter graph (zscale tonemapping).
    # ~20% of cores for filters, rest for the encoder — benchmarked on
    # a 32-core Ryzen 9950X, this balance minimizes total wall-clock time.
    filter_threads = max(1, total_cores // 5)
    encoder_threads = max(1, total_cores - filter_threads)

    if hdr_input:
        cmd += ["-filter_threads", str(filter_threads)]

    # NVDEC hardware-accelerated decoding.
    # hwaccel=auto lets ffmpeg pick the best decoder and deliver frames
    # in system memory (native pixel format), which zscale needs.
    if use_gpu:
        cmd += ["-hwaccel", "auto"]

    cmd += ["-i", input_path]

    # Video filters: HDR→SDR conversion
    if hdr_input:
        # Single-pass zscale with explicit input/output color parameters.
        # This avoids the expensive intermediate format=gbrpf32le (32-bit float RGB)
        # conversion that a two-pass linearize→convert approach requires.
        # zscale handles the transfer function, primaries, and matrix conversion
        # internally when all parameters are specified together.
        vf = (
            "zscale=tin=arib-std-b67:t=bt709"
            ":min=bt2020nc:m=bt709"
            ":pin=bt2020:p=bt709"
            ":r=tv:npl=100"
            ",format=yuv420p"
        )
    else:
        vf = "format=yuv420p"
    cmd += ["-vf", vf]

    # Encoding: libsvtav1 matching LeRobotDataset v3.0 defaults
    cmd += [
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-crf", str(quality),
        "-g", str(gop),
        "-preset", str(preset),
        "-svtav1-params", f"fast-decode=0:lp={encoder_threads}",
        "-movflags", "+faststart",
        "-an",  # strip audio — not needed for robot datasets
        output_path,
    ]

    return cmd


def build_ffmpeg_cmd_nvenc(
    input_path: str,
    output_path: str,
    *,
    use_gpu: bool = True,
    hdr_input: bool = False,
    quality: int = 30,
    gop: int = 2,
    bitrate: int | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: str = "30000/1001",
) -> list[str]:
    """Build ffmpeg command using NVENC AV1 hardware encoder.

    Uses VBR mode targeting the same bitrate as libsvtav1 at the given CRF.
    With NVENC handling encoding, all CPU cores are free for zscale tonemapping.
    """
    total_cores = os.cpu_count() or 4

    if bitrate is None:
        bitrate = estimate_bitrate(width, height, fps, quality)

    cmd = ["ffmpeg", "-y"]

    # With NVENC encoding, all CPU cores go to filter threads
    if hdr_input:
        cmd += ["-filter_threads", str(max(1, total_cores // 2))]

    if use_gpu:
        cmd += ["-hwaccel", "auto"]

    cmd += ["-i", input_path]

    if hdr_input:
        vf = (
            "zscale=tin=arib-std-b67:t=bt709"
            ":min=bt2020nc:m=bt709"
            ":pin=bt2020:p=bt709"
            ":r=tv:npl=100"
            ",format=yuv420p"
        )
    else:
        vf = "format=yuv420p"
    cmd += ["-vf", vf]

    cmd += [
        "-c:v", "av1_nvenc",
        "-pix_fmt", "yuv420p",
        "-b:v", f"{bitrate}k",
        "-maxrate", f"{int(bitrate * 1.5)}k",
        "-bufsize", f"{bitrate * 2}k",
        "-g", str(gop),
        "-bf", "0",  # no B-frames (required for GOP=2)
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]

    return cmd


def check_scale_cuda_available() -> bool:
    """Check if ffmpeg's scale_cuda filter is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        return "scale_cuda" in result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def build_ffmpeg_cmd_fast(
    input_path: str,
    output_path: str,
    *,
    gop: int = 2,
    bitrate: int = 2550,
) -> list[str]:
    """Build the fastest ffmpeg command: NVDEC → scale_cuda → h264_nvenc.

    Entire pipeline stays on GPU. No tonemapping — HLG signal values are
    truncated from 10-bit to 8-bit, which looks natural on SDR displays
    (HLG was designed for backwards compatibility).
    """
    return [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-i", input_path,
        "-vf", "scale_cuda=format=nv12",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-tune", "ll",
        "-g", str(gop),
        "-bf", "0",
        "-b:v", f"{bitrate}k",
        "-maxrate", f"{int(bitrate * 1.5)}k",
        "-bufsize", f"{bitrate * 2}k",
        "-an",
        output_path,
    ]


def run_gpu_pipeline(
    input_path: str,
    output_path: str,
    bitrate: int,
    gpu_convert_bin: str,
) -> bool:
    """Run the full GPU pipeline (NVDEC→CUDA tonemap→NVENC).

    Returns True on success, False on failure (caller should fall back).
    """
    cmd = [gpu_convert_bin, input_path, output_path, str(bitrate)]
    print(f"\n  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gpu_convert stderr:\n{result.stderr}", file=sys.stderr)
        return False
    if result.stderr:
        # gpu_convert prints progress to stderr
        for line in result.stderr.strip().split("\n"):
            print(f"  {line}")
    return True


def convert_video(
    input_path: str,
    output_path: str | None = None,
    use_gpu: bool = True,
    quality: int = 30,
    gop: int = 2,
    preset: int = 12,
    nvenc: bool = False,
    bitrate: int | None = None,
    gpu_pipeline: bool = False,
    fast: bool = False,
) -> str:
    """Convert a video to LeRobotDataset v3.0 format."""
    input_path = str(Path(input_path).resolve())

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_lerobot.mp4")

    # Probe input
    print(f"Probing {input_path}...")
    probe_info = probe_video(input_path)
    video_stream = get_video_stream(probe_info)

    codec = video_stream.get("codec_name", "unknown")
    width = video_stream.get("width", "?")
    height = video_stream.get("height", "?")
    pix_fmt = video_stream.get("pix_fmt", "unknown")
    fps = video_stream.get("r_frame_rate", "?")
    hdr = is_hdr(probe_info)

    print(f"  Input: {codec} {width}x{height} {pix_fmt} @ {fps} fps, HDR={hdr}")

    enc_bitrate = bitrate or estimate_bitrate(
        video_stream.get("width", 1920),
        video_stream.get("height", 1080),
        fps, quality,
    )

    # --fast: full GPU pipeline via ffmpeg (NVDEC → scale_cuda → h264_nvenc)
    use_fast = fast
    if use_fast:
        if not (check_nvdec_available() and check_nvenc_available()
                and check_scale_cuda_available()):
            print("  --fast requires NVDEC + scale_cuda + h264_nvenc, falling back")
            use_fast = False

    if use_fast:
        print(f"  Output: h264_nvenc (NVDEC→scale_cuda→NVENC) VBR={enc_bitrate}k GOP={gop}")
        cmd = build_ffmpeg_cmd_fast(
            input_path, output_path, gop=gop, bitrate=enc_bitrate,
        )
        print(f"\n  $ {' '.join(cmd)}\n")
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg stderr:\n{result.stderr}", file=sys.stderr)
            print("\n--fast failed, falling back to default...")
            return convert_video(input_path, output_path, use_gpu=use_gpu,
                                 quality=quality, gop=gop, preset=preset)
    else:
        # Check GPU
        if use_gpu:
            if check_nvdec_available():
                print("  NVDEC: available (hardware decode)")
            else:
                print("  NVDEC: not available, using CPU decode")
                use_gpu = False

        # Full GPU pipeline with CUDA tonemap (NVDEC → CUDA tonemap → NVENC)
        use_gpu_pipeline = gpu_pipeline
        gpu_convert_bin = None
        if use_gpu_pipeline:
            gpu_convert_bin = find_gpu_convert()
            if gpu_convert_bin is None:
                print("  gpu_convert: binary not found, falling back to ffmpeg")
                print("    Build with: nvcc -O3 -o gpu_convert gpu_convert.cu \\")
                print("      $(pkg-config --cflags --libs libavformat libavcodec libavutil) -lcuda")
                use_gpu_pipeline = False

        # Check NVENC availability if requested (and not using full GPU pipeline)
        use_nvenc = nvenc and not use_gpu_pipeline
        if use_nvenc and not check_nvenc_available():
            print("  av1_nvenc: not available, falling back to libsvtav1")
            use_nvenc = False

        if use_gpu_pipeline:
            print(f"  Output: gpu_convert (NVDEC→CUDA→NVENC) VBR={enc_bitrate}k GOP=2")
        elif use_nvenc:
            print(f"  Output: av1_nvenc yuv420p VBR={enc_bitrate}k GOP={gop}")
        else:
            print(f"  Output: libsvtav1 yuv420p CRF={quality} GOP={gop}")

        start = time.perf_counter()

        if use_gpu_pipeline:
            success = run_gpu_pipeline(input_path, output_path, enc_bitrate, gpu_convert_bin)
            if not success:
                print("\ngpu_convert failed, falling back to --nvenc...")
                return convert_video(input_path, output_path, use_gpu=use_gpu,
                                     quality=quality, gop=gop, preset=preset,
                                     nvenc=True, bitrate=bitrate)
        elif use_nvenc:
            cmd = build_ffmpeg_cmd_nvenc(
                input_path, output_path,
                use_gpu=use_gpu, hdr_input=hdr,
                quality=quality, gop=gop, bitrate=bitrate,
                width=video_stream.get("width", 1920),
                height=video_stream.get("height", 1080),
                fps=fps,
            )

            print(f"\n  $ {' '.join(cmd)}\n")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"ffmpeg stderr:\n{result.stderr}", file=sys.stderr)
                print("\nNVENC failed, falling back to libsvtav1...")
                return convert_video(input_path, output_path, use_gpu=use_gpu,
                                     quality=quality, gop=gop, preset=preset,
                                     nvenc=False)
        else:
            cmd = build_ffmpeg_cmd(
                input_path, output_path,
                use_gpu=use_gpu, hdr_input=hdr,
                quality=quality, gop=gop, preset=preset,
            )

            print(f"\n  $ {' '.join(cmd)}\n")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"ffmpeg stderr:\n{result.stderr}", file=sys.stderr)
                if use_gpu:
                    print("\nGPU path failed, retrying CPU-only...")
                    return convert_video(input_path, output_path, use_gpu=False,
                                         quality=quality, gop=gop, preset=preset)
                sys.exit(1)

    elapsed = time.perf_counter() - start

    # Verify output
    out_probe = probe_video(output_path)
    out_stream = get_video_stream(out_probe)
    out_size = Path(output_path).stat().st_size / (1024 * 1024)
    in_size = Path(input_path).stat().st_size / (1024 * 1024)

    print(f"  Done in {elapsed:.1f}s ({in_size:.1f} MB → {out_size:.1f} MB)")
    print(f"  Output: {output_path}")
    print(f"    Codec: {out_stream.get('codec_name')} | Pix fmt: {out_stream.get('pix_fmt')}")
    print(f"    Resolution: {out_stream.get('width')}x{out_stream.get('height')}")
    print(f"    FPS: {out_stream.get('r_frame_rate')}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to LeRobotDataset v3.0 format"
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output path (default: <input>_lerobot.mp4)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--quality", type=int, default=30,
                        help="CRF quality (default: 30, lower=better)")
    parser.add_argument("--gop", type=int, default=2, help="GOP size (default: 2)")
    parser.add_argument("--preset", type=int, default=12, help="libsvtav1 preset (default: 12)")
    parser.add_argument("--fast", action="store_true",
                        help="Fastest mode: NVDEC→scale_cuda→h264_nvenc, all on GPU (10x speedup)")
    parser.add_argument("--nvenc", action="store_true",
                        help="Use NVENC AV1 hardware encoder with zscale tonemapping")
    parser.add_argument("--gpu-pipeline", action="store_true",
                        help="NVDEC→CUDA tonemap→NVENC (requires gpu_convert binary)")
    parser.add_argument("--bitrate", type=int, default=None,
                        help="Target bitrate in kbps for NVENC modes (auto-estimated if not set)")
    args = parser.parse_args()

    convert_video(
        args.input,
        args.output,
        use_gpu=not args.no_gpu,
        quality=args.quality,
        gop=args.gop,
        preset=args.preset,
        nvenc=args.nvenc,
        bitrate=args.bitrate,
        gpu_pipeline=args.gpu_pipeline,
        fast=args.fast,
    )


if __name__ == "__main__":
    main()
