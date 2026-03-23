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


def convert_video(
    input_path: str,
    output_path: str | None = None,
    use_gpu: bool = True,
    quality: int = 30,
    gop: int = 2,
    preset: int = 12,
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
    print(f"  Output: libsvtav1 yuv420p CRF={quality} GOP={gop}")

    # Check GPU
    if use_gpu:
        if check_nvdec_available():
            print("  NVDEC: available (hardware decode)")
        else:
            print("  NVDEC: not available, using CPU decode")
            use_gpu = False

    cmd = build_ffmpeg_cmd(
        input_path, output_path,
        use_gpu=use_gpu, hdr_input=hdr,
        quality=quality, gop=gop, preset=preset,
    )

    print(f"\n  $ {' '.join(cmd)}\n")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"ffmpeg stderr:\n{result.stderr}", file=sys.stderr)
        if use_gpu:
            print("\nGPU path failed, retrying CPU-only...")
            return convert_video(input_path, output_path, use_gpu=False,
                                 quality=quality, gop=gop, preset=preset)
        sys.exit(1)

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
    args = parser.parse_args()

    convert_video(
        args.input,
        args.output,
        use_gpu=not args.no_gpu,
        quality=args.quality,
        gop=args.gop,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
