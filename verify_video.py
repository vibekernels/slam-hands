#!/usr/bin/env python3
"""Verify a video file matches LeRobotDataset v3.0 requirements.

Required: MP4 container, yuv420p pixel format, GOP=2, no audio.
Optional/info: codec, color metadata, file size.

Usage:
    python3 verify_video.py output.mp4
    python3 verify_video.py output.mp4 --gop 2 --verbose
"""

import argparse
import json
import subprocess
import sys

VALID_CODECS = {"av1", "h264", "hevc"}


def probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def probe_frames(path: str, num_frames: int = 60) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_frames", "-show_entries", "frame=pict_type,key_frame",
         "-read_intervals", f"%+#{num_frames}",
         "-print_format", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout).get("frames", [])


def verify(path: str, expected_gop: int = 2, verbose: bool = False) -> bool:
    info = probe(path)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video = None
    has_audio = False
    for s in streams:
        if s.get("codec_type") == "video" and video is None:
            video = s
        if s.get("codec_type") == "audio":
            has_audio = True

    if video is None:
        print("FAIL: no video stream found")
        return False

    checks = []

    def check(name: str, actual, expected, required: bool = True, match_fn=None):
        if match_fn:
            ok = match_fn(actual, expected)
        elif isinstance(expected, (list, tuple, set)):
            ok = actual in expected
        else:
            ok = actual == expected
        status = "OK" if ok else ("FAIL" if required else "WARN")
        checks.append(ok if required else True)  # warnings don't fail
        if verbose or not ok:
            label = "required" if required else "info"
            print(f"  {status}: {name} = {actual!r} (expected {expected!r}) [{label}]")

    codec = video.get("codec_name")
    print(f"Verifying: {path}")
    print(f"  Stream: {codec} {video.get('width')}x{video.get('height')} "
          f"{video.get('pix_fmt')} @ {video.get('r_frame_rate')} fps")

    # === Required checks ===

    # Container: must be MP4
    fmt_name = fmt.get("format_name", "")
    check("container", fmt_name, "mp4/mov",
          match_fn=lambda a, e: "mp4" in a or "mov" in a)

    # Pixel format: must be yuv420p
    check("pix_fmt", video.get("pix_fmt"), "yuv420p")

    # No audio
    check("audio", has_audio, False)

    # Codec: must be one lerobot supports
    check("codec", codec, VALID_CODECS)

    # GOP structure: check keyframe pattern
    frames = probe_frames(path, num_frames=expected_gop * 20)
    if len(frames) < expected_gop * 2:
        print(f"  FAIL: only {len(frames)} frames found, need at least {expected_gop * 2} [required]")
        checks.append(False)
    else:
        gop_ok = True
        for i, f in enumerate(frames):
            expected_key = (i % expected_gop == 0)
            actual_key = f.get("key_frame") == 1
            if actual_key != expected_key:
                gop_ok = False
                if verbose:
                    print(f"  FAIL: frame {i} key_frame={f.get('key_frame')} "
                          f"(expected {'1' if expected_key else '0'} for GOP={expected_gop}) [required]")
                break
        status = "OK" if gop_ok else "FAIL"
        checks.append(gop_ok)
        if verbose or not gop_ok:
            pattern = "".join("I" if f.get("key_frame") == 1 else "P" for f in frames[:10])
            print(f"  {status}: gop = {expected_gop} (pattern: {pattern}...) [required]")

    # === Info checks (warnings, don't fail) ===
    check("color_range", video.get("color_range"), ("tv", "mpeg"), required=False)
    check("color_space", video.get("color_space"), "bt709", required=False)
    check("color_transfer", video.get("color_transfer"), "bt709", required=False)
    check("color_primaries", video.get("color_primaries"), "bt709", required=False)

    passed = all(checks)
    n_required = sum(1 for c in checks if True)  # all are counted
    print(f"\n  {'PASS' if passed else 'FAIL'}: {sum(checks)}/{len(checks)} checks passed")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobotDataset v3.0 video compatibility")
    parser.add_argument("input", help="Video file to verify")
    parser.add_argument("--gop", type=int, default=2, help="Expected GOP size (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks, not just failures")
    args = parser.parse_args()

    ok = verify(args.input, expected_gop=args.gop, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
