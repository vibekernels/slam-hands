#!/usr/bin/env python3
"""Verify a video file matches the LeRobotDataset v3.0 spec.

Checks codec, pixel format, color metadata, GOP structure, container,
and audio absence. Returns exit code 0 if all checks pass, 1 otherwise.

Usage:
    python3 verify_video.py output.mp4
    python3 verify_video.py output.mp4 --gop 2 --verbose
"""

import argparse
import json
import subprocess
import sys


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

    def check(name: str, actual, expected, match_fn=None):
        if match_fn:
            ok = match_fn(actual, expected)
        elif isinstance(expected, (list, tuple)):
            ok = actual in expected
        else:
            ok = actual == expected
        status = "OK" if ok else "FAIL"
        checks.append(ok)
        if verbose or not ok:
            print(f"  {status}: {name} = {actual!r} (expected {expected!r})")

    print(f"Verifying: {path}")
    print(f"  Stream: {video.get('codec_name')} {video.get('width')}x{video.get('height')} "
          f"{video.get('pix_fmt')} @ {video.get('r_frame_rate')} fps")

    # Codec
    check("codec", video.get("codec_name"), "av1")

    # Container
    fmt_name = fmt.get("format_name", "")
    check("container", fmt_name, "mov,mp4,m4a,3gp,3g2,mj2",
          match_fn=lambda a, e: "mp4" in a or "mov" in a)

    # Pixel format
    check("pix_fmt", video.get("pix_fmt"), "yuv420p")

    # Color metadata
    check("color_range", video.get("color_range"), ("tv", "mpeg"))
    check("color_space", video.get("color_space"), "bt709")
    check("color_transfer", video.get("color_transfer"), "bt709")
    check("color_primaries", video.get("color_primaries"), "bt709")

    # No audio
    check("audio", has_audio, False)

    # GOP structure: check keyframe pattern
    frames = probe_frames(path, num_frames=expected_gop * 20)
    if len(frames) < expected_gop * 2:
        print(f"  FAIL: only {len(frames)} frames found, need at least {expected_gop * 2}")
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
                          f"(expected {'1' if expected_key else '0'} for GOP={expected_gop})")
                break
        status = "OK" if gop_ok else "FAIL"
        checks.append(gop_ok)
        if verbose or not gop_ok:
            pattern = "".join("I" if f.get("key_frame") == 1 else "P" for f in frames[:10])
            print(f"  {status}: gop = {expected_gop} (pattern: {pattern}...)")

    passed = all(checks)
    print(f"\n  {'PASS' if passed else 'FAIL'}: {sum(checks)}/{len(checks)} checks passed")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobotDataset v3.0 video format")
    parser.add_argument("input", help="Video file to verify")
    parser.add_argument("--gop", type=int, default=2, help="Expected GOP size (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks, not just failures")
    args = parser.parse_args()

    ok = verify(args.input, expected_gop=args.gop, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
