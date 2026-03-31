#!/usr/bin/env python3
"""Verify a pipeline output directory matches LeRobotDataset v3.0 and EgoVerse format.

Checks directory structure, info.json schema, parquet schemas and dtypes,
video format, and cross-file consistency.

Usage:
    python3 verify_output.py /path/to/output_dataset
    python3 verify_output.py /path/to/output_dataset --verbose
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ── Helpers ──────────────────────────────────────────────────────────────────

class Checker:
    """Accumulates pass/fail/warn checks with formatted output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []  # (ok, required, msg)

    def check(self, name: str, ok: bool, detail: str = "", required: bool = True):
        self.results.append((ok, required, name))
        status = "OK" if ok else ("FAIL" if required else "WARN")
        if self.verbose or not ok:
            msg = f"  {status}: {name}"
            if not ok and detail:
                msg += f" — {detail}"
            print(msg)
        return ok

    @property
    def passed(self):
        return all(ok for ok, req, _ in self.results if req)

    @property
    def summary(self):
        total = len(self.results)
        ok = sum(1 for r in self.results if r[0])
        fails = sum(1 for ok, req, _ in self.results if not ok and req)
        warns = sum(1 for ok, req, _ in self.results if not ok and not req)
        return total, ok, fails, warns


def ffprobe_json(path: str, *extra_args) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json"] + list(extra_args) + [path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


# ── Video checks ─────────────────────────────────────────────────────────────

VALID_CODECS = {"av1", "h264", "hevc"}


def verify_video(path: str, c: Checker, expected_gop: int = 2, expected_fps: float = None):
    """Verify video file meets LeRobot v3.0 requirements."""
    info = ffprobe_json(path, "-show_streams", "-show_format")
    if not info:
        c.check("video.probe", False, f"ffprobe failed on {path}")
        return None

    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video = None
    has_audio = False
    for s in streams:
        if s.get("codec_type") == "video" and video is None:
            video = s
        if s.get("codec_type") == "audio":
            has_audio = True

    if not c.check("video.stream_exists", video is not None, "no video stream"):
        return None

    codec = video.get("codec_name")
    pix_fmt = video.get("pix_fmt")
    fmt_name = fmt.get("format_name", "")

    print(f"  Video: {codec} {video.get('width')}x{video.get('height')} "
          f"{pix_fmt} @ {video.get('r_frame_rate')} fps")

    # Required
    c.check("video.container", "mp4" in fmt_name or "mov" in fmt_name,
            f"got {fmt_name!r}, need mp4")
    c.check("video.pix_fmt", pix_fmt == "yuv420p", f"got {pix_fmt!r}")
    c.check("video.no_audio", not has_audio, "audio stream found")
    c.check("video.codec", codec in VALID_CODECS, f"got {codec!r}")

    # GOP
    frames = ffprobe_json(path, "-select_streams", "v:0", "-show_frames",
                          "-show_entries", "frame=pict_type,key_frame",
                          "-read_intervals", f"%+#{expected_gop * 20}").get("frames", [])
    if len(frames) >= expected_gop * 2:
        gop_ok = all(
            (f.get("key_frame") == 1) == (i % expected_gop == 0)
            for i, f in enumerate(frames)
        )
        pattern = "".join("I" if f.get("key_frame") == 1 else "P" for f in frames[:10])
        c.check("video.gop", gop_ok, f"pattern: {pattern}..., expected GOP={expected_gop}")
    else:
        c.check("video.gop", False, f"only {len(frames)} frames, need {expected_gop * 2}")

    # FPS consistency
    if expected_fps is not None:
        num, den = video.get("r_frame_rate", "0/1").split("/")
        actual_fps = float(num) / float(den) if float(den) != 0 else 0
        fps_ok = abs(actual_fps - expected_fps) / max(expected_fps, 1e-9) < 0.01
        c.check("video.fps_match", fps_ok,
                f"video={actual_fps:.4f}, info.json={expected_fps:.4f}")

    # Color (info only)
    c.check("video.color_space", video.get("color_space") == "bt709",
            f"got {video.get('color_space')!r}", required=False)
    c.check("video.color_transfer", video.get("color_transfer") == "bt709",
            f"got {video.get('color_transfer')!r}", required=False)

    nb = video.get("nb_frames", "N/A")
    return int(nb) if nb != "N/A" else None


# ── info.json checks ─────────────────────────────────────────────────────────

REQUIRED_INFO_KEYS = {
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "chunks_size", "fps", "splits", "data_path", "video_path",
    "features",
}

REQUIRED_SCALAR_FEATURES = {
    "index": "int64",
    "frame_index": "int64",
    "episode_index": "int64",
    "timestamp": "float32",
    "task_index": "int64",
}


def verify_info_json(info: dict, c: Checker):
    """Verify info.json matches LeRobot v3.0 schema."""
    # Required top-level keys
    for key in REQUIRED_INFO_KEYS:
        c.check(f"info.has_{key}", key in info, f"missing from info.json")

    c.check("info.codebase_version", info.get("codebase_version") == "v3.0",
            f"got {info.get('codebase_version')!r}")

    c.check("info.total_episodes", isinstance(info.get("total_episodes"), int) and info["total_episodes"] > 0,
            f"got {info.get('total_episodes')!r}")
    c.check("info.total_frames", isinstance(info.get("total_frames"), int) and info["total_frames"] > 0,
            f"got {info.get('total_frames')!r}")
    c.check("info.fps", isinstance(info.get("fps"), (int, float)) and info["fps"] > 0,
            f"got {info.get('fps')!r}")

    # Splits
    splits = info.get("splits", {})
    c.check("info.splits.train", "train" in splits, f"got {splits!r}")

    # Path templates
    c.check("info.data_path",
            info.get("data_path") == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            f"got {info.get('data_path')!r}")
    c.check("info.video_path",
            info.get("video_path") == "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            f"got {info.get('video_path')!r}")

    # Required scalar features
    features = info.get("features", {})
    for feat_name, expected_dtype in REQUIRED_SCALAR_FEATURES.items():
        ft = features.get(feat_name)
        c.check(f"info.feature.{feat_name}", ft is not None, "missing")
        if ft:
            c.check(f"info.feature.{feat_name}.dtype", ft.get("dtype") == expected_dtype,
                    f"got {ft.get('dtype')!r}, expected {expected_dtype!r}")
            shape = tuple(ft.get("shape", []))
            c.check(f"info.feature.{feat_name}.shape", shape == (1,),
                    f"got {shape!r}, expected (1,)")

    # Video feature
    video_feats = {k: v for k, v in features.items() if v.get("dtype") == "video"}
    c.check("info.has_video_feature", len(video_feats) > 0, "no video feature found")
    for vk, vf in video_feats.items():
        vinfo = vf.get("info", {})
        c.check(f"info.video.{vk}.has_info", bool(vinfo), "missing 'info' sub-object")
        if vinfo:
            c.check(f"info.video.{vk}.pix_fmt", vinfo.get("video.pix_fmt") == "yuv420p",
                    f"got {vinfo.get('video.pix_fmt')!r}")
            c.check(f"info.video.{vk}.codec", vinfo.get("video.codec") in VALID_CODECS,
                    f"got {vinfo.get('video.codec')!r}")

    return features


# ── EgoVerse annotation checks ───────────────────────────────────────────────

EGOVERSE_FEATURES = {
    # (feature_name, expected_shape, dtype, required)
    "observation.slam.pose": ((7,), "float32", True),
    "observation.slam.intrinsics": ((4,), "float32", False),
    "observation.hand.left.keypoints_3d": ((21, 3), "float32", True),
    "observation.hand.left.keypoints_2d": ((21, 2), "float32", True),
    "observation.hand.left.detected": ((1,), "float32", True),
    "observation.hand.right.keypoints_3d": ((21, 3), "float32", True),
    "observation.hand.right.keypoints_2d": ((21, 2), "float32", True),
    "observation.hand.right.detected": ((1,), "float32", True),
    "observation.audio.transcript": ((1,), "string", False),
}


def verify_egoverse_features(features: dict, c: Checker):
    """Verify EgoVerse-specific annotation features are declared in info.json."""
    for feat_name, (expected_shape, expected_dtype, required) in EGOVERSE_FEATURES.items():
        ft = features.get(feat_name)
        if ft is None:
            c.check(f"egoverse.{feat_name}", False, "not declared in info.json", required=required)
            continue
        shape = tuple(ft.get("shape", []))
        c.check(f"egoverse.{feat_name}.shape", shape == expected_shape,
                f"got {shape}, expected {expected_shape}", required=required)
        c.check(f"egoverse.{feat_name}.dtype", ft.get("dtype") == expected_dtype,
                f"got {ft.get('dtype')!r}, expected {expected_dtype!r}", required=required)


# ── Data parquet checks ──────────────────────────────────────────────────────

def verify_data_parquet(path: Path, info: dict, c: Checker):
    """Verify data parquet schema and content against info.json."""
    table = pq.read_table(path)
    schema = table.schema
    num_rows = len(table)
    features = info.get("features", {})
    total_frames = info.get("total_frames", 0)
    fps = info.get("fps", 30)

    c.check("data.row_count", num_rows == total_frames,
            f"got {num_rows} rows, info.json says {total_frames}")

    # Check required columns exist
    col_names = set(schema.names)
    for col in ["index", "frame_index", "episode_index", "timestamp", "task_index"]:
        c.check(f"data.has_{col}", col in col_names, "column missing from parquet")

    # Check index is sequential 0..N-1
    if "index" in col_names:
        idx = table.column("index").to_pylist()
        c.check("data.index_sequential", idx == list(range(num_rows)),
                f"first={idx[0] if idx else '?'}, last={idx[-1] if idx else '?'}")

    # Check frame_index starts at 0
    if "frame_index" in col_names:
        fi = table.column("frame_index").to_pylist()
        c.check("data.frame_index_start", fi[0] == 0, f"starts at {fi[0]}")

    # Check timestamps are monotonically increasing
    if "timestamp" in col_names:
        ts = table.column("timestamp").to_pandas().values
        c.check("data.timestamps_monotonic", bool(np.all(np.diff(ts) >= 0)),
                f"non-monotonic at {np.argmin(np.diff(ts))}" if len(ts) > 1 else "")
        # Check last timestamp is reasonable
        expected_last = (num_rows - 1) / fps
        actual_last = float(ts[-1]) if len(ts) > 0 else 0
        c.check("data.timestamp_range", abs(actual_last - expected_last) < 0.1,
                f"last={actual_last:.3f}, expected ~{expected_last:.3f}")

    # Verify annotation columns match info.json features
    for feat_name, ft in features.items():
        if ft.get("dtype") == "video":
            # Video features should NOT be in parquet
            c.check(f"data.no_video_col.{feat_name}", feat_name not in col_names,
                    "video feature should not be in data parquet")
            continue
        shape = tuple(ft.get("shape", []))
        dtype = ft.get("dtype", "")

        if feat_name not in col_names:
            # Only warn for optional features
            if feat_name in REQUIRED_SCALAR_FEATURES:
                c.check(f"data.col.{feat_name}", False, "required column missing")
            continue

        col = table.column(feat_name)

        # Check dtype compatibility
        if shape == (1,):
            # Scalar column
            arrow_type = col.type
            if dtype == "float32":
                c.check(f"data.dtype.{feat_name}", str(arrow_type) == "float",
                        f"got {arrow_type}, expected float", required=False)
            elif dtype == "int64":
                c.check(f"data.dtype.{feat_name}", str(arrow_type) == "int64",
                        f"got {arrow_type}, expected int64")
            elif dtype == "string":
                c.check(f"data.dtype.{feat_name}",
                        "string" in str(arrow_type) or "utf8" in str(arrow_type),
                        f"got {arrow_type}, expected string")
        elif len(shape) == 1:
            # 1D list column — should be fixed_size_list<float32>[N]
            arrow_type = str(col.type)
            n = shape[0]
            c.check(f"data.type.{feat_name}",
                    f"fixed_size_list" in arrow_type or "list" in arrow_type,
                    f"got {arrow_type}, expected list type")
            # Verify element count
            first = col[0].as_py()
            if first is not None:
                c.check(f"data.len.{feat_name}", len(first) == n,
                        f"got {len(first)} elements, expected {n}")
        elif len(shape) == 2:
            # 2D array column — list of lists
            first = col[0].as_py()
            if first is not None:
                c.check(f"data.shape.{feat_name}",
                        len(first) == shape[0] and len(first[0]) == shape[1],
                        f"got [{len(first)}][{len(first[0]) if first else '?'}], expected {shape}")

    # Check no NaN in pose data
    if "observation.slam.pose" in col_names:
        poses = np.array(table.column("observation.slam.pose").to_pylist(), dtype=np.float32)
        nan_count = np.isnan(poses).sum()
        c.check("data.slam.no_nan", nan_count == 0,
                f"{nan_count} NaN values in SLAM poses", required=False)
        # Check quaternion norms ~1
        quats = poses[:, 3:]
        norms = np.linalg.norm(quats, axis=1)
        quat_ok = np.allclose(norms, 1.0, atol=0.01)
        c.check("data.slam.quaternion_norm", quat_ok,
                f"norm range [{norms.min():.4f}, {norms.max():.4f}]", required=False)

    # Check hand detection flags are 0 or 1
    for side in ["left", "right"]:
        col_name = f"observation.hand.{side}.detected"
        if col_name in col_names:
            vals = table.column(col_name).to_pandas().values.astype(np.float32)
            c.check(f"data.hand.{side}.detected_binary",
                    set(np.unique(vals)).issubset({0.0, 1.0}),
                    f"values: {np.unique(vals)}")
            pct = vals.mean() * 100
            c.check(f"data.hand.{side}.detection_rate",
                    pct > 10,
                    f"{pct:.1f}% — suspiciously low" if pct <= 10 else f"{pct:.1f}%",
                    required=False)

    return num_rows


# ── Episodes parquet checks ──────────────────────────────────────────────────

REQUIRED_EPISODE_COLS = {
    "episode_index", "length", "dataset_from_index", "dataset_to_index",
}


def verify_episodes_parquet(path: Path, info: dict, num_data_rows: int, c: Checker):
    """Verify episodes parquet schema and content."""
    table = pq.read_table(path)
    col_names = set(table.schema.names)

    for col in REQUIRED_EPISODE_COLS:
        c.check(f"episodes.has_{col}", col in col_names, "column missing")

    c.check("episodes.row_count", len(table) == info.get("total_episodes", 0),
            f"got {len(table)}, info.json says {info.get('total_episodes')}")

    if "length" in col_names:
        lengths = table.column("length").to_pylist()
        total = sum(lengths)
        c.check("episodes.total_length", total == num_data_rows,
                f"sum(length)={total}, data has {num_data_rows} rows")

    if "dataset_from_index" in col_names and "dataset_to_index" in col_names:
        froms = table.column("dataset_from_index").to_pylist()
        tos = table.column("dataset_to_index").to_pylist()
        c.check("episodes.index_range", froms[0] == 0 and tos[-1] == num_data_rows,
                f"from={froms[0]}, to={tos[-1]}, expected 0..{num_data_rows}")

    # Check for video chunk/file columns
    video_feats = {k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"}
    for vk in video_feats:
        chunk_col = f"videos/{vk}/chunk_index"
        file_col = f"videos/{vk}/file_index"
        c.check(f"episodes.has_{chunk_col}", chunk_col in col_names,
                "missing video chunk index", required=False)
        c.check(f"episodes.has_{file_col}", file_col in col_names,
                "missing video file index", required=False)


# ── Tasks parquet checks ─────────────────────────────────────────────────────

def verify_tasks_parquet(path: Path, info: dict, c: Checker):
    """Verify tasks parquet."""
    table = pq.read_table(path)
    col_names = set(table.schema.names)

    c.check("tasks.has_task_index", "task_index" in col_names, "column missing")
    c.check("tasks.has_task", "task" in col_names, "column missing")
    c.check("tasks.row_count", len(table) == info.get("total_tasks", 0),
            f"got {len(table)}, info.json says {info.get('total_tasks')}")


# ── Main verification ────────────────────────────────────────────────────────

def verify_dataset(dataset_dir: str, verbose: bool = False, gop: int = 2) -> bool:
    """Run all verification checks on a pipeline output directory."""
    root = Path(dataset_dir)
    c = Checker(verbose=verbose)

    print(f"Verifying dataset: {root}\n")

    # ── Directory structure ──
    print("[Structure]")
    c.check("dir.exists", root.is_dir(), f"{root} is not a directory")
    if not root.is_dir():
        return False

    info_path = root / "meta" / "info.json"
    c.check("dir.info.json", info_path.exists())

    data_parquets = sorted((root / "data").glob("*/*.parquet")) if (root / "data").exists() else []
    c.check("dir.data_parquet", len(data_parquets) > 0, "no parquet files in data/")

    ep_parquets = sorted((root / "meta" / "episodes").glob("*/*.parquet")) if (root / "meta" / "episodes").exists() else []
    c.check("dir.episodes_parquet", len(ep_parquets) > 0, "no parquet files in meta/episodes/")

    tasks_path = root / "meta" / "tasks.parquet"
    c.check("dir.tasks_parquet", tasks_path.exists())

    video_files = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    c.check("dir.video_files", len(video_files) > 0, "no mp4 files in videos/")

    # Check chunk-NNN/file-NNN naming convention
    for pf in data_parquets:
        name_ok = pf.parent.name.startswith("chunk-") and pf.name.startswith("file-")
        c.check(f"dir.naming.{pf.relative_to(root)}", name_ok,
                "expected chunk-NNN/file-NNN.parquet pattern")

    if not info_path.exists():
        print("\nCannot continue without info.json")
        return False

    # ── info.json ──
    print("\n[info.json]")
    with open(info_path) as f:
        info = json.load(f)
    features = verify_info_json(info, c)

    # ── EgoVerse features ──
    print("\n[EgoVerse format]")
    verify_egoverse_features(features, c)

    # ── Video ──
    print("\n[Video]")
    video_frame_count = None
    for vf in video_files:
        video_frame_count = verify_video(str(vf), c, expected_gop=gop,
                                         expected_fps=info.get("fps"))

    # ── Data parquet ──
    print("\n[Data parquet]")
    num_data_rows = 0
    for pf in data_parquets:
        num_data_rows = verify_data_parquet(pf, info, c)

    # Cross-check: video frame count vs parquet rows
    if video_frame_count is not None and num_data_rows > 0:
        c.check("cross.video_vs_data", video_frame_count == num_data_rows,
                f"video has {video_frame_count} frames, parquet has {num_data_rows} rows")

    # ── Episodes parquet ──
    print("\n[Episodes parquet]")
    for pf in ep_parquets:
        verify_episodes_parquet(pf, info, num_data_rows, c)

    # ── Tasks parquet ──
    print("\n[Tasks parquet]")
    if tasks_path.exists():
        verify_tasks_parquet(tasks_path, info, c)

    # ── Audio metadata ──
    audio_path = root / "meta" / "audio.json"
    if audio_path.exists():
        print("\n[Audio]")
        with open(audio_path) as f:
            audio = json.load(f)
        c.check("audio.has_transcript", "transcript" in audio, "missing 'transcript' key")
        c.check("audio.has_words", "words" in audio, "missing 'words' key", required=False)
        c.check("audio.has_segments", "segments" in audio, "missing 'segments' key", required=False)

    # ── Summary ──
    total, ok, fails, warns = c.summary
    print(f"\n{'=' * 50}")
    status = "PASS" if c.passed else "FAIL"
    print(f"  {status}: {ok}/{total} checks passed", end="")
    if fails:
        print(f", {fails} failures", end="")
    if warns:
        print(f", {warns} warnings", end="")
    print()
    print(f"{'=' * 50}")

    return c.passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify pipeline output matches LeRobotDataset v3.0 and EgoVerse format")
    parser.add_argument("input", help="Output dataset directory")
    parser.add_argument("--gop", type=int, default=2, help="Expected GOP size (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks, not just failures")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_dir():
        print(f"Error: {path} is not a directory")
        sys.exit(1)

    ok = verify_dataset(str(path), verbose=args.verbose, gop=args.gop)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
