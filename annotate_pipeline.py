#!/usr/bin/env python3
"""Annotate video with SLAM camera poses and hand pose estimation.

Produces a LeRobot v3.0 compatible dataset following the EgoVerse format:
  - Video converted to LeRobot format (via convert_video.py)
  - SLAM camera poses per frame (DROID-SLAM) — 6-DoF head/camera motion
  - 3D hand pose per frame (WiLoR) — 21 keypoints per hand, left + right

Usage:
    python3 annotate_pipeline.py /workspace/IMG_1443.MOV --output-dir /workspace/output_dataset
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from threading import Thread
from queue import Queue

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Video decode helpers
# ---------------------------------------------------------------------------

def _get_native_decode():
    """Import the native C++ async decoder module."""
    native_decode_path = os.path.join(os.path.dirname(__file__), "native_decode")
    if native_decode_path not in sys.path:
        sys.path.insert(0, native_decode_path)
    import native_decode
    return native_decode


def decode_video_frames(video_path: str, device: str = "cuda"):
    """Decode all video frames using native C++ decoder.

    Returns (frames_rgb_numpy, fps, width, height, num_frames).
    frames_rgb_numpy is a list of numpy arrays [H, W, 3] uint8 RGB.
    """
    nd = _get_native_decode()
    fps, width, height, num_frames_hint = nd.AsyncVideoDecoder.get_metadata(video_path)

    decoder = nd.AsyncVideoDecoder()
    # slam_w/slam_h don't matter — we only use the RGB output
    # Use a small SLAM size to minimize wasted work
    decoder.start(video_path, 64, 64, slam_only=False, queue_depth=128)

    frames = []
    while True:
        result = decoder.get_next()
        if result is None:
            break
        rgb_np, _ = result
        if rgb_np is not None:
            frames.append(np.array(rgb_np))
    decoder.stop()

    return frames, fps, width, height, len(frames)


def get_video_metadata(video_path: str):
    """Get video metadata without decoding frames."""
    nd = _get_native_decode()
    fps, width, height, num_frames = nd.AsyncVideoDecoder.get_metadata(video_path)
    return fps, width, height, int(num_frames)


def get_iphone_intrinsics(width: int, height: int):
    """Estimate camera intrinsics for iPhone rear camera."""
    fx = fy = max(width, height) * 0.75
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# SLAM annotation (DROID-SLAM)
# ---------------------------------------------------------------------------

def _prepare_slam_frame(rgb, w1, h1):
    """Convert RGB numpy frame to DROID-SLAM input tensor.

    DROID expects BGR [1, 3, H, W] uint8 at SLAM resolution.
    Uses GPU resize via F.interpolate for speed.
    """
    t = torch.as_tensor(rgb[:, :, ::-1].copy()).permute(2, 0, 1)  # [3, H, W]
    t = t.unsqueeze(0).float().cuda()
    t = F.interpolate(t, size=(h1, w1), mode="bilinear", align_corners=False)
    return t.byte()


def _prepare_slam_frames_batched(frames, w1, h1, num_workers=8):
    """Convert RGB numpy frames to DROID-SLAM tensors using multi-threaded CPU resize.

    Uses cv2.resize on CPU threads (much faster than per-frame GPU round-trip for
    full-res→SLAM-res downscale) then transfers small tensors to GPU.
    Returns list of [1, 3, H, W] uint8 CUDA tensors (same format as _prepare_slam_frame).
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    def _resize_one(rgb):
        small = cv2.resize(rgb[:, :, ::-1], (w1, h1), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).cuda()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(_resize_one, frames))


def _slerp_quaternions(q0, q1, t):
    """Spherical linear interpolation between quaternion arrays.

    q0, q1: [4] arrays (x, y, z, w format)
    t: scalar in [0, 1]
    """
    dot = np.clip(np.sum(q0 * q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def _interpolate_poses_simple(keyframe_tstamps, keyframe_poses, total_frames):
    """Interpolate poses for all frames using linear translation + slerp rotation.

    keyframe_tstamps: sorted array of keyframe indices
    keyframe_poses: [K, 7] array (tx, ty, tz, qx, qy, qz, qw)
    total_frames: total number of frames

    Returns: [total_frames, 7] interpolated poses
    """
    all_poses = np.zeros((total_frames, 7), dtype=np.float64)

    for i in range(total_frames):
        # Find surrounding keyframes
        idx = np.searchsorted(keyframe_tstamps, i, side='right') - 1
        idx = max(0, min(idx, len(keyframe_tstamps) - 1))

        if idx >= len(keyframe_tstamps) - 1:
            all_poses[i] = keyframe_poses[-1]
        elif keyframe_tstamps[idx] == i:
            all_poses[i] = keyframe_poses[idx]
        else:
            t0, t1 = keyframe_tstamps[idx], keyframe_tstamps[idx + 1]
            alpha = (i - t0) / max(t1 - t0, 1e-6)
            # Linear translation interpolation
            all_poses[i, :3] = (1 - alpha) * keyframe_poses[idx, :3] + alpha * keyframe_poses[idx + 1, :3]
            # Slerp quaternion interpolation
            all_poses[i, 3:] = _slerp_quaternions(keyframe_poses[idx, 3:], keyframe_poses[idx + 1, 3:], alpha)

    return all_poses


def run_slam_with_decode(video_path, fps, width, height, weights_path, device="cuda",
                         backend_steps=(7, 12), fast_traj=False):
    """Decode video and run DROID-SLAM in one pass (no double-decode).

    Decodes frames, prepares SLAM tensors (batched multi-threaded), runs tracking,
    backend optimization, and trajectory fill. Returns both SLAM results and decoded
    frames (for body inference).

    Args:
        backend_steps: Tuple of (step1, step2) for backend optimization iterations.
        fast_traj: If True, use simple linear/slerp interpolation instead of NN trajectory filler.
                   Saves ~3s GPU time but slightly less accurate non-keyframe poses.

    Returns:
        slam_result: dict with poses, intrinsics, slam_resolution
        frames: list of RGB numpy arrays
    """
    sys.path.insert(0, os.path.join(os.path.dirname(weights_path), "..", "droid_slam"))
    from droid import Droid

    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8

    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics_scaled = torch.as_tensor([
        fx * (w1 / w0), fy * (h1 / h0), cx * (w1 / w0), cy * (h1 / h0),
    ])

    slam_args = argparse.Namespace(
        weights=weights_path, buffer=512, image_size=[h1, w1],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device=device, backend_device=device,
    )

    # --- Decode all frames ---
    t_decode_start = time.perf_counter()
    nd = _get_native_decode()
    decoder = nd.AsyncVideoDecoder()
    decoder.start(video_path, w1, h1, slam_only=False, queue_depth=128)
    frames = []
    slam_bgrs = []
    while True:
        result = decoder.get_next()
        if result is None:
            break
        rgb_np, slam_bgr = result
        if rgb_np is not None:
            frames.append(np.array(rgb_np))
        slam_bgrs.append(slam_bgr)
    decoder.stop()
    num_frames = len(frames)
    t_decode = time.perf_counter() - t_decode_start
    print(f"  Decoded {num_frames} frames in {t_decode:.1f}s ({num_frames/t_decode:.0f} fps)")

    # --- DROID-SLAM ---
    droid = Droid(slam_args)
    print(f"  SLAM resolution: {w1}x{h1}")

    # SLAM BGR frames already decoded at correct resolution by native decoder
    t_track_start = time.perf_counter()
    slam_tensors = []

    t = 0
    for slam_bgr in slam_bgrs:
        tensor = torch.from_numpy(np.array(slam_bgr)).permute(2, 0, 1).unsqueeze(0).cuda()
        slam_tensors.append(tensor)
        droid.track(t, tensor, intrinsics=intrinsics_scaled)
        t += 1
    del slam_bgrs

    t_track = time.perf_counter() - t_track_start
    print(f"  Tracking: {t_track:.1f}s ({num_frames/t_track:.0f} fps)")

    # Backend optimization
    print(f"  Backend optimization ({backend_steps[0]}+{backend_steps[1]} steps)...")
    del droid.frontend
    torch.cuda.empty_cache()

    t_backend_start = time.perf_counter()
    droid.backend(backend_steps[0])
    torch.cuda.empty_cache()
    droid.backend(backend_steps[1])
    t_backend = time.perf_counter() - t_backend_start
    print(f"  Backend: {t_backend:.1f}s")

    # Trajectory fill
    if fast_traj:
        # Free SLAM tensors early since we don't need them for interpolation
        del slam_tensors
        torch.cuda.empty_cache()

        print("  Trajectory fill (fast linear/slerp)...")
        t_traj_start = time.perf_counter()
        N = droid.video.counter.value
        kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
        import lietorch
        kf_poses_se3 = lietorch.SE3(droid.video.poses[:N])
        kf_poses_raw = kf_poses_se3.inv().data.cpu().numpy()  # [N, 7]
        all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_raw, num_frames)
        t_traj = time.perf_counter() - t_traj_start
        print(f"  Trajectory fill: {t_traj:.2f}s ({N} keyframes → {num_frames} frames)")
    else:
        print("  Trajectory fill (NN refinement)...")
        t_traj_start = time.perf_counter()

        def frame_stream():
            for i, tensor in enumerate(slam_tensors):
                yield i, tensor, intrinsics_scaled

        traj = droid.traj_filler(frame_stream())
        all_poses = traj.inv().data.cpu().numpy()
        del slam_tensors
        t_traj = time.perf_counter() - t_traj_start
        print(f"  Trajectory fill: {t_traj:.1f}s")

    return {
        "poses": all_poses,
        "intrinsics": intrinsics_scaled.numpy(),
        "slam_resolution": (h1, w1),
    }, frames


def run_slam_native_decode(video_path, fps, width, height, weights_path, device="cuda",
                           backend_steps=(7, 12), fast_traj=False, collect_rgb=False):
    """Decode+SLAM with GIL-free C++ async decoder for true parallel overlap.

    The C++ decoder thread runs FFmpeg decode + swscale resize entirely without
    the GIL, while the Python main thread calls droid.track() (which runs CUDA
    kernels that also release the GIL). This gives true overlap: decode is fully
    hidden behind tracking.

    Args:
        collect_rgb: If True, use slam_only=False to also decode full-res RGB
            frames during SLAM. Eliminates the need for a separate body decode pass.
            Adds ~7s decode time but saves ~9s of a separate decode pass.

    Returns (slam_result, num_frames) if collect_rgb=False.
    Returns (slam_result, num_frames, frames) if collect_rgb=True, where frames
    is a list of numpy RGB arrays [H, W, 3] uint8.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(weights_path), "..", "droid_slam"))
    from droid import Droid

    nd = _get_native_decode()

    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8

    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics_scaled = torch.as_tensor([
        fx * (w1 / w0), fy * (h1 / h0), cx * (w1 / w0), cy * (h1 / h0),
    ])

    slam_args = argparse.Namespace(
        weights=weights_path, buffer=512, image_size=[h1, w1],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device=device, backend_device=device,
    )

    # Start C++ async decoder (GIL-free)
    # When collect_rgb=True, also produce full-res RGB frames for body inference
    decoder = nd.AsyncVideoDecoder()
    decoder.start(video_path, w1, h1, slam_only=not collect_rgb, queue_depth=128)

    droid = Droid(slam_args)
    mode_str = "RGB+SLAM" if collect_rgb else "SLAM-only"
    print(f"  SLAM resolution: {w1}x{h1} (native C++ decode, {mode_str})")

    # Stream frames: C++ decode thread runs in parallel with SLAM tracking
    # get_next() releases GIL while waiting, so decode thread runs concurrently
    t_track_start = time.perf_counter()
    slam_tensors = [] if not fast_traj else None
    rgb_frames = [] if collect_rgb else None
    t = 0
    while True:
        result = decoder.get_next()
        if result is None:
            break
        rgb_np, slam_bgr = result
        if collect_rgb and rgb_np is not None:
            rgb_frames.append(np.array(rgb_np))  # copy from pybind buffer
        tensor = torch.from_numpy(slam_bgr).permute(2, 0, 1).unsqueeze(0).cuda()
        if slam_tensors is not None:
            slam_tensors.append(tensor)
        droid.track(t, tensor, intrinsics=intrinsics_scaled)
        t += 1
    decoder.stop()
    num_frames = t

    t_track = time.perf_counter() - t_track_start
    print(f"  Native decode+tracking: {t_track:.1f}s ({num_frames/t_track:.0f} fps)")

    # Backend optimization
    print(f"  Backend optimization ({backend_steps[0]}+{backend_steps[1]} steps)...")
    del droid.frontend
    torch.cuda.empty_cache()

    t_backend_start = time.perf_counter()
    droid.backend(backend_steps[0])
    torch.cuda.empty_cache()
    droid.backend(backend_steps[1])
    t_backend = time.perf_counter() - t_backend_start
    print(f"  Backend: {t_backend:.1f}s")

    # Trajectory fill
    if fast_traj:
        torch.cuda.empty_cache()
        print("  Trajectory fill (fast linear/slerp)...")
        t_traj_start = time.perf_counter()
        N = droid.video.counter.value
        kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
        import lietorch
        kf_poses_se3 = lietorch.SE3(droid.video.poses[:N])
        kf_poses_raw = kf_poses_se3.inv().data.cpu().numpy()
        all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_raw, num_frames)
        t_traj = time.perf_counter() - t_traj_start
        print(f"  Trajectory fill: {t_traj:.2f}s ({N} keyframes → {num_frames} frames)")
    else:
        print("  Trajectory fill (NN refinement)...")
        t_traj_start = time.perf_counter()

        def frame_stream():
            for i, tensor in enumerate(slam_tensors):
                yield i, tensor, intrinsics_scaled

        traj = droid.traj_filler(frame_stream())
        all_poses = traj.inv().data.cpu().numpy()
        del slam_tensors
        t_traj = time.perf_counter() - t_traj_start
        print(f"  Trajectory fill: {t_traj:.1f}s")

    slam_result = {
        "poses": all_poses,
        "intrinsics": intrinsics_scaled.numpy(),
        "slam_resolution": (h1, w1),
    }

    if collect_rgb:
        return slam_result, num_frames, rgb_frames
    return slam_result, num_frames


def run_slam(frames, fps, width, height, weights_path, device="cuda"):
    """Run DROID-SLAM on pre-decoded video frames (non-streaming fallback).

    Returns dict with:
        poses: numpy [N, 7] (tx, ty, tz, qx, qy, qz, qw)
        intrinsics: numpy [4] (fx, fy, cx, cy) at SLAM resolution
    """
    sys.path.insert(0, os.path.join(os.path.dirname(weights_path), "..", "droid_slam"))
    from droid import Droid

    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8

    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics_scaled = torch.as_tensor([
        fx * (w1 / w0), fy * (h1 / h0), cx * (w1 / w0), cy * (h1 / h0),
    ])

    args = argparse.Namespace(
        weights=weights_path, buffer=512, image_size=[h1, w1],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device=device, backend_device=device,
    )

    droid = Droid(args)
    print(f"  SLAM resolution: {w1}x{h1}, intrinsics: fx={intrinsics_scaled[0]:.1f} fy={intrinsics_scaled[1]:.1f}")

    # Pre-compute all SLAM tensors once (avoids double decode in terminate)
    print("  Preparing SLAM tensors (batched)...")
    slam_tensors = _prepare_slam_frames_batched(frames, w1, h1)

    # Feed frames using cached tensors
    for t, tensor in enumerate(slam_tensors):
        droid.track(t, tensor, intrinsics=intrinsics_scaled)
        if t % 200 == 0:
            print(f"    SLAM frame {t}/{len(frames)}")

    # Re-use cached tensors for terminate stream
    def frame_stream():
        for t, tensor in enumerate(slam_tensors):
            yield t, tensor, intrinsics_scaled

    print("  Running global optimization...")
    traj_est = droid.terminate(frame_stream())

    del slam_tensors
    return {
        "poses": traj_est,
        "intrinsics": intrinsics_scaled.numpy(),
        "slam_resolution": (h1, w1),
    }


# ---------------------------------------------------------------------------
# Body pose annotation (Multi-HMR)
# ---------------------------------------------------------------------------

def _gpu_preprocess_body(rgb_tensor, img_size, mean, std):
    """GPU-accelerated preprocessing for Multi-HMR."""
    _, h, w = rgb_tensor.shape
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    x = rgb_tensor.unsqueeze(0).float()
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = img_size - new_h
    pad_w = img_size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    x = F.pad(x, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))

    x = x / 255.0
    x = (x - mean) / std
    return x


# ---------------------------------------------------------------------------
# Hand pose estimation (WiLoR) — EgoVerse format: 21 keypoints per hand
# ---------------------------------------------------------------------------

_wilor_cache = {}

def _load_wilor(wilor_dir, device="cuda", fast=True):
    """Load WiLoR model and YOLO hand detector. Cached after first call."""
    cache_key = (wilor_dir, device)
    if cache_key in _wilor_cache:
        return _wilor_cache[cache_key]

    sys.path.insert(0, wilor_dir)
    from wilor.models import load_wilor
    from ultralytics import YOLO

    orig_cwd = os.getcwd()
    os.chdir(wilor_dir)
    try:
        model, model_cfg = load_wilor(
            checkpoint_path=os.path.join(wilor_dir, 'pretrained_models', 'wilor_final.ckpt'),
            cfg_path=os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
        )
    finally:
        os.chdir(orig_cwd)
    model = model.to(device).eval()
    if fast:
        model = model.half()
        model.backbone.skip_blocks = True

    detector = YOLO(os.path.join(wilor_dir, 'pretrained_models', 'detector.pt')).to(device)

    _wilor_cache[cache_key] = (model, model_cfg, detector)
    return model, model_cfg, detector


def _load_wilor_cpu(wilor_dir):
    """Load WiLoR model on CPU in background thread (for overlap with SLAM)."""
    sys.path.insert(0, wilor_dir)
    # WiLoR config uses relative paths (./mano_data/...), so chdir temporarily
    orig_cwd = os.getcwd()
    os.chdir(wilor_dir)
    try:
        from wilor.models import load_wilor
        model, model_cfg = load_wilor(
            checkpoint_path=os.path.join(wilor_dir, 'pretrained_models', 'wilor_final.ckpt'),
            cfg_path=os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
        )
    finally:
        os.chdir(orig_cwd)
    return model, model_cfg


def run_hand_pose_wilor(frames, fps, width, height, wilor_dir, device="cuda",
                        stride=1, det_conf=0.3, preloaded=None):
    """Run WiLoR hand pose estimation on video frames.

    Returns dict with per-frame hand pose data in EgoVerse format:
    - left_hand_keypoints_3d: [N, 21, 3] — 21 keypoints in camera space (NaN if not detected)
    - right_hand_keypoints_3d: [N, 21, 3]
    - left_hand_keypoints_2d: [N, 21, 2] — projected 2D keypoints
    - right_hand_keypoints_2d: [N, 21, 2]
    - left_hand_detected: [N] bool
    - right_hand_detected: [N] bool
    - left_hand_wrist_pose: [N, 7] — wrist SE(3) as [tx,ty,tz,qx,qy,qz,qw] (NaN if not detected)
    - right_hand_wrist_pose: [N, 7]
    """
    sys.path.insert(0, wilor_dir)
    from wilor.utils import recursive_to
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils.renderer import cam_crop_to_full
    from ultralytics import YOLO

    num_frames = len(frames)

    # Load model
    if preloaded is not None:
        model_cpu, model_cfg = preloaded
        model = model_cpu.to(device).eval().half()
        model.backbone.skip_blocks = True
        detector = YOLO(os.path.join(wilor_dir, 'pretrained_models', 'detector.pt')).to(device)
        # Cache for future calls
        _wilor_cache[(wilor_dir, device)] = (model, model_cfg, detector)
    else:
        model, model_cfg, detector = _load_wilor(wilor_dir, device)

    # Stride: process every Nth frame, interpolate rest
    if stride > 1:
        sampled_indices = list(range(0, num_frames, stride))
        if sampled_indices[-1] != num_frames - 1:
            sampled_indices.append(num_frames - 1)
        print(f"  Stride={stride}: processing {len(sampled_indices)}/{num_frames} frames, interpolating rest")
    else:
        sampled_indices = list(range(num_frames))

    # Output arrays
    left_kp3d = np.full((num_frames, 21, 3), np.nan, dtype=np.float32)
    right_kp3d = np.full((num_frames, 21, 3), np.nan, dtype=np.float32)
    left_kp2d = np.full((num_frames, 21, 2), np.nan, dtype=np.float32)
    right_kp2d = np.full((num_frames, 21, 2), np.nan, dtype=np.float32)
    left_detected = np.zeros(num_frames, dtype=bool)
    right_detected = np.zeros(num_frames, dtype=bool)

    t_start = time.perf_counter()
    n_hands = 0

    for count, fi in enumerate(sampled_indices):
        img_rgb = frames[fi]
        img_bgr = img_rgb[:, :, ::-1].copy()

        # Detect hands
        dets = detector(img_bgr, conf=det_conf, verbose=False)[0]
        bboxes = []
        is_right_list = []
        for det in dets:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(bbox[:4].tolist())

        if len(bboxes) == 0:
            if count > 0 and count % 96 == 0:
                elapsed = time.perf_counter() - t_start
                fps_rate = (count + 1) / elapsed
                eta = (len(sampled_indices) - count - 1) / fps_rate
                print(f"    {count+1}/{len(sampled_indices)} frames ({fps_rate:.1f} fps, ETA {eta:.0f}s)")
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right_list)
        dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.0, fp16=True)
        dl = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        for batch in dl:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            bs = batch['img'].shape[0]
            for n in range(bs):
                joints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # [21, 3]
                is_r = batch['right'][n].cpu().numpy()
                joints_3d[:, 0] = (2 * is_r - 1) * joints_3d[:, 0]
                cam_t = pred_cam_t_full[n]
                joints_cam = joints_3d + cam_t  # 3D keypoints in camera space

                # 2D projection
                kp_2d = out['pred_keypoints_2d'][n].detach().cpu().numpy() if 'pred_keypoints_2d' in out else None

                if is_r > 0.5:
                    right_kp3d[fi] = joints_cam
                    right_detected[fi] = True
                    if kp_2d is not None:
                        right_kp2d[fi] = kp_2d[:, :2]
                else:
                    left_kp3d[fi] = joints_cam
                    left_detected[fi] = True
                    if kp_2d is not None:
                        left_kp2d[fi] = kp_2d[:, :2]
                n_hands += 1

        if count > 0 and count % 96 == 0:
            elapsed = time.perf_counter() - t_start
            fps_rate = (count + 1) / elapsed
            eta = (len(sampled_indices) - count - 1) / fps_rate
            print(f"    {count+1}/{len(sampled_indices)} frames ({fps_rate:.1f} fps, ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - t_start
    print(f"  WiLoR inference: {elapsed:.1f}s ({len(sampled_indices)/elapsed:.1f} fps), {n_hands} hands detected")

    # Interpolate strided frames
    if stride > 1:
        for arr in [left_kp3d, right_kp3d, left_kp2d, right_kp2d]:
            _interpolate_hand_keypoints(arr, sampled_indices, num_frames)
        # For boolean detection, use nearest-neighbor
        for det_arr in [left_detected, right_detected]:
            full = np.zeros(num_frames, dtype=bool)
            full[sampled_indices] = det_arr[sampled_indices]
            # Fill gaps with nearest sampled value
            for i in range(len(sampled_indices) - 1):
                a, b = sampled_indices[i], sampled_indices[i + 1]
                mid = (a + b) // 2
                full[a:mid+1] = det_arr[a]
                full[mid+1:b] = det_arr[b]
            det_arr[:] = full

    left_rate = left_detected.sum() / num_frames * 100
    right_rate = right_detected.sum() / num_frames * 100
    print(f"  Detection rate: left={left_rate:.1f}%, right={right_rate:.1f}%")

    return {
        "_model": "wilor",
        "left_hand_keypoints_3d": left_kp3d,
        "right_hand_keypoints_3d": right_kp3d,
        "left_hand_keypoints_2d": left_kp2d,
        "right_hand_keypoints_2d": right_kp2d,
        "left_hand_detected": left_detected,
        "right_hand_detected": right_detected,
    }


def _interpolate_hand_keypoints(arr, sampled_indices, num_frames):
    """Linearly interpolate [N, K, D] keypoint array between sampled frames."""
    for i in range(len(sampled_indices) - 1):
        a, b = sampled_indices[i], sampled_indices[i + 1]
        if b - a <= 1:
            continue
        va, vb = arr[a], arr[b]
        # If either endpoint is NaN, just copy the non-NaN one (or leave NaN)
        a_valid = not np.any(np.isnan(va))
        b_valid = not np.any(np.isnan(vb))
        if a_valid and b_valid:
            for j in range(a + 1, b):
                alpha = (j - a) / (b - a)
                arr[j] = va * (1 - alpha) + vb * alpha
        elif a_valid:
            arr[a+1:b] = va
        elif b_valid:
            arr[a+1:b] = vb
        # else: both NaN, leave as NaN


def run_body_pose(frames, fps, width, height, multi_hmr_dir, device="cuda"):
    """Run Multi-HMR body pose estimation on video frames."""
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ["XFORMERS_DISABLED"] = "1"

    orig_cwd = os.getcwd()
    os.chdir(multi_hmr_dir)
    sys.path.insert(0, multi_hmr_dir)
    from demo import load_model, get_camera_parameters, forward_model
    from utils import normalize_rgb

    model = load_model("multiHMR_896_L")
    os.chdir(orig_cwd)
    img_size = model.img_size
    K = get_camera_parameters(img_size, fov=60)

    print("  Compiling backbone with torch.compile...")
    model.backbone = torch.compile(model.backbone, mode="reduce-overhead")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    results = {
        "scores": [], "j3d": [], "j2d": [], "rotvec": [],
        "shape": [], "transl": [], "num_persons": [],
    }

    for t, rgb in enumerate(frames):
        rgb_tensor = torch.as_tensor(rgb).permute(2, 0, 1).to(device)
        x = _gpu_preprocess_body(rgb_tensor, img_size, mean, std)
        humans = forward_model(model, x, K, det_thresh=0.3, nms_kernel_size=3)

        if len(humans) > 0:
            best = max(humans, key=lambda h: h["scores"].item())
            results["scores"].append(best["scores"].cpu().item())
            results["j3d"].append(best["j3d"].cpu().numpy())
            results["j2d"].append(best["j2d"].cpu().float().numpy())
            results["rotvec"].append(best["rotvec"].cpu().numpy())
            results["shape"].append(best["shape"].cpu().numpy())
            results["transl"].append(best["transl"].cpu().numpy())
            results["num_persons"].append(len(humans))
        else:
            results["scores"].append(0.0)
            results["j3d"].append(np.zeros((127, 3), dtype=np.float32))
            results["j2d"].append(np.zeros((127, 2), dtype=np.float32))
            results["rotvec"].append(np.zeros((53, 3), dtype=np.float32))
            results["shape"].append(np.zeros((10,), dtype=np.float32))
            results["transl"].append(np.zeros((3,), dtype=np.float32))
            results["num_persons"].append(0)

        if t % 100 == 0:
            n_detected = sum(1 for s in results["scores"] if s > 0)
            print(f"    Body pose frame {t}/{len(frames)} ({n_detected}/{t+1} detected)")

    results["scores"] = np.array(results["scores"], dtype=np.float32)
    results["j3d"] = np.stack(results["j3d"])
    results["j2d"] = np.stack(results["j2d"])
    results["rotvec"] = np.stack(results["rotvec"])
    results["shape"] = np.stack(results["shape"])
    results["transl"] = np.stack(results["transl"])
    results["num_persons"] = np.array(results["num_persons"], dtype=np.int32)

    detection_rate = (results["scores"] > 0).mean() * 100
    print(f"  Body pose detection rate: {detection_rate:.1f}%")
    return results


# ---------------------------------------------------------------------------
# Body pose annotation (SAM-3D-Body) — optimized
# ---------------------------------------------------------------------------

def _interpolate_body_results(sampled_results, sampled_indices, num_frames, array_keys):
    """Linearly interpolate body pose results between sampled keyframes.

    Uses vectorized numpy interp for speed. Linear interpolation is acceptable
    at 30fps with stride<=3 since angular change per step is small.
    """
    full_results = {}
    full_results["_model"] = sampled_results.get("_model")

    all_frame_indices = np.arange(num_frames, dtype=np.float64)
    sample_x = np.array(sampled_indices, dtype=np.float64)

    for key in sampled_results:
        if key == "_model":
            continue
        val = sampled_results[key]
        if key in ("scores", "num_persons"):
            # Nearest-neighbor for discrete values
            full = np.zeros(num_frames, dtype=val.dtype)
            for i in range(len(sampled_indices)):
                start = sampled_indices[i]
                end = sampled_indices[i + 1] if i + 1 < len(sampled_indices) else num_frames
                full[start:end] = val[i]
            full_results[key] = full
        else:
            # Vectorized linear interpolation
            shape = val.shape[1:] if val.ndim > 1 else ()
            flat = val.reshape(len(sampled_indices), -1).astype(np.float64)
            D = flat.shape[1]
            full_flat = np.empty((num_frames, D), dtype=np.float32)
            for d in range(D):
                full_flat[:, d] = np.interp(all_frame_indices, sample_x, flat[:, d])
            if shape:
                full_results[key] = full_flat.reshape(num_frames, *shape)
            else:
                full_results[key] = full_flat.squeeze(-1)

    return full_results


_sam3d_cache = {}

def _load_sam3d_cpu(sam3d_dir, checkpoint_dir):
    """CPU-only phase of SAM-3D-Body loading (constructor + checkpoint).

    This is the slow part (~25s) that can run in a background thread while
    the GPU is busy with SLAM.
    """
    sys.path.insert(0, sam3d_dir)
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.config import get_config

    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    mhr_path = os.path.join(checkpoint_dir, "assets", "mhr_model.pt")
    config_path = os.path.join(checkpoint_dir, "model_config.yaml")

    cfg = get_config(config_path)
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    cfg.freeze()

    model = SAM3DBody(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # assign=True avoids copying weights (16s → 0s)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False, assign=True)

    print("  [Background] SAM-3D-Body CPU construction done")
    return model, cfg


def _load_sam3d_cached(sam3d_dir, checkpoint_dir, device, compile_model=False,
                       decoder_layers=None, _preloaded=None):
    """Load SAM-3D-Body model with caching to avoid repeated 10s+ construction.

    Args:
        _preloaded: Optional (model, cfg) tuple from _load_sam3d_cpu() to skip
                    the CPU-heavy construction phase (already done in background).
    """
    cache_key = (sam3d_dir, checkpoint_dir, device, compile_model, decoder_layers)
    if cache_key in _sam3d_cache:
        print("  Using cached SAM-3D-Body model")
        return _sam3d_cache[cache_key]

    if _preloaded is not None:
        model, cfg = _preloaded
        print("  Using pre-loaded SAM-3D-Body model (CPU phase was overlapped)")
    else:
        print("  Loading SAM-3D-Body model (mmap=True)...")
        model, cfg = _load_sam3d_cpu(sam3d_dir, checkpoint_dir)

    model.to(device).eval()
    model.backbone.half()
    model.backbone_dtype = torch.float16

    # Optionally reduce decoder layers (4 layers saves ~5% with <2mm quality loss)
    if decoder_layers is not None and decoder_layers < len(model.decoder.layers):
        print(f"  Reducing decoder from {len(model.decoder.layers)} to {decoder_layers} layers")
        model.decoder.layers = torch.nn.ModuleList(list(model.decoder.layers)[:decoder_layers])

    # Apply fused Triton kernels for ~7% backbone speedup (zero warmup cost)
    try:
        from triton_kernels import patch_backbone_with_triton
        n = patch_backbone_with_triton(model.backbone)
        if n > 0:
            print(f"  Applied fused Triton kernels to {n} backbone blocks")
    except Exception as e:
        print(f"  Triton kernel patching failed (non-fatal): {e}")

    # Optionally compile for ~30% faster inference (adds ~40s warmup)
    if compile_model:
        print("  Compiling model with torch.compile (this takes ~40s on first batch)...")
        model.forward_step = torch.compile(model.forward_step, mode="default", fullgraph=False)

    sys.path.insert(0, sam3d_dir)
    from sam_3d_body import SAM3DBodyEstimator
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    _sam3d_cache[cache_key] = (model, cfg, estimator)
    return model, cfg, estimator


def run_body_pose_sam3d(frames, fps, width, height, sam3d_dir, checkpoint_dir,
                        device="cuda", inference_type="body", stride=1, batch_size=32,
                        compile_model=False, decoder_layers=None):
    """Run SAM-3D-Body pose estimation on video frames.

    Args:
        inference_type: "body" (fast, ~20ms/frame batched) or "full" (body+hands, ~289ms/frame)
        stride: Process every Nth frame and interpolate between (1=all, 2=2x faster)
        batch_size: Frames per GPU batch for body-only mode (ignored for full)
        compile_model: Use torch.compile for ~30% faster inference (adds ~40s warmup)
        decoder_layers: Override number of decoder layers (None=all 6)
    """
    os.environ["XFORMERS_DISABLED"] = "1"

    sys.path.insert(0, sam3d_dir)
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.data.utils.prepare_batch import prepare_batch

    model, cfg, estimator = _load_sam3d_cached(sam3d_dir, checkpoint_dir, device,
                                                compile_model=compile_model,
                                                decoder_layers=decoder_layers)

    full_bbox = np.array([[0, 0, width, height]], dtype=np.float32)

    # Determine which frames to process
    num_frames = len(frames)
    sampled_indices = list(range(0, num_frames, stride))
    if sampled_indices[-1] != num_frames - 1:
        sampled_indices.append(num_frames - 1)
    n_sampled = len(sampled_indices)

    # Batched path only works for body-only (hand inference needs per-frame image access)
    use_batched = (inference_type == "body" and batch_size > 1)

    if stride > 1:
        print(f"  Stride={stride}: processing {n_sampled}/{num_frames} frames, interpolating rest")
    if use_batched:
        print(f"  Batched inference: bs={batch_size}, fp16 backbone")

    # Pre-compute camera intrinsics
    img_size = max(height, width)
    cam_int = torch.tensor(
        [[img_size, 0, width / 2], [0, img_size, height / 2], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )

    results = {
        "scores": [], "j3d": [], "j2d": [], "rotvec": [],
        "shape": [], "transl": [], "num_persons": [],
        "hand_pose_params": [], "scale_params": [], "expr_params": [],
    }

    t_start = time.perf_counter()

    if use_batched:
        # --- Batched body-only: N frames per forward pass (~20ms/frame at bs=8) ---
        prep_q = Queue(maxsize=2)

        def prep_worker():
            for chunk_start in range(0, n_sampled, batch_size):
                chunk_indices = sampled_indices[chunk_start:chunk_start + batch_size]
                chunk_frames = [frames[i] for i in chunk_indices]
                single_batches = [prepare_batch(f, estimator.transform, full_bbox) for f in chunk_frames]
                multi = {}
                for k in single_batches[0]:
                    v = single_batches[0][k]
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        try:
                            multi[k] = torch.cat([b[k] for b in single_batches], dim=0)
                        except Exception:
                            multi[k] = v
                    else:
                        multi[k] = v
                prep_q.put((chunk_indices, multi))
            prep_q.put(None)

        prep_thread = Thread(target=prep_worker, daemon=True)
        prep_thread.start()

        processed = 0
        while True:
            item = prep_q.get()
            if item is None:
                break
            chunk_indices, multi_batch = item
            bs = len(chunk_indices)

            for k in multi_batch:
                if isinstance(multi_batch[k], torch.Tensor):
                    multi_batch[k] = multi_batch[k].to(device, non_blocking=True)
            multi_batch["cam_int"] = cam_int.unsqueeze(0).expand(bs, -1, -1)
            model._initialize_batch(multi_batch)

            with torch.no_grad():
                pose_output = model.forward_step(multi_batch, decoder_type="body")

            mhr = pose_output["mhr"]
            for i in range(bs):
                results["scores"].append(1.0)
                results["j3d"].append(mhr["pred_keypoints_3d"][i].cpu().numpy())
                results["j2d"].append(mhr["pred_keypoints_2d"][i].cpu().numpy())
                results["rotvec"].append(mhr["body_pose"][i].cpu().numpy())
                results["shape"].append(mhr["shape"][i].cpu().numpy())
                results["transl"].append(mhr["pred_cam_t"][i].cpu().numpy())
                results["num_persons"].append(1)
                results["hand_pose_params"].append(mhr["hand"][i].cpu().numpy())
                results["scale_params"].append(mhr["scale"][i].cpu().numpy())
                results["expr_params"].append(mhr["face"][i].cpu().numpy())

            processed += bs
            if processed % 100 < bs or processed == bs:
                elapsed = time.perf_counter() - t_start
                rate = processed / elapsed
                eta = (n_sampled - processed) / rate if rate > 0 else 0
                print(f"    {processed}/{n_sampled} frames ({rate:.1f} fps, ETA {eta:.0f}s)")

        prep_thread.join()

    else:
        # --- Unbatched path: one frame at a time (for full inference or bs=1) ---
        _np = lambda v: v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v, dtype=np.float32)
        for si, idx in enumerate(sampled_indices):
            with torch.no_grad():
                humans = estimator.process_one_image(
                    frames[idx], bboxes=full_bbox, inference_type=inference_type,
                )
            if humans and len(humans) > 0:
                best = humans[0]
                results["scores"].append(1.0)
                results["j3d"].append(_np(best["pred_keypoints_3d"]))
                results["j2d"].append(_np(best["pred_keypoints_2d"]))
                results["rotvec"].append(_np(best["body_pose_params"]))
                results["shape"].append(_np(best["shape_params"]))
                results["transl"].append(_np(best["pred_cam_t"]))
                results["num_persons"].append(1)
                results["hand_pose_params"].append(_np(best["hand_pose_params"]))
                results["scale_params"].append(_np(best["scale_params"]))
                results["expr_params"].append(_np(best["expr_params"]))
            else:
                results["scores"].append(0.0)
                results["j3d"].append(np.zeros((70, 3), dtype=np.float32))
                results["j2d"].append(np.zeros((70, 2), dtype=np.float32))
                results["rotvec"].append(np.zeros((133,), dtype=np.float32))
                results["shape"].append(np.zeros((45,), dtype=np.float32))
                results["transl"].append(np.zeros((3,), dtype=np.float32))
                results["num_persons"].append(0)
                results["hand_pose_params"].append(np.zeros((108,), dtype=np.float32))
                results["scale_params"].append(np.zeros((28,), dtype=np.float32))
                results["expr_params"].append(np.zeros((72,), dtype=np.float32))

            if (si + 1) % 50 == 0 or si == 0:
                elapsed = time.perf_counter() - t_start
                rate = (si + 1) / elapsed
                eta = (n_sampled - si - 1) / rate if rate > 0 else 0
                print(f"    Frame {idx}/{num_frames} ({si+1}/{n_sampled}, {rate:.1f} fps, ETA {eta:.0f}s)")

    # Stack sampled results
    for key in results:
        if key == "num_persons":
            results[key] = np.array(results[key], dtype=np.int32)
        elif key == "scores":
            results[key] = np.array(results[key], dtype=np.float32)
        else:
            results[key] = np.stack(results[key])

    results["_model"] = "sam3d"

    if stride > 1:
        array_keys = [k for k in results if k not in ("_model", "scores", "num_persons")]
        results = _interpolate_body_results(results, sampled_indices, num_frames, array_keys)

    detection_rate = (results["scores"] > 0).mean() * 100
    print(f"  Body pose detection rate: {detection_rate:.1f}%")

    return results


# ---------------------------------------------------------------------------
# LeRobot v3.0 dataset writer
# ---------------------------------------------------------------------------

def write_lerobot_dataset(
    output_dir: str,
    video_path: str,
    slam_result: dict,
    hand_result: dict,
    fps: float,
    num_frames: int,
):
    """Assemble annotations into a LeRobot v3.0 dataset."""
    output_dir = Path(output_dir)

    video_dir = output_dir / "videos" / "observation.video" / "chunk-000"
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"

    for d in [video_dir, data_dir, meta_dir, episodes_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy video
    video_dest = video_dir / "file-000.mp4"
    shutil.copy2(video_path, video_dest)
    print(f"  Video: {video_dest}")

    # 2. Build per-frame data
    frame_indices = np.arange(num_frames, dtype=np.int64)
    timestamps = frame_indices / fps
    episode_indices = np.zeros(num_frames, dtype=np.int64)

    data = {
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "timestamp": timestamps,
    }

    # SLAM columns
    if slam_result is not None:
        poses = slam_result["poses"]  # [N, 7]
        if len(poses) >= num_frames:
            poses = poses[:num_frames]
        else:
            pad = np.tile(poses[-1:], (num_frames - len(poses), 1))
            poses = np.vstack([poses, pad])

        for i, name in enumerate(["tx", "ty", "tz", "qx", "qy", "qz", "qw"]):
            data[f"slam.pose.{name}"] = poses[:, i].astype(np.float32)

        intrinsics = slam_result["intrinsics"]
        for i, name in enumerate(["fx", "fy", "cx", "cy"]):
            data[f"slam.intrinsics.{name}"] = np.full(num_frames, intrinsics[i], dtype=np.float32)

    # Hand pose columns (EgoVerse format: 21 keypoints per hand)
    if hand_result is not None:
        for side in ["left", "right"]:
            kp3d = hand_result[f"{side}_hand_keypoints_3d"]  # [N, 21, 3]
            data[f"hand.{side}.keypoints_3d"] = [kp3d[i].flatten().tolist() for i in range(num_frames)]

            kp2d = hand_result[f"{side}_hand_keypoints_2d"]  # [N, 21, 2]
            data[f"hand.{side}.keypoints_2d"] = [kp2d[i].flatten().tolist() for i in range(num_frames)]

            data[f"hand.{side}.detected"] = hand_result[f"{side}_hand_detected"].astype(np.float32)

    # 3. Write Parquet
    table = pa.table(data)
    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(table, parquet_path)
    print(f"  Data: {parquet_path} ({num_frames} rows)")

    # 4. Write episode metadata
    episode_data = {
        "episode_index": [0],
        "length": [num_frames],
        "task_index": [0],
    }
    pq.write_table(pa.table(episode_data), episodes_dir / "file-000.parquet")

    # 5. Write tasks
    pq.write_table(pa.table({"task_index": [0], "task": ["default"]}), meta_dir / "tasks.parquet")

    # 6. Write info.json
    features = {
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "observation.video": {
            "dtype": "video",
            "shape": [],
            "video_info": {
                "video.fps": fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
            },
        },
    }

    if slam_result is not None:
        features["slam.pose"] = {"dtype": "float32", "shape": [7]}
        features["slam.intrinsics"] = {"dtype": "float32", "shape": [4]}

    if hand_result is not None:
        for side in ["left", "right"]:
            features[f"hand.{side}.keypoints_3d"] = {"dtype": "float32", "shape": [21, 3]}
            features[f"hand.{side}.keypoints_2d"] = {"dtype": "float32", "shape": [21, 2]}
            features[f"hand.{side}.detected"] = {"dtype": "float32", "shape": [1]}

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "total_episodes": 1,
        "total_frames": num_frames,
        "fps": fps,
        "features": features,
    }

    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Meta: {info_path}")

    return str(output_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Annotate video with SLAM + body pose for LeRobot")
    parser.add_argument("input", help="Input video file (e.g., IMG_1443.MOV)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output dataset directory")
    parser.add_argument("--skip-video-convert", action="store_true", help="Skip video format conversion")
    parser.add_argument("--skip-slam", action="store_true", help="Skip SLAM annotation")
    parser.add_argument("--skip-hands", action="store_true", help="Skip hand pose annotation")
    parser.add_argument("--hand-stride", type=int, default=1,
                        help="Process every Nth frame for hand pose, interpolate rest (1=all, 2=2x faster, 3=3x)")
    parser.add_argument("--hand-det-conf", type=float, default=0.3,
                        help="YOLO hand detection confidence threshold")
    parser.add_argument("--fast-traj", action="store_true",
                        help="Use fast linear/slerp trajectory interpolation instead of NN refinement (saves ~3s)")
    parser.add_argument("--slam-backend-steps", type=int, nargs=2, default=[7, 12],
                        help="Backend optimization steps (default: 7 12)")
    parser.add_argument("--droid-weights", default="/workspace/DROID-SLAM/checkpoints/droid.pth")
    parser.add_argument("--wilor-dir", default="/workspace/WiLoR")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_path = str(Path(args.input).resolve())
    if args.output_dir is None:
        args.output_dir = str(Path(input_path).parent / f"{Path(input_path).stem}_dataset")

    print(f"=" * 60)
    print(f"Annotation Pipeline")
    print(f"  Input: {input_path}")
    print(f"  Output: {args.output_dir}")
    if not args.skip_hands:
        print(f"  Hands: WiLoR (stride={args.hand_stride}, conf={args.hand_det_conf})")
    print(f"=" * 60)

    total_start = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=2)

    os.makedirs(args.output_dir, exist_ok=True)
    video_output = str(Path(args.output_dir) / "converted.mp4")
    phase1_time = 0
    video_convert_future = None

    # Start WiLoR CPU loading in background (overlaps with SLAM)
    wilor_cpu_future = None
    if not args.skip_hands:
        print(f"\n[Background] Starting WiLoR CPU load...")
        wilor_load_start = time.perf_counter()
        wilor_cpu_future = executor.submit(_load_wilor_cpu, args.wilor_dir)

    # Phase 2: Decode + SLAM (uses CPU decode + GPU tracking)
    slam_result = None
    frames = None
    num_frames = 0
    phase2_time = 0
    decode_time = 0
    if not args.skip_slam:
        print(f"\n[Phase 2] Decode + SLAM (DROID-SLAM)")
        phase2_start = time.perf_counter()

        torch.multiprocessing.set_start_method("spawn", force=True)
        fps, width, height, _ = get_video_metadata(input_path)

        # collect_rgb=True when hand inference needs frames — avoids separate decode pass
        need_body_frames = not args.skip_hands
        if need_body_frames:
            slam_result, num_frames, frames = run_slam_native_decode(
                input_path, fps, width, height, args.droid_weights, args.device,
                backend_steps=tuple(args.slam_backend_steps),
                fast_traj=args.fast_traj,
                collect_rgb=True,
            )
            print(f"  RGB frames collected during SLAM ({len(frames)} frames, no re-decode needed)")
        else:
            slam_result, num_frames = run_slam_native_decode(
                input_path, fps, width, height, args.droid_weights, args.device,
                backend_steps=tuple(args.slam_backend_steps),
                fast_traj=args.fast_traj,
                collect_rgb=False,
            )

        phase2_time = time.perf_counter() - phase2_start
        print(f"  Decode+SLAM completed in {phase2_time:.1f}s ({num_frames/phase2_time:.0f} fps)")
        print(f"  Poses shape: {slam_result['poses'].shape}")

        torch.cuda.empty_cache()
    else:
        print(f"\n[Phase 2] SLAM: Skipped")

    # Decode frames for hand inference if not already available
    if frames is None and not args.skip_hands:
        print(f"\n[Decode] Reading video frames for hand inference...")
        decode_start = time.perf_counter()
        frames, fps, width, height, num_frames = decode_video_frames(input_path)
        decode_time = time.perf_counter() - decode_start
        print(f"  Decoded {num_frames} frames ({width}x{height} @ {fps:.1f}fps) in {decode_time:.1f}s")
    elif frames is None and args.skip_hands:
        if num_frames == 0:
            _, _, _, num_frames = get_video_metadata(input_path)

    # Collect pre-loaded WiLoR model if background load was started
    wilor_preloaded = None
    if wilor_cpu_future is not None:
        if wilor_cpu_future.done():
            print(f"  WiLoR CPU load finished (fully overlapped with SLAM)")
        else:
            print(f"  Waiting for WiLoR CPU load to finish...")
        wilor_preloaded = wilor_cpu_future.result()
        wilor_load_time = time.perf_counter() - wilor_load_start
        print(f"  WiLoR CPU load took {wilor_load_time:.1f}s")

    # Start video conversion in background (CPU-bound ffmpeg overlaps with GPU body inference)
    if not args.skip_video_convert:
        from convert_video import convert_video
        print(f"\n[Background] Starting video conversion...")
        phase1_start = time.perf_counter()
        video_convert_future = executor.submit(
            convert_video, input_path, video_output, fast=True)

    # Phase 3: Hand pose (WiLoR — 21 keypoints per hand, EgoVerse format)
    hand_result = None
    phase3_time = 0
    if not args.skip_hands:
        print(f"\n[Phase 3] Hand pose (WiLoR)")
        phase3_start = time.perf_counter()

        hand_result = run_hand_pose_wilor(
            frames, fps, width, height,
            args.wilor_dir, args.device,
            stride=args.hand_stride,
            det_conf=args.hand_det_conf,
            preloaded=wilor_preloaded,
        )

        phase3_time = time.perf_counter() - phase3_start
        effective_fps = num_frames / phase3_time if phase3_time > 0 else 0
        print(f"  Hand pose completed in {phase3_time:.1f}s ({effective_fps:.0f} fps effective)")

        torch.cuda.empty_cache()
    else:
        print(f"\n[Phase 3] Hand pose: Skipped")

    # Free decoded frames to release ~11GB RAM
    del frames

    # Collect video conversion result
    if video_convert_future is not None:
        if video_convert_future.done():
            print(f"\n  Video conversion finished (overlapped with body)")
        else:
            print(f"\n  Waiting for video conversion to finish...")
        video_output = video_convert_future.result()
        phase1_time = time.perf_counter() - phase1_start
    elif not args.skip_video_convert:
        from convert_video import convert_video
        print(f"\n[Video conversion]")
        phase1_start = time.perf_counter()
        video_output = convert_video(input_path, video_output, fast=True)
        phase1_time = time.perf_counter() - phase1_start

    # Phase 5: Assembly
    print(f"\n[Phase 4] Assembling LeRobot dataset")
    phase4_start = time.perf_counter()

    dataset_path = write_lerobot_dataset(
        args.output_dir, video_output, slam_result, hand_result, fps, num_frames,
    )

    phase4_time = time.perf_counter() - phase4_start
    total_time = time.perf_counter() - total_start
    executor.shutdown(wait=False)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"    SLAM:          {phase2_time:.1f}s (native C++ decode overlapped)")
    if decode_time > 0:
        print(f"    Hand decode:   {decode_time:.1f}s")
    print(f"    Hand pose:     {phase3_time:.1f}s (WiLoR, model load overlapped with SLAM)")
    print(f"    Video convert: {phase1_time:.1f}s")
    print(f"    Assembly:      {phase4_time:.1f}s")
    print(f"  Output: {dataset_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
