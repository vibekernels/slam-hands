#!/usr/bin/env python3
"""Annotate video with SLAM camera poses, hand pose estimation, and audio transcription.

Produces a LeRobot v3.0 compatible dataset following the EgoVerse format:
  - Video converted to LeRobot format (via convert_video.py)
  - SLAM camera poses per frame (DROID-SLAM) — 6-DoF head/camera motion
  - 3D hand pose per frame (WiLoR) — 21 keypoints per hand, left + right
  - Audio transcription per frame (NVIDIA Parakeet) — word-level timestamps

Usage:
    python3 annotate_pipeline.py /workspace/IMG_1443.MOV --output-dir /workspace/output_dataset
"""

import argparse
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from threading import Thread
from queue import Queue

# ---------------------------------------------------------------------------
# Suppress known-harmless warnings from third-party libraries.
# Real errors still surface via stderr/exceptions.
# ---------------------------------------------------------------------------
# timm: "Importing from timm.models.layers is deprecated"
warnings.filterwarnings("ignore", message="Importing from timm.models.layers", category=FutureWarning)
# smplx/MANO: "You are using a MANO model, with only 10 shape coefficients"
logging.getLogger("smplx.body_models").setLevel(logging.ERROR)
# PyTorch: "torch.cross without specifying dim"
warnings.filterwarnings("ignore", message="Using torch.cross without specifying the dim")
# NeMo: verbose info/warnings about training data, CUDA graphs, etc.
for _nemo_logger in ("nemo", "nemo.collections", "nemo.utils", "nemo_logger"):
    logging.getLogger(_nemo_logger).setLevel(logging.ERROR)

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# PyTorch 2.6+ defaults to weights_only=True in torch.load, which breaks
# loading YOLO/WiLoR checkpoints that contain custom classes. Patch globally.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

_SENTINEL = object()  # End-of-stream marker for concurrent frame queue


def _install_pytorch_lightning_stub():
    """Replace pytorch_lightning with a minimal stub for inference.

    WiLoR inherits from pl.LightningModule (which is just nn.Module + training
    helpers) and uses pl.utilities.rank_zero_only as a decorator. For inference
    we only need the base class and a passthrough decorator. This avoids the
    3.4s import of the full pytorch_lightning + lightning_fabric stack.
    """
    import torch.nn as nn

    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

    class _RankZero:
        @staticmethod
        def rank_zero_only(fn):
            return fn

    class _Utilities:
        rank_zero = _RankZero()
        @staticmethod
        def rank_zero_only(fn):
            return fn

    class _FakePL:
        LightningModule = _LightningModule
        utilities = _Utilities()

    sys.modules['pytorch_lightning'] = _FakePL()
    sys.modules['pytorch_lightning.utilities'] = _Utilities()

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
    try:
        nd = _get_native_decode()
        fps, width, height, num_frames = nd.AsyncVideoDecoder.get_metadata(video_path)
        if width > 0 and height > 0:
            return fps, width, height, int(num_frames)
        # Some HEVC containers don't expose dimensions in codecpar before decode
    except (ImportError, AttributeError):
        pass
    # Fallback to ffprobe
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         video_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    # Parse fps from r_frame_rate "30000/1001" or "30/1"
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    # nb_frames may be "N/A" for some containers
    nb = info.get("nb_frames", "N/A")
    if nb != "N/A":
        nf = int(nb)
    else:
        dur = float(info.get("duration", "0"))
        nf = int(dur * fps)
    return fps, w, h, nf


def get_iphone_intrinsics(width: int, height: int):
    """Estimate camera intrinsics for iPhone rear camera."""
    fx = fy = max(width, height) * 0.75
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# SLAM annotation (DROID-SLAM)
# ---------------------------------------------------------------------------


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


def _hand_worker_selfdecode(
    video_path, wilor_dir, det_conf, stride,
    result_path, num_frames_est,
):
    """Child process: decode video independently + run YOLO + WiLoR.

    Child decodes the video with OpenCV and runs YOLO + WiLoR concurrently with
    SLAM in the parent. This allows the parent to use slam_only=True
    (fast native C++ decoder) while the child does its own decode.

    Inherits all Python imports via fork — no re-import needed.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from concurrent.futures import ThreadPoolExecutor as TPE

    try:
        torch.cuda.set_device(0)

        sys.path.insert(0, wilor_dir)
        from ultralytics import YOLO

        with torch.no_grad():
            t_load = time.perf_counter()

            # ── Phase 1: CPU-only work (overlaps with SLAM tracking) ──
            # Build WiLoR on CPU thread while we wait for GPU signal
            wilor_load_result = [None]
            def _build_wilor_bg():
                wilor_load_result[0] = _load_wilor_cpu(wilor_dir)
            wilor_build_thread = Thread(target=_build_wilor_bg, daemon=True)
            wilor_build_thread.start()

            orig_cwd = os.getcwd()
            os.chdir(wilor_dir)
            try:
                from wilor.configs import get_config as _get_wilor_cfg
                _cfg = _get_wilor_cfg(
                    os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
                    update_cachedir=True,
                )
            finally:
                os.chdir(orig_cwd)
            mean = 255.0 * np.array(_cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
            std = 255.0 * np.array(_cfg.MODEL.IMAGE_STD, dtype=np.float32)
            bbox_shape = _cfg.MODEL.get('BBOX_SHAPE', [192, 256])
            img_size_cfg = _cfg.MODEL.IMAGE_SIZE
            rescale_factor = 2.0
            BATCH_SIZE = 48
            YOLO_BATCH = 16

            # ── GPU work (runs concurrently with SLAM in parent) ──
            hand_stream = torch.cuda.Stream()
            wilor_stream = torch.cuda.Stream()

            with torch.cuda.stream(hand_stream):
                detector = YOLO(
                    os.path.join(wilor_dir, 'pretrained_models', 'detector.pt'),
                ).to('cuda')

            wilor_build_thread.join()
            model_cpu, model_cfg = wilor_load_result[0]

            with torch.cuda.stream(wilor_stream):
                model = model_cpu.to('cuda').eval().half()
                model.backbone.skip_blocks = True
            del model_cpu
            t_ready = time.perf_counter()
            print(f"  [Hand proc] Ready in {t_ready-t_load:.1f}s", flush=True)

            # Pre-allocate output arrays
            max_frames = max(num_frames_est + 100, 4000)
            left_kp3d = np.full((max_frames, 21, 3), np.nan, dtype=np.float32)
            right_kp3d = np.full((max_frames, 21, 3), np.nan, dtype=np.float32)
            left_kp2d = np.full((max_frames, 21, 2), np.nan, dtype=np.float32)
            right_kp2d = np.full((max_frames, 21, 2), np.nan, dtype=np.float32)
            left_detected = np.zeros(max_frames, dtype=bool)
            right_detected = np.zeros(max_frames, dtype=bool)

            # WiLoR inference thread
            crop_queue = Queue(maxsize=256)
            n_hands_done = [0]
            wilor_error = []

            def _wilor_inference_loop():
                try:
                    torch.cuda.set_device(0)
                    with torch.cuda.stream(wilor_stream), torch.no_grad():
                        crop_buf_imgs = []
                        crop_buf_meta = []
                        while True:
                            item = crop_queue.get()
                            if item is _SENTINEL:
                                break
                            img_t, meta = item
                            crop_buf_imgs.append(img_t)
                            crop_buf_meta.append(meta)
                            if len(crop_buf_imgs) >= BATCH_SIZE:
                                _run_wilor_batch(
                                    crop_buf_imgs[:BATCH_SIZE],
                                    crop_buf_meta[:BATCH_SIZE],
                                    model, model_cfg, 'cuda',
                                    left_kp3d, right_kp3d,
                                    left_kp2d, right_kp2d,
                                    left_detected, right_detected,
                                )
                                n_hands_done[0] += BATCH_SIZE
                                crop_buf_imgs = crop_buf_imgs[BATCH_SIZE:]
                                crop_buf_meta = crop_buf_meta[BATCH_SIZE:]
                        if crop_buf_imgs:
                            _run_wilor_batch(
                                crop_buf_imgs, crop_buf_meta,
                                model, model_cfg, 'cuda',
                                left_kp3d, right_kp3d,
                                left_kp2d, right_kp2d,
                                left_detected, right_detected,
                            )
                            n_hands_done[0] += len(crop_buf_imgs)
                except Exception:
                    import traceback
                    wilor_error.append(traceback.format_exc())

            wilor_thread = Thread(target=_wilor_inference_loop, daemon=True)
            wilor_thread.start()

            # ── Decode + YOLO detection (streaming from native decoder) ──
            t_yolo_start = time.perf_counter()
            total_crops = 0
            yolo_batch = []
            sampled_indices = []

            prep_pool = TPE(max_workers=4)

            def _preprocess_and_enqueue(fi, img_bgr, center, scale, is_right, img_wh):
                img_tensor, bbox_size = _preprocess_hand_crop(
                    img_bgr, center, scale, bbox_shape, img_size_cfg, is_right, mean, std,
                )
                crop_queue.put((
                    torch.from_numpy(img_tensor).half(),
                    (fi, center.copy(), bbox_size, img_wh, is_right),
                ))

            def _flush_yolo():
                nonlocal total_crops
                if not yolo_batch:
                    return
                fis = [x[0] for x in yolo_batch]
                imgs = [x[1] for x in yolo_batch]
                with torch.cuda.stream(hand_stream):
                    results = detector(imgs, conf=det_conf, verbose=False)
                for i, (det_fi, dets) in enumerate(zip(fis, results)):
                    bboxes, is_right_list = [], []
                    for det in dets:
                        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                        is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
                        bboxes.append(bbox[:4].tolist())
                    if bboxes:
                        boxes = np.stack(bboxes).astype(np.float32)
                        right_arr = np.stack(is_right_list).astype(np.float32)
                        img_bgr = imgs[i]
                        img_h, img_w = img_bgr.shape[:2]
                        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
                        scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
                        img_wh = np.array([img_w, img_h], dtype=np.float32)
                        for k in range(len(boxes)):
                            prep_pool.submit(
                                _preprocess_and_enqueue,
                                det_fi, img_bgr, centers[k], scales[k],
                                right_arr[k], img_wh,
                            )
                        total_crops += len(bboxes)

            # Stream frames from OpenCV through YOLO
            # OpenCV uses its own thread pool — doesn't compete with
            # parent's native decoder for FFmpeg CPU threads.
            cap = cv2.VideoCapture(video_path)
            fi = 0
            while True:
                ret, img_bgr = cap.read()
                if not ret:
                    break
                if stride <= 1 or fi % stride == 0:
                    sampled_indices.append(fi)
                    yolo_batch.append((fi, img_bgr))
                    if len(yolo_batch) >= YOLO_BATCH:
                        _flush_yolo()
                        yolo_batch = []
                fi += 1
            cap.release()
            _flush_yolo()
            yolo_batch = []

            prep_pool.shutdown(wait=True)
            crop_queue.put(_SENTINEL)

            num_frames = fi
            t_yolo = time.perf_counter() - t_yolo_start
            print(
                f"  [Hand proc] Decode+YOLO: {t_yolo:.1f}s "
                f"({num_frames/max(t_yolo,0.001):.0f} fps), {total_crops} crops",
                flush=True,
            )

            wilor_thread.join()
            if wilor_error:
                raise RuntimeError(f"WiLoR failed:\n{wilor_error[0]}")

            # Trim to actual frame count
            left_kp3d = left_kp3d[:num_frames]
            right_kp3d = right_kp3d[:num_frames]
            left_kp2d = left_kp2d[:num_frames]
            right_kp2d = right_kp2d[:num_frames]
            left_detected = left_detected[:num_frames]
            right_detected = right_detected[:num_frames]

            # Interpolate if strided
            if stride > 1 and len(sampled_indices) > 1:
                if sampled_indices[-1] != num_frames - 1:
                    sampled_indices.append(num_frames - 1)
                for arr in [left_kp3d, right_kp3d, left_kp2d, right_kp2d]:
                    _interpolate_hand_keypoints(arr, sampled_indices, num_frames)
                for det_arr in [left_detected, right_detected]:
                    full = np.zeros(num_frames, dtype=bool)
                    full[sampled_indices] = det_arr[sampled_indices]
                    for i in range(len(sampled_indices) - 1):
                        a, b = sampled_indices[i], sampled_indices[i + 1]
                        mid = (a + b) // 2
                        full[a:mid+1] = det_arr[a]
                        full[mid+1:b] = det_arr[b]
                    det_arr[:] = full

            left_rate = left_detected.sum() / num_frames * 100
            right_rate = right_detected.sum() / num_frames * 100
            t_total = time.perf_counter() - t_yolo_start
            n_hands = n_hands_done[0]
            print(
                f"  [Hand proc] Done: {t_total:.1f}s, {n_hands} hands, "
                f"left={left_rate:.1f}% right={right_rate:.1f}%",
                flush=True,
            )

            np.savez(
                result_path,
                left_kp3d=left_kp3d, right_kp3d=right_kp3d,
                left_kp2d=left_kp2d, right_kp2d=right_kp2d,
                left_detected=left_detected, right_detected=right_detected,
                num_frames=np.array([num_frames]),
            )

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _hand_worker_cuda(
    video_path, det_conf, stride,
    result_path, num_frames_est,
):
    """Child process: run CUDA hand pipeline (YOLO + WiLoR + MANO).

    Uses the cuda_hand binary with --video mode which decodes the video
    internally via FFmpeg and writes binary results to a file.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        cuda_hand_dir = os.path.join(os.path.dirname(__file__), 'cuda_hand')
        cuda_hand_bin = os.path.join(cuda_hand_dir, 'cuda_hand')
        weights_dir = os.path.join(cuda_hand_dir, 'data', 'weights')
        cuda_output = result_path.replace('.npz', '_cuda.bin')

        if not os.path.exists(cuda_hand_bin):
            raise FileNotFoundError(f"CUDA hand binary not found: {cuda_hand_bin}")

        print(f"  [CUDA Hand] Starting: video={video_path}, stride={stride}", flush=True)
        t_start = time.perf_counter()

        proc = subprocess.Popen(
            [cuda_hand_bin, '--weights-dir', weights_dir,
             '--video', video_path, '--stride', str(stride),
             '--det-conf', str(det_conf), '--output', cuda_output],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        stderr = proc.stderr.read()
        proc.wait(timeout=300)
        t_total = time.perf_counter() - t_start

        if proc.returncode != 0:
            print(f"  [CUDA Hand] FAILED (rc={proc.returncode})")
            print(f"  stderr: {stderr.decode()[:2000]}")
            sys.exit(1)

        # Parse binary output: header is [num_results, total_frames, stride]
        with open(cuda_output, 'rb') as f:
            num_results = struct.unpack('i', f.read(4))[0]
            num_frames = struct.unpack('i', f.read(4))[0]
            out_stride = struct.unpack('i', f.read(4))[0]

            left_kp3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
            right_kp3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
            left_kp2d = np.zeros((num_frames, 21, 2), dtype=np.float32)
            right_kp2d = np.zeros((num_frames, 21, 2), dtype=np.float32)
            left_detected = np.zeros(num_frames, dtype=bool)
            right_detected = np.zeros(num_frames, dtype=bool)

            sampled_indices = []
            for _ in range(num_results):
                fi = struct.unpack('i', f.read(4))[0]
                left_det = struct.unpack('B', f.read(1))[0]
                right_det = struct.unpack('B', f.read(1))[0]
                l_kp3d = np.frombuffer(f.read(63 * 4), dtype=np.float32).reshape(21, 3)
                r_kp3d = np.frombuffer(f.read(63 * 4), dtype=np.float32).reshape(21, 3)
                l_kp2d = np.frombuffer(f.read(42 * 4), dtype=np.float32).reshape(21, 2)
                r_kp2d = np.frombuffer(f.read(42 * 4), dtype=np.float32).reshape(21, 2)
                sampled_indices.append(fi)
                if fi < num_frames:
                    if left_det:
                        left_kp3d[fi] = l_kp3d
                        left_kp2d[fi] = l_kp2d
                        left_detected[fi] = True
                    if right_det:
                        right_kp3d[fi] = r_kp3d
                        right_kp2d[fi] = r_kp2d
                        right_detected[fi] = True

        # Interpolate if strided
        if out_stride > 1 and len(sampled_indices) > 1:
            if sampled_indices[-1] != num_frames - 1:
                sampled_indices.append(num_frames - 1)
            for arr in [left_kp3d, right_kp3d, left_kp2d, right_kp2d]:
                _interpolate_hand_keypoints(arr, sampled_indices, num_frames)
            for det_arr in [left_detected, right_detected]:
                full = np.zeros(num_frames, dtype=bool)
                full[sampled_indices] = det_arr[sampled_indices]
                for i in range(len(sampled_indices) - 1):
                    a, b = sampled_indices[i], sampled_indices[i + 1]
                    mid = (a + b) // 2
                    full[a:mid+1] = det_arr[a]
                    full[mid+1:b] = det_arr[b]
                det_arr[:] = full

        left_rate = left_detected.sum() / num_frames * 100
        right_rate = right_detected.sum() / num_frames * 100
        n_hands = left_detected.sum() + right_detected.sum()
        print(
            f"  [CUDA Hand] Done: {t_total:.1f}s, {n_hands} detections, "
            f"left={left_rate:.1f}% right={right_rate:.1f}%",
            flush=True,
        )
        print(f"  stderr: {stderr.decode()[:500]}", flush=True)

        np.savez(
            result_path,
            left_kp3d=left_kp3d, right_kp3d=right_kp3d,
            left_kp2d=left_kp2d, right_kp2d=right_kp2d,
            left_detected=left_detected, right_detected=right_detected,
            num_frames=np.array([num_frames]),
        )

        # Clean up
        try:
            os.remove(cuda_output)
        except OSError:
            pass

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_slam_and_hand_split(
    video_path, fps, width, height, weights_path, wilor_dir,
    device="cuda", backend_steps=(7, 12), fast_traj=False,
    det_conf=0.3, stride=1, post_fork_fn=None, num_frames_est=0,
):
    """Run DROID-SLAM and WiLoR in separate processes with independent decoding.

    The parent uses slam_only=True (fast: no full-res decode/resize/copy),
    and the child decodes the video independently. This removes ~8s of
    full-res decode + 6MB/frame shared memory copy from the parent's
    critical path.

    Returns (slam_result, hand_result, num_frames).
    """
    import multiprocessing as mp

    droid_slam_path = os.path.join(os.path.dirname(weights_path), "..", "droid_slam")
    nd = _get_native_decode()

    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8

    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics_scaled = torch.as_tensor([
        fx * (w1 / w0), fy * (h1 / h0), cx * (w1 / w0), cy * (h1 / h0),
    ])

    num_frames_est = num_frames_est if num_frames_est > 0 else 2000
    result_path = f"/tmp/hand_result_{os.getpid()}.npz"

    mp_ctx = mp.get_context('fork')

    # Fork child BEFORE any CUDA calls
    child = mp_ctx.Process(
        target=_hand_worker_cuda,
        args=(
            video_path, det_conf, stride,
            result_path, num_frames_est,
        ),
    )
    child.start()

    try:
        # Import Droid AFTER fork
        sys.path.insert(0, droid_slam_path)
        from droid import Droid

        slam_args = argparse.Namespace(
            weights=weights_path, buffer=512, image_size=[h1, w1],
            disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
            warmup=8, keyframe_thresh=4.0,
            frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
            backend_thresh=22.0, backend_radius=2, backend_nms=3,
            upsample=False, frontend_device=device, backend_device=device,
        )

        # Start decoder in slam_only mode (no full-res RGB decode/resize)
        decoder = nd.AsyncVideoDecoder()
        decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)

        # Start background tasks after fork
        if post_fork_fn is not None:
            post_fork_fn()

        droid = Droid(slam_args)
        print(f"  SLAM resolution: {w1}x{h1} (split mode, slam_only)")

        # ── SLAM tracking (no shared memory overhead) ──
        t_track_start = time.perf_counter()
        slam_tensors = [] if not fast_traj else None
        t = 0
        while True:
            result = decoder.get_next()
            if result is None:
                break
            _, slam_bgr = result
            tensor = torch.from_numpy(slam_bgr).permute(2, 0, 1).unsqueeze(0).cuda()
            if slam_tensors is not None:
                slam_tensors.append(tensor)
            droid.track(t, tensor, intrinsics=intrinsics_scaled)
            t += 1

        decoder.stop()
        num_frames = t

        t_track = time.perf_counter() - t_track_start
        print(f"  SLAM tracking: {t_track:.1f}s ({num_frames/t_track:.0f} fps)")

        # ── Backend optimization ──
        print(f"  Backend optimization ({backend_steps[0]}+{backend_steps[1]} steps)...")
        del droid.frontend
        torch.cuda.empty_cache()

        t_backend_start = time.perf_counter()
        droid.backend(backend_steps[0])
        torch.cuda.empty_cache()
        droid.backend(backend_steps[1])
        t_backend = time.perf_counter() - t_backend_start
        print(f"  Backend: {t_backend:.1f}s")

        # ── Trajectory fill ──
        if fast_traj:
            torch.cuda.empty_cache()
            t_traj_start = time.perf_counter()
            N = droid.video.counter.value
            kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
            import lietorch
            kf_poses_se3 = lietorch.SE3(droid.video.poses[:N])
            kf_poses_raw = kf_poses_se3.inv().data.cpu().numpy()
            all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_raw, num_frames)
            t_traj = time.perf_counter() - t_traj_start
            print(f"  Trajectory fill: {t_traj:.2f}s ({N} keyframes -> {num_frames} frames)")
        else:
            t_traj_start = time.perf_counter()
            def frame_stream():
                for i, tensor_item in enumerate(slam_tensors):
                    yield i, tensor_item, intrinsics_scaled
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

        # ── Wait for child process ──
        t_join_start = time.perf_counter()
        if child.is_alive():
            print(f"  Waiting for hand worker process to finish...")
        child.join(timeout=180)
        t_join = time.perf_counter() - t_join_start

        if child.exitcode != 0:
            raise RuntimeError(
                f"Hand worker process failed with exit code {child.exitcode}"
            )

        data = np.load(result_path)
        hand_num_frames = int(data["num_frames"][0]) if "num_frames" in data else num_frames
        hand_result = {
            "_model": "wilor", "_done": True,
            "left_hand_keypoints_3d": data["left_kp3d"],
            "right_hand_keypoints_3d": data["right_kp3d"],
            "left_hand_keypoints_2d": data["left_kp2d"],
            "right_hand_keypoints_2d": data["right_kp2d"],
            "left_hand_detected": data["left_detected"],
            "right_hand_detected": data["right_detected"],
        }
        if os.path.exists(result_path):
            os.remove(result_path)

        total = time.perf_counter() - t_track_start
        slam_total = t_track + t_backend + t_traj
        print(f"  Split total: {total:.1f}s (SLAM {slam_total:.1f}s, join wait {t_join:.1f}s)")

        return slam_result, hand_result, num_frames

    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Hand pose estimation (WiLoR) — EgoVerse format: 21 keypoints per hand
# ---------------------------------------------------------------------------


def _load_wilor_cpu(wilor_dir):
    """Load WiLoR model on CPU in background thread (for overlap with SLAM).

    Uses mmap + assign=True to avoid copying 2.5 GB of weights, reducing
    load_state_dict from ~15 s to ~0.3 s and minimizing GIL contention so
    SLAM + YOLO can run concurrently on the GPU without slowdown.
    """
    sys.path.insert(0, wilor_dir)
    orig_cwd = os.getcwd()
    os.chdir(wilor_dir)
    try:
        from wilor.configs import get_config
        from wilor.models.wilor import WiLoR

        model_cfg = get_config(
            os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
            update_cachedir=True,
        )
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE == 256:
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        _mano_dir = os.path.join(wilor_dir, 'mano_data')
        model_cfg.MANO.DATA_DIR = _mano_dir + '/'
        model_cfg.MANO.MODEL_PATH = _mano_dir + '/'
        model_cfg.MANO.MEAN_PARAMS = os.path.join(_mano_dir, 'mano_mean_params.npz')
        model_cfg.freeze()

        # Suppress smplx "only 10 shape coefficients" print during MANO init
        with open(os.devnull, 'w') as _devnull:
            _saved_stdout = sys.stdout
            sys.stdout = _devnull
            try:
                model = WiLoR(cfg=model_cfg, init_renderer=False)
            finally:
                sys.stdout = _saved_stdout

        # Fast mmap load: tensors stay as memory-mapped views (no copy)
        sd_path = os.path.join(wilor_dir, 'pretrained_models', 'wilor_final_statedict.pt')
        ckpt_path = os.path.join(wilor_dir, 'pretrained_models', 'wilor_final.ckpt')
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location='cpu', mmap=True, weights_only=True)
        else:
            sd = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=False)['state_dict']
        model.load_state_dict(sd, strict=False, assign=True)
    finally:
        os.chdir(orig_cwd)
    return model, model_cfg


def _preprocess_hand_crop(img_bgr, center, scale, bbox_shape, img_size, is_right, mean, std):
    """Fast hand crop preprocessing — replaces ViTDetDataset.__getitem__.

    Uses cv2.GaussianBlur instead of skimage.gaussian (3-5x faster),
    avoids full image copy when no blur needed, and uses vectorized normalization.
    Returns (img_tensor [3,H,W] float16, box_center, box_size, img_size_arr, right_val).
    """
    from wilor.datasets.utils import expand_to_aspect_ratio, gen_trans_from_patch_cv

    center_x, center_y = center
    bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max()
    patch_width = patch_height = img_size

    flip = is_right == 0

    # Blur to avoid aliasing — use cv2.GaussianBlur (much faster than skimage.gaussian)
    downsampling_factor = (bbox_size * 1.0) / patch_width / 2.0
    if downsampling_factor > 1.1:
        ksize = int(np.ceil((downsampling_factor - 1))) | 1  # ensure odd
        if ksize < 3:
            ksize = 3
        cvimg = cv2.GaussianBlur(img_bgr, (ksize, ksize), (downsampling_factor - 1) / 2)
    else:
        cvimg = img_bgr  # no copy needed — warpAffine doesn't modify input

    img_height, img_width = cvimg.shape[:2]
    cx, cy = center_x, center_y
    if flip:
        cvimg = cvimg[:, ::-1, :]
        cx = img_width - cx - 1

    trans = gen_trans_from_patch_cv(cx, cy, bbox_size, bbox_size,
                                   patch_width, patch_height, 1.0, 0)
    img_patch = cv2.warpAffine(cvimg, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # BGR→RGB, HWC→CHW, normalize — all vectorized
    img_patch = img_patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img_patch = (img_patch - mean[:, None, None]) / std[:, None, None]

    return img_patch, bbox_size


def run_hand_pose_wilor(frames, fps, width, height, wilor_dir, device="cuda",
                        stride=1, det_conf=0.3, preloaded=None):
    """Run WiLoR hand pose estimation on video frames.

    Uses two-pass architecture for speed:
    1. YOLO detection on all frames (batched)
    2. Cross-frame batched WiLoR inference (bs=48)

    Returns dict with per-frame hand pose data in EgoVerse format.
    """
    sys.path.insert(0, wilor_dir)
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    from ultralytics import YOLO

    num_frames = len(frames)

    # Load model
    if preloaded is not None:
        model_cpu, model_cfg = preloaded
        model = model_cpu.to(device).eval().half()
        model.backbone.skip_blocks = True
        detector = YOLO(os.path.join(wilor_dir, 'pretrained_models', 'detector.pt')).to(device)
    else:
        from wilor.models import load_wilor
        orig_cwd = os.getcwd()
        os.chdir(wilor_dir)
        try:
            model, model_cfg = load_wilor(
                checkpoint_path=os.path.join(wilor_dir, 'pretrained_models', 'wilor_final.ckpt'),
                cfg_path=os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
            )
        finally:
            os.chdir(orig_cwd)
        model = model.to(device).eval().half()
        model.backbone.skip_blocks = True
        detector = YOLO(os.path.join(wilor_dir, 'pretrained_models', 'detector.pt')).to(device)

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

    # Precompute config values
    mean = 255.0 * np.array(model_cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
    std = 255.0 * np.array(model_cfg.MODEL.IMAGE_STD, dtype=np.float32)
    bbox_shape = model_cfg.MODEL.get('BBOX_SHAPE', None)
    img_size_cfg = model_cfg.MODEL.IMAGE_SIZE
    rescale_factor = 2.0

    t_start = time.perf_counter()

    # ── Pass 1: YOLO detection on all frames (batched) ──
    print("  Pass 1/2: YOLO hand detection...")
    frame_dets = []
    total_crops = 0
    YOLO_BATCH = 32

    for batch_start in range(0, len(sampled_indices), YOLO_BATCH):
        batch_fis = sampled_indices[batch_start:batch_start + YOLO_BATCH]
        batch_imgs = [frames[fi][:, :, ::-1].copy() for fi in batch_fis]

        results = detector(batch_imgs, conf=det_conf, verbose=False)
        for fi, dets in zip(batch_fis, results):
            bboxes = []
            is_right_list = []
            for det in dets:
                bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(bbox[:4].tolist())
            if len(bboxes) > 0:
                boxes = np.stack(bboxes).astype(np.float32)
                right = np.stack(is_right_list).astype(np.float32)
                frame_dets.append((fi, boxes, right))
                total_crops += len(bboxes)

    t_det = time.perf_counter() - t_start
    print(f"    YOLO: {t_det:.1f}s ({len(sampled_indices)/t_det:.0f} fps), {total_crops} hand crops across {len(frame_dets)} frames")

    if total_crops == 0:
        print("  No hands detected in any frame")
        return {
            "_model": "wilor",
            "left_hand_keypoints_3d": left_kp3d,
            "right_hand_keypoints_3d": right_kp3d,
            "left_hand_keypoints_2d": left_kp2d,
            "right_hand_keypoints_2d": right_kp2d,
            "left_hand_detected": left_detected,
            "right_hand_detected": right_detected,
        }

    # ── Pass 2: Cross-frame batched WiLoR inference ──
    # Pipeline: CPU preprocessing in thread pool overlapped with GPU inference
    print(f"  Pass 2/2: WiLoR inference ({total_crops} crops, batch_size=48)...")
    t_wilor_start = time.perf_counter()
    BATCH_SIZE = 48
    n_hands = 0

    # Flatten all crops into a list of (fi, img_bgr, center, scale, is_right, img_wh)
    all_crop_specs = []
    for fi, boxes, right_arr in frame_dets:
        img_bgr = frames[fi][:, :, ::-1]  # RGB→BGR view (no copy)
        img_h, img_w = img_bgr.shape[:2]
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        img_wh = np.array([img_w, img_h], dtype=np.float32)
        for k in range(len(boxes)):
            all_crop_specs.append((fi, img_bgr, centers[k], scales[k], right_arr[k], img_wh))

    def _preprocess_one(spec):
        fi, img_bgr, center, scale, is_right, img_wh = spec
        img_tensor, bbox_size = _preprocess_hand_crop(
            img_bgr, center, scale, bbox_shape, img_size_cfg, is_right, mean, std
        )
        return (
            torch.from_numpy(img_tensor).half(),
            (fi, center.copy(), bbox_size, img_wh, is_right),
        )

    # Use thread pool for CPU preprocessing, overlap with GPU batches
    from concurrent.futures import ThreadPoolExecutor as TPE
    with TPE(max_workers=4) as prep_pool:
        # Submit all preprocessing jobs
        futures = [prep_pool.submit(_preprocess_one, spec) for spec in all_crop_specs]

        crop_buf_imgs = []
        crop_buf_meta = []
        for fut in futures:
            img_t, meta = fut.result()
            crop_buf_imgs.append(img_t)
            crop_buf_meta.append(meta)

            if len(crop_buf_imgs) >= BATCH_SIZE:
                _run_wilor_batch(
                    crop_buf_imgs[:BATCH_SIZE], crop_buf_meta[:BATCH_SIZE],
                    model, model_cfg, device,
                    left_kp3d, right_kp3d, left_kp2d, right_kp2d,
                    left_detected, right_detected,
                )
                n_hands += BATCH_SIZE
                crop_buf_imgs = crop_buf_imgs[BATCH_SIZE:]
                crop_buf_meta = crop_buf_meta[BATCH_SIZE:]

    # Flush remaining crops
    if crop_buf_imgs:
        _run_wilor_batch(
            crop_buf_imgs, crop_buf_meta,
            model, model_cfg, device,
            left_kp3d, right_kp3d, left_kp2d, right_kp2d,
            left_detected, right_detected,
        )
        n_hands += len(crop_buf_imgs)

    t_wilor = time.perf_counter() - t_wilor_start
    elapsed = time.perf_counter() - t_start
    print(f"    WiLoR: {t_wilor:.1f}s ({n_hands/max(t_wilor,0.001):.0f} hands/s)")
    print(f"  Total hand pose: {elapsed:.1f}s ({len(sampled_indices)/elapsed:.1f} fps), {n_hands} hands detected")

    # Interpolate strided frames
    if stride > 1:
        for arr in [left_kp3d, right_kp3d, left_kp2d, right_kp2d]:
            _interpolate_hand_keypoints(arr, sampled_indices, num_frames)
        for det_arr in [left_detected, right_detected]:
            full = np.zeros(num_frames, dtype=bool)
            full[sampled_indices] = det_arr[sampled_indices]
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


def _run_wilor_batch(crop_imgs, crop_meta, model, model_cfg, device,
                     left_kp3d, right_kp3d, left_kp2d, right_kp2d,
                     left_detected, right_detected):
    """Run WiLoR on a batch of preprocessed hand crops and write results into output arrays."""
    from wilor.utils.renderer import cam_crop_to_full

    bs = len(crop_imgs)
    img_batch = torch.stack(crop_imgs).to(device)

    # Build batch dict matching what WiLoR expects
    box_centers = torch.tensor(np.array([m[1] for m in crop_meta]), dtype=torch.float32, device=device)
    box_sizes = torch.tensor(np.array([m[2] for m in crop_meta]), dtype=torch.float32, device=device)
    img_sizes = torch.tensor(np.array([m[3] for m in crop_meta]), dtype=torch.float32, device=device)
    rights = torch.tensor(np.array([m[4] for m in crop_meta]), dtype=torch.float32, device=device)

    batch = {
        'img': img_batch,
        'box_center': box_centers,
        'box_size': box_sizes,
        'img_size': img_sizes,
        'right': rights,
        'personid': torch.zeros(bs, dtype=torch.int32, device=device),
    }

    with torch.no_grad():
        out = model(batch)

    multiplier = (2 * rights - 1)
    pred_cam = out['pred_cam']
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_sizes.max()
    pred_cam_t_full = cam_crop_to_full(
        pred_cam, box_centers, box_sizes, img_sizes, scaled_focal_length
    ).detach().cpu().numpy()

    joints_3d_all = out['pred_keypoints_3d'].detach().cpu().numpy()  # [B, 21, 3]
    rights_np = rights.cpu().numpy()
    has_kp2d = 'pred_keypoints_2d' in out
    kp2d_all = out['pred_keypoints_2d'].detach().cpu().numpy() if has_kp2d else None
    box_centers_np = box_centers.cpu().numpy()
    box_sizes_np = box_sizes.cpu().numpy()

    for n in range(bs):
        fi = crop_meta[n][0]
        is_r = rights_np[n]
        joints = joints_3d_all[n].copy()
        joints[:, 0] = (2 * is_r - 1) * joints[:, 0]
        cam_t = pred_cam_t_full[n]
        joints_cam = joints + cam_t

        kp_2d = kp2d_all[n][:, :2].copy() if has_kp2d else None

        if kp_2d is not None:
            # Transform from crop-relative to full-frame pixel coordinates
            # Flip x for handedness, then scale by box_size and translate by box_center
            kp_2d[:, 0] *= (2 * is_r - 1)
            kp_2d = kp_2d * box_sizes_np[n] + box_centers_np[n]

        if is_r > 0.5:
            right_kp3d[fi] = joints_cam
            right_detected[fi] = True
            if kp_2d is not None:
                right_kp2d[fi] = kp_2d
        else:
            left_kp3d[fi] = joints_cam
            left_detected[fi] = True
            if kp_2d is not None:
                left_kp2d[fi] = kp_2d


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


def _extract_audio(video_path: str, output_wav: str) -> bool:
    """Extract audio from video as 16 kHz mono WAV. Returns False if no audio stream."""
    import subprocess
    # Check for audio stream
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        return False
    # Extract as 16 kHz mono PCM (NeMo requirement)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_wav],
        capture_output=True, check=True,
    )
    return True


def _audio_subprocess_worker(video_path, fps, num_frames, model_name, result_file):
    """Run full audio pipeline in a subprocess (own GIL, no contention with SLAM).

    Writes result as JSON to result_file. Uses CPU-only inference.
    """
    try:
        import os as _os
        import logging as _logging
        import warnings as _warnings
        # Suppress NeMo/Megatron/OneLogger warnings before import
        _warnings.filterwarnings("ignore")
        for _name in ("nemo", "nemo.collections", "nemo.utils", "nemo_logger"):
            _logging.getLogger(_name).setLevel(_logging.ERROR)
        _logging.getLogger("onelogger").setLevel(_logging.ERROR)
        try:
            _os.nice(15)  # Lower priority to avoid starving SLAM
        except OSError:
            pass
        wav_path = result_file + ".wav"
        has_audio = _extract_audio(video_path, wav_path)
        if not has_audio:
            os.remove(wav_path) if os.path.exists(wav_path) else None
            with open(result_file, "w") as f:
                json.dump(None, f)
            return

        # Suppress NeMo's verbose warnings about training config, CUDA graphs, etc.
        import logging as _logging
        for _name in ("nemo", "nemo.collections", "nemo.utils", "nemo_logger"):
            _logging.getLogger(_name).setLevel(_logging.ERROR)

        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        model.eval()
        model = model.to("cpu")

        hypotheses = model.transcribe([wav_path], timestamps=True)
        os.remove(wav_path)

        hyp = hypotheses[0]
        transcript = hyp.text
        ts = hyp.timestamp
        segments = ts.get("segment", [])
        words = ts.get("word", [])

        frame_text = [""] * num_frames
        for seg in segments:
            start_frame = int(seg["start"] * fps)
            end_frame = min(int(seg["end"] * fps) + 1, num_frames)
            for fi in range(max(0, start_frame), end_frame):
                frame_text[fi] = seg["segment"]

        result = {
            "transcript": transcript,
            "segments": [
                {"text": s["segment"], "start": s["start"], "end": s["end"]}
                for s in segments
            ],
            "words": [
                {"text": w["word"], "start": w["start"], "end": w["end"]}
                for w in words
            ],
            "frame_text": frame_text,
        }
        with open(result_file, "w") as f:
            json.dump(result, f)
    except Exception:
        import traceback
        with open(result_file, "w") as f:
            json.dump({"_error": traceback.format_exc()}, f)


def start_audio_subprocess(video_path, fps, num_frames, model_name):
    """Launch audio transcription in a separate process (own GIL).

    Returns (subprocess.Popen, result_file_path).
    Call collect_audio_subprocess() to get the result.
    """
    import subprocess as sp
    import tempfile
    result_file = tempfile.mktemp(suffix=".json", prefix="audio_result_")
    # Run as a Python subprocess
    cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {os.path.dirname(__file__)!r}); "
        f"from annotate_pipeline import _audio_subprocess_worker; "
        f"_audio_subprocess_worker({video_path!r}, {fps}, {num_frames}, {model_name!r}, {result_file!r})"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Prevent CUDA init in subprocess
    # Limit PyTorch/MKL thread count to avoid starving SLAM for CPU
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    proc = sp.Popen(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, env=env)
    return proc, result_file


def collect_audio_subprocess(proc, result_file, timeout=120):
    """Collect audio result from subprocess. Returns dict or None."""
    proc.wait(timeout=timeout)
    if not os.path.exists(result_file):
        return None
    with open(result_file) as f:
        result = json.load(f)
    os.remove(result_file)
    if result is None:
        return None
    if "_error" in result:
        print(f"  Audio subprocess error:\n{result['_error']}")
        return None
    return result


# ---------------------------------------------------------------------------
# LeRobot v3.0 dataset writer
# ---------------------------------------------------------------------------

def _get_video_info(video_path: str, fps: float) -> dict:
    """Probe video file for codec, resolution, pixel format."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", video_path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)["streams"][0]
    return {
        "video.fps": fps,
        "video.height": int(info["height"]),
        "video.width": int(info["width"]),
        "video.channels": 3,
        "video.codec": info.get("codec_name", "h264"),
        "video.pix_fmt": info.get("pix_fmt", "yuv420p"),
        "video.is_depth_map": False,
        "has_audio": False,
    }


def write_lerobot_dataset(
    output_dir: str,
    video_path: str,
    slam_result: dict,
    hand_result: dict,
    fps: float,
    num_frames: int,
    audio_result: dict = None,
    input_video_path: str = None,
):
    """Assemble annotations into a LeRobot v3.0 dataset."""
    output_dir = Path(output_dir)

    VIDEO_KEY = "observation.video"
    video_dir = output_dir / "videos" / VIDEO_KEY / "chunk-000"
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"

    for d in [video_dir, data_dir, meta_dir, episodes_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy video (if it exists — may be skipped)
    video_dest = video_dir / "file-000.mp4"
    if video_path and os.path.exists(video_path):
        shutil.copy2(video_path, video_dest)
        print(f"  Video: {video_dest}")

    # Probe video for info.json
    if video_dest.exists():
        video_probe_path = str(video_dest)
    elif input_video_path and os.path.exists(input_video_path):
        video_probe_path = input_video_path
    else:
        video_probe_path = video_path
    vinfo = _get_video_info(video_probe_path, fps)

    # 2. Build per-frame data — all required LeRobot v3.0 columns
    frame_indices = np.arange(num_frames, dtype=np.int64)
    timestamps = (frame_indices / fps).astype(np.float32)
    episode_indices = np.zeros(num_frames, dtype=np.int64)

    data = {
        "index": frame_indices.copy(),            # global unique ID
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "timestamp": timestamps,
        "task_index": np.zeros(num_frames, dtype=np.int64),
    }

    # SLAM columns — fixed-size list columns with explicit float32 dtype
    if slam_result is not None:
        poses = slam_result["poses"]  # [N, 7]: tx,ty,tz,qx,qy,qz,qw
        if len(poses) >= num_frames:
            poses = poses[:num_frames]
        else:
            pad = np.tile(poses[-1:], (num_frames - len(poses), 1))
            poses = np.vstack([poses, pad])
        poses = poses.astype(np.float32)

        data["observation.slam.pose"] = pa.FixedSizeListArray.from_arrays(
            pa.array(poses.ravel(), type=pa.float32()), 7)

        intrinsics = slam_result["intrinsics"].astype(np.float32)  # [4]: fx,fy,cx,cy
        intrinsics_tiled = np.tile(intrinsics, (num_frames, 1))
        data["observation.slam.intrinsics"] = pa.FixedSizeListArray.from_arrays(
            pa.array(intrinsics_tiled.ravel(), type=pa.float32()), 4)

    # Hand pose columns — Array2D for keypoints, scalar for detected flag
    if hand_result is not None:
        for side in ["left", "right"]:
            kp3d = hand_result[f"{side}_hand_keypoints_3d"].astype(np.float32)  # [N, 21, 3]
            kp2d = hand_result[f"{side}_hand_keypoints_2d"].astype(np.float32)  # [N, 21, 2]

            # Store as nested list-of-lists for Array2D compatibility
            data[f"observation.hand.{side}.keypoints_3d"] = [
                [kp3d[i, j].tolist() for j in range(21)] for i in range(num_frames)
            ]
            data[f"observation.hand.{side}.keypoints_2d"] = [
                [kp2d[i, j].tolist() for j in range(21)] for i in range(num_frames)
            ]

            data[f"observation.hand.{side}.detected"] = hand_result[
                f"{side}_hand_detected"
            ].astype(np.float32)

    # Audio transcription columns
    if audio_result is not None:
        data["observation.audio.transcript"] = audio_result["frame_text"]

    # 3. Write Parquet
    table = pa.table(data)
    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(table, parquet_path)
    print(f"  Data: {parquet_path} ({num_frames} rows)")

    # 4. Write episode metadata (all columns required by lerobot v0.5)
    episode_data = {
        "episode_index": pa.array([0], type=pa.int64()),
        "length": pa.array([num_frames], type=pa.int64()),
        "task_index": pa.array([0], type=pa.int64()),
        "tasks": [["default"]],
        "dataset_from_index": pa.array([0], type=pa.int64()),
        "dataset_to_index": pa.array([num_frames], type=pa.int64()),
        "data/chunk_index": pa.array([0], type=pa.int32()),
        "data/file_index": pa.array([0], type=pa.int32()),
        "meta/episodes/chunk_index": pa.array([0], type=pa.int32()),
        "meta/episodes/file_index": pa.array([0], type=pa.int32()),
    }
    if video_dest.exists():
        episode_data[f"videos/{VIDEO_KEY}/chunk_index"] = pa.array([0], type=pa.int32())
        episode_data[f"videos/{VIDEO_KEY}/file_index"] = pa.array([0], type=pa.int32())
        episode_data[f"videos/{VIDEO_KEY}/from_timestamp"] = pa.array([0.0], type=pa.float32())
        episode_data[f"videos/{VIDEO_KEY}/to_timestamp"] = pa.array(
            [(num_frames - 1) / fps], type=pa.float32())
    pq.write_table(pa.table(episode_data), episodes_dir / "file-000.parquet")

    # 5. Write tasks
    pq.write_table(pa.table({"task_index": [0], "task": ["default"]}), meta_dir / "tasks.parquet")

    # 6. Write info.json (full LeRobot v3.0 schema)
    features = {
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [vinfo["video.height"], vinfo["video.width"], 3],
            "names": ["height", "width", "channels"],
            "info": vinfo,
        },
    }

    if slam_result is not None:
        features["observation.slam.pose"] = {
            "dtype": "float32",
            "shape": [7],
            "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        }
        features["observation.slam.intrinsics"] = {
            "dtype": "float32",
            "shape": [4],
            "names": ["fx", "fy", "cx", "cy"],
        }

    if hand_result is not None:
        for side in ["left", "right"]:
            features[f"observation.hand.{side}.keypoints_3d"] = {
                "dtype": "float32",
                "shape": [21, 3],
                "names": None,
            }
            features[f"observation.hand.{side}.keypoints_2d"] = {
                "dtype": "float32",
                "shape": [21, 2],
                "names": None,
            }
            features[f"observation.hand.{side}.detected"] = {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            }

    if audio_result is not None:
        features["observation.audio.transcript"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }
        # Save full transcript + word/segment timestamps as separate JSON
        audio_meta = {
            "transcript": audio_result["transcript"],
            "segments": audio_result["segments"],
            "words": audio_result["words"],
        }
        audio_meta_path = meta_dir / "audio.json"
        with open(audio_meta_path, "w") as f:
            json.dump(audio_meta, f, indent=2)
        print(f"  Audio: {audio_meta_path}")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": int(fps) if fps == int(fps) else fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
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
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio transcription")
    parser.add_argument("--asr-model", default="parakeet-tdt_ctc-110m",
                        help="NeMo ASR model name (default: parakeet-tdt_ctc-110m)")
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
    executor = ThreadPoolExecutor(max_workers=3)

    os.makedirs(args.output_dir, exist_ok=True)
    video_output = str(Path(args.output_dir) / "converted.mp4")
    phase1_time = 0
    phase1_start = time.perf_counter()
    video_convert_future = None
    video_convert_proc = None

    # Video conversion deferred until after SLAM+hands complete.
    # NVENC takes only ~2s on a free GPU but 30+s when competing with SLAM/YOLO/WiLoR
    # for GPU resources. Running it sequentially after inference is much faster overall.
    ffmpeg_cmd = None
    if not args.skip_video_convert:
        from convert_video import build_ffmpeg_cmd
        ffmpeg_cmd = build_ffmpeg_cmd(input_path, video_output, max_threads=4)
        if ffmpeg_cmd:
            print(f"[Video conversion] Will run after inference (NVENC, ~2s)")

    # Pre-import heavy modules so the forked child inherits them (zero re-import).
    # These imports must happen BEFORE `from droid import Droid` which initializes
    # the CUDA driver via droid_backends.so, preventing fork after that point.
    #
    # pytorch_lightning is stubbed: WiLoR only needs pl.LightningModule (a base class)
    # and pl.utilities.rank_zero_only (a decorator). The real import costs 3.4s for
    # training machinery we never use during inference.
    if not args.skip_hands:
        _install_pytorch_lightning_stub()
        import timm  # noqa: F401
        sys.path.insert(0, args.wilor_dir)
        from ultralytics import YOLO as _YOLO  # noqa: F401
        import smplx  # noqa: F401
        orig_cwd = os.getcwd()
        os.chdir(args.wilor_dir)
        try:
            from wilor.configs import get_config as _wc  # noqa: F401
            from wilor.models.wilor import WiLoR as _WiLoR  # noqa: F401
        finally:
            os.chdir(orig_cwd)
        print(f"  Modules pre-imported for fork")

    slam_result = None
    hand_result = None
    frames = None
    fps = 0
    num_frames = 0
    phase2_time = 0
    phase3_time = 0
    decode_time = 0
    audio_proc = None
    audio_result_file = None

    # Helper to start background audio + video convert
    def _start_bg():
        nonlocal video_convert_future, audio_proc, audio_result_file
        if not args.skip_video_convert and video_convert_future is None and video_convert_proc is None and ffmpeg_cmd is None:
            from convert_video import convert_video
            print(f"[Background] Starting video conversion...")
            def _nice_convert():
                try:
                    os.nice(10)
                except OSError:
                    pass
                return convert_video(input_path, video_output, fast=True)
            video_convert_future = executor.submit(_nice_convert)
        if not args.skip_audio and audio_proc is None:
            _audio_fps, _, _, _audio_nframes = get_video_metadata(input_path)
            print(f"[Background] Starting audio subprocess...")
            audio_proc, audio_result_file = start_audio_subprocess(
                input_path, _audio_fps, _audio_nframes, args.asr_model,
            )

    # ── Forked path: SLAM + hand pose in separate processes (no GIL contention) ──
    # Background tasks start AFTER fork via post_fork_fn to avoid deadlock.
    if not args.skip_slam and not args.skip_hands:
        print(f"\n[Forked] SLAM + Hand pose (DROID-SLAM + WiLoR)")
        phase2_start = time.perf_counter()

        fps, width, height, nf_est = get_video_metadata(input_path)

        slam_result, hand_result, num_frames = run_slam_and_hand_split(
            input_path, fps, width, height,
            args.droid_weights, args.wilor_dir,
            device=args.device,
            backend_steps=tuple(args.slam_backend_steps),
            fast_traj=args.fast_traj,
            det_conf=args.hand_det_conf,
            stride=args.hand_stride,
            post_fork_fn=_start_bg,
            num_frames_est=nf_est,
        )

        phase2_time = time.perf_counter() - phase2_start
        phase3_time = phase2_time  # combined
        print(f"  Poses shape: {slam_result['poses'].shape}")
        torch.cuda.empty_cache()

    # ── SLAM only (no hands) ──
    elif not args.skip_slam:
        _start_bg()
        print(f"\n[Phase 2] Decode + SLAM (DROID-SLAM)")
        phase2_start = time.perf_counter()
        fps, width, height, _ = get_video_metadata(input_path)

        slam_result, num_frames = run_slam_native_decode(
            input_path, fps, width, height, args.droid_weights, args.device,
            backend_steps=tuple(args.slam_backend_steps),
            fast_traj=args.fast_traj, collect_rgb=False,
        )
        phase2_time = time.perf_counter() - phase2_start
        print(f"  Decode+SLAM completed in {phase2_time:.1f}s ({num_frames/phase2_time:.0f} fps)")
        print(f"  Poses shape: {slam_result['poses'].shape}")
        torch.cuda.empty_cache()

    # ── Hands only (no SLAM) ──
    elif not args.skip_hands:
        _start_bg()
        print(f"\n[Phase 2] SLAM: Skipped")

        print(f"\n[Decode] Reading video frames for hand inference...")
        decode_start = time.perf_counter()
        frames, fps, width, height, num_frames = decode_video_frames(input_path)
        decode_time = time.perf_counter() - decode_start
        print(f"  Decoded {num_frames} frames ({width}x{height} @ {fps:.1f}fps) in {decode_time:.1f}s")

        print(f"\n[Phase 3] Hand pose (WiLoR)")
        phase3_start = time.perf_counter()
        hand_result = run_hand_pose_wilor(
            frames, fps, width, height,
            args.wilor_dir, args.device,
            stride=args.hand_stride,
            det_conf=args.hand_det_conf,
            preloaded=None,
        )
        phase3_time = time.perf_counter() - phase3_start
        effective_fps = num_frames / phase3_time if phase3_time > 0 else 0
        print(f"  Hand pose completed in {phase3_time:.1f}s ({effective_fps:.0f} fps effective)")
        torch.cuda.empty_cache()

    else:
        _start_bg()
        print(f"\n[Phase 2] SLAM: Skipped")
        print(f"\n[Phase 3] Hand pose: Skipped")
        if num_frames == 0:
            fps, _, _, num_frames = get_video_metadata(input_path)

    # Free decoded frames to release RAM
    if frames is not None:
        del frames

    # Collect background audio subprocess result
    audio_result = None
    audio_time = 0
    if not args.skip_audio and audio_proc is not None:
        print(f"\n[Audio] Transcription (Parakeet)")
        audio_start = time.perf_counter()
        if audio_proc.poll() is not None:
            print(f"  Audio subprocess finished (fully overlapped)")
        else:
            print(f"  Waiting for audio subprocess to finish...")
        audio_result = collect_audio_subprocess(audio_proc, audio_result_file)
        audio_time = time.perf_counter() - audio_start
        if audio_result is not None:
            # Update num_frames in frame_text if needed (metadata estimate vs actual)
            ft = audio_result["frame_text"]
            if num_frames > 0 and len(ft) != num_frames:
                if len(ft) < num_frames:
                    ft.extend([""] * (num_frames - len(ft)))
                else:
                    audio_result["frame_text"] = ft[:num_frames]
            print(f"  Audio transcription completed (waited {audio_time:.1f}s)")
        else:
            print(f"  Audio: no audio stream or subprocess failed")
            audio_time = 0

    # Run video conversion now that GPU is free (NVENC ~2s on idle GPU)
    if video_convert_proc is not None:
        if video_convert_proc.poll() is not None:
            print(f"\n  Video conversion finished (overlapped)")
        else:
            print(f"\n  Waiting for video conversion to finish...")
        video_convert_proc.wait()
        if video_convert_proc.returncode != 0:
            print(f"  WARNING: ffmpeg failed (code {video_convert_proc.returncode})")
        phase1_time = time.perf_counter() - phase1_start
    elif ffmpeg_cmd is not None:
        import subprocess as _sp
        print(f"\n[Video conversion] Running NVENC...")
        phase1_start = time.perf_counter()
        _vc = _sp.run(ffmpeg_cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        phase1_time = time.perf_counter() - phase1_start
        print(f"  Video conversion: {phase1_time:.1f}s")
        if _vc.returncode != 0:
            print(f"  WARNING: ffmpeg failed (code {_vc.returncode})")
    elif video_convert_future is not None:
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
        audio_result=audio_result, input_video_path=input_path,
    )

    # Clean up temp converted video (now copied into videos/ dir)
    if os.path.exists(video_output) and "converted.mp4" in video_output:
        os.remove(video_output)

    phase4_time = time.perf_counter() - phase4_start
    total_time = time.perf_counter() - total_start
    executor.shutdown(wait=False)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete!")
    print(f"  Total time: {total_time:.1f}s")
    if not args.skip_slam and not args.skip_hands:
        print(f"    SLAM+Hands:    {phase2_time:.1f}s (concurrent on same GPU)")
    else:
        if phase2_time > 0:
            print(f"    SLAM:          {phase2_time:.1f}s")
        if decode_time > 0:
            print(f"    Hand decode:   {decode_time:.1f}s")
        if phase3_time > 0:
            print(f"    Hand pose:     {phase3_time:.1f}s")
    if audio_time > 0:
        print(f"    Audio:         {audio_time:.1f}s")
    print(f"    Video convert: {phase1_time:.1f}s")
    print(f"    Assembly:      {phase4_time:.1f}s")
    print(f"  Output: {dataset_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
