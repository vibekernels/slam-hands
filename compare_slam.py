#!/usr/bin/env python3
"""Compare PyTorch DROID-SLAM vs CUDA DROID-SLAM outputs on same frames."""

import os, sys, time, struct, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from annotate_pipeline import get_video_metadata, get_iphone_intrinsics

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"
MAX_FRAMES = 200  # match what CUDA version has exported

fps, width, height, nf_est = get_video_metadata(VIDEO)
print(f"Video: {width}x{height} @ {fps}fps, ~{nf_est} frames")

# Compute SLAM resolution (same as pipeline)
h0, w0 = height, width
scale = np.sqrt((384 * 512) / (h0 * w0))
h1 = int(h0 * scale) // 8 * 8
w1 = int(w0 * scale) // 8 * 8
fx, fy, cx, cy = get_iphone_intrinsics(width, height)
fx_s = fx * (w1 / w0)
fy_s = fy * (h1 / h0)
cx_s = cx * (w1 / w0)
cy_s = cy * (h1 / h0)
print(f"SLAM resolution: {w1}x{h1}")
print(f"Intrinsics: fx={fx_s:.2f} fy={fy_s:.2f} cx={cx_s:.2f} cy={cy_s:.2f}")

# ============ PyTorch DROID-SLAM ============
print("\n" + "=" * 60)
print("PyTorch DROID-SLAM")
print("=" * 60)

import torch
sys.path.insert(0, '/workspace/DROID-SLAM')
sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
from annotate_pipeline import _get_native_decode
import argparse

nd = _get_native_decode()
slam_args = argparse.Namespace(
    weights=DROID_WEIGHTS, buffer=512, image_size=[h1, w1],
    disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
    warmup=8, keyframe_thresh=4.0,
    frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
    backend_thresh=22.0, backend_radius=2, backend_nms=3,
    upsample=False, frontend_device="cuda", backend_device="cuda",
)

from droid import Droid

decoder = nd.AsyncVideoDecoder()
decoder.start(VIDEO, w1, h1, slam_only=True, queue_depth=128)
droid = Droid(slam_args)

t0 = time.perf_counter()
t = 0
while t < MAX_FRAMES:
    result = decoder.get_next()
    if result is None:
        break
    _, slam_bgr = result
    tensor = torch.from_numpy(slam_bgr).permute(2, 0, 1).unsqueeze(0).cuda()
    droid.track(t, tensor, intrinsics=torch.as_tensor([fx_s, fy_s, cx_s, cy_s]))
    t += 1
decoder.stop()
t_pt = time.perf_counter() - t0
print(f"  Tracking: {t_pt:.1f}s ({t/t_pt:.0f} fps), {t} frames")

# Backend
del droid.frontend
torch.cuda.empty_cache()
droid.backend(7)
torch.cuda.empty_cache()
droid.backend(12)

# Extract poses
N = droid.video.counter.value
kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
import lietorch
kf_poses_se3 = lietorch.SE3(droid.video.poses[:N])
# .inv() converts from world-to-camera to camera-to-world
kf_poses_raw = kf_poses_se3.inv().data.cpu().numpy()

from annotate_pipeline import _interpolate_poses_simple
pt_all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_raw, t)

# Also get the raw internal poses (world-to-camera) for direct comparison
pt_internal_poses = droid.video.poses[:N].data.cpu().numpy()

print(f"  {N} keyframes at frames: {kf_tstamps.tolist()}")
print(f"  Output poses shape: {pt_all_poses.shape}")
print(f"  First 5 poses (camera-to-world):")
for i in range(min(5, len(pt_all_poses))):
    p = pt_all_poses[i]
    print(f"    Frame {i}: t=[{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}] q=[{p[3]:.4f} {p[4]:.4f} {p[5]:.4f} {p[6]:.4f}]")

del droid
torch.cuda.empty_cache()

# ============ CUDA DROID-SLAM ============
print("\n" + "=" * 60)
print("CUDA DROID-SLAM")
print("=" * 60)

# Update calibration to match pipeline intrinsics
calib_path = "/workspace/robot-video/cuda_slam/data/calib.bin"
with open(calib_path, "wb") as f:
    f.write(struct.pack("4f", fx_s, fy_s, cx_s, cy_s))

pose_file = "/tmp/cuda_poses.bin"
t0 = time.perf_counter()
result = subprocess.run(
    ["./cuda_slam/cuda_droid",
     "--weights", "cuda_slam/data/weights",
     "--frames", "cuda_slam/data/frames",
     "--calib", calib_path,
     "--max-frames", str(MAX_FRAMES),
     "--backend", "7", "12",
     "--cam-to-world",
     "--pose-output", pose_file],
    capture_output=True, text=True, cwd="/workspace/robot-video"
)
t_cuda = time.perf_counter() - t0
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# Load CUDA poses (new format: nk, timestamps[nk], poses[nk*7])
with open(pose_file, "rb") as f:
    nk_cuda = struct.unpack("i", f.read(4))[0]
    cuda_kf_timestamps = np.array(struct.unpack(f"{nk_cuda}i", f.read(nk_cuda * 4)))
    cuda_kf_poses = np.frombuffer(f.read(nk_cuda * 7 * 4), dtype=np.float32).reshape(nk_cuda, 7)
print(f"  {nk_cuda} keyframes at frames: {cuda_kf_timestamps.tolist()}")
print(f"  First 5 keyframe poses (camera-to-world):")
for i in range(min(5, nk_cuda)):
    p = cuda_kf_poses[i]
    print(f"    KF {i} (frame {cuda_kf_timestamps[i]}): t=[{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}] q=[{p[3]:.4f} {p[4]:.4f} {p[5]:.4f} {p[6]:.4f}]")

# Interpolate CUDA keyframe poses to all frames
cuda_all_poses = _interpolate_poses_simple(cuda_kf_timestamps, cuda_kf_poses, MAX_FRAMES)

# ============ Compare ============
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# PyTorch keyframe poses are camera-to-world (after .inv())
# CUDA poses are internal format - need to understand which convention
# PyTorch internal: droid.video.poses stores world-to-camera in lietorch SE3 format [tx,ty,tz,qx,qy,qz,qw]
# CUDA internal: poses are [tx,ty,tz,qx,qy,qz,qw] - also world-to-camera (same convention)

# For trajectory comparison, normalize both to same scale
# PyTorch: use interpolated camera-to-world poses
# CUDA: we have one pose per frame (all frames are "keyframes"), need to invert

# Check if CUDA poses look reasonable
cuda_traj = cuda_all_poses[:, :3]
pt_traj = pt_all_poses[:, :3]
n = min(len(cuda_traj), len(pt_traj))

print(f"\nTrajectory comparison ({n} frames):")
print(f"  PyTorch range: x=[{pt_traj[:n,0].min():.3f},{pt_traj[:n,0].max():.3f}] "
      f"y=[{pt_traj[:n,1].min():.3f},{pt_traj[:n,1].max():.3f}] "
      f"z=[{pt_traj[:n,2].min():.3f},{pt_traj[:n,2].max():.3f}]")
print(f"  CUDA range:    x=[{cuda_traj[:n,0].min():.3f},{cuda_traj[:n,0].max():.3f}] "
      f"y=[{cuda_traj[:n,1].min():.3f},{cuda_traj[:n,1].max():.3f}] "
      f"z=[{cuda_traj[:n,2].min():.3f},{cuda_traj[:n,2].max():.3f}]")

# Compute per-frame displacement from origin
pt_disp = np.linalg.norm(pt_traj[:n], axis=1)
cuda_disp = np.linalg.norm(cuda_traj[:n], axis=1)
print(f"\n  PyTorch max displacement: {pt_disp.max():.3f}")
print(f"  CUDA max displacement:    {cuda_disp.max():.3f}")
print(f"  PyTorch total path len:   {np.sum(np.linalg.norm(np.diff(pt_traj[:n], axis=0), axis=1)):.3f}")
print(f"  CUDA total path len:      {np.sum(np.linalg.norm(np.diff(cuda_traj[:n], axis=0), axis=1)):.3f}")

# Note: CUDA and PyTorch will differ because:
# 1. CUDA has no motion filter (processes every frame as keyframe)
# 2. CUDA has simplified edge management (no keyframe selection)
# 3. CUDA has no backend optimization
# 4. Different initialization and convergence behavior
print("\nNote: Differences expected - CUDA version uses simplified tracking")
print("(no motion filter, no keyframe selection, no backend optimization)")
