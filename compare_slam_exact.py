#!/usr/bin/env python3
"""Compare CUDA vs PyTorch DROID-SLAM using identical frame data."""

import os, sys, struct, argparse, time
import numpy as np
import torch
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"
FRAME_DIR = "/workspace/robot-video/cuda_slam/data/frames"
CALIB_FILE = "/workspace/robot-video/cuda_slam/data/calib.bin"

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast

def load_frames(frame_dir, max_frames=200):
    """Load pre-exported frames (same data CUDA binary uses)."""
    frames = []
    for i in range(max_frames):
        path = os.path.join(frame_dir, f"frame_{i:05d}.bin")
        if not os.path.exists(path):
            break
        with open(path, 'rb') as f:
            h, w = struct.unpack('2i', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.float32).reshape(h, w, 3)
            frames.append(data.copy())
    return frames

def main():
    # Load calibration
    with open(CALIB_FILE, 'rb') as f:
        fx, fy, cx, cy = struct.unpack('4f', f.read(16))
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    frames = load_frames(FRAME_DIR)
    h, w = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames at {w}x{h}")
    print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ============ Run PyTorch DROID-SLAM ============
    sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
    from droid import Droid

    intrinsics_t = torch.as_tensor(intrinsics)
    slam_args = argparse.Namespace(
        weights=DROID_WEIGHTS, buffer=512, image_size=[h, w],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.5,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device="cuda", backend_device="cuda",
    )

    droid = Droid(slam_args)

    t0 = time.perf_counter()
    for t, frame in enumerate(frames):
        # frame is [H, W, 3] BGR float32 (uint8-range values)
        # motion_filter.track expects [1, 3, H, W] tensor
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
        droid.track(t, tensor, intrinsics=intrinsics_t)
    t_pt = time.perf_counter() - t0

    N_pt = droid.video.counter.value
    pt_tstamps = droid.video.tstamp[:N_pt].cpu().numpy().astype(int)
    pt_poses = droid.video.poses[:N_pt].cpu().numpy()
    pt_disps = droid.video.disps[:N_pt].cpu().numpy()

    print(f"\nPyTorch: {N_pt} keyframes in {t_pt:.2f}s")
    print(f"  Timestamps: {pt_tstamps.tolist()}")
    for k in range(N_pt):
        p = pt_poses[k]
        d = pt_disps[k].mean()
        print(f"  KF{k} (f{pt_tstamps[k]:3d}): t=[{p[0]:8.5f} {p[1]:8.5f} {p[2]:8.5f}] disp={d:.4f}")

    del droid
    torch.cuda.empty_cache()

    # ============ Run CUDA DROID-SLAM ============
    import subprocess
    pose_file = "/tmp/cuda_exact_poses.bin"
    t0 = time.perf_counter()
    result = subprocess.run(
        ["./cuda_slam/cuda_droid",
         "--weights", "cuda_slam/data/weights",
         "--frames", FRAME_DIR,
         "--calib", CALIB_FILE,
         "--max-frames", str(len(frames)),
         "--cam-to-world",
         "--pose-output", pose_file],
        capture_output=True, text=True, cwd="/workspace/robot-video"
    )
    t_cu = time.perf_counter() - t0

    print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)

    # Parse CUDA stderr for STEP logs
    for line in result.stderr.split('\n'):
        if 'STEP[' in line and ('STEP[0]' in line or 'STEP[7]' in line or 'STEP[15]' in line):
            print(f"  CU {line.strip()}")

    # Load CUDA poses
    with open(pose_file, "rb") as f:
        nk_cuda = struct.unpack("i", f.read(4))[0]
        cuda_kf_timestamps = np.array(struct.unpack(f"{nk_cuda}i", f.read(nk_cuda * 4)))
        cuda_kf_poses = np.frombuffer(f.read(nk_cuda * 7 * 4), dtype=np.float32).reshape(nk_cuda, 7)

    print(f"\nCUDA: {nk_cuda} keyframes in {t_cu:.2f}s")
    print(f"  Timestamps: {cuda_kf_timestamps.tolist()}")
    for k in range(min(nk_cuda, 10)):
        p = cuda_kf_poses[k]
        print(f"  KF{k} (f{cuda_kf_timestamps[k]:3d}): t=[{p[0]:8.5f} {p[1]:8.5f} {p[2]:8.5f}]")

    # ============ Compare ============
    print("\n=== Comparison ===")

    # Find common keyframes
    common_ts = set(pt_tstamps) & set(cuda_kf_timestamps)
    print(f"Common keyframe timestamps: {sorted(common_ts)}")

    pt_idx = {int(t): i for i, t in enumerate(pt_tstamps)}
    cu_idx = {int(t): i for i, t in enumerate(cuda_kf_timestamps)}

    for ts in sorted(common_ts):
        pi, ci = pt_idx[ts], cu_idx[ts]
        pp, cp = pt_poses[pi], cuda_kf_poses[ci]
        t_diff = np.linalg.norm(pp[:3] - cp[:3])
        print(f"  Frame {ts}: t_diff={t_diff:.6f}  PT=[{pp[0]:.5f} {pp[1]:.5f} {pp[2]:.5f}]  CU=[{cp[0]:.5f} {cp[1]:.5f} {cp[2]:.5f}]")

if __name__ == "__main__":
    main()
