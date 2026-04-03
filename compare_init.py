#!/usr/bin/env python3
"""Compare CUDA vs PyTorch initialization state.

Runs both implementations through just the initialization phase and compares
keyframe timestamps, poses, and disparities to isolate where they diverge.
"""

import os, sys, time, struct, subprocess, argparse, tempfile, shutil, threading
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_pipeline import get_video_metadata, get_iphone_intrinsics, _get_native_decode

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"
CUDA_SLAM_BIN = "./cuda_slam/cuda_droid"
CUDA_SLAM_WEIGHTS = "cuda_slam/data/weights"


def decode_frames(video_path, h1, w1):
    nd = _get_native_decode()
    decoder = nd.AsyncVideoDecoder()
    decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)
    frames = []
    while True:
        result = decoder.get_next()
        if result is None:
            break
        _, slam_bgr = result
        frames.append(np.array(slam_bgr))
    decoder.stop()
    return frames


def run_pytorch_init(frames, h1, w1, intrinsics):
    """Run PyTorch DROID-SLAM through initialization only, dump state."""
    sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
    from droid import Droid

    intrinsics_t = torch.as_tensor(intrinsics)
    slam_args = argparse.Namespace(
        weights=DROID_WEIGHTS, buffer=512, image_size=[h1, w1],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.5,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device="cuda", backend_device="cuda",
    )

    droid = Droid(slam_args)

    # Feed frames until initialization completes
    for t, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        droid.track(t, tensor, intrinsics=intrinsics_t)

        # Check if initialization just completed
        if droid.frontend.is_initialized:
            break

    N = droid.video.counter.value
    tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
    poses = droid.video.poses[:N].cpu().numpy()  # [N, 7]
    disps = droid.video.disps[:N].cpu().numpy()   # [N, H, W]

    # Also get the next frame's pre-initialized state
    next_pose = droid.video.poses[N].cpu().numpy()
    next_disp = droid.video.disps[N].cpu().numpy()

    print(f"PyTorch init: {N} keyframes, tstamps={tstamps.tolist()}")

    del droid
    torch.cuda.empty_cache()

    return {
        "tstamps": tstamps, "poses": poses, "disps": disps,
        "next_pose": next_pose, "next_disp": next_disp,
        "num_keyframes": N,
    }


def run_cuda_init(frames, h1, w1, intrinsics):
    """Run CUDA DROID-SLAM, stop after init, dump state via pose output."""
    tmpdir = tempfile.mkdtemp(prefix="compare_init_")
    calib_path = os.path.join(tmpdir, "calib.bin")
    pose_path = os.path.join(tmpdir, "poses.bin")

    with open(calib_path, "wb") as f:
        f.write(struct.pack("4f", *intrinsics))

    # Only feed enough frames to trigger initialization + a couple more
    # We'll feed all frames but limit max-frames
    cmd = [
        CUDA_SLAM_BIN,
        "--weights", CUDA_SLAM_WEIGHTS,
        "--calib", calib_path,
        "--stdin", str(h1), str(w1),
        "--max-frames", str(len(frames)),
        "--pose-output", pose_path,
        "--frontend-window", "25",
        "--update-steps", "3",
        "--backend-radius", "2",
        "--backend", "0", "0",  # Skip backend
    ]

    stderr_path = os.path.join(tmpdir, "stderr.log")
    with open(stderr_path, "w") as stderr_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr_f)

        for frame in frames:
            frame_f32 = frame.astype(np.float32)
            proc.stdin.write(frame_f32.tobytes())
        proc.stdin.close()
        proc.wait()

    with open(stderr_path, "r") as f:
        stderr = f.read()

    if proc.returncode != 0:
        print(f"CUDA SLAM failed: {stderr[-500:]}")
        sys.exit(1)

    # Print debug
    for line in stderr.strip().split('\n'):
        if any(k in line for k in ['Init KF', 'mean_disp', 'Initialization', 'BA debug', 'BA dx', 'BA dz', 'CUDA BA#1', 'ii=', 'jj=', 'edge ', 'target[', 'weight[', '  KF']):
            print(f"  {line}")

    # Parse output
    with open(pose_path, "rb") as f:
        nk = struct.unpack("i", f.read(4))[0]
        kf_tstamps = np.array(struct.unpack(f"{nk}i", f.read(nk * 4)), dtype=np.int64)
        kf_poses = np.frombuffer(f.read(nk * 7 * 4), dtype=np.float32).reshape(nk, 7)

    print(f"CUDA init: {nk} keyframes, tstamps={kf_tstamps.tolist()}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"tstamps": kf_tstamps, "poses": kf_poses, "num_keyframes": nk}


def main():
    fps, width, height, _ = get_video_metadata(VIDEO)
    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8
    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics = np.array([fx*w1/w0, fy*h1/h0, cx*w1/w0, cy*h1/h0], dtype=np.float32)
    print(f"Video: {width}x{height}, SLAM: {w1}x{h1}")

    print("\nDecoding frames...")
    frames = decode_frames(VIDEO, h1, w1)
    print(f"  {len(frames)} frames")

    print("\n=== PyTorch ===")
    pt = run_pytorch_init(frames, h1, w1, intrinsics)

    print("\n=== CUDA ===")
    cu = run_cuda_init(frames, h1, w1, intrinsics)

    # Compare timestamps
    print(f"\n{'='*60}")
    print("INITIALIZATION COMPARISON")
    print(f"{'='*60}")

    N = min(pt["num_keyframes"], cu["num_keyframes"])
    print(f"  PT: {pt['num_keyframes']} KF, tstamps={pt['tstamps'][:N].tolist()}")
    print(f"  CU: {cu['num_keyframes']} KF, tstamps={cu['tstamps'][:N].tolist()}")

    # Compare poses at shared keyframes
    pt_map = {int(t): i for i, t in enumerate(pt["tstamps"])}
    cu_map = {int(t): i for i, t in enumerate(cu["tstamps"])}
    shared = sorted(set(pt_map.keys()) & set(cu_map.keys()))

    if shared:
        print(f"\n  Shared keyframes: {len(shared)} at frames {shared}")
        for f in shared[:8]:
            pi, ci = pt_map[f], cu_map[f]
            pt_t = pt["poses"][pi, :3]
            cu_t = cu["poses"][ci, :3]
            pt_q = pt["poses"][pi, 3:]
            cu_q = cu["poses"][ci, 3:]
            d = np.linalg.norm(pt_t - cu_t)
            print(f"  KF frame {f}: PT=[{pt_t[0]:8.5f} {pt_t[1]:8.5f} {pt_t[2]:8.5f}]  "
                  f"CU=[{cu_t[0]:8.5f} {cu_t[1]:8.5f} {cu_t[2]:8.5f}]  d={d:.5f}")

    # Compare disparities
    if "disps" in pt:
        print(f"\n  Disparity statistics (PyTorch):")
        for i in range(min(8, pt["num_keyframes"])):
            d = pt["disps"][i]
            print(f"    KF {i} (frame {pt['tstamps'][i]}): mean={d.mean():.4f} std={d.std():.4f} "
                  f"min={d.min():.4f} max={d.max():.4f}")


if __name__ == "__main__":
    main()
