#!/usr/bin/env python3
"""Compare CUDA SLAM vs PyTorch DROID-SLAM on identical decoded frames.

Decodes frames using the native C++ decoder, then feeds the SAME frames
to both CUDA (via stdin) and PyTorch DROID-SLAM. This isolates algorithm
differences from decode differences.
"""

import os, sys, time, struct, subprocess, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_pipeline import (
    get_video_metadata, get_iphone_intrinsics, _interpolate_poses_simple, _get_native_decode,
)

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"
CUDA_SLAM_BIN = "./cuda_slam/cuda_droid"
CUDA_SLAM_WEIGHTS = "cuda_slam/data/weights"
BACKEND_STEPS = (7, 12)


def decode_frames(video_path, h1, w1):
    """Decode all SLAM frames using native C++ decoder."""
    nd = _get_native_decode()
    decoder = nd.AsyncVideoDecoder()
    decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)
    frames = []
    while True:
        result = decoder.get_next()
        if result is None:
            break
        _, slam_bgr = result
        frames.append(np.array(slam_bgr))  # copy from pybind buffer
    decoder.stop()
    return frames


def run_pytorch_slam(frames, h1, w1, intrinsics, backend_steps):
    """Run PyTorch DROID-SLAM on pre-decoded frames."""
    sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
    from droid import Droid

    intrinsics_t = torch.as_tensor(intrinsics)
    slam_args = argparse.Namespace(
        weights=DROID_WEIGHTS, buffer=512, image_size=[h1, w1],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device="cuda", backend_device="cuda",
    )

    droid = Droid(slam_args)
    t0 = time.perf_counter()
    for t, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        droid.track(t, tensor, intrinsics=intrinsics_t)
    t_track = time.perf_counter() - t0
    n = len(frames)
    print(f"  Tracking: {t_track:.1f}s ({n/t_track:.0f} fps)")

    del droid.frontend
    torch.cuda.empty_cache()
    droid.backend(backend_steps[0])
    torch.cuda.empty_cache()
    droid.backend(backend_steps[1])

    N = droid.video.counter.value
    kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
    import lietorch
    kf_poses_c2w = lietorch.SE3(droid.video.poses[:N]).inv().data.cpu().numpy()
    all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_c2w, n)

    del droid
    torch.cuda.empty_cache()
    return {"poses": all_poses, "kf_timestamps": kf_tstamps, "n_keyframes": N, "num_frames": n}


def run_cuda_slam(frames, h1, w1, intrinsics, backend_steps):
    """Run CUDA DROID-SLAM on pre-decoded frames via stdin pipe."""
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="verify_slam_")
    calib_path = os.path.join(tmpdir, "calib.bin")
    pose_path = os.path.join(tmpdir, "poses.bin")

    with open(calib_path, "wb") as f:
        f.write(struct.pack("4f", *intrinsics))

    cmd = [
        CUDA_SLAM_BIN,
        "--weights", CUDA_SLAM_WEIGHTS,
        "--calib", calib_path,
        "--stdin", str(h1), str(w1),
        "--max-frames", str(len(frames)),
        "--cam-to-world",
        "--pose-output", pose_path,
        "--frontend-window", "25",
        "--update-steps", "3",
        "--backend-radius", "2",
        "--backend", str(backend_steps[0]), str(backend_steps[1]),
    ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    # Feed frames as raw float32 BGR HWC
    for frame in frames:
        frame_f32 = frame.astype(np.float32)
        proc.stdin.write(frame_f32.tobytes())
    proc.stdin.close()

    stdout = proc.stdout.read().decode()
    stderr = proc.stderr.read().decode()
    proc.wait()
    t_total = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  CUDA SLAM failed: {stderr[-500:]}")
        sys.exit(1)
    print(f"  Total: {t_total:.1f}s")

    # Parse num_frames from stdout
    n = len(frames)

    with open(pose_path, "rb") as f:
        nk = struct.unpack("i", f.read(4))[0]
        kf_tstamps = np.array(struct.unpack(f"{nk}i", f.read(nk * 4)), dtype=np.int64)
        kf_poses = np.frombuffer(f.read(nk * 7 * 4), dtype=np.float32).reshape(nk, 7)

    all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses, n)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {"poses": all_poses, "kf_timestamps": kf_tstamps, "n_keyframes": nk, "num_frames": n}


def quat_angle_diff(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0, 1)
    return 2 * np.degrees(np.arccos(dot))


def compare(pt, cuda):
    n = min(pt["num_frames"], cuda["num_frames"])
    pt_poses = pt["poses"][:n]
    cuda_poses = cuda["poses"][:n]
    pt_t, cuda_t = pt_poses[:, :3], cuda_poses[:, :3]
    pt_q, cuda_q = pt_poses[:, 3:], cuda_poses[:, 3:]

    print(f"\n{'='*60}")
    print(f"SLAM Comparison ({n} frames, identical input)")
    print(f"  PyTorch: {pt['n_keyframes']} keyframes")
    print(f"  CUDA:    {cuda['n_keyframes']} keyframes")
    print(f"{'='*60}")

    # Keyframe overlap
    pt_kf = set(pt["kf_timestamps"].tolist())
    cuda_kf = set(cuda["kf_timestamps"].tolist())
    overlap = pt_kf & cuda_kf
    print(f"  Keyframe overlap: {len(overlap)} shared out of PT={len(pt_kf)}, CUDA={len(cuda_kf)}")

    # Translation
    t_diff = np.linalg.norm(pt_t - cuda_t, axis=1)
    print(f"\nTranslation difference:")
    print(f"  Mean:   {t_diff.mean():.4f} m")
    print(f"  Median: {np.median(t_diff):.4f} m")
    print(f"  Max:    {t_diff.max():.4f} m (frame {np.argmax(t_diff)})")
    print(f"  p95:    {np.percentile(t_diff, 95):.4f} m")

    # Rotation
    q_diff = quat_angle_diff(pt_q, cuda_q)
    print(f"\nRotation difference:")
    print(f"  Mean:   {q_diff.mean():.2f}°")
    print(f"  Median: {np.median(q_diff):.2f}°")
    print(f"  Max:    {q_diff.max():.2f}° (frame {np.argmax(q_diff)})")
    print(f"  p95:    {np.percentile(q_diff, 95):.2f}°")

    # Trajectory stats
    pt_path = np.sum(np.linalg.norm(np.diff(pt_t, axis=0), axis=1))
    cuda_path = np.sum(np.linalg.norm(np.diff(cuda_t, axis=0), axis=1))
    print(f"\nTrajectory:")
    print(f"  Path length:  PT={pt_path:.4f}  CUDA={cuda_path:.4f}  ratio={cuda_path/pt_path:.3f}")
    print(f"  Max disp:     PT={np.linalg.norm(pt_t, axis=1).max():.4f}  "
          f"CUDA={np.linalg.norm(cuda_t, axis=1).max():.4f}")

    # RPE
    step = max(1, n // 100)
    rpe = [np.linalg.norm((pt_t[i+step]-pt_t[i]) - (cuda_t[i+step]-cuda_t[i]))
           for i in range(0, n-step, step)]
    rpe = np.array(rpe)
    print(f"\nRelative pose error (step={step}):")
    print(f"  Mean:   {rpe.mean():.4f} m")
    print(f"  Median: {np.median(rpe):.4f} m")
    print(f"  Max:    {rpe.max():.4f} m")

    # First poses
    print(f"\nFirst 5 poses:")
    for i in range(min(5, n)):
        d = np.linalg.norm(pt_poses[i,:3] - cuda_poses[i,:3])
        print(f"  F{i}: PT=[{pt_poses[i,0]:7.4f} {pt_poses[i,1]:7.4f} {pt_poses[i,2]:7.4f}]  "
              f"CU=[{cuda_poses[i,0]:7.4f} {cuda_poses[i,1]:7.4f} {cuda_poses[i,2]:7.4f}]  d={d:.4f}")


def main():
    fps, width, height, _ = get_video_metadata(VIDEO)
    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8
    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics = np.array([fx*w1/w0, fy*h1/h0, cx*w1/w0, cy*h1/h0], dtype=np.float32)
    print(f"Video: {width}x{height} @ {fps:.0f}fps, SLAM: {w1}x{h1}")
    print(f"Backend: {BACKEND_STEPS}")

    print("\nDecoding frames...")
    t0 = time.perf_counter()
    frames = decode_frames(VIDEO, h1, w1)
    print(f"  {len(frames)} frames in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== PyTorch DROID-SLAM ===")
    pt = run_pytorch_slam(frames, h1, w1, intrinsics, BACKEND_STEPS)
    print(f"  {pt['n_keyframes']} keyframes")

    print(f"\n=== CUDA DROID-SLAM (stdin, same frames) ===")
    cuda = run_cuda_slam(frames, h1, w1, intrinsics, BACKEND_STEPS)
    print(f"  {cuda['n_keyframes']} keyframes")

    compare(pt, cuda)


if __name__ == "__main__":
    main()
