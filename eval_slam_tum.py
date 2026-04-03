#!/usr/bin/env python3
"""Evaluate DROID-SLAM (PyTorch + CUDA) on TUM RGB-D benchmark sequences.

Runs both implementations with correct TUM intrinsics and computes
Absolute Trajectory Error (ATE) against ground truth.

Usage:
    python3 eval_slam_tum.py [--sequences fr1_desk fr1_room] [--cuda]
"""

import argparse
import os
import struct
import sys
import time

import cv2
import numpy as np

# ── TUM dataset constants ──────────────────────────────────────────────
# Freiburg1 calibration (from TUM website)
TUM_FR1_INTRINSICS = np.array([517.3, 516.5, 318.6, 255.3], dtype=np.float32)

SEQUENCES = {
    "fr1_desk": {
        "path": "benchmarks/tum/rgbd_dataset_freiburg1_desk",
        "intrinsics": TUM_FR1_INTRINSICS,
    },
    "fr1_room": {
        "path": "benchmarks/tum/rgbd_dataset_freiburg1_room",
        "intrinsics": TUM_FR1_INTRINSICS,
    },
}


# ── Trajectory alignment (Umeyama) ────────────────────────────────────
def align_umeyama(model, data):
    """Align two trajectories using Umeyama method (similarity transform).

    Args:
        model: Nx3 ground truth positions
        data:  Nx3 estimated positions
    Returns:
        aligned: Nx3 aligned estimated positions
        s, R, t: scale, rotation, translation of the alignment
    """
    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)

    model_c = model - mu_m
    data_c = data - mu_d

    sigma2 = (data_c ** 2).sum() / len(data)
    cov = model_c.T @ data_c / len(data)

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma2
    t = mu_m - s * R @ mu_d

    aligned = s * (R @ data.T).T + t
    return aligned, s, R, t


def compute_ate(gt_positions, est_positions):
    """Compute ATE statistics after Umeyama alignment."""
    aligned, s, R, t = align_umeyama(gt_positions, est_positions)
    errors = np.linalg.norm(gt_positions - aligned, axis=1)
    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "std": float(errors.std()),
        "scale": float(s),
        "n_frames": len(errors),
    }


# ── TUM file parsing ──────────────────────────────────────────────────
def parse_tum_file(path):
    """Parse TUM-format file (timestamp + numeric data columns)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Only parse numeric columns (skip filenames etc.)
            try:
                entries.append([float(x) for x in parts])
            except ValueError:
                # For rgb.txt: just grab the timestamp
                entries.append([float(parts[0])])
    return np.array(entries)


def associate_timestamps(ts_a, ts_b, max_diff=0.02):
    """Associate timestamps from two lists. Returns list of (idx_a, idx_b) pairs."""
    matches = []
    j_start = 0
    for i, ta in enumerate(ts_a):
        best_j, best_diff = -1, max_diff
        for j in range(j_start, len(ts_b)):
            diff = abs(ta - ts_b[j])
            if diff < best_diff:
                best_diff = diff
                best_j = j
            if ts_b[j] > ta + max_diff:
                break
        if best_j >= 0:
            matches.append((i, best_j))
            j_start = best_j + 1
    return matches


def load_tum_sequence(seq_dir):
    """Load TUM sequence: images, timestamps, ground truth."""
    rgb_data = parse_tum_file(os.path.join(seq_dir, "rgb.txt"))
    rgb_timestamps = rgb_data[:, 0]
    rgb_files = []
    with open(os.path.join(seq_dir, "rgb.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rgb_files.append(parts[1])

    gt_data = parse_tum_file(os.path.join(seq_dir, "groundtruth.txt"))
    gt_timestamps = gt_data[:, 0]
    gt_poses = gt_data[:, 1:]  # tx ty tz qx qy qz qw

    return rgb_timestamps, rgb_files, gt_timestamps, gt_poses


# ── Run PyTorch DROID-SLAM ─────────────────────────────────────────────
def run_pytorch_slam(seq_dir, intrinsics, max_frames=None):
    """Run PyTorch DROID-SLAM on a TUM sequence."""
    sys.path.insert(0, "/workspace/DROID-SLAM/droid_slam")
    import torch
    from droid import Droid

    rgb_timestamps, rgb_files, gt_timestamps, gt_poses = load_tum_sequence(seq_dir)

    slam_args = argparse.Namespace(
        weights="/workspace/DROID-SLAM/checkpoints/droid.pth",
        buffer=512, image_size=[480, 640],
        disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.5,
        warmup=8, keyframe_thresh=4.0,
        frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3,
        upsample=False, frontend_device="cuda", backend_device="cuda",
    )

    # Scale intrinsics to processing resolution
    # TUM images are 640x480, DROID processes at original resolution
    intrinsics_t = torch.as_tensor(intrinsics)

    droid = Droid(slam_args)

    n_frames = len(rgb_files) if max_frames is None else min(len(rgb_files), max_frames)

    t0 = time.perf_counter()
    for t in range(n_frames):
        img_path = os.path.join(seq_dir, rgb_files[t])
        img = cv2.imread(img_path)
        if img is None:
            continue
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        droid.track(t, tensor, intrinsics=intrinsics_t)
    elapsed = time.perf_counter() - t0

    # Run global backend BA (same as droid.terminate() but skip traj_filler)
    del droid.frontend
    torch.cuda.empty_cache()
    droid.backend(7)
    torch.cuda.empty_cache()
    droid.backend(12)
    elapsed = time.perf_counter() - t0

    N = droid.video.counter.value
    kf_indices = droid.video.tstamp[:N].cpu().numpy().astype(int)

    # droid.video.poses stores world-to-camera SE3 as lietorch format
    # Invert to get camera-to-world (matching TUM ground truth convention)
    import lietorch
    poses_se3 = lietorch.SE3(droid.video.poses[:N])
    cam_to_world = poses_se3.inv().data.cpu().numpy()  # [N, 7]
    kf_timestamps = rgb_timestamps[kf_indices]

    del droid
    torch.cuda.empty_cache()

    return {
        "timestamps": kf_timestamps,
        "poses": cam_to_world,
        "kf_indices": kf_indices,
        "n_keyframes": N,
        "elapsed": elapsed,
        "n_frames": n_frames,
    }


# ── Run CUDA DROID-SLAM ───────────────────────────────────────────────
def run_cuda_slam(seq_dir, intrinsics, max_frames=None):
    """Run CUDA DROID-SLAM on a TUM sequence."""
    import subprocess
    import tempfile

    rgb_timestamps, rgb_files, _, _ = load_tum_sequence(seq_dir)
    n_frames = len(rgb_files) if max_frames is None else min(len(rgb_files), max_frames)

    # Export frames to binary format expected by cuda_droid
    frame_dir = tempfile.mkdtemp(prefix="tum_frames_")
    calib_file = os.path.join(frame_dir, "calib.bin")

    # Write calibration
    with open(calib_file, "wb") as f:
        f.write(struct.pack("4f", *intrinsics))

    # Export frames as binary files
    h, w = None, None
    exported = 0
    for i in range(n_frames):
        img = cv2.imread(os.path.join(seq_dir, rgb_files[i]))
        if img is None:
            continue
        h, w = img.shape[:2]
        frame_path = os.path.join(frame_dir, f"frame_{i:05d}.bin")
        with open(frame_path, "wb") as f:
            f.write(struct.pack("2i", h, w))
            f.write(img.astype(np.float32).tobytes())
        exported += 1

    # Write meta.txt expected by cuda_droid
    with open(os.path.join(frame_dir, "meta.txt"), "w") as f:
        f.write(f"{h} {w} {exported} 1\n")

    pose_file = os.path.join(frame_dir, "poses.bin")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["./cuda_slam/cuda_droid",
         "--weights", "cuda_slam/data/weights",
         "--frames", frame_dir,
         "--calib", calib_file,
         "--max-frames", str(n_frames),
         "--backend", "7", "12",
         "--cam-to-world",
         "--pose-output", pose_file],
        capture_output=True, text=True,
        cwd="/workspace/robot-video",
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"CUDA SLAM failed: {result.stderr[:500]}")
        return None

    # Parse output
    with open(pose_file, "rb") as f:
        nk = struct.unpack("i", f.read(4))[0]
        kf_indices = np.array(struct.unpack(f"{nk}i", f.read(nk * 4)))
        kf_poses = np.frombuffer(f.read(nk * 7 * 4), dtype=np.float32).reshape(nk, 7)

    kf_timestamps = rgb_timestamps[kf_indices]

    # Cleanup
    import shutil
    shutil.rmtree(frame_dir)

    return {
        "timestamps": kf_timestamps,
        "poses": kf_poses,
        "kf_indices": kf_indices,
        "n_keyframes": nk,
        "elapsed": elapsed,
        "n_frames": n_frames,
    }


# ── Evaluate trajectory ───────────────────────────────────────────────
def quat_to_matrix(quat):
    """Convert quaternion [qx,qy,qz,qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def evaluate_trajectory(slam_result, seq_dir):
    """Evaluate SLAM result against TUM ground truth."""
    _, _, gt_timestamps, gt_poses = load_tum_sequence(seq_dir)

    # Associate SLAM timestamps with GT timestamps
    matches = associate_timestamps(slam_result["timestamps"], gt_timestamps)

    if len(matches) < 3:
        return None

    est_positions = []
    gt_positions = []
    for slam_idx, gt_idx in matches:
        est_positions.append(slam_result["poses"][slam_idx, :3])
        gt_positions.append(gt_poses[gt_idx, :3])

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    ate = compute_ate(gt_positions, est_positions)
    ate["n_matched"] = len(matches)
    ate["n_keyframes"] = slam_result["n_keyframes"]
    ate["elapsed"] = slam_result["elapsed"]
    ate["n_frames_input"] = slam_result["n_frames"]
    ate["fps"] = slam_result["n_frames"] / slam_result["elapsed"]

    return ate


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate DROID-SLAM on TUM RGB-D")
    parser.add_argument("--sequences", nargs="+",
                        default=["fr1_desk", "fr1_room"],
                        choices=list(SEQUENCES.keys()))
    parser.add_argument("--cuda", action="store_true",
                        help="Also evaluate CUDA DROID-SLAM")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (for quick testing)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  DROID-SLAM Evaluation on TUM RGB-D Benchmark")
    print("=" * 70)

    for seq_name in args.sequences:
        seq_info = SEQUENCES[seq_name]
        seq_dir = os.path.join(base_dir, seq_info["path"])

        if not os.path.isdir(seq_dir):
            print(f"\nSkipping {seq_name}: directory not found at {seq_dir}")
            continue

        print(f"\n{'─' * 70}")
        print(f"  Sequence: {seq_name}")
        print(f"  Path: {seq_dir}")
        print(f"  Intrinsics: fx={seq_info['intrinsics'][0]:.1f} fy={seq_info['intrinsics'][1]:.1f} "
              f"cx={seq_info['intrinsics'][2]:.1f} cy={seq_info['intrinsics'][3]:.1f}")
        print(f"{'─' * 70}")

        # PyTorch DROID-SLAM
        print("\n  [PyTorch DROID-SLAM]")
        pt_result = run_pytorch_slam(seq_dir, seq_info["intrinsics"], args.max_frames)
        print(f"    Keyframes: {pt_result['n_keyframes']} / {pt_result['n_frames']} frames")
        print(f"    Time: {pt_result['elapsed']:.2f}s ({pt_result['n_frames']/pt_result['elapsed']:.1f} fps)")

        ate = evaluate_trajectory(pt_result, seq_dir)
        if ate:
            print(f"    ATE RMSE:   {ate['rmse']:.4f} m")
            print(f"    ATE Mean:   {ate['mean']:.4f} m")
            print(f"    ATE Median: {ate['median']:.4f} m")
            print(f"    ATE Max:    {ate['max']:.4f} m")
            print(f"    Scale:      {ate['scale']:.4f}")
            print(f"    Matched:    {ate['n_matched']} / {ate['n_keyframes']} keyframes")
        else:
            print("    ERROR: Not enough matches for evaluation")

        # CUDA DROID-SLAM
        if args.cuda:
            print("\n  [CUDA DROID-SLAM]")
            cu_result = run_cuda_slam(seq_dir, seq_info["intrinsics"], args.max_frames)
            if cu_result:
                print(f"    Keyframes: {cu_result['n_keyframes']} / {cu_result['n_frames']} frames")
                print(f"    Time: {cu_result['elapsed']:.2f}s ({cu_result['n_frames']/cu_result['elapsed']:.1f} fps)")

                ate_cu = evaluate_trajectory(cu_result, seq_dir)
                if ate_cu:
                    print(f"    ATE RMSE:   {ate_cu['rmse']:.4f} m")
                    print(f"    ATE Mean:   {ate_cu['mean']:.4f} m")
                    print(f"    ATE Median: {ate_cu['median']:.4f} m")
                    print(f"    ATE Max:    {ate_cu['max']:.4f} m")
                    print(f"    Scale:      {ate_cu['scale']:.4f}")
                    print(f"    Matched:    {ate_cu['n_matched']} / {ate_cu['n_keyframes']} keyframes")
                else:
                    print("    ERROR: Not enough matches for evaluation")

    print(f"\n{'=' * 70}")
    print("  Reference ATE RMSE values for DROID-SLAM (from paper):")
    print("    fr1/desk: 0.018 m")
    print("    fr1/room: 0.047 m")
    print("=" * 70)


if __name__ == "__main__":
    main()
