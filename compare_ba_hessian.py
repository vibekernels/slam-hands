#!/usr/bin/env python3
"""Compare first BA Hessian/gradient between CUDA and PyTorch."""

import os, sys, struct, argparse, tempfile, shutil
import numpy as np
import torch
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_pipeline import get_video_metadata, get_iphone_intrinsics, _get_native_decode

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast

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

def main():
    fps, width, height, _ = get_video_metadata(VIDEO)
    h0, w0 = height, width
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale) // 8 * 8
    w1 = int(w0 * scale) // 8 * 8
    fx, fy, cx, cy = get_iphone_intrinsics(width, height)
    intrinsics = np.array([fx*w1/w0, fy*h1/h0, cx*w1/w0, cy*h1/h0], dtype=np.float32)
    h, w = h1 // 8, w1 // 8
    hw = h * w
    print(f"Video: {width}x{height}, SLAM: {w1}x{h1}, feat: {w}x{h}")

    print("\nDecoding frames...")
    frames = decode_frames(VIDEO, h1, w1)
    print(f"  {len(frames)} frames")

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

    # Feed frames until warmup
    for t, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            droid.filterx.track(t, tensor, intrinsics=intrinsics_t)
        if droid.video.counter.value == 8:
            break

    N = droid.video.counter.value
    print(f"\nPyTorch has {N} keyframes at warmup")

    # Manually run init first step
    frontend = droid.frontend
    frontend.t0 = 0
    frontend.t1 = N
    graph = frontend.graph
    graph.add_neighborhood_factors(0, N, r=3)

    print(f"Edges: {graph.ii.shape[0]}")
    print(f"  ii: {graph.ii.cpu().tolist()}")
    print(f"  jj: {graph.jj.cpu().tolist()}")

    # Run ONE GRU step
    from modules.corr import CorrBlock
    with torch.no_grad():
        with autocast(enabled=False):
            coords1, mask = droid.video.reproject(graph.ii, graph.jj)
            motn = torch.cat([coords1 - graph.coords0, graph.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        corr = graph.corr(coords1)

        with autocast(enabled=True):
            net_out, delta, weight, damping, upmask = \
                graph.update_op(graph.net, graph.inp, corr, motn, graph.ii, graph.jj)

        # Compute target and weight (matching graph.update() lines 233-238)
        target = coords1 + delta.float()
        weight_out = weight.float()
        graph.damping[torch.unique(graph.ii)] = damping

        # Build BA input (matching graph.update() lines 240-258)
        t0_ba = 1
        t1_ba = N
        EP = 1e-7

        # No inactive edges in first step
        ii = graph.ii
        jj = graph.jj
        ba_target = target
        ba_weight = weight_out

        damping_scaled = .2 * graph.damping[torch.unique(ii)].contiguous() + EP

        ba_target_fmt = ba_target.view(-1, h, w, 2).permute(0, 3, 1, 2).contiguous()
        ba_weight_fmt = ba_weight.view(-1, h, w, 2).permute(0, 3, 1, 2).contiguous()

        print(f"\n=== First BA inputs ===")
        print(f"  target: {ba_target_fmt.shape}, mean={ba_target_fmt.mean():.6f}")
        print(f"  weight: {ba_weight_fmt.shape}, mean={ba_weight_fmt.mean():.6f}")
        print(f"  damping: {damping_scaled.shape}, mean={damping_scaled.mean():.8f}")

        # Save target and weight for comparison
        np.save("/tmp/pt_ba0_target.npy", ba_target_fmt.cpu().numpy())
        np.save("/tmp/pt_ba0_weight.npy", ba_weight_fmt.cpu().numpy())

        # Now manually call droid_backends.ba to get dx, but we need to capture the Hessian.
        # Since the C++ backend doesn't expose the Hessian, let's replicate the BA in Python.

        import droid_backends

        # Get pre-BA state
        poses_pre = droid.video.poses[:N].clone()
        disps_pre = droid.video.disps[:N].clone()

        print(f"  poses[:3,0]: {poses_pre[:3,0].tolist()}")
        print(f"  disps_mean: {disps_pre.mean():.6f}")

        # Call BA with 1 iteration to match CUDA's single BA call
        droid_backends.ba(
            droid.video.poses, droid.video.disps, droid.video.intrinsics[0], droid.video.disps_sens,
            ba_target_fmt, ba_weight_fmt, damping_scaled, ii, jj,
            t0_ba, t1_ba, 1, 1e-4, 0.1, False)
        droid.video.disps.clamp_(min=0.001)

        poses_post = droid.video.poses[:N].clone()
        disps_post = droid.video.disps[:N].clone()

        # Compute dx from poses
        # dx is the se3 lie algebra update that was applied
        # We can approximate it from the pose change
        dx_norms = []
        for k in range(1, N):
            dt = poses_post[k, :3] - poses_pre[k, :3]
            dq = poses_post[k, 3:] - poses_pre[k, 3:]
            dx_norms.append(torch.cat([dt, dq[:3]]).norm().item())
        total_dx = sum(n**2 for n in dx_norms)**0.5
        print(f"\n  Post-BA (1 iter) approx dx_norm ~ {total_dx:.6f}")

        for k in range(N):
            p = poses_post[k]
            d = disps_post[k].mean()
            print(f"  KF{k}: t=[{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}] disp={d:.6f}")

        # Now replicate the Hessian computation manually in Python using the kernel output
        # Run projective_transform_kernel equivalent
        # Actually, let's just compare the final target/weight going to BA with CUDA's

    print("\n=== Comparing with CUDA dumps ===")

    def load_cuda_bin4(path):
        with open(path, 'rb') as f:
            dims = struct.unpack('4i', f.read(16))
            return np.frombuffer(f.read(), dtype=np.float32).reshape(dims)

    # Compare target
    cu_target = load_cuda_bin4("/tmp/cuda_ba0_target.bin")
    pt_target = np.load("/tmp/pt_ba0_target.npy")
    print(f"  target: CU shape={cu_target.shape} PT shape={pt_target.shape}")

    # Edge ordering differs — need to match
    pt_ii = graph.ii.cpu().tolist()
    pt_jj = graph.jj.cpu().tolist()

    # Build CUDA edge order
    cu_ii, cu_jj = [], []
    for i in range(N):
        for j in range(max(0, i-3), i):
            cu_ii.append(i); cu_jj.append(j)
            cu_jj.append(i); cu_ii.append(j)

    pt_edge_map = {(i,j): idx for idx, (i,j) in enumerate(zip(pt_ii, pt_jj))}
    cu_to_pt = {}
    for idx, (i, j) in enumerate(zip(cu_ii, cu_jj)):
        if (i,j) in pt_edge_map:
            cu_to_pt[idx] = pt_edge_map[(i,j)]

    target_diffs = []
    weight_diffs = []
    cu_weight = load_cuda_bin4("/tmp/cuda_ba0_weight.bin")
    pt_weight_np = np.load("/tmp/pt_ba0_weight.npy")
    for cu_e, pt_e in cu_to_pt.items():
        td = np.abs(cu_target[cu_e] - pt_target[pt_e]).max()
        wd = np.abs(cu_weight[cu_e] - pt_weight_np[pt_e]).max()
        target_diffs.append(td)
        weight_diffs.append(wd)
    print(f"  target max diff: {max(target_diffs):.8f}")
    print(f"  weight max diff: {max(weight_diffs):.8f}")

    # Compare Hessian
    if os.path.exists("/tmp/cuda_ba0_S.bin"):
        with open("/tmp/cuda_ba0_S.bin", "rb") as f:
            S_size = struct.unpack("i", f.read(4))[0]
            S_cu = np.frombuffer(f.read(), dtype=np.float64).reshape(S_size, S_size)
        with open("/tmp/cuda_ba0_b.bin", "rb") as f:
            b_size = struct.unpack("i", f.read(4))[0]
            b_cu = np.frombuffer(f.read(), dtype=np.float64).reshape(b_size)
        print(f"\n  CUDA Hessian: size={S_size}, diag mean={np.diag(S_cu).mean():.8f}")
        print(f"  CUDA b: mean={b_cu.mean():.8f}")
        print(f"  CUDA S[0,:6] = {S_cu[0,:6]}")
        print(f"  CUDA b[:6] = {b_cu[:6]}")

if __name__ == "__main__":
    main()
