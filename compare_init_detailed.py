#!/usr/bin/env python3
"""Detailed comparison: dump intermediate values from PyTorch init to compare with CUDA."""

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
    print(f"Video: {width}x{height}, SLAM: {w1}x{h1}")

    h, w = h1 // 8, w1 // 8
    print(f"Feature map: {w}x{h}")

    print("\nDecoding frames...")
    frames = decode_frames(VIDEO, h1, w1)
    print(f"  {len(frames)} frames")

    # Run PyTorch
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

    # Hook into the frontend to capture init state
    import types
    from modules.corr import CorrBlock
    import geom.projective_ops as pops

    # Feed frames until warmup
    for t, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        # Use motion filter's track
        with torch.no_grad():
            droid.filterx.track(t, tensor, intrinsics=intrinsics_t)

        if droid.video.counter.value == 8:
            break

    N = droid.video.counter.value
    print(f"\nPyTorch has {N} keyframes at warmup")
    tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
    print(f"Timestamps: {tstamps.tolist()}")

    outdir = "/tmp/compare_init_detail"
    os.makedirs(outdir, exist_ok=True)

    # Dump feature maps (FP16)
    fmaps = droid.video.fmaps[:N, 0].cpu().numpy()  # [N, 128, h, w] FP16→FP32
    np.save(f"{outdir}/pt_fmaps.npy", fmaps)
    print(f"fmaps shape: {fmaps.shape}, dtype: {fmaps.dtype}")
    print(f"  fmap[0] mean: {fmaps[0].mean():.6f} std: {fmaps[0].std():.6f}")
    print(f"  fmap[1] mean: {fmaps[1].mean():.6f} std: {fmaps[1].std():.6f}")

    # Dump nets/inps (FP16 stored in video)
    nets = droid.video.nets[:N].cpu().numpy()
    inps = droid.video.inps[:N].cpu().numpy()
    np.save(f"{outdir}/pt_nets.npy", nets)
    np.save(f"{outdir}/pt_inps.npy", inps)
    print(f"nets shape: {nets.shape}, nets[0] unique channels: {len(np.unique(nets[0], axis=0))}")
    print(f"  nets[0] mean: {nets[0].mean():.6f}, nets[1] mean: {nets[1].mean():.6f}")

    # Dump initial poses and disps
    poses0 = droid.video.poses[:N].cpu().numpy()
    disps0 = droid.video.disps[:N].cpu().numpy()
    np.save(f"{outdir}/pt_poses_pre.npy", poses0)
    np.save(f"{outdir}/pt_disps_pre.npy", disps0)
    print(f"\nPre-init state:")
    for k in range(N):
        print(f"  KF{k}: pose t=[{poses0[k,0]:.6f} {poses0[k,1]:.6f} {poses0[k,2]:.6f}] "
              f"q=[{poses0[k,3]:.6f} {poses0[k,4]:.6f} {poses0[k,5]:.6f} {poses0[k,6]:.6f}] "
              f"disp_mean={disps0[k].mean():.6f}")

    # Now manually run initialization to capture intermediate values
    frontend = droid.frontend
    frontend.t0 = 0
    frontend.t1 = N

    # Step 1: add_neighborhood_factors(0, t1, r=3)
    graph = frontend.graph
    graph.add_neighborhood_factors(0, N, r=3)

    print(f"\nAfter add_neighborhood_factors:")
    print(f"  edges: {graph.ii.shape[0]}")
    print(f"  ii: {graph.ii.cpu().numpy().tolist()}")
    print(f"  jj: {graph.jj.cpu().numpy().tolist()}")

    # Step 2: First update iteration
    # This calls graph.update(1, use_inactive=True) which means t0=1
    print(f"\n=== First GRU+BA step ===")

    # Manually replicate the update to capture intermediates
    with torch.no_grad():
        with autocast(enabled=False):
            coords1, mask = droid.video.reproject(graph.ii, graph.jj)
            motn = torch.cat([coords1 - graph.coords0, graph.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        with autocast(enabled=True):
            corr = graph.corr(coords1)

        # Dump correlation
        corr_np = corr.float().cpu().numpy()
        np.save(f"{outdir}/pt_corr_step0.npy", corr_np)
        print(f"  corr shape: {corr_np.shape}, mean: {corr_np.mean():.6f}")

        # Dump motion
        motn_np = motn.float().cpu().numpy()
        np.save(f"{outdir}/pt_motn_step0.npy", motn_np)
        print(f"  motn shape: {motn_np.shape}, mean: {motn_np.mean():.6f}")

        # Dump coords1
        coords1_np = coords1.float().cpu().numpy()
        np.save(f"{outdir}/pt_coords1_step0.npy", coords1_np)
        print(f"  coords1 shape: {coords1_np.shape}, mean: {coords1_np.mean():.6f}")

        # Dump nets (hidden state) going into GRU
        nets_np = graph.net.float().cpu().numpy()
        np.save(f"{outdir}/pt_nets_step0.npy", nets_np)
        print(f"  nets shape: {nets_np.shape}, mean: {nets_np.mean():.6f}")

        # Dump inps
        inps_np = graph.inp.float().cpu().numpy()
        np.save(f"{outdir}/pt_inps_step0.npy", inps_np)
        print(f"  inps shape: {inps_np.shape}, mean: {inps_np.mean():.6f}")

        # Run GRU
        with autocast(enabled=True):
            net_out, delta, weight, damping, upmask = \
                graph.update_op(graph.net, graph.inp, corr, motn, graph.ii, graph.jj)

        # Dump GRU outputs
        delta_np = delta.float().cpu().numpy()
        weight_np = weight.float().cpu().numpy()
        damping_np = damping.float().cpu().numpy()
        net_np = net_out.float().cpu().numpy()
        np.save(f"{outdir}/pt_delta_step0.npy", delta_np)
        np.save(f"{outdir}/pt_weight_step0.npy", weight_np)
        np.save(f"{outdir}/pt_damping_step0.npy", damping_np)
        print(f"  delta shape: {delta_np.shape}, mean: {delta_np.mean():.6f}")
        print(f"  weight shape: {weight_np.shape}, mean: {weight_np.mean():.6f}")
        print(f"  damping mean: {damping_np.mean():.6f}")

        # Run the full first iteration via graph.update
        # Reset state first
        graph.net = graph.update_op.agg.conv1(net_out.view(-1,128,h,w).half()).float()  # restore

    # Actually, just run the proper update
    # Reset graph state (re-add factors)
    del droid
    torch.cuda.empty_cache()

    # Simpler approach: run full init, dump post-init state
    droid = Droid(slam_args)
    for t, frame in enumerate(frames):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        droid.track(t, tensor, intrinsics=intrinsics_t)
        if droid.frontend.is_initialized:
            break

    N = droid.video.counter.value
    poses_post = droid.video.poses[:N].cpu().numpy()
    disps_post = droid.video.disps[:N].cpu().numpy()
    np.save(f"{outdir}/pt_poses_post.npy", poses_post)
    np.save(f"{outdir}/pt_disps_post.npy", disps_post)

    print(f"\nPost-init state ({N} keyframes):")
    for k in range(N):
        print(f"  KF{k} (frame {int(droid.video.tstamp[k])}): "
              f"t=[{poses_post[k,0]:.6f} {poses_post[k,1]:.6f} {poses_post[k,2]:.6f}] "
              f"disp_mean={disps_post[k].mean():.6f}")

    print(f"\nDumped to {outdir}/")
    del droid
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
