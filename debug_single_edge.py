#!/usr/bin/env python3
"""Compare single-edge GRU outputs between PyTorch and CUDA DROID-SLAM."""
import os, sys, struct, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/workspace/DROID-SLAM')
sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
from annotate_pipeline import get_video_metadata, get_iphone_intrinsics, _get_native_decode

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"

fps, width, height, _ = get_video_metadata(VIDEO)
h0, w0 = height, width
scale = np.sqrt((384 * 512) / (h0 * w0))
h1 = int(h0 * scale) // 8 * 8
w1 = int(w0 * scale) // 8 * 8
h, w = h1 // 8, w1 // 8  # 1/8 resolution
fx, fy, cx, cy = get_iphone_intrinsics(width, height)
fx_s, fy_s = fx * (w1 / w0), fy * (h1 / h0)
cx_s, cy_s = cx * (w1 / w0), cy * (h1 / h0)
intr = torch.as_tensor([fx_s, fy_s, cx_s, cy_s])
print(f"Video: {w1}x{h1}, intrinsics at SLAM res: {fx_s:.1f} {fy_s:.1f} {cx_s:.1f} {cy_s:.1f}")
print(f"1/8 res: {w}x{h}, hw={h*w}")

# ===== Load 2 frames =====
nd = _get_native_decode()
decoder = nd.AsyncVideoDecoder()
decoder.start(VIDEO, w1, h1, slam_only=True, queue_depth=128)

frames = []
for _ in range(10):
    result = decoder.get_next()
    if result is None: break
    _, slam_bgr = result
    tensor = torch.from_numpy(slam_bgr).permute(2, 0, 1).unsqueeze(0).float().cuda()
    frames.append(tensor)
decoder.stop()
print(f"Loaded {len(frames)} frames")

# ===== PyTorch: extract features and run single update =====
from droid_net import DroidNet
import geom.projective_ops as pops

net = DroidNet()
state_dict = torch.load(DROID_WEIGHTS)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# Trim 3-channel weights to 2-channel for delta/weight heads
for key in list(state_dict.keys()):
    if 'weight.2.' in key or 'delta.2.' in key:
        if state_dict[key].shape[0] == 3:
            state_dict[key] = state_dict[key][:2]
net.load_state_dict(state_dict)
net.cuda().eval()

# Normalize frames like DROID-SLAM does
MEAN = torch.as_tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1)
STD = torch.as_tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1)

with torch.no_grad():
    # Process frame 0 and frame 5 (which should be a keyframe)
    img0 = frames[0] / 255.0  # [1, 3, H, W]
    img5 = frames[5] / 255.0

    # Run feature extraction
    images = torch.stack([img0, img5], dim=1)  # [1, 2, 3, H, W]
    images = images.cuda()

    # Extract features - input is BGR [1, 2, 3, H, W]
    fmaps, gmaps, imaps = net.extract_features(images)
    # fmaps: [1, 2, 128, h, w], gmaps (net): [1, 2, 128, h, w], imaps (inp): [1, 2, 128, h, w]

    print(f"\nPyTorch features:")
    print(f"  fmap shape: {fmaps.shape}, range: [{fmaps.min():.4f}, {fmaps.max():.4f}]")
    print(f"  gmap shape: {gmaps.shape}, range: [{gmaps.min():.4f}, {gmaps.max():.4f}]")
    print(f"  imap shape: {imaps.shape}, range: [{imaps.min():.4f}, {imaps.max():.4f}]")

    # Build correlation between frame 0 and frame 5
    sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam/modules')
    from corr import CorrBlock
    corr_fn = CorrBlock(fmaps[:, 0:1], fmaps[:, 1:2])

    # Identity coords - CorrBlock expects [batch, num, h, w, 2]
    coords0 = pops.coords_grid(h, w, device='cuda').unsqueeze(0).unsqueeze(0)  # [1, 1, h, w, 2]
    coords1 = coords0.clone()

    # Lookup correlation at identity coords
    corr_vals = corr_fn(coords1)  # [1, 1, D*D*4, h, w]
    print(f"  Correlation shape: {corr_vals.shape}, range: [{corr_vals.min():.4f}, {corr_vals.max():.4f}]")
    print(f"  Correlation mean: {corr_vals.mean():.6f}, std: {corr_vals.std():.6f}")

    # Run single GRU update with zero motion
    flow = torch.zeros(1, 1, 4, h, w, device='cuda')
    gmap = gmaps[:, 0:1]  # net from frame 0
    imap = imaps[:, 0:1]  # inp from frame 0

    pt_net, pt_delta, pt_weight, pt_eta, pt_upmask = net.update(
        gmap, imap, corr_vals, flow,
        ii=torch.tensor([0], device='cuda'),
        jj=torch.tensor([1], device='cuda'))

    # delta and weight are in [B, N, H, W, 2] format (permuted in update)
    print(f"\nPyTorch GRU output:")
    print(f"  delta shape: {pt_delta.shape}, mean: {pt_delta.mean():.6f}, std: {pt_delta.std():.6f}")
    print(f"  delta range: [{pt_delta.min():.4f}, {pt_delta.max():.4f}]")
    print(f"  weight shape: {pt_weight.shape}, mean: {pt_weight.mean():.6f}")
    print(f"  weight range: [{pt_weight.min():.4f}, {pt_weight.max():.4f}]")

    # Look at specific values
    d = pt_delta[0, 0]  # [H, W, 2]
    print(f"  delta at (0,0): [{d[0,0,0]:.4f}, {d[0,0,1]:.4f}]")
    print(f"  delta at (h/2,w/2): [{d[h//2,w//2,0]:.4f}, {d[h//2,w//2,1]:.4f}]")

    # Flow magnitude (what motion filter uses)
    flow_mag = pt_delta.norm(dim=-1).mean().item()
    print(f"  flow magnitude (motion filter): {flow_mag:.4f}")

    # ===== Now compute CUDA-equivalent values in PyTorch =====
    print("\n\n===== Comparing intermediate values =====")

    # Get raw fmaps for frame 0 and 5
    f0 = fmaps[0, 0]  # [128, h, w]
    f5 = fmaps[0, 1]  # [128, h, w]

    # Correlation matrix: corr[src, tgt] = dot(f0[src], f5[tgt]) / 16
    f0_flat = f0.reshape(128, -1) / 4.0  # [128, hw]
    f5_flat = f5.reshape(128, -1) / 4.0
    corr_matrix = torch.matmul(f0_flat.T, f5_flat)  # [hw, hw]

    print(f"Correlation matrix: shape={corr_matrix.shape}")
    print(f"  range: [{corr_matrix.min():.4f}, {corr_matrix.max():.4f}]")
    print(f"  mean: {corr_matrix.mean():.6f}, std: {corr_matrix.std():.6f}")
    print(f"  diagonal mean: {corr_matrix.diag().mean():.6f}")

    # Check specific values
    print(f"  corr[0,0] = {corr_matrix[0,0]:.6f}")
    print(f"  corr[0,1] = {corr_matrix[0,1]:.6f}")
    print(f"  corr[1,0] = {corr_matrix[1,0]:.6f}")
    print(f"  corr[100,100] = {corr_matrix[100,100]:.6f}")

    # Also check the CorrBlock's internal values
    print(f"\nCorrBlock pyramid shapes:")
    for i, p in enumerate(corr_fn.corr_pyramid):
        print(f"  Level {i}: {p.shape}, range: [{p.min():.4f}, {p.max():.4f}]")

    # Check feature stats for comparison with CUDA
    print(f"\nFeature stats for frame 0:")
    print(f"  fmap: mean={f0.mean():.6f}, std={f0.std():.6f}")
    print(f"  net: mean={gmaps[0,0].mean():.6f}, std={gmaps[0,0].std():.6f}")
    print(f"  inp: mean={imaps[0,0].mean():.6f}, std={imaps[0,0].std():.6f}")

    print(f"\nFeature stats for frame 5:")
    print(f"  fmap: mean={fmaps[0,1].mean():.6f}, std={fmaps[0,1].std():.6f}")

    # Dump features to files for CUDA comparison
    np.save("/tmp/pt_fmap0.npy", f0.cpu().numpy())
    np.save("/tmp/pt_fmap5.npy", fmaps[0, 1].cpu().numpy())
    np.save("/tmp/pt_corr_matrix.npy", corr_matrix.cpu().numpy())
    np.save("/tmp/pt_corr_sampled.npy", corr_vals[0, 0].cpu().numpy())
    np.save("/tmp/pt_net0.npy", gmaps[0, 0].cpu().numpy())
    np.save("/tmp/pt_inp0.npy", imaps[0, 0].cpu().numpy())
    np.save("/tmp/pt_delta.npy", pt_delta[0, 0].cpu().numpy())
    print("\nSaved PyTorch intermediate values to /tmp/pt_*.npy")
