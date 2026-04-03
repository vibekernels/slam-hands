#!/usr/bin/env python3
"""Compare CUDA vs PyTorch feature extraction (fmaps, nets, inps)."""

import os, sys, struct, argparse, subprocess, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_pipeline import get_video_metadata, get_iphone_intrinsics, _get_native_decode

VIDEO = "/workspace/IMG_1466.mov"
DROID_WEIGHTS = "/workspace/DROID-SLAM/checkpoints/droid.pth"
CUDA_SLAM_BIN = "./cuda_slam/cuda_droid"
CUDA_SLAM_WEIGHTS = "cuda_slam/data/weights"

def decode_frames(video_path, h1, w1, max_frames=30):
    nd = _get_native_decode()
    decoder = nd.AsyncVideoDecoder()
    decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)
    frames = []
    while len(frames) < max_frames:
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

    # Decode a few frames
    frames = decode_frames(VIDEO, h1, w1, max_frames=3)
    print(f"Decoded {len(frames)} frames")

    # ====== PyTorch ======
    from functools import partial
    if torch.__version__.startswith("2"):
        autocast = partial(torch.autocast, device_type="cuda")
    else:
        autocast = torch.cuda.amp.autocast

    sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')
    from droid_net import DroidNet
    from collections import OrderedDict

    net = DroidNet()
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(DROID_WEIGHTS, map_location='cpu').items()])
    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
    net.load_state_dict(state_dict)
    net.to("cuda:0").eval()

    MEAN = torch.as_tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    STDV = torch.as_tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

    for fi, frame in enumerate(frames[:2]):
        print(f"\n=== Frame {fi} ===")
        # Preprocess same as PyTorch
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda()  # [3, H, W] BGR
        image = tensor.unsqueeze(0)  # [1, 3, H, W]
        inputs = image[:, [2,1,0]] / 255.0  # BGR->RGB, normalize
        inputs = inputs.sub_(MEAN).div_(STDV)
        inputs = inputs.unsqueeze(0)  # [1, 1, 3, H, W]

        with torch.no_grad():
            with autocast(enabled=True):
                pt_fmap = net.fnet(inputs).squeeze(0)  # [1, 128, h, w] FP16
                pt_cmap = net.cnet(inputs)  # [1, 1, 256, h, w] FP16
                pt_net_raw, pt_inp_raw = pt_cmap.split([128, 128], dim=2)
                pt_net = pt_net_raw.tanh().squeeze(0)  # [1, 128, h, w] FP16
                pt_inp = pt_inp_raw.relu().squeeze(0)  # [1, 128, h, w] FP16

        pt_fmap_np = pt_fmap.float().cpu().numpy()[0]  # [128, h, w]
        pt_net_np = pt_net.float().cpu().numpy()[0]     # [128, h, w]
        pt_inp_np = pt_inp.float().cpu().numpy()[0]     # [128, h, w]

        print(f"  PT fmap: mean={pt_fmap_np.mean():.6f} std={pt_fmap_np.std():.6f} "
              f"range=[{pt_fmap_np.min():.4f}, {pt_fmap_np.max():.4f}]")
        print(f"  PT net:  mean={pt_net_np.mean():.6f} std={pt_net_np.std():.6f}")
        print(f"  PT inp:  mean={pt_inp_np.mean():.6f} std={pt_inp_np.std():.6f}")

        # Save for comparison
        np.save(f"/tmp/pt_fmap_{fi}.npy", pt_fmap_np)
        np.save(f"/tmp/pt_net_{fi}.npy", pt_net_np)
        np.save(f"/tmp/pt_inp_{fi}.npy", pt_inp_np)

    del net
    torch.cuda.empty_cache()

    # ====== CUDA ======
    # Run CUDA with debug dump, only 2 frames
    tmpdir = tempfile.mkdtemp(prefix="compare_feat_")
    calib_path = os.path.join(tmpdir, "calib.bin")
    pose_path = os.path.join(tmpdir, "poses.bin")

    with open(calib_path, "wb") as f:
        f.write(struct.pack("4f", *intrinsics))

    cmd = [
        CUDA_SLAM_BIN,
        "--weights", CUDA_SLAM_WEIGHTS,
        "--calib", calib_path,
        "--stdin", str(h1), str(w1),
        "--max-frames", "2",
        "--pose-output", pose_path,
        "--debug-dump",
    ]

    stderr_path = os.path.join(tmpdir, "stderr.log")
    with open(stderr_path, "w") as stderr_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr_f)
        for frame in frames[:2]:
            proc.stdin.write(frame.astype(np.float32).tobytes())
        proc.stdin.close()
        proc.wait()

    with open(stderr_path, "r") as f:
        stderr = f.read()
    print(f"\nCUDA stderr: {stderr[:500]}")

    # Load CUDA fmaps
    for fi in range(2):
        cuda_fmap_path = f"/tmp/cuda_fmap{fi}.bin"
        if os.path.exists(cuda_fmap_path):
            with open(cuda_fmap_path, "rb") as f:
                shape = struct.unpack("3i", f.read(12))
                cu_fmap = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)

            pt_fmap = np.load(f"/tmp/pt_fmap_{fi}.npy")
            print(f"\n=== Frame {fi} fmap comparison ===")
            print(f"  CU fmap: shape={cu_fmap.shape}, mean={cu_fmap.mean():.6f}")
            print(f"  PT fmap: shape={pt_fmap.shape}, mean={pt_fmap.mean():.6f}")
            diff = np.abs(cu_fmap - pt_fmap)
            print(f"  Diff: mean={diff.mean():.8f}, max={diff.max():.8f}, "
                  f"pct_match(1e-3)={100*(diff < 1e-3).mean():.1f}%")

            # Check per-channel correlation
            for c in [0, 1, 63, 127]:
                cc = np.corrcoef(cu_fmap[c].flatten(), pt_fmap[c].flatten())[0, 1]
                print(f"  Ch{c}: CU_mean={cu_fmap[c].mean():.6f} PT_mean={pt_fmap[c].mean():.6f} corr={cc:.6f}")

    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
