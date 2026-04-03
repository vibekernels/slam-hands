#!/usr/bin/env python3
"""Compare CUDA exported weights against PyTorch checkpoint weights."""

import struct
import numpy as np
import torch

CKPT_PATH = '/workspace/DROID-SLAM/checkpoints/droid.pth'
WEIGHTS_DIR = '/workspace/robot-video/cuda_slam/data/weights'


def load_bin(name):
    """Load a .bin weight file saved by export_weights.py."""
    path = f'{WEIGHTS_DIR}/{name}.bin'
    with open(path, 'rb') as f:
        ndim = struct.unpack('i', f.read(4))[0]
        shape = tuple(struct.unpack('i', f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    return data


def pytorch_key_to_bin_name(key):
    """Convert PyTorch state dict key to .bin filename (without extension)."""
    # Strip 'model.' wrapper if present
    key = key.replace('module.', '')
    return key.replace('.', '_')


def compare(label, pt_tensor, bin_name):
    pt = pt_tensor.detach().cpu().float().numpy()
    cuda = load_bin(bin_name)
    max_diff = np.max(np.abs(pt - cuda))
    mean_diff = np.mean(np.abs(pt - cuda))
    match = np.allclose(pt, cuda, atol=1e-6)
    print(f"  {label}")
    print(f"    Shape   : pt={pt.shape}  cuda={cuda.shape}")
    print(f"    Max diff: {max_diff:.2e}  Mean diff: {mean_diff:.2e}  Match(<1e-6): {match}")
    return match


# Load checkpoint
print("Loading PyTorch checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location='cpu')

# The checkpoint may be wrapped; try common keys
if isinstance(ckpt, dict):
    if 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
else:
    state = ckpt

# Strip 'module.' prefix
cleaned = {k.replace('module.', ''): v for k, v in state.items()}

print(f"Checkpoint has {len(cleaned)} parameters.\n")
print("Sample keys (first 10):")
for k in sorted(cleaned.keys())[:10]:
    print(f"  {k}: {list(cleaned[k].shape)}")

print("\n--- Weight Comparisons ---\n")

results = []

# 1. fnet first conv layer weight
key = 'fnet.conv1.weight'
bin_name = pytorch_key_to_bin_name(key)
print(f"[1] fnet first conv1.weight  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

# 2. fnet conv1 bias
key = 'fnet.conv1.bias'
bin_name = pytorch_key_to_bin_name(key)
print(f"\n[2] fnet first conv1.bias  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

# 3. Deeper fnet residual block weight (layer3_0 conv1)
key = 'fnet.layer3.0.conv1.weight'
bin_name = pytorch_key_to_bin_name(key)
print(f"\n[3] fnet layer3_0 conv1.weight  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

# 4. fnet layer3_0 downsample conv (stride path)
key = 'fnet.layer3.0.downsample.0.weight'
bin_name = pytorch_key_to_bin_name(key)
print(f"\n[4] fnet layer3_0 downsample.weight  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

# 5. cnet first conv layer weight
key = 'cnet.conv1.weight'
bin_name = pytorch_key_to_bin_name(key)
print(f"\n[5] cnet conv1.weight  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

# 6. cnet layer2_0 conv2 (deeper)
key = 'cnet.layer2.0.conv2.weight'
bin_name = pytorch_key_to_bin_name(key)
print(f"\n[6] cnet layer2_0 conv2.weight  (bin: {bin_name})")
results.append(compare(key, cleaned[key], bin_name))

print("\n--- Summary ---")
print(f"All {len(results)} comparisons passed: {all(results)}")
if not all(results):
    print("FAILED comparisons detected!")
