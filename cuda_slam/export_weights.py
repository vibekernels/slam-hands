#!/usr/bin/env python3
"""Export DROID-SLAM weights and test data to binary files for the CUDA implementation."""

import os
import sys
import struct
import numpy as np
import torch

sys.path.insert(0, '/workspace/DROID-SLAM')
sys.path.insert(0, '/workspace/DROID-SLAM/droid_slam')


def save_tensor(path, tensor):
    """Save a tensor as raw float32 binary with a small header."""
    t = tensor.detach().cpu().float().contiguous().numpy()
    with open(path, 'wb') as f:
        # Header: ndim, then each dim size (as int32)
        f.write(struct.pack('i', t.ndim))
        for d in t.shape:
            f.write(struct.pack('i', d))
        f.write(t.tobytes())


def export_weights(ckpt_path, out_dir):
    """Export model weights to binary files."""
    os.makedirs(out_dir, exist_ok=True)

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Strip 'module.' prefix if present (DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace('module.', '')
        cleaned[key] = v

    # Save each parameter
    for name, param in sorted(cleaned.items()):
        fname = name.replace('.', '_') + '.bin'
        save_tensor(os.path.join(out_dir, fname), param)
        print(f'  {name:55s} {str(list(param.shape)):30s} -> {fname}')

    # Save a manifest
    with open(os.path.join(out_dir, 'manifest.txt'), 'w') as f:
        for name, param in sorted(cleaned.items()):
            fname = name.replace('.', '_') + '.bin'
            shape_str = ','.join(str(d) for d in param.shape)
            f.write(f'{name}\t{fname}\t{shape_str}\n')

    print(f'\nExported {len(cleaned)} parameters to {out_dir}/')


def export_test_frames(video_path, out_dir, max_frames=200, stride=2):
    """Export video frames as raw float32 RGB for testing."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target resolution matching DROID-SLAM defaults
    # The pipeline resizes to make dims divisible by 8
    tgt_h = (H // 8) * 8
    tgt_w = (W // 8) * 8

    # Save metadata
    with open(os.path.join(out_dir, 'meta.txt'), 'w') as f:
        f.write(f'{tgt_h} {tgt_w} {max_frames} {stride}\n')

    count = 0
    frame_idx = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        # Resize if needed
        if frame.shape[0] != tgt_h or frame.shape[1] != tgt_w:
            frame = cv2.resize(frame, (tgt_w, tgt_h))

        # Save as float32 RGB (HWC) - same format DROID-SLAM expects
        frame_f32 = frame.astype(np.float32)  # BGR, 0-255
        fname = os.path.join(out_dir, f'frame_{count:05d}.bin')
        with open(fname, 'wb') as f:
            f.write(struct.pack('ii', tgt_h, tgt_w))
            f.write(frame_f32.tobytes())

        count += 1

    cap.release()
    print(f'Exported {count} frames ({tgt_h}x{tgt_w}) to {out_dir}/')


def export_calibration(calib_path, out_dir):
    """Export camera calibration."""
    os.makedirs(out_dir, exist_ok=True)
    calib = np.loadtxt(calib_path)
    # calib is [fx, fy, cx, cy]
    with open(os.path.join(out_dir, 'calib.bin'), 'wb') as f:
        f.write(calib.astype(np.float32).tobytes())
    print(f'Exported calibration {calib} to {out_dir}/calib.bin')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/workspace/DROID-SLAM/droid.pth')
    parser.add_argument('--video', default=None, help='Video file to export test frames from')
    parser.add_argument('--calib', default=None, help='Calibration file')
    parser.add_argument('--out', default='/workspace/robot-video/cuda_slam/data')
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()

    print("=== Exporting weights ===")
    export_weights(args.checkpoint, os.path.join(args.out, 'weights'))

    if args.video:
        print("\n=== Exporting test frames ===")
        export_test_frames(args.video, os.path.join(args.out, 'frames'),
                          args.max_frames, args.stride)

    if args.calib:
        print("\n=== Exporting calibration ===")
        export_calibration(args.calib, os.path.join(args.out))
