#!/usr/bin/env python3
"""Evaluate WiLoR hand pose on FreiHAND evaluation set.

Feeds cropped hand images directly to WiLoR (skip YOLO detection) and
compares predicted 3D keypoints to ground truth.

Metrics:
  - MPJPE: Mean Per-Joint Position Error (mm)
  - PA-MPJPE: Procrustes-Aligned MPJPE (mm)
  - AUC: Area Under Curve (20-50mm thresholds)

Usage:
    python3 eval_hand_freihand.py [--max-samples N]
"""

import argparse
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", message="Importing from timm.models.layers", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using torch.cross without specifying the dim")

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

WILOR_DIR = "/workspace/WiLoR"
FREIHAND_DIR = "/workspace/robot-video/benchmarks/freihand/evaluation"


def procrustes_align(predicted, target):
    """Align predicted to target using Procrustes (translation + rotation + scale)."""
    mu_p = predicted.mean(axis=0)
    mu_t = target.mean(axis=0)
    p_c = predicted - mu_p
    t_c = target - mu_t

    H = p_c.T @ t_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])
    R = Vt.T @ sign_matrix @ U.T

    scale = np.trace(np.diag(S) @ sign_matrix) / (p_c ** 2).sum()
    aligned = scale * (p_c @ R.T) + mu_t
    return aligned


def compute_auc(errors, thresholds=np.linspace(20, 50, 100)):
    """Compute AUC (area under PCK curve) for given thresholds in mm."""
    pck = np.array([(errors < t).mean() for t in thresholds])
    auc = np.trapz(pck, thresholds) / (thresholds[-1] - thresholds[0])
    return auc


def load_wilor(device="cuda"):
    """Load WiLoR model using the same approach as annotate_pipeline.py."""
    # Stub pytorch_lightning
    import torch.nn as nn
    import types
    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *a, **kw): pass
        def log(self, *a, **kw): pass
    class _RankZero:
        @staticmethod
        def rank_zero_only(fn): return fn
    pl_stub = types.ModuleType("pytorch_lightning")
    pl_stub.LightningModule = _LightningModule
    pl_util_stub = types.ModuleType("pytorch_lightning.utilities")
    pl_util_stub.rank_zero_only = _RankZero.rank_zero_only
    pl_rz_stub = types.ModuleType("pytorch_lightning.utilities.rank_zero")
    pl_rz_stub.rank_zero_only = _RankZero.rank_zero_only
    pl_stub.utilities = pl_util_stub
    pl_util_stub.rank_zero = pl_rz_stub
    sys.modules["pytorch_lightning"] = pl_stub
    sys.modules["pytorch_lightning.utilities"] = pl_util_stub
    sys.modules["pytorch_lightning.utilities.rank_zero"] = pl_rz_stub

    sys.path.insert(0, WILOR_DIR)
    from wilor.configs import get_config
    from wilor.models.wilor import WiLoR

    orig_cwd = os.getcwd()
    os.chdir(WILOR_DIR)
    try:
        model_cfg = get_config(
            os.path.join(WILOR_DIR, 'pretrained_models', 'model_config.yaml'),
            update_cachedir=True,
        )
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE == 256:
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        _mano_dir = os.path.join(WILOR_DIR, 'mano_data')
        model_cfg.MANO.DATA_DIR = _mano_dir + '/'
        model_cfg.MANO.MODEL_PATH = _mano_dir + '/'
        model_cfg.MANO.MEAN_PARAMS = os.path.join(_mano_dir, 'mano_mean_params.npz')
        model_cfg.freeze()

        with open(os.devnull, 'w') as _devnull:
            _saved = sys.stdout
            sys.stdout = _devnull
            try:
                model = WiLoR(cfg=model_cfg, init_renderer=False)
            finally:
                sys.stdout = _saved

        sd_path = os.path.join(WILOR_DIR, 'pretrained_models', 'wilor_final_statedict.pt')
        ckpt_path = os.path.join(WILOR_DIR, 'pretrained_models', 'wilor_final.ckpt')
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location='cpu', mmap=True, weights_only=True)
        else:
            sd = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=False)['state_dict']
        model.load_state_dict(sd, strict=False, assign=True)
    finally:
        os.chdir(orig_cwd)

    model = model.to(device).eval().half()
    return model, model_cfg


def preprocess_freihand_crop(img_bgr, img_size, mean, std):
    """Preprocess a FreiHAND 224x224 crop for WiLoR.

    FreiHAND images are already cropped hands. Resize to img_size x img_size
    (256x256). The backbone internally slices to 256x192.
    """
    img_resized = cv2.resize(img_bgr, (img_size, img_size))

    # Convert BGR -> RGB, normalize
    img_rgb = img_resized[:, :, ::-1].astype(np.float32)
    img_rgb = (img_rgb - mean) / std

    # HWC -> CHW
    img_tensor = img_rgb.transpose(2, 0, 1)
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description="Evaluate WiLoR on FreiHAND")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    device = "cuda"

    # Load annotations
    anno_dir = os.path.join(FREIHAND_DIR, "anno")
    rgb_dir = os.path.join(FREIHAND_DIR, "rgb")
    anno_files = sorted(os.listdir(anno_dir))

    if args.max_samples:
        anno_files = anno_files[:args.max_samples]
    n_samples = len(anno_files)
    print(f"FreiHAND evaluation: {n_samples} samples")

    # Load GT
    gt_xyz = []
    gt_K = []
    img_paths = []
    for af in anno_files:
        with open(os.path.join(anno_dir, af)) as f:
            data = json.load(f)
        gt_xyz.append(np.array(data['xyz']))
        gt_K.append(np.array(data['K']))
        idx = af.replace('.json', '')
        img_paths.append(os.path.join(rgb_dir, f"{idx}.jpg"))

    gt_xyz = np.array(gt_xyz, dtype=np.float32)  # [N, 21, 3] in meters
    gt_K = np.array(gt_K, dtype=np.float32)  # [N, 3, 3]

    # Load model
    print("Loading WiLoR model...")
    model, model_cfg = load_wilor(device)

    img_size_cfg = model_cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(model_cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
    std = 255.0 * np.array(model_cfg.MODEL.IMAGE_STD, dtype=np.float32)

    # Process in batches
    print(f"Running WiLoR inference (batch_size={args.batch_size})...")
    pred_xyz = np.zeros((n_samples, 21, 3), dtype=np.float32)

    t0 = time.perf_counter()
    for batch_start in range(0, n_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_samples)
        bs = batch_end - batch_start

        # Preprocess batch
        imgs = []
        for i in range(batch_start, batch_end):
            img = cv2.imread(img_paths[i])
            img_t = preprocess_freihand_crop(img, img_size_cfg, mean, std)
            imgs.append(torch.from_numpy(img_t).half())

        img_batch = torch.stack(imgs).to(device)

        # For FreiHAND: image is the full crop, bbox covers entire image
        # box_center = [112, 112] (center of 224x224)
        # box_size = 224 (covers whole image)
        box_centers = torch.full((bs, 2), 112.0, dtype=torch.float32, device=device)
        box_sizes = torch.full((bs,), 224.0, dtype=torch.float32, device=device)
        img_sizes = torch.full((bs, 2), 224.0, dtype=torch.float32, device=device)
        rights = torch.ones(bs, dtype=torch.float32, device=device)  # FreiHAND is right hand

        batch = {
            'img': img_batch,
            'box_center': box_centers,
            'box_size': box_sizes,
            'img_size': img_sizes,
            'right': rights,
            'personid': torch.zeros(bs, dtype=torch.int32, device=device),
        }

        with torch.no_grad():
            out = model(batch)

        # Extract 3D keypoints (in camera-relative coords)
        joints_3d = out['pred_keypoints_3d'].detach().cpu().numpy()  # [B, 21, 3]

        # WiLoR outputs relative to root joint, we need camera-frame coords
        # pred_cam gives weak perspective camera [s, tx, ty]
        pred_cam = out['pred_cam'].detach().cpu().numpy()

        for n in range(bs):
            idx = batch_start + n
            joints = joints_3d[n].copy()
            # Add camera translation to get camera-frame coordinates
            s, tx, ty = pred_cam[n]
            # Weak perspective: tz = 2*focal/(img_size*s)
            focal = model_cfg.EXTRA.FOCAL_LENGTH
            tz = 2 * focal / (img_size_cfg * s)
            cam_t = np.array([tx, ty, tz])
            pred_xyz[idx] = joints + cam_t

        if (batch_start // args.batch_size) % 10 == 0:
            print(f"  {batch_end}/{n_samples} processed...")

    elapsed = time.perf_counter() - t0
    print(f"  Inference: {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/s)")

    # ── Compute metrics ──
    print("\n" + "=" * 60)
    print("  FreiHAND Hand Pose Evaluation Results")
    print("=" * 60)

    # Root-relative comparison (standard for hand pose)
    # Subtract wrist (joint 0) from both
    gt_rel = gt_xyz - gt_xyz[:, 0:1, :]      # [N, 21, 3]
    pred_rel = pred_xyz - pred_xyz[:, 0:1, :]  # [N, 21, 3]

    # Convert to mm
    gt_rel_mm = gt_rel * 1000.0
    pred_rel_mm = pred_rel * 1000.0

    # MPJPE (root-relative)
    per_joint_errors = np.linalg.norm(gt_rel_mm - pred_rel_mm, axis=2)  # [N, 21]
    mpjpe = per_joint_errors.mean()
    print(f"  MPJPE (root-relative):    {mpjpe:.1f} mm")

    # PA-MPJPE (Procrustes-aligned)
    pa_errors = np.zeros(n_samples)
    for i in range(n_samples):
        aligned = procrustes_align(pred_rel_mm[i], gt_rel_mm[i])
        pa_errors[i] = np.linalg.norm(gt_rel_mm[i] - aligned, axis=1).mean()
    pa_mpjpe = pa_errors.mean()
    print(f"  PA-MPJPE (aligned):       {pa_mpjpe:.1f} mm")

    # AUC (20-50mm)
    all_joint_errors = per_joint_errors.flatten()
    auc = compute_auc(all_joint_errors)
    print(f"  AUC (20-50mm):            {auc:.4f}")

    # Per-joint breakdown
    joint_names = [
        "Wrist", "Thumb1", "Thumb2", "Thumb3", "Thumb4",
        "Index1", "Index2", "Index3", "Index4",
        "Middle1", "Middle2", "Middle3", "Middle4",
        "Ring1", "Ring2", "Ring3", "Ring4",
        "Pinky1", "Pinky2", "Pinky3", "Pinky4",
    ]
    print("\n  Per-joint MPJPE:")
    joint_mpjpe = per_joint_errors.mean(axis=0)
    for j, (name, err) in enumerate(zip(joint_names, joint_mpjpe)):
        print(f"    {name:10s}: {err:6.1f} mm")

    print(f"\n  Reference values (FreiHAND leaderboard, top methods):")
    print(f"    PA-MPJPE: 6-8 mm")
    print(f"    AUC: 0.78-0.82")
    print("=" * 60)


if __name__ == "__main__":
    main()
