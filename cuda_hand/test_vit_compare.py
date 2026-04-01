#!/usr/bin/env python3
"""Compare CUDA vs Python WiLoR ViT backbone output."""

import struct, subprocess, sys, os
import numpy as np
import cv2
import torch

sys.path.insert(0, '/workspace/WiLoR')
sys.path.insert(0, '/workspace/robot-video')

_orig_load = torch.load
def _patched_load(*a, **kw):
    if 'weights_only' not in kw:
        kw['weights_only'] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ultralytics import YOLO
from wilor.datasets.utils import expand_to_aspect_ratio, gen_trans_from_patch_cv

# Get frame
cap = cv2.VideoCapture('/workspace/IMG_1466.mov')
ret, frame = cap.read()
cap.release()
H, W = frame.shape[:2]

# Run YOLO to get right hand detection
model = YOLO('/workspace/WiLoR/pretrained_models/detector.pt')
results = model(frame, conf=0.3, imgsz=512, verbose=False)
for r in results:
    for i in range(len(r.boxes)):
        cls = int(r.boxes[i].cls[0])
        if cls == 1:  # right hand
            bbox = r.boxes[i].xyxy[0].cpu().numpy()
            break
    break

x1, y1, x2, y2 = bbox
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
bw = (x2 - x1) * 2.0
bh = (y2 - y1) * 2.0
scale = np.array([bw, bh]) / 200.0
expanded = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=[192, 256])
bbox_size = expanded.max()
print(f"Right hand: center=({cx:.1f},{cy:.1f}), bbox_size={bbox_size:.1f}")

# Create Python crop (matching what pipeline does)
trans = gen_trans_from_patch_cv(cx, cy, bbox_size, bbox_size, 256, 256, 1.0, 0)
img_patch = cv2.warpAffine(frame, trans, (256, 256), flags=cv2.INTER_LINEAR)
rgb_patch = img_patch[:,:,::-1].copy()

mean = 255.0 * np.array([0.485, 0.456, 0.406])
std = 255.0 * np.array([0.229, 0.224, 0.225])
normalized = (rgb_patch.astype(np.float32).transpose(2, 0, 1) - mean[:, None, None]) / std[:, None, None]

# Feed through PyTorch WiLoR backbone
from wilor.models.wilor import WiLoRHandPoseEstimation
import yaml

cfg_path = '/workspace/WiLoR/pretrained_models/model_config.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Build model
from types import SimpleNamespace
model_cfg = SimpleNamespace(**cfg['MODEL'])
wilor_model = WiLoRHandPoseEstimation(model_cfg)
ckpt = torch.load('/workspace/WiLoR/pretrained_models/wilor_final.ckpt', map_location='cpu', weights_only=False)
state_dict = {k.replace('model.', '', 1): v for k, v in ckpt['state_dict'].items()}
wilor_model.load_state_dict(state_dict, strict=False)
wilor_model.eval().cuda().half()

# Run backbone
img_tensor = torch.from_numpy(normalized[np.newaxis]).cuda().half()
print(f"Input tensor: {img_tensor.shape}, mean={img_tensor.float().mean():.4f}")

with torch.no_grad():
    # Match what the pipeline does
    x = img_tensor[:,:,:,32:-32]  # crop to 256x192
    print(f"ViT input after crop: {x.shape}")

    # Run backbone
    temp_mano_params, pred_cam, pred_mano_feats, vit_out = wilor_model.backbone(x)

    # Extract token predictions
    print(f"\nPyTorch backbone outputs:")
    print(f"  temp_mano_params: {temp_mano_params.shape}, mean={temp_mano_params.float().mean():.6f}")
    print(f"  pred_cam: {pred_cam.shape}, values={pred_cam.float().cpu().numpy()}")
    print(f"  pred_mano_feats: {pred_mano_feats.shape}, mean={pred_mano_feats.float().mean():.6f}")
    print(f"  vit_out: {vit_out.shape if vit_out is not None else None}")

    # Show first few pose params
    pose = temp_mano_params[:, :96].float().cpu().numpy()
    shape = temp_mano_params[:, 96:106].float().cpu().numpy()
    print(f"  Pose params[:10]: {pose[0,:10]}")
    print(f"  Shape params: {shape[0]}")
    print(f"  Camera: {pred_cam.float().cpu().numpy()[0]}")

# Now get CUDA output for comparison
print("\n=== CUDA output ===")
data = struct.pack('III', W, H, 0) + frame.tobytes()
proc = subprocess.Popen(
    ['./cuda_hand', '--weights-dir', 'data/weights', '--det-conf', '0.3',
     '--output', '/tmp/cuda_hand_test.bin'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd='/workspace/robot-video/cuda_hand'
)
stdout, stderr = proc.communicate(data, timeout=120)

with open('/tmp/cuda_hand_test.bin', 'rb') as f:
    nf = struct.unpack('i', f.read(4))[0]
    frame_idx = struct.unpack('i', f.read(4))[0]
    left_det = struct.unpack('B', f.read(1))[0]
    right_det = struct.unpack('B', f.read(1))[0]
    left_kp3d = np.frombuffer(f.read(63*4), dtype=np.float32).reshape(21, 3)
    right_kp3d = np.frombuffer(f.read(63*4), dtype=np.float32).reshape(21, 3)
    left_kp2d = np.frombuffer(f.read(42*4), dtype=np.float32).reshape(21, 2)
    right_kp2d = np.frombuffer(f.read(42*4), dtype=np.float32).reshape(21, 2)

print(f"CUDA right kp3d[0]: {right_kp3d[0]}")
print(f"CUDA right kp2d[0]: {right_kp2d[0]}")

# Also compare with PyTorch MANO output
# Run full MANO forward with PyTorch
from wilor.models.mano_wrapper import MANODecoder
mano = MANODecoder(is_rhand=True).cuda()

with torch.no_grad():
    pose_rot6d = temp_mano_params[:, :96]
    betas = temp_mano_params[:, 96:106]

    # Convert rot6d to rotmat
    from wilor.utils.rotation_conversions import rotation_6d_to_matrix
    pose_rot6d_reshaped = pose_rot6d.reshape(-1, 16, 6)
    rotmat = rotation_6d_to_matrix(pose_rot6d_reshaped.reshape(-1, 6)).reshape(-1, 16, 3, 3)

    mano_out = mano(global_orient=rotmat[:, 0:1], hand_pose=rotmat[:, 1:],
                    betas=betas, transl=torch.zeros(1, 3).cuda().half())
    verts = mano_out.vertices.float().cpu().numpy()[0]
    joints = mano_out.joints.float().cpu().numpy()[0]
    print(f"\nPyTorch MANO joints[0] (wrist): {joints[0]}")

    # Camera projection
    cam = pred_cam.float().cpu().numpy()[0]
    print(f"Camera params: {cam}")

    # Perspective projection
    focal_length = 5000.0
    cam_t = torch.tensor([[cam[1], cam[2], 2*focal_length/(cam[0]*bbox_size + 1e-9)]]).cuda()
    print(f"cam_t: {cam_t.cpu().numpy()}")

    joints_t = torch.from_numpy(joints).cuda().unsqueeze(0) + cam_t.unsqueeze(1)
    proj = joints_t[:, :, :2] / joints_t[:, :, 2:3] * focal_length
    proj = proj.cpu().numpy()[0]

    # Transform back to image coords
    # proj is in crop coords (256x256 crop centered on cx,cy)
    # Need to apply inverse of the crop transform
    proj_img = proj * bbox_size / 256.0 + np.array([cx, cy])

    print(f"\nPyTorch projected wrist (image coords): {proj_img[0]}")
    print(f"CUDA wrist (image coords): {right_kp2d[0]}")
    print(f"Wrist diff: {np.linalg.norm(proj_img[0] - right_kp2d[0]):.1f} px")
