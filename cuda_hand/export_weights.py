#!/usr/bin/env python3
"""Export YOLO + WiLoR + MANO weights to binary files for the CUDA hand implementation.

Fuses BatchNorm into Conv/ConvTranspose layers at export time. Exports all tensors
in the same format as cuda_slam: raw float32 with [ndim, shape...] header.
"""

import os
import sys
import struct
import numpy as np
import torch


def save_tensor(path, tensor):
    """Save a tensor as raw float32 binary with a small header."""
    if isinstance(tensor, np.ndarray):
        t = tensor.astype(np.float32)
    else:
        t = tensor.detach().cpu().float().contiguous().numpy()
    with open(path, 'wb') as f:
        f.write(struct.pack('i', t.ndim))
        for d in t.shape:
            f.write(struct.pack('i', d))
        f.write(t.tobytes())


def save_int_tensor(path, tensor):
    """Save an integer tensor as raw int32 binary with header."""
    if isinstance(tensor, np.ndarray):
        t = tensor.astype(np.int32)
    else:
        t = tensor.detach().cpu().int().contiguous().numpy()
    with open(path, 'wb') as f:
        f.write(struct.pack('i', t.ndim))
        for d in t.shape:
            f.write(struct.pack('i', d))
        f.write(t.tobytes())


def fuse_conv_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=0.001):
    """Fuse BatchNorm into Conv weight and bias."""
    # BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    # Fused conv: W_new = gamma / sqrt(var+eps) * W, b_new = gamma * (-mean) / sqrt(var+eps) + beta
    inv_std = bn_weight / torch.sqrt(bn_var + eps)
    # For Conv2d: weight is [Co, Ci, kH, kW]
    w = conv_weight * inv_std.view(-1, 1, 1, 1)
    b = bn_bias - bn_mean * inv_std
    return w, b


def fuse_deconv_bn(deconv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=0.001):
    """Fuse BatchNorm into ConvTranspose2d weight and bias.
    ConvTranspose2d weight shape: [Ci, Co, kH, kW] (note: input channels first!)
    """
    inv_std = bn_weight / torch.sqrt(bn_var + eps)
    # Output channel is dim 1 for ConvTranspose2d
    w = deconv_weight * inv_std.view(1, -1, 1, 1)
    b = bn_bias - bn_mean * inv_std
    return w, b


def export_yolo(detector_path, out_dir):
    """Export YOLO detector weights with BN fused."""
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(detector_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().eval()

    manifest = []
    idx = 0

    def save(name, tensor):
        nonlocal idx
        fname = f'yolo_{name}.bin'
        save_tensor(os.path.join(out_dir, fname), tensor)
        shape_str = ','.join(str(d) for d in tensor.shape)
        manifest.append((name, fname, shape_str))
        print(f'  yolo_{name:50s} {str(list(tensor.shape)):30s}')
        idx += 1

    def export_conv_bn(prefix, conv_module, bn_module=None):
        """Export a Conv+BN+Act block as fused weight+bias."""
        w = conv_module.weight.data
        if bn_module is not None:
            w, b = fuse_conv_bn(w, bn_module.weight.data, bn_module.bias.data,
                                bn_module.running_mean, bn_module.running_var,
                                bn_module.eps)
        else:
            b = conv_module.bias.data if conv_module.bias is not None else torch.zeros(w.shape[0])
        save(f'{prefix}_weight', w)
        save(f'{prefix}_bias', b)

    # Export backbone + neck layers (0-21)
    for layer_idx, layer in enumerate(model.model):
        name = layer.__class__.__name__
        if name == 'Conv':
            export_conv_bn(f'l{layer_idx}', layer.conv, layer.bn)
        elif name == 'C2f':
            export_conv_bn(f'l{layer_idx}_cv1', layer.cv1.conv, layer.cv1.bn)
            export_conv_bn(f'l{layer_idx}_cv2', layer.cv2.conv, layer.cv2.bn)
            for bi, bottleneck in enumerate(layer.m):
                export_conv_bn(f'l{layer_idx}_m{bi}_cv1', bottleneck.cv1.conv, bottleneck.cv1.bn)
                export_conv_bn(f'l{layer_idx}_m{bi}_cv2', bottleneck.cv2.conv, bottleneck.cv2.bn)
        elif name == 'SPPF':
            export_conv_bn(f'l{layer_idx}_cv1', layer.cv1.conv, layer.cv1.bn)
            export_conv_bn(f'l{layer_idx}_cv2', layer.cv2.conv, layer.cv2.bn)
        elif name == 'Pose':
            # Detection head
            for si in range(layer.nl):
                # cv2: bbox regression
                for ci, conv in enumerate(layer.cv2[si]):
                    if hasattr(conv, 'conv'):  # Conv+BN+Act
                        export_conv_bn(f'head_cv2_s{si}_c{ci}', conv.conv, conv.bn)
                    elif hasattr(conv, 'weight'):  # plain Conv2d
                        save(f'head_cv2_s{si}_c{ci}_weight', conv.weight.data)
                        save(f'head_cv2_s{si}_c{ci}_bias', conv.bias.data)
                # cv3: class prediction
                for ci, conv in enumerate(layer.cv3[si]):
                    if hasattr(conv, 'conv'):
                        export_conv_bn(f'head_cv3_s{si}_c{ci}', conv.conv, conv.bn)
                    elif hasattr(conv, 'weight'):
                        save(f'head_cv3_s{si}_c{ci}_weight', conv.weight.data)
                        save(f'head_cv3_s{si}_c{ci}_bias', conv.bias.data)
                # cv4: keypoint prediction
                for ci, conv in enumerate(layer.cv4[si]):
                    if hasattr(conv, 'conv'):
                        export_conv_bn(f'head_cv4_s{si}_c{ci}', conv.conv, conv.bn)
                    elif hasattr(conv, 'weight'):
                        save(f'head_cv4_s{si}_c{ci}_weight', conv.weight.data)
                        save(f'head_cv4_s{si}_c{ci}_bias', conv.bias.data)
            # DFL conv
            save('dfl_weight', layer.dfl.conv.weight.data)
            if layer.dfl.conv.bias is not None:
                save('dfl_bias', layer.dfl.conv.bias.data)
            else:
                save('dfl_bias', torch.zeros(1))
            # Strides and anchors
            save('strides', layer.stride.float())

    # Save architecture info
    head = model.model[-1]
    info = {
        'nc': head.nc,
        'nl': head.nl,
        'reg_max': head.reg_max,
        'kpt_shape': head.kpt_shape,
        'no': head.no,
        'nk': head.nk,
    }
    import json
    with open(os.path.join(out_dir, 'yolo_info.json'), 'w') as f:
        json.dump(info, f)

    print(f'  Exported {idx} YOLO tensors')
    return manifest


def export_wilor(ckpt_path, out_dir):
    """Export WiLoR backbone + RefineNet weights."""
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt['state_dict']

    manifest = []
    idx = 0

    # Separate RefineNet deconv layers that need BN fusion
    # refine_net.deconv.first_conv: Conv2d (no BN - bnrelu_final=False)
    # refine_net.deconv.deconv.0: [ConvTranspose2d, BN, ReLU]
    # refine_net.deconv.deconv.1: [ConvTranspose2d, BN, ReLU, ConvTranspose2d, BN, ReLU]

    def save_kv(name, tensor):
        nonlocal idx
        fname = f'wilor_{name}.bin'
        save_tensor(os.path.join(out_dir, fname), tensor)
        shape_str = ','.join(str(d) for d in tensor.shape)
        manifest.append((name, fname, shape_str))
        idx += 1

    # Fuse RefineNet deconv BatchNorm layers
    fused_keys = set()

    # Branch 0: deconv.0 = [ConvTranspose2d(640->320, k4s2p1), BN(320), ReLU]
    w0 = sd['refine_net.deconv.deconv.0.0.weight']
    bn0_w = sd['refine_net.deconv.deconv.0.1.weight']
    bn0_b = sd['refine_net.deconv.deconv.0.1.bias']
    bn0_m = sd['refine_net.deconv.deconv.0.1.running_mean']
    bn0_v = sd['refine_net.deconv.deconv.0.1.running_var']
    w0_fused, b0_fused = fuse_deconv_bn(w0, bn0_w, bn0_b, bn0_m, bn0_v)
    save_kv('refine_net_deconv_branch0_0_weight', w0_fused)
    save_kv('refine_net_deconv_branch0_0_bias', b0_fused)
    for k in ['refine_net.deconv.deconv.0.0.weight',
              'refine_net.deconv.deconv.0.1.weight', 'refine_net.deconv.deconv.0.1.bias',
              'refine_net.deconv.deconv.0.1.running_mean', 'refine_net.deconv.deconv.0.1.running_var',
              'refine_net.deconv.deconv.0.1.num_batches_tracked']:
        fused_keys.add(k)

    # Branch 1, layer 0: ConvTranspose2d(640->320) + BN(320)
    w1a = sd['refine_net.deconv.deconv.1.0.weight']
    bn1a_w = sd['refine_net.deconv.deconv.1.1.weight']
    bn1a_b = sd['refine_net.deconv.deconv.1.1.bias']
    bn1a_m = sd['refine_net.deconv.deconv.1.1.running_mean']
    bn1a_v = sd['refine_net.deconv.deconv.1.1.running_var']
    w1a_f, b1a_f = fuse_deconv_bn(w1a, bn1a_w, bn1a_b, bn1a_m, bn1a_v)
    save_kv('refine_net_deconv_branch1_0_weight', w1a_f)
    save_kv('refine_net_deconv_branch1_0_bias', b1a_f)
    for k in ['refine_net.deconv.deconv.1.0.weight',
              'refine_net.deconv.deconv.1.1.weight', 'refine_net.deconv.deconv.1.1.bias',
              'refine_net.deconv.deconv.1.1.running_mean', 'refine_net.deconv.deconv.1.1.running_var',
              'refine_net.deconv.deconv.1.1.num_batches_tracked']:
        fused_keys.add(k)

    # Branch 1, layer 1: ConvTranspose2d(320->160) + BN(160)
    w1b = sd['refine_net.deconv.deconv.1.3.weight']
    bn1b_w = sd['refine_net.deconv.deconv.1.4.weight']
    bn1b_b = sd['refine_net.deconv.deconv.1.4.bias']
    bn1b_m = sd['refine_net.deconv.deconv.1.4.running_mean']
    bn1b_v = sd['refine_net.deconv.deconv.1.4.running_var']
    w1b_f, b1b_f = fuse_deconv_bn(w1b, bn1b_w, bn1b_b, bn1b_m, bn1b_v)
    save_kv('refine_net_deconv_branch1_1_weight', w1b_f)
    save_kv('refine_net_deconv_branch1_1_bias', b1b_f)
    for k in ['refine_net.deconv.deconv.1.3.weight',
              'refine_net.deconv.deconv.1.4.weight', 'refine_net.deconv.deconv.1.4.bias',
              'refine_net.deconv.deconv.1.4.running_mean', 'refine_net.deconv.deconv.1.4.running_var',
              'refine_net.deconv.deconv.1.4.num_batches_tracked']:
        fused_keys.add(k)

    # Export all other keys as-is
    skip_prefixes = ['mano.faces_tensor', 'mano.hand_components', 'mano.hand_mean',
                     'mano.pose_mean', 'mano.vertex_joint_selector']
    for key in sorted(sd.keys()):
        if key in fused_keys:
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        if 'num_batches_tracked' in key:
            continue
        name = key.replace('.', '_')
        t = sd[key]
        if t.dtype in (torch.int64, torch.long):
            fname = f'wilor_{name}.bin'
            save_int_tensor(os.path.join(out_dir, fname), t)
            shape_str = ','.join(str(d) for d in t.shape)
            manifest.append((name, fname, shape_str))
            idx += 1
        else:
            save_kv(name, t)
        print(f'  wilor_{name:55s} {str(list(t.shape)):30s}')

    print(f'  Exported {idx} WiLoR tensors')
    return manifest


def export_mano_mean_params(mean_params_path, out_dir):
    """Export MANO mean parameters."""
    os.makedirs(out_dir, exist_ok=True)
    mean = np.load(mean_params_path)
    manifest = []
    for key in ['pose', 'shape', 'cam']:
        fname = f'mano_mean_{key}.bin'
        save_tensor(os.path.join(out_dir, fname), mean[key])
        shape_str = ','.join(str(d) for d in mean[key].shape)
        manifest.append((f'mano_mean_{key}', fname, shape_str))
        print(f'  mano_mean_{key}: {mean[key].shape}')
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export YOLO + WiLoR + MANO weights')
    parser.add_argument('--wilor-dir', default='/workspace/WiLoR')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'data', 'weights'))
    args = parser.parse_args()

    wilor_dir = args.wilor_dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    all_manifest = []

    print("=== Exporting YOLO detector ===")
    m = export_yolo(os.path.join(wilor_dir, 'pretrained_models', 'detector.pt'), out_dir)
    all_manifest.extend(m)

    print("\n=== Exporting WiLoR model ===")
    m = export_wilor(os.path.join(wilor_dir, 'pretrained_models', 'wilor_final.ckpt'), out_dir)
    all_manifest.extend(m)

    print("\n=== Exporting MANO mean params ===")
    m = export_mano_mean_params(os.path.join(wilor_dir, 'mano_data', 'mano_mean_params.npz'), out_dir)
    all_manifest.extend(m)

    # Write manifest
    with open(os.path.join(out_dir, 'manifest.txt'), 'w') as f:
        for name, fname, shape_str in all_manifest:
            f.write(f'{name}\t{fname}\t{shape_str}\n')

    print(f'\nExported {len(all_manifest)} total tensors to {out_dir}/')


if __name__ == '__main__':
    main()
