"""Fused Triton kernels for ViT-H backbone acceleration.

Replaces elementwise kernel chains with fused operations:
1. fused_swiglu: chunk + silu + mul → 1 kernel (was 3 per block × 32 blocks)
2. fused_residual_layerscale: x + gamma * y → 1 kernel (was 2 per block × 2 × 32 blocks)
3. fused_residual_layerscale_layernorm: x + gamma * y + layernorm → 1 kernel (was 4 per block)

Target: ~30% of backbone time is elementwise ops; torch.compile fuses these for 33% speedup
but costs 41s warmup. These kernels give the fusion benefit with zero warmup.
"""

import torch
import triton
import triton.language as tl


# ─── Fused SwiGLU: silu(x1) * x2 from concatenated [x1|x2] input ───
# For ViT-H: N=3416 → use BLOCK_N=4096 to process entire row in one tile

@triton.jit
def _swiglu_fwd_kernel(
    X_ptr,       # [M, 2*N] input (output of w12 linear)
    Out_ptr,     # [M, N] output
    N,           # hidden dim (half of last dim of X)
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_off = row * stride_xm
    x1 = tl.load(X_ptr + x_off + cols, mask=mask).to(tl.float32)
    x2 = tl.load(X_ptr + x_off + N + cols, mask=mask).to(tl.float32)

    result = x1 * tl.sigmoid(x1) * x2
    tl.store(Out_ptr + row * stride_om + cols, result.to(tl.float16), mask=mask)


def fused_swiglu(x12: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation: replaces x12.chunk(2) + F.silu(x1) * x2."""
    orig_shape = x12.shape
    M = x12.numel() // x12.shape[-1]
    two_N = x12.shape[-1]
    N = two_N // 2

    x12_flat = x12.reshape(M, two_N)
    out = torch.empty(M, N, dtype=x12.dtype, device=x12.device)

    # Pick smallest power-of-2 >= N for the tile
    BLOCK_N = triton.next_power_of_2(N)

    _swiglu_fwd_kernel[(M,)](
        x12_flat, out, N,
        x12_flat.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N,
    )

    out_shape = list(orig_shape)
    out_shape[-1] = N
    return out.reshape(out_shape)


# ─── Fused residual + LayerScale: x + gamma * y ───

@triton.jit
def _residual_layerscale_kernel(
    X_ptr,       # [M, D] residual
    Y_ptr,       # [M, D] sublayer output
    Gamma_ptr,   # [D] layerscale parameter
    Out_ptr,     # [M, D] output
    D,
    stride_m,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * stride_m

    x = tl.load(X_ptr + off + cols, mask=mask).to(tl.float32)
    y = tl.load(Y_ptr + off + cols, mask=mask).to(tl.float32)
    g = tl.load(Gamma_ptr + cols, mask=mask).to(tl.float32)

    result = x + g * y
    tl.store(Out_ptr + off + cols, result.to(tl.float16), mask=mask)


def fused_residual_layerscale(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Fused x + gamma * y. Replaces LayerScale + residual add."""
    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    D = x.shape[-1]

    x_flat = x.reshape(M, D)
    y_flat = y.reshape(M, D)
    out = torch.empty_like(x_flat)

    BLOCK_D = triton.next_power_of_2(D)

    _residual_layerscale_kernel[(M,)](
        x_flat, y_flat, gamma, out, D,
        stride_m=x_flat.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out.reshape(orig_shape)


# ─── Fused residual + LayerScale + LayerNorm (single-tile, 2-pass) ───
# For D=1280, BLOCK_D=2048 fits entire row in one tile.
# Pass 1: compute residual, accumulate mean+var in registers
# Pass 2: normalize and store

@triton.jit
def _residual_layerscale_layernorm_kernel(
    X_ptr,        # [M, D] residual input
    Y_ptr,        # [M, D] sublayer output
    Gamma_ptr,    # [D] layerscale parameter
    LN_W_ptr,     # [D] layernorm weight
    LN_B_ptr,     # [D] layernorm bias
    Res_ptr,      # [M, D] updated residual output (x + gamma * y)
    Norm_ptr,     # [M, D] layernorm output
    D,
    eps,
    stride_m,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * stride_m

    # Load and compute residual
    x = tl.load(X_ptr + off + cols, mask=mask).to(tl.float32)
    y = tl.load(Y_ptr + off + cols, mask=mask).to(tl.float32)
    g = tl.load(Gamma_ptr + cols, mask=mask).to(tl.float32)
    res = x + g * y

    # Store residual (needed for next block's residual stream)
    tl.store(Res_ptr + off + cols, res.to(tl.float16), mask=mask)

    # Compute mean and variance in single pass (Welford online)
    mean = tl.sum(tl.where(mask, res, 0.0)) / D
    diff = tl.where(mask, res - mean, 0.0)
    var = tl.sum(diff * diff) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize with affine transform
    w = tl.load(LN_W_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(LN_B_ptr + cols, mask=mask).to(tl.float32)
    normed = (res - mean) * rstd * w + b
    tl.store(Norm_ptr + off + cols, normed.to(tl.float16), mask=mask)


def fused_residual_layerscale_layernorm(
    x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor,
    ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual + layerscale + layernorm.

    Computes:
        res = x + gamma * y
        norm = LayerNorm(res)
    Returns (res, norm).
    """
    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    D = x.shape[-1]

    x_flat = x.reshape(M, D)
    y_flat = y.reshape(M, D)
    res_out = torch.empty_like(x_flat)
    norm_out = torch.empty_like(x_flat)

    BLOCK_D = triton.next_power_of_2(D)

    _residual_layerscale_layernorm_kernel[(M,)](
        x_flat, y_flat, gamma, ln_weight, ln_bias,
        res_out, norm_out, D, eps,
        stride_m=x_flat.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return res_out.reshape(orig_shape), norm_out.reshape(orig_shape)


# ─── Monkey-patch functions ───

class FusedSwiGLUFFN_v2(torch.nn.Module):
    """Drop-in replacement for DINOv2 SwiGLUFFN (combined w12 linear)."""

    def __init__(self, original_mlp):
        super().__init__()
        self.w12 = original_mlp.w12
        self.w3 = original_mlp.w3

    def forward(self, x):
        x12 = self.w12(x)
        hidden = fused_swiglu(x12)
        return self.w3(hidden)


class FusedSwiGLUFFN_v3(torch.nn.Module):
    """Drop-in replacement for DINOv3 SwiGLUFFN (separate w1, w2 linears).

    Concatenates w1(x) and w2(x) into [w1|w2] so the fused_swiglu kernel
    can process both halves in a single pass.
    """

    def __init__(self, original_mlp):
        super().__init__()
        self.w1 = original_mlp.w1
        self.w2 = original_mlp.w2
        self.w3 = original_mlp.w3

    def forward(self, x):
        x12 = torch.cat([self.w1(x), self.w2(x)], dim=-1)
        hidden = fused_swiglu(x12)
        return self.w3(hidden)


def _make_fused_block_chain_forward(blocks, has_rope=False):
    """Create a fused forward for a chain of ViT blocks (DINOv2 or DINOv3).

    Fuses block[i]'s FFN residual+layerscale with block[i+1]'s norm1,
    eliminating standalone LayerNorm kernels.
    """
    block_data = []
    for blk in blocks:
        eps1 = blk.norm1.eps if hasattr(blk.norm1, 'eps') else 1e-6
        eps2 = blk.norm2.eps if hasattr(blk.norm2, 'eps') else 1e-6
        block_data.append({
            'norm1': blk.norm1,
            'attn': blk.attn,
            'ls1_gamma': blk.ls1.gamma if hasattr(blk.ls1, 'gamma') else None,
            'norm2': blk.norm2,
            'mlp': blk.mlp,
            'ls2_gamma': blk.ls2.gamma if hasattr(blk.ls2, 'gamma') else None,
            'eps1': eps1,
            'eps2': eps2,
        })

    def chain_forward(x, rope=None):
        norm1_out = block_data[0]['norm1'](x)

        for i, bd in enumerate(block_data):
            # Attention (pass rope if DINOv3)
            if has_rope:
                attn_out = bd['attn'](norm1_out, rope=rope)
            else:
                attn_out = bd['attn'](norm1_out)

            # Fused: x + gamma1 * attn_out + LayerNorm → norm2
            if bd['ls1_gamma'] is not None:
                x, norm2_out = fused_residual_layerscale_layernorm(
                    x, attn_out, bd['ls1_gamma'],
                    bd['norm2'].weight, bd['norm2'].bias, bd['eps2'],
                )
            else:
                x = x + attn_out
                norm2_out = bd['norm2'](x)

            # FFN
            ffn_out = bd['mlp'](norm2_out)

            # Fused: x + gamma2 * ffn_out + LayerNorm → next norm1
            if i < len(block_data) - 1:
                next_bd = block_data[i + 1]
                if bd['ls2_gamma'] is not None:
                    x, norm1_out = fused_residual_layerscale_layernorm(
                        x, ffn_out, bd['ls2_gamma'],
                        next_bd['norm1'].weight, next_bd['norm1'].bias, next_bd['eps1'],
                    )
                else:
                    x = x + ffn_out
                    norm1_out = next_bd['norm1'](x)
            else:
                if bd['ls2_gamma'] is not None:
                    x = fused_residual_layerscale(x, ffn_out, bd['ls2_gamma'])
                else:
                    x = x + ffn_out

        return x

    return chain_forward


def patch_backbone_with_triton(backbone):
    """Monkey-patch a DINOv2/v3 backbone to use fused Triton kernels.

    Supports both:
    - DINOv2: backbone.blocks is list of BlockChunks (each containing Blocks)
    - DINOv3: backbone.encoder.blocks is flat ModuleList of SelfAttentionBlocks

    Call this after model.backbone.half() and model.to(device).

    Returns:
        Number of blocks patched
    """
    patched = 0

    # DINOv3: Dinov3Backbone wraps DinoVisionTransformer as self.encoder
    encoder = getattr(backbone, 'encoder', backbone)

    if not hasattr(encoder, 'blocks'):
        return 0

    blocks = list(encoder.blocks)
    if not blocks:
        return 0

    # Detect version: DINOv3 blocks are direct SelfAttentionBlocks,
    # DINOv2 blocks are BlockChunks containing child Blocks
    first = blocks[0]
    is_v3 = hasattr(first, 'mlp') and hasattr(first, 'norm1')
    is_v2_chunked = not is_v3 and hasattr(first, 'children')

    if is_v3:
        # DINOv3: flat list of SelfAttentionBlock
        has_rope = getattr(encoder, 'rope_embed', None) is not None

        for blk in blocks:
            if hasattr(blk.mlp, 'w1') and hasattr(blk.mlp, 'w2'):
                blk.mlp = FusedSwiGLUFFN_v3(blk.mlp)
                patched += 1
            elif hasattr(blk.mlp, 'w12'):
                blk.mlp = FusedSwiGLUFFN_v2(blk.mlp)
                patched += 1

        # Replace block iteration with fused chain
        chain_fn = _make_fused_block_chain_forward(blocks, has_rope=has_rope)

        # Patch _get_intermediate_layers_not_chunked to use fused chain
        original_get_layers = encoder._get_intermediate_layers_not_chunked

        def _fused_get_intermediate_layers(x, n=1, _chain_fn=chain_fn, _encoder=encoder):
            x, (H, W) = _encoder.prepare_tokens_with_masks(x)
            total = len(_encoder.blocks)
            blocks_to_take = range(total - n, total) if isinstance(n, int) else n

            rope = None
            if _encoder.rope_embed is not None:
                rope = _encoder.rope_embed(H=H, W=W)

            # Run fused chain — returns final output only
            # For n=1 (last layer), this is all we need
            if blocks_to_take == range(total - 1, total):
                x = _chain_fn(x, rope=rope)
                return [x]
            else:
                # Fallback: need intermediate outputs, can't use fused chain
                output = []
                for i, blk in enumerate(_encoder.blocks):
                    x = blk(x, rope) if rope is not None else blk(x)
                    if i in blocks_to_take:
                        output.append(x)
                return output

        encoder._get_intermediate_layers_not_chunked = _fused_get_intermediate_layers

    elif is_v2_chunked:
        # DINOv2: list of BlockChunks
        for chunk in blocks:
            chunk_blocks = [b for b in chunk.children()
                           if hasattr(b, 'mlp') and hasattr(b, 'norm1')]
            if not chunk_blocks:
                continue
            for blk in chunk_blocks:
                if hasattr(blk.mlp, 'w12'):
                    blk.mlp = FusedSwiGLUFFN_v2(blk.mlp)
            chunk.forward = _make_fused_block_chain_forward(chunk_blocks)
            patched += len(chunk_blocks)

    return patched
