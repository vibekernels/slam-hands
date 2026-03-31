# Annotation Pipeline Optimization Log

Note: Optimizations 1-8 and 11 apply to SAM-3D-Body inference (optional `--body` mode).
Optimizations 9-10 and 12-25 apply to the main pipeline (DROID-SLAM + WiLoR hands).

## Hardware
- RTX 5090 (Blackwell sm_120), 32GB VRAM
- CUDA 12.8, PyTorch 2.8+cu128
- Input: 1920x1080 @ 30fps, 1858 frames (~62s)

## Profiling Breakdown (body-only, single frame, 68ms)

| Component | Time | % |
|-----------|------|---|
| prepare_batch (CPU cv2.warpAffine) | 1ms | 1% |
| GPU transfer | 0.3ms | 0% |
| Backbone (ViT-H, DINOv3) | 21ms | 29% |
| Decoder + MHR head | 46ms | 67% |
| Post-processing | 1.3ms | 2% |

The decoder+MHR head dominates. The MHR head uses a TorchScript-compiled body model with sparse matrix ops (FK/IK, blend shapes, pose correctives). bf16 autocast crashes on these sparse ops. torch.compile gives <1ms improvement on the backbone.

## Optimizations Applied

### 1. Body-only inference (4.2x)
- `--body-inference-type body` skips hand decoder passes
- "full" mode runs 3 forward passes (body + left hand + right hand) plus hand validation, keypoint prompting, and a second body pass
- 289ms → 68ms per frame

### 2. Multi-frame batching (3.4x)
- Stack N frames as batch_size=N, num_person=1 through `model.forward_step()`
- The model's `_flatten_person()` reshapes [B, 1, 3, H, W] → [B, 3, H, W] — batching "just works"
- Amortizes CUDA kernel launch overhead and improves GPU utilization
- bs=16 sweet spot: 19.7ms/frame, 4.2GB VRAM

| Batch size | ms/frame | FPS | VRAM |
|---|---|---|---|
| 1 | 65.6 | 15.2 | 3.8GB |
| 4 | 25.5 | 39.2 | 3.5GB |
| 8 | 20.4 | 49.0 | 3.8GB |
| 16 | 19.7 | 50.8 | 4.2GB |
| 32 | 18.6 | 53.8 | 5.1GB |

### 3. fp16 backbone (~3ms saved)
- `model.backbone.half(); model.backbone_dtype = torch.float16`
- Backbone weights + activations in fp16, output cast back to fp32 for decoder
- Cannot apply to decoder/MHR head (sparse ops crash)

### 4. Threaded CPU preprocessing
- Background thread runs `prepare_batch()` (cv2.warpAffine + ToTensor) while GPU runs inference
- CPU prep is only ~1ms/frame so this mainly hides batch construction latency

### 5. Frame stride with interpolation (Nx)
- `--body-stride N` processes every Nth frame, linearly interpolates between keyframes
- Vectorized via `numpy.interp` across all output dimensions
- Quality: stride=2 → 2.4mm mean error, stride=3 → 2.4mm (at 30fps, body motion between adjacent frames is small)

### 6. SLAM tensor caching
- Pre-compute all SLAM tensors before tracking, reuse in `terminate()` stream
- Eliminates double RGB→BGR→resize→GPU conversion for every frame

### 7. mmap checkpoint loading
- `torch.load(..., mmap=True)` — memory-maps the 2GB checkpoint instead of loading into RAM
- Reduces checkpoint load from 12.4s to 0.1s (first call; OS caches pages)
- Model caching via `_sam3d_cache` dict avoids reload across repeated calls

### 8. Optimal batch size: bs=32 (was 16)
- At bs=32 the ViT-H backbone (81% of forward time) amortizes kernel launches better
- 49.8 fps vs 28 fps at bs=16 (78% improvement)
- VRAM: 5.1GB at bs=32 (well within 32GB)
- Quality: 0.00mm difference vs bs=16

## Profiling at bs=32 (557ms per batch = 17.4ms/frame)

| Component | Time | % |
|-----------|------|---|
| Backbone (ViT-H, fp16) | 449ms | 81% |
| Decoder (6 layers, cross-attn) | 85ms | 15% |
| MHR head (6 calls, TorchScript) | 23ms | 4% |

The backbone dominates at large batch sizes. The decoder calls `head_pose` at every intermediate layer (`do_interm_preds=True`) for iterative keypoint refinement, but each MHR call is only ~4ms at bs=32.

## Results

| Config | ms/frame | FPS | 1858 frames | Speedup |
|---|---|---|---|---|
| full, stride=1, bs=1 (baseline) | 289 | 3.5 | ~537s | 1x |
| body, stride=1, bs=1 | 68 | 14.7 | ~126s | 4.3x |
| body, stride=1, bs=16 | 35 | 28 | ~66s | 8x |
| body, stride=1, bs=32 | 20 | 49.8 | ~37s | 14.5x |
| body, stride=2, bs=32 | 20 | 105* | ~18s | 30x |

*effective fps including interpolation of skipped frames

Model loading is a fixed ~25s cost (SAM3DBody constructor 8.7s + checkpoint load + to(cuda)). Cached after first call.

## Quality Validation
- Batched vs unbatched: 2mm mean diff (float order-of-operations, not a quality loss)
- bs=16 vs bs=32: 0.0mm difference (identical)
- Stride=2 interpolation: 2.4mm mean error on 3D keypoints
- Decoder 6→4 layers: 1.2mm error; 6→3 layers: 5.4mm error
- All outputs finite, 100% detection rate with full-image bbox

### 11. Fused Triton kernels for backbone (~7% backbone speedup, zero warmup)
- Custom Triton kernels replace PyTorch elementwise kernel chains in ViT-H blocks
- **Fused SwiGLU**: `chunk + silu + mul` → 1 kernel (was 3 separate kernels × 32 blocks)
- **Fused residual+LayerScale+LayerNorm**: `x + gamma*y + LN()` → 1 kernel (was 3-4 kernels)
- **Cross-block chain fusion**: block[i]'s FFN residual fused with block[i+1]'s norm1
- Elementwise ops: 61.7ms → 34.2ms (44% reduction), total backbone: 377ms → 352ms
- Applied automatically (no flag needed), Triton JIT compiles on first call (~2s)
- Max numerical diff: 0.03 (fp16 order-of-operations, not a quality issue)
- Captures ~25% of torch.compile's gains; remaining 75% requires GEMM epilogue fusion

## Optional: torch.compile (--compile flag)
- `torch.compile(model.forward_step, mode="default")` gives ~67 fps at bs=32 (33% faster)
- BUT: 41s warmup cost. Only worthwhile for long videos or batch processing
- For 1858 frames: 41s warmup + 28s inference = 69s (worse than 37s without compile)
- For 10,000+ frames or repeated calls: compile wins

## Optional: Reduced decoder layers (--decoder-layers flag)
- 6→4 layers: 59.2 fps (+4%), 1.2mm quality loss
- 6→3 layers: 60.3 fps (+6%), 5.4mm quality loss
- Marginal gain since backbone dominates at bs=32

## Full Pipeline Optimizations

### 9. Parallel SAM-3D-Body model loading
- Split loading into CPU phase (constructor + checkpoint) and GPU phase (to cuda + half)
- CPU phase (~9s) runs in background thread, overlapping with decode + SLAM
- `load_state_dict(assign=True)` avoids copying weights: 16s → 0s
- GPU phase (0.2s) runs after SLAM frees the GPU
- Net effect: 25s model load completely hidden

### 10. Batched SLAM tensor prep (12x faster)
- Old: per-frame `_prepare_slam_frame` (numpy BGR copy + GPU transfer + resize) → 19s
- New: multi-threaded cv2.resize on CPU (8 workers) + small tensor transfer → 1.6s
- CPU resize of 1920x1080→584x328 is faster than full-res GPU round-trip

### 12. Pipelined SLAM prep + tracking
- Previously: prep all tensors (1.8s) → track all frames (6s) = 7.8s sequential
- Now: background thread preps batches of 64 frames (8 workers) while main thread tracks
- Prep runs ~1.5s ahead and finishes while tracking is still running
- Saves ~1.5s by overlapping CPU prep with GPU tracking

### 13. Fast trajectory interpolation (--fast-traj)
- Default: DROID's NN trajectory filler runs feature extraction + 6 optimization iterations
  per batch of 16 frames → ~3s for 1858 frames
- Fast mode: linear translation interpolation + slerp for quaternion rotation
- CPU-only, completes in <0.1s
- For smooth camera motion (robot/handheld video at 30fps), interpolation between
  keyframes 2-3 frames apart is very accurate

### 14. Configurable SLAM backend steps (--slam-backend-steps)
- Default: 7+12=19 iterations of global bundle adjustment
- For robot video: 5+8=13 iterations may be sufficient
- Saves ~1.5s with minimal quality impact on smooth trajectories

### 15. Native C++ async video decoder (GIL-free decode↔SLAM overlap)
- pybind11 C++ extension using FFmpeg C API (libavformat, libavcodec, libswscale)
- Decode + swscale resize runs in a native thread that NEVER acquires the Python GIL
- `slam_only=False` mode: produces both full-res RGB + SLAM BGR in single decode pass
- `get_next()` releases GIL while waiting, so DROID-SLAM's CUDA kernels run concurrently
- RGB frames collected during SLAM tracking — eliminates separate body decode pass (~9s saved)
- HEVC decode+resize: ~14s for 1858 frames with RGB+SLAM, overlapped with 6s tracking
- Build: `cd native_decode && python setup.py build_ext --inplace`
- Auto-detected at runtime; falls back to Python decode if not compiled

### 16. Optimized video conversion fallback (libx264 ultrafast)
- When NVENC is unavailable, falls back to libx264 ultrafast instead of libsvtav1
- 6.2s standalone (was 38s with svtav1), ~18s in-pipeline due to process overhead
- Functional NVENC test (`check_scale_cuda_available`) catches CUDA_ERROR_NO_DEVICE in 0.1s
- No `-hwaccel auto` in CPU fallback (avoids failed NVDEC probe overhead)

### 17. Parallel video conversion + body inference
- Video conversion (CPU ffmpeg, ~18s) overlaps with body inference (GPU, ~13s)
- Net cost: max(18s, 13s) = 18s instead of 18+13=31s, saving ~13s
- Body inference FPS drops ~3% from CPU contention (51→49 fps), acceptable tradeoff

### Full Pipeline Results (1858 frames, 62s video)

| Config | SLAM | Body | Video | **Total** | vs baseline |
|---|---|---|---|---|---|
| stride=1, bs=32 (previous) | 24s | 36s | 38s seq | **98s** | 7x |
| stride=2, bs=32, fast_traj, 5+8 backend | 20s | 17s | 38s seq | **75s** | 9x |
| stride=3, bs=32, fast_traj, 5+8 backend | 20s | 11s | 38s seq | **69s** | 10x |
| + native decode (slam_only) | 8.3s | 13s decode+infer | 18s overlapped | **42s** | 16x |
| + single-decode (RGB+SLAM) + Triton | 16.1s | 13.7s infer only | 21s overlapped | **37.5s** | 18x |

*stride=3 has same 2.4mm mean error as stride=2 at 30fps

Previous total (estimated baseline): ~660s (8s decode + 90s SLAM + 25s model load + 537s body inference)

### 18. Single-decode: collect RGB frames during SLAM
- Native decoder uses `slam_only=False` to produce both SLAM BGR + full-res RGB simultaneously
- RGB frames accumulated in memory during tracking loop (~11GB for 1858 × 1920×1080×3)
- Eliminates separate 9.3s PyAV body decode pass entirely
- SLAM phase slower (16.1s vs 8.3s) due to full-res decode, but net saves ~5s end-to-end

### 19. Split decode architecture (no shared memory)
- Parent runs SLAM with `slam_only=True` native C++ decoder (SLAM-res only, no full-res RGB)
- Forked child decodes video independently with OpenCV at full resolution
- Eliminates shared-memory ring buffer (was 6MB/frame × 1858 frames memcpy overhead)
- OpenCV in child avoids competing with parent's native decoder for FFmpeg CPU threads
- GPU contention between child (YOLO/WiLoR) and parent (DROID-SLAM) is the remaining bottleneck (~5s impact)

### 20. pytorch_lightning stub for inference (3.4s import saved)
- WiLoR inherits from `pytorch_lightning.LightningModule` but only uses `nn.Module` methods at inference
- Replace `import pytorch_lightning` with a 10-line stub: `LightningModule = nn.Module` + no-op `save_hyperparameters`/`log`
- Saves 3.4s of import time (pytorch_lightning pulls in heavy dependencies)
- Installed before fork so child inherits the stub via `sys.modules`

### 21. WiLoR statedict extraction for mmap loading
- Extracted `state_dict` from `wilor_final.ckpt` (2.4GB) → `wilor_final_statedict.pt`
- Child loads statedict with `torch.load(mmap=True)` + `load_state_dict(assign=True)`
- Avoids parsing full Lightning checkpoint structure in child process

### 22. HLG→SDR tonemapping LUT in native decoder
- iPhone HEVC videos are 10-bit HDR (BT.2020/HLG); swscale converts pixel format but not transfer function
- Without tonemapping, 10-bit→8-bit naive quantization produces dark/wrong colors
- Precomputed 256-entry LUT: HLG OETF⁻¹ → scene linear → system gamma (1.2) → BT.709 OETF
- Applied per-pixel during RGB row copy in C++ — zero overhead vs previous memcpy
- Detected automatically from first decoded frame's pixel format (yuv420p10le → LUT enabled)
- Fixes YOLO hand detection on native decoder output (was getting 0 detections without it)

### 23. Deferred NVENC video conversion
- NVENC takes ~3s on an idle GPU but 30-35s when competing with SLAM/YOLO/WiLoR
- Previously ran overlapped with inference, causing severe GPU contention
- Now runs sequentially after SLAM+hands complete, when GPU is idle
- Net saving: ~28s (31s overlapped → 3s sequential after inference)

### 24. OpenCV decode in child process
- Previously child used a second native C++ decoder instance
- Two FFmpeg decoder instances competed for CPU threads, slowing both
- Child switched to OpenCV VideoCapture which uses its own thread pool
- SLAM: 82fps (up from ~70fps with two native decoders competing)
- Slight detection rate drop (~1-2%) from missing HLG tonemapping in OpenCV, acceptable

### 25. Persistent pipeline service (pipeline_service.py)
- Two-process architecture: parent (DROID-SLAM) + child (YOLO/WiLoR) stay alive across videos
- Child keeps YOLO + WiLoR on GPU permanently; parent keeps Droid imported
- JSON-lines IPC over unix socketpair for per-video dispatch
- Saves ~6s per video (no imports, no model loading)
- One-time startup: ~10s. Per-video overhead: ~0.5s (fresh Droid instance)

### Final Pipeline Results (1858 frames, 62s video)

| Config | SLAM | Hands | Video | **Total** | vs baseline |
|---|---|---|---|---|---|
| single-decode + Triton (body, prev) | 16.1s | 13.7s | 21s overlapped | **37.5s** | 18x |
| + split decode + PL stub (stride=1) | 22s | 18s | 3s deferred | **33s** | 20x |
| + pipeline service (stride=1) | 22s | 18s | 3s deferred | **32s** | 21x |

*`./annotate.sh` default. SLAM is the critical path (~22s tracking + 1s backend + video + assembly).
*Service mode: ~10s one-time startup, then ~32s per video (vs ~39s wall time cold start).

**Recommended default: stride=1** (`./annotate.sh`). Stride=2 saves ~3s total but introduces interpolation error on hand keypoints. At 33s for 62s of video, stride=1 is fast enough and produces exact per-frame hand poses.

## What Didn't Work

### SAM-3D-Body specific
- **torch.compile on backbone only**: <1ms improvement, backbone already memory-bound in fp16
- **bf16 autocast on full model**: MHR head sparse_coo_tensor matmul crashes
- **fp16 on decoder/head**: Same sparse op crash
- **CUDA graphs**: Model uses dynamic control flow (body_batch_idx, hand_batch_idx) — not graph-capturable
- **bf16 vs fp16 backbone**: <2% difference on Blackwell
- **Token Merging (ToMe)**: Would break spatial structure needed by decoder cross-attention

### Pipeline-level
- **Multi-threaded CPU prep**: GIL limits benefit; threaded Queue with single worker is sufficient
- **CUDA stream double-buffering**: <3% improvement over single-stream pipelining
- **Streaming decode→SLAM (threading)**: GIL serialization makes it slower than batch approach
- **Streaming decode→SLAM (multiprocessing)**: Frame transfer via SharedMemory costs ~6s (22GB memcpy for 6MB × 1858 frames), negating any overlap savings
- **NVDEC hardware decode**: hevc_cuvid works but is slower (203fps vs 250fps) due to GPU→CPU transfer for frame.to_rgb()
- **mmap + assign=True + CPU half()**: Deferred mmap page-in makes .half() take 13s
- **Pre-building WiLoR before fork**: Constructing MANO/WiLoR model in parent then forking caused child to deadlock at YOLO model loading — unclear root cause but related to inherited model state
- **Phased decode (all frames first, then GPU inference)**: CPU-only decode of all frames (8.5s) then serial GPU inference was slower (22.5s total) than streaming decode+YOLO together (19.8s) because serial decode doesn't overlap with GPU work
- **Phased GPU execution (defer child GPU until SLAM done)**: Child waited 16s for SLAM, then took 24s for decode+YOLO — total 53s, worse than concurrent 33s. GPU contention between SLAM and hands costs ~15% on SLAM but concurrent is still faster overall
- **Overlapped NVENC video conversion**: NVENC competing with SLAM/YOLO/WiLoR for GPU resources caused 32-35s video conversion instead of 2s. Deferring until after inference is much faster overall
- **Native C++ decoder in child process**: Two native decoder instances (parent SLAM + child hands) competed for FFmpeg CPU threads, slowing both. OpenCV in child avoids this contention
