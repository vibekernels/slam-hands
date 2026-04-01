#!/usr/bin/env python3
"""Benchmark the full pipeline in service mode."""
import time, os, sys, shutil

sys.path.insert(0, os.path.dirname(__file__))

output_dir = "/tmp/benchmark_output"

print("=" * 60)
print("LOADING SERVICE (models warm-up)...")
print("=" * 60)
t_load = time.perf_counter()

from pipeline_service import PipelineService
service = PipelineService(
    droid_weights="/workspace/DROID-SLAM/checkpoints/droid.pth",
    wilor_dir="/workspace/WiLoR",
)

t_load = time.perf_counter() - t_load
print(f"\nService loaded in {t_load:.1f}s")

# Run 1: with audio (broken, so near-zero)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 1: Full pipeline (audio broken = ~0s for audio)")
print("=" * 60)

t1 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_stride=1, hand_det_conf=0.3,
)
t1 = time.perf_counter() - t1
print(f"  => {t1:.1f}s")

# Run 2: explicitly skip audio for fair comparison
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 2: Skip audio explicitly")
print("=" * 60)

t2 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_stride=1, hand_det_conf=0.3,
    skip_audio=True,
)
t2 = time.perf_counter() - t2
print(f"  => {t2:.1f}s")

# Run 3: skip audio again (fully warm)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 3: Skip audio (fully warm, 2nd run)")
print("=" * 60)

t3 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_stride=1, hand_det_conf=0.3,
    skip_audio=True,
)
t3 = time.perf_counter() - t3
print(f"  => {t3:.1f}s")

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"  Model loading:        {t_load:.1f}s")
print(f"  Run 1 (audio broken): {t1:.1f}s")
print(f"  Run 2 (skip audio):   {t2:.1f}s")
print(f"  Run 3 (fully warm):   {t3:.1f}s")

service.shutdown()
