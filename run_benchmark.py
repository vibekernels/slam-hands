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

# Run 1: full pipeline (cold)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 1: Full pipeline (cold)")
print("=" * 60)

t1 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_det_conf=0.3, hand_stride=1,
)
t1 = time.perf_counter() - t1
print(f"  => {t1:.1f}s")

# Run 2: full pipeline (warm)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 2: Full pipeline (warm)")
print("=" * 60)

t2 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_det_conf=0.3, hand_stride=1,
)
t2 = time.perf_counter() - t2
print(f"  => {t2:.1f}s")

# Run 3: full pipeline (warm, 2nd)
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n" + "=" * 60)
print("RUN 3: Full pipeline (warm, 2nd)")
print("=" * 60)

t3 = time.perf_counter()
result = service.process_video(
    "/workspace/IMG_1466.mov", output_dir,
    fast_traj=True, hand_det_conf=0.3, hand_stride=1,
)
t3 = time.perf_counter() - t3
print(f"  => {t3:.1f}s")

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"  Model loading:    {t_load:.1f}s")
print(f"  Run 1 (cold):     {t1:.1f}s")
print(f"  Run 2 (warm):     {t2:.1f}s")
print(f"  Run 3 (warm 2nd): {t3:.1f}s")

service.shutdown()
