#!/usr/bin/env python3
"""Persistent annotation pipeline service.

Keeps SLAM warm across multiple videos. Hand processing uses the CUDA binary
(cuda_hand) which handles YOLO + WiLoR + MANO entirely on GPU without PyTorch.

Architecture:
    Parent process: DROID-SLAM (native C++ decoder, CUDA)
    Per-video:      cuda_hand binary (YOLO + WiLoR + MANO, pure CUDA)

Usage:
    # As a library (from visualizer.py or scripts):
    service = PipelineService(droid_weights)
    result = service.process_video("/path/to/video.mov", "/path/to/output")
    service.shutdown()

    # Standalone (reads jobs from stdin):
    python pipeline_service.py --listen
"""

import argparse
import gc
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch

# Import pipeline utilities
sys.path.insert(0, os.path.dirname(__file__))
from annotate_pipeline import (
    _get_native_decode,
    _interpolate_hand_keypoints,
    _interpolate_poses_simple,
    get_video_metadata,
    get_iphone_intrinsics,
    write_lerobot_dataset,
    start_audio_subprocess,
    collect_audio_subprocess,
)
from convert_video import build_ffmpeg_cmd

# ---------------------------------------------------------------------------
# CUDA hand processing helper
# ---------------------------------------------------------------------------

def _parse_cuda_hand_output(cuda_output, result_path):
    """Parse binary output from cuda_hand and save as npz."""
    with open(cuda_output, 'rb') as f:
        num_results = struct.unpack('i', f.read(4))[0]
        num_frames = struct.unpack('i', f.read(4))[0]
        out_stride = struct.unpack('i', f.read(4))[0]

        left_kp3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
        right_kp3d = np.zeros((num_frames, 21, 3), dtype=np.float32)
        left_kp2d = np.zeros((num_frames, 21, 2), dtype=np.float32)
        right_kp2d = np.zeros((num_frames, 21, 2), dtype=np.float32)
        left_detected = np.zeros(num_frames, dtype=bool)
        right_detected = np.zeros(num_frames, dtype=bool)

        sampled_indices = []
        for _ in range(num_results):
            fi = struct.unpack('i', f.read(4))[0]
            left_det = struct.unpack('B', f.read(1))[0]
            right_det = struct.unpack('B', f.read(1))[0]
            l_kp3d = np.frombuffer(f.read(63 * 4), dtype=np.float32).reshape(21, 3)
            r_kp3d = np.frombuffer(f.read(63 * 4), dtype=np.float32).reshape(21, 3)
            l_kp2d = np.frombuffer(f.read(42 * 4), dtype=np.float32).reshape(21, 2)
            r_kp2d = np.frombuffer(f.read(42 * 4), dtype=np.float32).reshape(21, 2)
            sampled_indices.append(fi)
            if fi < num_frames:
                if left_det:
                    left_kp3d[fi] = l_kp3d
                    left_kp2d[fi] = l_kp2d
                    left_detected[fi] = True
                if right_det:
                    right_kp3d[fi] = r_kp3d
                    right_kp2d[fi] = r_kp2d
                    right_detected[fi] = True

    # Interpolate if strided
    if out_stride > 1 and len(sampled_indices) > 1:
        if sampled_indices[-1] != num_frames - 1:
            sampled_indices.append(num_frames - 1)
        for arr in [left_kp3d, right_kp3d, left_kp2d, right_kp2d]:
            _interpolate_hand_keypoints(arr, sampled_indices, num_frames)
        for det_arr in [left_detected, right_detected]:
            full = np.zeros(num_frames, dtype=bool)
            full[sampled_indices] = det_arr[sampled_indices]
            for i in range(len(sampled_indices) - 1):
                a, b = sampled_indices[i], sampled_indices[i + 1]
                mid = (a + b) // 2
                full[a:mid + 1] = det_arr[a]
                full[mid + 1:b] = det_arr[b]
            det_arr[:] = full

    left_rate = left_detected.sum() / num_frames * 100
    right_rate = right_detected.sum() / num_frames * 100
    n_hands = left_detected.sum() + right_detected.sum()
    print(
        f"  [CUDA Hand] {n_hands} detections, "
        f"left={left_rate:.1f}% right={right_rate:.1f}%",
        flush=True,
    )

    np.savez(
        result_path,
        left_kp3d=left_kp3d, right_kp3d=right_kp3d,
        left_kp2d=left_kp2d, right_kp2d=right_kp2d,
        left_detected=left_detected, right_detected=right_detected,
        num_frames=np.array([num_frames]),
    )

    try:
        os.remove(cuda_output)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# PipelineService: orchestrates both processes
# ---------------------------------------------------------------------------

class PipelineService:
    """Persistent annotation pipeline. SLAM stays warm, hands use persistent CUDA binary.

    Usage:
        service = PipelineService(droid_weights="/path/to/droid.pth")
        result = service.process_video("/path/to/video.mov", "/path/to/output")
        service.shutdown()
    """

    def __init__(
        self,
        droid_weights="/workspace/DROID-SLAM/checkpoints/droid.pth",
        wilor_dir="/workspace/WiLoR",  # kept for API compat, unused
        device="cuda",
        asr_model="parakeet-tdt_ctc-110m",
    ):
        self.device = device
        self.asr_model = asr_model
        self._nd = None
        self._ready = False
        self._hand_proc = None

        t_start = time.perf_counter()
        self._start_hand_worker()
        self._init_slam()
        t_total = time.perf_counter() - t_start
        print(f"[Service] Ready in {t_total:.1f}s", flush=True)
        self._ready = True

    def _start_hand_worker(self):
        """Start persistent cuda_hand binary in --listen mode."""
        cuda_hand_dir = os.path.join(os.path.dirname(__file__), 'cuda_hand')
        cuda_hand_bin = os.path.join(cuda_hand_dir, 'cuda_hand')
        weights_dir = os.path.join(cuda_hand_dir, 'data', 'weights')

        if not os.path.exists(cuda_hand_bin):
            raise FileNotFoundError(f"CUDA hand binary not found: {cuda_hand_bin}")

        self._hand_proc = subprocess.Popen(
            [cuda_hand_bin, '--weights-dir', weights_dir, '--listen'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Collect stderr in background
        self._hand_stderr = []
        def _read_stderr():
            for line in self._hand_proc.stderr:
                self._hand_stderr.append(line.decode().rstrip())
                print(f"  [cuda_hand] {self._hand_stderr[-1]}", flush=True)
        self._stderr_thread = Thread(target=_read_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for READY
        ready_line = self._hand_proc.stdout.readline().decode().strip()
        if ready_line != "READY":
            raise RuntimeError(f"cuda_hand failed to start: {ready_line}")
        print("[Service] CUDA hand worker started", flush=True)

    def _init_slam(self):
        """Initialize CUDA DROID-SLAM binary and native decoder."""
        self._nd = _get_native_decode()
        # Locate the CUDA DROID-SLAM binary and weights
        self._cuda_slam_dir = os.path.join(os.path.dirname(__file__), "cuda_slam")
        self._cuda_slam_bin = os.path.join(self._cuda_slam_dir, "cuda_droid")
        self._cuda_slam_weights = os.path.join(self._cuda_slam_dir, "data", "weights")
        if not os.path.isfile(self._cuda_slam_bin):
            raise RuntimeError(f"CUDA DROID-SLAM binary not found: {self._cuda_slam_bin}")
        print("[Service] CUDA DROID-SLAM initialized", flush=True)

    def process_video(
        self,
        video_path,
        output_dir,
        fast_traj=True,
        backend_steps=None,
        hand_stride=2,
        hand_det_conf=0.3,
        skip_slam=False,
        skip_hands=False,
        skip_audio=False,
        skip_video_convert=False,
        on_progress=None,
    ):
        """Process one video end-to-end. Returns the output dataset path.

        on_progress: optional callback(phase: str, detail: str) for status updates.
        """
        if backend_steps is None:
            backend_steps = (2, 3) if fast_traj else (3, 5)

        video_path = str(Path(video_path).resolve())
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)
        video_output = str(Path(output_dir) / "converted.mp4")

        t_total_start = time.perf_counter()

        def _progress(phase, detail=""):
            if on_progress:
                on_progress(phase, detail)
            print(f"[Service] {phase} {detail}", flush=True)

        fps, width, height, nf_est = get_video_metadata(video_path)
        nf_est = nf_est if nf_est > 0 else 2000

        # ── Build ffmpeg command (deferred until after inference) ──
        ffmpeg_cmd = None
        if not skip_video_convert:
            ffmpeg_cmd = build_ffmpeg_cmd(video_path, video_output, max_threads=4)

        # ── Start audio in background ──
        audio_proc = None
        audio_result_file = None
        if not skip_audio:
            _progress("Audio", "starting transcription...")
            audio_proc, audio_result_file = start_audio_subprocess(
                video_path, fps, nf_est, self.asr_model,
            )

        # ── Start video conversion early (NVENC uses dedicated hardware) ──
        video_proc = None
        if ffmpeg_cmd is not None:
            _progress("Video", "starting (NVENC, background)...")
            video_proc = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t_vc = time.perf_counter()

        # ── Dispatch hand processing (persistent CUDA binary) ──
        hand_result = None
        result_path = f"/tmp/hand_result_svc_{os.getpid()}_{int(time.time())}.npz"
        cuda_output = result_path.replace('.npz', '_cuda.bin')
        if not skip_hands:
            _progress("Hands", "dispatching to CUDA hand worker...")
            job = json.dumps({
                "video": video_path,
                "output": cuda_output,
                "det_conf": hand_det_conf,
                "stride": hand_stride,
            }) + "\n"
            self._hand_proc.stdin.write(job.encode())
            self._hand_proc.stdin.flush()

        # ── Run SLAM in parent (concurrently with hand processing) ──
        slam_result = None
        num_frames = 0
        if not skip_slam:
            _progress("SLAM", "tracking...")
            slam_result, num_frames = self._run_slam(
                video_path, fps, width, height,
                fast_traj=fast_traj, backend_steps=backend_steps,
            )
            _progress("SLAM", f"done ({num_frames} frames)")

        # ── Collect hand result ──
        if not skip_hands:
            _progress("Hands", "waiting for CUDA hand result...")
            # Read response line from cuda_hand stdout
            resp_line = self._hand_proc.stdout.readline().decode().strip()
            resp = json.loads(resp_line)
            if resp.get("status") != "done":
                raise RuntimeError(f"CUDA hand failed: {resp}")
            t_hand = resp.get("time", "?")

            _parse_cuda_hand_output(cuda_output, result_path)

            data = np.load(result_path)
            hand_num_frames = int(data["num_frames"][0]) if "num_frames" in data else num_frames
            hand_result = {
                "_model": "wilor", "_done": True,
                "left_hand_keypoints_3d": data["left_kp3d"],
                "right_hand_keypoints_3d": data["right_kp3d"],
                "left_hand_keypoints_2d": data["left_kp2d"],
                "right_hand_keypoints_2d": data["right_kp2d"],
                "left_hand_detected": data["left_detected"],
                "right_hand_detected": data["right_detected"],
            }
            if num_frames == 0:
                num_frames = hand_num_frames
            if os.path.exists(result_path):
                os.remove(result_path)
            _progress("Hands", f"done ({t_hand}s)")

        if num_frames == 0:
            fps, _, _, num_frames = get_video_metadata(video_path)

        # ── Collect audio ──
        audio_result = None
        if audio_proc is not None:
            _progress("Audio", "collecting...")
            audio_result = collect_audio_subprocess(audio_proc, audio_result_file)
            if audio_result is not None:
                ft = audio_result["frame_text"]
                if num_frames > 0 and len(ft) != num_frames:
                    if len(ft) < num_frames:
                        ft.extend([""] * (num_frames - len(ft)))
                    else:
                        audio_result["frame_text"] = ft[:num_frames]

        # ── Wait for video conversion (started early, NVENC runs on dedicated HW) ──
        if video_proc is not None:
            _progress("Video", "waiting for NVENC...")
            video_proc.wait()
            _progress("Video", f"done ({time.perf_counter()-t_vc:.1f}s)")
            if video_proc.returncode != 0:
                print(f"  WARNING: ffmpeg failed (code {video_proc.returncode})", flush=True)

        # ── Assemble dataset ──
        _progress("Assembly", "writing dataset...")
        dataset_path = write_lerobot_dataset(
            output_dir, video_output, slam_result, hand_result, fps, num_frames,
            audio_result=audio_result, input_video_path=video_path,
        )

        # Cleanup temp converted video
        if os.path.exists(video_output) and "converted.mp4" in video_output:
            os.remove(video_output)

        t_total = time.perf_counter() - t_total_start
        _progress("Done", f"total {t_total:.1f}s")

        # Free CUDA memory between videos
        torch.cuda.empty_cache()
        gc.collect()

        return dataset_path

    def _run_slam(self, video_path, fps, width, height, fast_traj=True, backend_steps=(3, 5)):
        """Run CUDA DROID-SLAM on a single video, piping frames via stdin."""
        import tempfile
        import threading

        h0, w0 = height, width
        scale = np.sqrt((384 * 512) / (h0 * w0))
        h1 = int(h0 * scale) // 8 * 8
        w1 = int(w0 * scale) // 8 * 8

        fx, fy, cx, cy = get_iphone_intrinsics(width, height)
        fx_s = fx * (w1 / w0)
        fy_s = fy * (h1 / h0)
        cx_s = cx * (w1 / w0)
        cy_s = cy * (h1 / h0)
        intrinsics_scaled = np.array([fx_s, fy_s, cx_s, cy_s], dtype=np.float32)

        print(f"  SLAM resolution: {w1}x{h1}", flush=True)

        with tempfile.TemporaryDirectory(prefix="cuda_slam_") as tmpdir:
            calib_path = os.path.join(tmpdir, "calib.bin")
            pose_path = os.path.join(tmpdir, "poses.bin")

            with open(calib_path, "wb") as f:
                f.write(struct.pack("4f", fx_s, fy_s, cx_s, cy_s))

            # Build command with stdin mode (pipe frames directly, no disk I/O)
            cmd = [
                self._cuda_slam_bin,
                "--weights", self._cuda_slam_weights,
                "--calib", calib_path,
                "--stdin", str(h1), str(w1),
                "--max-frames", "99999",
                "--cam-to-world",
                "--pose-output", pose_path,
                "--frontend-window", "15",
                "--update-steps", "2",
                "--backend-radius", "1",
            ]
            if backend_steps and len(backend_steps) >= 2:
                cmd += ["--backend", str(backend_steps[0]), str(backend_steps[1])]

            t_slam_start = time.perf_counter()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Feed decoded frames via stdin in a background thread
            num_frames_box = [0]
            feed_error = [None]

            def feed_frames():
                decoder = self._nd.AsyncVideoDecoder()
                decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)
                n = 0
                try:
                    while True:
                        result = decoder.get_next()
                        if result is None:
                            break
                        _, slam_bgr = result
                        proc.stdin.write(slam_bgr.astype(np.float32).tobytes())
                        n += 1
                except (BrokenPipeError, OSError) as e:
                    feed_error[0] = e
                finally:
                    decoder.stop()
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                    num_frames_box[0] = n

            feeder = threading.Thread(target=feed_frames, daemon=True)
            feeder.start()

            stdout_bytes = proc.stdout.read()
            proc.wait()
            feeder.join()

            num_frames = num_frames_box[0]
            t_slam = time.perf_counter() - t_slam_start

            if proc.returncode != 0:
                stderr_text = proc.stderr.read().decode(errors="replace")
                print(f"  CUDA SLAM stderr: {stderr_text[-500:]}", flush=True)
                raise RuntimeError(f"CUDA DROID-SLAM failed (code {proc.returncode})")

            print(f"  CUDA SLAM: {t_slam:.1f}s ({num_frames} frames, "
                  f"{num_frames/t_slam:.0f} fps)", flush=True)

            # Read keyframe poses
            with open(pose_path, "rb") as f:
                nk = struct.unpack("i", f.read(4))[0]
                kf_timestamps = np.array(struct.unpack(f"{nk}i", f.read(nk * 4)), dtype=np.int64)
                kf_poses = np.frombuffer(f.read(nk * 7 * 4), dtype=np.float32).reshape(nk, 7)

        # Interpolate keyframe poses to all frames
        all_poses = _interpolate_poses_simple(kf_timestamps, kf_poses, num_frames)
        print(f"  Trajectory: {nk} keyframes -> {num_frames} frames", flush=True)

        slam_result = {
            "poses": all_poses,
            "intrinsics": intrinsics_scaled,
            "slam_resolution": (h1, w1),
        }

        return slam_result, num_frames

    def health_check(self):
        """Check service health."""
        hand_alive = self._hand_proc is not None and self._hand_proc.poll() is None
        return {"status": "alive" if hand_alive else "dead"}

    def shutdown(self):
        """Shut down the persistent cuda_hand worker."""
        if self._hand_proc and self._hand_proc.poll() is None:
            try:
                self._hand_proc.stdin.write(b"SHUTDOWN\n")
                self._hand_proc.stdin.flush()
                self._hand_proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._hand_proc.kill()
                self._hand_proc.wait(timeout=5)
        self._hand_proc = None
        print("[Service] Shutdown complete", flush=True)

    def __del__(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# Standalone mode: read jobs from stdin
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Persistent annotation pipeline service")
    parser.add_argument("--droid-weights", default="/workspace/DROID-SLAM/checkpoints/droid.pth")
    parser.add_argument("--wilor-dir", default="/workspace/WiLoR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="parakeet-tdt_ctc-110m")
    parser.add_argument("--listen", action="store_true",
                        help="Read JSON job requests from stdin")
    args = parser.parse_args()

    service = PipelineService(
        droid_weights=args.droid_weights,
        wilor_dir=args.wilor_dir,
        device=args.device,
        asr_model=args.asr_model,
    )

    if args.listen:
        print("[Service] Listening for jobs on stdin (JSON-lines)...", flush=True)
        print("[Service] Format: {\"video_path\": \"/path/to/video.mov\", \"output_dir\": \"/path/to/output\"}", flush=True)
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                except json.JSONDecodeError as e:
                    print(json.dumps({"error": str(e)}), flush=True)
                    continue

                if job.get("cmd") == "shutdown":
                    break
                if job.get("cmd") == "health":
                    print(json.dumps(service.health_check()), flush=True)
                    continue

                video_path = job.get("video_path")
                output_dir = job.get("output_dir")
                if not video_path or not output_dir:
                    print(json.dumps({"error": "Need video_path and output_dir"}), flush=True)
                    continue

                try:
                    t0 = time.perf_counter()
                    dataset_path = service.process_video(
                        video_path, output_dir,
                        fast_traj=job.get("fast_traj", True),
                        hand_stride=job.get("hand_stride", 1),
                        hand_det_conf=job.get("hand_det_conf", 0.3),
                        skip_slam=job.get("skip_slam", False),
                        skip_hands=job.get("skip_hands", False),
                        skip_audio=job.get("skip_audio", False),
                        skip_video_convert=job.get("skip_video_convert", False),
                    )
                    elapsed = time.perf_counter() - t0
                    print(json.dumps({
                        "status": "done",
                        "dataset_path": dataset_path,
                        "time": round(elapsed, 1),
                    }), flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(json.dumps({"status": "error", "error": str(e)}), flush=True)
        except KeyboardInterrupt:
            pass
    else:
        print("[Service] Ready. Call service.process_video() to process a video.", flush=True)
        print("[Service] Use --listen to read jobs from stdin.", flush=True)

    service.shutdown()


if __name__ == "__main__":
    main()
