#!/usr/bin/env python3
"""Persistent annotation pipeline service.

Keeps SLAM and hand models warm across multiple videos, eliminating ~13s of
per-video import and model loading overhead.

Architecture:
    Parent process: DROID-SLAM (native C++ decoder, CUDA)
    Child process:  YOLO + WiLoR hand detection (OpenCV decode, CUDA)

The child is forked BEFORE any CUDA calls and keeps models on GPU permanently.
Communication is via a JSON-lines protocol over a unix socketpair.

Usage:
    # As a library (from visualizer.py or scripts):
    service = PipelineService(droid_weights, wilor_dir)
    result = service.process_video("/path/to/video.mov", "/path/to/output")
    service.shutdown()

    # Standalone (reads jobs from stdin):
    python pipeline_service.py --listen
"""

import argparse
import gc
import json
import multiprocessing
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch

# Import pipeline utilities
sys.path.insert(0, os.path.dirname(__file__))
from annotate_pipeline import (
    _install_pytorch_lightning_stub,
    _get_native_decode,
    _load_wilor_cpu,
    _preprocess_hand_crop,
    _run_wilor_batch,
    _interpolate_hand_keypoints,
    _interpolate_poses_simple,
    get_video_metadata,
    get_iphone_intrinsics,
    write_lerobot_dataset,
    start_audio_subprocess,
    collect_audio_subprocess,
    _SENTINEL,
)
from convert_video import build_ffmpeg_cmd

# ---------------------------------------------------------------------------
# IPC helpers: length-prefixed JSON messages over a socket
# ---------------------------------------------------------------------------

def _send_msg(sock, obj):
    """Send a JSON message with a 4-byte length prefix."""
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_msg(sock):
    """Receive a length-prefixed JSON message. Returns dict or None on EOF."""
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    length = struct.unpack("!I", hdr)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data)


# ---------------------------------------------------------------------------
# Hand worker: persistent child process
# ---------------------------------------------------------------------------

def _hand_worker_service(wilor_dir, child_sock_fd):
    """Long-running hand detection process. Reuses YOLO + WiLoR across videos.

    Forked before CUDA init — inherits all Python imports from parent.
    Communicates via JSON messages over a unix socketpair.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sock = socket.fromfd(child_sock_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    os.close(child_sock_fd)  # close the raw fd, we use the socket object

    from concurrent.futures import ThreadPoolExecutor as TPE

    try:
        torch.cuda.set_device(0)
        sys.path.insert(0, wilor_dir)
        from ultralytics import YOLO

        # ── One-time model loading ──
        t_load = time.perf_counter()

        # WiLoR on CPU in background thread
        wilor_load_result = [None]
        def _build_wilor_bg():
            wilor_load_result[0] = _load_wilor_cpu(wilor_dir)
        wilor_thread = Thread(target=_build_wilor_bg, daemon=True)
        wilor_thread.start()

        # Load config for preprocessing constants
        orig_cwd = os.getcwd()
        os.chdir(wilor_dir)
        try:
            from wilor.configs import get_config as _get_wilor_cfg
            cfg = _get_wilor_cfg(
                os.path.join(wilor_dir, 'pretrained_models', 'model_config.yaml'),
                update_cachedir=True,
            )
        finally:
            os.chdir(orig_cwd)

        mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        std = 255.0 * np.array(cfg.MODEL.IMAGE_STD, dtype=np.float32)
        bbox_shape = cfg.MODEL.get('BBOX_SHAPE', [192, 256])
        img_size_cfg = cfg.MODEL.IMAGE_SIZE
        rescale_factor = 2.0
        BATCH_SIZE = 48
        YOLO_BATCH = 16

        # YOLO to GPU
        hand_stream = torch.cuda.Stream()
        wilor_stream = torch.cuda.Stream()
        with torch.cuda.stream(hand_stream):
            detector = YOLO(
                os.path.join(wilor_dir, 'pretrained_models', 'detector.pt'),
            ).to('cuda')

        # WiLoR to GPU
        wilor_thread.join()
        model_cpu, model_cfg = wilor_load_result[0]
        with torch.cuda.stream(wilor_stream):
            model = model_cpu.to('cuda').eval().half()
            model.backbone.skip_blocks = True
        del model_cpu

        t_ready = time.perf_counter() - t_load
        print(f"  [Hand service] Models loaded in {t_ready:.1f}s", flush=True)
        _send_msg(sock, {"status": "ready"})

        # ── Per-video processing loop ──
        while True:
            msg = _recv_msg(sock)
            if msg is None:
                break  # parent closed socket
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                break
            if cmd == "health":
                vram = torch.cuda.memory_allocated() / 1e6
                _send_msg(sock, {"status": "alive", "vram_mb": round(vram)})
                continue
            if cmd != "process":
                _send_msg(sock, {"status": "error", "message": f"Unknown cmd: {cmd}"})
                continue

            video_path = msg["video_path"]
            det_conf = msg.get("det_conf", 0.3)
            stride = msg.get("stride", 1)
            num_frames_est = msg.get("num_frames_est", 2000)
            result_path = msg["result_path"]

            try:
                t_start = time.perf_counter()
                _process_one_video(
                    video_path, det_conf, stride, num_frames_est, result_path,
                    detector, model, model_cfg, hand_stream, wilor_stream,
                    mean, std, bbox_shape, img_size_cfg, rescale_factor,
                    BATCH_SIZE, YOLO_BATCH,
                )
                elapsed = time.perf_counter() - t_start
                _send_msg(sock, {"status": "done", "result_path": result_path, "time": round(elapsed, 1)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                _send_msg(sock, {"status": "error", "message": str(e)})

            # Cleanup between videos
            torch.cuda.empty_cache()

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        sock.close()


def _process_one_video(
    video_path, det_conf, stride, num_frames_est, result_path,
    detector, model, model_cfg, hand_stream, wilor_stream,
    mean, std, bbox_shape, img_size_cfg, rescale_factor,
    BATCH_SIZE, YOLO_BATCH,
):
    """Process a single video through YOLO + WiLoR. Uses pre-loaded models."""
    from concurrent.futures import ThreadPoolExecutor as TPE

    with torch.no_grad():
        max_frames = max(num_frames_est + 100, 4000)
        left_kp3d = np.full((max_frames, 21, 3), np.nan, dtype=np.float32)
        right_kp3d = np.full((max_frames, 21, 3), np.nan, dtype=np.float32)
        left_kp2d = np.full((max_frames, 21, 2), np.nan, dtype=np.float32)
        right_kp2d = np.full((max_frames, 21, 2), np.nan, dtype=np.float32)
        left_detected = np.zeros(max_frames, dtype=bool)
        right_detected = np.zeros(max_frames, dtype=bool)

        crop_queue = Queue(maxsize=256)
        n_hands_done = [0]
        wilor_error = []

        def _wilor_inference_loop():
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(wilor_stream), torch.no_grad():
                    crop_buf_imgs = []
                    crop_buf_meta = []
                    while True:
                        item = crop_queue.get()
                        if item is _SENTINEL:
                            break
                        img_t, meta = item
                        crop_buf_imgs.append(img_t)
                        crop_buf_meta.append(meta)
                        if len(crop_buf_imgs) >= BATCH_SIZE:
                            _run_wilor_batch(
                                crop_buf_imgs[:BATCH_SIZE],
                                crop_buf_meta[:BATCH_SIZE],
                                model, model_cfg, 'cuda',
                                left_kp3d, right_kp3d,
                                left_kp2d, right_kp2d,
                                left_detected, right_detected,
                            )
                            n_hands_done[0] += BATCH_SIZE
                            crop_buf_imgs = crop_buf_imgs[BATCH_SIZE:]
                            crop_buf_meta = crop_buf_meta[BATCH_SIZE:]
                    if crop_buf_imgs:
                        _run_wilor_batch(
                            crop_buf_imgs, crop_buf_meta,
                            model, model_cfg, 'cuda',
                            left_kp3d, right_kp3d,
                            left_kp2d, right_kp2d,
                            left_detected, right_detected,
                        )
                        n_hands_done[0] += len(crop_buf_imgs)
            except Exception:
                import traceback
                wilor_error.append(traceback.format_exc())

        wilor_thread = Thread(target=_wilor_inference_loop, daemon=True)
        wilor_thread.start()

        # ── YOLO detection (streaming from OpenCV) ──
        t_yolo_start = time.perf_counter()
        total_crops = 0
        yolo_batch = []
        sampled_indices = []
        prep_pool = TPE(max_workers=4)

        def _preprocess_and_enqueue(fi, img_bgr, center, scale, is_right, img_wh):
            img_tensor, bbox_size = _preprocess_hand_crop(
                img_bgr, center, scale, bbox_shape, img_size_cfg, is_right, mean, std,
            )
            crop_queue.put((
                torch.from_numpy(img_tensor).half(),
                (fi, center.copy(), bbox_size, img_wh, is_right),
            ))

        def _flush_yolo():
            nonlocal total_crops
            if not yolo_batch:
                return
            fis = [x[0] for x in yolo_batch]
            imgs = [x[1] for x in yolo_batch]
            with torch.cuda.stream(hand_stream):
                results = detector(imgs, conf=det_conf, verbose=False)
            for i, (det_fi, dets) in enumerate(zip(fis, results)):
                bboxes, is_right_list = [], []
                for det in dets:
                    bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                    is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())
                    bboxes.append(bbox[:4].tolist())
                if bboxes:
                    boxes = np.stack(bboxes).astype(np.float32)
                    right_arr = np.stack(is_right_list).astype(np.float32)
                    img_bgr = imgs[i]
                    img_h, img_w = img_bgr.shape[:2]
                    centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
                    scales = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
                    img_wh = np.array([img_w, img_h], dtype=np.float32)
                    for k in range(len(boxes)):
                        prep_pool.submit(
                            _preprocess_and_enqueue,
                            det_fi, img_bgr, centers[k], scales[k],
                            right_arr[k], img_wh,
                        )
                    total_crops += len(bboxes)

        cap = cv2.VideoCapture(video_path)
        fi = 0
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                break
            if stride <= 1 or fi % stride == 0:
                sampled_indices.append(fi)
                yolo_batch.append((fi, img_bgr))
                if len(yolo_batch) >= YOLO_BATCH:
                    _flush_yolo()
                    yolo_batch = []
            fi += 1
        cap.release()
        _flush_yolo()
        yolo_batch = []

        prep_pool.shutdown(wait=True)
        crop_queue.put(_SENTINEL)

        num_frames = fi
        t_yolo = time.perf_counter() - t_yolo_start
        print(
            f"  [Hand service] Decode+YOLO: {t_yolo:.1f}s "
            f"({num_frames/max(t_yolo,0.001):.0f} fps), {total_crops} crops",
            flush=True,
        )

        wilor_thread.join()
        if wilor_error:
            raise RuntimeError(f"WiLoR failed:\n{wilor_error[0]}")

        # Trim
        left_kp3d = left_kp3d[:num_frames]
        right_kp3d = right_kp3d[:num_frames]
        left_kp2d = left_kp2d[:num_frames]
        right_kp2d = right_kp2d[:num_frames]
        left_detected = left_detected[:num_frames]
        right_detected = right_detected[:num_frames]

        # Interpolate if strided
        if stride > 1 and len(sampled_indices) > 1:
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
                    full[a:mid+1] = det_arr[a]
                    full[mid+1:b] = det_arr[b]
                det_arr[:] = full

        left_rate = left_detected.sum() / num_frames * 100
        right_rate = right_detected.sum() / num_frames * 100
        t_total = time.perf_counter() - t_yolo_start
        print(
            f"  [Hand service] Done: {t_total:.1f}s, {n_hands_done[0]} hands, "
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


# ---------------------------------------------------------------------------
# PipelineService: orchestrates both processes
# ---------------------------------------------------------------------------

class PipelineService:
    """Persistent annotation pipeline. Models loaded once, reused per video.

    Usage:
        service = PipelineService(droid_weights="/path/to/droid.pth",
                                  wilor_dir="/path/to/WiLoR")
        result = service.process_video("/path/to/video.mov", "/path/to/output")
        service.shutdown()
    """

    def __init__(
        self,
        droid_weights="/workspace/DROID-SLAM/checkpoints/droid.pth",
        wilor_dir="/workspace/WiLoR",
        device="cuda",
        asr_model="parakeet-tdt_ctc-110m",
    ):
        self.droid_weights = droid_weights
        self.wilor_dir = wilor_dir
        self.device = device
        self.asr_model = asr_model
        self.hand_sock = None
        self.hand_proc = None
        self.Droid = None
        self._nd = None
        self._ready = False

        t_start = time.perf_counter()
        self._pre_import()
        self._start_hand_worker()
        self._init_slam()
        self._wait_hand_ready()
        t_total = time.perf_counter() - t_start
        print(f"[Service] Ready in {t_total:.1f}s", flush=True)
        self._ready = True

    def _pre_import(self):
        """Pre-import heavy modules so the forked child inherits them."""
        _install_pytorch_lightning_stub()
        import timm  # noqa: F401
        sys.path.insert(0, self.wilor_dir)
        from ultralytics import YOLO  # noqa: F401
        import smplx  # noqa: F401
        orig_cwd = os.getcwd()
        os.chdir(self.wilor_dir)
        try:
            from wilor.configs import get_config  # noqa: F401
            from wilor.models.wilor import WiLoR  # noqa: F401
        finally:
            os.chdir(orig_cwd)
        print("[Service] Modules pre-imported", flush=True)

    def _start_hand_worker(self):
        """Fork hand worker BEFORE any CUDA calls."""
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        mp_ctx = multiprocessing.get_context('fork')
        self.hand_proc = mp_ctx.Process(
            target=_hand_worker_service,
            args=(self.wilor_dir, child_sock.fileno()),
            daemon=True,
        )
        self.hand_proc.start()
        child_sock.close()
        self.hand_sock = parent_sock

    def _wait_hand_ready(self):
        """Wait for child to finish loading models and signal ready."""
        msg = _recv_msg(self.hand_sock)
        if msg is None or msg.get("status") != "ready":
            raise RuntimeError(f"Hand worker failed to start: {msg}")

    def _init_slam(self):
        """Import DROID-SLAM (loads CUDA backend). Done after fork."""
        droid_slam_path = os.path.join(
            os.path.dirname(self.droid_weights), "..", "droid_slam"
        )
        sys.path.insert(0, droid_slam_path)
        from droid import Droid
        self.Droid = Droid
        self._nd = _get_native_decode()
        print("[Service] DROID-SLAM initialized", flush=True)

    def process_video(
        self,
        video_path,
        output_dir,
        fast_traj=True,
        backend_steps=(7, 12),
        hand_stride=1,
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

        # ── Dispatch hand processing to child ──
        hand_result = None
        result_path = f"/tmp/hand_result_svc_{os.getpid()}_{int(time.time())}.npz"
        if not skip_hands:
            _progress("Hands", "dispatching to worker...")
            _send_msg(self.hand_sock, {
                "cmd": "process",
                "video_path": video_path,
                "det_conf": hand_det_conf,
                "stride": hand_stride,
                "num_frames_est": nf_est,
                "result_path": result_path,
            })

        # ── Run SLAM in parent (concurrently with child's hand processing) ──
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
            _progress("Hands", "waiting for result...")
            hand_msg = _recv_msg(self.hand_sock)
            if hand_msg is None:
                raise RuntimeError("Hand worker connection lost")
            if hand_msg.get("status") == "error":
                raise RuntimeError(f"Hand worker: {hand_msg.get('message')}")
            if hand_msg.get("status") != "done":
                raise RuntimeError(f"Unexpected hand status: {hand_msg}")

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
            _progress("Hands", f"done ({hand_msg.get('time', '?')}s)")

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

        # ── Video conversion (GPU now idle) ──
        if ffmpeg_cmd is not None:
            _progress("Video", "converting (NVENC)...")
            t_vc = time.perf_counter()
            vc = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _progress("Video", f"done ({time.perf_counter()-t_vc:.1f}s)")
            if vc.returncode != 0:
                print(f"  WARNING: ffmpeg failed (code {vc.returncode})", flush=True)

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

    def _run_slam(self, video_path, fps, width, height, fast_traj=True, backend_steps=(7, 12)):
        """Run DROID-SLAM on a single video. Fresh Droid instance each time."""
        h0, w0 = height, width
        scale = np.sqrt((384 * 512) / (h0 * w0))
        h1 = int(h0 * scale) // 8 * 8
        w1 = int(w0 * scale) // 8 * 8

        fx, fy, cx, cy = get_iphone_intrinsics(width, height)
        intrinsics_scaled = torch.as_tensor([
            fx * (w1 / w0), fy * (h1 / h0), cx * (w1 / w0), cy * (h1 / h0),
        ])

        slam_args = argparse.Namespace(
            weights=self.droid_weights, buffer=512, image_size=[h1, w1],
            disable_vis=True, stereo=False, beta=0.3, filter_thresh=2.4,
            warmup=8, keyframe_thresh=4.0,
            frontend_thresh=16.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
            backend_thresh=22.0, backend_radius=2, backend_nms=3,
            upsample=False, frontend_device=self.device, backend_device=self.device,
        )

        decoder = self._nd.AsyncVideoDecoder()
        decoder.start(video_path, w1, h1, slam_only=True, queue_depth=128)

        droid = self.Droid(slam_args)
        print(f"  SLAM resolution: {w1}x{h1}", flush=True)

        # ── Tracking ──
        t_track_start = time.perf_counter()
        slam_tensors = [] if not fast_traj else None
        t = 0
        while True:
            result = decoder.get_next()
            if result is None:
                break
            _, slam_bgr = result
            tensor = torch.from_numpy(slam_bgr).permute(2, 0, 1).unsqueeze(0).cuda()
            if slam_tensors is not None:
                slam_tensors.append(tensor)
            droid.track(t, tensor, intrinsics=intrinsics_scaled)
            t += 1
        decoder.stop()
        num_frames = t

        t_track = time.perf_counter() - t_track_start
        print(f"  SLAM tracking: {t_track:.1f}s ({num_frames/t_track:.0f} fps)", flush=True)

        # ── Backend ──
        del droid.frontend
        torch.cuda.empty_cache()
        t_be = time.perf_counter()
        droid.backend(backend_steps[0])
        torch.cuda.empty_cache()
        droid.backend(backend_steps[1])
        print(f"  Backend: {time.perf_counter()-t_be:.1f}s", flush=True)

        # ── Trajectory ──
        if fast_traj:
            torch.cuda.empty_cache()
            N = droid.video.counter.value
            kf_tstamps = droid.video.tstamp[:N].cpu().numpy().astype(np.int64)
            import lietorch
            kf_poses_se3 = lietorch.SE3(droid.video.poses[:N])
            kf_poses_raw = kf_poses_se3.inv().data.cpu().numpy()
            all_poses = _interpolate_poses_simple(kf_tstamps, kf_poses_raw, num_frames)
            print(f"  Trajectory: {N} keyframes -> {num_frames} frames", flush=True)
        else:
            def frame_stream():
                for i, tensor_item in enumerate(slam_tensors):
                    yield i, tensor_item, intrinsics_scaled
            traj = droid.traj_filler(frame_stream())
            all_poses = traj.inv().data.cpu().numpy()
            del slam_tensors
            print(f"  Trajectory: NN fill", flush=True)

        slam_result = {
            "poses": all_poses,
            "intrinsics": intrinsics_scaled.numpy(),
            "slam_resolution": (h1, w1),
        }

        # Free DROID memory
        del droid
        torch.cuda.empty_cache()

        return slam_result, num_frames

    def health_check(self):
        """Check if hand worker is alive. Returns dict with status."""
        if not self.hand_proc or not self.hand_proc.is_alive():
            return {"status": "dead"}
        try:
            _send_msg(self.hand_sock, {"cmd": "health"})
            # Set a timeout for the response
            self.hand_sock.settimeout(5.0)
            try:
                msg = _recv_msg(self.hand_sock)
            finally:
                self.hand_sock.settimeout(None)
            return msg or {"status": "unresponsive"}
        except (socket.timeout, OSError):
            return {"status": "unresponsive"}

    def shutdown(self):
        """Gracefully shut down the hand worker."""
        if self.hand_sock:
            try:
                _send_msg(self.hand_sock, {"cmd": "shutdown"})
            except OSError:
                pass
            self.hand_sock.close()
            self.hand_sock = None
        if self.hand_proc:
            self.hand_proc.join(timeout=10)
            if self.hand_proc.is_alive():
                self.hand_proc.kill()
                self.hand_proc.join(timeout=5)
            self.hand_proc = None
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
