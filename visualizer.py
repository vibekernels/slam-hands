#!/usr/bin/env python3
"""Browser-based visualizer for LeRobot v3.0 datasets.

Displays video with hand pose overlay and SLAM camera trajectory.

Usage:
    python3 visualizer.py /workspace/output_test
    python3 visualizer.py /workspace/output_test --port 8080
"""

import argparse
import json
import math
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pyarrow.parquet as pq


def load_dataset(dataset_dir: str) -> dict:
    """Load LeRobot dataset and convert to JSON-serializable dict."""
    dataset_dir = Path(dataset_dir)

    # Load info
    with open(dataset_dir / "meta" / "info.json") as f:
        info = json.load(f)

    # Load parquet
    table = pq.read_table(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    num_frames = len(table)

    # Find video file
    video_path = None
    for feat_name, feat in info["features"].items():
        if feat.get("dtype") == "video":
            vp = dataset_dir / "videos" / feat_name / "chunk-000" / "file-000.mp4"
            if vp.exists():
                video_path = str(vp)
                break
    if video_path is None:
        # fallback: find any mp4
        for mp4 in dataset_dir.rglob("*.mp4"):
            video_path = str(mp4)
            break

    # Extract frame data
    frames = []
    has_slam = "observation.slam.pose" in table.column_names
    has_left = "observation.hand.left.detected" in table.column_names
    has_right = "observation.hand.right.detected" in table.column_names

    # Get video dimensions from info
    video_feat = info["features"].get("observation.video", {})
    video_shape = video_feat.get("shape", [1080, 1920, 3])
    vid_h, vid_w = video_shape[0], video_shape[1]

    # Get intrinsics if available (for SLAM resolution, not video resolution)
    slam_intrinsics = None
    if has_slam and "observation.slam.intrinsics" in table.column_names:
        intr = table.column("observation.slam.intrinsics")[0].as_py()
        slam_intrinsics = intr  # [fx, fy, cx, cy] at SLAM resolution

    for i in range(num_frames):
        frame = {
            "t": table.column("timestamp")[i].as_py(),
        }

        if has_slam:
            pose = table.column("observation.slam.pose")[i].as_py()
            frame["slam"] = [float(v) for v in pose]

        for side in ["left", "right"]:
            det_col = f"observation.hand.{side}.detected"
            kp2d_col = f"observation.hand.{side}.keypoints_2d"
            kp3d_col = f"observation.hand.{side}.keypoints_3d"

            if det_col not in table.column_names:
                continue

            detected = table.column(det_col)[i].as_py()
            if detected > 0.5:
                kp2d_flat = table.column(kp2d_col)[i].as_py()
                # Convert from normalized coords to pixel coords
                # kp2d values are in range ~[-0.5, 0.5], centered at image center
                kp2d_pixels = []
                for j in range(0, len(kp2d_flat), 2):
                    px, py = kp2d_flat[j], kp2d_flat[j + 1]
                    if math.isnan(px):
                        kp2d_pixels.extend([None, None])
                    else:
                        # Already in pixel coordinates (full-frame)
                        kp2d_pixels.extend([round(px, 1), round(py, 1)])
                frame[f"{side}_kp2d"] = kp2d_pixels
            else:
                frame[f"{side}_kp2d"] = None

        frames.append(frame)

    return {
        "info": info,
        "num_frames": num_frames,
        "fps": info.get("fps", 30),
        "video_path": video_path,
        "video_width": vid_w,
        "video_height": vid_h,
        "has_slam": has_slam,
        "has_left_hand": has_left,
        "has_right_hand": has_right,
        "frames": frames,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LeRobot Dataset Visualizer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; }

.container { display: flex; height: 100vh; }

.video-panel {
    flex: 1; display: flex; flex-direction: column; min-width: 0;
}

.video-wrapper {
    position: relative; flex: 1; display: flex; align-items: center;
    justify-content: center; background: #000; overflow: hidden;
}

video { max-width: 100%; max-height: 100%; }

canvas#overlay {
    position: absolute; top: 0; left: 0; pointer-events: none;
}

.controls {
    background: #16213e; padding: 10px 16px; display: flex; align-items: center; gap: 12px;
}

.controls button {
    background: #0f3460; color: #eee; border: none; padding: 6px 14px;
    border-radius: 4px; cursor: pointer; font-size: 14px;
}
.controls button:hover { background: #1a5276; }

.controls input[type="range"] { flex: 1; accent-color: #e94560; }

.controls .time { font-size: 13px; font-variant-numeric: tabular-nums; min-width: 100px; }

.side-panel {
    width: 360px; background: #16213e; display: flex; flex-direction: column;
    border-left: 1px solid #0f3460;
}

.side-panel h3 {
    padding: 12px 16px 8px; font-size: 14px; color: #e94560;
    border-bottom: 1px solid #0f3460;
}

.slam-view {
    flex: 1; position: relative; min-height: 200px;
}

canvas#slam-canvas { width: 100%; height: 100%; }

.info-panel {
    padding: 12px 16px; font-size: 12px; line-height: 1.6;
    border-top: 1px solid #0f3460; max-height: 200px; overflow-y: auto;
}

.info-panel .label { color: #888; }
.info-panel .val { color: #53d8fb; font-variant-numeric: tabular-nums; }
.info-panel .detected { color: #4ade80; }
.info-panel .not-detected { color: #f87171; }

.legend {
    padding: 8px 16px; font-size: 11px; display: flex; gap: 16px;
    border-top: 1px solid #0f3460;
}
.legend span::before {
    content: ''; display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 4px; vertical-align: middle;
}
.legend .left::before { background: #4ade80; }
.legend .right::before { background: #f59e0b; }
.legend .slam-dot::before { background: #e94560; }

.toggle-row {
    padding: 8px 16px; display: flex; gap: 12px; font-size: 12px;
    border-bottom: 1px solid #0f3460;
}
.toggle-row label { cursor: pointer; display: flex; align-items: center; gap: 4px; }
.toggle-row input { accent-color: #e94560; }
</style>
</head>
<body>
<div class="container">
    <div class="video-panel">
        <div class="video-wrapper" id="video-wrapper">
            <video id="video" preload="auto"></video>
            <canvas id="overlay"></canvas>
        </div>
        <div class="controls">
            <button id="playBtn">Play</button>
            <input type="range" id="scrubber" min="0" max="1000" value="0">
            <span class="time" id="timeDisplay">0:00.00 / 0:00</span>
            <span class="time" id="frameDisplay">F: 0 / 0</span>
        </div>
    </div>
    <div class="side-panel">
        <h3>SLAM Camera Trajectory</h3>
        <div class="slam-view">
            <canvas id="slam-canvas"></canvas>
        </div>
        <div class="toggle-row">
            <label><input type="checkbox" id="showLeft" checked> Left hand</label>
            <label><input type="checkbox" id="showRight" checked> Right hand</label>
            <label><input type="checkbox" id="showBones" checked> Bones</label>
        </div>
        <div class="legend">
            <span class="left">Left hand</span>
            <span class="right">Right hand</span>
            <span class="slam-dot">Camera pos</span>
        </div>
        <h3>Frame Info</h3>
        <div class="info-panel" id="info-panel"></div>
    </div>
</div>

<script>
// MANO hand skeleton connectivity (21 joints)
const HAND_BONES = [
    [0,1],[1,2],[2,3],[3,4],       // thumb
    [0,5],[5,6],[6,7],[7,8],       // index
    [0,9],[9,10],[10,11],[11,12],  // middle
    [0,13],[13,14],[14,15],[15,16],// ring
    [0,17],[17,18],[18,19],[19,20] // pinky
];

const FINGER_COLORS_LEFT = ['#22c55e','#4ade80','#86efac','#bbf7d0','#dcfce7'];
const FINGER_COLORS_RIGHT = ['#f59e0b','#fbbf24','#fcd34d','#fde68a','#fef3c7'];

let DATA = null;
let currentFrame = 0;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const scrubber = document.getElementById('scrubber');
const playBtn = document.getElementById('playBtn');
const timeDisplay = document.getElementById('timeDisplay');
const frameDisplay = document.getElementById('frameDisplay');
const infoPanel = document.getElementById('info-panel');
const slamCanvas = document.getElementById('slam-canvas');
const slamCtx = slamCanvas.getContext('2d');
const showLeft = document.getElementById('showLeft');
const showRight = document.getElementById('showRight');
const showBones = document.getElementById('showBones');

// Load data
fetch('/api/data').then(r => r.json()).then(data => {
    DATA = data;
    video.src = '/video';
    scrubber.max = data.num_frames - 1;
    video.addEventListener('loadedmetadata', () => {
        resizeOverlay();
        drawFrame(0);
    });
});

function resizeOverlay() {
    const wrapper = document.getElementById('video-wrapper');
    const vw = video.videoWidth || DATA.video_width;
    const vh = video.videoHeight || DATA.video_height;
    const wrapW = wrapper.clientWidth;
    const wrapH = wrapper.clientHeight;
    const scale = Math.min(wrapW / vw, wrapH / vh);
    const dispW = vw * scale;
    const dispH = vh * scale;
    const offX = (wrapW - dispW) / 2;
    const offY = (wrapH - dispH) / 2;
    overlay.width = wrapW;
    overlay.height = wrapH;
    overlay._offX = offX;
    overlay._offY = offY;
    overlay._scale = scale;
    overlay._vidW = vw;
    overlay._vidH = vh;

    // Resize SLAM canvas
    const slamRect = slamCanvas.parentElement.getBoundingClientRect();
    slamCanvas.width = slamRect.width * devicePixelRatio;
    slamCanvas.height = slamRect.height * devicePixelRatio;
    slamCanvas.style.width = slamRect.width + 'px';
    slamCanvas.style.height = slamRect.height + 'px';
    slamCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

window.addEventListener('resize', () => { resizeOverlay(); drawFrame(currentFrame); });

function vidToCanvas(px, py) {
    return [
        overlay._offX + px * overlay._scale,
        overlay._offY + py * overlay._scale
    ];
}

function drawHandKeypoints(kp2d, isRight) {
    if (!kp2d) return;
    const colors = isRight ? FINGER_COLORS_RIGHT : FINGER_COLORS_LEFT;
    const baseColor = isRight ? '#f59e0b' : '#4ade80';
    const jointRadius = 3 * overlay._scale;

    // Draw bones
    if (showBones.checked) {
        for (let bi = 0; bi < HAND_BONES.length; bi++) {
            const [a, b] = HAND_BONES[bi];
            const ax = kp2d[a*2], ay = kp2d[a*2+1];
            const bx = kp2d[b*2], by = kp2d[b*2+1];
            if (ax == null || bx == null) continue;
            const [cx1, cy1] = vidToCanvas(ax, ay);
            const [cx2, cy2] = vidToCanvas(bx, by);
            const fingerIdx = Math.floor(bi / 4);
            ctx.strokeStyle = colors[fingerIdx];
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.moveTo(cx1, cy1);
            ctx.lineTo(cx2, cy2);
            ctx.stroke();
        }
    }

    // Draw joints
    ctx.globalAlpha = 1.0;
    for (let j = 0; j < 21; j++) {
        const x = kp2d[j*2], y = kp2d[j*2+1];
        if (x == null) continue;
        const [cx, cy] = vidToCanvas(x, y);
        const fingerIdx = j === 0 ? 0 : Math.floor((j - 1) / 4);
        ctx.fillStyle = colors[fingerIdx];
        ctx.beginPath();
        ctx.arc(cx, cy, j === 0 ? jointRadius * 1.5 : jointRadius, 0, Math.PI * 2);
        ctx.fill();
        // outline
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

function drawSlamTrajectory(frameIdx) {
    if (!DATA.has_slam) return;
    const w = slamCanvas.width / devicePixelRatio;
    const h = slamCanvas.height / devicePixelRatio;
    slamCtx.clearRect(0, 0, w, h);

    // Collect all poses
    const poses = [];
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < DATA.num_frames; i++) {
        const slam = DATA.frames[i].slam;
        if (!slam) continue;
        const tx = slam[0], tz = slam[2];
        poses.push({ x: tx, z: tz, idx: i });
        if (tx < minX) minX = tx;
        if (tx > maxX) maxX = tx;
        if (tz < minZ) minZ = tz;
        if (tz > maxZ) maxZ = tz;
    }
    if (poses.length === 0) return;

    // Add padding
    const rangeX = (maxX - minX) || 0.01;
    const rangeZ = (maxZ - minZ) || 0.01;
    const padding = 30;
    const plotW = w - padding * 2;
    const plotH = h - padding * 2;
    const scale = Math.min(plotW / rangeX, plotH / rangeZ);
    const cx = w / 2;
    const cy = h / 2;
    const midX = (minX + maxX) / 2;
    const midZ = (minZ + maxZ) / 2;

    function toScreen(px, pz) {
        return [cx + (px - midX) * scale, cy + (pz - midZ) * scale];
    }

    // Grid
    slamCtx.strokeStyle = '#2a2a4a';
    slamCtx.lineWidth = 0.5;
    const gridStep = Math.pow(10, Math.floor(Math.log10(rangeX)));
    for (let gx = Math.floor(minX / gridStep) * gridStep; gx <= maxX + gridStep; gx += gridStep) {
        const [sx] = toScreen(gx, 0);
        slamCtx.beginPath(); slamCtx.moveTo(sx, 0); slamCtx.lineTo(sx, h); slamCtx.stroke();
    }
    for (let gz = Math.floor(minZ / gridStep) * gridStep; gz <= maxZ + gridStep; gz += gridStep) {
        const [, sy] = toScreen(0, gz);
        slamCtx.beginPath(); slamCtx.moveTo(0, sy); slamCtx.lineTo(w, sy); slamCtx.stroke();
    }

    // Draw trajectory line
    slamCtx.strokeStyle = '#53d8fb44';
    slamCtx.lineWidth = 1.5;
    slamCtx.beginPath();
    for (let i = 0; i < poses.length; i++) {
        const [sx, sy] = toScreen(poses[i].x, poses[i].z);
        if (i === 0) slamCtx.moveTo(sx, sy);
        else slamCtx.lineTo(sx, sy);
    }
    slamCtx.stroke();

    // Highlight trajectory up to current frame
    slamCtx.strokeStyle = '#53d8fb';
    slamCtx.lineWidth = 2;
    slamCtx.beginPath();
    for (let i = 0; i < poses.length && poses[i].idx <= frameIdx; i++) {
        const [sx, sy] = toScreen(poses[i].x, poses[i].z);
        if (i === 0) slamCtx.moveTo(sx, sy);
        else slamCtx.lineTo(sx, sy);
    }
    slamCtx.stroke();

    // Start marker
    const [sx0, sy0] = toScreen(poses[0].x, poses[0].z);
    slamCtx.fillStyle = '#4ade80';
    slamCtx.beginPath(); slamCtx.arc(sx0, sy0, 5, 0, Math.PI * 2); slamCtx.fill();

    // Current position marker
    const curPose = DATA.frames[frameIdx].slam;
    if (curPose) {
        const [scx, scy] = toScreen(curPose[0], curPose[2]);
        // Direction indicator from quaternion
        const qx = curPose[3], qy = curPose[4], qz = curPose[5], qw = curPose[6];
        const fwdX = 2 * (qx*qz + qw*qy);
        const fwdZ = 1 - 2*(qx*qx + qy*qy);
        const dirLen = 12;
        const [dsx, dsy] = [scx + fwdX * dirLen * scale * 0.5, scy + fwdZ * dirLen * scale * 0.5];

        slamCtx.strokeStyle = '#e94560';
        slamCtx.lineWidth = 2;
        slamCtx.beginPath(); slamCtx.moveTo(scx, scy); slamCtx.lineTo(dsx, dsy); slamCtx.stroke();

        slamCtx.fillStyle = '#e94560';
        slamCtx.beginPath(); slamCtx.arc(scx, scy, 6, 0, Math.PI * 2); slamCtx.fill();
        slamCtx.strokeStyle = '#fff';
        slamCtx.lineWidth = 1.5;
        slamCtx.stroke();
    }

    // Axis labels
    slamCtx.fillStyle = '#888';
    slamCtx.font = '10px system-ui';
    slamCtx.fillText('X (m)', w - 40, h - 5);
    slamCtx.fillText('Z (m)', 5, 12);
    slamCtx.fillText(`${(rangeX*1000).toFixed(0)}mm range`, 5, h - 5);
}

function drawFrame(idx) {
    if (!DATA) return;
    currentFrame = idx;
    const frame = DATA.frames[idx];

    // Clear overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw hand keypoints
    if (showLeft.checked && frame.left_kp2d) {
        drawHandKeypoints(frame.left_kp2d, false);
    }
    if (showRight.checked && frame.right_kp2d) {
        drawHandKeypoints(frame.right_kp2d, true);
    }

    // Draw SLAM
    drawSlamTrajectory(idx);

    // Update info panel
    const ts = frame.t.toFixed(3);
    let html = `<span class="label">Frame:</span> <span class="val">${idx}</span> &nbsp; `;
    html += `<span class="label">Time:</span> <span class="val">${ts}s</span><br>`;

    if (frame.slam) {
        const s = frame.slam;
        html += `<span class="label">Camera:</span> <span class="val">`;
        html += `t=[${s[0].toFixed(4)}, ${s[1].toFixed(4)}, ${s[2].toFixed(4)}]</span><br>`;
        html += `<span class="label">&nbsp;</span> <span class="val">`;
        html += `q=[${s[3].toFixed(4)}, ${s[4].toFixed(4)}, ${s[5].toFixed(4)}, ${s[6].toFixed(4)}]</span><br>`;
    }

    html += `<span class="label">Left hand:</span> `;
    html += frame.left_kp2d
        ? `<span class="detected">detected</span>`
        : `<span class="not-detected">not detected</span>`;
    html += `<br>`;

    html += `<span class="label">Right hand:</span> `;
    html += frame.right_kp2d
        ? `<span class="detected">detected</span>`
        : `<span class="not-detected">not detected</span>`;

    infoPanel.innerHTML = html;

    // Update time display
    const dur = DATA.num_frames / DATA.fps;
    const cur = frame.t;
    const fmt = t => `${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`;
    timeDisplay.textContent = `${fmt(cur)} / ${fmt(dur)}`;
    frameDisplay.textContent = `F: ${idx} / ${DATA.num_frames - 1}`;
}

// Video sync
video.addEventListener('timeupdate', () => {
    if (!DATA) return;
    const t = video.currentTime;
    // Find closest frame
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < DATA.num_frames; i++) {
        const d = Math.abs(DATA.frames[i].t - t);
        if (d < bestDist) { bestDist = d; best = i; }
        if (DATA.frames[i].t > t + 0.1) break; // early exit
    }
    scrubber.value = best;
    drawFrame(best);
});

// Scrubber
scrubber.addEventListener('input', () => {
    const idx = parseInt(scrubber.value);
    if (DATA) {
        video.currentTime = DATA.frames[idx].t;
        drawFrame(idx);
    }
});

// Play/pause
playBtn.addEventListener('click', () => {
    if (video.paused) { video.play(); playBtn.textContent = 'Pause'; }
    else { video.pause(); playBtn.textContent = 'Play'; }
});

video.addEventListener('play', () => playBtn.textContent = 'Pause');
video.addEventListener('pause', () => playBtn.textContent = 'Play');

// Toggle checkboxes
showLeft.addEventListener('change', () => drawFrame(currentFrame));
showRight.addEventListener('change', () => drawFrame(currentFrame));
showBones.addEventListener('change', () => drawFrame(currentFrame));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
    if (!DATA) return;
    if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
    if (e.code === 'ArrowLeft') {
        e.preventDefault();
        const idx = Math.max(0, currentFrame - (e.shiftKey ? 10 : 1));
        scrubber.value = idx;
        video.currentTime = DATA.frames[idx].t;
        drawFrame(idx);
    }
    if (e.code === 'ArrowRight') {
        e.preventDefault();
        const idx = Math.min(DATA.num_frames - 1, currentFrame + (e.shiftKey ? 10 : 1));
        scrubber.value = idx;
        video.currentTime = DATA.frames[idx].t;
        drawFrame(idx);
    }
});
</script>
</body>
</html>
"""


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the visualizer HTML, video, and data API."""

    dataset = None  # set by main()
    video_path = None

    def do_HEAD(self):
        """Handle HEAD requests (browser sends these for video)."""
        path = urlparse(self.path).path
        if path == "/video" and self.video_path and os.path.exists(self.video_path):
            file_size = os.path.getsize(self.video_path)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # Send data without video_path (served separately)
            data = {k: v for k, v in self.dataset.items() if k != "video_path"}
            self.wfile.write(json.dumps(data).encode())

        elif path == "/video":
            self._serve_video()

        else:
            self.send_error(404)

    def _serve_video(self):
        vp = self.video_path
        if not vp or not os.path.exists(vp):
            self.send_error(404, "Video not found")
            return

        file_size = os.path.getsize(vp)
        range_header = self.headers.get("Range")

        if range_header:
            # Handle range requests for video seeking
            range_val = range_header.strip().split("=")[1]
            start, end = range_val.split("-")
            start = int(start)
            end = int(end) if end else file_size - 1
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(vp, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(vp, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def log_message(self, format, *args):
        # Suppress noisy request logs except errors
        if args and "200" not in str(args[0]) and "206" not in str(args[0]):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Browser-based LeRobot dataset visualizer")
    parser.add_argument("dataset", help="Path to LeRobot dataset directory")
    parser.add_argument("--port", type=int, default=8888, help="HTTP server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"  {dataset['num_frames']} frames, {dataset['fps']:.1f} fps")
    print(f"  Video: {dataset['video_path']}")
    print(f"  SLAM: {'yes' if dataset['has_slam'] else 'no'}")
    print(f"  Left hand: {'yes' if dataset['has_left_hand'] else 'no'}")
    print(f"  Right hand: {'yes' if dataset['has_right_hand'] else 'no'}")

    VisualizerHandler.dataset = dataset
    VisualizerHandler.video_path = dataset["video_path"]

    server = HTTPServer((args.host, args.port), VisualizerHandler)
    print(f"\nVisualizer running at http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()


if __name__ == "__main__":
    main()
