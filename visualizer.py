#!/usr/bin/env python3
"""Browser-based visualizer for LeRobot v3.0 datasets.

Displays video with hand pose overlay and 3D scene (Three.js) with
camera trajectory and 3D hand poses.

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

    with open(dataset_dir / "meta" / "info.json") as f:
        info = json.load(f)

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
        for mp4 in dataset_dir.rglob("*.mp4"):
            video_path = str(mp4)
            break

    # Video dimensions
    video_feat = info["features"].get("observation.video", {})
    video_shape = video_feat.get("shape", [1080, 1920, 3])
    vid_h, vid_w = video_shape[0], video_shape[1]

    has_slam = "observation.slam.pose" in table.column_names
    has_left = "observation.hand.left.detected" in table.column_names
    has_right = "observation.hand.right.detected" in table.column_names

    frames = []
    for i in range(num_frames):
        frame = {
            "t": round(table.column("timestamp")[i].as_py(), 6),
        }

        if has_slam:
            pose = table.column("observation.slam.pose")[i].as_py()
            frame["slam"] = [round(float(v), 6) for v in pose]

        for side in ["left", "right"]:
            det_col = f"observation.hand.{side}.detected"
            kp2d_col = f"observation.hand.{side}.keypoints_2d"
            kp3d_col = f"observation.hand.{side}.keypoints_3d"

            if det_col not in table.column_names:
                continue

            detected = table.column(det_col)[i].as_py()
            if detected > 0.5:
                # 2D keypoints (pixel coords)
                kp2d_flat = table.column(kp2d_col)[i].as_py()
                kp2d_pixels = []
                for j in range(0, len(kp2d_flat), 2):
                    px, py = kp2d_flat[j], kp2d_flat[j + 1]
                    if math.isnan(px):
                        kp2d_pixels.extend([None, None])
                    else:
                        kp2d_pixels.extend([round(px, 1), round(py, 1)])
                frame[f"{side}_kp2d"] = kp2d_pixels

                # 3D keypoints (camera frame)
                if kp3d_col in table.column_names:
                    kp3d_flat = table.column(kp3d_col)[i].as_py()
                    kp3d_rounded = []
                    for v in kp3d_flat:
                        fv = float(v)
                        kp3d_rounded.append(None if math.isnan(fv) else round(fv, 5))
                    frame[f"{side}_kp3d"] = kp3d_rounded
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
    width: 420px; background: #16213e; display: flex; flex-direction: column;
    border-left: 1px solid #0f3460;
}

.side-panel h3 {
    padding: 12px 16px 8px; font-size: 14px; color: #e94560;
    border-bottom: 1px solid #0f3460;
}

.scene-view {
    flex: 1; position: relative; min-height: 300px;
}

#scene-container { width: 100%; height: 100%; }

.info-panel {
    padding: 12px 16px; font-size: 12px; line-height: 1.6;
    border-top: 1px solid #0f3460; max-height: 180px; overflow-y: auto;
}

.info-panel .label { color: #888; }
.info-panel .val { color: #53d8fb; font-variant-numeric: tabular-nums; }
.info-panel .detected { color: #4ade80; }
.info-panel .not-detected { color: #f87171; }

.toggle-row {
    padding: 8px 16px; display: flex; gap: 12px; font-size: 12px;
    border-bottom: 1px solid #0f3460; flex-wrap: wrap;
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
        <h3>3D Scene</h3>
        <div class="scene-view">
            <div id="scene-container"></div>
        </div>
        <div class="toggle-row">
            <label><input type="checkbox" id="showLeft" checked> Left hand</label>
            <label><input type="checkbox" id="showRight" checked> Right hand</label>
            <label><input type="checkbox" id="showBones" checked> Bones</label>
            <label><input type="checkbox" id="show3DHands" checked> 3D hands</label>
            <label><input type="checkbox" id="followCam" checked> Follow camera</label>
        </div>
        <h3>Frame Info</h3>
        <div class="info-panel" id="info-panel"></div>
    </div>
</div>

<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.171.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.171.0/examples/jsm/"
    }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// MANO hand skeleton connectivity (21 joints)
const HAND_BONES = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
];

const FINGER_COLORS_LEFT = ['#22c55e','#4ade80','#86efac','#bbf7d0','#dcfce7'];
const FINGER_COLORS_RIGHT = ['#f59e0b','#fbbf24','#fcd34d','#fde68a','#fef3c7'];

let DATA = null;
let currentFrame = 0;
let lastDrawnFrame = -1;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const scrubber = document.getElementById('scrubber');
const playBtn = document.getElementById('playBtn');
const timeDisplay = document.getElementById('timeDisplay');
const frameDisplay = document.getElementById('frameDisplay');
const infoPanel = document.getElementById('info-panel');
const showLeft = document.getElementById('showLeft');
const showRight = document.getElementById('showRight');
const showBones = document.getElementById('showBones');
const show3DHands = document.getElementById('show3DHands');
const followCam = document.getElementById('followCam');

// --- Three.js scene setup ---
const sceneContainer = document.getElementById('scene-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);

const camera3D = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
camera3D.position.set(0, 2, 2);
camera3D.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
sceneContainer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera3D, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 0);

// Grid
const grid = new THREE.GridHelper(10, 20, 0x2a2a4a, 0x1a1a3a);
scene.add(grid);

// Axis helper
const axes = new THREE.AxesHelper(0.3);
scene.add(axes);

// Ambient + directional light
scene.add(new THREE.AmbientLight(0x404060, 2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(2, 5, 3);
scene.add(dirLight);

// --- Trajectory line ---
let trajectoryLine = null;
let trajectoryHighlight = null;
let trajectoryPositions = [];

function buildTrajectory() {
    if (!DATA || !DATA.has_slam) return;

    const positions = [];
    for (let i = 0; i < DATA.num_frames; i++) {
        const slam = DATA.frames[i].slam;
        if (!slam) continue;
        positions.push(new THREE.Vector3(slam[0], slam[1], slam[2]));
    }
    trajectoryPositions = positions;

    if (positions.length < 2) return;

    // Full trajectory (dim)
    const geom = new THREE.BufferGeometry().setFromPoints(positions);
    const mat = new THREE.LineBasicMaterial({ color: 0x53d8fb, opacity: 0.2, transparent: true });
    trajectoryLine = new THREE.Line(geom, mat);
    scene.add(trajectoryLine);

    // Highlighted portion (bright)
    const highlightGeom = new THREE.BufferGeometry().setFromPoints(positions);
    const highlightMat = new THREE.LineBasicMaterial({ color: 0x53d8fb, linewidth: 2 });
    trajectoryHighlight = new THREE.Line(highlightGeom, highlightMat);
    scene.add(trajectoryHighlight);

    // Start marker
    const startGeo = new THREE.SphereGeometry(0.015, 8, 8);
    const startMat = new THREE.MeshBasicMaterial({ color: 0x4ade80 });
    const startMarker = new THREE.Mesh(startGeo, startMat);
    startMarker.position.copy(positions[0]);
    scene.add(startMarker);
}

// --- Camera frustum ---
const frustumGroup = new THREE.Group();
scene.add(frustumGroup);

function buildCameraFrustum() {
    // Simple wireframe frustum
    const s = 0.04; // near plane half-size
    const d = 0.08; // depth
    const ar = 16/9;
    const vertices = new Float32Array([
        // near plane corners
        -s*ar, -s, -0.001,   s*ar, -s, -0.001,
         s*ar, -s, -0.001,   s*ar,  s, -0.001,
         s*ar,  s, -0.001,  -s*ar,  s, -0.001,
        -s*ar,  s, -0.001,  -s*ar, -s, -0.001,
        // far plane corners
        -d*ar, -d, -d*2,   d*ar, -d, -d*2,
         d*ar, -d, -d*2,   d*ar,  d, -d*2,
         d*ar,  d, -d*2,  -d*ar,  d, -d*2,
        -d*ar,  d, -d*2,  -d*ar, -d, -d*2,
        // connecting edges
        -s*ar, -s, -0.001,  -d*ar, -d, -d*2,
         s*ar, -s, -0.001,   d*ar, -d, -d*2,
         s*ar,  s, -0.001,   d*ar,  d, -d*2,
        -s*ar,  s, -0.001,  -d*ar,  d, -d*2,
    ]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0xe94560 });
    const lines = new THREE.LineSegments(geom, mat);
    frustumGroup.add(lines);

    // Small sphere at camera origin
    const sphereGeo = new THREE.SphereGeometry(0.012, 8, 8);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0xe94560 });
    frustumGroup.add(new THREE.Mesh(sphereGeo, sphereMat));
}
buildCameraFrustum();

// --- 3D hand skeleton objects ---
const handGroups = { left: new THREE.Group(), right: new THREE.Group() };
scene.add(handGroups.left);
scene.add(handGroups.right);

const handJointMeshes = { left: [], right: [] };
const handBoneLines = { left: null, right: null };

function buildHandSkeleton(side) {
    const color = side === 'right' ? 0xf59e0b : 0x4ade80;
    const group = handGroups[side];

    // 21 joint spheres
    const sphereGeo = new THREE.SphereGeometry(0.005, 6, 6);
    const mat = new THREE.MeshBasicMaterial({ color });
    for (let j = 0; j < 21; j++) {
        const mesh = new THREE.Mesh(sphereGeo, mat);
        mesh.visible = false;
        group.add(mesh);
        handJointMeshes[side].push(mesh);
    }

    // Bone lines (20 bones * 2 points = 40 vertices)
    const positions = new Float32Array(HAND_BONES.length * 2 * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const lineMat = new THREE.LineBasicMaterial({ color, opacity: 0.7, transparent: true });
    handBoneLines[side] = new THREE.LineSegments(geom, lineMat);
    handBoneLines[side].visible = false;
    group.add(handBoneLines[side]);
}
buildHandSkeleton('left');
buildHandSkeleton('right');

function updateHand3D(side, kp3d, slamPose) {
    const group = handGroups[side];
    const joints = handJointMeshes[side];
    const boneLine = handBoneLines[side];

    if (!kp3d || !slamPose || !show3DHands.checked) {
        joints.forEach(m => m.visible = false);
        boneLine.visible = false;
        return;
    }

    // WiLoR 3D keypoints are in WiLoR's camera frame with z~40m (weak-perspective depth).
    // We center on the wrist, keep relative joint structure, and place in front of the SLAM camera.
    const camPos = new THREE.Vector3(slamPose[0], slamPose[1], slamPose[2]);
    const camQuat = new THREE.Quaternion(slamPose[3], slamPose[4], slamPose[5], slamPose[6]);

    // Read all joints, find wrist (joint 0) as origin
    const raw = [];
    for (let j = 0; j < 21; j++) {
        const x = kp3d[j * 3], y = kp3d[j * 3 + 1], z = kp3d[j * 3 + 2];
        if (x == null || isNaN(x)) { raw.push(null); continue; }
        raw.push(new THREE.Vector3(x, y, z));
    }
    if (!raw[0]) {
        joints.forEach(m => m.visible = false);
        boneLine.visible = false;
        return;
    }

    const wrist = raw[0].clone();

    // Place hand 0.3m in front of camera (along camera -Z in SLAM convention)
    // Offset left/right hands slightly so they don't overlap
    const offset = new THREE.Vector3(side === 'left' ? -0.06 : 0.06, -0.05, -0.3);
    offset.applyQuaternion(camQuat);
    const handOrigin = camPos.clone().add(offset);

    const worldPositions = [];
    for (let j = 0; j < 21; j++) {
        if (!raw[j]) {
            joints[j].visible = false;
            worldPositions.push(null);
            continue;
        }
        // Relative to wrist, keeping WiLoR's scale (units ~ meters)
        const rel = raw[j].clone().sub(wrist);
        // Rotate relative position into SLAM world frame
        rel.applyQuaternion(camQuat);
        const world = handOrigin.clone().add(rel);
        joints[j].position.copy(world);
        joints[j].visible = true;
        worldPositions.push(world);
    }

    // Update bone lines
    const posAttr = boneLine.geometry.getAttribute('position');
    let anyVisible = false;
    for (let bi = 0; bi < HAND_BONES.length; bi++) {
        const [a, b] = HAND_BONES[bi];
        const pa = worldPositions[a], pb = worldPositions[b];
        if (pa && pb) {
            posAttr.setXYZ(bi * 2, pa.x, pa.y, pa.z);
            posAttr.setXYZ(bi * 2 + 1, pb.x, pb.y, pb.z);
            anyVisible = true;
        } else {
            posAttr.setXYZ(bi * 2, 0, 0, 0);
            posAttr.setXYZ(bi * 2 + 1, 0, 0, 0);
        }
    }
    posAttr.needsUpdate = true;
    boneLine.visible = anyVisible;
}

// --- Update trajectory highlight ---
function updateTrajectoryHighlight(frameIdx) {
    if (!trajectoryHighlight || trajectoryPositions.length === 0) return;
    // Show trajectory up to current frame
    const count = Math.min(frameIdx + 1, trajectoryPositions.length);
    trajectoryHighlight.geometry.setDrawRange(0, count);
}

// --- Update frustum pose ---
function updateFrustum(slamPose) {
    if (!slamPose) {
        frustumGroup.visible = false;
        return;
    }
    frustumGroup.visible = true;
    frustumGroup.position.set(slamPose[0], slamPose[1], slamPose[2]);
    frustumGroup.quaternion.set(slamPose[3], slamPose[4], slamPose[5], slamPose[6]);
}

// --- Resize Three.js ---
function resizeScene() {
    const rect = sceneContainer.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    renderer.setPixelRatio(devicePixelRatio);
    camera3D.aspect = w / h;
    camera3D.updateProjectionMatrix();
}

// --- 2D overlay ---
function resizeOverlay() {
    if (!DATA) return;
    const wrapper = document.getElementById('video-wrapper');
    const vw = video.videoWidth || DATA.video_width;
    const vh = video.videoHeight || DATA.video_height;
    const wrapW = wrapper.clientWidth;
    const wrapH = wrapper.clientHeight;
    const scale = Math.min(wrapW / vw, wrapH / vh);
    const dispW = vw * scale;
    const dispH = vh * scale;
    overlay.width = wrapW;
    overlay.height = wrapH;
    overlay._offX = (wrapW - dispW) / 2;
    overlay._offY = (wrapH - dispH) / 2;
    overlay._scale = scale;
}

function vidToCanvas(px, py) {
    return [
        overlay._offX + px * overlay._scale,
        overlay._offY + py * overlay._scale
    ];
}

function drawHandKeypoints2D(kp2d, isRight) {
    if (!kp2d) return;
    const colors = isRight ? FINGER_COLORS_RIGHT : FINGER_COLORS_LEFT;
    const jointRadius = 3 * overlay._scale;

    if (showBones.checked) {
        for (let bi = 0; bi < HAND_BONES.length; bi++) {
            const [a, b] = HAND_BONES[bi];
            const ax = kp2d[a*2], ay = kp2d[a*2+1];
            const bx = kp2d[b*2], by = kp2d[b*2+1];
            if (ax == null || bx == null) continue;
            const [cx1, cy1] = vidToCanvas(ax, ay);
            const [cx2, cy2] = vidToCanvas(bx, by);
            ctx.strokeStyle = colors[Math.floor(bi / 4)];
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.moveTo(cx1, cy1);
            ctx.lineTo(cx2, cy2);
            ctx.stroke();
        }
    }

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
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// --- Frame drawing ---
function drawFrame(idx) {
    if (!DATA || idx < 0 || idx >= DATA.num_frames) return;
    currentFrame = idx;
    const frame = DATA.frames[idx];

    // 2D overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (showLeft.checked && frame.left_kp2d) drawHandKeypoints2D(frame.left_kp2d, false);
    if (showRight.checked && frame.right_kp2d) drawHandKeypoints2D(frame.right_kp2d, true);

    // 3D scene updates
    updateFrustum(frame.slam);
    updateTrajectoryHighlight(idx);
    updateHand3D('left', frame.left_kp3d || null, frame.slam);
    updateHand3D('right', frame.right_kp3d || null, frame.slam);

    // Follow camera mode
    if (followCam.checked && frame.slam) {
        const target = new THREE.Vector3(frame.slam[0], frame.slam[1], frame.slam[2]);
        controls.target.lerp(target, 0.15);
    }

    // Info panel
    const ts = frame.t.toFixed(3);
    let html = `<span class="label">Frame:</span> <span class="val">${idx}</span> &nbsp; `;
    html += `<span class="label">Time:</span> <span class="val">${ts}s</span><br>`;
    if (frame.slam) {
        const s = frame.slam;
        html += `<span class="label">Pos:</span> <span class="val">[${s[0].toFixed(4)}, ${s[1].toFixed(4)}, ${s[2].toFixed(4)}]</span><br>`;
        html += `<span class="label">Quat:</span> <span class="val">[${s[3].toFixed(3)}, ${s[4].toFixed(3)}, ${s[5].toFixed(3)}, ${s[6].toFixed(3)}]</span><br>`;
    }
    html += `<span class="label">Left:</span> ${frame.left_kp2d ? '<span class="detected">detected</span>' : '<span class="not-detected">-</span>'} &nbsp; `;
    html += `<span class="label">Right:</span> ${frame.right_kp2d ? '<span class="detected">detected</span>' : '<span class="not-detected">-</span>'}`;
    infoPanel.innerHTML = html;

    // Controls display
    const dur = DATA.num_frames / DATA.fps;
    const cur = frame.t;
    const fmt = t => `${Math.floor(t/60)}:${(t%60).toFixed(2).padStart(5,'0')}`;
    timeDisplay.textContent = `${fmt(cur)} / ${fmt(dur)}`;
    frameDisplay.textContent = `F: ${idx} / ${DATA.num_frames - 1}`;
}

// --- requestAnimationFrame sync loop (fixes lag) ---
function animationLoop() {
    requestAnimationFrame(animationLoop);

    if (DATA && !video.paused) {
        // Compute frame from video time
        const t = video.currentTime;
        const idx = Math.min(Math.round(t * DATA.fps), DATA.num_frames - 1);
        if (idx !== lastDrawnFrame) {
            scrubber.value = idx;
            drawFrame(idx);
            lastDrawnFrame = idx;
        }
    }

    controls.update();
    renderer.render(scene, camera3D);
}

// --- Load data ---
fetch('/api/data').then(r => r.json()).then(data => {
    DATA = data;
    video.src = '/video';
    scrubber.max = data.num_frames - 1;
    video.addEventListener('loadedmetadata', () => {
        resizeOverlay();
        resizeScene();
        buildTrajectory();
        drawFrame(0);
        // Center orbit on trajectory midpoint
        if (trajectoryPositions.length > 0) {
            const mid = trajectoryPositions[Math.floor(trajectoryPositions.length / 2)];
            controls.target.copy(mid);
            camera3D.position.set(mid.x + 1, mid.y + 1.5, mid.z + 1);
        }
    });
});

window.addEventListener('resize', () => {
    resizeOverlay();
    resizeScene();
    drawFrame(currentFrame);
});

// Scrubber
scrubber.addEventListener('input', () => {
    const idx = parseInt(scrubber.value);
    if (DATA) {
        video.currentTime = DATA.frames[idx].t;
        drawFrame(idx);
        lastDrawnFrame = idx;
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
[showLeft, showRight, showBones, show3DHands, followCam].forEach(el =>
    el.addEventListener('change', () => drawFrame(currentFrame))
);

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
        lastDrawnFrame = idx;
    }
    if (e.code === 'ArrowRight') {
        e.preventDefault();
        const idx = Math.min(DATA.num_frames - 1, currentFrame + (e.shiftKey ? 10 : 1));
        scrubber.value = idx;
        video.currentTime = DATA.frames[idx].t;
        drawFrame(idx);
        lastDrawnFrame = idx;
    }
});

// Start animation loop
requestAnimationFrame(animationLoop);
</script>
</body>
</html>
"""


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the visualizer HTML, video, and data API."""

    dataset = None
    video_path = None

    def do_HEAD(self):
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
