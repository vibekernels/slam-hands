import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  View,
  Text,
  Pressable,
  AppState,
  Platform,
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useFrameProcessor,
  runAtTargetFps,
} from "react-native-vision-camera";
import { Worklets } from "react-native-worklets-core";
import * as Speech from "expo-speech";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { detectHandGesture } from "@/lib/handGesture";
import { analyzeGesture, thumbIndexDistance, type GestureState, type HandForGesture } from "@/lib/gesture";
import { saveClip, updateClipMeta } from "@/lib/storage";
import { uploadClip } from "@/lib/upload";

// ── Constants ────────────────────────────────────────────────────────
const GESTURE_HOLD_MS = 750; // hold OK gesture this long to trigger
const TOGGLE_COOLDOWN_MS = 3000; // minimum gap between toggles
const DETECTION_FPS = 3; // hand detection frequency

export default function CameraScreen() {
  const router = useRouter();
  const cameraRef = useRef<Camera>(null);

  // ── Permissions ──────────────────────────────────────────────────
  const [cameraPermission, setCameraPermission] = useState<string | null>(null);
  const [micPermission, setMicPermission] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const cam = await Camera.requestCameraPermission();
      const mic = await Camera.requestMicrophonePermission();
      setCameraPermission(cam);
      setMicPermission(mic);
    })();
  }, []);

  // ── Camera device & format ───────────────────────────────────────
  const device = useCameraDevice("back", {
    physicalDevices: ["ultra-wide-angle-camera"],
  });

  const format = useCameraFormat(device, [
    { videoResolution: { width: 1920, height: 1080 } },
    { fps: 30 },
  ]);

  // ── App state (pause camera when backgrounded) ───────────────────
  const [isActive, setIsActive] = useState(true);
  useEffect(() => {
    const sub = AppState.addEventListener("change", (state) => {
      setIsActive(state === "active");
    });
    return () => sub.remove();
  }, []);

  // ── Recording state ──────────────────────────────────────────────
  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);
  const recordingStartTime = useRef(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Gesture state ────────────────────────────────────────────────
  const [gestureInfo, setGestureInfo] = useState<GestureState>({
    anyHandOk: false,
    leftOk: false,
    rightOk: false,
    handsDetected: 0,
  });
  const bothOkSince = useRef<number | null>(null);
  const lastToggleTime = useRef(0);
  const [gestureProgress, setGestureProgress] = useState(0);

  // ── Voice announcements ──────────────────────────────────────────
  const speak = useCallback((text: string) => {
    Speech.speak(text, {
      language: "en-US",
      rate: 1.1,
      pitch: 1.0,
    });
  }, []);

  // ── Recording control ────────────────────────────────────────────
  const startRecording = useCallback(() => {
    if (isRecordingRef.current || !cameraRef.current) return;

    isRecordingRef.current = true;
    setIsRecording(true);
    recordingStartTime.current = Date.now();
    setRecordingDuration(0);

    // Update duration display every second
    durationInterval.current = setInterval(() => {
      setRecordingDuration(
        (Date.now() - recordingStartTime.current) / 1000
      );
    }, 1000);

    speak("Recording");

    cameraRef.current.startRecording({
      onRecordingFinished: async (video) => {
        const duration = (Date.now() - recordingStartTime.current) / 1000;
        try {
          const clip = await saveClip(`file://${video.path}`, duration);
          // Auto-upload in background
          uploadClip(clip.uri, clip.filename).then(async (result) => {
            await updateClipMeta(clip.filename, {
              uploading: false,
              uploaded: result.success,
              error: result.error,
            });
            if (result.success) {
              speak("Upload complete");
            }
          });
          await updateClipMeta(clip.filename, { uploading: true });
        } catch (err) {
          console.error("Failed to save clip:", err);
        }
      },
      onRecordingError: (error) => {
        console.error("Recording error:", error);
        isRecordingRef.current = false;
        setIsRecording(false);
        if (durationInterval.current) clearInterval(durationInterval.current);
        speak("Recording error");
      },
    });
  }, [speak]);

  const stopRecording = useCallback(async () => {
    if (!isRecordingRef.current || !cameraRef.current) return;

    isRecordingRef.current = false;
    setIsRecording(false);
    if (durationInterval.current) clearInterval(durationInterval.current);

    speak("Stopped");

    await cameraRef.current.stopRecording();
  }, [speak]);

  const toggleRecording = useCallback(() => {
    const now = Date.now();
    if (now - lastToggleTime.current < TOGGLE_COOLDOWN_MS) return;
    lastToggleTime.current = now;

    if (isRecordingRef.current) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [startRecording, stopRecording]);

  // ── Parse raw native landmark arrays into {x, y} objects ──────────
  const parseRawHands = useCallback((rawHands: any[]): HandForGesture[] => {
    const lm = (arr: number[]) => ({ x: arr[0], y: arr[1] });
    return rawHands.map((h: any) => ({
      chirality: h.chirality,
      thumbTip: lm(h.thumbTip),
      thumbIP: lm(h.thumbIP),
      indexTip: lm(h.indexTip),
      indexDIP: lm(h.indexDIP),
      indexMCP: lm(h.indexMCP),
      middleTip: lm(h.middleTip),
      middleMCP: lm(h.middleMCP),
      ringTip: lm(h.ringTip),
      ringMCP: lm(h.ringMCP),
      littleTip: lm(h.littleTip),
      littleMCP: lm(h.littleMCP),
      wrist: lm(h.wrist),
    }));
  }, []);

  // ── Gesture processing (called from frame processor via runOnJS) ─
  const onGestureDetected = useCallback(
    (result: any) => {
      if (!result || !result.hands) {
        setGestureInfo({
          anyHandOk: false,
          leftOk: false,
          rightOk: false,
          handsDetected: 0,
        });
        bothOkSince.current = null;
        setGestureProgress(0);
        return;
      }

      const hands = parseRawHands(result.hands);
      const state = analyzeGesture(hands);
      setGestureInfo(state);

      if (state.anyHandOk) {
        const now = Date.now();
        if (!bothOkSince.current) {
          bothOkSince.current = now;
        }
        const elapsed = now - bothOkSince.current;
        setGestureProgress(Math.min(elapsed / GESTURE_HOLD_MS, 1));

        if (elapsed >= GESTURE_HOLD_MS) {
          toggleRecording();
          bothOkSince.current = null;
          setGestureProgress(0);
        }
      } else {
        bothOkSince.current = null;
        setGestureProgress(0);
      }
    },
    [toggleRecording]
  );

  // ── Frame processor ──────────────────────────────────────────────
  const onGestureJS = Worklets.createRunOnJS(onGestureDetected);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";
      runAtTargetFps(DETECTION_FPS, () => {
        "worklet";
        const result = detectHandGesture(frame);
        onGestureJS(result);
      });
    },
    [onGestureJS]
  );

  // ── Fallback: tap to toggle (for testing / Android) ──────────────
  const handleTapToggle = useCallback(() => {
    toggleRecording();
  }, [toggleRecording]);

  // ── Render ───────────────────────────────────────────────────────
  if (cameraPermission !== "granted" || micPermission !== "granted") {
    return (
      <View style={styles.centered}>
        <Ionicons name="videocam-off" size={64} color="#666" />
        <Text style={styles.permissionText}>
          Camera and microphone permissions are required.
        </Text>
        <Pressable
          style={styles.permissionButton}
          onPress={async () => {
            const cam = await Camera.requestCameraPermission();
            const mic = await Camera.requestMicrophonePermission();
            setCameraPermission(cam);
            setMicPermission(mic);
          }}
        >
          <Text style={styles.permissionButtonText}>Grant Permissions</Text>
        </Pressable>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permissionText}>
          No ultra-wide camera found on this device.
        </Text>
      </View>
    );
  }

  const formatDuration = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={isActive}
        video={true}
        audio={true}
        videoHdr={false} // non-HDR as requested
        fps={30}
        frameProcessor={frameProcessor}
        onError={(e) => console.error("Camera error:", e)}
      />

      {/* ── Recording indicator ─────────────────────────────────── */}
      {isRecording && (
        <View style={styles.recordingBanner}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>
            REC {formatDuration(recordingDuration)}
          </Text>
        </View>
      )}

      {/* ── Gesture status overlay ──────────────────────────────── */}
      <View style={styles.gestureOverlay}>
        <Text style={styles.gestureText}>
          {gestureInfo.handsDetected === 0
            ? "Show hands"
            : gestureInfo.anyHandOk
              ? "Hold..."
              : `Hands: ${gestureInfo.handsDetected} | L:${gestureInfo.leftOk ? "OK" : "--"} R:${gestureInfo.rightOk ? "OK" : "--"}${gestureInfo.debugDist ? ` | d:${gestureInfo.debugDist}` : ""}`}
        </Text>
        {gestureProgress > 0 && (
          <View style={styles.progressBarBg}>
            <View
              style={[
                styles.progressBarFill,
                { width: `${gestureProgress * 100}%` },
              ]}
            />
          </View>
        )}
      </View>

      {/* ── Tap fallback + settings ─────────────────────────────── */}
      <View style={styles.bottomBar}>
        <Pressable
          style={[
            styles.recordButton,
            isRecording && styles.recordButtonActive,
          ]}
          onPress={handleTapToggle}
        >
          <View
            style={[
              styles.recordButtonInner,
              isRecording && styles.recordButtonInnerActive,
            ]}
          />
        </Pressable>
      </View>

      {/* ── Settings button ─────────────────────────────────────── */}
      <Pressable
        style={styles.settingsButton}
        onPress={() => router.push("/settings")}
      >
        <Ionicons name="settings-outline" size={24} color="#fff" />
      </Pressable>
    </View>
  );
}

// ── Styles ─────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  centered: {
    flex: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  permissionText: {
    color: "#aaa",
    fontSize: 16,
    textAlign: "center",
    marginTop: 16,
  },
  permissionButton: {
    marginTop: 24,
    backgroundColor: "#ff3b30",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
  },
  permissionButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },

  // Recording banner
  recordingBanner: {
    position: "absolute",
    top: Platform.OS === "ios" ? 60 : 40,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 8,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: "#ff3b30",
  },
  recordingText: {
    color: "#ff3b30",
    fontSize: 18,
    fontWeight: "700",
    fontVariant: ["tabular-nums"],
  },

  // Gesture overlay
  gestureOverlay: {
    position: "absolute",
    top: Platform.OS === "ios" ? 100 : 80,
    left: 16,
    right: 16,
    alignItems: "center",
  },
  gestureText: {
    color: "#fff",
    fontSize: 14,
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: "hidden",
  },
  progressBarBg: {
    width: 120,
    height: 4,
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: 2,
    marginTop: 8,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    backgroundColor: "#34c759",
    borderRadius: 2,
  },

  // Bottom bar with record button
  bottomBar: {
    position: "absolute",
    bottom: 100,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  recordButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
  },
  recordButtonActive: {
    borderColor: "#ff3b30",
  },
  recordButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "#ff3b30",
  },
  recordButtonInnerActive: {
    width: 28,
    height: 28,
    borderRadius: 6,
    backgroundColor: "#ff3b30",
  },

  // Settings
  settingsButton: {
    position: "absolute",
    top: Platform.OS === "ios" ? 60 : 40,
    right: 16,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "center",
    alignItems: "center",
  },
});
