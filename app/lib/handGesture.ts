/**
 * VisionCamera frame processor plugin wrapper for hand gesture detection.
 * The native plugin (iOS) uses Apple Vision's VNDetectHumanHandPoseRequest.
 */
import { VisionCameraProxy, type Frame } from "react-native-vision-camera";

const plugin = VisionCameraProxy.initFrameProcessorPlugin("detectHandGesture", {});

export interface HandLandmark {
  x: number; // normalized [0,1], origin bottom-left (Vision coords)
  y: number;
  confidence: number;
}

export interface DetectedHand {
  chirality: "left" | "right";
  thumbTip: HandLandmark;
  thumbIP: HandLandmark;
  thumbMP: HandLandmark;
  thumbCMC: HandLandmark;
  indexTip: HandLandmark;
  indexDIP: HandLandmark;
  indexPIP: HandLandmark;
  indexMCP: HandLandmark;
  middleTip: HandLandmark;
  middleDIP: HandLandmark;
  middlePIP: HandLandmark;
  middleMCP: HandLandmark;
  ringTip: HandLandmark;
  ringDIP: HandLandmark;
  ringPIP: HandLandmark;
  ringMCP: HandLandmark;
  littleTip: HandLandmark;
  littleDIP: HandLandmark;
  littlePIP: HandLandmark;
  littleMCP: HandLandmark;
  wrist: HandLandmark;
}

export interface HandGestureResult {
  hands: DetectedHand[];
}

/** Parse a [x, y, confidence] array from native into a HandLandmark. */
function parseLandmark(raw: number[]): HandLandmark {
  return { x: raw[0], y: raw[1], confidence: raw[2] };
}

/** Parse a raw native hand dict into a typed DetectedHand. */
function parseHand(raw: Record<string, any>): DetectedHand {
  return {
    chirality: raw.chirality,
    thumbTip: parseLandmark(raw.thumbTip),
    thumbIP: parseLandmark(raw.thumbIP),
    thumbMP: parseLandmark(raw.thumbMP),
    thumbCMC: parseLandmark(raw.thumbCMC),
    indexTip: parseLandmark(raw.indexTip),
    indexDIP: parseLandmark(raw.indexDIP),
    indexPIP: parseLandmark(raw.indexPIP),
    indexMCP: parseLandmark(raw.indexMCP),
    middleTip: parseLandmark(raw.middleTip),
    middleDIP: parseLandmark(raw.middleDIP),
    middlePIP: parseLandmark(raw.middlePIP),
    middleMCP: parseLandmark(raw.middleMCP),
    ringTip: parseLandmark(raw.ringTip),
    ringDIP: parseLandmark(raw.ringDIP),
    ringPIP: parseLandmark(raw.ringPIP),
    ringMCP: parseLandmark(raw.ringMCP),
    littleTip: parseLandmark(raw.littleTip),
    littleDIP: parseLandmark(raw.littleDIP),
    littlePIP: parseLandmark(raw.littlePIP),
    littleMCP: parseLandmark(raw.littleMCP),
    wrist: parseLandmark(raw.wrist),
  };
}

/**
 * Call from within a VisionCamera frame processor (worklet context).
 * Returns null when no hands are detected or the plugin is unavailable.
 */
export function detectHandGesture(frame: Frame): HandGestureResult | null {
  "worklet";
  if (!plugin) return null;
  const raw = plugin.call(frame) as Record<string, any> | null | undefined;
  if (!raw || !raw.hands) return null;
  // NOTE: parsing happens on the worklet thread — keep it lightweight.
  // The native side already filters low-confidence landmarks.
  return raw as unknown as HandGestureResult;
}
