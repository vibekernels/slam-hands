/**
 * OK-hand gesture recognition from hand landmarks.
 *
 * An "OK" gesture is defined as:
 * 1. Thumb tip touches (or nearly touches) the index finger tip
 * 2. Middle, ring, and pinky fingers are extended (not curled)
 *
 * Landmarks use Apple Vision normalized coordinates: origin at bottom-left,
 * (1,1) at top-right.
 */

interface Point {
  x: number;
  y: number;
}

function dist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export interface HandForGesture {
  chirality: "left" | "right";
  thumbTip: Point;
  thumbIP: Point;
  indexTip: Point;
  indexDIP: Point;
  indexMCP: Point;
  middleTip: Point;
  middleMCP: Point;
  ringTip: Point;
  ringMCP: Point;
  littleTip: Point;
  littleMCP: Point;
  wrist: Point;
}

/**
 * Returns true if the hand is making an OK gesture.
 *
 * Criteria:
 * - Thumb tip to index tip distance < 0.15 (normalized)
 * - At least 2 of middle/ring/pinky are extended
 *   (fingertip further from wrist than its MCP joint)
 */
export function thumbIndexDistance(hand: HandForGesture): number {
  return dist(hand.thumbTip, hand.indexTip);
}

export function isOpenHand(hand: HandForGesture): boolean {
  // All five fingers extended: each fingertip is further from wrist than its MCP
  const margin = 1.2;
  const thumbExtended = dist(hand.thumbTip, hand.wrist) > dist(hand.thumbIP, hand.wrist) * margin;
  const indexExtended = dist(hand.indexTip, hand.wrist) > dist(hand.indexMCP, hand.wrist) * margin;
  const middleExtended = dist(hand.middleTip, hand.wrist) > dist(hand.middleMCP, hand.wrist) * margin;
  const ringExtended = dist(hand.ringTip, hand.wrist) > dist(hand.ringMCP, hand.wrist) * margin;
  const littleExtended = dist(hand.littleTip, hand.wrist) > dist(hand.littleMCP, hand.wrist) * margin;

  return thumbExtended && indexExtended && middleExtended && ringExtended && littleExtended;
}

export interface GestureState {
  anyHandOk: boolean;
  leftOk: boolean;
  rightOk: boolean;
  handsDetected: number;
  debugDist?: string;
}

/**
 * Analyze an array of detected hands and return the aggregate gesture state.
 */
export function analyzeGesture(hands: HandForGesture[]): GestureState {
  let leftOk = false;
  let rightOk = false;
  let minDist = Infinity;

  for (const hand of hands) {
    const d = thumbIndexDistance(hand);
    if (d < minDist) minDist = d;
    const ok = isOpenHand(hand);
    if (ok && hand.chirality === "left") leftOk = true;
    if (ok && hand.chirality === "right") rightOk = true;
  }

  return {
    anyHandOk: leftOk || rightOk,
    leftOk,
    rightOk,
    handsDetected: hands.length,
    debugDist: hands.length > 0 ? minDist.toFixed(3) : undefined,
  };
}
