
import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";

let detectorPromise = null;

// Phase 1: Initialize MediaPipe
export function initializeHandLandmarker() {

  if (!detectorPromise) {
    detectorPromise = (async () => {

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm"
      );

      const detector = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "CPU"
          },
          runningMode: "IMAGE",
          numHands: 1
        }
      );

      return detector;
    })();
  }

  return detectorPromise.catch(error => {
    detectorPromise = null;
    throw error;
  });
}

// Phase 2: Detect the 21 hand landmarks
export async function detectHandLandmarks(image) {

  const detector = await initializeHandLandmarker();

  const result = detector.detect(image);

  const landmarks = result.landmarks?.[0] || [];

  if (landmarks.length !== 21) {
    return {
      detected: false,
      landmarks: [],
      wrist: null
    };
  }

  return {
    detected: true,
    landmarks: landmarks,
    wrist: landmarks[0]
  };
}


/* ==========================================
   PHASE 3 - WATCH PLACEMENT CALCULATION
========================================== */

export function calculateWatchPlacement(
  landmarks,
  imageWidth,
  imageHeight
) {

  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  // Convert normalized landmark positions into pixels
  function getPoint(index) {
    return {
      x: landmarks[index].x * imageWidth,
      y: landmarks[index].y * imageHeight
    };
  }

  const wrist = getPoint(0);
  const indexFinger = getPoint(5);
  const middleFinger = getPoint(9);
  const pinkyFinger = getPoint(17);

  // Step 1: Estimate palm width
  const palmWidth = Math.hypot(
    indexFinger.x - pinkyFinger.x,
    indexFinger.y - pinkyFinger.y
  );

  // Step 2: Calculate direction from palm to wrist
  const dx = wrist.x - middleFinger.x;
  const dy = wrist.y - middleFinger.y;

  const distance = Math.hypot(dx, dy);

  if (
    !Number.isFinite(palmWidth) ||
    !Number.isFinite(distance) ||
    palmWidth < 5 ||
    distance < 5
  ) {
    return null;
  }

  // Step 3: Normalize forearm direction
  const directionX = dx / distance;
  const directionY = dy / distance;

  // Step 4: Estimate watch center beyond wrist landmark
  const offset = palmWidth * 0.55;

  const watchX = wrist.x + directionX * offset;
  const watchY = wrist.y + directionY * offset;

  // Step 5: Estimate watch dimensions
  const watchWidth = palmWidth * 0.68;
  const watchHeight = watchWidth * 1.35;

  // Step 6: Calculate rotation angle
  // Assumes the watch image is vertically oriented
  const angleRadians =
    Math.atan2(directionY, directionX) -
    Math.PI / 2;

  const angleDegrees =
    angleRadians * (180 / Math.PI);

  return {
    x: Math.round(watchX),
    y: Math.round(watchY),
    width: Math.round(watchWidth),
    height: Math.round(watchHeight),
    angle: Number(angleDegrees.toFixed(1)),

    wristX: Math.round(wrist.x),
    wristY: Math.round(wrist.y),

    centerInsideImage:
      watchX >= 0 &&
      watchX < imageWidth &&
      watchY >= 0 &&
      watchY < imageHeight
  };
}



