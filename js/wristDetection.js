
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
