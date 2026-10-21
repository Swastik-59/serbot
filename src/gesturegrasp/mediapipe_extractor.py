"""
MediaPipe Hands wrapper for GestureGrasp: 21 landmarks per hand from a BGR frame.
Frames are resized so the long side is 640 px before detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("gesturegrasp.mediapipe")


@dataclass
class HandLandmarkResult:
    landmarks: np.ndarray          # (21, 3) normalized coordinates [0, 1]
    handedness: str                # "Left" or "Right"
    handedness_confidence: float   # 0.0 - 1.0
    bounding_box_norm: tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) in [0, 1]


class MediaPipeHandExtractor:
    """MediaPipe Hands, up to two hands per frame."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        target_long_side: int = 640,
    ):
        self.target_long_side = target_long_side
        self.mp_hands = mp.solutions.hands
        self._detector = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if self._detector:
            self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize so the longest side is target_long_side (never upscales)."""
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= self.target_long_side:
            return frame
        scale = self.target_long_side / float(long_side)
        return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame_bgr: np.ndarray) -> list[HandLandmarkResult]:
        """Detected hands in a BGR frame. Coordinates are normalised, so resizing does not change them."""
        frame_rgb = cv2.cvtColor(self.resize_frame(frame_bgr), cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)
        if not results or not results.multi_hand_landmarks:
            return []

        hands: list[HandLandmarkResult] = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
            handedness, conf = "Right", 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                h_class = results.multi_handedness[idx].classification[0]
                handedness, conf = h_class.label, float(h_class.score)
            xs, ys = lm_array[:, 0], lm_array[:, 1]
            hands.append(HandLandmarkResult(
                landmarks=lm_array,
                handedness=handedness,
                handedness_confidence=conf,
                bounding_box_norm=(float(ys.min()), float(xs.min()), float(ys.max()), float(xs.max())),
            ))
        return hands

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_landmarks_list: list[np.ndarray]) -> np.ndarray:
        """Draw the 21-point hand skeleton on a copy of the frame."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        for lms in hand_landmarks_list:
            pts = [(int(np.clip(lms[i, 0] * w, 0, w - 1)), int(np.clip(lms[i, 1] * h, 0, h - 1))) for i in range(21)]
            for a, b in self.mp_hands.HAND_CONNECTIONS:
                cv2.line(annotated, pts[a], pts[b], (255, 255, 255), 2)
            for p in pts:
                cv2.circle(annotated, p, 4, (0, 255, 0), -1)
        return annotated
