"""
Lightweight YOLO Object Detection for GestureGrasp.
Wraps Ultralytics nano models (default yolo11n.pt / fallback yolov8n.pt).
Includes offline mock mode to avoid mandatory runtime weight downloads during development/testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.gesturegrasp.pointing import DetectedObject, PointingRay, DEFAULT_ALLOWED_OBJECTS

logger = logging.getLogger("gesturegrasp.detector")


class ObjectDetector:
    """YOLO Object Detector with graceful fallback and offline mock support."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        allow_list: Optional[list[str]] = None,
        conf_threshold: float = 0.35,
        use_mock: bool = False,
    ):
        self.model_name = model_name
        self.allow_list = set(allow_list or DEFAULT_ALLOWED_OBJECTS)
        self.conf_threshold = conf_threshold
        self.use_mock = use_mock
        self.active_model_name = "mock" if use_mock else model_name
        self._model = None
        self.device = "cpu"

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Attempt loading real Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO

            weights_path = Path(self.model_name)
            # Check if file exists locally or if ultralytics can load it
            logger.info(f"Loading YOLO detector: {self.model_name}")
            self._model = YOLO(self.model_name)
            self.active_model_name = self.model_name
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
            logger.info(f"YOLO detector '{self.model_name}' loaded on {self.device}.")
        except Exception as e:
            logger.warning(
                f"Failed to load YOLO model '{self.model_name}': {e}. "
                "Switching to offline MockDetector mode."
            )
            self.use_mock = True
            self.active_model_name = "mock_fallback"

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """Detect allow-listed objects in the RGB/BGR frame."""
        h, w = frame.shape[:2]

        if self.use_mock or self._model is None:
            # Deterministic mock objects for offline development & tests
            mock_objects = [
                DetectedObject(
                    label="bottle",
                    confidence=0.88,
                    bbox_xyxy=(w * 0.65, h * 0.30, w * 0.80, h * 0.75),
                    center=(w * 0.725, h * 0.525),
                ),
                DetectedObject(
                    label="cup",
                    confidence=0.82,
                    bbox_xyxy=(w * 0.20, h * 0.45, w * 0.35, h * 0.70),
                    center=(w * 0.275, h * 0.575),
                ),
                DetectedObject(
                    label="book",
                    confidence=0.76,
                    bbox_xyxy=(w * 0.40, h * 0.65, w * 0.60, h * 0.85),
                    center=(w * 0.50, h * 0.75),
                ),
            ]
            return [obj for obj in mock_objects if obj.label in self.allow_list]

        # Real YOLO inference
        try:
            results = self._model(frame, conf=self.conf_threshold, verbose=False,
                                  device=self.device, half=self.device != "cpu")
            detected = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self._model.names.get(cls_id, str(cls_id)).lower()
                    conf = float(box.conf[0])

                    if label in self.allow_list and conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        detected.append(
                            DetectedObject(
                                label=label,
                                confidence=conf,
                                bbox_xyxy=(x1, y1, x2, y2),
                                center=(cx, cy),
                            )
                        )
            return detected
        except Exception as e:
            logger.error(f"Inference error in YOLO: {e}")
            return []

    def draw_annotations(
        self,
        frame: np.ndarray,
        objects: list[DetectedObject],
        selected_target: Optional[dict[str, Any]] = None,
        pointing_ray: Optional[PointingRay] = None,
    ) -> np.ndarray:
        """Render bounding boxes, labels, pointing ray, and target highlight on frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        selected_label = selected_target.get("label") if selected_target else None

        # 1. Draw detected objects
        for obj in objects:
            is_selected = selected_label and obj.label == selected_label
            color = (0, 255, 0) if is_selected else (255, 140, 0)  # Green if selected, Cyan/Orange otherwise
            thickness = 3 if is_selected else 2

            x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{obj.label.upper()} {obj.confidence:.2f}"
            if is_selected:
                label_text += " [TARGET]"

            # Background for label text
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if is_selected else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw 2D Pointing Ray
        if pointing_ray is not None:
            ox, oy = map(int, pointing_ray.origin)
            tx, ty = map(int, pointing_ray.tip)
            dx, dy = pointing_ray.direction

            # Draw MCP to Tip segment (Hand ray base)
            cv2.circle(annotated, (ox, oy), 6, (0, 0, 255), -1)  # Red MCP origin
            cv2.circle(annotated, (tx, ty), 6, (0, 255, 255), -1)  # Yellow tip
            cv2.line(annotated, (ox, oy), (tx, ty), (0, 255, 255), 2)

            # Extended projected ray
            ray_length = max(w, h)
            end_x = int(ox + dx * ray_length)
            end_y = int(oy + dy * ray_length)
            cv2.line(annotated, (tx, ty), (end_x, end_y), (0, 255, 255), 2, cv2.LINE_AA)

        return annotated
