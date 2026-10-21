"""
2D Pointing Ray & Object Selection Engine for GestureGrasp.
Extracts 2D image-plane ray from Index MCP (landmark 5) to Index Tip (landmark 8).
Selects target object from YOLO detections using ray-box intersection, angular deviation,
and temporal consistency tracking.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Landmark constants
INDEX_MCP = 5
INDEX_TIP = 8

DEFAULT_ALLOWED_OBJECTS = [
    "bottle",
    "cup",
    "cell phone",
    "remote",
    "book",
    "mouse",
    "apple",
    "banana",
]


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]

    @property
    def x1(self) -> float:
        return self.bbox_xyxy[0]

    @property
    def y1(self) -> float:
        return self.bbox_xyxy[1]

    @property
    def x2(self) -> float:
        return self.bbox_xyxy[2]

    @property
    def y2(self) -> float:
        return self.bbox_xyxy[3]


@dataclass
class PointingRay:
    origin: tuple[float, float]     # (x, y) pixel coordinates of Landmark 5 (Index MCP)
    tip: tuple[float, float]        # (x, y) pixel coordinates of Landmark 8 (Index Tip)
    direction: tuple[float, float]  # Unit vector (dx, dy)
    angle_deg: float                # Angle in degrees relative to horizontal


class TargetConsistencyTracker:
    """Which object has the user been pointing at over the SELECT hold?

    A sliding window of the last `required_hold_time` seconds votes: the target is stable once the
    window is full and one object holds at least `min_share` of its frames. A trembling ray that
    grazes a neighbour for a frame or two no longer resets the hold, while a deliberate move to
    another object still takes over as soon as it wins the window.
    """

    def __init__(self, required_hold_time: float = 0.80, min_share: float = 0.6):
        self.required_hold_time = required_hold_time
        self.min_share = min_share
        self.history: deque[tuple[float, Optional[str]]] = deque()
        self.current_candidate_label: Optional[str] = None
        self.share: float = 0.0
        self._since: Optional[float] = None

    def update(self, target_label: Optional[str], timestamp: Optional[float] = None) -> tuple[float, bool]:
        """Returns (hold_progress 0..1, is_stable)."""
        now = time.time() if timestamp is None else timestamp
        if self._since is None:
            self._since = now
        self.history.append((now, target_label))
        while self.history and self.history[0][0] < now - self.required_hold_time - 1e-6:
            self.history.popleft()

        votes = Counter(label for _, label in self.history if label)
        if not votes:
            self.current_candidate_label, self.share, self._since = None, 0.0, None
            self.history.clear()
            return 0.0, False
        label, count = votes.most_common(1)[0]
        self.current_candidate_label = label
        self.share = count / len(self.history)
        covered = min(1.0, (now - self._since) / max(self.required_hold_time, 1e-6) + 1e-6)
        progress = covered if self.share >= self.min_share else 0.0
        return min(progress, 1.0), progress >= 1.0

    def reset(self) -> None:
        self.history.clear()
        self.current_candidate_label = None
        self.share = 0.0
        self._since = None


class PointingEngine:
    """
    Computes 2D ray from hand landmarks and selects the pointed object.
    Depth Ambiguity Note: MediaPipe RGB landmarks operate on a 2D projection.
    Objects behind one another on the line of sight cannot be disambiguated in 3D
    without depth sensing or user confirmation.
    """

    def __init__(
        self,
        max_angular_deviation_deg: float = 15.0,
        allowed_classes: Optional[list[str]] = None,
        min_confidence: float = 0.35,
    ):
        self.max_angular_deviation_deg = max_angular_deviation_deg
        self.allowed_classes = set(allowed_classes or DEFAULT_ALLOWED_OBJECTS)
        self.min_confidence = min_confidence
        self.tracker = TargetConsistencyTracker()
        self._last_seen: dict[str, dict[str, Any]] = {}
        self._stable = False

    def compute_ray(
        self,
        landmarks: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> Optional[PointingRay]:
        """
        landmarks: (21, 3) or (63,) array in normalized [0, 1] coordinates.
        frame_shape: (height, width).
        """
        lm = np.asarray(landmarks, dtype=np.float32)
        if lm.ndim == 1:
            lm = lm.reshape((21, 3))

        h, w = frame_shape[0], frame_shape[1]

        # Landmark 5 (Index MCP) and Landmark 8 (Index Tip)
        x_mcp = float(lm[INDEX_MCP, 0] * w)
        y_mcp = float(lm[INDEX_MCP, 1] * h)

        x_tip = float(lm[INDEX_TIP, 0] * w)
        y_tip = float(lm[INDEX_TIP, 1] * h)

        dx = x_tip - x_mcp
        dy = y_tip - y_mcp
        length = math.hypot(dx, dy)

        if length < 5.0:  # hand too small / landmark degenerate
            return None

        unit_dx = dx / length
        unit_dy = dy / length
        angle_deg = math.degrees(math.atan2(unit_dy, unit_dx))

        return PointingRay(
            origin=(x_mcp, y_mcp),
            tip=(x_tip, y_tip),
            direction=(unit_dx, unit_dy),
            angle_deg=angle_deg,
        )

    def _ray_box_intersection(
        self,
        ray: PointingRay,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """Check if 2D ray starting from ray.origin intersects the axis-aligned bounding box."""
        x1, y1, x2, y2 = bbox
        ox, oy = ray.origin
        dx, dy = ray.direction

        # Slab method for 2D ray / AABB intersection
        tmin = -float("inf")
        tmax = float("inf")

        # X slab
        if abs(dx) > 1e-7:
            tx1 = (x1 - ox) / dx
            tx2 = (x2 - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))
        else:
            if ox < x1 or ox > x2:
                return False

        # Y slab
        if abs(dy) > 1e-7:
            ty1 = (y1 - oy) / dy
            ty2 = (y2 - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
        else:
            if oy < y1 or oy > y2:
                return False

        # Must intersect in forward direction of ray (tmax >= tmin and tmax >= 0)
        return (tmax >= tmin) and (tmax >= 0)

    def _angular_deviation(
        self,
        ray: PointingRay,
        target_center: tuple[float, float],
    ) -> float:
        """Calculate angular deviation in degrees between pointing ray and target center."""
        ox, oy = ray.origin
        cx, cy = target_center
        vx = cx - ox
        vy = cy - oy
        v_dist = math.hypot(vx, vy)
        if v_dist < 1e-6:
            return 0.0

        v_dx = vx / v_dist
        v_dy = vy / v_dist

        rx, ry = ray.direction
        dot = rx * v_dx + ry * v_dy
        dot = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot)
        return math.degrees(angle_rad)

    def stable_target(self) -> Optional[dict[str, Any]]:
        """The object that won the pointing vote over the full hold, or None if none did."""
        label = self.tracker.current_candidate_label
        if not self._stable or label is None or label not in self._last_seen:
            return None
        return dict(self._last_seen[label], is_stable=True, vote_share=round(self.tracker.share, 2))

    def _vote(self, label: Optional[str], timestamp: Optional[float]) -> tuple[float, bool]:
        progress, stable = self.tracker.update(label, timestamp)
        self._stable = stable
        return progress, stable

    def select_target(
        self,
        ray: Optional[PointingRay],
        objects: list[DetectedObject],
        timestamp: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Best object for this frame's ray, and a vote for it in the hold window.
        Priority: allow-listed objects with confidence >= min_confidence; direct ray/box hits
        (nearest first); otherwise smallest angular deviation within max_angular_deviation_deg.
        """
        if ray is None or not objects:
            self._vote(None, timestamp)
            return None
        candidates = [
            obj for obj in objects
            if obj.label.lower() in self.allowed_classes and obj.confidence >= self.min_confidence
        ]
        best_obj, best_dev, direct = None, 0.0, False
        hits = [obj for obj in candidates if self._ray_box_intersection(ray, obj.bbox_xyxy)]
        if hits:
            best_obj = min(hits, key=lambda o: math.hypot(o.center[0] - ray.origin[0], o.center[1] - ray.origin[1]))
            best_dev, direct = self._angular_deviation(ray, best_obj.center), True
        else:
            scored = []
            for obj in candidates:
                dev = self._angular_deviation(ray, obj.center)
                if dev <= self.max_angular_deviation_deg:
                    dist = math.hypot(obj.center[0] - ray.origin[0], obj.center[1] - ray.origin[1])
                    scored.append((dev * 1.5 + dist * 0.01, dev, obj))
            if scored:
                _, best_dev, best_obj = min(scored, key=lambda x: x[0])
        if best_obj is None:
            self._vote(None, timestamp)
            return None

        progress, stable = self._vote(best_obj.label, timestamp)
        result = {
            "label": best_obj.label,
            "confidence": best_obj.confidence,
            "bbox": list(best_obj.bbox_xyxy),
            "center": list(best_obj.center),
            "angular_deviation_deg": round(best_dev, 2),
            "direct_hit": direct,
            "hold_progress": progress,
            "is_stable": stable,
        }
        self._last_seen[best_obj.label] = result
        return result
