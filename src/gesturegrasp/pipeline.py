"""
Core GestureGrasp Pipeline API. Completely decoupled from Streamlit.

    process(frame)  = perceive (MediaPipe hand + YOLO objects)  ->  decide(...)  ->  draw
    decide(...)     = classifier -> IntentEngine -> pointing/target -> FSM -> robot

decide() takes landmarks and detections directly, so tests can drive the decision logic without
a camera.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import joblib
import numpy as np

from robot.base import BaseRobot, RobotState, RobotStatus
from robot.simulated import SimulatedRobot
from src.gesturegrasp.detector import ObjectDetector
from src.gesturegrasp.features import FEATURE_VERSION_2D, extract_features_2d
from src.gesturegrasp.intent_engine import IntentEngine
from src.gesturegrasp.mediapipe_extractor import MediaPipeHandExtractor
from src.gesturegrasp.pointing import DetectedObject, PointingEngine, PointingRay
from src.gesturegrasp.state_machine import GestureStateMachine

logger = logging.getLogger("gesturegrasp.pipeline")

@dataclass
class PipelineResult:
    gesture: str
    confidence: float
    probabilities: dict[str, float]
    intent: str
    state: str
    hold_progress: float
    detected_objects: list[DetectedObject]
    selected_target: Optional[dict[str, Any]]
    pointing_ray: Optional[PointingRay]
    robot_status: RobotStatus
    latency_ms: float
    fps: float
    hand_detected: bool
    landmarks: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    event_log: list[str] = field(default_factory=list)
    intent_fired: bool = False
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture": self.gesture,
            "confidence": round(self.confidence, 3),
            "intent": self.intent,
            "intent_fired": self.intent_fired,
            "state": self.state,
            "hold_progress": round(self.hold_progress, 2),
            "selected_target": self.selected_target,
            "latency_ms": round(self.latency_ms, 1),
            "fps": round(self.fps, 1),
            "hand_detected": self.hand_detected,
            "robot_state": self.robot_status.state.value,
        }


def _is_bgr_frame(frame: object) -> bool:
    """True when the array is a non-empty BGR image."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        return False
    return frame.dtype in (np.uint8, np.float32, np.float64)


class GesturePipeline:
    """The master end-to-end perception, intent, and robot control pipeline."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        robot: Optional[BaseRobot] = None,
        use_mock_detector: bool = False,
    ):
        self.config_path = config_path or Path("configs/intents.yaml")
        self.hand_extractor = MediaPipeHandExtractor(static_image_mode=False)
        self.detector = ObjectDetector(use_mock=use_mock_detector)
        self.intent_engine = IntentEngine(self.config_path)
        self._now: Optional[float] = None
        self.state_machine = GestureStateMachine(self.config_path, clock=self._clock)
        self.pointing_engine = PointingEngine()
        self.robot = robot or SimulatedRobot()

        self._load_classifier(model_path, scaler_path)

        self._task_dispatched = False
        self._return_dispatched = False
        self._last_fsm_state = self.state_machine.current_state
        self._last_frame_time: Optional[float] = None
        self.fps = 0.0

    def _clock(self) -> float:
        return self._now if self._now is not None else time.time()

    def _load_classifier(self, model_path: Optional[Path], scaler_path: Optional[Path]) -> None:
        """Load the trained model and its scaler. There is no fallback: guessing gestures is unsafe."""
        m_path = model_path or Path("models/calibrated_model.joblib")
        s_path = scaler_path or m_path.parent / "scaler.joblib"
        meta_path = m_path.parent / "metadata.json"
        missing = [str(p) for p in (m_path, s_path, meta_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Trained model files missing: {', '.join(missing)}. "
                                    "Train with: python -m src.gesturegrasp.train_v2")
        self.feature_version = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_version")
        if self.feature_version != FEATURE_VERSION_2D:
            raise ValueError(f"{m_path} uses feature set {self.feature_version}; this code needs "
                             f"{FEATURE_VERSION_2D}. Retrain with: python -m src.gesturegrasp.train_v2")
        # joblib unpickles: only load models/ files this project's own training wrote.
        self.model = joblib.load(m_path)
        self.scaler = joblib.load(s_path)
        logger.info("Loaded gesture model %s (feature set %s)", m_path, self.feature_version)

    def predict_probabilities(self, landmarks: np.ndarray, frame_shape: tuple[int, int]) -> dict[str, float]:
        """Class probabilities for one hand. landmarks: (21, 2|3) MediaPipe-normalised."""
        h, w = frame_shape[:2]
        features = extract_features_2d(np.asarray(landmarks, dtype=np.float32)[:, :2], aspect=w / h)
        probs = self.model.predict_proba(self.scaler.transform(features.reshape(1, -1)))[0]
        return {str(c): float(p) for c, p in zip(self.model.classes_, probs)}

    # -- per-frame API --------------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None,
                draw_annotations: bool = True) -> PipelineResult:
        """Camera frame in, full result out."""
        t_start = time.perf_counter()
        now = time.time() if timestamp is None else timestamp
        if not _is_bgr_frame(frame_bgr):
            logger.warning("Skipping frame: expected a non-empty BGR image with shape (H, W, 3).")
            return self._empty_result(t_start)

        hands = self.hand_extractor.process_frame(frame_bgr)
        objects = self.detector.detect(frame_bgr)
        hand = self._commanding_hand(hands, frame_bgr.shape[:2])
        result = self.decide(
            landmarks=hand.landmarks if hand else None,
            frame_shape=frame_bgr.shape[:2],
            detected_objects=objects,
            timestamp=now,
        )
        if draw_annotations:
            annotated = frame_bgr.copy()
            if hand is not None:
                annotated = self.hand_extractor.draw_landmarks(annotated, [hand.landmarks])
            pointing = result.gesture == "point"  # draw the ray and target only while pointing
            annotated = self.detector.draw_annotations(
                annotated, objects,
                selected_target=result.selected_target if pointing else None,
                pointing_ray=result.pointing_ray if pointing else None)
            result.annotated_frame = annotated
        result.latency_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    def _commanding_hand(self, hands: list, frame_shape: tuple[int, int]):
        """With two hands in view (one often resting), act on the one showing the clearest command.
        MediaPipe's hand order is arbitrary, so taking hands[0] can silently ignore the gesture."""
        if len(hands) < 2:
            return hands[0] if hands else None
        non_command = {name for name, meta in self.intent_engine.gestures_cfg.items() if not meta.get("actionable")}

        def command_strength(hand) -> float:
            probs = self.predict_probabilities(hand.landmarks, frame_shape)
            return max((p for g, p in probs.items() if g not in non_command), default=0.0)

        return max(hands, key=command_strength)

    def decide(self, landmarks: Optional[np.ndarray], frame_shape: tuple[int, int],
               detected_objects: list[DetectedObject], timestamp: float) -> PipelineResult:
        """Everything after perception. landmarks: (21, 2|3) normalised, or None when no hand."""
        t_start = time.perf_counter()
        now = timestamp
        self._now = now
        if self._last_frame_time is not None and now > self._last_frame_time:
            current = 1.0 / (now - self._last_frame_time)
            self.fps = 0.8 * self.fps + 0.2 * current if self.fps > 0 else current
        self._last_frame_time = now

        hand_detected = landmarks is not None
        raw_probs = self.predict_probabilities(landmarks, frame_shape) if hand_detected else {}
        intent_res = self.intent_engine.update(raw_probs=raw_probs, current_state=self.state_machine.current_state,
                                               hand_detected=hand_detected, timestamp=now)

        pointing_ray = None
        selected_target = None
        if hand_detected:
            pointing_ray = self.pointing_engine.compute_ray(np.asarray(landmarks)[:, :2], frame_shape)
            if pointing_ray is not None:
                selected_target = self.pointing_engine.select_target(pointing_ray, detected_objects, timestamp=now)
        else:
            self.pointing_engine.tracker.update(None, now)

        if intent_res.intent_fired:
            # SELECT uses the object that won the pointing vote over the hold, never a single frame.
            stable_target = self.pointing_engine.stable_target() if intent_res.intent == "SELECT_TARGET" else None
            accepted = self.state_machine.handle_intent(intent_res.intent, target_info=stable_target)
            if not accepted and intent_res.intent == "SELECT_TARGET":
                self.intent_engine.cancel_cooldown()
        self.state_machine.check_timeouts(now)
        self._sync_robot()

        sm = self.state_machine
        return PipelineResult(
            gesture=intent_res.gesture,
            confidence=intent_res.confidence,
            probabilities=intent_res.smoothed_probabilities,
            intent=intent_res.intent,
            state=sm.current_state,
            hold_progress=intent_res.hold_progress,
            detected_objects=detected_objects,
            selected_target=selected_target,
            pointing_ray=pointing_ray,
            robot_status=self.robot.get_status(),
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
            fps=self.fps,
            hand_detected=hand_detected,
            landmarks=None if landmarks is None else np.asarray(landmarks),
            event_log=[f"[{e.from_state} -> {e.to_state}] {e.trigger}: {e.details}" for e in sm.event_log[-5:]],
            intent_fired=intent_res.intent_fired,
            raw_probabilities=raw_probs,
            reason=intent_res.reason,
        )

    def _sync_robot(self) -> None:
        """Drive the robot from the FSM state. Each FSM state issues its robot command once."""
        state = self.state_machine.current_state
        previous, self._last_fsm_state = self._last_fsm_state, state
        status = self.robot.get_status()

        if state == "STOPPED":
            if not status.is_stopped:
                self.robot.emergency_stop()
            return
        if status.is_stopped:
            # FSM left STOPPED (WAKE), so release the robot where it stands.
            self.robot.resume()
            status = self.robot.get_status()

        if previous == "AWAIT_FEEDBACK" and state == "IDLE":
            # ACKNOWLEDGE: hand the object over, then go home.
            self.robot.release()
            self.robot.return_to_dock()
        elif previous == "AWAIT_FEEDBACK" and state == "LISTENING":
            # REJECT of a delivered object: set it down by the user and wait for a new choice.
            self.robot.release()

        if state != "EXECUTING":
            self._task_dispatched = False
        if state != "RETURNING":
            self._return_dispatched = False

        if state == "EXECUTING":
            target = self.state_machine.proposed_target
            if not self._task_dispatched and target:
                goal = self.robot.resolve_target(target)
                if goal is not None:
                    self._task_dispatched = self.robot.navigate_to(goal[0], goal[1], label=target.get("label", "target"))
            elif self._task_dispatched and status.state == RobotState.IDLE and status.holding_object:
                # Fetched the object and brought it to the user.
                self.state_machine.notify_task_complete()

        elif state == "RETURNING":
            if not self._return_dispatched:
                self._return_dispatched = self.robot.return_to_dock()
            elif status.state == RobotState.IDLE:
                self.state_machine.notify_dock_reached()

    def _empty_result(self, t_start: float) -> PipelineResult:
        return PipelineResult(
            gesture="no_gesture", confidence=0.0, probabilities={}, intent="NONE",
            state=self.state_machine.current_state, hold_progress=0.0, detected_objects=[],
            selected_target=None, pointing_ray=None, robot_status=self.robot.get_status(),
            latency_ms=(time.perf_counter() - t_start) * 1000.0, fps=self.fps, hand_detected=False,
        )

    def reset(self) -> None:
        """Reset pipeline, intent engine, FSM, and robot."""
        self.intent_engine.reset()
        self.state_machine.reset()
        self.pointing_engine.tracker.reset()
        # MediaPipe in tracking mode carries the last hand ROI forward; a fresh tracker forgets it.
        self.hand_extractor.close()
        self.hand_extractor = MediaPipeHandExtractor(static_image_mode=False)
        self.robot.reset()
        self._task_dispatched = False
        self._return_dispatched = False
        self._last_fsm_state = self.state_machine.current_state
