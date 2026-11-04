"""
Temporal Intent Engine for GestureGrasp.
Converts noisy per-frame classifier probabilities into robust, safe HRI commands.
Implements:
- EMA probability smoothing (alpha=0.6)
- Top-1 vs Runner-up confidence margin gating (>=0.20)
- Hold-to-fire temporal integration
- Dropped frame tolerance (max 2 consecutive frames, and never longer than at the 15 fps it was evaluated at)
- Cooldown refractory period (1.0 sec)
- Emergency STOP priority override
- FSM state gating
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger("gesturegrasp.intent_engine")

# The evaluation ran at 15 fps, where the frame limit forgets a hand after (2 + 1) / 15 = 0.2 s. On a Pi
# at 8 fps the same limit would bridge 0.375 s without seeing a hand, so a lost hand also expires by time.
EVALUATED_FPS = 15.0


@dataclass
class IntentResult:
    gesture: str
    intent: str
    confidence: float
    raw_probabilities: dict[str, float]
    smoothed_probabilities: dict[str, float]
    hold_progress: float
    intent_fired: bool
    gated: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


class IntentEngine:
    """
    Deterministic perception-to-action mediator.
    Never lets a single raw frame actuate the robot.
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path("configs/intents.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.gestures_cfg = cfg["gestures"]
        engine_cfg = cfg.get("intent_engine", {})
        self.ema_alpha = engine_cfg.get("ema_alpha", 0.6)
        self.margin_threshold = engine_cfg.get("margin_threshold", 0.20)
        self.cooldown_seconds = engine_cfg.get("cooldown_seconds", 1.00)
        self.max_dropped_frames = engine_cfg.get("max_dropped_frames", 2)
        self.max_dropped_seconds = (self.max_dropped_frames + 1) / EVALUATED_FPS
        self.stop_priority = engine_cfg.get("stop_priority", True)

        self.transitions = cfg["state_machine"]["transitions"]

        # Runtime state
        self._smoothed_probs: dict[str, float] = {}
        self._candidate_gesture: Optional[str] = None
        self._candidate_start_time: float = 0.0
        self._dropped_frames: int = 0
        self._last_hand_time: Optional[float] = None
        self._last_fire_time: float = -1000.0
        self._last_fired_intent: Optional[str] = None

    def reset(self) -> None:
        """Reset internal temporal buffers."""
        self._smoothed_probs.clear()
        self._candidate_gesture = None
        self._candidate_start_time = 0.0
        self._dropped_frames = 0
        self._last_hand_time = None
        self._last_fire_time = -1000.0
        self._last_fired_intent = None

    def cancel_cooldown(self) -> None:
        """The last fired intent was refused downstream (e.g. SELECT with no stable target), so
        let the user keep holding the gesture instead of waiting out the cooldown."""
        self._last_fire_time = -1000.0

    def update(
        self,
        raw_probs: dict[str, float],
        current_state: str,
        hand_detected: bool = True,
        timestamp: Optional[float] = None,
    ) -> IntentResult:
        """
        Process a new frame's probability distribution and return the intent status.
        """
        now = time.time() if timestamp is None else timestamp

        if not hand_detected or not raw_probs:
            self._dropped_frames += 1
            gap = None if self._last_hand_time is None else now - self._last_hand_time
            hand_gone = self._dropped_frames > self.max_dropped_frames or (
                gap is not None and gap >= self.max_dropped_seconds)
            if hand_gone:
                self._candidate_gesture = None
                self._candidate_start_time = 0.0
                self._smoothed_probs.clear()

            return IntentResult(
                gesture="no_gesture",
                intent="NONE",
                confidence=0.0,
                raw_probabilities=raw_probs or {},
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=False,
                reason="No hand detected" if hand_gone else "No hand detected (buffer active)",
                timestamp=now,
            )

        # Hand is detected: reset dropped frame counter
        self._dropped_frames = 0
        self._last_hand_time = now

        # Step 1: Exponential Moving Average (EMA) smoothing
        for gesture_name, prob in raw_probs.items():
            if gesture_name not in self._smoothed_probs:
                self._smoothed_probs[gesture_name] = prob
            else:
                self._smoothed_probs[gesture_name] = (
                    self.ema_alpha * prob + (1.0 - self.ema_alpha) * self._smoothed_probs[gesture_name]
                )

        # Step 2: Sort smoothed probabilities to get Top-1 and Runner-up
        sorted_probs = sorted(self._smoothed_probs.items(), key=lambda x: x[1], reverse=True)
        top1_gesture, top1_prob = sorted_probs[0]
        runner_up_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        margin = top1_prob - runner_up_prob

        # Gesture metadata lookup
        g_info = self.gestures_cfg.get(top1_gesture, {})
        intent = g_info.get("intent", "NONE")
        is_actionable = g_info.get("actionable", False)
        threshold = g_info.get("min_conf", 0.70)
        required_hold = g_info.get("hold_time", 0.70)

        # Emergency STOP Priority check
        # If stop is high confidence, it can override other gesture holds
        is_stop = intent == "STOP"

        # Check 1: Cooldown check (STOP bypasses cooldown)
        time_since_fire = now - self._last_fire_time
        if not is_stop and time_since_fire < self.cooldown_seconds:
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=True,
                reason=f"Cooldown active ({self.cooldown_seconds - time_since_fire:.2f}s remaining)",
                timestamp=now,
            )

        # Check 2: Confidence threshold
        if top1_prob < threshold:
            self._candidate_gesture = None
            self._candidate_start_time = 0.0
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=False,
                reason=f"Below confidence threshold ({top1_prob:.2f} < {threshold:.2f})",
                timestamp=now,
            )

        # Check 3: Margin check
        if margin < self.margin_threshold:
            self._candidate_gesture = None
            self._candidate_start_time = 0.0
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=False,
                reason=f"Insufficient margin ({margin:.2f} < {self.margin_threshold:.2f})",
                timestamp=now,
            )

        # Non-actionable gestures (no_gesture, fist) do not trigger robot commands
        if not is_actionable:
            self._candidate_gesture = None
            self._candidate_start_time = 0.0
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=False,
                reason="Non-actionable gesture class",
                timestamp=now,
            )

        # Check 4: State gating check
        allowed_intents = self.transitions.get(current_state, {})
        # Note: STOP is allowed from any active state
        can_transition = (intent in allowed_intents) or (is_stop and current_state not in ("IDLE", "STOPPED"))

        if not can_transition:
            # Gated by state machine
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=0.0,
                intent_fired=False,
                gated=True,
                reason=f"Intent '{intent}' not permitted in state '{current_state}'",
                timestamp=now,
            )

        # Step 3: Hold-to-fire progression
        if self._candidate_gesture != top1_gesture:
            self._candidate_gesture = top1_gesture
            self._candidate_start_time = now

        hold_elapsed = now - self._candidate_start_time
        progress = min(1.0, hold_elapsed / max(required_hold, 0.01))

        if progress >= 1.0:
            # Intent fires!
            self._last_fire_time = now
            self._last_fired_intent = intent
            self._candidate_gesture = None
            self._candidate_start_time = 0.0
            self._smoothed_probs.clear()

            logger.info(f"Intent FIRED: {intent} (gesture={top1_gesture}, conf={top1_prob:.2f})")
            return IntentResult(
                gesture=top1_gesture,
                intent=intent,
                confidence=top1_prob,
                raw_probabilities=raw_probs,
                smoothed_probabilities=dict(self._smoothed_probs),
                hold_progress=1.0,
                intent_fired=True,
                gated=False,
                reason="Hold duration satisfied, command fired",
                timestamp=now,
            )

        # Still holding
        return IntentResult(
            gesture=top1_gesture,
            intent=intent,
            confidence=top1_prob,
            raw_probabilities=raw_probs,
            smoothed_probabilities=dict(self._smoothed_probs),
            hold_progress=progress,
            intent_fired=False,
            gated=False,
            reason=f"Holding {intent} ({progress*100:.0f}%)",
            timestamp=now,
        )
