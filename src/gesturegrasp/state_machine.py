"""
Finite State Machine for GestureGrasp HRI Workflow.
Implements the exact 7 states, transition matrix, timeouts, and STOP priority override.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger("gesturegrasp.state_machine")


@dataclass
class FSMEvent:
    timestamp: float
    from_state: str
    to_state: str
    trigger: str
    details: str


class GestureStateMachine:
    """Context-aware state machine governing robot behavior."""

    VALID_STATES = [
        "IDLE",
        "LISTENING",
        "TARGET_PROPOSED",
        "EXECUTING",
        "AWAIT_FEEDBACK",
        "RETURNING",
        "STOPPED",
    ]

    def __init__(self, config_path: Optional[Path] = None, clock: Callable[[], float] = time.time):
        # The pipeline passes its frame clock, so timeouts follow frame timestamps (tests can fake time).
        self.clock = clock
        if config_path is None:
            config_path = Path("configs/intents.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        fsm_cfg = cfg.get("state_machine", {})
        self.transitions = fsm_cfg.get("transitions", {})
        self.timeouts = fsm_cfg.get("timeouts", {
            "IDLE": 0,
            "LISTENING": 30.0,
            "TARGET_PROPOSED": 15.0,
            "EXECUTING": 120.0,
            "AWAIT_FEEDBACK": 30.0,
            "RETURNING": 60.0,
            "STOPPED": 0,
        })

        self.current_state = fsm_cfg.get("initial_state", "IDLE")
        self.proposed_target: Optional[dict[str, Any]] = None
        self.state_entry_time = self.clock()
        self.event_log: list[FSMEvent] = []

        self._record_event("INIT", self.current_state, "System startup", "INITIALIZED")

    def _record_event(self, from_state: str, to_state: str, trigger: str, details: str) -> None:
        event = FSMEvent(
            timestamp=self.clock(),
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            details=details,
        )
        self.event_log.append(event)
        if len(self.event_log) > 50:
            self.event_log.pop(0)

    def transition_to(self, new_state: str, trigger: str, details: str = "") -> bool:
        if new_state not in self.VALID_STATES:
            logger.error(f"Invalid target state: {new_state}")
            return False

        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = self.clock()
        self._record_event(old_state, new_state, trigger, details)
        logger.info(f"FSM Transition: [{old_state}] --({trigger})--> [{new_state}] ({details})")
        return True

    def handle_intent(
        self,
        intent: str,
        target_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Process an actionable intent from the IntentEngine."""
        # 1. EMERGENCY STOP Override: Allowed from all states except IDLE
        if intent == "STOP":
            if self.current_state != "IDLE":
                return self.transition_to("STOPPED", "STOP", "Emergency stop priority override")
            return False

        # 2. State-specific intent processing
        if self.current_state == "IDLE":
            if intent == "WAKE":
                return self.transition_to("LISTENING", "WAKE", "User wake gesture recognized")

        elif self.current_state == "LISTENING":
            if intent == "SELECT_TARGET":
                if target_info and target_info.get("label"):
                    self.proposed_target = dict(target_info)
                    return self.transition_to(
                        "TARGET_PROPOSED",
                        "SELECT_TARGET",
                        f"Target selected: {target_info.get('label')}",
                    )
                else:
                    logger.debug("SELECT_TARGET received in LISTENING, but no valid object targeted.")
                    return False
            elif intent == "RETURN_HOME":
                return self.transition_to("RETURNING", "RETURN_HOME", "Aborted to dock")

        elif self.current_state == "TARGET_PROPOSED":
            if intent == "CONFIRM":
                return self.transition_to(
                    "EXECUTING",
                    "CONFIRM",
                    f"User confirmed retrieval for: {self.proposed_target.get('label') if self.proposed_target else 'object'}",
                )
            elif intent == "REJECT":
                self.proposed_target = None
                return self.transition_to("LISTENING", "REJECT", "User rejected target proposal")
            elif intent == "SELECT_TARGET":
                if target_info and target_info.get("label"):
                    self.proposed_target = dict(target_info)
                    self._record_event(
                        self.current_state,
                        self.current_state,
                        "RESELECT_TARGET",
                        f"Updated target to: {target_info.get('label')}",
                    )
                    self.state_entry_time = self.clock()  # reset proposal timeout
                    return True
            elif intent == "RETURN_HOME":
                return self.transition_to("RETURNING", "RETURN_HOME", "Aborted to dock")

        elif self.current_state == "EXECUTING":
            if intent == "RETURN_HOME":
                return self.transition_to("RETURNING", "RETURN_HOME", "Execution aborted to dock")

        elif self.current_state == "AWAIT_FEEDBACK":
            if intent == "ACKNOWLEDGE":
                self.proposed_target = None
                return self.transition_to("IDLE", "ACKNOWLEDGE", "Interaction completed successfully")
            elif intent == "REJECT":
                # Same recovery as TARGET_PROPOSED --REJECT--> LISTENING:
                # the robot keeps or returns the object, and the user can point again.
                self.proposed_target = None
                return self.transition_to(
                    "LISTENING",
                    "REJECT",
                    "User rejected the delivered object; robot holds or returns it",
                )
            elif intent == "WAKE":
                self.proposed_target = None
                return self.transition_to("LISTENING", "WAKE", "New interaction started")
            elif intent == "RETURN_HOME":
                return self.transition_to("RETURNING", "RETURN_HOME", "Interaction ended with dock return")

        elif self.current_state == "RETURNING":
            # In returning, manual/gesture STOP can pause return
            pass

        elif self.current_state == "STOPPED":
            if intent == "WAKE":
                return self.transition_to("LISTENING", "WAKE", "Resumed from stopped state")

        logger.debug(f"Ignored intent '{intent}' in state '{self.current_state}'")
        return False

    def notify_task_complete(self) -> bool:
        """Called when robot finishes retrieval task."""
        if self.current_state == "EXECUTING":
            return self.transition_to("AWAIT_FEEDBACK", "TASK_COMPLETE", "Robot reached user with object")
        return False

    def notify_dock_reached(self) -> bool:
        """Called when robot returns to dock."""
        if self.current_state == "RETURNING":
            self.proposed_target = None
            return self.transition_to("IDLE", "DOCK_REACHED", "Robot docked and charging")
        return False

    def check_timeouts(self, now: Optional[float] = None) -> Optional[str]:
        """Check if current state has timed out."""
        curr_time = self.clock() if now is None else now
        timeout_limit = self.timeouts.get(self.current_state, 0)
        if timeout_limit <= 0:
            return None

        elapsed = curr_time - self.state_entry_time
        if elapsed >= timeout_limit:
            old = self.current_state
            if old == "LISTENING":
                self.transition_to("IDLE", "TIMEOUT", f"Listening timed out after {timeout_limit}s")
            elif old == "TARGET_PROPOSED":
                self.proposed_target = None
                self.transition_to("LISTENING", "TIMEOUT", f"Target proposal timed out after {timeout_limit}s")
            elif old == "AWAIT_FEEDBACK":
                self.proposed_target = None
                self.transition_to("IDLE", "TIMEOUT", f"Awaiting feedback timed out after {timeout_limit}s")
            elif old == "RETURNING":
                self.transition_to("IDLE", "TIMEOUT", "Docking timeout fallback")
            elif old == "EXECUTING":
                self.transition_to("LISTENING", "TIMEOUT", "Execution timeout fallback")
            return old
        return None

    def reset(self) -> None:
        self.current_state = "IDLE"
        self.proposed_target = None
        self.state_entry_time = self.clock()
        self._record_event("RESET", "IDLE", "MANUAL_RESET", "State machine reset to IDLE")
