"""Unit tests for GestureStateMachine transitions, timeouts, and safety gating."""

import pytest
from src.gesturegrasp.state_machine import GestureStateMachine


@pytest.fixture
def fsm():
    sm = GestureStateMachine()
    sm.reset()
    return sm


def test_initial_state(fsm):
    assert fsm.current_state == "IDLE"


def test_nominal_hri_flow(fsm):
    # IDLE --WAKE--> LISTENING
    assert fsm.handle_intent("WAKE") is True
    assert fsm.current_state == "LISTENING"

    # LISTENING --SELECT_TARGET + object--> TARGET_PROPOSED
    target = {"label": "bottle", "confidence": 0.88, "center": (450.0, 220.0)}
    assert fsm.handle_intent("SELECT_TARGET", target_info=target) is True
    assert fsm.current_state == "TARGET_PROPOSED"
    assert fsm.proposed_target["label"] == "bottle"

    # TARGET_PROPOSED --CONFIRM--> EXECUTING
    assert fsm.handle_intent("CONFIRM") is True
    assert fsm.current_state == "EXECUTING"

    # EXECUTING --TASK_COMPLETE--> AWAIT_FEEDBACK
    assert fsm.notify_task_complete() is True
    assert fsm.current_state == "AWAIT_FEEDBACK"

    # AWAIT_FEEDBACK --ACKNOWLEDGE--> IDLE
    assert fsm.handle_intent("ACKNOWLEDGE") is True
    assert fsm.current_state == "IDLE"


def test_rejection_flow(fsm):
    fsm.handle_intent("WAKE")
    target = {"label": "cup", "confidence": 0.82}
    fsm.handle_intent("SELECT_TARGET", target_info=target)
    assert fsm.current_state == "TARGET_PROPOSED"

    # User rejects proposed target
    assert fsm.handle_intent("REJECT") is True
    assert fsm.current_state == "LISTENING"
    assert fsm.proposed_target is None


def test_await_feedback_reject_returns_to_listening(fsm):
    fsm.handle_intent("WAKE")
    fsm.handle_intent("SELECT_TARGET", target_info={"label": "bottle", "confidence": 0.9})
    fsm.handle_intent("CONFIRM")
    assert fsm.notify_task_complete() is True
    assert fsm.current_state == "AWAIT_FEEDBACK"
    assert fsm.proposed_target is not None

    assert fsm.handle_intent("REJECT") is True
    assert fsm.current_state == "LISTENING"
    assert fsm.proposed_target is None


def test_invalid_transition_rejected(fsm):
    # CONFIRM directly from IDLE must be ignored
    assert fsm.current_state == "IDLE"
    assert fsm.handle_intent("CONFIRM") is False
    assert fsm.current_state == "IDLE"


def test_emergency_stop_from_any_active_state(fsm):
    # STOP from LISTENING
    fsm.handle_intent("WAKE")
    assert fsm.handle_intent("STOP") is True
    assert fsm.current_state == "STOPPED"

    # Resume with WAKE
    assert fsm.handle_intent("WAKE") is True
    assert fsm.current_state == "LISTENING"


def test_state_timeout(fsm):
    fsm.handle_intent("WAKE")
    assert fsm.current_state == "LISTENING"

    # Fast forward past 30s timeout
    timed_out_state = fsm.check_timeouts(now=fsm.state_entry_time + 35.0)
    assert timed_out_state == "LISTENING"
    assert fsm.current_state == "IDLE"
