"""Unit tests for IntentEngine temporal logic, hold progression, margin gating, and STOP priority."""

from pathlib import Path

import pytest
import yaml

from src.gesturegrasp.intent_engine import IntentEngine


@pytest.fixture
def engine(tmp_path):
    # Timing logic is under test, not tuned thresholds, so pin every threshold to the shipped default.
    cfg = yaml.safe_load(Path("configs/intents.yaml").read_text(encoding="utf-8"))
    for name, meta in cfg["gestures"].items():
        meta.pop("min_conf", None)
        meta["default_threshold"] = 0.65 if name == "stop" else 0.70
    path = tmp_path / "intents.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    eng = IntentEngine(path)
    eng.reset()
    return eng


def test_margin_rejection(engine):
    # Top 1 = 0.40, Runner up = 0.35 -> margin = 0.05 < 0.20
    probs = {"point": 0.40, "call": 0.35, "no_gesture": 0.25}
    res = engine.update(probs, current_state="LISTENING", timestamp=100.0)
    assert res.intent_fired is False
    assert "Insufficient margin" in res.reason or "Below confidence threshold" in res.reason


def test_hold_to_fire(engine):
    # Provide consistent high confidence "call" (WAKE)
    # Required hold time for call is 0.60s
    probs = {"call": 0.90, "no_gesture": 0.10}

    # Frame 1 at t=0
    res1 = engine.update(probs, current_state="IDLE", timestamp=0.0)
    assert res1.intent_fired is False
    assert res1.hold_progress < 1.0

    # Frame 2 at t=0.30s (halfway)
    res2 = engine.update(probs, current_state="IDLE", timestamp=0.30)
    assert res2.intent_fired is False
    assert 0.4 <= res2.hold_progress <= 0.6

    # Frame 3 at t=0.65s (exceeds 0.60s hold)
    res3 = engine.update(probs, current_state="IDLE", timestamp=0.65)
    assert res3.intent_fired is True
    assert res3.intent == "WAKE"


def test_dropped_frame_tolerance(engine):
    probs = {"call": 0.90, "no_gesture": 0.10}
    # Start hold
    engine.update(probs, current_state="IDLE", timestamp=0.0)
    engine.update(probs, current_state="IDLE", timestamp=0.20)

    # 1 dropped frame: buffer holds
    res_drop1 = engine.update({}, current_state="IDLE", hand_detected=False, timestamp=0.25)
    assert "buffer active" in res_drop1.reason

    # Resume with valid hand at t=0.65s -> should fire because buffer preserved candidate
    res_resume = engine.update(probs, current_state="IDLE", timestamp=0.65)
    assert res_resume.intent_fired is True


def test_cooldown_refractory(engine):
    probs = {"call": 0.90, "no_gesture": 0.10}
    # Fire at t=0.65s
    res1 = engine.update(probs, current_state="IDLE", timestamp=0.0)
    res2 = engine.update(probs, current_state="IDLE", timestamp=0.65)
    assert res2.intent_fired is True

    # Immediately attempt another actionable gesture during 1.0s cooldown
    probs_point = {"point": 0.92, "no_gesture": 0.08}
    res_blocked = engine.update(probs_point, current_state="LISTENING", timestamp=0.90)
    assert res_blocked.intent_fired is False
    assert res_blocked.gated is True
    assert "Cooldown active" in res_blocked.reason


def test_stop_priority_override(engine):
    # STOP (hold time 0.25s) has emergency priority
    probs_stop = {"stop": 0.95, "no_gesture": 0.05}

    engine.update(probs_stop, current_state="EXECUTING", timestamp=0.0)
    res_stop = engine.update(probs_stop, current_state="EXECUTING", timestamp=0.30)
    assert res_stop.intent_fired is True
    assert res_stop.intent == "STOP"
