"""Test configuration files schema and integrity."""

from pathlib import Path
import pytest
import yaml

INTENTS_FILE = Path("configs/intents.yaml")


def test_configs_exist():
    assert INTENTS_FILE.exists(), "configs/intents.yaml must exist"


def test_intents_schema():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "gestures" in cfg
    gestures = cfg["gestures"]

    expected_gestures = [
        "call", "point", "like", "dislike", "stop", "peace", "ok", "no_gesture", "fist"
    ]
    for g in expected_gestures:
        assert g in gestures, f"Gesture '{g}' missing from configs/intents.yaml"
        g_data = gestures[g]
        assert "intent" in g_data
        assert "actionable" in g_data
        assert "min_conf" in g_data
        assert 0.0 <= g_data["min_conf"] <= 1.0
        assert "hold_time" in g_data
        assert g_data["hold_time"] >= 0.0

    # Actionable vs Non-actionable
    assert gestures["no_gesture"]["actionable"] is False
    assert gestures["fist"]["actionable"] is False
    assert gestures["stop"]["actionable"] is True
    assert gestures["point"]["actionable"] is True


def test_state_machine_schema():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "state_machine" in cfg
    fsm_cfg = cfg["state_machine"]
    expected_states = [
        "IDLE", "LISTENING", "TARGET_PROPOSED", "EXECUTING", "AWAIT_FEEDBACK", "RETURNING", "STOPPED"
    ]
    for st in expected_states:
        assert st in fsm_cfg["states"]
        assert st in fsm_cfg["timeouts"]

    transitions = fsm_cfg["transitions"]
    assert "IDLE" in transitions
    assert transitions["IDLE"]["WAKE"] == "LISTENING"
    assert transitions["LISTENING"]["SELECT_TARGET"] == "TARGET_PROPOSED"
    assert transitions["TARGET_PROPOSED"]["CONFIRM"] == "EXECUTING"
    assert transitions["TARGET_PROPOSED"]["REJECT"] == "LISTENING"
    assert transitions["AWAIT_FEEDBACK"]["REJECT"] == "LISTENING"
    reject_states = cfg["gestures"]["dislike"]["states"]
    assert "AWAIT_FEEDBACK" in reject_states
    assert "TARGET_PROPOSED" in reject_states


def test_pointing_schema():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pointing = cfg.get("pointing", {})
    assert pointing.get("origin_landmark") == 5
    assert pointing.get("direction_landmark") == 8
    assert pointing.get("max_angular_deviation_deg") == 15.0
    assert "allowed_classes" in pointing
    assert len(pointing["allowed_classes"]) >= 6
