"""Integration tests for GesturePipeline."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from src.gesturegrasp.pipeline import GesturePipeline


@pytest.fixture
def pipeline():
    return GesturePipeline(use_mock_detector=True)


def test_pipeline_blank_frame(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process(frame, timestamp=0.0)

    assert res is not None
    assert res.state == "IDLE"
    assert res.hand_detected is False
    assert res.latency_ms > 0.0
    assert len(res.detected_objects) >= 1


def test_pipeline_malformed_frame_does_not_raise(pipeline):
    res = pipeline.process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res.hand_detected is False
    assert res.annotated_frame is None

    res_gray = pipeline.process(np.zeros((32, 32), dtype=np.uint8))
    assert res_gray.hand_detected is False


def _fast_robot():
    from robot.simulated import SimulatedRobot
    return SimulatedRobot(speed=10_000.0, grasp_duration=0.0)


def test_executing_dispatches_once_and_reaches_await_feedback(pipeline):
    robot = _fast_robot()
    pipeline.robot = robot
    sm = pipeline.state_machine
    sm.handle_intent("WAKE")
    sm.handle_intent("SELECT_TARGET", target_info={"label": "bottle", "center": [300.0, 200.0]})
    sm.handle_intent("CONFIRM")
    assert sm.current_state == "EXECUTING"

    for _ in range(30):
        pipeline._sync_robot()
        robot.step_simulation(dt=0.5)
    pipeline._sync_robot()

    assert sm.current_state == "AWAIT_FEEDBACK"
    navigations = [n for n in robot.get_status().notifications if "Navigating toward" in n]
    assert len(navigations) == 1


def test_wake_after_stop_releases_robot(pipeline):
    robot = _fast_robot()
    pipeline.robot = robot
    sm = pipeline.state_machine
    sm.handle_intent("WAKE")
    sm.handle_intent("STOP")
    pipeline._sync_robot()
    assert robot.get_status().is_stopped is True

    sm.handle_intent("WAKE")
    pipeline._sync_robot()
    assert sm.current_state == "LISTENING"
    assert robot.get_status().is_stopped is False
    assert robot.navigate_to(200.0, 200.0, "cup") is True


def test_pipeline_structured_dict(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process(frame, timestamp=1.0)
    data = res.to_dict()

    assert "gesture" in data
    assert "intent" in data
    assert "state" in data
    assert "latency_ms" in data
    assert "robot_state" in data


def test_commanding_hand_prefers_the_gesturing_hand(pipeline):
    # MediaPipe's hand order is arbitrary; a resting hand must not hide the command.
    resting = SimpleNamespace(landmarks="rest", handedness="Left")
    pointing = SimpleNamespace(landmarks="point", handedness="Right")
    fake = {"rest": {"no_gesture": 0.97, "point": 0.03}, "point": {"point": 0.9, "no_gesture": 0.1}}
    pipeline.predict_probabilities = lambda lm, shape: fake[lm]
    assert pipeline._commanding_hand([resting, pointing], (480, 640)) is pointing
    assert pipeline._commanding_hand([pointing, resting], (480, 640)) is pointing


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_v2"):
        GesturePipeline(model_path=tmp_path / "calibrated_model.joblib", use_mock_detector=True)


GESTURES = ["call", "point", "like", "dislike", "stop", "peace", "ok", "fist"]


@pytest.mark.skipif(not Path("data/raw/call").exists(), reason="needs the downloaded HaGRID sample images")
@pytest.mark.parametrize("gesture", GESTURES)
def test_real_photo_is_classified(gesture):
    # Same path as a camera frame from the Pi: MediaPipe -> features -> trained model.
    image = cv2.imread(str(sorted(Path(f"data/raw/{gesture}").glob("*.jpg"))[36]))
    pipe = GesturePipeline(use_mock_detector=True)
    pipe.hand_extractor.close()
    from src.gesturegrasp.mediapipe_extractor import MediaPipeHandExtractor
    pipe.hand_extractor = MediaPipeHandExtractor(static_image_mode=True)  # a still photo, not a stream
    result = pipe.process(image, timestamp=0.0, draw_annotations=False)
    assert result.hand_detected
    assert max(result.raw_probabilities, key=result.raw_probabilities.get) == gesture
