"""Unit tests for 2D pointing ray, ray-box intersection, and target consistency."""

import math
import numpy as np
import pytest
from src.gesturegrasp.pointing import (
    DetectedObject,
    PointingEngine,
    PointingRay,
    TargetConsistencyTracker,
)


@pytest.fixture
def pointing_engine():
    return PointingEngine(max_angular_deviation_deg=15.0)


def test_pointing_ray_computation(pointing_engine):
    landmarks = np.zeros((21, 3), dtype=np.float32)
    # Landmark 5 (Index MCP): (0.5, 0.5)
    landmarks[5] = [0.5, 0.5, 0.0]
    # Landmark 8 (Index Tip): (0.7, 0.5) pointing directly RIGHT along +X
    landmarks[8] = [0.7, 0.5, 0.0]

    ray = pointing_engine.compute_ray(landmarks, frame_shape=(480, 640))
    assert ray is not None
    assert isinstance(ray, PointingRay)
    # Origin is at (320, 240), tip is at (448, 240)
    assert pytest.approx(ray.direction[0], 0.01) == 1.0
    assert pytest.approx(ray.direction[1], 0.01) == 0.0
    assert pytest.approx(ray.angle_deg, 0.1) == 0.0


def test_ray_box_direct_intersection(pointing_engine):
    # Ray shooting horizontally from (100, 200) to the right
    ray = PointingRay(origin=(100.0, 200.0), tip=(150.0, 200.0), direction=(1.0, 0.0), angle_deg=0.0)

    # Box right in the line of fire: (300, 150, 400, 250)
    target_obj = DetectedObject(label="bottle", confidence=0.90, bbox_xyxy=(300.0, 150.0, 400.0, 250.0), center=(350.0, 200.0))

    selected = pointing_engine.select_target(ray, [target_obj], timestamp=0.0)
    assert selected is not None
    assert selected["label"] == "bottle"
    assert selected["direct_hit"] is True


def test_angular_deviation_rejection(pointing_engine):
    # Ray shooting horizontally (0 deg)
    ray = PointingRay(origin=(100.0, 200.0), tip=(150.0, 200.0), direction=(1.0, 0.0), angle_deg=0.0)

    # Object placed at 30 degrees elevation (exceeds 15 deg cutoff)
    # Target center at (100 + 100*cos(30), 200 - 100*sin(30)) = (186.6, 150.0)
    far_obj = DetectedObject(label="cup", confidence=0.85, bbox_xyxy=(170.0, 135.0, 200.0, 165.0), center=(185.0, 150.0))

    selected = pointing_engine.select_target(ray, [far_obj], timestamp=0.0)
    assert selected is None, "Object exceeding 15 degrees should be rejected"


def test_target_vote_needs_a_full_hold():
    tracker = TargetConsistencyTracker(required_hold_time=0.80)
    p1, c1 = tracker.update("bottle", timestamp=0.0)
    assert c1 is False
    p2, c2 = tracker.update("bottle", timestamp=0.40)
    assert 0.4 <= p2 <= 0.6 and c2 is False
    _, c3 = tracker.update("bottle", timestamp=0.80)
    assert c3 is True and tracker.current_candidate_label == "bottle"


def test_target_vote_ignores_tremor_flicker():
    tracker = TargetConsistencyTracker(required_hold_time=0.80)
    frames = [(i / 15, "cup" if i in (4, 9) else "bottle") for i in range(13)]  # 2 stray frames
    for t, label in frames:
        _, stable = tracker.update(label, timestamp=t)
    assert stable is True and tracker.current_candidate_label == "bottle"


def test_target_vote_follows_a_deliberate_switch():
    tracker = TargetConsistencyTracker(required_hold_time=0.80)
    t = 0.0
    for _ in range(15):
        tracker.update("bottle", timestamp=t)
        t += 1 / 15
    for _ in range(15):  # user now points at the cup for a full second
        _, stable = tracker.update("cup", timestamp=t)
        t += 1 / 15
    assert stable is True and tracker.current_candidate_label == "cup"


def test_no_target_for_a_full_window_clears_the_vote():
    tracker = TargetConsistencyTracker(required_hold_time=0.80)
    for i in range(13):
        tracker.update("bottle", timestamp=i / 15)
    for i in range(13, 30):
        _, stable = tracker.update(None, timestamp=i / 15)
    assert stable is False and tracker.current_candidate_label is None
