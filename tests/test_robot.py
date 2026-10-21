"""Unit tests for robot abstraction interface and simulated robot backend."""

import pytest
from robot.base import RobotState
from robot.simulated import SimulatedRobot


@pytest.fixture
def robot():
    return SimulatedRobot(dock_pos=(100.0, 100.0), speed=100.0, grasp_duration=1.0)


def test_initial_dock_state(robot):
    status = robot.get_status()
    assert status.state == RobotState.IDLE
    assert status.current_position == (100.0, 100.0)
    assert status.is_stopped is False


def test_navigation_and_progress(robot):
    robot.navigate_to(200.0, 100.0, label="bottle")
    status = robot.get_status()
    assert status.state == RobotState.NAVIGATING
    assert status.target_label == "bottle"

    # Step simulation 0.5s at 100 u/s -> traveled 50 units (halfway)
    robot.step_simulation(dt=0.5)
    status2 = robot.get_status()
    assert 0.4 <= status2.navigation_progress <= 0.6
    assert status2.current_position[0] > 140.0


def test_arrival_starts_grasp_without_recursing(robot):
    robot.navigate_to(450.0, 220.0, "bottle")
    for _ in range(12):
        robot.step_simulation(dt=0.6)
    assert robot.get_status().state.value in {"GRASPING", "DELIVERING", "IDLE"}


def test_emergency_stop_halts_movement(robot):
    robot.navigate_to(300.0, 300.0, label="cup")
    robot.step_simulation(dt=0.2)

    robot.emergency_stop()
    status = robot.get_status()
    assert status.state == RobotState.EMERGENCY_STOPPED
    assert status.is_stopped is True
    assert status.speed == 0.0

    # Further navigation must be rejected while stopped
    can_nav = robot.navigate_to(100.0, 100.0)
    assert can_nav is False


def test_resume_clears_stop_in_place(robot):
    robot.navigate_to(300.0, 100.0, label="cup")
    robot.step_simulation(dt=0.5)
    robot.emergency_stop()
    stopped_at = robot.get_status().current_position

    robot.resume()
    status = robot.get_status()
    assert status.is_stopped is False
    assert status.state == RobotState.IDLE
    assert status.current_position == stopped_at


def test_reset_clears_stop(robot):
    robot.emergency_stop()
    assert robot.get_status().is_stopped is True

    robot.reset()
    status = robot.get_status()
    assert status.state == RobotState.IDLE
    assert status.is_stopped is False
    assert status.current_position == (100.0, 100.0)
