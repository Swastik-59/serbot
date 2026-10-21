"""Robot backends. A real robot later implements robot.base.BaseRobot; until then SimulatedRobot runs."""

from robot.base import BaseRobot, RobotState, RobotStatus
from robot.simulated import SimulatedRobot

__all__ = ["BaseRobot", "RobotState", "RobotStatus", "SimulatedRobot"]
