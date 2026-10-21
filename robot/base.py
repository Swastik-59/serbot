"""
Base Robot Abstraction Interface for GestureGrasp.
Defines required methods and status contracts for physical and simulated robot platforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RobotState(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    GRASPING = "GRASPING"
    DELIVERING = "DELIVERING"
    RETURNING = "RETURNING"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"
    ERROR = "ERROR"


@dataclass
class RobotStatus:
    state: RobotState = RobotState.IDLE
    current_position: tuple[float, float] = (100.0, 100.0)
    target_position: Optional[tuple[float, float]] = None
    target_label: Optional[str] = None
    holding_object: Optional[str] = None
    battery_level: float = 100.0  # percentage
    speed: float = 0.0
    navigation_progress: float = 0.0  # 0.0 to 1.0
    last_action: str = "INITIALIZED"
    last_update: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notifications: list[str] = field(default_factory=list)
    is_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "current_position": list(self.current_position),
            "target_position": list(self.target_position) if self.target_position else None,
            "target_label": self.target_label,
            "holding_object": self.holding_object,
            "battery_level": round(self.battery_level, 1),
            "speed": round(self.speed, 1),
            "navigation_progress": round(self.navigation_progress, 2),
            "last_action": self.last_action,
            "last_update": self.last_update,
            "notifications": self.notifications[-5:],
            "is_stopped": self.is_stopped,
        }


class BaseRobot(ABC):
    """Abstract Base Class for all robot execution backends."""

    @abstractmethod
    def navigate_to(self, x: float, y: float, label: Optional[str] = None) -> bool:
        """Start navigation to target coordinates (x, y)."""
        pass

    @abstractmethod
    def grasp(self, target_label: Optional[str] = None) -> bool:
        """Attempt to grasp the specified target object."""
        pass

    @abstractmethod
    def return_to_dock(self) -> bool:
        """Navigate back to dock home position."""
        pass

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Instantly stop all motors and physical motions."""
        pass

    @abstractmethod
    def notify(self, message: str) -> None:
        """Send notification or log message to robot telemetry."""
        pass

    @abstractmethod
    def get_status(self) -> RobotStatus:
        """Query current telemetry status of the robot."""
        pass

    @abstractmethod
    def release(self) -> bool:
        """Hand the held object to the user (open the gripper)."""
        pass

    def resolve_target(self, target: dict[str, Any]) -> Optional[tuple[float, float]]:
        """Where to drive for a selected target. Default: the target's 'world' position, else its centre.
        A real robot would localise the object with its own sensors here."""
        pos = target.get("world") or target.get("center")
        return (float(pos[0]), float(pos[1])) if pos is not None else None

    @abstractmethod
    def resume(self) -> None:
        """Clear an emergency stop and hold position in IDLE. The user re-woke the system."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset robot to IDLE docked state."""
        pass
