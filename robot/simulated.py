"""
Simulated Robot Backend for GestureGrasp.

Top-down 2D arena (units = cm-ish pixels). A fetch is:
    navigate_to(object) -> grasp (grasp_duration) -> DELIVERING to the user -> IDLE holding the object
    release() hands it over; return_to_dock() drives home.
Stands in for the real robot until one is connected (implement robot.base.BaseRobot for it).
The clock is injectable: wall time live, a fake clock in tests for exact, repeatable timings.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from robot.base import BaseRobot, RobotState, RobotStatus

DEFAULT_ARENA_OBJECTS: dict[str, tuple[float, float]] = {
    "bottle": (450.0, 220.0),
    "cup": (280.0, 360.0),
    "book": (380.0, 420.0),
    "cell phone": (520.0, 310.0),
    "remote": (200.0, 200.0),
    "apple": (350.0, 180.0),
}


class SimulatedRobot(BaseRobot):
    """Full-featured simulated robot for demonstration and offline testing."""

    def __init__(
        self,
        dock_pos: tuple[float, float] = (100.0, 100.0),
        user_pos: tuple[float, float] = (300.0, 520.0),
        speed: float = 120.0,
        grasp_duration: float = 1.5,
        clock: Callable[[], float] = time.time,
        arena_objects: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self.dock_pos = dock_pos
        self.user_pos = user_pos
        self.speed = speed  # units per second
        self.grasp_duration = grasp_duration
        self.clock = clock
        self._status = RobotStatus(
            state=RobotState.IDLE,
            current_position=dock_pos,
            battery_level=98.5,
        )
        self._last_tick_time = self.clock()
        self._navigation_start_pos = dock_pos
        self._grasp_elapsed = 0.0
        self._advancing = False
        self.odometer = 0.0
        self.arena_objects: dict[str, tuple[float, float]] = dict(arena_objects or DEFAULT_ARENA_OBJECTS)

    # -- BaseRobot ------------------------------------------------------------------------------
    def notify(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._status.notifications.append(entry)
        if len(self._status.notifications) > 20:
            self._status.notifications.pop(0)

    def get_status(self) -> RobotStatus:
        self._tick()
        self._status.last_update = datetime.now(timezone.utc).isoformat()
        return self._status

    def resolve_target(self, target: dict[str, Any]) -> Optional[tuple[float, float]]:
        label = target.get("label")
        if label in self.arena_objects:
            return self.arena_objects[label]
        return super().resolve_target(target)

    def navigate_to(self, x: float, y: float, label: Optional[str] = None) -> bool:
        if self._status.is_stopped:
            self.notify("ERROR: Cannot navigate while EMERGENCY STOPPED.")
            return False
        self._tick()
        self._start_drive(RobotState.NAVIGATING, (float(x), float(y)), label)
        self._status.last_action = f"NAVIGATE_TO({x:.1f}, {y:.1f})"
        self.notify(f"Navigating toward {label or 'target'} at ({x:.0f}, {y:.0f})")
        return True

    def grasp(self, target_label: Optional[str] = None) -> bool:
        if self._status.is_stopped:
            self.notify("ERROR: Cannot grasp while EMERGENCY STOPPED.")
            return False
        self._status.state = RobotState.GRASPING
        self._status.speed = 0.0
        self._status.target_label = target_label
        self._status.last_action = f"GRASP({target_label or 'object'})"
        self._grasp_elapsed = 0.0
        self.notify(f"Initiating mechanical grasp for {target_label or 'target'}")
        return True

    def deliver_to_user(self) -> bool:
        if self._status.is_stopped:
            return False
        self._start_drive(RobotState.DELIVERING, self.user_pos, "USER")
        self._status.last_action = "DELIVER_TO_USER"
        self.notify(f"Bringing {self._status.holding_object} to the user")
        return True

    def release(self) -> bool:
        if self._status.is_stopped or not self._status.holding_object:
            return False
        self.notify(f"Handed {self._status.holding_object} to the user")
        self._status.holding_object = None
        self._status.last_action = "RELEASE"
        return True

    def return_to_dock(self) -> bool:
        if self._status.is_stopped:
            self.notify("ERROR: Cannot return while EMERGENCY STOPPED.")
            return False
        self._tick()
        self._start_drive(RobotState.RETURNING, self.dock_pos, "DOCK")
        self._status.last_action = "RETURN_TO_DOCK"
        self.notify("Returning to charging dock")
        return True

    def emergency_stop(self) -> bool:
        self._status.state = RobotState.EMERGENCY_STOPPED
        self._status.is_stopped = True
        self._status.speed = 0.0
        self._status.target_position = None
        self._status.last_action = "EMERGENCY_STOP"
        self.notify("CRITICAL: EMERGENCY STOP ENGAGED. All movement arrested.")
        return True

    def resume(self) -> None:
        self._status.state = RobotState.IDLE
        self._status.is_stopped = False
        self._status.target_position = None
        self._status.speed = 0.0
        self._last_tick_time = self.clock()
        self.notify("Emergency stop cleared. Holding position.")

    def reset(self) -> None:
        self._status.state = RobotState.IDLE
        self._status.current_position = self.dock_pos
        self._status.target_position = None
        self._status.target_label = None
        self._status.holding_object = None
        self._status.speed = 0.0
        self._status.navigation_progress = 0.0
        self._status.last_action = "RESET_TO_IDLE"
        self._status.is_stopped = False
        self._last_tick_time = self.clock()
        self.odometer = 0.0
        self.notify("Robot reset to idle docked state.")

    # -- simulation -----------------------------------------------------------------------------
    def step_simulation(self, dt: float = 0.1) -> RobotStatus:
        """Explicitly step simulation by delta time (useful in non-realtime loops)."""
        self._advance(dt)
        return self._status

    def _start_drive(self, state: RobotState, target: tuple[float, float], label: Optional[str]) -> None:
        self._status.state = state
        self._status.target_position = target
        if label not in ("USER", "DOCK"):
            self._status.target_label = label
        self._status.navigation_progress = 0.0
        self._navigation_start_pos = self._status.current_position
        self._last_tick_time = self.clock()

    def _tick(self) -> None:
        now = self.clock()
        dt = min(max(now - self._last_tick_time, 0.0), 0.5)  # clamp to avoid huge delta jumps
        self._last_tick_time = now
        self._advance(dt)

    def _advance(self, dt: float) -> None:
        # Arrival handlers call methods that tick again; never re-enter on the same arrival.
        if self._advancing or self._status.is_stopped:
            return
        self._advancing = True
        try:
            self._advance_body(dt)
        finally:
            self._advancing = False

    def _advance_body(self, dt: float) -> None:
        self._status.battery_level = max(5.0, self._status.battery_level - 0.005 * dt)
        state = self._status.state

        if state in (RobotState.NAVIGATING, RobotState.DELIVERING, RobotState.RETURNING):
            if not self._status.target_position:
                self._status.state = RobotState.IDLE
                return
            tx, ty = self._status.target_position
            cx, cy = self._status.current_position
            dist = math.hypot(tx - cx, ty - cy)
            step = self.speed * dt
            if dist <= step or dist < 2.0:
                self.odometer += dist
                self._status.current_position = (tx, ty)
                self._status.speed = 0.0
                self._status.navigation_progress = 1.0
                self._status.target_position = None
                self._arrive(state)
            else:
                self.odometer += step
                self._status.speed = self.speed
                self._status.current_position = (cx + (tx - cx) / dist * step, cy + (ty - cy) / dist * step)
                sx, sy = self._navigation_start_pos
                total = math.hypot(tx - sx, ty - sy)
                if total > 1e-3:
                    done = math.hypot(self._status.current_position[0] - sx, self._status.current_position[1] - sy)
                    self._status.navigation_progress = min(1.0, done / total)

        elif state == RobotState.GRASPING:
            self._grasp_elapsed += dt
            if self._grasp_elapsed >= self.grasp_duration:
                self._status.holding_object = self._status.target_label or "object"
                self.notify(f"Object {self._status.holding_object} securely grasped.")
                self.deliver_to_user()

    def _arrive(self, state: RobotState) -> None:
        if state == RobotState.NAVIGATING:
            self.notify(f"Reached {self._status.target_label}. Grasping.")
            self.grasp(self._status.target_label)
        elif state == RobotState.DELIVERING:
            self._status.state = RobotState.IDLE
            self.notify(f"Arrived at the user with {self._status.holding_object}.")
        elif state == RobotState.RETURNING:
            self._status.state = RobotState.IDLE
            self._status.target_label = None
            self.notify("Docking complete. Robot in home station.")
