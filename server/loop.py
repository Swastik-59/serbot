"""The processing loop on the Pi: newest camera frame -> pipeline -> stored result -> repeat."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from src.gesturegrasp.camera import CameraSource

logger = logging.getLogger("gesturegrasp.loop")


class CameraLoop:
    """Runs on its own thread. Each frame id is processed at most once, and frames that arrive while the
    pipeline is busy are skipped, so a slow pipeline lowers the frame rate instead of adding delay."""

    def __init__(self, camera: CameraSource, process: Callable[[np.ndarray, float, int], dict[str, Any]],
                 paused: Callable[[], bool] = lambda: False):
        self.camera = camera
        self._process = process
        self._paused = paused
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished: deque[float] = deque(maxlen=30)
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.latency_ms = 0.0
        self.frame_age_ms = 0.0

    def start(self) -> "CameraLoop":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-loop", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        last_id = 0
        while not self._stop.is_set():
            frame = self.camera.wait_for_frame(last_id, timeout=0.25)
            if frame is None:
                continue
            image, frame_id, stamp = frame
            if last_id:
                self.skipped += max(0, frame_id - last_id - 1)
            last_id = frame_id
            if self._paused():
                continue
            started = time.perf_counter()
            try:
                self._process(image, stamp, frame_id)
            except Exception:  # one bad frame must not end the loop that STOP gestures depend on
                self.errors += 1
                logger.exception("Frame %d failed", frame_id)
                continue
            self.processed += 1
            self.latency_ms = (time.perf_counter() - started) * 1000.0
            self.frame_age_ms = (time.time() - stamp) * 1000.0
            self._finished.append(time.monotonic())

    def stats(self) -> dict[str, Any]:
        stamps = list(self._finished)
        fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
        return {
            "fps": round(fps, 1),
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "latency_ms": round(self.latency_ms, 1),
            "frame_age_ms": round(self.frame_age_ms, 1),
            "paused": self._paused(),
        }
