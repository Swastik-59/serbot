"""Speaks prompts on the Pi through espeak-ng. Never blocks the processing loop."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger("gesturegrasp.speech")


class Speaker:
    """A newer prompt replaces one still being spoken. Missing espeak-ng disables speech, not the server."""

    def __init__(self, engine: str = "espeak-ng", words_per_minute: int = 140):
        self.words_per_minute = words_per_minute
        self._engine = shutil.which(engine)
        self._current: Optional[subprocess.Popen] = None
        if self._engine is None:
            logger.warning("%s not found, so prompts will not be spoken. Install: sudo apt install espeak-ng", engine)

    @property
    def available(self) -> bool:
        return self._engine is not None

    def say(self, text: str) -> None:
        if self._engine is None or not text:
            return
        if self._current is not None and self._current.poll() is None:
            self._current.terminate()
        self._current = subprocess.Popen([self._engine, "-s", str(self.words_per_minute), text],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self) -> None:
        if self._current is not None and self._current.poll() is None:
            self._current.terminate()
