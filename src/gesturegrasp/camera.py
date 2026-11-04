"""
Camera sources for the edge pipeline.

The Raspberry Pi 5 pulls video from an AI-Thinker ESP32-CAM over Wi-Fi. Every source keeps only the
newest frame: there is no queue, so a slow pipeline skips frames instead of acting on old ones.

Why MJPEG is parsed here instead of using cv2.VideoCapture:
    - FFmpeg buffers inside VideoCapture, which adds latency we cannot see or control.
    - A dead stream can block VideoCapture.read() with no usable timeout. Here a socket read timeout
      plus a stall check turn a rebooting ESP32 into a clean reconnect.
    - Splitting on JPEG markers costs almost nothing, and frames we skip are never decoded.

Why bytes are read with read1() instead of requests' iter_content():
    iter_content(n) can wait for n bytes, which holds the end of a frame back until the next frame
    arrives (one frame of added latency). read1() returns whatever has arrived.

The ESP32 stream server handles ONE client, so the Pi must be the only consumer of /stream.
Browsers watch the Pi's /video, never the ESP32 directly.
"""

from __future__ import annotations

import http.client
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import urllib3

logger = logging.getLogger("gesturegrasp.camera")

CONNECTING = "connecting"
STREAMING = "streaming"
LOST = "lost"

DEFAULT_CAMERA_URL = "http://esp32cam.local:81/stream"
FLIP_MODES = ("none", "h", "v", "both")
MAX_FRAME_BYTES = 4 * 1024 * 1024

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_SIZE = 64 * 1024
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_BARE_HOST = re.compile(r"^[A-Za-z0-9.-]+(:\d+)?$")
_FETCH_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException, OSError)

Frame = tuple[np.ndarray, int, float]


class MjpegParser:
    """Splits a byte stream into JPEG frames by SOI/EOI markers. Keeps partial frames between chunks.

    Entropy-coded JPEG data escapes 0xFF, so FF D9 only ends a frame. The OV2640 writes no embedded
    thumbnails, so there are no nested images to skip over.
    """

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES):
        self.max_frame_bytes = max_frame_bytes
        self.dropped = 0
        self._buf = bytearray()
        self._scan = 0
        self._in_frame = False

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk
        frames: list[bytes] = []
        while True:
            if not self._in_frame:
                start = self._buf.find(_SOI)
                if start < 0:
                    # A trailing 0xFF may be the first half of an SOI split across two chunks.
                    keep = 1 if self._buf.endswith(b"\xff") else 0
                    del self._buf[:len(self._buf) - keep]
                    return frames
                del self._buf[:start]
                self._in_frame = True
                self._scan = len(_SOI)
            end = self._buf.find(_EOI, self._scan)
            if end < 0:
                if len(self._buf) > self.max_frame_bytes:
                    self.dropped += 1
                    self._buf.clear()
                    self._in_frame = False
                else:
                    self._scan = max(len(_SOI), len(self._buf) - 1)  # EOI may straddle chunks
                return frames
            frames.append(bytes(self._buf[:end + len(_EOI)]))
            del self._buf[:end + len(_EOI)]
            self._in_frame = False


def _strip_credentials(url: str) -> str:
    parts = urlsplit(url)
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class CameraSource:
    """Latest-frame slot shared by the reader thread and the consumer."""

    kind = "base"

    def __init__(self, flip: str = "none"):
        if flip not in FLIP_MODES:
            raise ValueError(f"flip must be one of {FLIP_MODES}, got {flip!r}")
        self.flip = flip
        self._hardware_flip = False
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._payload: Optional[Union[bytes, np.ndarray]] = None
        self._frame_id = 0
        self._timestamp = 0.0
        self._decoded: Optional[Frame] = None
        self._bad_id = 0
        self._arrivals: deque[float] = deque(maxlen=30)
        self._state = CONNECTING
        self.frames_received = 0
        self.decode_errors = 0
        self.decode_ms = 0.0
        self.last_error: Optional[str] = None

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self) -> "CameraSource":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=f"camera-{self.kind}", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        raise NotImplementedError

    # -- producer side --------------------------------------------------------------------------
    def _publish(self, payload: Union[bytes, np.ndarray], arrivals: int = 1) -> None:
        """Replace the slot. Whatever was there and never read is simply gone."""
        now = time.time()
        with self._cond:
            self._payload = payload
            self._frame_id += 1
            self._timestamp = now
            self._decoded = None
            self.frames_received += arrivals
            self._arrivals.extend([now] * arrivals)
            if self._state != STREAMING:
                logger.info("Camera %s -> %s", self._state, STREAMING)
            self._state = STREAMING
            self._cond.notify_all()

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        with self._cond:
            if state != self._state:
                logger.info("Camera %s -> %s%s", self._state, state, f" ({error})" if error else "")
            self._state = state
            if error:
                self.last_error = error

    # -- consumer side --------------------------------------------------------------------------
    def read(self) -> Optional[Frame]:
        """Newest frame as (frame_bgr, frame_id, timestamp), or None if there is none yet or it is
        undecodable. Decoding happens here, so frames nobody reads are never decoded."""
        with self._cond:
            if self._decoded is not None:
                return self._decoded
            payload, frame_id, stamp = self._payload, self._frame_id, self._timestamp
        if payload is None:
            return None
        started = time.perf_counter()
        frame = payload if isinstance(payload, np.ndarray) else _decode_jpeg(payload)
        if not isinstance(payload, np.ndarray):
            self.decode_ms = (time.perf_counter() - started) * 1000.0
        if frame is None:
            with self._cond:
                self.decode_errors += 1
                self._bad_id = frame_id
            return None
        if self.flip != "none" and not self._hardware_flip:
            frame = cv2.flip(frame, {"h": 1, "v": 0, "both": -1}[self.flip])
        decoded = (frame, frame_id, stamp)
        with self._cond:
            if self._frame_id == frame_id:
                self._decoded = decoded
        return decoded

    def wait_for_frame(self, after_id: int, timeout: float = 0.5) -> Optional[Frame]:
        """Block until a frame newer than after_id exists, then return the newest. None on timeout."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._frame_id > max(after_id, self._bad_id) or self._stop.is_set(), timeout)
            if not ready or self._stop.is_set():
                return None
        return self.read()

    def status(self) -> dict[str, Any]:
        with self._cond:
            stamps = list(self._arrivals)
            stamp, frame_id, state = self._timestamp, self._frame_id, self._state
        fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
        return {
            "kind": self.kind,
            "state": state,
            "fps": round(fps, 1),
            "last_frame_age_ms": round((time.time() - stamp) * 1000.0, 1) if stamp else None,
            "frame_id": frame_id,
            "frames_received": self.frames_received,
            "decode_errors": self.decode_errors,
            "flip": self.flip,
            "last_error": self.last_error,
        }


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size else None


class Esp32CamSource(CameraSource):
    """Reads the ESP32-CAM's MJPEG stream on a background thread and keeps only the newest frame.

    Reconnects with exponential backoff when the board drops or reboots. If /stream keeps failing it
    polls /capture instead and retries the stream every so often. Settings go through /control on
    every connection because the ESP32 forgets them when it reboots.
    """

    kind = "esp32cam"

    def __init__(self, stream_url: str, control_url: Optional[str] = None, framesize: Optional[str] = None,
                 quality: Optional[int] = None, flip: str = "none", connect_timeout: float = 3.0,
                 read_timeout: float = 2.5, stall_timeout: float = 3.0, backoff_min: float = 0.5,
                 backoff_max: float = 8.0, capture_after_failures: int = 2, capture_retry_seconds: float = 15.0,
                 capture_max_fps: float = 10.0, status_interval: float = 5.0):
        super().__init__(flip=flip)
        self.stream_url = stream_url
        self.control_url = control_url.rstrip("/") if control_url else None
        self.framesize = framesize
        self.quality = quality
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.stall_timeout = stall_timeout
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.capture_after_failures = capture_after_failures
        self.capture_retry_seconds = capture_retry_seconds
        self.capture_max_fps = capture_max_fps
        self.status_interval = status_interval
        self.mode = "stream"
        self.reconnects = 0
        self.esp32_status: Optional[dict[str, Any]] = None
        self._status_thread: Optional[threading.Thread] = None
        self._session = requests.Session()
        self._session.trust_env = False  # the camera is on the LAN; never send it through a proxy

    def start(self) -> "Esp32CamSource":
        super().start()
        if self.control_url and self._status_thread is None:
            self._status_thread = threading.Thread(target=self._poll_status, name="camera-status", daemon=True)
            self._status_thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        super().stop(timeout)
        if self._status_thread is not None:
            self._status_thread.join(timeout)
            self._status_thread = None

    def status(self) -> dict[str, Any]:
        return {**super().status(), "url": _strip_credentials(self.stream_url), "mode": self.mode,
                "reconnects": self.reconnects, "esp32": self.esp32_status}

    # -- reader thread --------------------------------------------------------------------------
    def _run(self) -> None:
        backoff = self.backoff_min
        failures = 0
        while not self._stop.is_set():
            self.mode = "stream"
            got_frames = self._read_stream()
            if self._stop.is_set():
                return
            failures = 0 if got_frames else failures + 1
            if got_frames:
                backoff = self.backoff_min
            self.reconnects += 1
            self._set_state(LOST)
            if failures >= self.capture_after_failures and self.control_url and self._poll_capture():
                backoff = self.backoff_min
                continue
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.backoff_max)

    def _read_stream(self) -> bool:
        """One connection's lifetime. True if it delivered at least one frame."""
        self._apply_settings()
        parser = MjpegParser()
        got_frames = False
        try:
            with self._session.get(self.stream_url, stream=True,
                                   timeout=(self.connect_timeout, self.read_timeout)) as resp:
                resp.raise_for_status()
                last_frame = time.monotonic()
                while not self._stop.is_set():
                    chunk = resp.raw.read1(_READ_SIZE)
                    if not chunk:
                        raise ConnectionError("stream ended")
                    frames = parser.feed(chunk)
                    if frames:
                        self._publish(frames[-1], arrivals=len(frames))
                        got_frames = True
                        last_frame = time.monotonic()
                    elif time.monotonic() - last_frame > self.stall_timeout:
                        raise TimeoutError(f"no complete frame for {self.stall_timeout:.1f} s")
        except _FETCH_ERRORS as exc:
            if not self._stop.is_set():
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Camera stream failed: %s", self.last_error)
        return got_frames

    def _poll_capture(self) -> bool:
        """Poll /capture until the stream is due another try. True if it produced any frame."""
        interval = 1.0 / self.capture_max_fps
        deadline = time.monotonic() + self.capture_retry_seconds
        got_frames = False
        self.mode = "capture"
        while not self._stop.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
            try:
                resp = self._session.get(f"{self.control_url}/capture",
                                         timeout=(self.connect_timeout, self.read_timeout))
                resp.raise_for_status()
            except _FETCH_ERRORS as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return got_frames
            if resp.content.startswith(_SOI):
                self._publish(resp.content)
                got_frames = True
            self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        return got_frames

    def _apply_settings(self) -> None:
        """Best effort. A failure here must not stop the stream, so it only logs."""
        self._hardware_flip = False
        if not self.control_url:
            return
        settings: list[tuple[str, Union[str, int]]] = []
        if self.framesize:
            settings.append(("framesize", self.framesize))
        if self.quality is not None:
            settings.append(("quality", self.quality))
        if self.flip != "none":
            settings.append(("hmirror", int(self.flip in ("h", "both"))))
            settings.append(("vflip", int(self.flip in ("v", "both"))))
        try:
            for var, val in settings:
                resp = self._session.get(f"{self.control_url}/control", params={"var": var, "val": val},
                                         timeout=(self.connect_timeout, self.read_timeout))
                resp.raise_for_status()
        except _FETCH_ERRORS as exc:
            logger.warning("Could not apply camera settings: %s", exc)
            return
        self._hardware_flip = self.flip != "none"

    def _poll_status(self) -> None:
        with requests.Session() as session:
            session.trust_env = False
            while not self._stop.is_set():
                try:
                    resp = session.get(f"{self.control_url}/status", timeout=(1.5, 1.5))
                    resp.raise_for_status()
                    self.esp32_status = resp.json()
                except (*_FETCH_ERRORS, ValueError):
                    self.esp32_status = None
                self._stop.wait(self.status_interval)


class VideoFileSource(CameraSource):
    """Replays a video file at its own frame rate, looping. For testing without hardware."""

    kind = "video"

    def __init__(self, path: Union[str, Path], flip: str = "none", loop: bool = True):
        super().__init__(flip=flip)
        self.path = str(path)
        self.loop = loop

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self._set_state(LOST, f"cannot open {self.path}")
            return
        interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0)
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if not self.loop:
                        self._set_state(LOST, "end of file")
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._publish(frame)
                self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        finally:
            cap.release()

    def status(self) -> dict[str, Any]:
        return {**super().status(), "url": self.path}


class ImageSource(CameraSource):
    """Offers one still image as a stream, each repeat with a new frame id. For testing without hardware."""

    kind = "image"

    def __init__(self, path: Union[str, Path], flip: str = "none", fps: float = 15.0):
        super().__init__(flip=flip)
        self.path = str(path)
        self.fps = fps

    def _run(self) -> None:
        image = cv2.imread(self.path)
        if image is None:
            self._set_state(LOST, f"cannot read {self.path}")
            return
        while not self._stop.is_set():
            self._publish(image)
            self._stop.wait(1.0 / self.fps)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "url": self.path}


def resolve_stream_url(source: str) -> tuple[str, Optional[str]]:
    """Turn 'esp32cam.local', '192.168.4.20' or a full URL into (stream_url, control_url).

    control_url is None for a URL that does not look like our firmware (the path is not /stream),
    because /capture, /status and /control only exist there.
    """
    if "://" not in source:
        source = f"http://{source}"
    parts = urlsplit(source)
    host = parts.hostname or ""
    if parts.username or parts.password:
        auth = parts.username or ""
        auth = f"{auth}:{parts.password}@" if parts.password else f"{auth}@"
        host = auth + host
    control = urlunsplit((parts.scheme, host, "", "", ""))
    if parts.path in ("", "/"):
        return urlunsplit((parts.scheme, f"{host}:{parts.port or 81}", "/stream", "", "")), control
    return source, control if parts.path.rstrip("/").endswith("/stream") else None


def make_source(source: str, control_url: Optional[str] = None, framesize: Optional[str] = None,
                quality: Optional[int] = None, flip: str = "none") -> CameraSource:
    """Build a source from an http(s) URL, a bare ESP32 host name or IP, a video file or a still image."""
    path = Path(source)
    if "://" not in source:
        if path.is_file():
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                return ImageSource(path, flip=flip)
            return VideoFileSource(path, flip=flip)
        if path.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES:
            raise FileNotFoundError(f"Camera source file not found: {source}")
        if not _BARE_HOST.match(source):
            raise ValueError(f"Not a camera URL, host name or file: {source}")
    stream_url, derived_control = resolve_stream_url(source)
    return Esp32CamSource(stream_url, control_url=control_url or derived_control, framesize=framesize,
                          quality=quality, flip=flip)
