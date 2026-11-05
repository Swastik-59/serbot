"""
GestureGrasp laptop client for the AI-Thinker ESP32-CAM.

Reads the ESP32-CAM's MJPEG stream, sends frames as JPEG over a WebSocket to the server running on this
laptop, and prints the server's prompt whenever the interaction state changes. The model never runs here.

    ESP32-CAM --MJPEG over HTTP--> this client --JPEG over WebSocket--> python -m server

The stream is read on a background thread that keeps only the newest frame (src/gesturegrasp/camera.py),
because the ESP32 produces frames faster than the model consumes them. Sending is lock-step: the next
frame is taken only after the previous result arrives, so nothing queues up and no command fires late.

The ESP32 stream serves one client, so close any browser tab showing it while this runs.

Usage:
    python laptop_client/esp32_client.py --esp32 192.168.1.50 --check
    python laptop_client/esp32_client.py --esp32 192.168.1.50
    python laptop_client/esp32_client.py --esp32 192.168.1.50 --show --flip --token SECRET --fps 15
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import urllib3
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gesturegrasp.camera import LOST, STREAMING, Esp32CamSource, MjpegParser, resolve_stream_url  # noqa: E402

logger = logging.getLogger("esp32_client")

DEFAULT_SERVER = "ws://127.0.0.1:8000/ws/stream"
DEFAULT_QUALITY = 80
RECONNECT_SECONDS = 2.0
READ_TIMEOUT_SECONDS = 5.0
RESULT_TIMEOUT_SECONDS = 30.0
NO_FRAME_HINT_SECONDS = 10.0
NO_FRAME_HINT = "No frames from ESP32. Check power, Wi-Fi, and that no browser tab is using the stream."
PIPED_STATUS_SECONDS = 2.0
CHECK_SECONDS = 3.0
CHECK_MIN_FPS = 3.0
CHECK_MIN_WIDTH = 480
CHECK_SNAPSHOT = Path("esp32_check.jpg")
_FETCH_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, http.client.HTTPException, OSError)


@dataclass
class FrameOptions:
    width: int = 640
    quality: Optional[int] = None  # None keeps the ESP32's own JPEG whenever the picture is unchanged
    flip: bool = False


def normalise_esp32(value: str) -> tuple[str, str]:
    """'192.168.1.50', '192.168.1.50:81' or a full URL into (stream_url, status_url)."""
    value = value.strip()
    stream_url, _ = resolve_stream_url(value)
    parts = urlsplit(stream_url)
    if not parts.hostname:
        raise ValueError(f"Not an ESP32 address: {value!r}")
    return stream_url, urlunsplit((parts.scheme, parts.hostname, "/status", "", ""))


def add_token(server: str, token: Optional[str]) -> str:
    if not token:
        return server
    return server + ("&" if "?" in server else "?") + f"token={quote(token, safe='')}"


def prepare_frame(frame: np.ndarray, jpeg: Optional[bytes],
                  options: FrameOptions) -> tuple[Optional[bytes], np.ndarray]:
    """What to send, and the picture it shows. The original JPEG goes through when nothing changed it,
    which saves CPU and a second round of JPEG loss."""
    changed = False
    if options.flip:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        changed = True
    if frame.shape[1] > options.width:
        scale = options.width / frame.shape[1]
        frame = cv2.resize(frame, (options.width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        changed = True
    if jpeg is not None and not changed and options.quality is None:
        return jpeg, frame
    quality = options.quality if options.quality is not None else DEFAULT_QUALITY
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return (encoded.tobytes() if ok else None), frame


def make_speaker(enabled: bool) -> Callable[[str], None]:
    """Say prompts aloud with Windows SAPI. Never blocks the frame loop. Skipped on other systems."""
    engine = shutil.which("powershell") if enabled and os.name == "nt" else None
    if engine is None:
        return lambda text: None
    current: list[Optional[subprocess.Popen]] = [None]

    def say(text: str) -> None:
        if current[0] is not None and current[0].poll() is None:
            current[0].terminate()  # a newer prompt replaces one still being spoken
        quoted = text.replace("'", "''")
        script = ("Add-Type -AssemblyName System.Speech; "
                  f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{quoted}')")
        try:
            current[0] = subprocess.Popen([engine, "-NoProfile", "-NonInteractive", "-Command", script],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            current[0] = None

    return say


class Preview:
    """Optional window with the live picture and the current state. q closes it; streaming carries on."""

    WINDOW = "GestureGrasp ESP32-CAM (q closes)"

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._opened = False

    def show(self, frame: np.ndarray, result: dict) -> None:
        if not self.enabled:
            return
        picture = frame.copy()
        for row, text in enumerate(overlay_lines(result)):
            y = 28 + row * 28
            cv2.putText(picture, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(picture, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        try:
            cv2.imshow(self.WINDOW, picture)
            self._opened = True
        except cv2.error:
            logger.warning("This OpenCV build has no preview window. Continuing without --show.")
            self.enabled = False
        self.pump()

    def pump(self) -> None:
        """Keep the window responsive, including while no frames are arriving."""
        if not self._opened:
            return
        pressed = cv2.waitKey(1) & 0xFF
        if pressed == ord("q") or cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            self.enabled = False

    def close(self) -> None:
        if self._opened:
            self._opened = False
            cv2.destroyAllWindows()


def overlay_lines(result: dict) -> list[str]:
    if "error" in result:
        return [f"Server: {result['error']}"]
    gesture = f"{result.get('gesture', '?')} {result.get('confidence', 0.0):.2f}"
    if result.get("hold_progress"):
        gesture += f"  hold {result['hold_progress']:.0%}"
    return [str(result.get("state", "?")), gesture, str(result.get("message", ""))]


def status_line(result: dict) -> str:
    hold = f" hold {result['hold_progress']:.0%}" if result["hold_progress"] else ""
    aim = f" -> {result['pointing_at']}" if result.get("pointing_at") else ""
    return f"{result['gesture']:<11} {result['confidence']:.2f}{hold}{aim}  | {result['latency_ms']:.0f} ms"


class Console:
    """Prints prompts on state change and a live status line, as the Pi client did. When piped (run_serbot)
    the status line would be one line per frame, so it is thinned out instead of redrawn in place."""

    def __init__(self, say: Callable[[str], None]):
        self._say = say
        self._state: Optional[str] = None
        self._live = sys.stdout.isatty()
        self._last_piped = 0.0

    def show(self, result: dict) -> None:
        if "error" in result:
            logger.warning("Server: %s", result["error"])
        elif result["state"] != self._state:
            self._state = result["state"]
            print(f"\n>>> [{result['state']}] {result['message']}", flush=True)
            self._say(result["message"])
        elif self._live:
            print(f"\r{status_line(result)}        ", end="", flush=True)
        elif time.monotonic() - self._last_piped >= PIPED_STATUS_SECONDS:
            self._last_piped = time.monotonic()
            print(status_line(result), flush=True)


class CameraWatch:
    """One clear line when the ESP32 stream changes state, and a hint during a long silence."""

    def __init__(self, source: Esp32CamSource, stream_url: str, hint_after: float = NO_FRAME_HINT_SECONDS):
        self._source = source
        self._stream_url = stream_url
        self._hint_after = hint_after
        self._state: Optional[str] = None
        self._quiet_since = time.monotonic()

    def frame_arrived(self) -> None:
        self._quiet_since = time.monotonic()

    def poll(self) -> None:
        status = self._source.status()
        if status["state"] != self._state:
            self._state = status["state"]
            if self._state == STREAMING:
                print(f"\nESP32 stream connected: {self._stream_url}", flush=True)
            elif self._state == LOST:
                reason = (status["last_error"] or "no data").split(":")[0]
                print(f"\nESP32 stream lost ({reason}). Retrying every {RECONNECT_SECONDS:.0f} s...", flush=True)
        if time.monotonic() - self._quiet_since >= self._hint_after:
            self._quiet_since = time.monotonic()
            print(f"\n{NO_FRAME_HINT}", flush=True)


def run_stream(server: str, source: Esp32CamSource, options: FrameOptions, console: Console, preview: Preview,
               watch: CameraWatch, max_fps: float, max_frames: int = 0, sent: int = 0) -> int:
    """One WebSocket connection's lifetime. Returns the running count of frames sent."""
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_id = 0
    with connect(server, max_size=2**22, open_timeout=10) as ws:
        logger.info("Connected to server")
        while not max_frames or sent < max_frames:
            started = time.monotonic()
            watch.poll()
            preview.pump()
            frame = source.wait_for_frame(last_id, timeout=0.25)
            if frame is None:
                continue
            image, last_id, _ = frame
            watch.frame_arrived()
            payload, picture = prepare_frame(image, source.jpeg_of(last_id), options)
            if payload is None:
                continue
            ws.send(payload)
            result = json.loads(ws.recv(timeout=RESULT_TIMEOUT_SECONDS))  # lock-step: wait for the answer
            sent += 1
            console.show(result)
            preview.show(picture, result)
            spare = min_interval - (time.monotonic() - started)
            if spare > 0:
                time.sleep(spare)
    return sent


def stream_forever(server: str, source: Esp32CamSource, options: FrameOptions, console: Console, preview: Preview,
                   watch: CameraWatch, max_fps: float, max_frames: int = 0) -> None:
    """Reconnect to the server for as long as it takes. The ESP32 reconnects by itself on its own thread."""
    sent = 0
    while True:
        try:
            sent = run_stream(server, source, options, console, preview, watch, max_fps, max_frames, sent)
            return
        except (WebSocketException, OSError, TimeoutError, json.JSONDecodeError) as exc:
            reason = str(exc).split("\n")[0] or type(exc).__name__
            logger.warning("Server connection lost (%s). Retrying in %.0f s...", reason, RECONNECT_SECONDS)
            time.sleep(RECONNECT_SECONDS)


# -- --check --------------------------------------------------------------------------------------
@dataclass
class CheckResult:
    connected: bool = False
    frames: int = 0
    seconds: float = 0.0
    total_bytes: int = 0
    size: Optional[tuple[int, int]] = None
    snapshot: Optional[bytes] = None
    error: str = ""
    corrupt: int = field(default=0)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def _read_frames(resp: requests.Response, seconds: float, result: CheckResult) -> None:
    parser = MjpegParser()
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        chunk = resp.raw.read1(64 * 1024)
        if not chunk:
            raise ConnectionError("stream ended")
        for jpeg in parser.feed(chunk):
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or not frame.size:
                result.corrupt += 1
                continue
            result.frames += 1
            result.total_bytes += len(jpeg)
            result.size = (frame.shape[1], frame.shape[0])
            result.snapshot = jpeg
        result.seconds = time.monotonic() - started


def collect_frames(stream_url: str, seconds: float = CHECK_SECONDS,
                   read_timeout: float = READ_TIMEOUT_SECONDS) -> CheckResult:
    """Open the stream and gather frames for about `seconds`. A drop part-way keeps what arrived."""
    result = CheckResult()
    with requests.Session() as session:
        session.trust_env = False
        try:
            with session.get(stream_url, stream=True, timeout=(3.0, read_timeout)) as resp:
                resp.raise_for_status()
                result.connected = True
                print(f"Connected to {stream_url}", flush=True)
                _read_frames(resp, seconds, result)
        except requests.exceptions.ReadTimeout as exc:
            result.connected = True  # the board accepted the connection and then said nothing
            result.error = type(exc).__name__
        except _FETCH_ERRORS as exc:
            result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return result


def print_failure(cause: str, fix: str) -> int:
    print("RESULT: FAILED")
    print(f"Likely cause: {cause}")
    print(f"Fix: {fix}")
    return 1


def print_status(status_url: str) -> None:
    with requests.Session() as session:
        session.trust_env = False
        try:
            resp = session.get(status_url, timeout=3.0)
            resp.raise_for_status()
            print(f"Status: {resp.text.strip()}")
        except _FETCH_ERRORS:
            print(f"Status: {status_url} is unreachable. Continuing with the stream.")


def run_check(stream_url: str, status_url: str, snapshot: Path = CHECK_SNAPSHOT, seconds: float = CHECK_SECONDS,
              read_timeout: float = READ_TIMEOUT_SECONDS) -> int:
    """Diagnose the ESP32 stream. Returns the process exit code: 0 on OK, 1 on failure."""
    print_status(status_url)
    result = collect_frames(stream_url, seconds, read_timeout)
    if not result.connected:
        return print_failure(f"Cannot connect to the ESP32 ({result.error}).",
                             "Check the IP address, ping it, make sure both are on the same Wi-Fi, "
                             "and allow it through the firewall.")
    if result.frames == 0:
        return print_failure("Connected, but no frames arrived.",
                             "Close any other tab or program using the stream, and check the board's power.")
    print(f"Received {result.frames} frames in {result.seconds:.1f} s  (~{result.fps:.1f} fps)")
    width, height = result.size
    print(f"Frame size: {width}x{height}")
    print(f"Average JPEG size: {result.total_bytes / result.frames / 1024:.1f} KB")
    if result.corrupt:
        print(f"Corrupt frames skipped: {result.corrupt}")
    snapshot.write_bytes(result.snapshot)
    print(f"Saved snapshot: {snapshot}")
    if width < CHECK_MIN_WIDTH:
        print("Resolution is low; hands may be too small for reliable tracking. Use FRAMESIZE_VGA in the sketch.")
    if result.fps < CHECK_MIN_FPS:
        return print_failure(f"Only {result.fps:.1f} fps arrived.",
                             "Weak Wi-Fi: move the board closer to the router, or reduce the resolution.")
    print("RESULT: OK")
    return 0


# -- entry point ----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream ESP32-CAM frames to the GestureGrasp server")
    parser.add_argument("--esp32", required=True, help="192.168.1.50, 192.168.1.50:81 or a full stream URL")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Server WebSocket (default {DEFAULT_SERVER})")
    parser.add_argument("--token", help="Shared secret if the server was started with --token")
    parser.add_argument("--fps", type=float, default=15.0, help="Upper bound on frames per second sent to the model")
    parser.add_argument("--width", type=int, default=640, help="Frames wider than this are scaled down")
    parser.add_argument("--quality", type=int, help=f"JPEG quality 1-100 when re-encoding (default {DEFAULT_QUALITY}). "
                                                    "Setting it re-encodes every frame")
    parser.add_argument("--flip", action="store_true", help="Rotate 180 degrees in software")
    parser.add_argument("--show", action="store_true", help="Preview window with the state; q closes it")
    parser.add_argument("--speak", action="store_true", help="Say prompts aloud (Windows only)")
    parser.add_argument("--check", action="store_true", help="Diagnose the ESP32 stream, then exit")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = forever)")
    return parser


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stream_url, status_url = normalise_esp32(args.esp32)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check:
        return run_check(stream_url, status_url)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("gesturegrasp.camera").setLevel(logging.ERROR)  # CameraWatch reports these in one line
    for name in ("SIGTERM", "SIGBREAK"):  # how run_serbot asks this process to stop
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)

    source = Esp32CamSource(stream_url, read_timeout=READ_TIMEOUT_SECONDS, stall_timeout=READ_TIMEOUT_SECONDS,
                            backoff_min=RECONNECT_SECONDS, backoff_max=RECONNECT_SECONDS)
    options = FrameOptions(width=args.width, quality=args.quality, flip=args.flip)
    preview = Preview(args.show)
    console = Console(make_speaker(args.speak))
    try:
        source.start()
        stream_forever(add_token(args.server, args.token), source, options, console, preview,
                       CameraWatch(source, stream_url), args.fps, args.max_frames)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        source.stop(timeout=2.0)
        preview.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
