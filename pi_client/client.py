"""
GestureGrasp Raspberry Pi camera client.

Captures frames on the Pi, sends them as JPEG over a WebSocket to the laptop server, and prints
the server's prompt whenever the interaction state changes. The model never runs on the Pi.

Lock-step: the next frame is captured only after the previous result arrives, so a slow network
or a slow model drops frames instead of building a backlog (old frames would mean late commands).

Usage on the Pi:
    python client.py --server ws://192.168.1.21:8000/ws/stream
    python client.py --server ws://192.168.1.21:8000/ws/stream --source 0        # USB webcam
    python client.py --server ws://LAPTOP:8000/ws/stream --source clip.mp4       # replay a file
    python client.py --server ws://192.168.1.21:8000/ws/stream --speak           # say prompts aloud
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

logger = logging.getLogger("pi_client")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def open_picamera(width: int, height: int) -> Optional[Callable[[], Optional[np.ndarray]]]:
    """Raspberry Pi CSI camera via picamera2. Returns None when picamera2 is not installed."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        return None
    cam = Picamera2()
    # picamera2's "RGB888" stores pixels as [B, G, R], i.e. OpenCV-native order.
    cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
    cam.start()
    logger.info("Using Raspberry Pi camera (picamera2) at %dx%d", width, height)
    return lambda: cam.capture_array()


def open_source(source: str, width: int, height: int) -> Callable[[], Optional[np.ndarray]]:
    """Return a zero-arg function that yields the next BGR frame (None when the source is exhausted)."""
    if source == "picamera":
        reader = open_picamera(width, height)
        if reader is not None:
            return reader
        logger.warning("picamera2 not available, falling back to camera index 0")
        source = "0"

    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read image {path}")
        logger.info("Replaying still image %s (for testing)", path)
        return lambda: image.copy()

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/video source {source!r}")
    logger.info("Using OpenCV source %s", source)

    def read() -> Optional[np.ndarray]:
        ok, frame = cap.read()
        return frame if ok else None

    return read


def make_speaker(enabled: bool) -> Callable[[str], None]:
    """Say prompts aloud with espeak-ng (sudo apt install -y espeak-ng). Never blocks the frame loop."""
    if not enabled:
        return lambda text: None
    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if engine is None:
        logger.warning("--speak needs espeak-ng: sudo apt install -y espeak-ng. Continuing without speech.")
        return lambda text: None
    current: list[Optional[subprocess.Popen]] = [None]

    def say(text: str) -> None:
        if current[0] is not None and current[0].poll() is None:
            current[0].terminate()  # a newer prompt replaces one still being spoken
        current[0] = subprocess.Popen([engine, "-s", "140", text],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return say


def run(server: str, read_frame: Callable[[], Optional[np.ndarray]], *, max_fps: float, quality: int,
        width: int, max_frames: int = 0, on_result: Optional[Callable[[dict], None]] = None,
        say: Callable[[str], None] = lambda text: None) -> int:
    """Stream until the source ends or max_frames is reached. Returns frames sent."""
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    sent = 0
    last_state = None
    with connect(server, max_size=2**22, open_timeout=10) as ws:
        logger.info("Connected to %s", server)
        while not max_frames or sent < max_frames:
            started = time.monotonic()
            frame = read_frame()
            if frame is None:
                logger.info("Source finished.")
                break
            if frame.shape[1] > width:
                scale = width / frame.shape[1]
                frame = cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            ws.send(jpeg.tobytes())
            result = json.loads(ws.recv())
            sent += 1

            if on_result:
                on_result(result)
            if "error" in result:
                logger.warning("Server: %s", result["error"])
            elif result["state"] != last_state:
                last_state = result["state"]
                print(f"\n>>> [{result['state']}] {result['message']}", flush=True)
                say(result["message"])
            else:
                hold = f" hold {result['hold_progress']:.0%}" if result["hold_progress"] else ""
                aim = f" -> {result['pointing_at']}" if result.get("pointing_at") else ""
                print(f"\r{result['gesture']:<11} {result['confidence']:.2f}{hold}{aim}"
                      f"  | {result['latency_ms']:.0f} ms        ", end="", flush=True)

            spare = min_interval - (time.monotonic() - started)
            if spare > 0:
                time.sleep(spare)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Pi camera frames to the GestureGrasp server")
    parser.add_argument("--server", required=True, help="ws://<laptop-ip>:8000/ws/stream")
    parser.add_argument("--source", default="picamera",
                        help="'picamera' (default), a camera index like 0, a video file, or an image file")
    parser.add_argument("--token", help="Shared secret if the server was started with --token")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0, help="Upper bound on frames per second")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = forever)")
    parser.add_argument("--speak", action="store_true", help="Say each new prompt aloud (needs espeak-ng)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = args.server + (("&" if "?" in args.server else "?") + f"token={args.token}" if args.token else "")
    read_frame = open_source(args.source, args.width, args.height)
    say = make_speaker(args.speak)

    # Reconnect forever: the laptop may restart, Wi-Fi may drop.
    while True:
        try:
            run(server, read_frame, max_fps=args.fps, quality=args.quality,
                width=args.width, max_frames=args.max_frames, say=say)
            return
        except (ConnectionClosed, OSError, TimeoutError) as exc:
            logger.warning("Connection lost (%s). Retrying in 2 s...", exc)
            time.sleep(2)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
