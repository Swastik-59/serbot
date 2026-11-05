"""Run the server. It works out where it is running, or pass --mode:

    python -m server                 # laptop: frames arrive from laptop_client/esp32_client.py
    python -m server --mode pi       # Raspberry Pi 5: the server reads the ESP32-CAM itself

Every flag has an environment variable, so the systemd unit can keep settings out of its command line.
"""

import argparse
import importlib.metadata
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from src.gesturegrasp.camera import DEFAULT_CAMERA_URL, FLIP_MODES

_TOKEN_IN_URL = re.compile(r"(token=)[^&\s\"']+")
_PROJECT_VENV = Path(__file__).resolve().parents[1] / ".venv"
_MAX_PROTOBUF_MAJOR = 4  # MediaPipe crashes at start-up on anything newer
MODES = ("auto", "pi", "laptop")


def resolve_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    return "pi" if platform.machine().lower() in ("aarch64", "arm64", "armv7l") else "laptop"


def camera_for_mode(mode: str, camera_url: Optional[str]) -> str:
    """An explicit camera always wins, so a laptop can also watch an ESP32-CAM."""
    return camera_url or (DEFAULT_CAMERA_URL if mode == "pi" else "none")


def protobuf_is_compatible() -> bool:
    try:
        return int(importlib.metadata.version("protobuf").split(".")[0]) <= _MAX_PROTOBUF_MAJOR
    except importlib.metadata.PackageNotFoundError:
        return True  # let the normal import report it


def venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"


def ensure_compatible_python(venv: Path = _PROJECT_VENV) -> None:
    """A global Python often has a newer protobuf for other tools. Run from the project's .venv instead of
    letting MediaPipe fail with a long traceback, and explain the fix if there is no .venv to run from."""
    if protobuf_is_compatible():
        return
    python = venv_python(venv)
    if not python.exists() or Path(sys.prefix).resolve() == venv.resolve():
        raise SystemExit(f"protobuf {importlib.metadata.version('protobuf')} breaks MediaPipe.\n"
                         "Install the pinned version in a virtual environment: "
                         "pip install protobuf==4.25.9 mediapipe==0.10.14")
    print(f"Using {python}, because this Python's protobuf is too new for MediaPipe.", flush=True)
    try:
        code = subprocess.call([str(python), "-m", "server", *sys.argv[1:]])
    except KeyboardInterrupt:
        code = 0
    raise SystemExit(code)


class RedactToken(logging.Filter):
    """Uvicorn logs WebSocket paths with their query string, which would print ?token=..."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = _TOKEN_IN_URL.sub(r"\1***", record.getMessage())
        record.msg, record.args = message, None
        return True


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="GestureGrasp: ESP32-CAM to Raspberry Pi 5 to web page")
    parser.add_argument("--mode", choices=MODES, default=env("SERBOT_MODE", "auto"),
                        help="laptop = no camera of its own, pi = reads the ESP32-CAM. auto picks by CPU")
    parser.add_argument("--host", default=env("SERBOT_HOST", "0.0.0.0"),
                        help="0.0.0.0 lets phones on the network connect")
    parser.add_argument("--port", type=int, default=int(env("SERBOT_PORT", "8000")))
    parser.add_argument("--camera-url", "--source", dest="camera_url",
                        default=env("SERBOT_CAMERA_URL"),
                        help="http://<ip>:81/stream, esp32cam.local, a video file, an image, or none "
                             f"(default: none on a laptop, {DEFAULT_CAMERA_URL} on the Pi)")
    parser.add_argument("--camera-control-url", default=env("SERBOT_CAMERA_CONTROL_URL"),
                        help="Base URL of the ESP32's /control and /status (default: same host as the stream)")
    parser.add_argument("--framesize", default=env("SERBOT_FRAMESIZE", "vga"),
                        help="ESP32 frame size, e.g. vga or qvga")
    parser.add_argument("--quality", type=int, default=int(env("SERBOT_QUALITY", "12")),
                        help="ESP32 JPEG quality, 10 (best) to 63")
    parser.add_argument("--flip", choices=FLIP_MODES, default=env("SERBOT_FLIP", "none"),
                        help="h = mirror, v = upside down, both = rotated 180 degrees")
    parser.add_argument("--speak", action="store_true", default=env("SERBOT_SPEAK", "") == "1",
                        help="Speak prompts on the Pi through espeak-ng")
    parser.add_argument("--max-hands", type=int, choices=(1, 2), default=int(env("SERBOT_MAX_HANDS", "2")),
                        help="1 is about twice as fast, but a second hand in view can hide the command")
    parser.add_argument("--token", help="Shared secret for /reset, /infer and /ws/stream (or set SERBOT_TOKEN)")
    parser.add_argument("--mock-detector", action="store_true", help="Fake objects instead of YOLO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_compatible_python()
    mode = resolve_mode(args.mode)
    os.environ["SERBOT_CAMERA_URL"] = camera_for_mode(mode, args.camera_url)
    os.environ["SERBOT_FRAMESIZE"] = args.framesize
    os.environ["SERBOT_QUALITY"] = str(args.quality)
    os.environ["SERBOT_FLIP"] = args.flip
    os.environ["SERBOT_SPEAK"] = "1" if args.speak else "0"
    os.environ["SERBOT_MAX_HANDS"] = str(args.max_hands)
    if args.camera_control_url:
        os.environ["SERBOT_CAMERA_CONTROL_URL"] = args.camera_control_url
    if args.mock_detector:
        os.environ["SERBOT_MOCK_DETECTOR"] = "1"
    if args.token:
        os.environ["SERBOT_TOKEN"] = args.token

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactToken())
    host = "localhost" if mode == "laptop" else "<this-pi>"
    logging.getLogger("gesturegrasp.server").info("%s version. Open http://%s:%d/", mode.capitalize(), host, args.port)
    # Import after env vars are set: create_app() reads them. log_config=None sends uvicorn's logs through
    # the root handler above, and access_log=False keeps page polling (and ?token=) out of the journal.
    uvicorn.run("server.app:app", host=args.host, port=args.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
