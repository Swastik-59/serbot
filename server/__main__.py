"""Run the system on the Pi: python -m server --camera-url http://esp32cam.local:81/stream

Every flag has an environment variable, so the systemd unit can keep settings out of its command line.
"""

import argparse
import logging
import os
import re

import uvicorn

from src.gesturegrasp.camera import DEFAULT_CAMERA_URL, FLIP_MODES

_TOKEN_IN_URL = re.compile(r"(token=)[^&\s\"']+")


class RedactToken(logging.Filter):
    """Uvicorn logs WebSocket paths with their query string, which would print ?token=..."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = _TOKEN_IN_URL.sub(r"\1***", record.getMessage())
        record.msg, record.args = message, None
        return True


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="GestureGrasp: ESP32-CAM to Raspberry Pi 5 to web page")
    parser.add_argument("--host", default=env("SERBOT_HOST", "0.0.0.0"),
                        help="0.0.0.0 lets phones on the network connect")
    parser.add_argument("--port", type=int, default=int(env("SERBOT_PORT", "8000")))
    parser.add_argument("--camera-url", "--source", dest="camera_url",
                        default=env("SERBOT_CAMERA_URL", DEFAULT_CAMERA_URL),
                        help="http://<ip>:81/stream, esp32cam.local, a video file, an image, or none")
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
    parser.add_argument("--token", help="Shared secret for /reset, /infer and /ws/stream (or set SERBOT_TOKEN)")
    parser.add_argument("--mock-detector", action="store_true", help="Fake objects instead of YOLO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["SERBOT_CAMERA_URL"] = args.camera_url
    os.environ["SERBOT_FRAMESIZE"] = args.framesize
    os.environ["SERBOT_QUALITY"] = str(args.quality)
    os.environ["SERBOT_FLIP"] = args.flip
    os.environ["SERBOT_SPEAK"] = "1" if args.speak else "0"
    if args.camera_control_url:
        os.environ["SERBOT_CAMERA_CONTROL_URL"] = args.camera_control_url
    if args.mock_detector:
        os.environ["SERBOT_MOCK_DETECTOR"] = "1"
    if args.token:
        os.environ["SERBOT_TOKEN"] = args.token

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactToken())
    logging.getLogger("gesturegrasp.server").info("Open http://<this-pi>:%d/ on any device on the network", args.port)
    # Import after env vars are set: create_app() reads them. log_config=None sends uvicorn's logs through
    # the root handler above, and access_log=False keeps page polling (and ?token=) out of the journal.
    uvicorn.run("server.app:app", host=args.host, port=args.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
