"""Run the inference server: python -m server --host 0.0.0.0 --port 8000"""

import argparse
import logging
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="GestureGrasp inference server")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets the Raspberry Pi on your LAN connect")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock-detector", action="store_true", help="Fake objects instead of YOLO")
    parser.add_argument("--token", help="Shared secret the Pi must send (or set SERBOT_TOKEN)")
    args = parser.parse_args()

    if args.mock_detector:
        os.environ["SERBOT_MOCK_DETECTOR"] = "1"
    if args.token:
        os.environ["SERBOT_TOKEN"] = args.token

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Import after env vars are set: create_app() reads them.
    logging.getLogger("gesturegrasp.server").info(
        "Playground: http://localhost:%d/playground  |  Pi connects to ws://<this-laptop-ip>:%d/ws/stream",
        args.port, args.port)
    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
