"""
Start everything on the laptop with one command: the server, then the ESP32-CAM client.

    python run_serbot.py --esp32 192.168.1.50
    python run_serbot.py --esp32 192.168.1.50 --show --flip --token SECRET --fps 15

The server starts with no camera of its own, because the ESP32 stream serves one client and that client
is laptop_client/esp32_client.py. Ctrl+C stops both.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Optional

ROOT = Path(__file__).resolve().parent
CLIENT = ROOT / "laptop_client" / "esp32_client.py"
HEALTH_TIMEOUT_SECONDS = 90.0
CLIENT_STOP_SECONDS = 3.0

_print_lock = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the GestureGrasp server and the ESP32-CAM client")
    parser.add_argument("--esp32", required=True, help="192.168.1.50, 192.168.1.50:81 or a full stream URL")
    parser.add_argument("--token", help="Shared secret for the server and the client")
    parser.add_argument("--mock-detector", action="store_true", help="Fake objects instead of YOLO")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--fps", type=float, help="Upper bound on frames per second sent to the model")
    parser.add_argument("--flip", action="store_true", help="Rotate 180 degrees in software")
    parser.add_argument("--show", action="store_true", help="Preview window with the state; q closes it")
    parser.add_argument("--speak", action="store_true", help="Say prompts aloud (Windows only)")
    return parser


def server_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-u", "-m", "server", "--camera-url", "none", "--port", str(args.port)]
    if args.token:
        command += ["--token", args.token]
    if args.mock_detector:
        command.append("--mock-detector")
    return command


def client_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-u", str(CLIENT), "--esp32", args.esp32,
               "--server", f"ws://127.0.0.1:{args.port}/ws/stream"]
    if args.token:
        command += ["--token", args.token]
    if args.fps is not None:
        command += ["--fps", str(args.fps)]
    command += [flag for flag, on in (("--flip", args.flip), ("--show", args.show), ("--speak", args.speak)) if on]
    return command


def say(prefix: str, text: str) -> None:
    with _print_lock:
        print(f"[{prefix}] {text}", flush=True)


def _pump(stream: IO[bytes], prefix: str) -> None:
    for raw in stream:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            say(prefix, line)


def start(command: list[str], prefix: str) -> subprocess.Popen:
    """Start a child in its own process group, so Ctrl+C reaches only this script, which stops it in order."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, **group)
    threading.Thread(target=_pump, args=(process.stdout, prefix), daemon=True).start()
    return process


def port_in_use(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def is_healthy(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2.0) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_until_healthy(url: str, server: subprocess.Popen, timeout: float = HEALTH_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False
        if is_healthy(url):
            return True
        time.sleep(0.5)
    return False


def kill_tree(process: subprocess.Popen) -> None:
    """Stop a process and its children. The server may have handed over to the project's .venv, so on
    Windows the real server is a child, and it must go first while the parent can still name it."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5.0)


def request_stop(process: subprocess.Popen) -> None:
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGTERM)


def stop_client(process: Optional[subprocess.Popen]) -> None:
    """Ask the client to stop, so it can close its socket and window, then make sure it did."""
    if process is None or process.poll() is not None:
        return
    try:
        request_stop(process)
        process.wait(timeout=CLIENT_STOP_SECONDS)
    except (subprocess.TimeoutExpired, OSError):
        kill_tree(process)


def supervise(server: subprocess.Popen, client: subprocess.Popen) -> int:
    """Run until one side stops, and report which."""
    while True:
        if client.poll() is not None:
            return client.returncode
        if server.poll() is not None:
            say("serbot", f"The server stopped (exit code {server.returncode}).")
            return 1
        time.sleep(0.2)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def install_stop_signals() -> None:
    """Ctrl+Break, or a terminate from outside, must clean up the same way Ctrl+C does."""
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupt)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    install_stop_signals()
    if port_in_use(args.port):
        say("serbot", f"Port {args.port} is already in use. Stop the other server, or pass --port.")
        return 1

    server: Optional[subprocess.Popen] = None
    client: Optional[subprocess.Popen] = None
    try:
        say("serbot", "Starting the server. The models take a little while to load...")
        server = start(server_command(args), "server")
        health_url = f"http://127.0.0.1:{args.port}/health"
        if not wait_until_healthy(health_url, server):
            say("serbot", f"The server did not become ready at {health_url}. See its output above.")
            return 1
        say("serbot", f"Server ready. Page: http://localhost:{args.port}/   Ctrl+C stops everything.")
        client = start(client_command(args), "client")
        return supervise(server, client)
    except KeyboardInterrupt:
        say("serbot", "Stopping...")
        return 0
    finally:
        stop_client(client)
        if server is not None:
            kill_tree(server)


if __name__ == "__main__":
    sys.exit(main())
