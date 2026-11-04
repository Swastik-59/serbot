"""A local stand-in for the ESP32-CAM firmware: /stream, /capture, /status, /control over real sockets."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

BOUNDARY = "123456789000000000000987654321"


def jpeg_bytes(value: int = 0, height: int = 120, width: int = 160) -> bytes:
    """A small real JPEG. value sets the brightness, so a test can tell frames apart after decoding."""
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


class FakeEsp32Cam:
    """Serves numbered frames. Tests can drop the stream, refuse it, or stall it."""

    def __init__(self, fps: float = 100.0, chunked: bool = True, chunk_size: Optional[int] = None, port: int = 0):
        self.fps = fps
        self.chunked = chunked
        self.chunk_size = chunk_size  # split each write so frames cross read boundaries
        self.stream_enabled = True
        self.capture_enabled = True
        self.stall = False
        self.status_payload = {"framesize": "vga", "quality": 12, "fps": 24.5, "rssi": -58, "ip": "127.0.0.1"}
        self.controls: list[tuple[str, str]] = []
        self.stream_clients = 0
        self.capture_requests = 0
        self.frames_sent = 0
        self._generation = 0
        self._lock = threading.Lock()

        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if chunked else "HTTP/1.0"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                parts = urlsplit(self.path)
                route = {"/stream": fake._stream, "/capture": fake._capture,
                         "/status": fake._status, "/control": fake._control}.get(parts.path)
                if route is None:
                    self.send_error(404)
                    return
                route(self, parse_qs(parts.query))

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def stream_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/stream"

    @property
    def control_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeEsp32Cam":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stream_enabled = False
        self._generation += 1
        self._server.shutdown()
        self._server.server_close()

    def drop_streams(self) -> None:
        """Cut every open stream, as a rebooting ESP32 would."""
        self._generation += 1

    def _next_frame(self) -> bytes:
        with self._lock:
            self.frames_sent += 1
            return jpeg_bytes(self.frames_sent % 256)

    def _write(self, handler: BaseHTTPRequestHandler, data: bytes) -> None:
        size = self.chunk_size or len(data)
        for start in range(0, len(data), size):
            piece = data[start:start + size]
            if self.chunked:
                handler.wfile.write(f"{len(piece):X}\r\n".encode() + piece + b"\r\n")
            else:
                handler.wfile.write(piece)
            handler.wfile.flush()

    def _stream(self, handler: BaseHTTPRequestHandler, query: dict) -> None:
        if not self.stream_enabled:
            handler.send_error(503)
            return
        handler.send_response(200)
        handler.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
        if self.chunked:
            handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        self.stream_clients += 1
        generation = self._generation
        try:
            while generation == self._generation and self.stream_enabled:
                if self.stall:
                    time.sleep(0.01)
                    continue
                jpeg = self._next_frame()
                part = (f"\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                        .encode() + jpeg)
                self._write(handler, part)
                time.sleep(1.0 / self.fps)
        except OSError:
            pass
        handler.close_connection = True

    def _capture(self, handler: BaseHTTPRequestHandler, query: dict) -> None:
        if not self.capture_enabled:
            handler.send_error(503)
            return
        self.capture_requests += 1
        jpeg = self._next_frame()
        handler.send_response(200)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.end_headers()
        handler.wfile.write(jpeg)

    def _status(self, handler: BaseHTTPRequestHandler, query: dict) -> None:
        body = json.dumps(self.status_payload).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _control(self, handler: BaseHTTPRequestHandler, query: dict) -> None:
        self.controls.append((query.get("var", [""])[0], query.get("val", [""])[0]))
        handler.send_response(200)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
