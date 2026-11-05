"""Laptop client and launcher tests against the fake ESP32-CAM. Local sockets only, no hardware."""

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import uvicorn

import run_serbot
from fake_esp32 import BOUNDARY, FakeEsp32Cam, jpeg_bytes
from laptop_client import esp32_client as client
from server.app import create_app
from src.gesturegrasp.camera import Esp32CamSource, MjpegParser
from src.gesturegrasp.pipeline import GesturePipeline

CHECK_TIMEOUT = 0.5


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def source_for(cam: FakeEsp32Cam) -> Esp32CamSource:
    return Esp32CamSource(cam.stream_url, connect_timeout=1.0, read_timeout=0.5, stall_timeout=0.5,
                          backoff_min=0.02, backoff_max=0.02)


@pytest.fixture
def fake_cam():
    cam = FakeEsp32Cam(fps=60.0).start()
    yield cam
    cam.stop()


class CorruptEvery(FakeEsp32Cam):
    """Every other frame has valid JPEG markers around garbage."""

    def _next_frame(self) -> bytes:
        self.frames_sent += 1
        return b"\xff\xd8corrupt\xff\xd9" if self.frames_sent % 2 else jpeg_bytes(90)


# -- addresses and options ------------------------------------------------------------------------
@pytest.mark.parametrize("value, stream, status", [
    ("192.168.1.50", "http://192.168.1.50:81/stream", "http://192.168.1.50/status"),
    ("192.168.1.50:81", "http://192.168.1.50:81/stream", "http://192.168.1.50/status"),
    ("  192.168.1.50 ", "http://192.168.1.50:81/stream", "http://192.168.1.50/status"),
    ("http://192.168.1.50:81/stream", "http://192.168.1.50:81/stream", "http://192.168.1.50/status"),
    ("esp32cam.local", "http://esp32cam.local:81/stream", "http://esp32cam.local/status"),
])
def test_address_normalisation(value, stream, status):
    assert client.normalise_esp32(value) == (stream, status)


def test_an_empty_address_is_rejected():
    with pytest.raises(ValueError):
        client.normalise_esp32("  ")


def test_token_is_appended_like_the_old_client():
    assert client.add_token("ws://h/ws", None) == "ws://h/ws"
    assert client.add_token("ws://h/ws", "abc") == "ws://h/ws?token=abc"
    assert client.add_token("ws://h/ws?x=1", "abc") == "ws://h/ws?x=1&token=abc"
    assert client.add_token("ws://h/ws", "a b&c") == "ws://h/ws?token=a%20b%26c"


def test_defaults_match_the_brief():
    args = client.build_parser().parse_args(["--esp32", "192.168.1.50"])
    assert args.server == "ws://127.0.0.1:8000/ws/stream"
    assert (args.fps, args.width, args.max_frames, args.flip, args.show, args.check) == (15.0, 640, 0, False, False, False)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME[:, :320] = 200


def test_original_jpeg_is_forwarded_untouched():
    original = jpeg_bytes(30)
    payload, _ = client.prepare_frame(FRAME, original, client.FrameOptions())
    assert payload is original


@pytest.mark.parametrize("options", [client.FrameOptions(flip=True), client.FrameOptions(quality=50),
                                     client.FrameOptions(width=320)])
def test_changes_are_re_encoded(options):
    original = jpeg_bytes(30)
    payload, _ = client.prepare_frame(FRAME, original, options)
    assert payload != original and payload.startswith(b"\xff\xd8")


def test_a_wide_frame_is_scaled_down_only():
    wide = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert client.prepare_frame(wide, None, client.FrameOptions())[1].shape == (360, 640, 3)
    small = np.zeros((120, 160, 3), dtype=np.uint8)
    assert client.prepare_frame(small, None, client.FrameOptions())[1].shape == (120, 160, 3)


def test_flip_rotates_180_degrees():
    _, flipped = client.prepare_frame(FRAME, None, client.FrameOptions(flip=True))
    assert flipped[0, 0, 0] == 0 and flipped[0, -1, 0] == 200


def test_speaker_never_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(client.shutil, "which", lambda name: None)
    client.make_speaker(True)("hello")
    client.make_speaker(False)("hello")


def test_a_quote_cannot_break_out_of_the_spoken_text(monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append(args)

        def poll(self):
            return 0

    monkeypatch.setattr(client.os, "name", "nt")
    monkeypatch.setattr(client.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(client.subprocess, "Popen", FakePopen)
    client.make_speaker(True)("it's; Remove-Item x")
    assert "'it''s; Remove-Item x'" in calls[0][-1]


def test_console_prints_prompt_once_then_status(capsys):
    said = []
    console = client.Console(said.append)
    result = {"state": "IDLE", "message": "Show the call-me sign to start.", "gesture": "call", "confidence": 1.0,
              "hold_progress": 0.2, "latency_ms": 54.0}
    console.show(result)
    console.show(result)
    out = capsys.readouterr().out
    assert out.count(">>> [IDLE] Show the call-me sign to start.") == 1
    assert said == ["Show the call-me sign to start."]
    assert client.status_line(result) == "call        1.00 hold 20%  | 54 ms"


# -- stream reading -------------------------------------------------------------------------------
def test_stream_frames_decode(fake_cam):
    source = source_for(fake_cam).start()
    try:
        image, _, _ = source.wait_for_frame(0, timeout=3.0)
        assert image.shape == (120, 160, 3)
    finally:
        source.stop()


def test_jpeg_of_gives_the_original_bytes_only_for_the_newest_frame():
    source = Esp32CamSource("http://127.0.0.1:1/stream")
    source._publish(jpeg_bytes(10))
    source._publish(jpeg_bytes(20))
    assert source.jpeg_of(2) == jpeg_bytes(20)
    assert source.jpeg_of(1) is None


def test_old_frames_are_dropped(fake_cam):
    source = source_for(fake_cam).start()
    try:
        _, first, _ = source.wait_for_frame(0, timeout=3.0)
        time.sleep(0.2)  # a slow model: the camera carries on without it
        _, next_id, _ = source.wait_for_frame(first, timeout=1.0)
        assert next_id - first > 3
    finally:
        source.stop()


def test_a_corrupt_frame_is_skipped():
    cam = CorruptEvery(fps=60.0).start()
    source = source_for(cam).start()
    try:
        good, last = 0, 0
        deadline = time.monotonic() + 5.0
        while good < 3 and time.monotonic() < deadline:
            frame = source.wait_for_frame(last, timeout=0.5)
            if frame is not None:
                last = frame[1]
                good += 1
        assert good == 3
        assert wait_until(lambda: source.status()["decode_errors"] >= 1)
    finally:
        source.stop()
        cam.stop()


def test_reconnects_after_the_camera_drops(fake_cam):
    source = source_for(fake_cam).start()
    try:
        _, first, _ = source.wait_for_frame(0, timeout=3.0)
        fake_cam.drop_streams()
        assert wait_until(lambda: fake_cam.stream_clients >= 2)
        assert source.wait_for_frame(first, timeout=3.0) is not None
    finally:
        source.stop()


def test_camera_watch_reports_loss_and_the_hint(capsys):
    source = Esp32CamSource("http://127.0.0.1:1/stream")
    watch = client.CameraWatch(source, source.stream_url, hint_after=0.0)
    source._set_state("lost", "ConnectionError: refused")
    watch.poll()
    out = capsys.readouterr().out
    assert "ESP32 stream lost (ConnectionError)" in out
    assert client.NO_FRAME_HINT in out


# -- --check --------------------------------------------------------------------------------------
def test_check_passes_against_the_camera(fake_cam, tmp_path, capsys):
    snapshot = tmp_path / "check.jpg"
    code = client.run_check(fake_cam.stream_url, f"{fake_cam.control_url}/status", snapshot, seconds=0.4)
    out = capsys.readouterr().out
    assert code == 0
    assert f"Connected to {fake_cam.stream_url}" in out
    assert "Received " in out and " frames in " in out and "fps)" in out
    assert "Frame size: 160x120" in out
    assert f"Saved snapshot: {snapshot}" in out
    assert "Average JPEG size:" in out
    assert out.strip().endswith("RESULT: OK")
    assert '"rssi": -58' in out
    assert cv2.imread(str(snapshot)) is not None


def test_check_warns_about_low_resolution(fake_cam, tmp_path, capsys):
    client.run_check(fake_cam.stream_url, f"{fake_cam.control_url}/status", tmp_path / "c.jpg", seconds=0.3)
    assert "Resolution is low; hands may be too small" in capsys.readouterr().out


def test_check_continues_when_status_is_missing(fake_cam, tmp_path, capsys):
    code = client.run_check(fake_cam.stream_url, "http://127.0.0.1:1/status", tmp_path / "c.jpg", seconds=0.3)
    out = capsys.readouterr().out
    assert code == 0 and "is unreachable" in out


def test_check_fails_on_a_closed_port(tmp_path, capsys):
    port = free_port()
    code = client.run_check(f"http://127.0.0.1:{port}/stream", f"http://127.0.0.1:{port}/status", tmp_path / "c.jpg")
    out = capsys.readouterr().out
    assert code == 1
    assert "RESULT: FAILED" in out and "ping" in out and "firewall" in out
    assert "RESULT: OK" not in out


def test_check_fails_when_connected_but_silent(tmp_path, capsys):
    # An ESP32 busy with another client accepts the connection and then says nothing.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        code = client.run_check(f"http://127.0.0.1:{port}/stream", f"http://127.0.0.1:{free_port()}/status",
                                tmp_path / "c.jpg", read_timeout=CHECK_TIMEOUT)
    out = capsys.readouterr().out
    assert code == 1
    assert "RESULT: FAILED" in out and "other tab" in out and "power" in out


def test_check_fails_on_a_very_low_frame_rate(tmp_path, capsys):
    cam = FakeEsp32Cam(fps=1.0).start()
    try:
        code = client.run_check(cam.stream_url, f"{cam.control_url}/status", tmp_path / "c.jpg", seconds=1.5,
                                read_timeout=2.0)
    finally:
        cam.stop()
    out = capsys.readouterr().out
    assert code == 1
    assert "RESULT: FAILED" in out and "Weak Wi-Fi" in out


def test_main_returns_the_check_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert client.main(["--esp32", f"127.0.0.1:{free_port()}", "--check"]) == 1


# -- end to end -----------------------------------------------------------------------------------
@pytest.fixture
def live_server(monkeypatch):
    """A real server on a real port, with the mock detector and no camera."""
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    app = create_app(GesturePipeline(use_mock_detector=True))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_until(lambda: server.started, timeout=30.0)
    yield SimpleNamespace(url=f"ws://127.0.0.1:{port}/ws/stream", port=port, app=app, server=server)
    server.should_exit = True
    thread.join(timeout=10.0)


def collect(source, url, frames, options=None, max_fps=0.0):
    results: list[dict] = []
    console = SimpleNamespace(show=results.append)
    watch = client.CameraWatch(source, source.stream_url)
    client.stream_forever(url, source, options or client.FrameOptions(), console, client.Preview(False), watch,
                          max_fps, frames)
    return results


def test_frames_reach_the_server_and_results_come_back(fake_cam, live_server):
    source = source_for(fake_cam).start()
    try:
        results = collect(source, live_server.url, 5)
    finally:
        source.stop()
    assert len(results) == 5
    assert all("state" in r and "gesture" in r for r in results)
    service = live_server.app.state.service
    assert service.latest_result["state"] == "IDLE"  # what the page reads from /latest
    assert service.latest_jpeg is not None           # and from /video


def test_client_waits_for_a_late_server_then_connects(fake_cam, live_server, monkeypatch):
    monkeypatch.setattr(client, "RECONNECT_SECONDS", 0.05)
    source = source_for(fake_cam).start()
    attempts = []
    real = client.connect

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionRefusedError("server not up yet")
        return real(*args, **kwargs)

    monkeypatch.setattr(client, "connect", flaky)
    try:
        assert len(collect(source, live_server.url, 2)) == 2
    finally:
        source.stop()
    assert len(attempts) == 2


def test_fps_limit_holds(fake_cam, live_server):
    source = source_for(fake_cam).start()
    started = time.monotonic()
    try:
        collect(source, live_server.url, 6, max_fps=20.0)
    finally:
        source.stop()
    assert time.monotonic() - started >= 0.25


def test_script_runs_directly_from_the_project_root():
    done = subprocess.run([sys.executable, "laptop_client/esp32_client.py", "--help"], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0
    for flag in ("--esp32", "--server", "--token", "--fps", "--width", "--quality", "--flip", "--show", "--check",
                 "--max-frames"):
        assert flag in done.stdout


def test_token_is_never_printed(capsys, caplog):
    source = Esp32CamSource("http://127.0.0.1:1/stream")
    url = client.add_token("ws://127.0.0.1:1/ws/stream", "hunter2")
    console, preview = SimpleNamespace(show=print), client.Preview(False)
    with pytest.raises(OSError):
        client.run_stream(url, source, client.FrameOptions(), console, preview,
                          client.CameraWatch(source, source.stream_url), 0.0)
    assert "hunter2" not in capsys.readouterr().out + caplog.text


# -- launcher -------------------------------------------------------------------------------------
def launcher_args(*extra: str):
    return run_serbot.build_parser().parse_args(["--esp32", "192.168.1.50", *extra])


def test_server_gets_no_camera_so_the_client_owns_the_stream():
    command = run_serbot.server_command(launcher_args())
    assert command[0] == sys.executable
    assert command[command.index("--camera-url") + 1] == "none"
    assert "--mock-detector" not in command and "--token" not in command


def test_options_are_passed_through():
    args = launcher_args("--flip", "--show", "--token", "SECRET", "--fps", "12", "--mock-detector", "--port", "9000")
    server = run_serbot.server_command(args)
    assert "--mock-detector" in server and server[server.index("--token") + 1] == "SECRET"
    client_command = run_serbot.client_command(args)
    assert client_command[client_command.index("--esp32") + 1] == "192.168.1.50"
    assert client_command[client_command.index("--server") + 1] == "ws://127.0.0.1:9000/ws/stream"
    assert client_command[client_command.index("--fps") + 1] == "12.0"
    assert {"--flip", "--show"} <= set(client_command)
    assert client_command[client_command.index("--token") + 1] == "SECRET"


def test_output_is_prefixed(capsys):
    seen = []

    def collected() -> bool:
        seen.append(capsys.readouterr().out)
        return "[client] hello" in "".join(seen)

    process = run_serbot.start([sys.executable, "-c", "print('hello')"], "client")
    process.wait(timeout=30)
    assert wait_until(collected)


def test_kill_tree_takes_the_children_with_it():
    # The same shape as the protobuf hand-over: a parent waiting on the real server.
    port = free_port()
    child = f"import socket,time; s=socket.socket(); s.bind(('127.0.0.1',{port})); s.listen(); time.sleep(60)"
    parent = f"import subprocess,sys; subprocess.call([sys.executable,'-c',{child!r}])"
    process = run_serbot.start([sys.executable, "-c", parent], "server")
    try:
        assert wait_until(lambda: run_serbot.port_in_use(port), timeout=15.0)
    finally:
        run_serbot.kill_tree(process)
    assert wait_until(lambda: not run_serbot.port_in_use(port), timeout=5.0)


def test_launcher_stops_its_children_when_it_is_told_to_stop():
    # Ctrl+Break or closing the console must not leave a server holding the port.
    port = free_port()
    child = f"import socket,time; s=socket.socket(); s.bind(('127.0.0.1',{port})); s.listen(); time.sleep(60)"
    fake_server = f"import subprocess,sys; subprocess.call([sys.executable,'-c',{child!r}])"
    script = ("import sys, time; sys.path.insert(0, '.')\n"
              "import run_serbot\n"
              "run_serbot.install_stop_signals()\n"
              f"server = run_serbot.start([sys.executable, '-c', {fake_server!r}], 'server')\n"
              "try:\n"
              "    print('ready', flush=True)\n"
              "    for _ in range(600):\n"
              "        time.sleep(0.1)\n"  # like the real launcher: on Windows a long sleep ignores Ctrl+Break
              "except KeyboardInterrupt:\n"
              "    pass\n"
              "finally:\n"
              "    run_serbot.kill_tree(server)\n")
    process = run_serbot.start([sys.executable, "-c", script], "launcher")
    try:
        assert wait_until(lambda: run_serbot.port_in_use(port), timeout=20.0)
        run_serbot.request_stop(process)
        process.wait(timeout=15.0)
        assert wait_until(lambda: not run_serbot.port_in_use(port), timeout=5.0)
    finally:
        run_serbot.kill_tree(process)


def test_launcher_refuses_a_busy_port(capsys):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert run_serbot.main(["--esp32", "127.0.0.1", "--port", str(port)]) == 1
    assert "already in use" in capsys.readouterr().out


def test_health_wait_gives_up_when_the_server_dies():
    dead = run_serbot.start([sys.executable, "-c", "raise SystemExit(3)"], "server")
    dead.wait(timeout=30)
    assert run_serbot.wait_until_healthy("http://127.0.0.1:1/health", dead, timeout=5.0) is False


def test_batch_file_calls_the_launcher():
    assert "python run_serbot.py %*" in Path("run_serbot.bat").read_text(encoding="utf-8")


def test_multipart_headers_are_stripped_from_the_frames():
    jpeg = jpeg_bytes(70)
    part = f"\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode() + jpeg
    assert MjpegParser().feed(part * 2) == [jpeg, jpeg]
