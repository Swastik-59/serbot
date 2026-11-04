"""Camera source tests against the fake ESP32-CAM. Local sockets only, no hardware."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.gesturegrasp.camera import (CONNECTING, LOST, STREAMING, Esp32CamSource, ImageSource, MjpegParser,
                                     VideoFileSource, make_source, resolve_stream_url)
from fake_esp32 import FakeEsp32Cam, jpeg_bytes


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def quick_source(cam: FakeEsp32Cam, **kwargs) -> Esp32CamSource:
    defaults = dict(control_url=cam.control_url, connect_timeout=1.0, read_timeout=0.5, stall_timeout=0.3,
                    backoff_min=0.01, backoff_max=0.05, capture_retry_seconds=0.3, status_interval=0.05)
    return Esp32CamSource(cam.stream_url, **{**defaults, **kwargs})


@pytest.fixture
def fake_cam():
    cam = FakeEsp32Cam().start()
    yield cam
    cam.stop()


# -- parser --------------------------------------------------------------------------------------
def test_parser_finds_a_frame_between_multipart_headers():
    jpeg = jpeg_bytes(10)
    frames = MjpegParser().feed(b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n--boundary\r\n")
    assert frames == [jpeg]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64, 1000])
def test_parser_handles_frames_split_at_any_chunk_boundary(chunk_size):
    jpegs = [jpeg_bytes(v) for v in (10, 20, 30)]
    stream = b"".join(b"\r\n--b\r\nContent-Length: 1\r\n\r\n" + j for j in jpegs) + b"\r\n"
    parser = MjpegParser()
    frames = []
    for start in range(0, len(stream), chunk_size):
        frames += parser.feed(stream[start:start + chunk_size])
    assert frames == jpegs


def test_parser_drops_a_frame_that_never_ends_and_recovers():
    jpeg = jpeg_bytes(10)
    parser = MjpegParser(max_frame_bytes=len(jpeg) * 2)
    assert parser.feed(b"\xff\xd8" + b"\x00" * (len(jpeg) * 3)) == []
    assert parser.dropped == 1
    assert parser.feed(jpeg) == [jpeg]


def test_parser_returns_every_frame_in_one_chunk():
    jpegs = [jpeg_bytes(v) for v in (1, 2)]
    assert MjpegParser().feed(jpegs[0] + jpegs[1]) == jpegs


# -- stream ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("chunked", [True, False], ids=["chunked", "plain"])
def test_reads_frames_from_either_http_framing(chunked):
    cam = FakeEsp32Cam(chunked=chunked).start()
    source = quick_source(cam).start()
    try:
        frame = source.wait_for_frame(0, timeout=3.0)
        assert frame is not None
        image, frame_id, stamp = frame
        assert image.shape == (120, 160, 3)
        assert frame_id >= 1 and abs(time.time() - stamp) < 2.0
    finally:
        source.stop()
        cam.stop()


def test_frames_split_across_small_writes_still_arrive():
    cam = FakeEsp32Cam(chunk_size=11).start()
    source = quick_source(cam).start()
    try:
        assert source.wait_for_frame(0, timeout=3.0) is not None
        assert source.decode_errors == 0
    finally:
        source.stop()
        cam.stop()


def test_a_frame_is_delivered_without_waiting_for_the_next_one():
    # One frame per second: if the reader waited to fill a buffer, frame 1 would need frame 2 to arrive.
    cam = FakeEsp32Cam(fps=1.0).start()
    source = quick_source(cam, read_timeout=2.0, stall_timeout=3.0).start()
    try:
        assert wait_until(lambda: cam.frames_sent >= 1)
        assert source.wait_for_frame(0, timeout=0.5) is not None
    finally:
        source.stop()
        cam.stop()


def test_only_the_latest_frame_is_kept(fake_cam):
    source = quick_source(fake_cam).start()
    try:
        assert wait_until(lambda: source.frames_received >= 20)
        _, frame_id, _ = source.wait_for_frame(0, timeout=1.0)
        time.sleep(0.15)  # the consumer is busy; frames arrive and replace each other
        _, next_id, _ = source.wait_for_frame(frame_id, timeout=1.0)
        assert next_id - frame_id > 3  # jumped straight to the newest, no backlog to drain
    finally:
        source.stop()


def test_wait_times_out_when_no_new_frame_exists():
    source = Esp32CamSource("http://127.0.0.1:1/stream")
    assert source.wait_for_frame(0, timeout=0.01) is None


def test_undecodable_frame_is_counted_and_skipped():
    source = Esp32CamSource("http://127.0.0.1:1/stream")
    source._publish(b"\xff\xd8garbage\xff\xd9")
    assert source.read() is None
    assert source.decode_errors == 1
    assert source.wait_for_frame(0, timeout=0.01) is None  # does not spin on the bad frame
    source._publish(jpeg_bytes(5))
    assert source.wait_for_frame(1, timeout=0.5) is not None


# -- recovery -------------------------------------------------------------------------------------
def test_state_goes_connecting_streaming_lost_streaming(fake_cam):
    source = quick_source(fake_cam, capture_after_failures=99)
    assert source.status()["state"] == CONNECTING
    source.start()
    try:
        assert wait_until(lambda: source.status()["state"] == STREAMING)
        fake_cam.stream_enabled = False
        fake_cam.drop_streams()
        assert wait_until(lambda: source.status()["state"] == LOST)
        before = source.status()["frame_id"]
        fake_cam.stream_enabled = True
        assert wait_until(lambda: source.status()["state"] == STREAMING and source.status()["frame_id"] > before)
        assert source.status()["reconnects"] >= 1
        assert fake_cam.stream_clients >= 2
    finally:
        source.stop()


def test_connecting_is_logged_for_the_journal(fake_cam, caplog):
    caplog.set_level("INFO", logger="gesturegrasp.camera")
    source = quick_source(fake_cam).start()
    try:
        assert wait_until(lambda: source.status()["state"] == STREAMING)
    finally:
        source.stop()
    assert "Camera connecting -> streaming" in caplog.text


def test_a_stalled_stream_is_treated_as_lost(fake_cam):
    source = quick_source(fake_cam, capture_after_failures=99).start()
    try:
        assert wait_until(lambda: source.status()["state"] == STREAMING)
        fake_cam.stall = True
        assert wait_until(lambda: source.status()["state"] == LOST)
    finally:
        source.stop()


def test_backoff_grows_and_stays_capped():
    source = Esp32CamSource("http://127.0.0.1:1/stream", backoff_min=0.5, backoff_max=8.0)
    delays = []
    source._read_stream = lambda: False
    source._stop.wait = lambda delay=None: delays.append(delay) or len(delays) >= 7
    source._stop.is_set = lambda: len(delays) >= 7
    source._run()
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_falls_back_to_capture_when_stream_fails(fake_cam):
    fake_cam.stream_enabled = False
    source = quick_source(fake_cam).start()
    try:
        assert wait_until(lambda: fake_cam.capture_requests >= 2)
        assert source.wait_for_frame(0, timeout=1.0) is not None
        assert source.status()["mode"] == "capture"
        assert source.status()["state"] == STREAMING
    finally:
        source.stop()


def test_returns_to_the_stream_after_capture_fallback(fake_cam):
    fake_cam.stream_enabled = False
    source = quick_source(fake_cam, capture_retry_seconds=0.1).start()
    try:
        assert wait_until(lambda: fake_cam.capture_requests >= 1)
        fake_cam.stream_enabled = True
        assert wait_until(lambda: fake_cam.stream_clients >= 1 and source.status()["mode"] == "stream")
    finally:
        source.stop()


# -- control and status ---------------------------------------------------------------------------
def test_settings_are_sent_on_every_connection(fake_cam):
    source = quick_source(fake_cam, framesize="vga", quality=12, flip="v", capture_after_failures=99).start()
    try:
        assert wait_until(lambda: source.status()["state"] == STREAMING)
        expected = [("framesize", "vga"), ("quality", "12"), ("hmirror", "0"), ("vflip", "1")]
        assert fake_cam.controls == expected
        fake_cam.drop_streams()  # a rebooted ESP32 has forgotten its settings
        assert wait_until(lambda: len(fake_cam.controls) >= 8)
        assert fake_cam.controls[4:8] == expected
    finally:
        source.stop()


def test_status_reports_esp32_data(fake_cam):
    source = quick_source(fake_cam).start()
    try:
        assert wait_until(lambda: source.status()["esp32"] is not None)
        status = source.status()
        assert status["esp32"]["rssi"] == -58
        assert status["fps"] >= 0 and status["url"] == fake_cam.stream_url
    finally:
        source.stop()


def test_software_flip_applies_when_the_camera_cannot_flip():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, 0] = 255
    ok, buf = cv2.imencode(".png", frame)
    source = Esp32CamSource("http://127.0.0.1:1/stream", flip="h")
    source._publish(buf.tobytes())
    image, _, _ = source.read()
    assert image[0, -1, 0] == 255 and image[0, 0, 0] == 0


def test_credentials_never_appear_in_status():
    source = Esp32CamSource("http://user:secret@10.0.0.2:81/stream")
    assert "secret" not in str(source.status())


# -- other sources --------------------------------------------------------------------------------
def test_image_source_gives_every_repeat_a_new_id(tmp_path):
    path = tmp_path / "hand.jpg"
    path.write_bytes(jpeg_bytes(50))
    source = ImageSource(path, fps=200.0).start()
    try:
        first = source.wait_for_frame(0, timeout=1.0)
        second = source.wait_for_frame(first[1], timeout=1.0)
        assert second[1] > first[1]
    finally:
        source.stop()


def test_video_file_source_loops(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 100.0, (64, 48))
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    source = VideoFileSource(path).start()
    try:
        assert wait_until(lambda: source.frames_received > 5)  # more frames than the file holds
    finally:
        source.stop()


@pytest.mark.parametrize("source, stream, control", [
    ("esp32cam.local", "http://esp32cam.local:81/stream", "http://esp32cam.local"),
    ("192.168.4.20", "http://192.168.4.20:81/stream", "http://192.168.4.20"),
    ("http://192.168.4.20:81/stream", "http://192.168.4.20:81/stream", "http://192.168.4.20"),
    ("http://192.168.4.20", "http://192.168.4.20:81/stream", "http://192.168.4.20"),
    ("http://10.0.0.5:8080/video", "http://10.0.0.5:8080/video", None),
])
def test_stream_url_resolution(source, stream, control):
    assert resolve_stream_url(source) == (stream, control)


def test_make_source_picks_the_right_kind(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(jpeg_bytes())
    video = tmp_path / "b.mp4"
    video.write_bytes(b"x")
    assert isinstance(make_source(str(image)), ImageSource)
    assert isinstance(make_source(str(video)), VideoFileSource)
    assert isinstance(make_source("esp32cam.local"), Esp32CamSource)


def test_make_source_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        make_source("missing.mp4")


def test_control_url_flag_overrides_the_derived_one():
    source = make_source("http://10.0.0.5:81/stream", control_url="http://10.0.0.6/")
    assert source.control_url == "http://10.0.0.6"
