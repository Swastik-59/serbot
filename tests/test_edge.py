"""The on-Pi path: processing loop, YOLO thread, NumPy classifier, frame-rate timing, speech, flags."""

import logging
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fake_esp32 import FakeEsp32Cam, jpeg_bytes
from server.__main__ import RedactToken, build_parser
from server.app import InferenceService, create_app
from server.loop import CameraLoop
from server.speech import Speaker
from src.gesturegrasp import detector as detector_module
from src.gesturegrasp.camera import Esp32CamSource, ImageSource
from src.gesturegrasp.detector import ObjectDetector
from src.gesturegrasp.intent_engine import IntentEngine
from src.gesturegrasp.numpy_mlp import NumpyMLPClassifier, load_numpy_model
from src.gesturegrasp.pipeline import GesturePipeline


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def quick_source(cam: FakeEsp32Cam) -> Esp32CamSource:
    return Esp32CamSource(cam.stream_url, control_url=cam.control_url, connect_timeout=1.0, read_timeout=0.5,
                          backoff_min=0.01, backoff_max=0.05, status_interval=0.05)


@pytest.fixture
def fake_cam():
    cam = FakeEsp32Cam(fps=60.0).start()
    yield cam
    cam.stop()


@pytest.fixture
def live_client(fake_cam, monkeypatch):
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    with TestClient(create_app(GesturePipeline(use_mock_detector=True), camera=quick_source(fake_cam))) as client:
        yield client


# -- server loop ----------------------------------------------------------------------------------
def test_loop_produces_results_from_the_camera(live_client):
    assert wait_until(lambda: not live_client.get("/latest").json().get("waiting"))
    body = live_client.get("/latest").json()
    assert body["state"] == "IDLE" and body["frame_id"] >= 1
    assert body["frame_size"] == [160, 120]
    assert live_client.app.state.service.latest_jpeg is not None


def test_camera_endpoint_reports_source_and_loop(live_client):
    assert wait_until(lambda: live_client.get("/camera").json()["loop"]["processed"] >= 3)
    assert wait_until(lambda: live_client.get("/camera").json()["esp32"] is not None)
    body = live_client.get("/camera").json()
    assert body["state"] == "streaming" and body["kind"] == "esp32cam"
    assert body["esp32"]["rssi"] == -58
    assert body["last_frame_age_ms"] is not None
    assert body["loop"]["fps"] >= 0 and body["loop"]["errors"] == 0


def test_camera_endpoint_when_running_without_one(monkeypatch):
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    with TestClient(create_app(GesturePipeline(use_mock_detector=True))) as client:
        assert client.get("/camera").json() == {"state": "disabled"}
        assert client.get("/health").json()["camera"] == "disabled"


def test_stop_needs_no_token_while_the_loop_runs(fake_cam, monkeypatch):
    monkeypatch.setenv("SERBOT_TOKEN", "s3cret")
    with TestClient(create_app(GesturePipeline(use_mock_detector=True), camera=quick_source(fake_cam))) as client:
        assert wait_until(lambda: client.get("/camera").json()["loop"]["processed"] >= 2)
        response = client.post("/stop")
        assert response.status_code == 200
        assert response.json()["state"] == "STOPPED"
        seen = client.get("/camera").json()["loop"]["processed"]
        assert wait_until(lambda: client.get("/camera").json()["loop"]["processed"] > seen + 2)
        assert client.get("/state").json()["state"] == "STOPPED"  # new frames must not release it
        assert client.post("/reset").status_code == 401


def test_stop_works_when_the_camera_is_gone(monkeypatch):
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    source = Esp32CamSource("http://127.0.0.1:1/stream", connect_timeout=0.05, backoff_min=0.01, backoff_max=0.02)
    with TestClient(create_app(GesturePipeline(use_mock_detector=True), camera=source)) as client:
        assert wait_until(lambda: client.get("/camera").json()["state"] == "lost")
        assert client.post("/stop").json()["state"] == "STOPPED"


def test_dev_stream_pauses_the_camera_loop(live_client):
    assert wait_until(lambda: live_client.get("/camera").json()["loop"]["processed"] >= 2)
    with live_client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(jpeg_bytes())
        ws.receive_json()
        assert live_client.get("/camera").json()["loop"]["paused"] is True
        held = live_client.get("/camera").json()["loop"]["processed"]
        time.sleep(0.15)
        assert live_client.get("/camera").json()["loop"]["processed"] <= held + 1
    assert wait_until(lambda: live_client.get("/camera").json()["loop"]["processed"] > held + 1)


def test_each_frame_id_is_processed_once(fake_cam):
    source = quick_source(fake_cam).start()
    ids: list[int] = []
    loop = CameraLoop(source, lambda frame, stamp, frame_id: ids.append(frame_id)).start()
    try:
        assert wait_until(lambda: len(ids) >= 25)
    finally:
        loop.stop()
        source.stop()
    assert len(ids) == len(set(ids)) and ids == sorted(ids)


def test_no_backlog_when_inference_is_slower_than_capture(fake_cam):
    source = quick_source(fake_cam).start()
    ages: list[float] = []

    def slow(frame, stamp, frame_id):
        time.sleep(0.1)  # 10 fps pipeline behind a 60 fps camera
        ages.append(time.time() - stamp)

    loop = CameraLoop(source, slow).start()
    try:
        assert wait_until(lambda: len(ages) >= 8)
    finally:
        loop.stop()
        source.stop()
    assert max(ages) < 0.4  # a backlog would make this grow with every frame
    assert loop.skipped > 8


def test_a_failing_frame_does_not_end_the_loop(fake_cam):
    source = quick_source(fake_cam).start()
    calls: list[int] = []

    def flaky(frame, stamp, frame_id):
        calls.append(frame_id)
        if len(calls) == 1:
            raise RuntimeError("bad frame")

    loop = CameraLoop(source, flaky).start()
    try:
        assert wait_until(lambda: len(calls) >= 3)
        assert loop.stats()["errors"] == 1
    finally:
        loop.stop()
        source.stop()


def test_server_can_run_from_a_still_image(tmp_path, monkeypatch):
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    path = tmp_path / "hand.jpg"
    path.write_bytes(jpeg_bytes(40))
    with TestClient(create_app(GesturePipeline(use_mock_detector=True), camera=ImageSource(path, fps=100.0))) as client:
        assert wait_until(lambda: client.get("/camera").json()["loop"]["processed"] >= 3)


def test_root_and_viewer_pages_are_served(live_client):
    assert "GestureGrasp Playground" in live_client.get("/").text
    assert "EMERGENCY STOP" in live_client.get("/viewer").text


def test_page_works_from_any_device():
    page = Path("server/static/playground.html").read_text(encoding="utf-8")
    assert "ESP32-CAM (via Pi)" in page
    assert "localhost" not in page and "127.0.0.1" not in page


# -- YOLO thread ----------------------------------------------------------------------------------
class SlowModel:
    """Stands in for YOLO. Blocks for `delay` and answers with one bottle."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[dict] = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        box = SimpleNamespace(cls=[39], conf=[0.9], xyxy=[SimpleNamespace(tolist=lambda: [10.0, 20.0, 30.0, 40.0])])
        return [SimpleNamespace(boxes=[box], names={39: "bottle"})]


def real_detector(model: SlowModel, **kwargs) -> ObjectDetector:
    det = ObjectDetector(use_mock=True, **kwargs)
    det.use_mock = False
    det._model = model
    return det


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def test_yolo_never_blocks_the_gesture_loop():
    model = SlowModel(delay=0.3)
    det = real_detector(model)
    try:
        started = time.perf_counter()
        for _ in range(20):
            det.detect_latest(FRAME)
        assert time.perf_counter() - started < 0.1
        assert wait_until(lambda: det.detect_latest(FRAME) != [])
        assert det.detect_latest(FRAME)[0].label == "bottle"
    finally:
        det.close()


def test_yolo_runs_at_its_own_rate():
    model = SlowModel()
    det = real_detector(model, rate_hz=20.0)
    try:
        started = time.monotonic()
        while time.monotonic() - started < 0.5:
            det.detect_latest(FRAME)  # the gesture loop can call far more often than YOLO runs
            time.sleep(0.002)
    finally:
        det.close()
    assert 3 <= len(model.calls) <= 12


def test_old_boxes_are_dropped_when_the_frame_size_changes():
    det = real_detector(SlowModel())
    try:
        assert wait_until(lambda: det.detect_latest(FRAME) != [])
        assert det.detect_latest(np.zeros((24, 32, 3), dtype=np.uint8)) == []
    finally:
        det.close()


def test_yolo_is_called_for_a_cpu_with_no_gpu_options():
    model = SlowModel()
    assert real_detector(model).detect(FRAME)[0].label == "bottle"
    assert model.calls[0]["device"] == "cpu" and model.calls[0]["imgsz"] == 320
    assert "half" not in model.calls[0]


def test_mock_detector_answers_straight_away():
    det = ObjectDetector(use_mock=True)
    assert {o.label for o in det.detect_latest(FRAME)} >= {"bottle"}
    assert det._worker is None


def test_live_pipeline_does_not_wait_for_yolo():
    pipe = GesturePipeline(use_mock_detector=True)
    pipe.detector.close()
    pipe.detector = real_detector(SlowModel(delay=2.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        started = time.perf_counter()
        pipe.process(frame, timestamp=0.0, live=True)
        assert time.perf_counter() - started < 1.5
    finally:
        pipe.detector.close()
        pipe.hand_extractor.close()


def test_detector_uses_the_model_named_by_the_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert detector_module.default_model() == "yolo11n.pt"
    (tmp_path / "yolo11n_ncnn_model").mkdir()
    (tmp_path / "yolo_best.txt").write_text("yolo11n_ncnn_model", encoding="utf-8")
    assert detector_module.default_model() == "yolo11n_ncnn_model"
    (tmp_path / "yolo_best.txt").write_text("deleted_model", encoding="utf-8")
    assert detector_module.default_model() == "yolo11n.pt"


# -- NumPy classifier -----------------------------------------------------------------------------
MODELS = Path("models")


def test_numpy_model_matches_the_trained_model():
    pytest.importorskip("torch")
    import joblib  # only for the trained files committed in models/
    model = joblib.load(MODELS / "calibrated_model.joblib")
    scaler = joblib.load(MODELS / "scaler.joblib")
    exported = load_numpy_model(MODELS / "mlp_weights.npz", MODELS / "calibrated_model.joblib", MODELS / "scaler.joblib")
    assert exported is not None
    numpy_model, numpy_scaler = exported
    raw = np.random.default_rng(1).normal(scaler.mean_, scaler.scale_ * 2.0, (2000, scaler.mean_.size))
    original = model.predict_proba(scaler.transform(raw))
    assert np.abs(original - numpy_model.predict_proba(numpy_scaler.transform(raw))).max() < 1e-5
    assert list(numpy_model.classes_) == list(model.classes_)


def test_pipeline_runs_the_exported_model():
    pipe = GesturePipeline(use_mock_detector=True)
    pipe.hand_extractor.close()
    assert isinstance(pipe.model, NumpyMLPClassifier)


def test_a_stale_export_is_ignored(tmp_path):
    for name in ("calibrated_model.joblib", "scaler.joblib", "metadata.json", "mlp_weights.npz"):
        shutil.copy(MODELS / name, tmp_path / name)
    with open(tmp_path / "calibrated_model.joblib", "ab") as model:
        model.write(b"retrained")
    assert load_numpy_model(tmp_path / "mlp_weights.npz", tmp_path / "calibrated_model.joblib",
                            tmp_path / "scaler.joblib") is None


# -- timing at Pi frame rates ---------------------------------------------------------------------
PROBS = {"stop": 0.95, "no_gesture": 0.05}


def run_engine(fps: float, state: str = "LISTENING", gaps=()) -> tuple[IntentEngine, float]:
    """Feed a held STOP at this frame rate and return when it fired. gaps = frames with no hand."""
    engine = IntentEngine(Path("configs/intents.yaml"))
    for frame in range(int(fps * 3)):
        now = frame / fps
        result = engine.update({} if frame in gaps else PROBS, current_state=state,
                               hand_detected=frame not in gaps, timestamp=now)
        if result.intent_fired:
            return engine, now
    return engine, float("inf")


@pytest.mark.parametrize("fps", [8.0, 10.0, 15.0, 30.0])
def test_stop_fires_on_time_at_any_frame_rate(fps):
    _, fired_at = run_engine(fps)
    assert 0.25 <= fired_at <= 0.25 + 1.0 / fps + 1e-9


def test_a_lost_hand_expires_at_the_evaluated_time_on_a_slow_pi():
    # At 8 fps the second missing frame is 0.25 s after the hand, past the 0.2 s seen at 15 fps.
    engine = IntentEngine(Path("configs/intents.yaml"))
    engine.update(PROBS, "LISTENING", timestamp=0.0)
    assert engine.update({}, "LISTENING", hand_detected=False, timestamp=0.125).reason.endswith("(buffer active)")
    assert engine.update({}, "LISTENING", hand_detected=False, timestamp=0.25).reason == "No hand detected"
    assert engine._candidate_gesture is None


def test_the_frame_limit_is_unchanged_at_the_evaluated_rate():
    engine = IntentEngine(Path("configs/intents.yaml"))
    engine.update(PROBS, "LISTENING", timestamp=0.0)
    step = 1.0 / 15.0
    assert "buffer active" in engine.update({}, "LISTENING", hand_detected=False, timestamp=step).reason
    assert "buffer active" in engine.update({}, "LISTENING", hand_detected=False, timestamp=2 * step).reason
    assert "buffer active" not in engine.update({}, "LISTENING", hand_detected=False, timestamp=3 * step).reason


def test_one_missed_frame_does_not_break_a_hold_at_8_fps():
    _, fired_at = run_engine(8.0, gaps={1})
    assert fired_at < 0.5


# -- speech ---------------------------------------------------------------------------------------
class FakeProcess:
    def __init__(self, args, **kwargs):
        self.args = args
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.alive = False


def test_speaker_is_harmless_without_espeak(monkeypatch):
    monkeypatch.setattr("server.speech.shutil.which", lambda name: None)
    speaker = Speaker()
    assert speaker.available is False
    speaker.say("hello")


def test_a_newer_prompt_replaces_the_older_one(monkeypatch):
    monkeypatch.setattr("server.speech.shutil.which", lambda name: "/usr/bin/espeak-ng")
    monkeypatch.setattr("server.speech.subprocess.Popen", FakeProcess)
    speaker = Speaker()
    speaker.say("first")
    first = speaker._current
    speaker.say("second")
    assert first.alive is False and speaker._current.args[-1] == "second"


def test_service_speaks_once_per_state_change():
    said: list[str] = []
    service = InferenceService(GesturePipeline(use_mock_detector=True), SimpleNamespace(say=said.append))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    service.process(frame)
    service.process(frame)
    assert said == ["Show the call-me sign to start."]
    service.emergency_stop()
    assert said[-1].startswith("Stopped")
    assert len(said) == 2
    service.pipeline.hand_extractor.close()


# -- flags ----------------------------------------------------------------------------------------
def test_flags_have_environment_defaults(monkeypatch):
    monkeypatch.setenv("SERBOT_CAMERA_URL", "http://10.0.0.9:81/stream")
    monkeypatch.setenv("SERBOT_FLIP", "both")
    monkeypatch.setenv("SERBOT_SPEAK", "1")
    args = build_parser().parse_args([])
    assert args.camera_url == "http://10.0.0.9:81/stream"
    assert args.flip == "both" and args.speak is True


def test_source_is_another_name_for_camera_url():
    assert build_parser().parse_args(["--source", "clip.mp4"]).camera_url == "clip.mp4"


def test_token_is_redacted_from_logs():
    record = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, 'WebSocket /ws/stream?token=hunter2&x=1 [%s]',
                               ("accepted",), None)
    assert RedactToken().filter(record) is True
    assert "hunter2" not in record.getMessage()
    assert "accepted" in record.getMessage()
