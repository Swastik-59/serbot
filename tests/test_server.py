"""FastAPI server tests. Mock object detector, no network, no camera."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from src.gesturegrasp.pipeline import GesturePipeline


def _jpeg(h: int = 480, w: int = 640) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SERBOT_TOKEN", raising=False)
    with TestClient(create_app(GesturePipeline(use_mock_detector=True))) as c:
        yield c


def test_health_reports_loaded_components(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["detector"] == "mock"


def test_infer_returns_pipeline_result(client):
    res = client.post("/infer", content=_jpeg())
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "IDLE"
    assert body["hand_detected"] is False
    assert body["message"].startswith("Show the call-me sign")
    assert {o["label"] for o in body["objects"]} >= {"bottle"}


def test_infer_rejects_non_image(client):
    assert client.post("/infer", content=b"not a jpeg").status_code == 400


def test_websocket_lockstep(client):
    with client.websocket_connect("/ws/stream") as ws:
        for _ in range(3):
            ws.send_bytes(_jpeg())
            assert ws.receive_json()["state"] == "IDLE"
        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()


def test_stop_holds_even_from_idle(client):
    body = client.post("/stop").json()
    assert body["state"] == "STOPPED"
    assert body["robot"]["is_stopped"] is True
    # A following frame must not silently release the stop.
    after = client.post("/infer", content=_jpeg()).json()
    assert after["state"] == "STOPPED"
    assert after["robot"]["is_stopped"] is True


def test_reset_returns_to_idle(client):
    client.post("/stop")
    assert client.post("/reset").json()["state"] == "IDLE"


def test_token_required_when_configured(monkeypatch):
    monkeypatch.setenv("SERBOT_TOKEN", "s3cret")
    with TestClient(create_app(GesturePipeline(use_mock_detector=True))) as c:
        assert c.post("/infer", content=_jpeg()).status_code == 401
        assert c.post("/infer?token=s3cret", content=_jpeg()).status_code == 200
        assert c.post("/stop").status_code == 200  # stop never needs the token


def test_config_describes_gestures_and_model(client):
    cfg = client.get("/config").json()
    assert cfg["gestures"]["stop"]["intent"] == "STOP"
    assert 0 < cfg["gestures"]["point"]["min_conf"] <= 1
    assert cfg["model"]["feature_version"] == "2.0"
    assert "bottle" in cfg["allowed_objects"]
    assert "bottle" in cfg["map"]["objects"]


def test_latest_mirrors_the_last_frame(client):
    assert client.get("/latest").json().get("waiting") is True
    sent = client.post("/infer", content=_jpeg()).json()
    assert client.get("/latest").json()["state"] == sent["state"]


def test_result_carries_overlay_and_model_details(client):
    body = client.post("/infer", content=_jpeg(480, 640)).json()
    assert body["frame_size"] == [640, 480]
    for key in ("probabilities", "raw_probabilities", "margin", "intent_fired", "reason", "landmarks", "pointing_ray"):
        assert key in body


def test_playground_page_is_served(client):
    res = client.get("/playground")
    assert res.status_code == 200
    assert "GestureGrasp Playground" in res.text


@pytest.mark.skipif(not Path("data/raw/point").exists(), reason="needs the downloaded HaGRID sample images")
def test_real_hand_photo_sends_landmarks_and_probabilities(client):
    photo = sorted(Path("data/raw/point").glob("*.jpg"))[36].read_bytes()
    body = client.post("/infer", content=photo).json()
    assert body["hand_detected"] is True
    assert len(body["landmarks"]) == 21 and len(body["landmarks"][0]) == 2
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert body["gesture"] == "point"
