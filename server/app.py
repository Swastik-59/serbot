"""
GestureGrasp server. Runs the whole system on the Raspberry Pi 5.

A background loop pulls the newest frame from the ESP32-CAM, runs the full GesturePipeline
(MediaPipe -> classifier -> IntentEngine -> pointing/YOLO -> FSM -> robot) and stores the result.
Any browser on the network opens the page and watches.

Endpoints:
    GET  /            The web page (same as /playground).
    GET  /playground  Live view with every model detail. Also takes a webcam or a photo for development.
    GET  /viewer      Minimal page: video, state and STOP, for old devices.
    GET  /video       MJPEG stream of annotated frames. Browsers watch this, never the ESP32.
    GET  /camera      Camera state, fps, frame age, ESP32 /status data, and the processing loop's speed.
    GET  /state       Current FSM + robot status.
    GET  /latest      Last frame's full result.
    GET  /config      Gestures, thresholds, states, robot map and model scores, for the page.
    GET  /health      Liveness + which model / detector is loaded.
    POST /stop        Emergency stop from any device, in every state. Never needs a token.
    POST /reset       Reset FSM, intent engine and robot to IDLE.

Optional development endpoints (not needed on the Pi):
    POST /infer       One JPEG in, one JSON result out, for curl and photos.
    WS   /ws/stream   Send one binary JPEG per message, get one JSON result back (lock-step).
                      The camera loop pauses while a client is connected, so two sources never
                      feed one state machine.

Config (environment variables, or the flags of `python -m server`):
    SERBOT_CAMERA_URL          Stream URL, host name, video file or image. "none" runs without a camera.
    SERBOT_CAMERA_CONTROL_URL  Base URL of the ESP32's /control, /status and /capture.
    SERBOT_FRAMESIZE           ESP32 frame size name, e.g. vga.
    SERBOT_QUALITY             ESP32 JPEG quality, lower is better (10-63).
    SERBOT_FLIP                none, h, v or both.
    SERBOT_SPEAK               "1" speaks prompts on the Pi through espeak-ng.
    SERBOT_TOKEN               If set, clients must send it as ?token=... or header X-Token.
    SERBOT_MOCK_DETECTOR       "1" uses fixed fake objects instead of YOLO (no weights needed).

Run:
    python -m server --host 0.0.0.0 --port 8000 --camera-url http://esp32cam.local:81/stream
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse

from server.loop import CameraLoop
from server.speech import Speaker
from src.gesturegrasp.camera import DEFAULT_CAMERA_URL, MAX_FRAME_BYTES, CameraSource, make_source
from src.gesturegrasp.pipeline import GesturePipeline, PipelineResult

logger = logging.getLogger("gesturegrasp.server")

# Short spoken/on-screen prompt per FSM state, so the user always knows what to do next.
STATE_PROMPTS = {
    "IDLE": "Show the call-me sign to start.",
    "LISTENING": "Point at the object you want.",
    "TARGET_PROPOSED": "Thumbs up to fetch the {target}, thumbs down to choose again.",
    "EXECUTING": "Fetching the {target}.",
    "AWAIT_FEEDBACK": "Here is the {target}. Show OK when done, thumbs down if it is wrong.",
    "RETURNING": "Going back to the dock.",
    "STOPPED": "Stopped. Show the call-me sign to continue.",
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _camera_from_env() -> Optional[CameraSource]:
    source = os.environ.get("SERBOT_CAMERA_URL", DEFAULT_CAMERA_URL).strip()
    if source.lower() in {"", "none", "off"}:
        logger.info("Running without a camera: use /infer or /ws/stream.")
        return None
    quality = os.environ.get("SERBOT_QUALITY", "").strip()
    return make_source(source,
                       control_url=os.environ.get("SERBOT_CAMERA_CONTROL_URL") or None,
                       framesize=os.environ.get("SERBOT_FRAMESIZE") or None,
                       quality=int(quality) if quality else None,
                       flip=os.environ.get("SERBOT_FLIP") or "none")


def decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    if not data or len(data) > MAX_FRAME_BYTES:
        return None
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size else None


class InferenceService:
    """Owns the single pipeline. One lock serialises frames, since the FSM is one shared session."""

    def __init__(self, pipeline: GesturePipeline, speaker: Optional[Speaker] = None):
        self.pipeline = pipeline
        self.dev_streams = 0  # open /ws/stream clients; the camera loop pauses while there are any
        self._speaker = speaker
        self._spoken_state: Optional[str] = None
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_result: Optional[dict[str, Any]] = None
        self._frame_count = 0
        self._commands: list[dict[str, Any]] = []  # fired commands, newest last (kept short)

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None,
                frame_id: Optional[int] = None) -> dict[str, Any]:
        with self._lock:
            result = self.pipeline.process(frame, timestamp=timestamp)
            self._frame_count += 1
            if result.annotated_frame is not None:
                ok, buf = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._latest_jpeg = buf.tobytes()
            if result.intent_fired:
                self._commands.append({
                    "id": self._frame_count, "time": time.time(), "gesture": result.gesture,
                    "intent": result.intent, "state": result.state,
                    "target": _target_label(self.pipeline.state_machine.proposed_target),
                })
                del self._commands[:-20]
            self._latest_result = self._serialize(result, frame.shape[:2])
            self._latest_result["frame_id"] = frame_id
            self._announce(self._latest_result["state"], self._latest_result["message"])
            return self._latest_result

    @property
    def latest_result(self) -> Optional[dict[str, Any]]:
        return self._latest_result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            sm = self.pipeline.state_machine
            return {
                "state": sm.current_state,
                "proposed_target": _target_label(sm.proposed_target),
                "message": self._message(sm.current_state, sm.proposed_target),
                "robot": self.pipeline.robot.get_status().to_dict(),
                "events": self._events(),
                "frames_processed": self._frame_count,
            }

    def emergency_stop(self) -> dict[str, Any]:
        with self._lock:
            self.pipeline.robot.emergency_stop()
            sm = self.pipeline.state_machine
            if sm.current_state != "STOPPED":
                # Direct transition: FSM handle_intent("STOP") ignores IDLE, but a manual stop must hold.
                sm.transition_to("STOPPED", "MANUAL_STOP", "Emergency stop from API")
            self._announce(sm.current_state, self._message(sm.current_state, sm.proposed_target))
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.pipeline.reset()
            self._commands.clear()
            sm = self.pipeline.state_machine
            self._announce(sm.current_state, self._message(sm.current_state, sm.proposed_target))
        return self.snapshot()

    def close(self) -> None:
        if self._speaker is not None:
            self._speaker.close()

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        return self._latest_jpeg

    def _announce(self, state: str, message: str) -> None:
        """Speak once per FSM state change. Non-blocking, so holding the lock here is fine."""
        if state != self._spoken_state:
            self._spoken_state = state
            if self._speaker is not None:
                self._speaker.say(message)

    def _events(self) -> list[str]:
        return [
            f"[{e.from_state} -> {e.to_state}] {e.trigger}: {e.details}"
            for e in self.pipeline.state_machine.event_log[-5:]
        ]

    @staticmethod
    def _message(state: str, target: Optional[dict[str, Any]]) -> str:
        return STATE_PROMPTS.get(state, "").format(target=_target_label(target) or "object")

    def _serialize(self, r: PipelineResult, frame_shape: tuple[int, int]) -> dict[str, Any]:
        sm = self.pipeline.state_machine
        ranked = sorted(r.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        h, w = frame_shape
        # The ray is only meaningful while the hand is pointing; other gestures get no target.
        pointing = r.gesture == "point" and r.hand_detected
        target = r.selected_target if pointing else None
        return {
            "hand_detected": r.hand_detected,
            "gesture": r.gesture,
            "confidence": round(r.confidence, 3),
            "top3": [{"gesture": g, "p": round(p, 3)} for g, p in ranked[:3]],
            # Smoothed over frames (what the IntentEngine acts on) and this frame alone.
            "probabilities": {g: round(p, 4) for g, p in r.probabilities.items()},
            "raw_probabilities": {g: round(p, 4) for g, p in r.raw_probabilities.items()},
            "margin": round(ranked[0][1] - ranked[1][1], 3) if len(ranked) > 1 else None,
            "intent": r.intent,
            "intent_fired": r.intent_fired,
            "reason": r.reason,
            "hold_progress": round(r.hold_progress, 2),
            "state": r.state,
            "message": self._message(r.state, sm.proposed_target),
            "pointing_at": _target_label(target),
            "pointing": None if target is None else {
                "label": target["label"],
                "angular_deviation_deg": target.get("angular_deviation_deg"),
                "direct_hit": target.get("direct_hit"),
                "hold_progress": round(target.get("hold_progress", 0.0), 2),
            },
            "proposed_target": _target_label(sm.proposed_target),
            "frame_size": [w, h],
            # Normalised (0..1) hand landmarks and the pointing ray in frame pixels, for overlays.
            "landmarks": None if r.landmarks is None else np.round(np.asarray(r.landmarks)[:, :2], 4).tolist(),
            "pointing_ray": None if r.pointing_ray is None or not pointing else {
                "origin": [round(v, 1) for v in r.pointing_ray.origin],
                "tip": [round(v, 1) for v in r.pointing_ray.tip],
                "direction": [round(v, 4) for v in r.pointing_ray.direction],
            },
            "objects": [
                {"label": o.label, "confidence": round(o.confidence, 2), "bbox": [round(v, 1) for v in o.bbox_xyxy]}
                for o in r.detected_objects
            ],
            "robot": r.robot_status.to_dict(),
            "events": self._events(),
            "commands": list(self._commands),
            "latency_ms": round(r.latency_ms, 1),
        }


def _target_label(target: Optional[dict[str, Any]]) -> Optional[str]:
    return target.get("label") if target else None


def create_app(pipeline: Optional[GesturePipeline] = None, camera: Optional[CameraSource] = None) -> FastAPI:
    """Build the app. Tests pass their own pipeline and, if they want the loop, a camera.
    Production passes neither: both are built at startup from the environment."""
    token = os.environ.get("SERBOT_TOKEN") or None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        speaker = Speaker() if _env_flag("SERBOT_SPEAK") else None
        if pipeline is None:
            mock = _env_flag("SERBOT_MOCK_DETECTOR")
            logger.info("Loading GesturePipeline (mock detector=%s)...", mock)
            svc = InferenceService(GesturePipeline(use_mock_detector=mock), speaker)
            source = camera or _camera_from_env()
        else:
            svc = InferenceService(pipeline, speaker)
            source = camera
        app.state.service = svc
        app.state.camera = source
        app.state.loop = None
        if source is not None:
            source.start()
            app.state.loop = CameraLoop(source, lambda frame, stamp, frame_id: svc.process(frame, stamp, frame_id),
                                        paused=lambda: svc.dev_streams > 0).start()
        logger.info("Server ready.")
        yield
        if app.state.loop is not None:
            app.state.loop.stop()
        if source is not None:
            source.stop()
        svc.close()
        svc.pipeline.hand_extractor.close()

    app = FastAPI(title="GestureGrasp Server", lifespan=lifespan)

    def check_token(supplied: Optional[str]) -> None:
        if token and supplied != token:
            raise HTTPException(status_code=401, detail="Bad or missing token")

    def service(request_or_ws) -> InferenceService:
        return request_or_ws.app.state.service

    @app.get("/health")
    def health(request: Request):
        p = service(request).pipeline
        return {
            "status": "ok",
            "classifier": type(p.model).__name__,
            "detector": p.detector.active_model_name,
            "detector_device": p.detector.device,
            "feature_version": p.feature_version,
            "robot": type(p.robot).__name__,
            "camera": request.app.state.camera.status()["state"] if request.app.state.camera else "disabled",
        }

    @app.get("/camera")
    def camera_status(request: Request):
        """Source state, fps, frame age, ESP32 /status data, and how fast the loop keeps up."""
        source, loop = request.app.state.camera, request.app.state.loop
        if source is None:
            return {"state": "disabled"}
        return {**source.status(), "loop": loop.stats()}

    @app.post("/infer")
    async def infer(request: Request):
        check_token(request.query_params.get("token") or request.headers.get("x-token"))
        frame = decode_jpeg(await request.body())
        if frame is None:
            raise HTTPException(status_code=400, detail="Body must be one JPEG/PNG image under 4 MB")
        return await run_in_threadpool(service(request).process, frame)

    @app.get("/state")
    def state(request: Request):
        return service(request).snapshot()

    @app.get("/latest")
    def latest(request: Request):
        """Last frame's full result, whichever source it came from."""
        return service(request).latest_result or {"hand_detected": False, "waiting": True}

    @app.get("/config")
    def config(request: Request):
        """What the playground needs to explain results: gestures, thresholds, states, map, model scores."""
        p = service(request).pipeline
        cfg = yaml.safe_load(p.config_path.read_text(encoding="utf-8"))
        meta = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
        report_path = Path("reports/evaluation_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        live = report.get("live_mediapipe_test") or {}
        robot = p.robot
        return {
            "gestures": {g: {k: m.get(k) for k in ("intent", "actionable", "hold_time", "min_conf", "description")}
                         for g, m in cfg["gestures"].items()},
            "states": cfg["state_machine"]["states"],
            "margin_threshold": cfg["intent_engine"]["margin_threshold"],
            "allowed_objects": cfg["pointing"]["allowed_classes"],
            "model": {
                "name": meta.get("champion_model"), "feature_version": meta.get("feature_version"),
                "classes": meta.get("classes"), "trained": meta.get("timestamp"),
                "live_macro_f1": live.get("macro_f1"), "stop_recall": live.get("stop_recall"),
                "age_60_plus_macro_f1": (report.get("age_60_plus_test") or {}).get("macro_f1"),
                "temporal": {g: v.get("fired_correctly") for g, v in
                             ((report.get("temporal") or {}).get("per_gesture") or {}).items()},
                "detector": p.detector.active_model_name, "detector_device": p.detector.device,
            },
            "map": {
                "dock": list(getattr(robot, "dock_pos", (100.0, 100.0))),
                "user": list(getattr(robot, "user_pos", (300.0, 520.0))),
                "objects": {k: list(v) for k, v in (getattr(robot, "arena_objects", {}) or {}).items()},
            },
        }

    @app.get("/playground", response_class=HTMLResponse)
    def playground():
        return (Path(__file__).parent / "static" / "playground.html").read_text(encoding="utf-8")

    @app.post("/stop")
    def stop(request: Request):
        # No token check: anyone in the room must be able to stop the robot.
        return service(request).emergency_stop()

    @app.post("/reset")
    def reset(request: Request):
        check_token(request.query_params.get("token") or request.headers.get("x-token"))
        return service(request).reset()

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        supplied = ws.query_params.get("token") or ws.headers.get("x-token")
        if token and supplied != token:
            await ws.close(code=1008)
            return
        await ws.accept()
        svc = service(ws)
        client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
        logger.info("Development stream connected: %s", client)
        svc.dev_streams += 1
        try:
            while True:
                data = await ws.receive_bytes()
                frame = decode_jpeg(data)
                if frame is None:
                    await ws.send_json({"error": "undecodable frame"})
                    continue
                result = await run_in_threadpool(svc.process, frame)
                await ws.send_json(result)
        except WebSocketDisconnect:
            logger.info("Development stream disconnected: %s", client)
        finally:
            svc.dev_streams -= 1

    @app.get("/video")
    async def video(request: Request):
        svc = service(request)

        async def frames():
            last = None
            while not await request.is_disconnected():
                jpeg = svc.latest_jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.03)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return playground()

    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        return VIEWER_HTML

    return app


VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>GestureGrasp Server</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 img{max-width:100%;border:1px solid #333;background:#000;min-width:320px;min-height:240px}
 #msg{font-size:1.6em;margin:12px 0}
 button{font-size:1.4em;padding:12px 28px;background:#c00;color:#fff;border:0;border-radius:6px;cursor:pointer}
 pre{background:#222;padding:8px;max-width:520px;white-space:pre-wrap}
</style></head><body>
<h2>GestureGrasp</h2>
<div id="msg">&hellip;</div>
<div class="row">
 <img src="/video" alt="Waiting for frames from the ESP32-CAM">
 <div><button onclick="fetch('/stop',{method:'POST'})">EMERGENCY STOP</button>
 <button style="background:#444" onclick="fetch('/reset'+location.search,{method:'POST'})">Reset</button>
 <pre id="state"></pre></div>
</div>
<script>
async function poll(){
 try{const s=await (await fetch('/state')).json();
  document.getElementById('msg').textContent=s.state+': '+s.message;
  document.getElementById('state').textContent=JSON.stringify(s,null,1);}catch(e){}
 setTimeout(poll,500);}
poll();
</script></body></html>"""


app = create_app()
