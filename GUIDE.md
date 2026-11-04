# GestureGrasp guide: ESP32-CAM + Raspberry Pi 5 + web page

This guide takes you from nothing to a working system. An ESP32-CAM streams video over Wi-Fi to a
Raspberry Pi 5. The Pi runs the whole pipeline and serves a web page, which any phone or computer on the
network can open. For the gestures themselves, see [GESTURES.md](GESTURES.md).

```
 ESP32-CAM                     Raspberry Pi 5                                Any browser
 ─────────                     ──────────────                                ───────────
 OV2640 ──MJPEG over Wi-Fi──►  capture thread (newest frame only)
                               → MediaPipe hands → gesture model
                               → IntentEngine (hold, smoothing)
                               → pointing + YOLO (its own thread)
                               → state machine → robot (simulated for now)
                               → web server, port 8000  ──────────────────►  http://<pi>:8000/
```

The ESP32-CAM serves one client, so the Pi must be the only consumer of its stream. Browsers watch the
Pi's `/video`, never the ESP32 directly.

**Just want to see it work?** No hardware needed. On any PC with the requirements installed:

```bash
python -m server --source some_photo.jpg --mock-detector
```

Then open **http://localhost:8000/**. The webcam tab also works there (2.5).

---

## 1. What is where

| Path | What it is |
|---|---|
| `esp32_cam/` | Firmware, `config.h` and the flashing guide |
| `server/` | FastAPI server: `python -m server`. Processing loop, speech, REST endpoints |
| `server/static/playground.html` | The web page, a single self-contained file |
| `src/gesturegrasp/camera.py` | Reads the ESP32-CAM stream; also video files and images for testing |
| `src/gesturegrasp/pipeline.py` | One frame in, one decision out: hands → model → intent → pointing → state machine → robot |
| `src/gesturegrasp/` (others) | Hand tracking, features, intent engine, state machine, pointing, YOLO, training |
| `models/` | The trained model: `calibrated_model.joblib`, `scaler.joblib`, `metadata.json`, and `mlp_weights.npz` (the same model for NumPy) |
| `configs/intents.yaml` | Single source of truth: gestures, hold times, thresholds, states, objects |
| `robot/` | Robot interface (`base.py`) and the simulated robot used until real hardware exists |
| `scripts/` | `benchmark.py`, `export_yolo_pi.py`, `export_mlp_numpy.py`, `draw_diagrams.py` |
| `deploy/gesturegrasp.service` | Starts the server when the Pi boots |
| `requirements-pi.txt` | Everything the Pi needs |
| `requirements-dev.txt` | GPU PC: training and `download_dataset.py` only |
| `reports/` | Latest evaluation numbers from training |
| `tests/` | `pytest -q`. `tests/fake_esp32.py` is a stand-in for the camera |

---

## 2. Raspberry Pi 5: install and start

Flash the camera first: [esp32_cam/README.md](esp32_cam/README.md). Keep its IP address handy.

### 2.1 Install

Use **Raspberry Pi OS Bookworm 64-bit**, which has Python 3.11. Newer releases ship a Python that MediaPipe
has no wheel for.

```bash
sudo apt update
sudo apt install -y python3-venv espeak-ng libgl1 libglib2.0-0 libportaudio2 avahi-daemon libnss-mdns
git clone https://github.com/Swastik-59/serbot.git
cd serbot
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-pi.txt
```

Two pins in that file matter, so don't upgrade them on their own:

- **`mediapipe==0.10.14`** is the closest version to the one used for evaluation that has an aarch64 wheel
  (0.10.11 has none). It gave identical landmarks and identical classifications on 1,251 hand photos, on x86.
- **`protobuf==4.25.9`**. A newer one breaks MediaPipe at start-up.

### 2.2 Export YOLO for this Pi

Do this once, while the Pi has internet:

```bash
python scripts/export_yolo_pi.py
```

It downloads `yolo11n.pt`, exports NCNN, times both, and writes the winner to `yolo_best.txt`, which the
server then loads. The script has Ultralytics' auto-install turned off on purpose, because mid-run it had
upgraded protobuf and torch and broken both MediaPipe and YOLO.

### 2.3 Start the server

Run everything from the project root.

```bash
python -m server --camera-url http://esp32cam.local:81/stream
```

| Flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--camera-url`, `--source` | `SERBOT_CAMERA_URL` | `http://esp32cam.local:81/stream` | Stream URL, `esp32cam.local`, an IP, a video file, an image, or `none` |
| `--camera-control-url` | `SERBOT_CAMERA_CONTROL_URL` | same host, port 80 | Base URL of the ESP32's `/control`, `/status`, `/capture` |
| `--framesize` | `SERBOT_FRAMESIZE` | `vga` | ESP32 frame size, sent on every connection |
| `--quality` | `SERBOT_QUALITY` | `12` | ESP32 JPEG quality, 10 (best) to 63 |
| `--flip` | `SERBOT_FLIP` | `none` | `h` mirror, `v` upside down, `both` rotated 180° |
| `--speak` | `SERBOT_SPEAK` | off | Speak prompts on the Pi with espeak-ng |
| `--max-hands` | `SERBOT_MAX_HANDS` | `2` | See 2.6 |
| `--token` | `SERBOT_TOKEN` | none | Secret for `/reset`, `/infer`, `/ws/stream`. Never needed for `/stop` |
| `--mock-detector` | `SERBOT_MOCK_DETECTOR` | off | Fixed test objects instead of YOLO |
| `--host`, `--port` | `SERBOT_HOST`, `SERBOT_PORT` | `0.0.0.0`, `8000` | Where to listen |

When it is ready you will see:

```
Open http://<this-pi>:8000/ on any device on the network
Server ready.
Uvicorn running on http://0.0.0.0:8000
Camera connecting -> streaming
```

Stop it with Ctrl+C.

### 2.4 Check it

```bash
curl http://localhost:8000/health
curl http://localhost:8000/camera
```

`/health` should show `"classifier":"NumpyMLPClassifier"` and `"feature_version":"2.0"`, with `"detector"`
naming your YOLO export. `/camera` should show `"state":"streaming"`.

### 2.5 The web page

Open **http://\<pi-ip\>:8000/** on any device on the network.

| Source | What it does |
|---|---|
| **ESP32-CAM (via Pi)** | The default: watches the live annotated video |
| **Your webcam** | Development: sends this device's webcam to the pipeline. The camera loop pauses while this is on, so two sources never feed one state machine |
| **A photo** | Development: analyses one uploaded image (shows what the model reads; no commands, since nothing is held) |

What it shows:

- **The sentence:** a plain-language reading of your hand, with a bar that fills while you hold
  the gesture and a "Sent ✓" badge when a command fires.
- **Camera badge:** connected or lost, frames per second, frame age and Wi-Fi signal.
- **What the model sees:** confidence for all 9 classes, each with a tick at the confidence that
  gesture needs.
- **Decision:** confidence vs. required, lead over the runner-up gesture, hold time, the command,
  and in plain words why it did or didn't fire.
- **Pointing & objects:** the object being pointed at, how far off-centre, how steadily, and every
  object detected.
- **Speed:** server processing time, camera-to-result time, frames per second.
- **The robot:** its task step by step, a top-down map of it moving, and the log of commands sent.

Try this:

1. Show 🤙 **call me** until the bar fills. The step track moves from *Asleep* to *Listening*.
2. 👉 **Point** at a bottle, cup or book. The sentence names it.
3. 👍 **Thumbs up**. The map shows the robot driving out, picking it up and bringing it to *you*.
4. 👌 **OK** once it says *Delivered*. The robot hands it over and returns to the dock.

Any time: ✋ **stop**, or press **Stop robot**.

Notes:

- **Webcam tab:** browsers only allow the webcam on `localhost` or HTTPS, so on a phone use the ESP32-CAM tab.
- **Spoken prompts:** `--speak` speaks on the Pi. *Speak prompts here* on the page speaks in that browser
  instead, which suits a phone or tablet next to the user.
- **Token:** if the server was started with `--token`, open the page as `http://<pi>:8000/?token=mysecret`.
  Viewing and STOP work without it; the token is for reset and sending frames.
- **Fonts:** the page loads them from Google Fonts. Offline, such as on the Pi's hotspot, it falls back to
  system fonts and works the same.

### 2.6 Measure and tune speed

```bash
python scripts/benchmark.py --source http://esp32cam.local:81/stream
```

Stop the server first: the ESP32 stream serves one client. This prints the time for each stage (decode,
MediaPipe, classifier, YOLO, draw and encode), the real loop's frame rate, and how old each frame was when
its result was ready. The targets are at least 10 fps and under 250 ms.

If the Pi falls short, try these in order:

1. **Cooling.** An active cooler keeps a Pi 5 from throttling under this load.
2. **`--max-hands 1`.** With 2, MediaPipe runs its palm detector on every frame until it finds a second
   hand, which cost about twice as much in our test (58 ms vs 28 ms). The price: if two hands are in view,
   a resting hand can hide the one making the command.
3. **`--framesize qvga`** is cheaper per frame, but a hand at 1–2 m may be too small to detect.

---

## 3. Start automatically, and networking

### 3.1 Start on boot

```bash
sudo cp deploy/gesturegrasp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gesturegrasp
journalctl -u gesturegrasp -f      # watch it run (Ctrl+C to leave)
```

Edit `User`, `WorkingDirectory` and the paths in the unit first if yours differ. Current Pi images ask you
to create a user, so yours may not be `pi`. Settings and secrets go in `/etc/default/gesturegrasp`, one per
line, such as `SERBOT_TOKEN=...` or `SERBOT_SPEAK=1`.

It restarts by itself if it stops, and it reconnects to the camera by itself if the ESP32 reboots.

### 3.2 Use the IP if `.local` does not resolve

`esp32cam.local` needs `avahi-daemon` and `libnss-mdns` on the Pi, and a network that passes multicast. If it
does not resolve, use the IP printed on the ESP32's serial log and reserve that address in your router:

```bash
python -m server --camera-url http://192.168.1.50:81/stream
```

### 3.3 Find the Pi's address

```bash
hostname -I
```

`http://<hostname>.local:8000/` also works on most phones and computers.

### 3.4 Optional: the Pi as its own Wi-Fi hotspot

With no router around, the Pi can make the network. The ESP32-CAM and your phone join it. Needs hardware check.

```bash
sudo nmcli device wifi hotspot ifname wlan0 con-name gesturegrasp ssid GestureGrasp password "choose-8-or-more-characters"
sudo nmcli connection modify gesturegrasp connection.autoconnect yes 802-11-wireless.band bg
sudo nmcli connection up gesturegrasp
```

`802-11-wireless.band bg` keeps it on 2.4 GHz, the only band the ESP32 can see. Put the same SSID and password
in `esp32_cam/config.h` and flash again. The Pi is then at `10.42.0.1`, so the page is at
`http://10.42.0.1:8000/`. If `esp32cam.local` does not resolve, find the camera's address with:

```bash
cat /var/lib/NetworkManager/dnsmasq-wlan0.leases
```

A hotspot has no internet, so finish 2.1 and 2.2 first.

---

## 4. Endpoints and messages

### 4.1 HTTP endpoints

| Endpoint | Token? | Purpose |
|---|---|---|
| `GET /`, `/playground` | no | The web page |
| `GET /viewer` | no | Minimal page: video, state, STOP |
| `GET /video` | no | MJPEG stream of annotated frames |
| `GET /camera` | no | Camera state, fps, frame age, ESP32 status, loop speed |
| `GET /health` | no | Loaded model, detector and camera state |
| `GET /state` | no | Current state, prompt, robot status, recent events |
| `GET /latest` | no | The last frame's full result |
| `GET /config` | no | Gestures, thresholds, states, robot map and model scores |
| `POST /stop` | **never** | Emergency stop, works in every state, even with the camera down. Anyone in the room must be able to stop the robot |
| `POST /reset` | yes | Back to IDLE, robot docked |
| `POST /infer` | yes | Development: body = one JPEG; returns the same JSON as `/latest` |
| `WS /ws/stream` | yes | Development: one binary JPEG per message, one JSON reply per frame. The camera loop pauses while a client is connected |

```bash
curl -X POST http://<pi>:8000/stop
curl -X POST --data-binary @hand.jpg http://<pi>:8000/infer
```

The server runs one shared session: one camera, one robot, one state machine.

### 4.2 `/camera`

```json
{
  "kind": "esp32cam", "state": "streaming", "mode": "stream",
  "fps": 19.7, "last_frame_age_ms": 31.0, "frame_id": 380, "frames_received": 380,
  "reconnects": 0, "decode_errors": 0, "flip": "none", "last_error": null,
  "url": "http://esp32cam.local:81/stream",
  "esp32": {"framesize": "vga", "quality": 12, "fps": 19.8, "uptime_s": 311, "free_heap": 150000,
            "free_psram": 3400000, "rssi": -58, "ip": "192.168.1.50", "hmirror": 0, "vflip": 0},
  "loop": {"fps": 12.1, "processed": 300, "skipped": 78, "errors": 0,
           "latency_ms": 60.4, "frame_age_ms": 112.1, "paused": false}
}
```

| Field | Meaning |
|---|---|
| `state` | `connecting` (no frame yet), `streaming`, or `lost` (retrying) |
| `mode` | `stream`, or `capture` when `/stream` keeps failing and the Pi polls `/capture` instead |
| `fps`, `last_frame_age_ms` | Frames arriving from the ESP32, and how old the newest one is |
| `esp32` | The board's own `/status`, refreshed every 5 s. `null` if unreachable |
| `loop.fps`, `loop.latency_ms` | How fast the pipeline keeps up, and the last frame's processing time |
| `loop.skipped` | Frames replaced before the pipeline reached them. Normal: this is how backlog is avoided |
| `loop.frame_age_ms` | Time from a frame reaching the Pi to its result |

### 4.3 `/latest`

Shortened; `...` marks trimmed lists:

```json
{
  "hand_detected": true,
  "gesture": "point", "confidence": 0.993, "margin": 0.988,
  "top3": [{"gesture": "point", "p": 0.993}, {"gesture": "no_gesture", "p": 0.005}, {"gesture": "fist", "p": 0.001}],
  "probabilities": {"point": 0.993, "no_gesture": 0.005, "fist": 0.001, "...": "..."},
  "raw_probabilities": {"point": 0.997, "...": "..."},
  "intent": "SELECT_TARGET", "intent_fired": false, "hold_progress": 0.62,
  "reason": "Holding SELECT_TARGET (62%)",
  "state": "LISTENING",
  "message": "Point at the object you want.",
  "pointing_at": "bottle",
  "pointing": {"label": "bottle", "angular_deviation_deg": 4.2, "direct_hit": true, "hold_progress": 0.6},
  "proposed_target": null,
  "frame_id": 1042,
  "frame_size": [640, 480],
  "landmarks": [[0.512, 0.744], "... 21 points (x, y), 0..1 of the frame"],
  "pointing_ray": {"origin": [331.0, 352.4], "tip": [377.9, 301.2], "direction": [0.675, -0.737]},
  "objects": [{"label": "bottle", "confidence": 0.87, "bbox": [412.0, 150.3, 470.8, 281.0]}],
  "robot": {"state": "IDLE", "current_position": [100.0, 100.0], "holding_object": null, "is_stopped": false, "...": "..."},
  "events": ["[IDLE -> LISTENING] WAKE: User wake gesture recognized"],
  "commands": [{"id": 41, "time": 1760377169.3, "gesture": "call", "intent": "WAKE", "state": "LISTENING", "target": null}],
  "latency_ms": 41.7
}
```

| Field | Meaning |
|---|---|
| `gesture`, `confidence` | Most likely gesture after smoothing over recent frames, and its probability |
| `probabilities` / `raw_probabilities` | All 9 classes: smoothed (what decisions use) / this frame alone |
| `margin` | Lead of the top gesture over the runner-up (must be ≥ 0.20 to count) |
| `intent`, `hold_progress`, `intent_fired` | The command this gesture means, how much of its hold is done (0–1), and whether it fired on this frame |
| `reason` | Why the decision logic did or did not act |
| `state`, `message` | Current step of the task and the prompt to show or speak |
| `pointing_at`, `pointing`, `pointing_ray` | The object under the finger. Filled **only while the gesture is point**, otherwise `null` |
| `proposed_target` | The object waiting for thumbs up / down |
| `frame_id` | The camera frame this result came from |
| `landmarks`, `frame_size` | Hand points as fractions of the frame, and the frame's width/height in pixels |
| `objects` | Allowed objects YOLO found, with boxes in frame pixels. On a live stream they can be a few frames old |
| `robot` | Robot status: state, position, what it holds, whether it is stopped |
| `events`, `commands` | Last 5 state changes, and the last 20 commands that fired (also kept between frames) |
| `latency_ms` | Processing time for this frame |

**The system moves forward on frames.** The state machine, timeouts and simulated robot advance each
time a frame is processed. If the camera is lost they pause where they are until frames resume. `/stop` still
works during that time.

---

## 5. The model

### 5.1 What it is

- **Input:** 21 hand landmarks from MediaPipe Hands (x, y), turned into 126 features: positions,
  finger bend angles, fingertip distances, finger extension, hand roll. No handedness needed.
- **Classifier:** a neural network (126 → 512 → 256 → 128 → 9) trained on a GPU,
  temperature-calibrated so its confidence matches reality. On the Pi it runs from
  `models/mlp_weights.npz` with NumPy alone: the same weights, without torch or scikit-learn.
- **Classes:** the 7 commands plus `no_gesture` and `fist`, which are trained "not a command" classes.
- **Training data:** 435,939 hands from 40,159 people (HaGRID v2). It also includes open palms
  (trained as STOP), everyday poses (trained as "not a command"), and every hand at random
  rotations.
- **On top of the model:** the IntentEngine smooths confidence over frames and needs a full hold
  before a command fires. The state machine only accepts commands that make sense in the current
  step. Pointing needs the same object for most of the hold.

### 5.2 How good it is

All numbers are on people held out of training. The full figures are in `reports/evaluation_report.json`.

| Measure | Result |
|---|---|
| Live path (our MediaPipe on real photos), macro-F1 | 0.979 |
| STOP recall | 99.0% |
| Users aged 60+ (2,209 hands), macro-F1 | 0.985 |
| Hands at random rotations, macro-F1 | 0.988 |
| Commands that fire within 2 s with tremor-like jitter | 95–99.8% |
| False commands from idle/everyday hands | 0.04 per minute |

It has not been validated with real elderly or Parkinson's users yet. Simulated tremor is a stress
test, not clinical evidence.

### 5.3 Retrain

Only needed if you change the gestures, thresholds or training data. This runs on a GPU PC, not on the Pi.
Takes ~3 minutes on an RTX 4060.

```bash
# GPU build of PyTorch first, otherwise ultralytics pulls in the CPU-only build
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-dev.txt

python download_dataset.py --check      # what is missing (the annotations are already on disk)
python download_dataset.py              # fetch missing annotation files (never past 10 GB in data/raw)
python -m src.gesturegrasp.train_v2     # build features, train on GPU, evaluate, save to models/
python scripts/export_mlp_numpy.py      # refresh models/mlp_weights.npz; checks it matches the new model
pytest -q
```

Training:
- writes `models/` and `reports/`,
- updates each gesture's `min_conf` in `configs/intents.yaml` (chosen on validation users),
- rebuilds `data/processed/hagrid_landmarks.parquet` if it is missing.

Copy `models/` and `configs/intents.yaml` to the Pi and restart the server. If you skip the export, the Pi
notices `mlp_weights.npz` no longer matches and falls back to the joblib files, which is slower to load.

### 5.4 Tune behaviour without retraining

Everything is in `configs/intents.yaml`. Restart the server after editing.

- `hold_time` per gesture: how long to hold before it fires (longer = fewer accidents, slower).
- `min_conf` per gesture: the confidence needed (rewritten by retraining).
- `intent_engine.cooldown_seconds`, `margin_threshold`, `ema_alpha`: pacing and smoothing.
- `state_machine.timeouts`: how long each step waits.
- `pointing.allowed_classes`, `max_angular_deviation_deg`: what can be pointed at, and how precisely.

---

## 6. Connecting a real robot later

The server drives `robot.simulated.SimulatedRobot`. A real robot gets a class that implements
`robot.base.BaseRobot`:
- `navigate_to`, `grasp`, `release`, `return_to_dock`, `emergency_stop`, `resume`, `reset`,
  `get_status`, `notify`;
- optionally `resolve_target`, to turn a selected object into a position the robot can drive to.

Pass it to `GesturePipeline(robot=...)` where the server builds its pipeline (`server/app.py`, in
`lifespan`). The state machine calls the robot methods once per step and waits for the robot to
report it is IDLE and holding the object before it asks the user for OK.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| No frames, `/camera` says `lost` | Test the camera from the Pi: `curl -s http://<esp32-ip>/status`. If that fails, check power and Wi-Fi on the ESP32's serial log. Then check the URL in `--camera-url` |
| `esp32cam.local` does not resolve | Use the IP (3.2). Check `avahi-daemon` and `libnss-mdns` are installed |
| Stream stalls or stays on `connecting` | Something else holds the ESP32 stream: it serves one client. Close any browser tab on `:81/stream` and stop `scripts/benchmark.py` |
| Drops out now and then, serial log shows `Brownout detector was triggered` | The power supply. Use a stable 5 V, at least 1 A, 2 A preferred, on the 5V pin, with short wires |
| `/camera` shows `mode: capture` | `/stream` keeps failing, so the Pi is polling `/capture` at a lower rate. Same checks as above |
| High latency, `loop.frame_age_ms` over 250 | Check the Pi is not throttling (`vcgencmd get_throttled` should print `0x0`), then see 2.6 |
| Low fps on the Pi | Run `scripts/benchmark.py` to see which stage is slow, then see 2.6. Low `fps` under `/camera` itself points to Wi-Fi: a low RSSI (under about -70 dBm) means move the camera or antenna closer |
| Wrong colours or a tint | White balance on the camera: `curl "http://<esp32-ip>/control?var=awb&val=1"`, and let it settle for a few seconds in steady light |
| Picture upside down or mirrored | `--flip v`, `--flip h` or `--flip both` |
| Hand not detected at a distance | Keep `--framesize vga` or larger; at QVGA a hand at 1–2 m is too small. Better light helps, and so does keeping the whole hand in view |
| Gestures fire too easily / too slowly | Raise / lower `hold_time` in `configs/intents.yaml` |
| Server: `Trained model files missing` | `models/` is incomplete: copy it from the repo or a training PC |
| `AttributeError: ... has no attribute 'label'` from MediaPipe | protobuf was upgraded. `pip install protobuf==4.25.9` |
| Log repeats `operator torchvision::nms does not exist` | torch and torchvision no longer match, and YOLO finds nothing. Reinstall both from `requirements-pi.txt` |
| Page says *test objects (no YOLO)* without `--mock-detector` | YOLO failed to load, so fixed fake objects are in use. Check the log, and that step 2.2 ran. A hotspot has no internet to download weights |
| `--speak` is silent | `sudo apt install -y espeak-ng`, and check the Pi's audio output. The Pi 5 has no audio jack, so use a USB or Bluetooth speaker |
| Page: "The browser blocked the camera" | The webcam tab needs `localhost` or HTTPS. On a phone, use the ESP32-CAM tab |
| Page: "Server not reachable" | The server is not running, or it runs on another port: use that port in the URL |
| Reset does nothing | The server has `--token`: open the page with `?token=<same secret>` |
| Robot frozen on the map | No frames are being processed (see 4.3). Check the camera badge |

---

## 8. Hardware smoke-test checklist

Everything above was tested without the physical boards, against `tests/fake_esp32.py`. These steps need the
real hardware, in this order:

1. **Flash:** follow [esp32_cam/README.md](esp32_cam/README.md). The firmware has not been compiled yet, so check
   that it builds on Arduino-ESP32 core 3.x. The serial log should show `Camera ready, PID 0x26` and an IP.
2. **Capture:** open `http://<esp32-ip>/capture` and `http://<esp32-ip>/status`. The image should be 640x480.
3. **From the Pi:** `curl -s http://<esp32-ip>/status`, then
   `curl -s --max-time 3 -o /dev/null -w "%{size_download}\n" http://<esp32-ip>:81/stream`. Expect a few hundred KB.
4. **Install:** `pip install` from 2.1 finishes on the Pi, and `python -c "import mediapipe; print(mediapipe.__version__)"`
   prints `0.10.14`.
5. **Export:** `python scripts/export_yolo_pi.py` prints a fastest backend, and `pytest -q` still passes after it.
6. **Benchmark:** `python scripts/benchmark.py --source http://<esp32-ip>:81/stream`. Record fps and frame age. The
   targets are at least 10 fps and under 250 ms.
7. **Server:** `python -m server --camera-url http://<esp32-ip>:81/stream`. `/camera` shows `streaming`.
8. **Page:** open `http://<pi-ip>:8000/` on a phone. Video plays and the badge shows fps and Wi-Fi signal.
9. **WAKE:** hold 🤙 for 0.6 s at 1–2 m. The page moves to *Listening*.
10. **STOP:** hold ✋. The page shows *Stopped* within about half a second. Press **Stop robot** as well.
11. **Recovery:** unplug the ESP32-CAM. The badge shows *lost* within a few seconds. Plug it back in: video
    returns by itself, with no restart of the server.
12. **Reboot:** with the service enabled, reboot the Pi and check the page comes back by itself.
