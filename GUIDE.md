# GestureGrasp guide: Raspberry Pi camera + laptop server + model

This guide takes you from nothing to a working system. A Raspberry Pi with a camera streams video to a
laptop, the laptop runs the gesture model, and the result (a spoken prompt and the robot's state) comes
back to the Pi. For the gestures themselves, see [GESTURES.md](GESTURES.md).

```
 Raspberry Pi                                   Laptop (NVIDIA GPU)
 ────────────                                   ───────────────────
 camera ─► pi_client/client.py  ──JPEG frames──►  server (FastAPI, port 8000)
           prints / speaks prompt ◄──JSON result──   MediaPipe hands → gesture model (GPU-trained MLP)
                                                     → IntentEngine (hold, smoothing) → pointing + YOLO
                                                     → state machine → robot (simulated for now)
```

The Pi does no AI work. It only captures, compresses and sends frames, so any Pi 4 or Pi 5 is enough.

**Just want to see it work?** No Pi needed. On the laptop:

```bash
pip install -r requirements.txt     # (GPU users: install PyTorch first, see 2.1)
python -m server
```

Then open **http://localhost:8000/playground**, click **Turn on webcam** and show your hand (2.6).

---

## 1. What is where

| Path | What it is |
|---|---|
| `server/` | FastAPI server: `python -m server`. WebSocket for the Pi, REST endpoints, browser viewer |
| `server/static/playground.html` | The playground demo page (2.6), a single self-contained file |
| `pi_client/` | Everything the Pi needs: `client.py`, `requirements.txt`, autostart service file |
| `src/gesturegrasp/pipeline.py` | One frame in, one decision out: hands → model → intent → pointing → state machine → robot |
| `src/gesturegrasp/` (others) | Hand tracking, features, intent engine, state machine, pointing, YOLO, training |
| `models/` | The trained model: `calibrated_model.joblib`, `scaler.joblib`, `metadata.json` |
| `configs/intents.yaml` | Single source of truth: gestures, hold times, thresholds, states, objects |
| `robot/` | Robot interface (`base.py`) and the simulated robot used until real hardware exists |
| `download_dataset.py` | Fetches HaGRID training annotations (only needed to retrain) |
| `reports/` | Latest evaluation numbers from training |
| `tests/` | `pytest -q` |

---

## 2. Laptop: install and start the server

Tested on Windows 11, Python 3.11, RTX 4060. Run everything from the project root.

### 2.1 Install

```bash
# 1. GPU build of PyTorch FIRST (otherwise ultralytics pulls in the CPU-only build)
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"     # must print True

# 2. Everything else
pip install -r requirements.txt
```

No NVIDIA GPU? Skip step 1. `pip install -r requirements.txt` then installs CPU PyTorch, and the
server still works, just slower (YOLO takes ~60 ms per frame instead of ~14 ms).

### 2.2 Find the laptop's IP address

```powershell
ipconfig        # look for "IPv4 Address" on your Wi-Fi adapter, e.g. 192.168.1.21
```

The Pi and the laptop must be on the same network. Give the laptop a fixed IP (a DHCP reservation
in your router) so the Pi's setting never goes stale.

### 2.3 Open port 8000 in Windows Firewall

Windows asks the first time the server starts; choose **Private networks**. Or do it once from an
**administrator** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "GestureGrasp server" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

### 2.4 Start the server

```bash
python -m server                       # listens on 0.0.0.0:8000, real YOLO on the GPU
python -m server --token mysecret      # require a shared secret from the Pi (recommended)
python -m server --mock-detector       # fixed fake objects instead of YOLO (testing without objects)
python -m server --port 9000           # another port
```

The first start downloads the YOLO weights `yolo11n.pt` (~5 MB) once. When it is ready you will see:

```
Playground: http://localhost:8000/playground  |  Pi connects to ws://<this-laptop-ip>:8000/ws/stream
Server ready.
Uvicorn running on http://0.0.0.0:8000
```

Stop it with Ctrl+C.

### 2.5 Check it

```bash
curl http://localhost:8000/health
```
Expect: `"classifier":"TorchMLPClassifier"`, `"detector_device":"cuda:0"`, `"feature_version":"2.0"`.

Open **http://localhost:8000/playground** to try the model (see 2.6), or **http://localhost:8000/**
for the plain view of the Pi's annotated stream with an **EMERGENCY STOP** button.

### 2.6 The playground: see what the model can do

**http://localhost:8000/playground** is an interactive demo page served by the same server. Every
frame goes through the same model and decision logic the Pi uses.

| Source | What it does |
|---|---|
| **Your webcam** | Streams the laptop's webcam live. Point at a bottle and the page writes "You're pointing at the **bottle**." |
| **Raspberry Pi camera** | Watches whatever the Pi is streaming, with the same panels |
| **A photo** | Analyses one uploaded image (shows what the model reads; no commands, since nothing is held) |

What it shows:
- **The sentence:** a plain-language reading of your hand, with a bar that fills while you hold
  the gesture and a "Sent ✓" badge when a command fires.
- **What the model sees:** confidence for all 9 classes, each with a tick at the confidence that
  gesture needs.
- **Decision:** confidence vs. required, lead over the runner-up gesture, hold time, the command,
  and in plain words why it did or didn't fire.
- **Pointing & objects:** the object being pointed at, how far off-centre, how steadily, and every
  object detected.
- **Speed:** server processing time, round trip, frames per second.
- **The robot:** its task step by step, a top-down map of it moving, and the log of commands sent.
- **On the video:** the hand skeleton, the pointing ray and object boxes, with the target highlighted.

Try this:

1. Show 🤙 **call me** until the bar fills. The step track moves from *Asleep* to *Listening*.
2. 👉 **Point** at a bottle, cup or book. The sentence names it, and the ray and box turn amber and teal.
3. 👍 **Thumbs up**. The map shows the robot driving out, picking it up and bringing it to *you*.
4. 👌 **OK** once it says *Delivered*. The robot hands it over and returns to the dock.

Any time: ✋ **stop**, or press **Stop robot**.

Notes:
- **Webcam access:** browsers only allow the webcam on `localhost` (or HTTPS). Use webcam mode on the
  laptop itself; other devices on the network can use the Pi and photo modes.
- **No objects handy:** with `python -m server --mock-detector`, three fixed test objects (bottle,
  cup, book) stand in for YOLO, so you can try pointing without real objects. The chip at the top
  says *test objects (no YOLO)* when this is on.
- **Shared session:** the playground drives the same single session as the Pi, so if the Pi is
  streaming too, both feed one robot. **Start over** resets it and **Stop robot** stops it.
- **Photo mode** starts over before each photo, so the photo is judged on its own.
- **Token:** if the server was started with `--token`, open the page as
  `http://localhost:8000/playground?token=mysecret`.
- **Fonts:** the page loads them from Google Fonts. Offline, it falls back to system fonts and works
  the same.

---

## 3. Raspberry Pi: set up the camera client

Tested target: Raspberry Pi 4 or 5, Raspberry Pi OS Bookworm (64-bit), Pi Camera Module or a USB webcam.

### 3.1 Prepare the Pi

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2     # Pi camera support (skip picamera2 for USB-only)
sudo apt install -y espeak-ng                          # optional: speak prompts aloud
rpicam-hello -t 2000                                   # Pi camera only: a 2 s preview proves the camera works
```

### 3.2 Copy the client to the Pi

Only the `pi_client/` folder is needed. From the laptop (replace `pi@raspberrypi.local`):

```bash
scp -r pi_client pi@raspberrypi.local:~/
```

### 3.3 Install the Python packages

```bash
cd ~/pi_client
python3 -m venv --system-site-packages venv    # --system-site-packages lets the venv see apt's picamera2
source venv/bin/activate
pip install -r requirements.txt
```

### 3.4 Run it

First check the Pi can reach the server:
```bash
curl http://192.168.1.21:8000/health
```

Then start streaming:
```bash
python client.py --server ws://192.168.1.21:8000/ws/stream                    # Pi camera
python client.py --server ws://192.168.1.21:8000/ws/stream --source 0         # USB webcam
python client.py --server ws://192.168.1.21:8000/ws/stream --speak            # say prompts aloud
python client.py --server ws://192.168.1.21:8000/ws/stream --token mysecret   # if the server uses --token
```

What you see:
```
>>> [IDLE] Show the call-me sign to start.
call        1.00 hold 60%  | 54 ms
>>> [LISTENING] Point at the object you want.
```
The `>>>` lines are the prompts (spoken with `--speak`). The live line shows the recognised gesture,
its confidence, hold progress, the object being pointed at, and the round-trip time.

| Option | Default | Meaning |
|---|---|---|
| `--server` | (required) | `ws://<laptop-ip>:8000/ws/stream` |
| `--source` | `picamera` | `picamera`, a camera index (`0`), a video file, or an image file |
| `--token` | none | Shared secret, if the server was started with `--token` |
| `--width` / `--height` | 640 / 480 | Capture size; frames wider than `--width` are scaled down |
| `--fps` | 15 | Upper limit on frames per second |
| `--quality` | 80 | JPEG quality (lower = less Wi-Fi traffic) |
| `--speak` | off | Speak each new prompt with espeak-ng |
| `--max-frames` | 0 | Stop after N frames (0 = run forever) |

### 3.5 Start automatically when the Pi boots

1. Edit `~/pi_client/gesturegrasp-client.service`: the user name, the paths, the laptop IP, and the
   options on the `ExecStart` line.
2. Install and enable it:
   ```bash
   sudo cp ~/pi_client/gesturegrasp-client.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now gesturegrasp-client
   journalctl -u gesturegrasp-client -f      # watch it run (Ctrl+C to leave)
   ```
It restarts by itself if it stops, and the client reconnects by itself if the laptop restarts.

---

## 4. How the Pi and the server talk

### 4.1 The camera stream (WebSocket)

`ws://<laptop>:8000/ws/stream` (add `?token=...` if the server has one).

- **Pi → server:** one binary message = one JPEG frame.
- **Server → Pi:** one JSON message per frame.
- **Lock-step:** the Pi sends the next frame only after the reply arrives. If Wi-Fi or the model
  slows down, frames are skipped rather than queued, so a command never fires from a stale frame.

Reply example (shortened; `...` marks trimmed lists):

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
| `landmarks`, `frame_size` | Hand points as fractions of the frame, and the frame's width/height in pixels |
| `objects` | Allowed objects YOLO found, with boxes in frame pixels |
| `robot` | Robot status: state, position, what it holds, whether it is stopped |
| `events`, `commands` | Last 5 state changes, and the last 20 commands that fired (also kept between frames) |
| `latency_ms` | Server processing time for this frame |

A frame that is not a valid image gets `{"error": "undecodable frame"}` and the stream continues.

**The system moves forward on frames.** The state machine, timeouts and simulated robot advance each
time a frame arrives, with or without a hand in it. While the Pi streams, that is continuous. If no
client sends frames, the robot pauses where it is until frames resume.

### 4.2 HTTP endpoints

| Endpoint | Token? | Purpose |
|---|---|---|
| `GET /health` | no | Loaded model, detector and device |
| `GET /state` | no | Current state, prompt, robot status, recent events |
| `POST /stop` | **never** | Emergency stop, works in every state. Anyone in the room must be able to stop the robot |
| `POST /reset` | yes | Back to IDLE, robot docked |
| `POST /infer` | yes | Body = one JPEG; returns the same JSON as the stream (for testing with curl) |
| `GET /playground` | no | Interactive demo page (2.6) |
| `GET /config` | no | Gestures, thresholds, states, robot map and model scores (used by the playground) |
| `GET /latest` | no | The last frame's full result, whoever sent it |
| `GET /` | no | Browser viewer with the annotated stream and a STOP button |
| `GET /video` | no | MJPEG stream of annotated frames (used by the viewer) |

```bash
curl -X POST --data-binary @hand.jpg http://192.168.1.21:8000/infer
curl -X POST http://192.168.1.21:8000/stop
```

The server runs one shared session: one camera, one robot, one state machine.

---

## 5. The model

### 5.1 What it is

- **Input:** 21 hand landmarks from MediaPipe Hands (x, y), turned into 126 features: positions,
  finger bend angles, fingertip distances, finger extension, hand roll. No handedness needed.
- **Classifier:** a neural network (126 → 512 → 256 → 128 → 9) trained on the GPU,
  temperature-calibrated so its confidence matches reality. ~0.3 ms per hand on the CPU.
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

Only needed if you change the gestures, thresholds or training data. Takes ~3 minutes on the RTX 4060.

```bash
python download_dataset.py --check      # what is missing (the annotations are already on disk)
python download_dataset.py              # fetch missing annotation files (never past 10 GB in data/raw)
python -m src.gesturegrasp.train_v2     # build features, train on GPU, evaluate, save to models/
pytest -q                               # 56 tests
```

Training:
- writes `models/` and `reports/`,
- updates each gesture's `min_conf` in `configs/intents.yaml` (chosen on validation users),
- rebuilds `data/processed/hagrid_landmarks.parquet` if it is missing.

Restart the server afterwards so it loads the new model.

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
| Pi: `Connection refused` / timeout | Server not running, wrong IP, or the firewall rule is missing (2.3). Test with `curl http://<ip>:8000/health` from the Pi |
| Pi: connects then closes at once (code 1008) | Server has `--token`; start the client with the same `--token` |
| Pi: `Cannot open camera/video source` | Pi camera: install `python3-picamera2` and use the `--system-site-packages` venv. USB: try `--source 0` or `1` |
| Colours look wrong (blue skin) in the viewer | Use the default `--source picamera`; it already outputs the colour order OpenCV expects |
| `/health` shows `detector_device: cpu` | CUDA PyTorch is not installed: redo step 2.1 part 1 and check it prints `True` |
| Server: `Trained model files missing` | `models/` is incomplete: retrain (5.3) |
| Gestures fire too easily / too slowly | Raise / lower `hold_time` in `configs/intents.yaml` |
| Hand not detected | Better light, hand 0.5–2 m from the camera, whole hand in view |
| Slow (> 150 ms per frame) | Lower `--quality` or `--width` on the Pi, use 5 GHz Wi-Fi, check the GPU is used |
| `--speak` is silent | `sudo apt install -y espeak-ng`, and check the Pi's audio output (`raspi-config`) |
| Playground: "The browser blocked the camera" | Allow the camera for the page (camera icon in the address bar), and open it as `http://localhost:8000/playground`, not via the laptop's IP |
| Playground: "Server not reachable" | The server is not running, or it runs on another port: use that port in the URL |
| Playground: "Wrong or missing token" | Add `?token=<same secret>` to the page URL |
| Playground Pi mode: "No frames yet" | Start `client.py` on the Pi; the page only watches, it does not start the Pi |
| Playground: no objects found | Real YOLO only knows the objects in `pointing.allowed_classes`; put one in view, or start the server with `--mock-detector` |
| Robot frozen on the map | Nothing is sending frames (see 4.1): turn the webcam on or start the Pi client |
