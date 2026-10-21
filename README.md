<div align="center">

# 🤙 GestureGrasp

### Hand-sign control for an assistive fetching robot

**Point at a bottle, give a thumbs up, and the robot brings it to you.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Hands-0097A7?logo=google&logoColor=white">
  <img alt="YOLO11" src="https://img.shields.io/badge/YOLO11n-Ultralytics-111F68">
  <img alt="Raspberry Pi" src="https://img.shields.io/badge/Raspberry%20Pi-4%20%7C%205-A22846?logo=raspberrypi&logoColor=white">
</p>

<p>
  <img alt="Macro-F1" src="https://img.shields.io/badge/live%20macro--F1-0.979-2ea44f">
  <img alt="STOP recall" src="https://img.shields.io/badge/STOP%20recall-99%25-2ea44f">
  <img alt="Users 60+" src="https://img.shields.io/badge/users%2060%2B%20F1-0.985-2ea44f">
  <img alt="False commands" src="https://img.shields.io/badge/false%20commands-0.04%2Fmin-2ea44f">
</p>

[Quick start](#-quick-start) · [Gestures](#-the-gestures) · [Playground](#-the-playground) · [How it works](#-how-it-works) · [Results](#-results) · [Hardware](#-hardware) · [Docs](#-documentation)

</div>

---

## 💡 Why

Speech and touch controls are hard for many of the people who would benefit most from a helper robot:
older adults, people with Parkinson's, people with missing fingers or stiff joints. GestureGrasp lets
them ask for an object with a few simple, forgiving hand signs. No wearable, no app, no voice.

- 🖐️ **Forgiving by design:** small tremor is fine, either hand works, gestures can be tilted, and an
  open palm with spread fingers still counts as STOP.
- 🛑 **Safety first:** STOP has the shortest hold, always wins, and `POST /stop` works without a token
  so anyone in the room can halt the robot.
- 🙅 **Ignores everyday hands:** fists, resting hands, holding a cup and other signs are trained as
  "not a command".
- ✅ **Nothing moves without confirmation:** every target is proposed first and needs a thumbs up.

---

## 🧭 How it works

A **Raspberry Pi** with a camera streams video to a **laptop server**. The laptop runs the gesture model
on the GPU and sends back the system's state and a prompt, which the Pi prints and can speak aloud.
The Pi does no AI work, so any Pi 4 or Pi 5 is enough.

<p align="center">
  <img src="architecture.png" alt="GestureGrasp system architecture" width="720">
</p>

```
Pi camera ─► pi_client ──JPEG over WebSocket──► server: hands → model → intent → pointing/YOLO → state machine → robot
             speaks prompt ◄─────── JSON ───────
```

| Stage | What it does |
|---|---|
| **Hand landmarks** | MediaPipe Hands finds 21 points per hand |
| **Gesture classifier** | 126 features (positions, bend angles, fingertip distances, roll) → GPU-trained MLP (126 → 512 → 256 → 128 → 9), temperature-calibrated, ~0.3 ms per hand |
| **Intent engine** | Smooths confidence over frames, needs a full hold before a command fires, STOP has priority |
| **Pointing + YOLO** | A ray along the index finger picks the nearest allowed object within 15°, voted over the hold |
| **State machine** | `IDLE → LISTENING → TARGET_PROPOSED → EXECUTING → AWAIT_FEEDBACK`, plus `RETURNING` and `STOPPED`. Only accepts commands that make sense in the current step |
| **Robot** | `BaseRobot` interface. A simulated robot drives, grasps and delivers until real hardware is connected |

Every gesture, hold time, threshold, state and object lives in one file:
[`configs/intents.yaml`](configs/intents.yaml).

---

## ✋ The gestures

Hold a gesture steady for its hold time and it fires once.

| | Gesture | Command | Hold | What happens |
|:-:|---|---|:-:|---|
| 🤙 | **Call me** | `WAKE` | 0.6 s | Wakes the system; it starts listening |
| 👉 | **Point** | `SELECT_TARGET` | 0.8 s | Proposes the object you point at. Nothing moves yet |
| 👍 | **Thumbs up** | `CONFIRM` | 0.7 s | The robot fetches the proposed object |
| 👎 | **Thumbs down** | `REJECT` | 0.7 s | Cancels the proposal (or a wrong delivery) so you can point again |
| ✋ | **Stop** | `STOP` | 0.25 s | Emergency stop, at any time |
| ✌️ | **Peace** | `RETURN_HOME` | 1.0 s | Cancels the task and sends the robot to its dock |
| 👌 | **OK** | `ACKNOWLEDGE` | 0.8 s | You have the object: the robot lets go and goes home |

**A complete fetch:**

```
🤙 Call me      →  "Point at the object you want."
👉 Point        →  "Thumbs up to fetch the bottle, thumbs down to choose again."
👍 Thumbs up    →  "Fetching the bottle."   (robot drives, grasps, brings it to you)
                   "Here is the bottle. Show OK when done, thumbs down if it is wrong."
👌 OK           →  robot hands it over and returns to the dock
```

**Objects you can point at:** `bottle`, `cup`, `cell phone`, `remote`, `book`, `mouse`, `apple`, `banana`.

Full details, reliability per gesture and tips for users: **[GESTURES.md](GESTURES.md)**.

---

## 🚀 Quick start

### Laptop (server)

Tested on Windows 11, Python 3.11, NVIDIA RTX 4060.

```bash
# 1. GPU build of PyTorch first (skip on machines without an NVIDIA GPU)
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130

# 2. Everything else
pip install -r requirements.txt

# 3. Start the server
python -m server
```

Then open **http://localhost:8000/playground**. The first start downloads the YOLO weights
(`yolo11n.pt`, ~5 MB) once.

<details>
<summary><b>Server options</b></summary>

```bash
python -m server --token mysecret      # require a shared secret from the Pi (recommended)
python -m server --mock-detector       # fixed fake objects instead of YOLO (no real objects needed)
python -m server --port 9000           # another port
```

</details>

### Raspberry Pi (camera client)

Copy the `pi_client/` folder to the Pi, then:

```bash
sudo apt install -y python3-venv python3-picamera2 espeak-ng
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install -r requirements.txt
python client.py --server ws://<laptop-ip>:8000/ws/stream --speak
```

```
>>> [IDLE] Show the call-me sign to start.
call        1.00 hold 60%  | 54 ms
>>> [LISTENING] Point at the object you want.
```

To start the client on boot, use the systemd unit in
[`pi_client/gesturegrasp-client.service`](pi_client/gesturegrasp-client.service) (see GUIDE.md §3.5).

> **No Pi?** The playground can use the laptop's webcam, so you can try the whole system on the laptop alone.

---

## 🎮 The playground

`http://localhost:8000/playground` is an interactive demo that runs every frame through the same model
and decision logic the Pi uses. It explains, in plain words, what the model reads from your hand:
*"You're pointing at the **bottle**."*

| Panel | Shows |
|---|---|
| **The sentence** | A plain-language reading of your hand, a hold bar and a "Sent ✓" badge when a command fires |
| **What the model sees** | Confidence for all 9 classes, with the threshold each gesture needs |
| **Decision** | Confidence vs. required, lead over the runner-up, hold time, and why it did or didn't fire |
| **Pointing & objects** | The target, how far off-centre, how steady, and every detected object |
| **Speed** | Server time, round trip, frames per second |
| **The robot** | Task step by step, a top-down map of the robot moving, and the command log |

Sources: **your webcam**, **the Raspberry Pi camera**, or **an uploaded photo**.

---

## 🔌 API

| Endpoint | Purpose |
|---|---|
| `WS /ws/stream` | Pi sends JPEG frames, server replies with one JSON result per frame (lock-step, no stale frames) |
| `POST /stop` | 🛑 Emergency stop. Never needs a token |
| `POST /reset` | Back to IDLE, robot docked |
| `POST /infer` | One JPEG in, the same JSON out (handy with `curl`) |
| `GET /health` · `/state` · `/config` · `/latest` | Model/device info, current state, configuration, last result |
| `GET /` · `/video` · `/playground` | Browser viewer with a STOP button, MJPEG stream, demo page |

```bash
curl http://localhost:8000/health
curl -X POST --data-binary @hand.jpg http://localhost:8000/infer
curl -X POST http://localhost:8000/stop
```

The full message format is in [GUIDE.md §4](GUIDE.md#4-how-the-pi-and-the-server-talk).

---

## 📊 Results

Trained on **435,939 hands from 40,159 people** ([HaGRID v2](https://github.com/hukenovs/hagrid)),
including open palms, everyday poses and random rotations. All numbers below are on **people held
out of training** (`reports/evaluation_report.json`).

| Measure | Result |
|---|:-:|
| Live path (our MediaPipe on real photos), macro-F1 | **0.979** |
| STOP recall | **99.0%** |
| Users aged 60+ (2,209 hands), macro-F1 | **0.985** |
| Hands at random rotations, macro-F1 | **0.988** |
| Commands that fire within 2 s under tremor-like jitter | **95–99.8%** |
| False commands from idle / everyday hands | **0.04 per minute** |

<details>
<summary><b>Model selection</b></summary>

| Model | Val macro-F1 (annotations) | Val macro-F1 (live) | Device |
|---|:-:|:-:|---|
| Logistic Regression | 0.977 | 0.927 | CPU |
| HistGradientBoosting | 0.984 | 0.957 | CPU |
| **TorchMLP** ✅ | **0.987** | **0.975** | RTX 4060 |

</details>

### Retrain

Only needed when you change gestures, thresholds or training data (~3 minutes on an RTX 4060):

```bash
python download_dataset.py --check      # see what is missing
python download_dataset.py              # fetch annotations (capped at 10 GB)
python -m src.gesturegrasp.train_v2     # build features, train on GPU, evaluate, save to models/
pytest -q                               # run the tests
```

---

## 🔧 Hardware

<p align="center">
  <img src="pin_diagram.png" alt="GestureGrasp hardware pin diagram" width="720">
</p>

An ESP32-CAM or Pi camera feeds the Raspberry Pi, the Pi talks to the laptop over Wi-Fi, and the
SerBot robot receives commands through the `BaseRobot` interface. To plug in a real robot, implement
`robot.base.BaseRobot` and pass it to `GesturePipeline(robot=...)` in `server/app.py`
(GUIDE.md §6).

---

## 📁 Project layout

```
server/                 FastAPI server (python -m server); static/playground.html is the demo page
pi_client/              Raspberry Pi camera client + systemd autostart unit
src/gesturegrasp/
  pipeline.py           frame → decision: hands, model, intent, pointing, state machine, robot
  mediapipe_extractor.py, features.py, torch_mlp.py     landmarks → features → classifier
  intent_engine.py, state_machine.py, pointing.py, detector.py
  hagrid_landmarks.py, train_v2.py                       training data + GPU training
robot/                  robot interface + simulated robot
configs/intents.yaml    gestures, hold times, thresholds, states, objects
models/                 trained model (calibrated MLP, scaler, metadata)
reports/                evaluation results
download_dataset.py     fetches HaGRID training annotations
tests/                  pytest -q
```

---

## 📚 Documentation

| Doc | What's inside |
|---|---|
| **[GUIDE.md](GUIDE.md)** | Full setup and operation: laptop server, Raspberry Pi client, protocol, model, retraining, tuning, troubleshooting |
| **[GESTURES.md](GESTURES.md)** | The 7 gestures, what each does, reliability, timeouts, tips for users |
| **[DECISIONS.md](DECISIONS.md)** | Why things are built the way they are |

---

## 🚧 Status and limits

- 🧪 **No clinical testing yet.** Not yet tested with real elderly or Parkinson's users. Simulated
  tremor is a stress test, not clinical evidence.
- 🤖 **Simulated robot.** The robot is simulated until the SerBot hardware link exists.
- 📐 **2D pointing.** Objects in the same line of sight can't be told apart (the nearer one is
  proposed), which is why every target needs a thumbs up before the robot moves.
- 📷 **Lighting and distance matter.** Best with the whole hand in view, 0.5–2 m from the camera.

---

<div align="center">

Built with MediaPipe, PyTorch, Ultralytics YOLO and FastAPI · Trained on HaGRID v2

</div>
