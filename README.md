# GestureGrasp

Hand-sign control for an assistive fetching robot, built for people who find speech or touch
controls hard: older adults, people with Parkinson's, people with missing fingers. Point at a bottle,
give a thumbs up, and the robot brings it to you.

A **Raspberry Pi** with a camera streams video to a **laptop server**, which runs the gesture model on
the GPU and sends back the system's state and a prompt for the user. The Pi prints the prompt and can
speak it aloud.

```
Pi camera ─► pi_client ──JPEG over WebSocket──► server: hands → model → intent → pointing/YOLO → state machine → robot
             speaks prompt ◄─────── JSON ───────
```

- **Playground:** `http://localhost:8000/playground` shows live, in words, what the model reads from
  your hand ("You're pointing at the bottle"), with every confidence and decision behind it.
- **[GUIDE.md](GUIDE.md):** setup and operation. Laptop server, Raspberry Pi client, how they talk,
  the model, retraining, troubleshooting.
- **[GESTURES.md](GESTURES.md):** the 7 supported hand gestures, what each does, and how reliable it is.
- **[DECISIONS.md](DECISIONS.md):** why things are built the way they are.

## Quick start

Laptop (project root):
```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU
pip install -r requirements.txt
python -m server                       # then open http://localhost:8000/playground
```

Raspberry Pi (copy the `pi_client/` folder over first):
```bash
sudo apt install -y python3-venv python3-picamera2 espeak-ng
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install -r requirements.txt
python client.py --server ws://<laptop-ip>:8000/ws/stream --speak
```

## Layout

```
server/        FastAPI server the Pi connects to (python -m server); static/playground.html is the demo page
pi_client/     Raspberry Pi camera client + autostart service file
src/gesturegrasp/
  pipeline.py           frame → decision: hands, model, intent, pointing, state machine, robot
  mediapipe_extractor.py, features.py, torch_mlp.py      hand landmarks → features → classifier
  intent_engine.py, state_machine.py, pointing.py, detector.py
  hagrid_landmarks.py, train_v2.py                        training data + training (GPU)
robot/         robot interface + simulated robot (real robot to come)
configs/intents.yaml   gestures, hold times, thresholds, states, objects
models/        trained model        reports/  evaluation results
download_dataset.py    fetch HaGRID training annotations
tests/         pytest -q
```

## Status and limits

- **Accuracy:** on people never seen in training, live-path macro-F1 is 0.979, STOP recall 99.0%,
  and users aged 60+ score 0.985. Details in GUIDE.md §5.2.
- **No clinical testing yet:** not yet tested with real elderly or Parkinson's users.
- **Simulated robot:** the robot is simulated until hardware is connected (GUIDE.md §6).
- **2D pointing:** objects in the same line of sight cannot be told apart, so every target needs a
  thumbs-up before the robot moves.
