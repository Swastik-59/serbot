---
description: Build GestureGrasp: a contactless hand-gesture intent system for SerBot using HaGRID, MediaPipe, ML classification, temporal intent logic, pointing, YOLO object detection, robot simulation, and a Streamlit demo.
---

# GestureGrasp Project Goal

## Goal
Build a complete contactless HRI prototype for SerBot:

Camera → Hand Detection → MediaPipe Landmarks → Gesture Classifier → Intent Engine → Pointing/Object Selection → YOLO Object Detection → User Confirmation → Robot Action → Feedback.

This is a complete system, not only a gesture classifier.

## Intent Mapping
Use these 7 actionable intents:

- call → WAKE: start interaction
- point → SELECT_TARGET: select pointed object
- like → CONFIRM: approve proposed target
- dislike → REJECT: reject target
- stop → STOP: immediately stop
- peace → RETURN_HOME: abort and return to dock
- ok → ACKNOWLEDGE: finish interaction

Also train `no_gesture` and `fist` as non-actionable rejection/hard-negative classes.

`point` means SELECT_TARGET, not FETCH.

Store all mappings in `configs/intents.yaml`; do not duplicate them in Python.

## State Machine
States:
`IDLE, LISTENING, TARGET_PROPOSED, EXECUTING, AWAIT_FEEDBACK, RETURNING, STOPPED`

Transitions:
- IDLE --WAKE--> LISTENING
- LISTENING --SELECT_TARGET + valid object--> TARGET_PROPOSED
- TARGET_PROPOSED --CONFIRM--> EXECUTING
- TARGET_PROPOSED --REJECT--> LISTENING
- EXECUTING --complete--> AWAIT_FEEDBACK
- AWAIT_FEEDBACK --ACKNOWLEDGE--> IDLE
- active --RETURN_HOME--> RETURNING-->IDLE
- any active state --STOP--> STOPPED
- STOPPED --WAKE--> LISTENING

STOP always has priority.

## Intent Engine
Classifier outputs probabilities; a deterministic IntentEngine converts them to commands.

Use:
- EMA smoothing, alpha 0.6
- configurable confidence threshold
- top1 vs runner-up margin >= 0.20
- hold-to-fire
- max 2 dropped/no-hand frames
- 1 sec cooldown
- state gating
- state timeouts

Default hold times:
WAKE .60s, SELECT_TARGET .80s, CONFIRM .70s, REJECT .70s, STOP .25s, RETURN_HOME 1.0s, ACKNOWLEDGE .80s.

Never execute a robot command from one noisy frame.

## Dataset
Do NOT use Kaggle. Use official HaGRID direct archives.

Required:
`call, dislike, fist, like, ok, peace, stop, point, no_gesture, annotations`

Do NOT download the full dataset.

Create `download_dataset.py` with:
- direct URLs
- `--check` and `--redownload`
- streaming/tqdm
- retry/resume
- extraction
- validation
- `reports/download_manifest.json`
- clear failure messages

Use the official HaGRID downloader URLs supplied in the execution prompt. Do not invent URLs.

## Sampling
Default:
- actionable classes: max 3000 each
- no_gesture: max 5000
- fist: max 2000
- max 10 samples per user/class
Prioritize user diversity.

## Landmark Pipeline
Use MediaPipe Hands for training AND live inference.

Settings:
`static_image_mode=True`, `max_num_hands=2`, `min_detection_confidence=0.5`

Resize long side to ~640px. Process full frame. Use annotations only to associate the correct detected hand.

Save 21x3 landmarks plus:
`sample_id, gesture, intent, user_id, handedness, split, image size, source`

Output: `data/processed/landmarks.parquet`

Report detection rates overall and per class.

## Features
Normalize landmarks using:
- wrist as origin
- wrist-to-middle-MCP scale
- canonical left/right mirroring

Do not rotate-normalize because orientation matters.

Add:
- finger flexion angles
- fingertip/wrist distances
- fingertip distances
- palm dimensions
- finger extension ratios
- palm orientation

Use `FEATURE_VERSION="1.0"` and save it in model metadata.

## Augmentation
Training data only:
- small Gaussian landmark noise
- scale jitter
- small rotation
- translation noise
- single-joint dropout

Never change gesture semantics.

## Training
Compare:
`LogisticRegression, SVC(probability=True), RandomForestClassifier, HistGradientBoostingClassifier, MLPClassifier`

Optional XGBoost/LightGBM if useful.

Use subject-independent splits. Prefer official splits; otherwise GroupShuffleSplit. Use GroupKFold for model selection with `user_id`.

Select using grouped validation macro-F1, not training accuracy.

Save:
`models/gesture_model.joblib`
`models/calibrated_model.joblib`
`models/scaler.joblib`
`models/metadata.json`

Never fabricate metrics.

## Evaluation
Generate:
- accuracy
- macro precision/recall/F1
- weighted F1
- per-class metrics
- confusion matrix
- calibration/ECE/Brier score
- threshold sweep
- NO_GESTURE false acceptance
- STOP recall
- false triggers/minute
- inference latency

Save under `reports/`.

Sweep thresholds 0.50–0.95 and update intent thresholds only from measured validation results.

## Pointing + Object Detection
For `point`, use MediaPipe landmark 5 (index MCP) → 8 (index tip) to create a 2D image-plane pointing ray.

Use Ultralytics lightweight YOLO. Default `yolo11n.pt`; if unsupported, use a supported nano model and record its name.

Default allow-list:
`bottle, cup, cell phone, remote, book, mouse, apple, banana`

Target logic:
1. test ray/box intersection
2. prioritize direct hits
3. otherwise choose smallest angular deviation
4. reject if deviation > 15°
5. target must remain stable during the SELECT hold
6. reset if target changes
7. no target = no action

Document 2D depth ambiguity.

## Robot
Create:
`robot/base.py`
`robot/simulated.py`
`robot/serbot_stub.py`

Interface:
`navigate_to, grasp, return_to_dock, emergency_stop, notify, get_status`

Simulated robot must animate navigation/grasp/return and support STOP.

Never invent SerBot SDK calls.

## Own Data
Create a collector for real SerBot-camera data:
`subject_id, gesture, distance, lighting, timestamp, landmarks`

Save in `data/own/`.

Later compare HaGRID-only vs HaGRID+own data on held-out own subjects. Own data is not required for first training.

## Robustness
Test:
- low light
- overexposure
- blur
- downscaling/compression
- reduced hand size
- landmark noise
- temporal jitter
- simulated tremor-like jitter

Do not call simulated tremor real-user validation.

## Streamlit
Create `app/streamlit_app.py` with:

1. Live Demo
2. Scenario Simulator
3. Intent Table
4. Model Report
5. Dataset Explorer
6. System Architecture
7. Experiments

Live Demo must show:
camera, landmarks, gesture/confidence, top-3, intent, hold progress, FSM state, detected objects, pointing ray, selected target, robot state, FPS/latency, event log, large EMERGENCY STOP.

Scenario Simulator must work without a camera and include:
WAKE, POINT→BOTTLE, POINT→CUP, CONFIRM, REJECT, STOP, RETURN HOME, ACKNOWLEDGE, and `RUN COMPLETE DEMO`.

Complete demo:
WAKE → POINT BOTTLE → CONFIRM → NAVIGATE → GRASP → RETURN → ACKNOWLEDGE

Also include reject, STOP, timeout, ambiguous-target, no-object and low-confidence scenarios.

## Core API
Create:
`GesturePipeline.process(frame)`

Return:
gesture, confidence, probabilities, intent, state, hold progress, objects, selected target, pointing ray, robot status, latency.

Core pipeline must not depend on Streamlit.

## Master Pipeline
User workflow:
1. `python download_dataset.py`
2. `python main.py --all`
3. `streamlit run app/streamlit_app.py`

`main.py --all` runs:
environment check → dataset inspection → sampling → landmark extraction → features → augmentation → grouped split → training → CV/model selection → calibration → thresholds → evaluation → robustness → pointing tests → robot tests → model export → reports → pytest.

Also support:
`--prepare --extract --train --evaluate --robustness --report --demo --help`

Use caching/checkpoints so expensive work resumes.

## Tests
Create pytest tests for:
features, mirroring, intent hold/cooldown/state gating, STOP priority, timeouts, pointing geometry, target consistency, robot simulation and config validation.

Run `pytest -q`.

## Quality
Initial targets:
- macro F1 >= 0.85
- STOP recall >= 0.90 where achievable

Never manipulate metrics. If below target, inspect confusion, imbalance, MediaPipe failures and user distribution, make a documented change, retrain and report before/after results.

## Documentation
Maintain:
`README.md`
`PROGRESS.md`
`DECISIONS.md`
`reports/REPORT.md`

Document architecture, commands, dataset, training, evaluation, Streamlit, camera setup, own-data collection, SerBot integration and limitations.

Do not claim elderly/tremor validation without real participants. Document 2D pointing, domain gap, camera-distance limits, MediaPipe failures, simulated robot limits and missing SerBot SDK.

## Engineering Rules
- inspect before editing
- preserve unrelated user work
- use pathlib, logging and seed 42
- configuration-driven behavior
- verify outputs
- never fabricate metrics or SDK calls
- never silently ignore failures
- never download the full HaGRID dataset
- never use Kaggle
- diagnose → fix → rerun → verify

## Definition of Done
`download_dataset.py` downloads/extracts required data.
`python main.py --all` produces verified model/evaluation artifacts.
`streamlit run app/streamlit_app.py` launches the demo.
`pytest -q` passes.