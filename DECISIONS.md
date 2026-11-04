# Architecture & Engineering Decisions (GestureGrasp)

> This is a history log. Entries are kept as written; where one names code that DEC-017 removed
> (the Streamlit app, the scenario simulator, feature set 1.0, the image downloader, `main.py`),
> or that DEC-018 replaced (`pi_client/`, the camera module on the Pi, a second machine running the
> models), it records why that existed at the time.

## DEC-001: Execution Environment & Python Stack
- **Context:** System is Windows with Python 3.11.9.
- **Packages:** Installed `opencv-python`, `mediapipe` (1.0.1), `ultralytics` (8.4.164), `streamlit` (1.64.0), `pyarrow` (25.0.1), `scikit-learn` (1.6.1), `torch` (2.14.0), `pandas` (3.0.5), `numpy` (2.2.2).
- **Decision:** Use standard scikit-learn algorithms with calibrated probabilities, OpenCV for frame processing, MediaPipe Hands for 21-landmark 3D coordinate detection, and Ultralytics YOLO (`yolo11n.pt`) for real-time lightweight object detection.

## DEC-002: Intent & State Machine Centralization
- **Context:** Specification strictly forbids duplicating intent definitions in Python code and mandates single-source-of-truth configuration.
- **Decision:** All 7 actionable intents (`WAKE`, `SELECT_TARGET`, `CONFIRM`, `REJECT`, `STOP`, `RETURN_HOME`, `ACKNOWLEDGE`) plus 2 negative classes (`no_gesture`, `fist`), hold durations, confidence margins, cooldowns, and allowable state transitions are defined in `configs/intents.yaml`. The python runtime loads and validates this file at startup.

## DEC-003: HaGRID Dataset Management & Storage Constraints
- **Context:** Drive C: has limited free space. Each full HaGRID class zip is tens of gigabytes (for example `call.zip` is about 36 GB). Downloading the nine full archives plus annotations would exhaust the disk.
- **Decision:**
  1. `download_dataset.py` keeps the official SberCloud URLs for the nine classes `call`, `point`, `like`, `dislike`, `stop`, `peace`, `ok`, `no_gesture`, and `fist`, plus `annotations.zip`.
  2. It does not download a full zip. `remotezip` reads each archive's central directory with an HTTP range request, samples images per class, and fetches only those compressed members.
  3. `--check` prints that per-image plan and the disk assessment. It does not transfer image bytes.
  4. Size is budget-driven: `--budget-gb` (default 9) and `--hard-cap-gb` (default 10). There is no `MAX_DATASET_BYTES` constant.
  5. Annotation JSON for every selected class is taken from all three splits (`train`, `val`, and `test`) under `data/raw/annotations/<split>/<class>.json`. The nine-class annotation set is about 148 MB compressed, inside the default 400 MB `--annotations-budget-mb` reserve.

## DEC-004: Pointing Geometry & 2D Projection
- **Context:** MediaPipe 2D image coordinates are normalized [0, 1]. Depth estimation from single-camera RGB has fundamental scale ambiguity.
- **Decision:** Pointing ray is strictly calculated in the 2D image plane from Landmark 5 (Index MCP) through Landmark 8 (Index Tip). Ray intersection with YOLO bounding boxes uses angular deviation and distance, rejecting targets exceeding a 15-degree threshold. Depth ambiguity is explicitly flagged in UI and documentation.

## DEC-005: Robot Abstraction Layer
- **Context:** The SerBot physical hardware and proprietary SDK are not yet connected to the host environment.
- **Decision:** Implement `BaseRobot` interface in `robot/base.py`. `SimulatedRobot` (`robot/simulated.py`) provides full 2D coordinate-based navigation, grasping, returning, obstacle/status simulation, and safety STOP with millisecond timestamps and telemetry. `SerBotStub` (`robot/serbot_stub.py`) provides an explicit hardware-ready bridge raising informative notifications until hardware drivers are provisioned.

## DEC-006: User-Controlled Dataset Download and Budget Cap
- **Context:** User mandate prohibits ANY automatic dataset downloading by the AI agent during development, verification, or pipeline execution. The downloader plans against `--budget-gb` (default 9 GB) and refuses to start a transfer whose plan exceeds `--hard-cap-gb` (default 10 GB).
- **Decision:**
  1. The agent will NEVER run `python download_dataset.py` without `--check`, and will not trigger any other HTTP dataset fetch.
  2. Only non-destructive `python download_dataset.py --check` is allowed during agent work.
  3. Before any transfer, the script prints the per-class sample plan. If that plan exceeds the hard cap it exits with: `"Plan exceeds the hard cap. Adjust --images-per-class / --budget-gb / --classes before downloading."` A disk-space pre-check can also stop the run. There is no `MAX_DATASET_BYTES` constant and no `--all-classes` flag; the nine classes above are the default set.
  4. The download manifest is `reports/download_manifest.json` with `total_bytes_transferred` and `results[<class>] = {requested, newly_fetched, files_on_disk}` (plus an `annotations` entry). Readers must tolerate a missing file or an older shape and continue from whatever is already under `data/raw/`.
  5. `main.py --all` will NEVER auto-download data. If dataset files are missing, it halts with:
     ```
     Dataset not found.

     Run:
     python download_dataset.py

     after reviewing the download plan.
     ```
  6. For offline testing, unit tests, and validation without a human download, unit tests and sample fixtures in `tests/` use deterministic mock landmarks matching the MediaPipe schema (21 points × 3 coordinates, `FEATURE_VERSION = "1.0"`).

## DEC-007: STOP Gesture Coverage (open trade-off)
- **Context:** `configs/intents.yaml` maps STOP to the single HaGRID class `stop`. An earlier design also considered `palm` as a STOP synonym, because this is the safety-critical emergency gesture and a second class would make that command more robust to pose variation.
- **Decision:** Not taken in this pass. Adding `palm` means editing `download_dataset.py` `CLASS_SPECS` and downloading another image sample, which changes the dataset size. The current download should finish first. The user decides whether to add `palm` after that, and no class-list or download change is made until then.

## DEC-008: Train on HaGRID annotation landmarks, not only on downloaded images
- **Context:** `annotations_with_landmarks` carries 21 MediaPipe landmarks (x, y) for every labelled hand. On 5,948 local images the median gap to our own MediaPipe output is 0.0016 of the image width, so they are the same landmark source.
- **Decision:** `src/gesturegrasp/hagrid_landmarks.py` builds `data/processed/hagrid_landmarks.parquet` (~436k hands, ~40k users) from the annotation JSON. The downloaded images are kept as the *live-path* check: our MediaPipe on real photos, true aspect ratio, held-out users only.
- **Trade-off:** annotations store no image size. Rows assume 3:4 portrait (the shape of ~88% of sampled images) and training jitters the aspect ratio ±15%.

## DEC-009: Feature set 2.0 (x/y only, no handedness)
- Annotation landmarks have no z and no handedness, so 2.0 uses x/y in true image geometry (x × width/height), wrist-centred and scaled, plus roll-free coordinates, roll sin/cos, bend angles, tip distances and extension ratios (126 values). Training mirrors hands at random instead of reading MediaPipe's handedness, which is often wrong for sideways hands. The pipeline picks 1.0 or 2.0 from `models/metadata.json`.

## DEC-010: STOP synonym and everyday-pose negatives (resolves DEC-007)
- `palm` (spread fingers) trains as `stop`: users with tremor, stiffness or missing fingers often cannot hold the fingers together. Everyday poses (grabbing, grip, rock, three, mute, little/middle finger, thumb_index) train as `no_gesture`, capped at 15k hands each. Configured under `training_data` in `configs/intents.yaml`. Fetched with `python download_dataset.py --annotations-only ...` (0.19 GB transfer, `data/raw` 8.33 → 9.41 GB, under the 10 GB cap the script enforces).

## DEC-011: Rotation augmentation per gesture
- Pointing happens in every direction and a sideways palm must still STOP. All gestures train with ±180° roll except `like` / `dislike` (±20°), whose direction is their meaning. Before this, `point` accuracy fell to 47% at 90° and `stop` to 0%.

## DEC-012: GPU MLP champion, temperature-scaled
- `TorchMLPClassifier` (126 → 512 → 256 → 128 → 9, LayerNorm/GELU) trains on the RTX 4060 with fresh augmentation every epoch (~3 s/epoch). It is compared against logistic regression and histogram gradient boosting on validation users; selection averages annotation and live-path macro-F1. One temperature is fitted on validation users. The champion is saved CPU-side and predicts one hand in ~0.3 ms.

## DEC-013: Pick the gesturing hand when two are visible
- HaGRID labels the second hand in a photo `no_gesture`; 477 of our 6,009 extracted rows were the wrong hand. Live rows now take the label of the annotated hand they match. At run time the pipeline classifies every visible hand and acts on the one with the strongest command, since MediaPipe's hand order is arbitrary and a resting hand on a lap must not hide a STOP.

## DEC-014: Simulator runs the real decision code on a simulated clock
- `src/gesturegrasp/simulator.py` places real held-out HaGRID hands in a virtual scene and calls `GesturePipeline.decide()`. FSM and robot take an injectable clock, so a run is exact for a seed. Only perception is replaced. The robot now delivers to the user (DELIVERING), hands over on ACKNOWLEDGE and then docks.

## DEC-015: Pointing target by majority vote (deviation from "reset if target changes")
- **Context:** The spec required the same object on every frame of the 0.8 s SELECT hold. With a 5 Hz tremor, the ray grazes a neighbouring object for a frame and the hold resets; the simulator's emergency-stop scenario passed 5% of runs under severe tremor.
- **Decision:** `TargetConsistencyTracker` votes over a sliding 0.8 s window; the target is stable once the window is full and one object holds ≥ 60% of its frames. A deliberate switch still wins within the window. A SELECT refused for lack of a stable target refunds the IntentEngine cooldown, so the user can keep pointing. The same scenario now passes 65% under severe tremor and 85% without.

## DEC-016: Responsive confidence thresholds
- The old sweep picked the *highest* threshold that kept precision ≥ 0.90, which set every command to 0.95 and dropped holds whenever confidence flickered. `train_v2` now picks the *lowest* threshold with per-frame precision ≥ 0.98 and recall ≥ 0.80 (STOP: lowest with recall ≥ 0.90). Hold time, the 0.20 margin and state gating still guard against false triggers; measured false triggers stay ≤ 0.15/min.

## DEC-017: Focus on the Pi client, server and model; remove everything else
- **Kept:** `pi_client/`, `server/`, the live pipeline (hand tracking, features 2.0, model, intent engine, state machine, pointing, YOLO), the simulated robot behind `robot.base.BaseRobot`, and the training path (`download_dataset.py` → `hagrid_landmarks.py` → `train_v2.py`).
- **Removed:**
  - the Streamlit dashboard and the scenario simulator;
  - feature set 1.0 and its heuristic no-model fallback (a missing model now fails loudly instead of guessing);
  - the synthetic-data, robustness and own-data-collector modules;
  - `main.py` and its 23 logging stages; `dataset.py` and image extraction;
  - the image-sampling half of `download_dataset.py` (it now fetches annotation JSON only, with classes read from `intents.yaml`);
  - the SerBot stub, the duplicate `gesture_model.joblib`, old logs, backups and caches.
- **Not removed:**
  - The downloaded HaGRID photos in `data/raw/<class>/` (7.5 GB) and `data/processed/landmarks.parquet`. The photos back the real-photo tests, and the parquet holds the live-path evaluation rows; both are data that cannot be cheaply recreated.
  - `.agents/`, `.claude/` and `skills-lock.json`, which are coding-agent tooling, not project code.
- `intents.yaml` keeps one `min_conf` per gesture (the duplicate `default_threshold` is gone).
- The Pi client gained `--speak` (espeak-ng) and a systemd unit for autostart.

## DEC-018: Fully on the edge: ESP32-CAM to Raspberry Pi 5 to web page
- **Context:** The system needed a Pi with a Pi camera module plus a laptop with a GPU. The target is two devices: an AI-Thinker ESP32-CAM and a Raspberry Pi 5, with any browser as the screen. The trained model, gesture semantics, thresholds, FSM and robot interface stay as they are.
- **ESP32-CAM instead of the Pi camera module:** it is cheap, has its own Wi-Fi, and can sit anywhere in the room away from the Pi. `pi_client/` and picamera2 are removed.
- **All inference on the Pi:** a loop in the FastAPI lifespan takes the newest frame, runs `GesturePipeline` and stores the result and annotated JPEG. Browsers only watch. `/infer` and `/ws/stream` stay as development endpoints, and the loop pauses while a `/ws/stream` client is connected so two sources never feed one FSM.
- **YOLO decoupled from the gesture loop:** STOP has a 0.25 s hold, so it must not wait for object boxes. YOLO runs on its own low-priority thread at 3 Hz with `imgsz=320`, and the loop reuses the last boxes. Boxes can be a few frames old, which suits objects that are not moving. `/infer` still detects synchronously, because one photo has no later frame to catch up on.
- **Latest-frame-only capture:** a one-frame slot, no queue. Frames that arrive while the pipeline is busy are replaced and never decoded, so slow inference lowers the frame rate instead of adding delay. Old frames would fire commands late, which is worse than fewer frames.
- **`CAMERA_GRAB_LATEST`:** the same rule on the ESP32, so `esp_camera_fb_get()` never returns a stale queued frame.
- **MJPEG parsed by hand, not `cv2.VideoCapture`:** FFmpeg's internal buffering adds unseen latency, and a dead stream can block `read()` with no usable timeout. Reading with `requests` and `read1()` gives real timeouts, clean reconnects with exponential backoff, and a `/capture` fallback. `read1()` also avoids `iter_content`, which can hold a finished frame back until the next one arrives.
- **VGA 640x480, quality 12 by default:** at QVGA a hand at 1-2 m is too small for MediaPipe. The firmware starts with UXGA-sized buffers so frame size can change at run time, and falls back to VGA then QVGA if that much memory cannot be found.
- **MediaPipe 0.10.14 instead of 0.10.11:** 0.10.11 has no aarch64 wheel. Compared on 1,350 HaGRID photos (x86), both found hands in the same 1,251, the landmarks matched to five decimal places, and the classifier agreed on all of them. protobuf must stay at 4.25.x.
- **NumPy classifier (`models/mlp_weights.npz`):** the same weights, so the classifier needs neither torch nor unpickling. Max abs probability difference is 1.6e-6. It stores hashes of the joblib files, so a stale export after a retrain is ignored, not trusted.
- **`max_num_hands` stays 2:** it is what lets a resting hand sit in view (DEC-013), even though MediaPipe then runs its palm detector on every frame, about twice the cost. `--max-hands 1` is an opt-in for a Pi that falls short of 10 fps.
- **Lost hands also expire by time:** the 2-frame rule forgets a hand after 0.2 s at the evaluated 15 fps, but it would stretch to 0.375 s at 8 fps. The engine now also forgets it after 0.2 s of wall-clock time, so a slow Pi behaves as evaluated. Behaviour at 15 fps and above is unchanged. Hold times were already wall-clock.
- **Ultralytics auto-install turned off in `export_yolo_pi.py`:** mid-export it upgraded protobuf and torch, which broke MediaPipe and made YOLO fail on every frame. The NCNN exporters are pinned in `requirements-pi.txt` instead. ONNX needs protobuf 6.31 or newer, which cannot share an environment with MediaPipe, so it stays opt-in and in a separate venv.
- **Trade-off:** if the camera is lost, frames stop, so the FSM and simulated robot pause and STOP gestures cannot be seen. `POST /stop` still works. A real robot must stop itself when `/camera` reports `lost`; that belongs in the SerBot driver.
