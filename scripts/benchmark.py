"""
Measure the pipeline on this machine, stage by stage, for any camera source.

    python scripts/benchmark.py --source http://esp32cam.local:81/stream
    python scripts/benchmark.py --source data/raw/point/some_photo.jpg --seconds 10
    python scripts/benchmark.py --source clip.mp4 --mock-detector --json bench.json

Part 1 times each stage on its own. Part 2 runs the real loop (YOLO on its own thread) and reports the
frame rate and how old each frame was once its result existed. That age starts when the frame reaches
the Pi, so add the ESP32's capture and Wi-Fi time (about 50-80 ms, needs hardware check) for glass-to-state.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gesturegrasp.camera import DEFAULT_CAMERA_URL, make_source  # noqa: E402
from src.gesturegrasp.pipeline import GesturePipeline  # noqa: E402

TARGET_FPS = 10.0
TARGET_LATENCY_MS = 250.0
YOLO_EVERY = 5  # YOLO is timed on every 5th frame of part 1, which is how rarely the loop needs it


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"mean": round(statistics.fmean(ordered), 2), "p50": round(ordered[len(ordered) // 2], 2),
            "p95": round(ordered[int(len(ordered) * 0.95)], 2), "n": len(ordered)}


def timed(fn, *args):
    started = time.perf_counter()
    value = fn(*args)
    return value, (time.perf_counter() - started) * 1000.0


def measure_stages(source, pipe: GesturePipeline, seconds: float) -> dict[str, dict[str, float]]:
    stages: dict[str, list[float]] = {"decode": [], "mediapipe": [], "mlp": [], "yolo": [], "draw_encode": []}
    fallback_hand = np.random.default_rng(0).random((21, 3), dtype=np.float32)
    deadline, last_id, count = time.monotonic() + seconds, 0, 0
    while time.monotonic() < deadline:
        item = source.wait_for_frame(last_id, timeout=1.0)
        if item is None:
            continue
        frame, last_id, _ = item
        count += 1
        if source.decode_ms:
            stages["decode"].append(source.decode_ms)
        hands, ms = timed(pipe.hand_extractor.process_frame, frame)
        stages["mediapipe"].append(ms)
        hand = hands[0].landmarks if hands else fallback_hand
        stages["mlp"].append(timed(pipe.predict_probabilities, hand, frame.shape[:2])[1])
        objects = []
        if count % YOLO_EVERY == 1:
            objects, ms = timed(pipe.detector.detect, frame)
            stages["yolo"].append(ms)
        started = time.perf_counter()
        drawn = pipe.hand_extractor.draw_landmarks(frame, [hand])
        drawn = pipe.detector.draw_annotations(drawn, objects)
        cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 70])
        stages["draw_encode"].append((time.perf_counter() - started) * 1000.0)
    return {name: summary(values) for name, values in stages.items() if values}


def measure_loop(source, pipe: GesturePipeline, seconds: float) -> dict[str, float]:
    ages, processed, last_id, first_id = [], 0, 0, None
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        item = source.wait_for_frame(last_id, timeout=1.0)
        if item is None:
            continue
        frame, last_id, stamp = item
        first_id = first_id or last_id
        result = pipe.process(frame, timestamp=stamp, live=True)
        cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        ages.append((time.time() - stamp) * 1000.0)
        processed += 1
    elapsed = time.monotonic() - started
    return {"fps": round(processed / elapsed, 1), "frames": processed,
            "skipped": max(0, last_id - (first_id or 0) + 1 - processed),
            "age_mean_ms": round(statistics.fmean(ages), 1) if ages else 0.0,
            "age_p95_ms": round(sorted(ages)[int(len(ages) * 0.95)], 1) if ages else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_CAMERA_URL, help="Stream URL, host, video file or image")
    parser.add_argument("--seconds", type=float, default=15.0, help="Length of each part")
    parser.add_argument("--mock-detector", action="store_true", help="Skip YOLO")
    parser.add_argument("--max-hands", type=int, choices=(1, 2), default=2)
    parser.add_argument("--json", type=Path, help="Also write the numbers to this file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    pipe = GesturePipeline(use_mock_detector=args.mock_detector, max_hands=args.max_hands)
    source = make_source(args.source).start()
    try:
        if source.wait_for_frame(0, timeout=10.0) is None:
            raise SystemExit(f"No frames from {args.source}: {source.status().get('last_error')}")
        stages = measure_stages(source, pipe, args.seconds)
        loop = measure_loop(source, pipe, args.seconds)
        camera = source.status()
    finally:
        source.stop()
        pipe.detector.close()
        pipe.hand_extractor.close()

    print(f"\nSource {args.source}: {camera['fps']} fps arriving, classifier {type(pipe.model).__name__}, "
          f"detector {pipe.detector.active_model_name}")
    print(f"{'stage':<14}{'mean ms':>10}{'p50':>9}{'p95':>9}{'n':>7}")
    for name, s in stages.items():
        print(f"{name:<14}{s['mean']:>10.2f}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['n']:>7}")
    print(f"\nReal loop: {loop['fps']} fps, {loop['skipped']} frames skipped, "
          f"frame age at result {loop['age_mean_ms']} ms (p95 {loop['age_p95_ms']} ms)")
    print(f"Targets: {'PASS' if loop['fps'] >= TARGET_FPS else 'FAIL'} at least {TARGET_FPS:g} fps, "
          f"{'PASS' if loop['age_p95_ms'] < TARGET_LATENCY_MS else 'FAIL'} under {TARGET_LATENCY_MS:g} ms")
    if args.json:
        args.json.write_text(json.dumps({"source": args.source, "stages": stages, "loop": loop}, indent=2))


if __name__ == "__main__":
    main()
