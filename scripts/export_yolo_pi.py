"""
Export yolo11n to NCNN (and optionally ONNX) for CPU inference on the Raspberry Pi 5, compare speeds,
and make the fastest one the default.

Run this on the Pi from the project root, once, while it has internet (it downloads yolo11n.pt and the
exporters):
    python scripts/export_yolo_pi.py            # exports NCNN, then benchmarks PyTorch vs NCNN
    python scripts/export_yolo_pi.py --onnx     # also exports and benchmarks ONNX

The winner is written to yolo_best.txt, and the server loads what that file names. The ranking differs
between machines, so run this on the Pi itself. Without the file the server uses yolo11n.pt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gesturegrasp.detector import BEST_MODEL_FILE, DEFAULT_IMGSZ  # noqa: E402


def time_model(weights: str, imgsz: int, runs: int) -> float:
    model = YOLO(weights, task="detect")
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    model(frame, imgsz=imgsz, device="cpu", verbose=False)  # the first call loads everything
    started = time.perf_counter()
    for _ in range(runs):
        model(frame, imgsz=imgsz, device="cpu", verbose=False)
    return (time.perf_counter() - started) / runs * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--onnx", action="store_true", help="Also export and time ONNX")
    args = parser.parse_args()

    candidates = {"pytorch": args.weights,
                  "ncnn": YOLO(args.weights).export(format="ncnn", imgsz=args.imgsz)}
    if args.onnx:
        candidates["onnx"] = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz)

    times = {name: time_model(path, args.imgsz, args.runs) for name, path in candidates.items()}
    print(f"\nyolo11n at imgsz={args.imgsz}, CPU, mean of {args.runs} runs:")
    for name, ms in sorted(times.items(), key=lambda kv: kv[1]):
        print(f"  {name:<8} {ms:7.1f} ms  ({1000.0 / ms:5.1f} fps)  {candidates[name]}")
    best = min(times, key=times.get)
    Path(BEST_MODEL_FILE).write_text(str(candidates[best]), encoding="utf-8")
    print(f"Fastest here: {best}. The server will use it ({BEST_MODEL_FILE}).")


if __name__ == "__main__":
    main()
