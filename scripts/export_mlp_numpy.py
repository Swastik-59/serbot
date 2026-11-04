"""
Export the trained gesture MLP to models/mlp_weights.npz, so the classifier needs no torch at run time.

Run on any machine that has torch, after training:
    python scripts/export_mlp_numpy.py

It refuses to write the file unless the NumPy model matches the original (max abs prob diff < 1e-5).
The joblib files are only read. Nothing is retrained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gesturegrasp.numpy_mlp import NumpyMLPClassifier, NumpyScaler, file_sha256  # noqa: E402

MAX_PROB_DIFF = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/calibrated_model.joblib"))
    parser.add_argument("--scaler", type=Path, default=Path("models/scaler.joblib"))
    parser.add_argument("--out", type=Path, default=Path("models/mlp_weights.npz"))
    args = parser.parse_args()

    # joblib unpickles: only point this at files this project's own training wrote.
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    state = {k: v.numpy().astype(np.float32) for k, v in model._state.items()}
    layers = sorted({int(k.split(".")[0]) for k in state})
    # Each hidden block is Linear, LayerNorm, GELU, Dropout; the output layer stands alone.
    linear, norm = layers[0::2][:-1], layers[1::2]
    arrays: dict[str, np.ndarray] = {}
    for i, (lin, nrm) in enumerate(zip(linear, norm)):
        arrays |= {f"w{i}": state[f"{lin}.weight"], f"b{i}": state[f"{lin}.bias"],
                   f"norm_w{i}": state[f"{nrm}.weight"], f"norm_b{i}": state[f"{nrm}.bias"]}
    arrays |= {"out_w": state[f"{layers[-1]}.weight"], "out_b": state[f"{layers[-1]}.bias"]}
    bundle = {
        **arrays,
        "n_hidden": np.array(len(linear)),
        "classes": np.asarray(model.classes_).astype(str),
        "temperature": np.array(model.temperature, dtype=np.float64),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "model_sha256": np.array(file_sha256(args.model)),
        "scaler_sha256": np.array(file_sha256(args.scaler)),
    }

    rng = np.random.default_rng(42)
    raw = rng.normal(scaler.mean_, scaler.scale_ * 2.0, (5000, scaler.mean_.size))
    original = model.predict_proba(scaler.transform(raw))
    exported = NumpyMLPClassifier(bundle)
    rebuilt = exported.predict_proba(NumpyScaler(bundle["scaler_mean"], bundle["scaler_scale"]).transform(raw))
    diff = float(np.abs(original - rebuilt).max())
    if diff >= MAX_PROB_DIFF:
        raise SystemExit(f"Not exported: max abs prob diff {diff:.2e} is over {MAX_PROB_DIFF:.0e}")

    np.savez_compressed(args.out, **bundle)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB), max abs prob diff {diff:.2e}")


if __name__ == "__main__":
    main()
