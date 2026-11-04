"""
The trained gesture MLP without torch or scikit-learn at run time.

scripts/export_mlp_numpy.py writes models/mlp_weights.npz from the trained joblib files. The weights
are the same; this file only repeats the forward pass (Linear, LayerNorm, GELU) in NumPy. The npz stores
hashes of the joblib files it came from, so after a retrain a stale export is ignored, not trusted.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("gesturegrasp.numpy_mlp")

_LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default
_SQRT2 = math.sqrt(2.0)
_erf = np.frompyfunc(math.erf, 1, 1)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gelu(x: np.ndarray) -> np.ndarray:
    # torch.nn.GELU() is the exact form, so this uses the real erf, not the tanh approximation.
    return 0.5 * x * (1.0 + _erf(x / _SQRT2).astype(x.dtype))


class NumpyScaler:
    """StandardScaler.transform, with the fitted mean and scale."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_


class NumpyMLPClassifier:
    """Same predict_proba and classes_ as TorchMLPClassifier."""

    def __init__(self, data):
        self.classes_ = data["classes"]
        self.temperature = float(data["temperature"])
        self._hidden = [(data[f"w{i}"], data[f"b{i}"], data[f"norm_w{i}"], data[f"norm_b{i}"])
                        for i in range(int(data["n_hidden"]))]
        self._out_w = data["out_w"]
        self._out_b = data["out_b"]

    def _logits(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for w, b, norm_w, norm_b in self._hidden:
            h = h @ w.T + b
            mean = h.mean(-1, keepdims=True)
            var = ((h - mean) ** 2).mean(-1, keepdims=True)
            h = (h - mean) / np.sqrt(var + _LAYER_NORM_EPS) * norm_w + norm_b
            h = _gelu(h)
        return h @ self._out_w.T + self._out_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._logits(x) / self.temperature
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self._logits(x).argmax(1)]


def load_numpy_model(weights_path: Path, model_path: Path,
                     scaler_path: Path) -> Optional[tuple[NumpyMLPClassifier, NumpyScaler]]:
    """The exported model, or None if there is no export or it no longer matches the joblib files."""
    if not weights_path.exists():
        return None
    data = np.load(weights_path, allow_pickle=False)
    if str(data["model_sha256"]) != file_sha256(model_path) or str(data["scaler_sha256"]) != file_sha256(scaler_path):
        logger.warning("%s is out of date; using the joblib model. Refresh it with scripts/export_mlp_numpy.py",
                       weights_path)
        return None
    return NumpyMLPClassifier(data), NumpyScaler(data["scaler_mean"], data["scaler_scale"])
