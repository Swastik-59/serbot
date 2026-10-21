"""
Large-scale training (feature set 2.0) on HaGRID annotation landmarks, GPU MLP vs CPU baselines.

Data:   data/processed/hagrid_landmarks.parquet (annotation landmarks, ~436k hands, 40k users)
        + data/processed/landmarks.parquet (our MediaPipe run on real images; true aspect).
Split:  by user_id into train / val / test (70 / 15 / 15). No person is in two splits.
Select: champion by mean of val macro-F1 on annotation rows and on live MediaPipe rows.
Report: per-frame metrics on test users, separately for
          - annotation rows (large, aspect assumed),
          - live MediaPipe rows (exact run-time path: our MediaPipe on real images),
          - users aged 60+ (HaGRID meta age),
        and temporal metrics from simulated 15 fps clips through the real IntentEngine.

Run:  python -m src.gesturegrasp.train_v2
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.gesturegrasp.features import FEATURE_VERSION_2D, augment_points_2d, extract_features_2d, feature_names_2d
from src.gesturegrasp.hagrid_landmarks import build_annotation_landmarks, load_mediapipe_rows
from src.gesturegrasp.intent_engine import IntentEngine
from src.gesturegrasp.torch_mlp import TorchMLPClassifier

logger = logging.getLogger("gesturegrasp.train_v2")

# Which FSM state each gesture is meant for, when replaying clips through the IntentEngine.
INTENDED_STATE = {
    "call": "IDLE", "point": "LISTENING", "like": "TARGET_PROPOSED", "dislike": "TARGET_PROPOSED",
    "stop": "EXECUTING", "peace": "LISTENING", "ok": "AWAIT_FEEDBACK",
}


def calculate_ece(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the top-class confidence."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def write_min_conf(config_path: Path, min_conf: dict[str, float]) -> None:
    """Rewrite each gesture's min_conf line in intents.yaml in place, keeping comments and layout."""
    out, gesture, in_gestures, written = [], None, False, set()
    stamp = datetime.now(timezone.utc).isoformat()
    for line in Path(config_path).read_text(encoding="utf-8").splitlines(keepends=True):
        if line.strip() and not line.startswith(" "):
            in_gestures = line.startswith("gestures:")
        header = re.match(r"^  (\w+):\s*$", line)
        if in_gestures and header:
            gesture = header.group(1)
        if in_gestures and gesture in min_conf and line.strip().startswith("min_conf:"):
            line = f"    min_conf: {min_conf[gesture]:.2f}\n"
            written.add(gesture)
        elif line.startswith("  updated_at:"):
            line = f'  updated_at: "{stamp}"\n'
        out.append(line)
    missing = set(min_conf) - written
    if missing:
        raise ValueError(f"intents.yaml has no min_conf line for: {sorted(missing)}")
    Path(config_path).write_text("".join(out), encoding="utf-8")
    logger.info("Wrote per-gesture min_conf to %s", config_path)


def select_min_conf_responsive(y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray,
                               gestures_cfg: dict[str, Any], min_precision: float = 0.98,
                               min_recall: float = 0.80) -> dict[str, float]:
    """Per-gesture min_conf that favours responsiveness: the LOWEST threshold in 0.50-0.95 that keeps
    per-frame precision >= min_precision (and recall >= min_recall). The IntentEngine's hold time,
    margin and state gating guard against false triggers on top; temporal clips measure the result.
    STOP keeps the lowest threshold with recall >= 0.90 (a missed STOP is the costly error)."""
    y_true = np.asarray(y_true)
    pred = classes[y_prob.argmax(1)]
    conf = y_prob.max(1)
    grid = [round(float(v), 2) for v in np.arange(0.50, 0.96, 0.05)]
    chosen: dict[str, float] = {}
    for gesture, meta in gestures_cfg.items():
        if gesture not in set(map(str, classes)):
            continue
        rows = []
        for th in grid:
            hit = (pred == gesture) & (conf >= th)
            tp = int((hit & (y_true == gesture)).sum())
            precision = tp / max(int(hit.sum()), 1)
            recall = tp / max(int((y_true == gesture).sum()), 1)
            rows.append((th, precision, recall))
        if meta.get("intent") == "STOP":
            ok = [r for r in rows if r[2] >= 0.90]
        elif meta.get("actionable"):
            ok = [r for r in rows if r[1] >= min_precision and r[2] >= min_recall]
        else:
            ok = rows
        chosen[gesture] = min(ok)[0] if ok else max(rows, key=lambda r: (r[1], r[0]))[0]
    return chosen


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _points(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["points"].to_numpy()).astype(np.float32).reshape(-1, 21, 2)


def split_users(users: np.ndarray, seed: int) -> dict[str, set]:
    unique = np.unique(users)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_i, rest_i = next(gss.split(unique, groups=unique))
    rest = unique[rest_i]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_i, test_i = next(gss2.split(rest, groups=rest))
    return {"train": set(unique[train_i]), "val": set(rest[val_i]), "test": set(rest[test_i])}


def per_frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = {str(c): {"precision": round(float(pi), 4), "recall": round(float(ri), 4),
                          "f1_score": round(float(fi), 4), "support": int(si)}
                 for c, pi, ri, fi, si in zip(classes, p, r, f, s)}
    conf = y_prob.max(1)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "stop_recall": per_class.get("stop", {}).get("recall", 0.0),
        "expected_calibration_error": round(calculate_ece((y_pred == y_true).astype(int), conf), 4),
        "average_brier_score": round(float(np.mean([
            brier_score_loss((y_true == c).astype(int), y_prob[:, i]) for i, c in enumerate(classes)])), 4),
        "per_class": per_class,
    }


def simulate_clips(predict_proba, scaler, points: np.ndarray, aspect: np.ndarray, labels: np.ndarray,
                   classes: np.ndarray, intents_path: Path, rng: np.random.Generator,
                   seconds: float = 2.0, fps: int = 15) -> dict[str, Any]:
    """Replay each hand as a jittered clip through IntentEngine; count what actually fires."""
    frames = int(seconds * fps)
    n = len(points)
    rep_pts = np.repeat(points, frames, axis=0)
    rep_asp = np.repeat(aspect, frames)
    # Small per-frame jitter + slow drift: a resting, trembling hand rather than a new pose each frame.
    jittered, _ = augment_points_2d(rep_pts, rep_asp, rng, aspect_jitter=0.0, rotation_deg=3.0,
                                    noise=0.006, mirror_prob=0.0)
    probs = predict_proba(scaler.transform(extract_features_2d(jittered, rep_asp))).reshape(n, frames, -1)
    cfg = yaml.safe_load(Path(intents_path).read_text(encoding="utf-8"))
    intent_of = {g: m["intent"] for g, m in cfg["gestures"].items()}

    engine = IntentEngine(intents_path)
    logging.getLogger("gesturegrasp.intent_engine").setLevel(logging.WARNING)

    def run(state: str, clip: np.ndarray) -> tuple[str, float]:
        engine.reset()
        for i in range(frames):
            res = engine.update(dict(zip(map(str, classes), clip[i])), current_state=state,
                                hand_detected=True, timestamp=i / fps)
            if res.intent_fired:
                return res.intent, i / fps
        return "NONE", float("nan")

    out: dict[str, Any] = {"clip_seconds": seconds, "fps": fps, "per_gesture": {}}
    for gesture, state in INTENDED_STATE.items():
        idx = np.where(labels == gesture)[0]
        fired = [run(state, probs[i]) for i in idx]
        correct = [t for intent, t in fired if intent == intent_of[gesture]]
        wrong = sum(1 for intent, _ in fired if intent not in ("NONE", intent_of[gesture]))
        out["per_gesture"][gesture] = {
            "clips": int(len(idx)), "state": state,
            "fired_correctly": round(len(correct) / max(len(idx), 1), 4),
            "fired_wrong_intent": round(wrong / max(len(idx), 1), 4),
            "median_seconds_to_fire": round(float(np.median(correct)), 2) if correct else None,
        }
    neg = np.where(np.isin(labels, ["no_gesture", "fist"]))[0]
    minutes = len(neg) * seconds / 60.0
    for state in ("IDLE", "LISTENING", "TARGET_PROPOSED"):
        fired = [run(state, probs[i])[0] for i in neg]
        bad = [f for f in fired if f != "NONE"]
        out[f"false_triggers_per_minute_{state.lower()}"] = round(len(bad) / max(minutes, 1e-9), 3)
        out[f"false_trigger_intents_{state.lower()}"] = pd.Series(bad, dtype=str).value_counts().to_dict()
    out["negative_clip_minutes"] = round(minutes, 1)
    return out


def train_v2(models_dir: Path = Path("models"), reports_dir: Path = Path("reports"),
             intents_path: Path = Path("configs/intents.yaml"), seed: int = 42,
             write_thresholds: bool = True) -> dict[str, Any]:
    t_start = time.time()
    rng = np.random.default_rng(seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path("data/processed/hagrid_landmarks.parquet")
    ann = pd.read_parquet(ann_path) if ann_path.exists() else build_annotation_landmarks(output=ann_path)
    live = load_mediapipe_rows()
    df = pd.concat([ann, live], ignore_index=True)
    logger.info("Rows: %d annotation + %d live MediaPipe, %d users", len(ann), len(live), df["user_id"].nunique())

    split = split_users(df["user_id"].to_numpy(), seed)
    part = {k: df[df["user_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}
    for k, v in part.items():
        logger.info("%-5s %7d rows, %6d users", k, len(v), v["user_id"].nunique())

    tr = part["train"]
    tr_pts, tr_asp, y_tr = _points(tr), tr["aspect"].to_numpy(np.float32), tr["gesture"].to_numpy(str)
    scaler = StandardScaler().fit(extract_features_2d(tr_pts, tr_asp))

    def feats(frame: pd.DataFrame) -> np.ndarray:
        return scaler.transform(extract_features_2d(_points(frame), frame["aspect"].to_numpy(np.float32)))

    aug_cfg = yaml.safe_load(Path(intents_path).read_text(encoding="utf-8")).get("training_data", {}) or {}
    rot_limit = np.where(np.isin(y_tr, aug_cfg.get("oriented_gestures", ["like", "dislike"])),
                         float(aug_cfg.get("oriented_rotation_deg", 20.0)),
                         float(aug_cfg.get("rotation_deg", 180.0))).astype(np.float32)

    def make_epoch_x(epoch: int) -> np.ndarray:
        if epoch == 0:
            return scaler.transform(extract_features_2d(tr_pts, tr_asp))
        p, a = augment_points_2d(tr_pts, tr_asp, np.random.default_rng(seed + epoch), rotation_deg=rot_limit)
        return scaler.transform(extract_features_2d(p, a))

    val = part["val"]
    x_val, y_val = feats(val), val["gesture"].to_numpy(str)
    val_live = (val["source"] == "mediapipe_live").to_numpy()

    def selection_score(model) -> dict[str, float]:
        pred = model.predict(x_val)
        ann_f1 = macro_f1(y_val[~val_live], pred[~val_live])
        live_f1 = macro_f1(y_val[val_live], pred[val_live]) if val_live.any() else ann_f1
        return {"val_macro_f1_annotation": round(ann_f1, 4), "val_macro_f1_live": round(live_f1, 4),
                "selection_score": round((ann_f1 + live_f1) / 2, 4)}

    candidates: dict[str, Any] = {}
    results: dict[str, Any] = {}

    # CPU baselines on the original rows plus one augmented copy.
    p1, a1 = augment_points_2d(tr_pts, tr_asp, np.random.default_rng(seed + 1000), rotation_deg=rot_limit)
    x_aug = np.concatenate([make_epoch_x(0), scaler.transform(extract_features_2d(p1, a1))])
    y_aug = np.concatenate([y_tr, y_tr])
    for name, model in {
        "LogisticRegression": LogisticRegression(max_iter=300, C=1.0, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                               class_weight="balanced", random_state=seed),
    }.items():
        t0 = time.time()
        model.fit(x_aug, y_aug)
        candidates[name] = model
        results[name] = {**selection_score(model), "train_seconds": round(time.time() - t0, 1), "device": "cpu"}
        logger.info("%-22s %s", name, results[name])

    t0 = time.time()
    mlp = TorchMLPClassifier(seed=seed).fit_epochs(make_epoch_x, y_tr, x_val, y_val, macro_f1)
    candidates["TorchMLP"] = mlp
    import torch
    results["TorchMLP"] = {**selection_score(mlp), "train_seconds": round(time.time() - t0, 1),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                           "epochs_run": len(mlp.history_)}
    logger.info("%-22s %s", "TorchMLP", results["TorchMLP"])

    champion_name = max(results, key=lambda k: results[k]["selection_score"])
    champion = candidates[champion_name]
    logger.info("Champion: %s", champion_name)

    if isinstance(champion, TorchMLPClassifier):
        temp = champion.calibrate_temperature(x_val, y_val)
        logger.info("Temperature scaling on val users: T = %.3f", temp)
        calibrated = champion
    else:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(champion), method="sigmoid").fit(x_val, y_val)
    classes = np.asarray(calibrated.classes_)

    intents_cfg = yaml.safe_load(Path(intents_path).read_text(encoding="utf-8"))
    val_prob = calibrated.predict_proba(x_val)
    min_conf = select_min_conf_responsive(y_val, val_prob, classes, intents_cfg["gestures"])
    if write_thresholds:
        write_min_conf(intents_path, min_conf)

    # ---- test users -------------------------------------------------------------------------
    te = part["test"]
    x_te, y_te = feats(te), te["gesture"].to_numpy(str)
    prob_te = calibrated.predict_proba(x_te)
    pred_te = classes[prob_te.argmax(1)]
    is_live = (te["source"] == "mediapipe_live").to_numpy()
    is_old = (te["age"] >= 60).to_numpy()
    ann_metrics = per_frame_metrics(y_te[~is_live], pred_te[~is_live], prob_te[~is_live], classes)
    report: dict[str, Any] = {
        "feature_version": FEATURE_VERSION_2D,
        "champion_model": champion_name,
        **{k: v for k, v in ann_metrics.items() if k != "n"},
        "test_rows": ann_metrics["n"],
        "live_mediapipe_test": per_frame_metrics(y_te[is_live], pred_te[is_live], prob_te[is_live], classes),
        "age_60_plus_test": per_frame_metrics(y_te[is_old], pred_te[is_old], prob_te[is_old], classes),
        "age_under_60_test": per_frame_metrics(y_te[~is_old & ~is_live], pred_te[~is_old & ~is_live],
                                               prob_te[~is_old & ~is_live], classes),
        "candidates": results,
        "selected_min_conf": min_conf,
        "threshold_split": "val_users",
        "data": {"annotation_rows": int(len(ann)), "live_rows": int(len(live)),
                 "users": int(df["user_id"].nunique()),
                 "rows_per_split": {k: int(len(v)) for k, v in part.items()},
                 "class_counts_train": tr["gesture"].value_counts().to_dict()},
    }
    # Same test hands, turned to a random roll within each gesture's allowed range (any direction
    # for most; +/-20 deg for thumbs up/down). Measures pointing / STOP at sideways angles.
    te_ann = te[~is_live]
    te_rot_limit = np.where(np.isin(te_ann["gesture"], aug_cfg.get("oriented_gestures", ["like", "dislike"])),
                            float(aug_cfg.get("oriented_rotation_deg", 20.0)), float(aug_cfg.get("rotation_deg", 180.0)))
    rp, ra = augment_points_2d(_points(te_ann), te_ann["aspect"].to_numpy(np.float32), np.random.default_rng(seed + 7),
                               aspect_jitter=0.0, noise=0.0, mirror_prob=0.0, rotation_deg=te_rot_limit)
    rprob = calibrated.predict_proba(scaler.transform(extract_features_2d(rp, ra)))
    report["rotated_test"] = per_frame_metrics(te_ann["gesture"].to_numpy(str), classes[rprob.argmax(1)], rprob, classes)

    if isinstance(champion, TorchMLPClassifier):
        report["temperature"] = round(champion.temperature, 4)
        report["training_history"] = champion.history_

    # Single-row latency, as the live pipeline calls it.
    lat = []
    for i in range(200):
        t0 = time.perf_counter()
        calibrated.predict_proba(x_te[i % len(x_te):i % len(x_te) + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    report["latency_ms"] = {"mean": round(float(np.mean(lat)), 3), "p50": round(float(np.percentile(lat, 50)), 3),
                            "p95": round(float(np.percentile(lat, 95)), 3)}

    # Temporal behaviour through the real IntentEngine on a sample of test users' hands.
    sample = te[~is_live].sample(frac=1.0, random_state=seed).groupby("gesture").head(400)
    temporal = simulate_clips(calibrated.predict_proba, scaler, _points(sample), sample["aspect"].to_numpy(np.float32),
                              sample["gesture"].to_numpy(str), classes, intents_path, rng)
    report["temporal"] = temporal
    report["false_triggers_per_minute"] = temporal["false_triggers_per_minute_listening"]

    sweep = []
    conf = prob_te.max(1)
    for th in np.arange(0.50, 0.96, 0.05):
        keep = conf >= th
        sweep.append({"threshold": round(float(th), 2), "coverage": round(float(keep.mean()), 4),
                      "accuracy_on_accepted": round(float(accuracy_score(y_te[keep], pred_te[keep])) if keep.any() else 1.0, 4),
                      "accepted_count": int(keep.sum())})

    # ---- save -------------------------------------------------------------------------------
    joblib.dump(calibrated, models_dir / "calibrated_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    metadata = {
        "feature_version": FEATURE_VERSION_2D, "feature_dim": len(feature_names_2d()),
        "champion_model": champion_name, "classes": list(map(str, classes)),
        "cv_results": results, "timestamp": datetime.now(timezone.utc).isoformat(), "random_seed": seed,
        "test_macro_f1": report["macro_f1"], "test_accuracy": report["accuracy"],
        "live_test_macro_f1": report["live_mediapipe_test"].get("macro_f1"),
        "stop_recall": report["stop_recall"], "python_version": sys.version,
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    labels = list(map(str, classes))
    for name, mask in (("confusion_matrix.json", ~is_live), ("confusion_matrix_live.json", is_live)):
        cm = confusion_matrix(y_te[mask], pred_te[mask], labels=classes) if mask.any() else np.zeros((len(labels),) * 2)
        (reports_dir / name).write_text(json.dumps({"labels": labels, "matrix": cm.astype(int).tolist()}, indent=2),
                                        encoding="utf-8")
    (reports_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    report["total_seconds"] = round(time.time() - t_start, 1)
    (reports_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    logger.info("Test (annotation) macro-F1 %.4f | live MediaPipe macro-F1 %s | 60+ macro-F1 %s | STOP recall %.4f",
                report["macro_f1"], report["live_mediapipe_test"].get("macro_f1"),
                report["age_60_plus_test"].get("macro_f1"), report["stop_recall"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_v2()
