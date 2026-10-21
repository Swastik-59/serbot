"""
Large-scale landmark dataset from HaGRID annotation JSON.

HaGRID v2's annotations_with_landmarks carry 21 MediaPipe hand landmarks (x, y; normalised to the
image) for every labelled hand, so the full ~30k-hands-per-class set is usable without the images.
Checked on 5,948 local images: median |annotation - our MediaPipe output| = 0.0016 of the image.

The annotations do not store image size. Landmark x is normalised by width and y by height, so
geometry depends on the aspect ratio. ~88% of sampled HaGRID images are 3:4 portrait, so
ASSUMED_ASPECT is used for annotation rows and training jitters the aspect to cover the rest.
Rows extracted by our own MediaPipe run (data/processed/landmarks.parquet) keep their real aspect.

Output: data/processed/hagrid_landmarks.parquet with columns
    sample_id, gesture, source_class, user_id, split, age, aspect, source, points (42 floats)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("gesturegrasp.hagrid_landmarks")

ASSUMED_ASPECT = 0.75  # width / height of a 1440x1920 portrait HaGRID image
SPLITS = ("train", "val", "test")


def class_mapping(intents_cfg: dict[str, Any]) -> tuple[dict[str, str], int]:
    """HaGRID source class -> training label, plus the per-source-class cap."""
    mapping = {name: name for name in intents_cfg["gestures"]}
    extra = intents_cfg.get("training_data", {}) or {}
    for source, target in (extra.get("aliases") or {}).items():
        if target not in intents_cfg["gestures"]:
            raise ValueError(f"Alias {source} -> {target}: {target} is not a gesture in intents.yaml")
        mapping[source] = target
    for source in extra.get("hard_negatives") or []:
        mapping[source] = "no_gesture"
    return mapping, int(extra.get("max_hands_per_source_class", 0) or 0)


def _first_age(meta: Optional[dict]) -> float:
    for value in ((meta or {}).get("age") or []):
        if value is not None:
            return float(value)
    return float("nan")


def build_annotation_landmarks(
    annotations_dir: Path = Path("data/raw/annotations"),
    intents_path: Path = Path("configs/intents.yaml"),
    output: Path = Path("data/processed/hagrid_landmarks.parquet"),
    seed: int = 42,
) -> pd.DataFrame:
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_cfg = yaml.safe_load(f)
    mapping, cap = class_mapping(intents_cfg)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source_class, label in mapping.items():
        class_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            path = Path(annotations_dir) / split / f"{source_class}.json"
            if not path.exists():
                missing.append(f"{split}/{source_class}.json")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for image_id, rec in payload.items():
                hands = rec.get("hand_landmarks") or []
                for hand_idx, (hand_label, lm) in enumerate(zip(rec.get("labels") or [], hands)):
                    if not lm or len(lm) != 21:
                        continue
                    # A gesture image's second hand is labelled no_gesture; keep that label as is.
                    target = label if hand_label == source_class else mapping.get(hand_label)
                    if target is None:
                        continue
                    class_rows.append({
                        "sample_id": f"{image_id}#{hand_idx}",
                        "gesture": target,
                        "source_class": hand_label,
                        "user_id": str(rec.get("user_id", "unknown")),
                        "split": split,
                        "age": _first_age(rec.get("meta")),
                        "aspect": ASSUMED_ASPECT,
                        "source": "hagrid_annotation",
                        "points": np.asarray(lm, dtype=np.float32).reshape(-1).tolist(),
                    })
        if cap and len(class_rows) > cap and source_class not in intents_cfg["gestures"]:
            keep = rng.choice(len(class_rows), size=cap, replace=False)
            class_rows = [class_rows[i] for i in sorted(keep)]
        logger.info("%-14s -> %-10s %6d hands", source_class, label, len(class_rows))
        rows.extend(class_rows)

    if missing:
        logger.warning("Missing annotation files (run download_dataset.py --annotations-only ...): %s",
                       ", ".join(missing))
    if not rows:
        raise RuntimeError(f"No annotation landmarks found under {annotations_dir}")

    df = pd.DataFrame(rows).drop_duplicates("sample_id")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info("Wrote %d hands (%d users) to %s", len(df), df["user_id"].nunique(), output)
    return df


def load_mediapipe_rows(path: Path = Path("data/processed/landmarks.parquet"),
                        annotations_dir: Path = Path("data/raw/annotations"),
                        match_tolerance: float = 0.02) -> pd.DataFrame:
    """Our own MediaPipe output on real images, in the same row format (real aspect, x/y only).

    A HaGRID photo often shows a second, relaxed hand annotated no_gesture. When MediaPipe found
    that hand instead of the gesturing one, the image's class is the wrong label for the row. Each
    row therefore takes the label of the annotated hand it matches; rows that match no annotated
    hand within match_tolerance (mean |dx|, |dy| as a fraction of the image) are dropped.
    """
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    annotations: dict[str, Any] = {}
    for gesture in df["gesture"].unique():
        for split in SPLITS:
            f = Path(annotations_dir) / split / f"{gesture}.json"
            if f.exists():
                annotations.update(json.loads(f.read_text(encoding="utf-8")))

    rows, relabelled, dropped = [], 0, 0
    for r in df.itertuples():
        pts = np.asarray(r.landmarks, dtype=np.float32).reshape(21, 3)[:, :2]
        label = r.gesture
        ann = annotations.get(str(r.sample_id))
        if ann:
            candidates = [(float(np.abs(np.asarray(lm, np.float32) - pts).mean()), lab)
                          for lm, lab in zip(ann.get("hand_landmarks") or [], ann.get("labels") or [])
                          if lm and len(lm) == 21]
            if candidates:
                dist, lab = min(candidates)
                if dist > match_tolerance:
                    dropped += 1
                    continue
                if lab != label:
                    relabelled += 1
                    label = lab
        rows.append({
            "sample_id": f"{r.sample_id}#mp", "gesture": label, "source_class": label,
            "user_id": str(r.user_id), "split": getattr(r, "split", "unknown"), "age": np.nan,
            "aspect": np.float32(r.image_width / r.image_height), "source": "mediapipe_live",
            "points": pts.reshape(-1).tolist(),
        })
    logger.info("Live MediaPipe rows: %d kept, %d relabelled to the matched annotated hand, %d dropped (no match).",
                len(rows), relabelled, dropped)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_annotation_landmarks()
