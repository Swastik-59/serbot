"""
Build the hand-sign pictograms for the web page from real hand landmarks.

For each gesture this finds the median hand across data/processed/landmarks.parquet, then keeps the real
hand closest to it, so every pictogram is one person's actual hand and not an idealised drawing. It prints
a JavaScript object to paste into SIGNS in server/static/playground.html.

    python scripts/make_sign_glyphs.py
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_TIP = 0, 4, 5, 8, 9, 20
KEEP_ORIENTATION = {"like", "dislike", "fist"}  # up versus down is what thumbs mean; a fist has no clear axis
GESTURES = ["call", "point", "like", "dislike", "stop", "peace", "ok", "fist", "no_gesture"]
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "pinky": (17, 18, 19, 20)}

# Most pointing hands aim at the camera and lose the finger in 2D, so the typical hand makes a poor sign.
# This one was picked by eye from the clearest candidates. Its index finger is drawn on top.
CHOSEN_SAMPLE = {"point": "3dde0aa3"}


def turn(xy: np.ndarray, vector: np.ndarray, target_degrees: float) -> np.ndarray:
    """Rotate the hand so `vector` lies along the target direction (0 = right, 90 = up on screen)."""
    angle = -np.radians(target_degrees) - np.arctan2(vector[1], vector[0])  # screen y runs downward
    c, s = np.cos(angle), np.sin(angle)
    return xy @ np.array([[c, s], [-s, c]])


def normalise(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """True-shape points with the wrist at the origin, as a right hand, turned so the sign reads clearly."""
    points = np.asarray(row.landmarks, dtype=float).reshape(21, 3)
    xy = points[:, :2] * (row.image_width, row.image_height)
    xy -= xy[WRIST]
    if row.handedness == "Left":
        xy[:, 0] *= -1
    xy /= np.linalg.norm(xy[MIDDLE_MCP])
    if row.gesture == "point":
        xy = turn(xy, xy[INDEX_TIP] - xy[INDEX_MCP], 90)
    elif row.gesture == "call":
        xy = turn(xy, xy[PINKY_TIP] - xy[THUMB_TIP], 0)  # thumb and little finger level, like a phone
    elif row.gesture not in KEEP_ORIENTATION:
        xy = turn(xy, xy[MIDDLE_MCP], 90)
    return xy, points[:, 2]


def pick_row(rows: pd.DataFrame, chosen: Optional[str]):
    if chosen:
        return next(row for row in rows.itertuples(index=False) if str(row.sample_id).startswith(chosen))
    rows = rows[rows.annotation_matched]
    shapes = np.stack([normalise(row)[0] for row in rows.itertuples(index=False)])
    median = np.median(shapes, axis=0)
    best = int(np.argmin(np.linalg.norm(shapes - median, axis=2).mean(axis=1)))
    return next(row for i, row in enumerate(rows.itertuples(index=False)) if i == best)


def glyph(gesture: str, rows: pd.DataFrame) -> dict:
    xy, depth = normalise(pick_row(rows, CHOSEN_SAMPLE.get(gesture)))
    low, high = xy.min(axis=0), xy.max(axis=0)
    fitted = (xy - (low + high) / 2) / (high - low).max()  # centred, largest side 1
    order = sorted(FINGERS, key=lambda name: -depth[list(FINGERS[name])].mean())  # far fingers drawn first
    if gesture in CHOSEN_SAMPLE:
        order = [name for name in order if name != "index"] + ["index"]
    return {"p": np.round(fitted, 3).tolist(), "order": order}


def main() -> None:
    data = pd.read_parquet("data/processed/landmarks.parquet")
    data = data[data.landmarks.notna()]
    print(json.dumps({name: glyph(name, data[data.gesture == name]) for name in GESTURES}, separators=(",", ":")))


if __name__ == "__main__":
    main()
