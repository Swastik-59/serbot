"""
Hand features for GestureGrasp (feature set 2.0).

x/y only, vectorised, no handedness input. HaGRID's annotation landmarks are 2D, so z is not used.
Handedness is not used either: training mirrors hands at random, so the model covers both hands,
and MediaPipe's handedness label (often wrong for a sideways hand) cannot flip the features.
Input points are MediaPipe-normalised (x / width, y / height); multiplying x by the aspect ratio
(width / height) restores true hand geometry before any angle or distance is taken.
"""

from __future__ import annotations

from typing import Union

import numpy as np

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

FEATURE_VERSION_2D = "2.0"

_FINGER_CHAINS = [  # (base, joint, next) triples whose bend angle is measured
    (WRIST, THUMB_CMC, THUMB_MCP), (THUMB_CMC, THUMB_MCP, THUMB_IP), (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP, INDEX_PIP), (INDEX_MCP, INDEX_PIP, INDEX_DIP), (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP, RING_PIP), (RING_MCP, RING_PIP, RING_DIP), (RING_PIP, RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP, PINKY_PIP), (PINKY_MCP, PINKY_PIP, PINKY_DIP), (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]
_TIP_PAIRS = [(a, b) for i, a in enumerate(FINGERTIP_INDICES) for b in FINGERTIP_INDICES[i + 1:]]
_PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def feature_names_2d() -> list[str]:
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    names = [f"lm_{i}_{a}" for i in range(21) for a in "xy"]
    names += [f"up_lm_{i}_{a}" for i in range(21) for a in "xy"]
    names += ["roll_sin", "roll_cos"]
    names += [f"bend_{f}_{j}" for f in fingers for j in ("base", "mid", "end")]
    names += [f"tip_wrist_{f}" for f in fingers]
    names += [f"tip_{fingers[FINGERTIP_INDICES.index(a)]}_{fingers[FINGERTIP_INDICES.index(b)]}" for a, b in _TIP_PAIRS]
    names += [f"ext_ratio_{f}" for f in fingers]
    names += [f"tip_palm_{f}" for f in fingers]
    return names


def extract_features_2d(points: np.ndarray, aspect: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """points: (21, 2) or (N, 21, 2) normalised landmarks. aspect: width / height. Returns (126,) or (N, 126)."""
    p = np.asarray(points, dtype=np.float32)
    single = p.ndim == 2
    if single:
        p = p[None]
    if p.shape[1:] != (21, 2):
        raise ValueError(f"Expected (N, 21, 2) points, got {p.shape}")
    p = p.copy()
    p[:, :, 0] *= np.asarray(aspect, dtype=np.float32).reshape(-1, 1)

    centred = p - p[:, WRIST:WRIST + 1]
    scale = np.linalg.norm(centred[:, MIDDLE_MCP], axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    n = centred / scale[:, None, None]

    # Roll: angle that turns wrist->middle-MCP to point straight up (-y). Kept as sin/cos, so the
    # model still knows orientation (thumbs up vs thumbs down) while "up" coords stay roll-free.
    axis = n[:, MIDDLE_MCP]
    roll = np.arctan2(axis[:, 0], -axis[:, 1])
    c, s = np.cos(roll), np.sin(roll)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    up = np.einsum("nij,nkj->nki", rot, n)

    def dist(a: int, b: int) -> np.ndarray:
        return np.linalg.norm(n[:, a] - n[:, b], axis=1)

    bends = []
    for a, b, cc in _FINGER_CHAINS:
        v1, v2 = n[:, a] - n[:, b], n[:, cc] - n[:, b]
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-7)
        bends.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    tip_wrist = [dist(t, WRIST) for t in FINGERTIP_INDICES]
    tip_pairs = [dist(a, b) for a, b in _TIP_PAIRS]
    ext = [dist(t, WRIST) / (dist(m, WRIST) + 1e-6) for m, t in zip(MCP_INDICES, FINGERTIP_INDICES)]
    palm_centre = n[:, _PALM].mean(1)
    tip_palm = [np.linalg.norm(n[:, t] - palm_centre, axis=1) for t in FINGERTIP_INDICES]

    out = np.concatenate(
        [n.reshape(len(n), -1), up.reshape(len(n), -1), np.stack([np.sin(roll), np.cos(roll)], 1),
         np.stack(bends, 1), np.stack(tip_wrist, 1), np.stack(tip_pairs, 1), np.stack(ext, 1),
         np.stack(tip_palm, 1)],
        axis=1,
    ).astype(np.float32)
    return out[0] if single else out


def augment_points_2d(points: np.ndarray, aspect: np.ndarray, rng: np.random.Generator,
                      aspect_jitter: float = 0.15, rotation_deg: Union[float, np.ndarray] = 15.0,
                      noise: float = 0.012, mirror_prob: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Random mirror, aspect jitter, in-plane rotation, scale and per-joint jitter (tremor-like).

    points (N, 21, 2) normalised; returns new (points, aspect). rotation_deg may be per row (N,).
    Rotation and noise are applied in
    true-geometry space and mapped back, so they never invent a squashed hand.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    asp = np.asarray(aspect, dtype=np.float32).copy()
    n = len(pts)
    true = pts.copy()
    true[:, :, 0] *= asp[:, None]
    wrist = true[:, :1].copy()
    hand = true - wrist

    mirror = rng.random(n) < mirror_prob
    hand[mirror, :, 0] *= -1.0
    limit = np.broadcast_to(np.asarray(rotation_deg, dtype=np.float32), (n,))
    theta = np.deg2rad(rng.uniform(-1.0, 1.0, n) * limit).astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    hand = np.einsum("nij,nkj->nki", rot, hand)
    hand *= rng.uniform(0.9, 1.1, (n, 1, 1)).astype(np.float32)
    size = np.linalg.norm(hand[:, MIDDLE_MCP], axis=1)[:, None, None]
    hand += rng.normal(0.0, 1.0, hand.shape).astype(np.float32) * noise * rng.uniform(0, 1, (n, 1, 1)) * size * 4

    true = hand + wrist
    # Jitter the aspect we *claim*, which squashes geometry the way an unknown image shape would.
    claimed = asp * rng.uniform(1 - aspect_jitter, 1 + aspect_jitter, n).astype(np.float32)
    true[:, :, 0] /= claimed[:, None]
    return true, claimed
