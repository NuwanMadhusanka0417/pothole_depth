"""Confidence scoring helpers for the hybrid pipeline.

The hybrid pipeline produces *several* sources of confidence information:

  * geometric per-frame consistency,
  * SfM scale-recovery MAD,
  * agreement between geometric and SfM depths.

These are folded into a single number in [0, 1] using simple, easy-to-
explain operators. Keep them dumb on purpose - the estimator config
file controls weights.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def consistency_confidence(values: Iterable[float], *, eps: float = 1e-6) -> float:
    """Return ``1 / (1 + relative_std)`` for a set of measurements.

    Returns 0.0 if the input is empty / non-finite / mean ~ 0.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    m = float(np.mean(arr))
    if abs(m) < eps:
        return 0.0
    rel_std = float(np.std(arr) / abs(m))
    return float(np.clip(1.0 / (1.0 + rel_std), 0.0, 1.0))


def agreement_confidence(d_a: float, d_b: float, *, eps: float = 1e-6) -> float:
    """Confidence based on the relative gap between two depth estimates.

    Maps a 0% disagreement -> 1.0 and a 100% disagreement -> 0.5.
    """
    if not (np.isfinite(d_a) and np.isfinite(d_b)):
        return 0.0
    mean = 0.5 * (abs(d_a) + abs(d_b))
    if mean < eps:
        return 0.0
    rel = abs(d_a - d_b) / mean
    return float(np.clip(1.0 / (1.0 + rel), 0.0, 1.0))


def fuse_confidences(
    geo_conf: float,
    sfm_conf: float,
    agreement: float,
    *,
    geometric_weight: float = 0.55,
    sfm_weight: float = 0.45,
) -> float:
    """Combine geometric / SfM / agreement scores into a single value.

    The weights mirror :file:`pipeline.yaml`. Agreement multiplies the
    weighted mean so that disagreement always *reduces* confidence.
    """
    geo_conf = float(np.clip(geo_conf, 0.0, 1.0))
    sfm_conf = float(np.clip(sfm_conf, 0.0, 1.0))
    agreement = float(np.clip(agreement, 0.0, 1.0))

    total_w = geometric_weight + sfm_weight
    if total_w <= 0:
        return 0.0
    base = (geometric_weight * geo_conf + sfm_weight * sfm_conf) / total_w
    return float(np.clip(base * agreement, 0.0, 1.0))
