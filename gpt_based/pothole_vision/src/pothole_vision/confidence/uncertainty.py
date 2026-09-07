"""Uncertainty estimation."""

from __future__ import annotations

import numpy as np

from pothole_vision.utils.math import robust_median, mad_outlier_mask


def multi_frame_consensus(depths_cm: list[float], mad_threshold: float = 3.5) -> tuple[float, float]:
    """Robust fused depth from multiple observations."""
    arr = np.array(depths_cm, dtype=np.float64)
    inliers = mad_outlier_mask(arr, mad_threshold)
    clean = arr[inliers & np.isfinite(arr)]
    if clean.size == 0:
        return float("nan"), float("nan")
    med = robust_median(clean)
    unc = float(np.std(clean)) if clean.size > 1 else float("nan")
    return med, unc


def fuse_confidence(*scores: float, depth_confidence: float = 0.0) -> float:
    weights = [0.15, 0.15, 0.25, 0.2, 0.1, 0.15]
    values = list(scores) + [depth_confidence]
    values = values[: len(weights)]
    while len(values) < len(weights):
        values.append(0.0)
    return float(np.clip(np.dot(weights, values), 0, 1))


def should_reject_measurement(
    geometry_confidence: float,
    scale_confidence: float,
    num_points: int,
    config: dict | None = None,
) -> tuple[bool, str | None]:
    if geometry_confidence < 0.3:
        return True, "insufficient_geometry"
    if scale_confidence < 0.4:
        return True, "metric_scale_uncertainty"
    if num_points < 20:
        return True, "insufficient_point_cloud"
    return False, None
