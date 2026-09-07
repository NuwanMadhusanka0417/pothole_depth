"""Temporal depth fusion."""

from __future__ import annotations

import numpy as np

from pothole_vision.utils.math import mad_outlier_mask, robust_median


def fuse_depth_observations(observations: list[float]) -> tuple[float, float]:
    """Multi-frame consensus: median with MAD-based uncertainty."""
    arr = np.array(observations, dtype=np.float64)
    mask = mad_outlier_mask(arr)
    filtered = arr[mask]
    if filtered.size == 0:
        return float("nan"), float("nan")
    med = robust_median(filtered)
    mad = np.median(np.abs(filtered - med))
    uncertainty = 1.4826 * mad  # scaled MAD as uncertainty proxy
    return med, float(uncertainty)


def weighted_depth_fusion(
    geometric: np.ndarray,
    neural: np.ndarray,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    w = weights or {"reprojection": 0.3, "metric_depth": 0.3, "road_surface": 0.2, "temporal": 0.2}
    wg = w.get("reprojection", 0.3)
    wn = w.get("metric_depth", 0.3)
    total = wg + wn
    if total == 0:
        return geometric
    return (wg * geometric + wn * neural) / total
