"""Depth measurement with robust statistics."""

from __future__ import annotations

import numpy as np

from pothole_vision.utils.math import deepest_region_median, percentile_robust_max


def compute_depth_stats(
    depths: np.ndarray,
    deepest_percentile: float = 98.0,
    robust_fraction: float = 0.03,
) -> dict:
    clean = depths[np.isfinite(depths) & (depths > 0)]
    if clean.size == 0:
        return {
            "maximum_depth_m": None,
            "mean_depth_m": None,
            "median_depth_m": None,
            "raw_max_m": None,
            "depth_p95": None,
            "depth_p99": None,
        }
    return {
        "maximum_depth_m": deepest_region_median(clean, robust_fraction),
        "mean_depth_m": float(np.mean(clean)),
        "median_depth_m": float(np.median(clean)),
        "raw_max_m": float(np.max(clean)),
        "depth_p95": percentile_robust_max(clean, 95),
        "depth_p99": percentile_robust_max(clean, 99),
    }
