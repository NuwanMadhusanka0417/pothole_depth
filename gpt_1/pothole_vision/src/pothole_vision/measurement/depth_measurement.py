"""Pothole depth measurement."""

from __future__ import annotations

import numpy as np

from pothole_vision.utils.math import percentile_robust_max, robust_median


def compute_depth_metrics(
    depths: np.ndarray,
    deepest_percentile: float = 97.5,
    deepest_region_fraction: float = 0.03,
) -> dict[str, float]:
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        return {
            "maximum_depth_m": float("nan"),
            "robust_maximum_depth_m": float("nan"),
            "mean_depth_m": float("nan"),
            "median_depth_m": float("nan"),
            "depth_p95_m": float("nan"),
            "depth_p99_m": float("nan"),
        }

    return {
        "maximum_depth_m": float(np.max(valid)),
        "robust_maximum_depth_m": percentile_robust_max(
            valid, deepest_percentile, deepest_region_fraction
        ),
        "mean_depth_m": float(np.mean(valid)),
        "median_depth_m": robust_median(valid),
        "depth_p95_m": float(np.percentile(valid, 95)),
        "depth_p99_m": float(np.percentile(valid, 99)),
    }
