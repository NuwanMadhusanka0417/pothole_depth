"""Metric scale estimation from depth/SfM cues."""

from __future__ import annotations

import numpy as np

from pothole_vision.models import ScaleEstimate
from pothole_vision.utils.math import robust_median_with_mad


def estimate_scale(
    z_sfm: np.ndarray,
    z_metric: np.ndarray,
    pothole_mask: np.ndarray | None = None,
    config: dict | None = None,
) -> ScaleEstimate:
    cfg = config or {}
    mad_threshold = float(cfg.get("mad_threshold", 3.5))
    min_points = int(cfg.get("min_points", 50))
    exclude_pothole = cfg.get("exclude_pothole_pixels", True)

    valid = np.isfinite(z_sfm) & np.isfinite(z_metric) & (z_sfm > 1e-6) & (z_metric > 0)
    if exclude_pothole and pothole_mask is not None:
        valid &= ~pothole_mask

    if valid.sum() < min_points:
        return ScaleEstimate(
            scale=float("nan"),
            confidence=0.0,
            number_of_points=int(valid.sum()),
            dispersion=0.0,
            source="insufficient_points",
        )

    ratios = z_metric[valid] / z_sfm[valid]
    scale, dispersion, n = robust_median_with_mad(ratios, mad_threshold)

    if n < min_points or not np.isfinite(scale):
        confidence = 0.0
    else:
        cv = dispersion / max(abs(scale), 1e-6)
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    return ScaleEstimate(
        scale=scale,
        confidence=confidence,
        number_of_points=n,
        dispersion=dispersion,
        source="robust_median",
    )
