"""Metric scale estimation from monocular SfM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pothole_vision.utils.math import mad_outlier_mask, robust_median


@dataclass
class ScaleEstimate:
    scale: float
    confidence: float
    number_of_points: int
    dispersion: float
    source: str


def estimate_scale(
    sfm_depths: np.ndarray,
    metric_depths: np.ndarray,
    pothole_mask: np.ndarray | None = None,
) -> ScaleEstimate:
    """Robust scale s = Z_metric / Z_sfm on road points (exclude pothole region)."""
    z_sfm = sfm_depths.ravel()
    z_met = metric_depths.ravel()

    valid = np.isfinite(z_sfm) & np.isfinite(z_met) & (z_sfm > 1e-6) & (z_met > 1e-6)
    if pothole_mask is not None:
        pm = pothole_mask.ravel().astype(bool)
        valid &= ~pm

    if valid.sum() < 10:
        return ScaleEstimate(
            scale=float("nan"), confidence=0.0, number_of_points=0,
            dispersion=float("inf"), source="insufficient_points",
        )

    ratios = z_met[valid] / z_sfm[valid]
    outlier_mask = mad_outlier_mask(ratios)
    filtered = ratios[outlier_mask]

    if filtered.size < 5:
        return ScaleEstimate(
            scale=float("nan"), confidence=0.0, number_of_points=int(filtered.size),
            dispersion=float("inf"), source="insufficient_inliers",
        )

    scale = robust_median(filtered)
    dispersion = float(np.std(filtered))
    confidence = max(0.0, min(1.0, 1.0 - dispersion / (scale + 1e-6)))

    return ScaleEstimate(
        scale=float(scale),
        confidence=float(confidence),
        number_of_points=int(filtered.size),
        dispersion=dispersion,
        source="robust_median",
    )
