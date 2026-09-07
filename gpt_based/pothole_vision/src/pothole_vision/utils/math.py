"""Math utilities: robust statistics, geometry helpers."""

from __future__ import annotations

import numpy as np


def robust_median(values: np.ndarray) -> float:
    """Median of 1D array, ignoring NaN."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.median(clean))


def mad_outlier_mask(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Return boolean mask of inliers using median absolute deviation."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.zeros_like(values, dtype=bool)
    med = np.median(clean)
    mad = np.median(np.abs(clean - med))
    if mad < 1e-12:
        return np.isfinite(values)
    modified_z = 0.6745 * (values - med) / mad
    return np.abs(modified_z) < threshold


def robust_median_with_mad(values: np.ndarray, mad_threshold: float = 3.5) -> tuple[float, float, int]:
    """Robust median with MAD-based outlier rejection."""
    values = np.asarray(values, dtype=np.float64)
    inliers = mad_outlier_mask(values, mad_threshold)
    inlier_vals = values[inliers & np.isfinite(values)]
    if inlier_vals.size == 0:
        return float("nan"), 0.0, 0
    med = float(np.median(inlier_vals))
    dispersion = float(np.std(inlier_vals)) if inlier_vals.size > 1 else 0.0
    return med, dispersion, int(inlier_vals.size)


def normalized_to_pixel(
    points: list[list[float]], width: int, height: int
) -> np.ndarray:
    """Convert normalized [x,y] points to pixel coordinates."""
    arr = np.array(points, dtype=np.float64)
    px = arr.copy()
    px[:, 0] *= width
    px[:, 1] *= height
    return px


def pixel_to_normalized(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert pixel coordinates to normalized [0,1]."""
    arr = np.asarray(points, dtype=np.float64).copy()
    arr[:, 0] /= max(width, 1)
    arr[:, 1] /= max(height, 1)
    return arr


def polygon_area(points: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def percentile_robust_max(values: np.ndarray, percentile: float = 98.0) -> float:
    """Robust maximum using high percentile instead of raw max."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.percentile(clean, percentile))


def deepest_region_median(values: np.ndarray, fraction: float = 0.03) -> float:
    """Median of deepest fraction of values."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    n = max(1, int(len(clean) * fraction))
    deepest = np.sort(clean)[-n:]
    return float(np.median(deepest))
