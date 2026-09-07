"""Math utilities: robust statistics, geometry helpers."""

from __future__ import annotations

import numpy as np


def robust_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def mad_outlier_mask(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Modified Z-score outlier mask using MAD."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-12:
        return np.ones_like(arr, dtype=bool)
    modified_z = 0.6745 * (arr - med) / mad
    return np.abs(modified_z) <= threshold


def robust_median_filtered(values: np.ndarray, mad_threshold: float = 3.5) -> tuple[float, int]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    mask = mad_outlier_mask(arr, mad_threshold)
    filtered = arr[mask]
    if filtered.size == 0:
        return float("nan"), 0
    return float(np.median(filtered)), int(filtered.size)


def percentile_robust_max(values: np.ndarray, percentile: float = 97.5, region_fraction: float = 0.03) -> float:
    """Robust maximum: median of deepest region_fraction of points above percentile."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    cutoff = np.percentile(arr, percentile)
    deep = arr[arr >= cutoff]
    if deep.size == 0:
        return float(np.max(arr))
    k = max(1, int(len(deep) * region_fraction))
    deepest = np.sort(deep)[-k:]
    return float(np.median(deepest))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / (area_a + area_b - inter))


def point_to_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Perpendicular distance from points to plane ax+by+cz+d=0."""
    a, b, c, d = plane
    norm = np.sqrt(a * a + b * b + c * c)
    if norm < 1e-12:
        return np.full(len(points), np.nan)
    return np.abs(points @ np.array([a, b, c]) + d) / norm
