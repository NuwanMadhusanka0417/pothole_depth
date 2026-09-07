"""Detection-track association metrics."""

from __future__ import annotations

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def bbox_iou(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def centroid_distance(
    c_a: tuple[float, float], c_b: tuple[float, float], diag: float
) -> float:
    dx = c_a[0] - c_b[0]
    dy = c_a[1] - c_b[1]
    return float(np.hypot(dx, dy) / max(diag, 1.0))


def association_cost(
    detection_mask: np.ndarray,
    detection_bbox: tuple[int, int, int, int],
    detection_centroid: tuple[float, float],
    track_mask: np.ndarray,
    track_bbox: tuple[int, int, int, int],
    track_centroid: tuple[float, float],
    image_diag: float,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Lower is better. Returns cost in [0, 1] approximately."""
    miou = mask_iou(detection_mask, track_mask)
    biou = bbox_iou(detection_bbox, track_bbox)
    cdist = centroid_distance(detection_centroid, track_centroid, image_diag)
    score = weights[0] * miou + weights[1] * biou + weights[2] * (1.0 - cdist)
    return 1.0 - score
