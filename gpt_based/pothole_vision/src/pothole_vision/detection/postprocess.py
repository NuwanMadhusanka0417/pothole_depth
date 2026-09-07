"""Detection post-processing."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.models import PotholeDetection


def filter_detections(
    detections: list[PotholeDetection],
    roi_mask: np.ndarray,
    min_area: int = 80,
    min_mask_ratio: float = 0.3,
) -> list[PotholeDetection]:
    """Filter by ROI overlap and minimum area."""
    filtered = []
    for det in detections:
        in_roi = det.binary_mask & roi_mask
        area = int(np.sum(in_roi))
        if area < min_area:
            continue
        ratio = area / max(det.area_pixels, 1)
        if ratio < min_mask_ratio:
            continue
        det.binary_mask = in_roi
        det.area_pixels = area
        filtered.append(det)
    return filtered


def refine_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological cleanup of segmentation mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m.astype(bool)
