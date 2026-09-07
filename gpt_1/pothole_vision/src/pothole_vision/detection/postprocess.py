"""Detection post-processing."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.detection.detector import PotholeDetection


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_factor * peri, True)
    return approx.reshape(-1, 2).astype(np.float64)


def filter_detections_by_roi(
    detections: list[PotholeDetection],
    roi_mask: np.ndarray,
    min_visible_ratio: float = 0.5,
) -> list[PotholeDetection]:
    filtered = []
    for det in detections:
        masked = cv2.bitwise_and(det.binary_mask, roi_mask)
        visible = int(np.sum(masked > 0))
        if det.area_pixels == 0:
            continue
        if visible / det.area_pixels >= min_visible_ratio:
            det.area_pixels = visible
            det.binary_mask = masked
            filtered.append(det)
    return filtered


def assign_zone(centroid_y: float, height: int, zones: dict) -> str:
    ny = centroid_y / height
    if ny >= zones.get("depth_start_y", 0.74):
        return "C"
    if ny >= zones.get("tracking_start_y", 0.69):
        return "B"
    if ny >= zones.get("detection_start_y", 0.64):
        return "A"
    return "A"
