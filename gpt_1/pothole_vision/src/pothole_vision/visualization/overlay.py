"""Video and frame overlay visualization."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.detection.detector import PotholeDetection
from pothole_vision.tracking.track import Track


def draw_roi_polygon(image: np.ndarray, polygon_norm: list[tuple[float, float]], color=(0, 255, 255)) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_norm], dtype=np.int32)
    cv2.polylines(out, [pts], True, color, 2)
    return out


def draw_zone_lines(image: np.ndarray, zones: dict) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for key, color in [
        ("detection_start_y", (100, 100, 255)),
        ("tracking_start_y", (100, 255, 100)),
        ("depth_start_y", (255, 100, 100)),
        ("depth_end_y", (255, 100, 100)),
    ]:
        if key in zones:
            y = int(zones[key] * h)
            cv2.line(out, (0, y), (w, y), color, 1)
            cv2.putText(out, key, (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def draw_detection(image: np.ndarray, det: PotholeDetection, track_id: str | None = None) -> np.ndarray:
    out = image.copy()
    if det.polygon is not None and len(det.polygon) > 0:
        pts = det.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    label = track_id or f"{det.confidence:.2f}"
    cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def draw_track_overlay(
    image: np.ndarray,
    track: Track,
    measurement: dict | None = None,
) -> np.ndarray:
    out = image.copy()
    det = track.latest_detection()
    if det is None:
        return out
    out = draw_detection(out, det, track.track_id)

    y = 30
    lines = [f"{track.track_id} [{track.status.value}]"]
    if measurement:
        if measurement.get("metric_depth_available"):
            lines.append(f"Depth: {measurement.get('maximum_depth_cm', 0):.1f} cm")
            lines.append(f"Conf: {measurement.get('overall_confidence', 0)*100:.0f}%")
        else:
            lines.append("Depth: unreliable")
            lines.append(f"Reason: {measurement.get('reason', 'unknown')}")

    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    return out
