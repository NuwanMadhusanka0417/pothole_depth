"""Frame overlay visualization."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.models import PotholeDetection, PotholeMeasurement, Track


def draw_detections(
    image: np.ndarray,
    detections: list[PotholeDetection],
    track_ids: dict[int, str] | None = None,
) -> np.ndarray:
    out = image.copy()
    for i, det in enumerate(detections):
        color = (0, 0, 255)
        cv2.polylines(out, [det.polygon.astype(np.int32)], True, color, 2)
        x1, y1, x2, y2 = det.bbox
        label = track_ids.get(i, "") if track_ids else ""
        text = f"{label} {det.confidence:.2f}".strip()
        cv2.putText(out, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def draw_roi_and_zones(
    image: np.ndarray,
    roi_overlay: np.ndarray,
    show_zones: bool = True,
) -> np.ndarray:
    return roi_overlay


def draw_tracks(
    image: np.ndarray,
    frame_index: int,
    tracks: list[Track],
    measurements: dict[str, PotholeMeasurement] | None = None,
) -> np.ndarray:
    out = image.copy()
    for track in tracks:
        det_idx = None
        for i, d in enumerate(track.detections):
            if d.frame_index == frame_index:
                det_idx = i
                break
        if det_idx is None:
            continue
        det = track.detections[det_idx]
        color = (0, 255, 0) if track.status.value != "REJECTED" else (128, 128, 128)
        cv2.polylines(out, [det.polygon.astype(np.int32)], True, color, 2)

        meas = (measurements or {}).get(track.track_id)
        lines = [track.track_id, f"status: {track.status.value}", f"conf: {det.confidence:.2f}"]
        if meas and meas.metric_depth_available and meas.maximum_depth_cm is not None:
            lines.append(f"Depth: {meas.maximum_depth_cm:.1f} ± {meas.depth_uncertainty_cm or 0:.1f} cm")
            if meas.width_cm:
                lines.append(f"W: {meas.width_cm:.0f} cm L: {meas.length_cm:.0f} cm")
            lines.append(f"Confidence: {meas.overall_confidence * 100:.0f}%")
        elif meas and meas.measurement_status == "unreliable":
            lines.append("Depth: unreliable")
            if meas.rejection_reason:
                lines.append(f"Reason: {meas.rejection_reason}")

        x1, y1 = det.bbox[0], det.bbox[1]
        for j, line in enumerate(lines):
            cv2.putText(
                out, line, (x1, max(y1 - 5 - j * 18, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )
    return out


def compose_frame(
    image: np.ndarray,
    roi_debug: np.ndarray | None,
    detections: list[PotholeDetection],
    tracks: list[Track] | None = None,
    frame_index: int = 0,
    measurements: dict[str, PotholeMeasurement] | None = None,
) -> np.ndarray:
    base = roi_debug if roi_debug is not None else image.copy()
    if tracks:
        return draw_tracks(base, frame_index, tracks, measurements)
    return draw_detections(base, detections)
