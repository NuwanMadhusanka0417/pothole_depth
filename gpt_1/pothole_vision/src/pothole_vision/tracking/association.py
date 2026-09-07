"""Track-detection association."""

from __future__ import annotations

import numpy as np

from pothole_vision.detection.detector import PotholeDetection
from pothole_vision.tracking.track import Track
from pothole_vision.utils.math import bbox_iou, mask_iou


def predict_centroid(track: Track, flow_dx: float = 0, flow_dy: float = 0) -> tuple[float, float]:
    if not track.centroid_history:
        return (0.0, 0.0)
    cx, cy = track.centroid_history[-1]
    return (cx + flow_dx, cy + flow_dy)


def association_cost(
    track: Track,
    detection: PotholeDetection,
    iou_weight: float = 0.5,
    centroid_weight: float = 0.3,
    mask_weight: float = 0.2,
    max_centroid_dist: float = 150.0,
) -> float:
    if not track.bbox_history:
        return 0.0

    miou = mask_iou(track.mask_history[-1], detection.binary_mask)
    biou = bbox_iou(track.bbox_history[-1], detection.bbox)

    pred_cx, pred_cy = predict_centroid(track)
    dx = detection.centroid[0] - pred_cx
    dy = detection.centroid[1] - pred_cy
    dist = np.sqrt(dx * dx + dy * dy)
    centroid_score = max(0.0, 1.0 - dist / max_centroid_dist)

    return iou_weight * biou + mask_weight * miou + centroid_weight * centroid_score


def match_detections_to_tracks(
    tracks: list[Track],
    detections: list[PotholeDetection],
    iou_threshold: float = 0.15,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches, unmatched_track_indices, unmatched_det_indices)."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            cost_matrix[ti, di] = association_cost(track, det)

    matches: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()

    # Greedy assignment by highest cost
    flat_indices = np.argsort(cost_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        ti, di = divmod(int(flat_idx), len(detections))
        if ti in used_t or di in used_d:
            continue
        if cost_matrix[ti, di] < iou_threshold:
            continue
        matches.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    return matches, unmatched_t, unmatched_d
