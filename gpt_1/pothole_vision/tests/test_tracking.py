"""Tests for tracking association."""

import numpy as np

from pothole_vision.detection.detector import PotholeDetection
from pothole_vision.tracking.association import association_cost, match_detections_to_tracks
from pothole_vision.tracking.track import Track
from pothole_vision.utils.math import bbox_iou, mask_iou


def _make_det(cx, cy, size=50, frame=0):
    mask = np.zeros((480, 640), dtype=np.uint8)
    x, y = int(cx), int(cy)
    mask[y - 10:y + 10, x - 10:x + 10] = 255
    return PotholeDetection(
        bbox=(x - 10, y - 10, x + 10, y + 10),
        polygon=np.array([[x - 10, y - 10], [x + 10, y + 10]]),
        binary_mask=mask,
        confidence=0.9,
        centroid=(cx, cy),
        area_pixels=int(mask.sum() / 255),
        frame_index=frame,
    )


def test_mask_iou_identical():
    mask = np.ones((100, 100), dtype=np.uint8)
    assert mask_iou(mask, mask) == 1.0


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_track_association():
    track = Track(track_id="PH_000001", start_frame=0)
    det1 = _make_det(100, 200, frame=0)
    track.detections.append(det1)
    track.centroid_history.append(det1.centroid)
    track.mask_history.append(det1.binary_mask)
    track.bbox_history.append(det1.bbox)

    det2 = _make_det(105, 205, frame=1)
    matches, unmatched_t, unmatched_d = match_detections_to_tracks([track], [det2])
    assert len(matches) == 1
