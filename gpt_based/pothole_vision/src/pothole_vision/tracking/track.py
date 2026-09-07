"""Track data structure helpers."""

from __future__ import annotations

from pothole_vision.models import PotholeDetection, Track, TrackStatus


def update_track_from_detection(track: Track, detection: PotholeDetection) -> None:
    track.detections.append(detection)
    track.centroid_history.append(detection.centroid)
    track.mask_history.append(detection.binary_mask)
    track.bbox_history.append(detection.bbox)
    track.end_frame = detection.frame_index
    if track.status == TrackStatus.NEW:
        track.status = TrackStatus.ACTIVE
