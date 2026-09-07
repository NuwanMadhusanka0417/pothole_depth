"""Quality-based confidence helpers."""

from __future__ import annotations


def tracking_confidence_from_frames(frame_count: int, min_frames: int = 10) -> float:
    if frame_count < min_frames:
        return frame_count / min_frames * 0.5
    return min(1.0, 0.5 + 0.5 * (frame_count - min_frames) / min_frames)


def geometry_confidence_from_sfm(inlier_ratio: float, parallax_deg: float) -> float:
    return min(1.0, inlier_ratio * 0.7 + min(parallax_deg / 2.0, 1.0) * 0.3)
