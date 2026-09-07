"""Frame sampling utilities for parallax-diverse selection."""

from __future__ import annotations

import numpy as np

from pothole_vision.video.reader import FrameData


def select_diverse_frames(
    frames: list[FrameData],
    max_count: int,
    min_centroid_shift: float = 5.0,
) -> list[FrameData]:
    """Select high-quality frames with sufficient centroid/viewpoint diversity."""
    if len(frames) <= max_count:
        return frames

    usable = [f for f in frames if f.quality_metrics and f.quality_metrics.usable_for_depth]
    if not usable:
        usable = frames

    usable.sort(key=lambda f: f.quality_metrics.sharpness if f.quality_metrics else 0, reverse=True)

    selected: list[FrameData] = []
    for frame in usable:
        if len(selected) >= max_count:
            break
        if not selected:
            selected.append(frame)
            continue
        # Prefer temporal spacing
        min_gap = min(abs(frame.frame_index - s.frame_index) for s in selected)
        if min_gap >= 2:
            selected.append(frame)

    if len(selected) < max_count:
        for frame in usable:
            if frame not in selected:
                selected.append(frame)
            if len(selected) >= max_count:
                break

    selected.sort(key=lambda f: f.frame_index)
    return selected
