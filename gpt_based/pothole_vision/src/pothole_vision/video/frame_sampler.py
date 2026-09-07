"""Frame sampling utilities."""

from __future__ import annotations

from pothole_vision.models import FrameData


def select_diverse_frames(
    frames: list[FrameData],
    scores: list[float],
    max_count: int,
    min_frame_gap: int = 3,
) -> list[FrameData]:
    """Select high-quality frames with minimum temporal gap for parallax diversity."""
    if not frames:
        return []
    ranked = sorted(zip(scores, frames), key=lambda x: x[0], reverse=True)
    selected: list[FrameData] = []
    selected_indices: list[int] = []

    for score, frame in ranked:
        if len(selected) >= max_count:
            break
        if any(abs(frame.frame_index - idx) < min_frame_gap for idx in selected_indices):
            continue
        selected.append(frame)
        selected_indices.append(frame.frame_index)

    return sorted(selected, key=lambda f: f.frame_index)
