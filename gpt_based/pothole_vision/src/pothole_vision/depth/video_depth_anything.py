"""Video Depth Anything adapter (optional)."""

from __future__ import annotations

from pothole_vision.depth.single_frame_depth import SingleFrameDepthEstimator
from pothole_vision.models import DepthResult, FrameData
from pothole_vision.utils.logging import log_tag


class VideoDepthAnythingEstimator(SingleFrameDepthEstimator):
    """Optional backend — falls back if not installed."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._available = False
        try:
            # Placeholder for actual VDA import
            self._available = False
        except ImportError:
            self._available = False

    def estimate(self, frames: list[FrameData], masks=None) -> DepthResult:
        if not self._available:
            log_tag("DEPTH", "Video Depth Anything not installed — using fallback")
            return super().estimate(frames, masks)
        raise NotImplementedError("Install Video Depth Anything per README")
