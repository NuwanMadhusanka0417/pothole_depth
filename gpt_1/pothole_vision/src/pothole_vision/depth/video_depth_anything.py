"""Video Depth Anything adapter (optional)."""

from __future__ import annotations

import numpy as np

from pothole_vision.depth.base import DepthEstimator, DepthResult
from pothole_vision.depth.single_frame_depth import FallbackDepthEstimator
from pothole_vision.utils.logging import log_stage


class VideoDepthAnythingEstimator(DepthEstimator):
    """Adapter for Video Depth Anything — falls back if not installed."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._model = None
        self._fallback = FallbackDepthEstimator()
        self._try_load()

    def _try_load(self) -> None:
        try:
            # Placeholder for official Video Depth Anything integration
            # pip install from: https://github.com/DepthAnything/Video-Depth-Anything
            import importlib
            importlib.import_module("video_depth_anything")
            log_stage("DEPTH", "Video Depth Anything module found")
            self._available = True
        except ImportError:
            log_stage("DEPTH", "Video Depth Anything not installed — using fallback")
            self._available = False

    def estimate(self, frames: list[np.ndarray], masks: list[np.ndarray] | None = None) -> DepthResult:
        if not self._available:
            result = self._fallback.estimate(frames, masks)
            result.model_name = "fallback (video_depth_anything unavailable)"
            return result
        # Integration point for official API
        return self._fallback.estimate(frames, masks)
