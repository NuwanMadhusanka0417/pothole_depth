"""Single-frame depth fallback (relative, not metric)."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.depth.base import DepthEstimator
from pothole_vision.models import DepthResult, FrameData
from pothole_vision.utils.logging import log_tag


class SingleFrameDepthEstimator(DepthEstimator):
    """Fallback using gradient-based pseudo-depth. NOT metric."""

    def estimate(self, frames: list[FrameData], masks: list[np.ndarray] | None = None) -> DepthResult:
        if not frames:
            raise ValueError("No frames provided")
        frame = frames[len(frames) // 2]
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Simple relative depth proxy: lower in image = closer (dashcam assumption)
        h, w = gray.shape
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        depth = y_coords * np.ones((h, w), dtype=np.float32)
        depth = depth * (1.0 + 0.1 * (128 - gray) / 128.0)
        valid = np.ones((h, w), dtype=bool)
        log_tag("DEPTH", "single_frame relative depth (NOT metric)")
        return DepthResult(
            depth_map=depth,
            is_metric=False,
            scale_confidence=0.0,
            model_name="gradient_fallback",
            temporal_consistency=0.0,
            valid_mask=valid,
        )
