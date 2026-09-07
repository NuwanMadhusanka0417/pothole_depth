"""Fallback relative depth from gradient-based proxy (no external model required)."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.depth.base import DepthEstimator, DepthResult


class FallbackDepthEstimator(DepthEstimator):
    """Simple relative depth proxy — NOT metric. Used when advanced models unavailable."""

    def estimate(self, frames: list[np.ndarray], masks: list[np.ndarray] | None = None) -> DepthResult:
        frame = frames[len(frames) // 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        # Perspective prior: lower in image = closer (relative only)
        depth = (yy / h).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        valid = np.ones((h, w), dtype=np.uint8) * 255
        if masks:
            valid = masks[-1]
        return DepthResult(
            depth_map=depth,
            is_metric=False,
            scale_confidence=0.0,
            model_name="fallback_perspective_prior",
            temporal_consistency=0.5,
            valid_mask=valid,
        )
