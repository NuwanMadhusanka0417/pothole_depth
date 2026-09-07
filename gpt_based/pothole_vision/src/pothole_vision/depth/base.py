"""Depth estimator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pothole_vision.models import DepthResult, FrameData


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frames: list[FrameData], masks: list[np.ndarray] | None = None) -> DepthResult:
        pass
