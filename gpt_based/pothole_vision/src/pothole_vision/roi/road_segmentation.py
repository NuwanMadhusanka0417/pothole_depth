"""Optional road segmentation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pothole_vision.models import FrameData


class RoadSegmentation(ABC):
    @abstractmethod
    def segment(self, frame: FrameData) -> np.ndarray:
        """Return boolean road mask same size as frame."""


class StaticROISegmentation(RoadSegmentation):
    """Fallback: use fixed ROI as road mask."""

    def __init__(self, roi_mask_fn) -> None:
        self._roi_mask_fn = roi_mask_fn

    def segment(self, frame: FrameData) -> np.ndarray:
        return self._roi_mask_fn(frame.width, frame.height)
