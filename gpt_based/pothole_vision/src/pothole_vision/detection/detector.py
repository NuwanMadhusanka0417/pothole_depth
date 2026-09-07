"""Pothole detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pothole_vision.models import FrameData, PotholeDetection


class PotholeDetector(ABC):
    @abstractmethod
    def predict(self, frame: FrameData, roi_mask: np.ndarray) -> list[PotholeDetection]:
        """Detect potholes; all coordinates in original image space."""

    def close(self) -> None:
        pass
