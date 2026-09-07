"""Optional semantic road segmentation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RoadSegmentation(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return binary road mask in original image coordinates."""
        ...


class StaticROISegmentation(RoadSegmentation):
    """Fallback: use provided static mask."""

    def __init__(self, static_mask: np.ndarray) -> None:
        self.static_mask = static_mask

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.static_mask.shape[:2] != (h, w):
            import cv2
            return cv2.resize(self.static_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return self.static_mask
