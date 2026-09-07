"""Pothole detection interfaces and models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PotholeDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in original coords
    polygon: np.ndarray  # Nx2 float in original coords
    binary_mask: np.ndarray  # HxW uint8 in original image size
    confidence: float
    centroid: tuple[float, float]
    area_pixels: int
    frame_index: int = 0
    class_name: str = "pothole"
    zone: str = "B"  # A, B, or C


class PotholeDetector(ABC):
    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray | None = None,
        frame_index: int = 0,
    ) -> list[PotholeDetection]:
        ...
