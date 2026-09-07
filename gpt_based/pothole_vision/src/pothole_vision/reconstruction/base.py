"""Reconstruction backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pothole_vision.geometry.camera import Camera
from pothole_vision.models import FrameData, ReconstructionResult


class ReconstructionBackend(ABC):
    @abstractmethod
    def reconstruct(
        self,
        frames: list[FrameData],
        masks: list[np.ndarray] | None,
        camera: Camera,
        geometry_mask: np.ndarray | None = None,
    ) -> ReconstructionResult:
        pass
