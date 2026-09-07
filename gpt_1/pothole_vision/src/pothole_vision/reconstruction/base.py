"""3D reconstruction backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReconstructionResult:
    camera_poses: list[np.ndarray] = field(default_factory=list)
    point_cloud_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    frame_depths: list[np.ndarray] = field(default_factory=list)
    reprojection_errors: list[float] = field(default_factory=list)
    quality_metrics: dict = field(default_factory=dict)
    scale_status: str = "unknown"


class ReconstructionBackend(ABC):
    @abstractmethod
    def reconstruct(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        camera,
    ) -> ReconstructionResult:
        ...
