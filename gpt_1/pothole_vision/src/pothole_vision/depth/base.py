"""Depth estimation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthResult:
    depth_map: np.ndarray
    is_metric: bool
    scale_confidence: float
    model_name: str
    temporal_consistency: float
    valid_mask: np.ndarray


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frames: list[np.ndarray], masks: list[np.ndarray] | None = None) -> DepthResult:
        ...
