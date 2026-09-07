"""Bundle adjustment placeholder — extensible for future optimization."""

from __future__ import annotations

import numpy as np


def refine_poses(poses: list[np.ndarray], points: np.ndarray) -> list[np.ndarray]:
    """Placeholder: return poses unchanged. Replace with g2o/Ceres/pycolmap BA."""
    return poses
