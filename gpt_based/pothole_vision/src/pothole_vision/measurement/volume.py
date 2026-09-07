"""Volume estimation."""

from __future__ import annotations

import numpy as np


def estimate_volume_grid(depths: np.ndarray, cell_area_m2: float) -> float:
    """Integrate depth over pothole grid cells."""
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.sum(valid) * cell_area_m2)
