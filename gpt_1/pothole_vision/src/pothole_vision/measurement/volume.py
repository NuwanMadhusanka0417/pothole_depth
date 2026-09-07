"""Volume estimation."""

from __future__ import annotations

import numpy as np


def estimate_volume_grid(depths: np.ndarray, pixel_area_m2: float) -> float:
    """Integrate depth over pothole region."""
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.sum(valid) * pixel_area_m2)
