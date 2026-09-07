"""Fixed normalized ROI and mask operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from pothole_vision.geometry.coordinates import normalized_to_pixel


@dataclass
class ROIConfig:
    road_polygon: list[tuple[float, float]]
    bonnet_y: float | None = None
    overlay_top_y: float | None = 0.08
    manual_exclusions: list[list[tuple[float, float]]] | None = None


def load_roi_config(path: Path) -> ROIConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    polygon = [tuple(p) for p in raw.get("road_polygon", [])]
    exclusions = raw.get("manual_exclusions") or []
    excl_tuples = [[tuple(p) for p in poly] for poly in exclusions]
    return ROIConfig(
        road_polygon=polygon,
        bonnet_y=raw.get("bonnet_y"),
        overlay_top_y=raw.get("overlay_top_y"),
        manual_exclusions=excl_tuples,
    )


def save_roi_config(path: Path, config: ROIConfig) -> None:
    data: dict[str, Any] = {
        "road_polygon": [list(p) for p in config.road_polygon],
        "bonnet_y": config.bonnet_y,
        "overlay_top_y": config.overlay_top_y,
        "manual_exclusions": [[list(p) for p in poly] for poly in (config.manual_exclusions or [])],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


class FixedRoadROI:
    """Static road region mask from normalized polygon — preserves full image size."""

    def __init__(self, config: ROIConfig) -> None:
        self.config = config

    def create_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if not self.config.road_polygon:
            mask[:] = 255
            return mask

        pts = normalized_to_pixel(self.config.road_polygon, width, height)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

        if self.config.bonnet_y is not None:
            bonnet_line = int(self.config.bonnet_y * height)
            mask[bonnet_line:, :] = 0

        if self.config.overlay_top_y is not None:
            overlay_line = int(self.config.overlay_top_y * height)
            mask[:overlay_line, :] = 0

        for exclusion in self.config.manual_exclusions or []:
            ex_pts = normalized_to_pixel(exclusion, width, height)
            cv2.fillPoly(mask, [ex_pts.reshape(-1, 1, 2)], 0)

        return mask

    def zone_mask(self, width: int, height: int, y_start: float, y_end: float = 1.0) -> np.ndarray:
        """Create mask for processing zone based on normalized Y."""
        mask = self.create_mask(width, height)
        y0 = int(y_start * height)
        y1 = int(y_end * height)
        zone = np.zeros((height, width), dtype=np.uint8)
        zone[y0:y1, :] = 255
        return cv2.bitwise_and(mask, zone)

    def apply_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask for visualization — does not crop."""
        masked = image.copy()
        masked[mask == 0] = (masked[mask == 0] * 0.3).astype(np.uint8)
        return masked
