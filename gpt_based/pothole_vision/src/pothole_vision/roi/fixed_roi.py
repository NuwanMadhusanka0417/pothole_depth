"""Fixed normalized ROI for road region."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from pothole_vision.roi.masks import (
    bonnet_mask,
    combine_masks,
    exclude_polygon,
    overlay_exclusion_mask,
    polygon_to_mask,
)
from pothole_vision.utils.math import normalized_to_pixel, pixel_to_normalized


@dataclass
class ROIConfig:
    road_polygon: list[list[float]]
    bonnet_y: float | None = None
    manual_exclusions: list[list[list[float]]] | None = None
    overlay_exclusion: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ROIConfig:
        return cls(
            road_polygon=data.get("road_polygon", []),
            bonnet_y=data.get("bonnet_y"),
            manual_exclusions=data.get("manual_exclusions", []),
            overlay_exclusion=data.get("overlay_exclusion"),
        )

    def save(self, path: Path) -> None:
        data = {
            "road_polygon": self.road_polygon,
            "bonnet_y": self.bonnet_y,
            "manual_exclusions": self.manual_exclusions or [],
            "overlay_exclusion": self.overlay_exclusion,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> ROIConfig:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


class FixedRoadROI:
    """Static road ROI from normalized polygon coordinates."""

    def __init__(self, config: ROIConfig | dict[str, Any]) -> None:
        if isinstance(config, dict):
            self.config = ROIConfig.from_dict(config)
        else:
            self.config = config

    def get_static_mask(self, width: int, height: int) -> np.ndarray:
        """Build M_static = road AND NOT bonnet AND NOT overlay AND NOT exclusions."""
        if not self.config.road_polygon:
            return np.ones((height, width), dtype=bool)

        poly_px = normalized_to_pixel(self.config.road_polygon, width, height)
        m_road = polygon_to_mask(poly_px, height, width)
        m_bonnet = bonnet_mask(height, width, self.config.bonnet_y)
        m_overlay = overlay_exclusion_mask(height, width, self.config.overlay_exclusion)

        mask = combine_masks(m_road, m_bonnet, m_overlay)

        for exclusion in self.config.manual_exclusions or []:
            if exclusion:
                ex_px = normalized_to_pixel(exclusion, width, height)
                mask = exclude_polygon(mask, ex_px)

        return mask

    def get_zone_mask(
        self, width: int, height: int, y_start: float, y_end: float = 1.0
    ) -> np.ndarray:
        """Horizontal band within static ROI (normalized Y)."""
        static = self.get_static_mask(width, height)
        y0 = int(y_start * height)
        y1 = int(y_end * height)
        band = np.zeros((height, width), dtype=bool)
        band[y0:y1, :] = True
        return static & band

    def draw_debug(self, image: np.ndarray, zones: dict[str, float] | None = None) -> np.ndarray:
        """Draw ROI polygon and optional zone lines on image copy."""
        out = image.copy()
        h, w = out.shape[:2]
        poly_px = normalized_to_pixel(self.config.road_polygon, w, h).astype(np.int32)
        cv2.polylines(out, [poly_px], True, (0, 255, 255), 2)

        overlay = out.copy()
        cv2.fillPoly(overlay, [poly_px], (0, 255, 255))
        out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)

        if zones:
            for name, y_norm in zones.items():
                if y_norm is not None:
                    y_px = int(y_norm * h)
                    color = {"detection_start_y": (255, 128, 0), "tracking_start_y": (0, 128, 255),
                             "depth_start_y": (0, 255, 0), "depth_end_y": (0, 200, 0)}.get(
                        name, (200, 200, 200)
                    )
                    cv2.line(out, (0, y_px), (w, y_px), color, 1)
                    cv2.putText(out, name, (10, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return out
