"""Confidence and uncertainty scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceScores:
    detection_confidence: float
    tracking_confidence: float
    geometry_confidence: float
    scale_confidence: float
    road_surface_confidence: float
    depth_confidence: float
    overall_confidence: float

    def to_dict(self) -> dict:
        return {
            "detection_confidence": self.detection_confidence,
            "tracking_confidence": self.tracking_confidence,
            "geometry_confidence": self.geometry_confidence,
            "scale_confidence": self.scale_confidence,
            "road_surface_confidence": self.road_surface_confidence,
            "depth_confidence": self.depth_confidence,
            "overall_confidence": self.overall_confidence,
        }


def compute_overall_confidence(
    detection: float,
    tracking: float,
    geometry: float,
    scale: float,
    surface: float,
    depth: float,
) -> ConfidenceScores:
    weights = [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]
    values = [detection, tracking, geometry, scale, surface, depth]
    overall = sum(w * v for w, v in zip(weights, values))
    return ConfidenceScores(
        detection_confidence=detection,
        tracking_confidence=tracking,
        geometry_confidence=geometry,
        scale_confidence=scale,
        road_surface_confidence=surface,
        depth_confidence=depth,
        overall_confidence=overall,
    )
