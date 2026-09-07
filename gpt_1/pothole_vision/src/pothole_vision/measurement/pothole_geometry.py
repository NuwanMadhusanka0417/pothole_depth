"""Combined pothole geometry measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pothole_vision.depth.temporal_fusion import fuse_depth_observations
from pothole_vision.measurement.depth_measurement import compute_depth_metrics
from pothole_vision.measurement.dimensions import compute_dimensions_2d
from pothole_vision.measurement.volume import estimate_volume_grid


@dataclass
class PotholeMeasurement:
    metric_depth_available: bool
    measurement_status: str  # accepted | approximate | unreliable
    reason: str | None
    maximum_depth_cm: float | None
    mean_depth_cm: float | None
    median_depth_cm: float | None
    depth_uncertainty_cm: float | None
    width_cm: float | None
    length_cm: float | None
    area_m2: float | None
    volume_m3: float | None
    raw_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_depth_available": self.metric_depth_available,
            "measurement_status": self.measurement_status,
            "reason": self.reason,
            "maximum_depth_cm": self.maximum_depth_cm,
            "mean_depth_cm": self.mean_depth_cm,
            "median_depth_cm": self.median_depth_cm,
            "depth_uncertainty_cm": self.depth_uncertainty_cm,
            "width_cm": self.width_cm,
            "length_cm": self.length_cm,
            "area_m2": self.area_m2,
            "volume_m3": self.volume_m3,
        }


def measure_pothole(
    depths_perpendicular: np.ndarray,
    boundary_points: np.ndarray,
    pixel_size_m: float,
    multi_frame_depths: list[float] | None = None,
    scale_confidence: float = 0.0,
    geometry_confidence: float = 0.0,
    surface_confidence: float = 0.0,
    confidence_thresholds: dict | None = None,
) -> PotholeMeasurement:
    cfg = confidence_thresholds or {}
    accept = cfg.get("metric_accept", 0.85)
    approx = cfg.get("approximate_accept", 0.65)

    overall = (scale_confidence + geometry_confidence + surface_confidence) / 3.0

    if overall < approx:
        return PotholeMeasurement(
            metric_depth_available=False,
            measurement_status="unreliable",
            reason="insufficient_geometry_or_scale",
            maximum_depth_cm=None,
            mean_depth_cm=None,
            median_depth_cm=None,
            depth_uncertainty_cm=None,
            width_cm=None,
            length_cm=None,
            area_m2=None,
            volume_m3=None,
            raw_metrics={},
        )

    if scale_confidence < 0.5:
        return PotholeMeasurement(
            metric_depth_available=False,
            measurement_status="unreliable",
            reason="metric_scale_uncertainty",
            maximum_depth_cm=None,
            mean_depth_cm=None,
            median_depth_cm=None,
            depth_uncertainty_cm=None,
            width_cm=None,
            length_cm=None,
            area_m2=None,
            volume_m3=None,
            raw_metrics={},
        )

    metrics = compute_depth_metrics(depths_perpendicular)
    if multi_frame_depths:
        fused, uncertainty = fuse_depth_observations(multi_frame_depths)
        metrics["fused_depth_m"] = fused
        metrics["uncertainty_m"] = uncertainty
    else:
        uncertainty = float(np.std(depths_perpendicular[np.isfinite(depths_perpendicular)]))

    dims = compute_dimensions_2d(boundary_points, pixel_size_m)
    pixel_area = pixel_size_m ** 2
    volume = estimate_volume_grid(depths_perpendicular, pixel_area)

    status = "accepted" if overall >= accept else "approximate"

    return PotholeMeasurement(
        metric_depth_available=True,
        measurement_status=status,
        reason=None,
        maximum_depth_cm=metrics["robust_maximum_depth_m"] * 100,
        mean_depth_cm=metrics["mean_depth_m"] * 100,
        median_depth_cm=metrics["median_depth_m"] * 100,
        depth_uncertainty_cm=uncertainty * 100,
        width_cm=dims["width_m"] * 100,
        length_cm=dims["length_m"] * 100,
        area_m2=dims["area_m2"],
        volume_m3=volume,
        raw_metrics={**metrics, **dims},
    )
