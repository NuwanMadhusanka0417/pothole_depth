"""JSON/CSV result serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pothole_vision.models import PotholeDetection, PotholeMeasurement, Track


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def detection_to_dict(det: PotholeDetection) -> dict[str, Any]:
    return {
        "frame_index": det.frame_index,
        "bbox": list(det.bbox),
        "confidence": det.confidence,
        "centroid": list(det.centroid),
        "area_pixels": det.area_pixels,
        "polygon": det.polygon.tolist(),
    }


def save_detections_json(path: Path, all_detections: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, cls=NumpyEncoder)


def measurement_to_dict(m: PotholeMeasurement) -> dict[str, Any]:
    return {
        "pothole_id": m.pothole_id,
        "metric_depth_available": m.metric_depth_available,
        "maximum_depth_cm": m.maximum_depth_cm,
        "mean_depth_cm": m.mean_depth_cm,
        "median_depth_cm": m.median_depth_cm,
        "depth_uncertainty_cm": m.depth_uncertainty_cm,
        "width_cm": m.width_cm,
        "length_cm": m.length_cm,
        "area_m2": m.area_m2,
        "volume_m3": m.volume_m3,
        "detection_confidence": m.detection_confidence,
        "tracking_confidence": m.tracking_confidence,
        "geometry_confidence": m.geometry_confidence,
        "scale_confidence": m.scale_confidence,
        "road_surface_confidence": m.road_surface_confidence,
        "depth_confidence": m.depth_confidence,
        "overall_confidence": m.overall_confidence,
        "measurement_status": m.measurement_status,
        "rejection_reason": m.rejection_reason,
    }


def track_summary(track: Track) -> dict[str, Any]:
    confs = [d.confidence for d in track.detections]
    return {
        "track_id": track.track_id,
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "num_frames": len(track.detections),
        "status": track.status.value,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "best_frames": track.best_frames,
    }
