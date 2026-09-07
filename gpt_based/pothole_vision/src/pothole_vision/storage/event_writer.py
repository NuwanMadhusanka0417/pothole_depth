"""Per-pothole event folder writer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pothole_vision.models import PotholeDetection, PotholeMeasurement, Track
from pothole_vision.storage.results import measurement_to_dict, NumpyEncoder


class EventWriter:
    def __init__(self, events_dir: Path) -> None:
        self.events_dir = events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        track: Track,
        measurement: PotholeMeasurement | None,
        best_frame_image: np.ndarray | None = None,
        annotated_image: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> Path:
        event_dir = self.events_dir / track.track_id
        event_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "pothole_id": track.track_id,
            "start_frame": track.start_frame,
            "end_frame": track.end_frame,
            "num_observations": len(track.detections),
            "status": track.status.value,
        }
        if track.detections:
            metadata["detection_confidence"] = float(
                np.mean([d.confidence for d in track.detections])
            )

        if measurement:
            metadata.update(measurement_to_dict(measurement))

        with (event_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        if measurement:
            with (event_dir / "measurement.json").open("w", encoding="utf-8") as f:
                json.dump(measurement_to_dict(measurement), f, indent=2)

        if best_frame_image is not None:
            cv2.imwrite(str(event_dir / "best_frame.jpg"), best_frame_image)
        if annotated_image is not None:
            cv2.imwrite(str(event_dir / "annotated_frame.jpg"), annotated_image)
        if mask is not None:
            cv2.imwrite(str(event_dir / "mask.png"), (mask.astype(np.uint8) * 255))

        return event_dir

    @staticmethod
    def save_detection_snapshot(
        events_dir: Path,
        detection: PotholeDetection,
        frame_image: np.ndarray,
        suffix: str = "",
    ) -> Path:
        """Save single-frame detection image for detection-only mode."""
        name = f"frame_{detection.frame_index:06d}{suffix}.jpg"
        out_dir = events_dir / "detection_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        vis = frame_image.copy()
        cv2.polylines(vis, [detection.polygon.astype(np.int32)], True, (0, 0, 255), 2)
        path = out_dir / name
        cv2.imwrite(str(path), vis)
        return path
