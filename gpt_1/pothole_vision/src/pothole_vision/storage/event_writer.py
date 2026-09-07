"""Per-pothole event folder writer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pothole_vision.reconstruction.point_cloud import save_ply
from pothole_vision.tracking.track import Track
from pothole_vision.visualization.depth_map import depth_to_colormap


class EventWriter:
    def __init__(self, events_dir: Path) -> None:
        self.events_dir = events_dir
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        track: Track,
        metadata: dict[str, Any],
        best_frame: np.ndarray | None = None,
        annotated_frame: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        depth_map: np.ndarray | None = None,
        point_cloud: np.ndarray | None = None,
        point_colors: np.ndarray | None = None,
        measurement: dict | None = None,
    ) -> Path:
        event_dir = self.events_dir / track.track_id
        event_dir.mkdir(parents=True, exist_ok=True)

        meta_path = event_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if measurement:
            with (event_dir / "measurement.json").open("w", encoding="utf-8") as f:
                json.dump(measurement, f, indent=2)

        if best_frame is not None:
            cv2.imwrite(str(event_dir / "best_frame.jpg"), best_frame)
        if annotated_frame is not None:
            cv2.imwrite(str(event_dir / "annotated_frame.jpg"), annotated_frame)
        if mask is not None:
            cv2.imwrite(str(event_dir / "mask.png"), mask)
        if depth_map is not None:
            cv2.imwrite(str(event_dir / "depth_map.png"), depth_to_colormap(depth_map))
        if point_cloud is not None and len(point_cloud) > 0:
            save_ply(event_dir / "point_cloud.ply", point_cloud, point_colors)

        return event_dir

    def write_event_clip(self, event_dir: Path, frames: list[np.ndarray], fps: float) -> None:
        if not frames:
            return
        h, w = frames[0].shape[:2]
        out_path = event_dir / "event_clip.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            writer.write(f)
        writer.release()
