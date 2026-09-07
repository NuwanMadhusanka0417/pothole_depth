"""Multi-frame pothole tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pothole_vision.detection.detector import PotholeDetection


class TrackStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    READY_FOR_DEPTH = "READY_FOR_DEPTH"
    FINISHED = "FINISHED"
    REJECTED = "REJECTED"


@dataclass
class Track:
    track_id: str
    start_frame: int
    end_frame: int = 0
    detections: list[PotholeDetection] = field(default_factory=list)
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    mask_history: list[np.ndarray] = field(default_factory=list)
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    quality_history: list[float] = field(default_factory=list)
    best_frames: list[int] = field(default_factory=list)
    status: TrackStatus = TrackStatus.NEW
    rejection_reason: str | None = None

    @property
    def detection_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([d.confidence for d in self.detections]))

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    def latest_detection(self) -> PotholeDetection | None:
        return self.detections[-1] if self.detections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "status": self.status.value,
            "detection_confidence": self.detection_confidence,
            "best_frames": self.best_frames,
            "rejection_reason": self.rejection_reason,
        }
