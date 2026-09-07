"""Multi-frame pothole tracker."""

from __future__ import annotations

import numpy as np

from pothole_vision.models import FrameData, PotholeDetection, Track, TrackStatus
from pothole_vision.tracking.association import association_cost, mask_iou
from pothole_vision.tracking.track import update_track_from_detection
from pothole_vision.utils.logging import log_tag


class PotholeTracker:
    """Associate detections across frames using mask/bbox/centroid cues."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("tracking", config)
        self.max_age = int(cfg.get("max_age", 30))
        self.min_track_frames = int(cfg.get("min_track_frames", 3))
        self.mask_iou_threshold = float(cfg.get("mask_iou_threshold", 0.15))
        self.centroid_max_dist = float(cfg.get("centroid_max_distance_ratio", 0.15))
        self.min_ready_frames = int(cfg.get("min_frames_ready_for_depth", 10))
        self._tracks: list[Track] = []
        self._next_id = 1
        self._missed: dict[str, int] = {}

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def _new_track_id(self) -> str:
        tid = f"PH_{self._next_id:06d}"
        self._next_id += 1
        return tid

    def update(
        self,
        frame: FrameData,
        detections: list[PotholeDetection],
        depth_zone_mask: np.ndarray | None = None,
    ) -> list[Track]:
        diag = float(np.hypot(frame.width, frame.height))
        matched_tracks: set[str] = set()
        active = [t for t in self._tracks if t.status not in (TrackStatus.FINISHED, TrackStatus.REJECTED)]

        for det in detections:
            best_track: Track | None = None
            best_cost = 1.0

            for track in active:
                if track.track_id in matched_tracks:
                    continue
                if not track.mask_history:
                    continue
                cost = association_cost(
                    det.binary_mask,
                    det.bbox,
                    det.centroid,
                    track.mask_history[-1],
                    track.bbox_history[-1],
                    track.centroid_history[-1],
                    diag,
                )
                miou = mask_iou(det.binary_mask, track.mask_history[-1])
                if cost < best_cost and miou >= self.mask_iou_threshold:
                    best_cost = cost
                    best_track = track

            if best_track is not None:
                update_track_from_detection(best_track, det)
                matched_tracks.add(best_track.track_id)
                self._missed[best_track.track_id] = 0
                self._update_status(best_track, depth_zone_mask)
            else:
                tid = self._new_track_id()
                track = Track(track_id=tid, start_frame=det.frame_index)
                update_track_from_detection(track, det)
                self._tracks.append(track)
                matched_tracks.add(tid)
                self._missed[tid] = 0
                log_tag("TRACK", f"created {tid} frame={det.frame_index}")

        for track in active:
            if track.track_id not in matched_tracks:
                self._missed[track.track_id] = self._missed.get(track.track_id, 0) + 1
                if self._missed[track.track_id] > self.max_age:
                    track.status = TrackStatus.FINISHED
                    log_tag("TRACK", f"{track.track_id} finished age={self.max_age}")

        return self._tracks

    def _update_status(self, track: Track, depth_zone_mask: np.ndarray | None) -> None:
        n = len(track.detections)
        if n >= self.min_ready_frames and track.status == TrackStatus.ACTIVE:
            if depth_zone_mask is not None and track.mask_history:
                in_zone = np.any(track.mask_history[-1] & depth_zone_mask)
                if in_zone:
                    track.status = TrackStatus.READY_FOR_DEPTH
            elif n >= self.min_ready_frames * 2:
                track.status = TrackStatus.READY_FOR_DEPTH

    def get_active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status not in (TrackStatus.FINISHED, TrackStatus.REJECTED)]
