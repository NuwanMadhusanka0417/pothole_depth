"""Pothole multi-frame tracker."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.detection.detector import PotholeDetection
from pothole_vision.tracking.association import match_detections_to_tracks
from pothole_vision.tracking.track import Track, TrackStatus
from pothole_vision.utils.logging import log_stage


class PotholeTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_track_frames: int = 5,
        min_frames_for_depth: int = 10,
        max_best_frames: int = 20,
        iou_threshold: float = 0.15,
        depth_zone_y: float = 0.74,
        image_height: int = 1080,
    ) -> None:
        self.max_age = max_age
        self.min_track_frames = min_track_frames
        self.min_frames_for_depth = min_frames_for_depth
        self.max_best_frames = max_best_frames
        self.iou_threshold = iou_threshold
        self.depth_zone_y = depth_zone_y
        self.image_height = image_height
        self.tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self._next_id = 1
        self._miss_counts: dict[str, int] = {}

    def _new_id(self) -> str:
        tid = f"PH_{self._next_id:06d}"
        self._next_id += 1
        return tid

    def update(self, frame_index: int, detections: list[PotholeDetection]) -> list[Track]:
        active = [t for t in self.tracks if t.status not in (TrackStatus.FINISHED, TrackStatus.REJECTED)]
        matches, unmatched_t, unmatched_d = match_detections_to_tracks(
            active, detections, self.iou_threshold
        )

        for ti, di in matches:
            track = active[ti]
            det = detections[di]
            self._update_track(track, det, frame_index)

        for ti in unmatched_t:
            track = active[ti]
            self._miss_counts[track.track_id] = self._miss_counts.get(track.track_id, 0) + 1
            if self._miss_counts[track.track_id] > self.max_age:
                self._finalize_track(track, frame_index)

        for di in unmatched_d:
            det = detections[di]
            track = Track(track_id=self._new_id(), start_frame=frame_index)
            self._update_track(track, det, frame_index)
            track.status = TrackStatus.NEW
            self.tracks.append(track)
            log_stage("TRACK", f"created {track.track_id}")

        return self.tracks

    def _update_track(self, track: Track, det: PotholeDetection, frame_index: int) -> None:
        track.detections.append(det)
        track.centroid_history.append(det.centroid)
        track.mask_history.append(det.binary_mask)
        track.bbox_history.append(det.bbox)
        track.end_frame = frame_index
        self._miss_counts[track.track_id] = 0

        if det.centroid[1] / self.image_height >= self.depth_zone_y:
            if track.frame_count >= self.min_frames_for_depth:
                track.status = TrackStatus.READY_FOR_DEPTH
            else:
                track.status = TrackStatus.ACTIVE
        elif track.frame_count >= self.min_track_frames:
            track.status = TrackStatus.ACTIVE
        else:
            track.status = TrackStatus.NEW

    def _finalize_track(self, track: Track, frame_index: int) -> None:
        track.end_frame = frame_index
        if track.frame_count < self.min_track_frames:
            track.status = TrackStatus.REJECTED
            track.rejection_reason = "too_few_frames"
        else:
            track.status = TrackStatus.FINISHED
            self._select_best_frames(track)
        self.finished_tracks.append(track)
        self.tracks = [t for t in self.tracks if t.track_id != track.track_id]
        log_stage("TRACK", f"{track.track_id} finalized status={track.status.value} frames={track.frame_count}")

    def _select_best_frames(self, track: Track) -> None:
        scores = []
        for i, det in enumerate(track.detections):
            sharpness = track.quality_history[i] if i < len(track.quality_history) else 100.0
            zone_bonus = 2.0 if det.centroid[1] / self.image_height >= self.depth_zone_y else 1.0
            scores.append((det.confidence * zone_bonus + sharpness * 0.001, det.frame_index))
        scores.sort(reverse=True)
        selected = []
        for _, fidx in scores:
            if len(selected) >= self.max_best_frames:
                break
            if not selected or all(abs(fidx - s) >= 3 for s in selected):
                selected.append(fidx)
        track.best_frames = sorted(selected)

    def finalize_all(self, frame_index: int) -> None:
        for track in list(self.tracks):
            self._finalize_track(track, frame_index)

    @property
    def all_tracks(self) -> list[Track]:
        return self.tracks + self.finished_tracks
