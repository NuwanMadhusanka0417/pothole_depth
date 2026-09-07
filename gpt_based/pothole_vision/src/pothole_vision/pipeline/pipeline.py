"""Main processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pothole_vision.detection.yolo_seg import create_detector
from pothole_vision.models import FrameData
from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.roi.fixed_roi import FixedRoadROI, ROIConfig
from pothole_vision.storage.database import ResultsDatabase
from pothole_vision.storage.event_writer import EventWriter
from pothole_vision.storage.results import detection_to_dict, save_detections_json
from pothole_vision.tracking.tracker import PotholeTracker
from pothole_vision.utils.logging import log_tag, setup_logging
from pothole_vision.utils.config import resolve_path
from pothole_vision.video.quality import assess_frame_quality
from pothole_vision.video.reader import VideoReader
from pothole_vision.visualization.overlay import compose_frame


class PotholePipeline:
    """Orchestrates video processing stages."""

    def __init__(self, config: dict, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        setup_logging(
            config.get("logging", {}).get("level", "INFO"),
            config.get("logging", {}).get("structured", True),
        )
        self.roi = FixedRoadROI(config)
        self.detector = create_detector(config)
        self.tracker = PotholeTracker(config)
        self.output_dir = resolve_path(config, "output_dir", self.project_root)
        self.events_dir = resolve_path(config, "events_dir", self.project_root)
        self.cache_dir = resolve_path(config, "cache_dir", self.project_root)

    def run(
        self,
        video_path: Path,
        mode: PipelineMode = PipelineMode.DETECTION,
        start_frame: int = 0,
        max_frames: int | None = None,
        duration_seconds: float | None = None,
        debug: bool = False,
    ) -> dict:
        video_path = Path(video_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        video_id = video_path.stem
        run_dir = self.output_dir / video_id
        run_dir.mkdir(parents=True, exist_ok=True)

        all_detections_json: list[dict] = []
        frame_detections: dict[int, list] = {}

        with VideoReader(video_path) as reader:
            meta = reader.metadata
            log_tag(
                "VIDEO",
                f"{meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.frame_count} frames",
            )

            roi_mask = self.roi.get_static_mask(meta.width, meta.height)
            zones = self.config.get("zones", {})
            depth_zone = self.roi.get_zone_mask(
                meta.width, meta.height,
                zones.get("depth_start_y", 0.74),
                zones.get("depth_end_y", 0.88),
            )

            if self.config.get("output", {}).get("save_roi_debug", True):
                sample = reader.read_frame(max(0, start_frame))
                if sample:
                    roi_vis = self.roi.draw_debug(sample.image, zones)
                    cv2.imwrite(str(run_dir / "roi_debug.jpg"), roi_vis)
                    log_tag("ROI", f"loaded {len(self.roi.config.road_polygon)}-point polygon")

            stride = int(self.config.get("video", {}).get("frame_stride", 1))
            if max_frames is None:
                max_frames = self.config.get("video", {}).get("max_frames")

            writer = None
            if self.config.get("output", {}).get("annotated_video", True):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = run_dir / "annotated.mp4"
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, meta.fps / stride, (meta.width, meta.height)
                )

            event_writer = EventWriter(self.events_dir)
            db = ResultsDatabase(run_dir / "results.db")
            db.insert_video(str(video_path), {
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_seconds": meta.duration_seconds,
            })

            quality_cfg = self.config.get("quality", {})
            frames_iter = reader.iter_frames(
                start_frame=start_frame,
                max_frames=max_frames,
                stride=stride,
                duration_seconds=duration_seconds,
            )

            for frame in tqdm(frames_iter, desc="Processing frames"):
                frame.quality_metrics = assess_frame_quality(frame, roi_mask, quality_cfg)

                if not frame.quality_metrics.usable_for_detection:
                    if writer:
                        roi_vis = self.roi.draw_debug(frame.image, zones)
                        writer.write(roi_vis)
                    continue

                detections = self.detector.predict(frame, roi_mask)
                frame_detections[frame.frame_index] = detections

                for det in detections:
                    log_tag(
                        "DETECT",
                        f"PH candidate frame={frame.frame_index} conf={det.confidence:.2f}",
                    )
                    rec = detection_to_dict(det)
                    rec["timestamp_seconds"] = frame.timestamp_seconds
                    all_detections_json.append(rec)

                    if mode == PipelineMode.DETECTION:
                        event_writer.save_detection_snapshot(
                            self.events_dir, det, frame.image
                        )

                if mode in (PipelineMode.TRACKING, PipelineMode.GEOMETRY, PipelineMode.FULL):
                    self.tracker.update(frame, detections, depth_zone)

                roi_vis = self.roi.draw_debug(frame.image, zones)
                if mode == PipelineMode.DETECTION:
                    annotated = compose_frame(frame.image, roi_vis, detections)
                else:
                    annotated = compose_frame(
                        frame.image, roi_vis, detections,
                        tracks=self.tracker.tracks,
                        frame_index=frame.frame_index,
                    )

                if writer:
                    writer.write(annotated)

            if writer:
                writer.release()

            if self.config.get("output", {}).get("save_detections_json", True):
                save_detections_json(run_dir / "detections.json", all_detections_json)

            if mode in (PipelineMode.TRACKING, PipelineMode.GEOMETRY, PipelineMode.FULL):
                for track in self.tracker.tracks:
                    if track.detections:
                        best = max(track.detections, key=lambda d: d.confidence)
                        idx = track.detections.index(best)
                        event_writer.write_event(
                            track, None,
                            mask=track.mask_history[idx] if track.mask_history else None,
                        )

            db.close()

        log_tag("PIPELINE", f"Complete mode={mode.value} output={run_dir}")
        return {
            "video_id": video_id,
            "output_dir": str(run_dir),
            "num_detections": len(all_detections_json),
            "num_tracks": len(self.tracker.tracks) if mode != PipelineMode.DETECTION else 0,
        }
