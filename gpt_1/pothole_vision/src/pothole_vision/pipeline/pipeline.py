"""Main processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pothole_vision.confidence.quality_score import geometry_confidence_from_sfm, tracking_confidence_from_frames
from pothole_vision.confidence.uncertainty import compute_overall_confidence
from pothole_vision.depth.scale_estimation import estimate_scale
from pothole_vision.depth.single_frame_depth import FallbackDepthEstimator
from pothole_vision.depth.video_depth_anything import VideoDepthAnythingEstimator
from pothole_vision.detection.postprocess import assign_zone
from pothole_vision.detection.yolo_seg import YOLOSegmentationDetector
from pothole_vision.geometry.camera import Camera
from pothole_vision.geometry.intrinsics import load_camera_intrinsics
from pothole_vision.measurement.pothole_geometry import measure_pothole
from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.reconstruction.colmap_backend import COLMAPBackend
from pothole_vision.reconstruction.triangulation import OpenCVTriangulationBackend
from pothole_vision.reconstruction.vggt_backend import VGGTBackend
from pothole_vision.roi.fixed_roi import FixedRoadROI, load_roi_config
from pothole_vision.storage.database import ResultsDatabase
from pothole_vision.storage.event_writer import EventWriter
from pothole_vision.storage.results import save_detections_json, save_tracks_json
from pothole_vision.tracking.tracker import PotholeTracker
from pothole_vision.utils.config import AppConfig, load_config, load_nested_config, resolve_path
from pothole_vision.utils.video_paths import paths_for_video, resolve_video_path
from pothole_vision.utils.logging import log_stage, setup_logging
from pothole_vision.video.reader import VideoReader
from pothole_vision.visualization.overlay import draw_detection, draw_roi_polygon, draw_zone_lines


class PotholePipeline:
    def __init__(
        self,
        config: AppConfig,
        project_root: Path,
        mode: PipelineMode = PipelineMode.DETECTION,
        debug: bool = False,
        force: bool = False,
    ) -> None:
        self.config = config
        self.root = project_root
        self.mode = mode
        self.debug = debug
        self.force = force

        setup_logging(config.logging.level)

        roi_path = resolve_path(self.root, config.roi["config_file"])
        self.roi = FixedRoadROI(load_roi_config(roi_path))
        self.roi_polygon = load_roi_config(roi_path).road_polygon

        det_cfg = load_nested_config(config, "detection", self.root)
        self.detector = YOLOSegmentationDetector(project_root=self.root, **{
            "model_path": det_cfg.get("model_path"),
            "fallback_model": det_cfg.get("fallback_model", "yolov8n-seg.pt"),
            "confidence": det_cfg.get("confidence_threshold", 0.25),
            "iou": det_cfg.get("iou_threshold", 0.45),
            "inference_size": det_cfg.get("inference_size", 640),
            "device": det_cfg.get("device", "auto"),
        })

        cam_path = resolve_path(self.root, config.camera.config_file)
        self.camera = Camera(load_camera_intrinsics(cam_path))

        self.base_output_dir = resolve_path(self.root, config.paths.output_dir)
        self.base_events_dir = resolve_path(self.root, config.paths.events_dir)
        self.base_cache_dir = resolve_path(self.root, config.paths.cache_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_path: Path,
        start_frame: int = 0,
        duration: float | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        video_path = resolve_video_path(video_path, self.root)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        run_paths = paths_for_video(
            video_path,
            self.base_output_dir,
            self.base_events_dir,
            self.base_cache_dir,
        )
        self.output_dir = run_paths.output_dir
        self.events_dir = run_paths.events_dir
        self.cache_dir = run_paths.cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        video_id = run_paths.video_id
        log_stage("VIDEO", f"processing {video_path.name} -> output/{video_id}/")

        with VideoReader(video_path) as reader:
            meta = reader.metadata
            log_stage("VIDEO", f"{meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.frame_count} frames")

            roi_mask = self.roi.create_mask(meta.width, meta.height)
            log_stage("ROI", f"loaded {len(self.roi_polygon)}-point polygon")

            if self.debug:
                sample = reader.read_frame(min(start_frame, meta.frame_count - 1))
                if sample:
                    roi_vis = draw_roi_polygon(sample.image, self.roi_polygon)
                    roi_vis = draw_zone_lines(roi_vis, self.config.zones.model_dump())
                    cv2.imwrite(str(self.output_dir / "roi_debug.jpg"), roi_vis)

            tracker = None
            if self.mode in (PipelineMode.TRACKING, PipelineMode.GEOMETRY, PipelineMode.FULL):
                tracker = PotholeTracker(
                    max_age=self.config.tracking.max_age,
                    min_track_frames=self.config.tracking.min_track_frames,
                    min_frames_for_depth=self.config.tracking.min_frames_for_depth,
                    max_best_frames=self.config.tracking.max_best_frames,
                    iou_threshold=self.config.tracking.iou_threshold,
                    depth_zone_y=self.config.zones.depth_start_y,
                    image_height=meta.height,
                )

            all_detections: list[dict] = []
            track_map: dict[str, Any] = {}
            annotated_path = run_paths.annotated_video
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                meta.fps,
                (meta.width, meta.height),
            )

            quality_cfg = self.config.quality.model_dump()
            zones_cfg = self.config.zones.model_dump()

            for frame in reader.iter_frames(
                start_frame=start_frame,
                max_frames=max_frames or self.config.video.max_frames,
                stride=self.config.video.frame_stride,
                duration_seconds=duration or self.config.video.duration_seconds,
                quality_config=quality_cfg,
            ):
                dets = self.detector.predict(frame.image, roi_mask, frame.frame_index)
                for det in dets:
                    det.zone = assign_zone(det.centroid[1], frame.height, zones_cfg)

                if tracker:
                    tracker.update(frame.frame_index, dets)
                    for t in tracker.tracks:
                        track_map[t.track_id] = t

                overlay = draw_roi_polygon(frame.image, self.roi_polygon)
                overlay = draw_zone_lines(overlay, zones_cfg)
                for det in dets:
                    tid = None
                    if tracker:
                        for t in tracker.tracks:
                            if t.latest_detection() and t.latest_detection().frame_index == frame.frame_index:
                                if np.allclose(t.latest_detection().centroid, det.centroid, atol=5):
                                    tid = t.track_id
                    overlay = draw_detection(overlay, det, tid)

                writer.write(overlay)

                for det in dets:
                    all_detections.append({
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "bbox": det.bbox,
                        "confidence": det.confidence,
                        "centroid": det.centroid,
                        "area_pixels": det.area_pixels,
                        "zone": det.zone,
                    })
                    log_stage("DETECT", f"PH candidate frame={frame.frame_index} conf={det.confidence:.2f}")

            if tracker:
                tracker.finalize_all(meta.frame_count - 1)

            writer.release()

            save_detections_json(self.output_dir / "detections.json", all_detections)

            if tracker:
                save_tracks_json(
                    self.output_dir / "tracks.json",
                    [t.to_dict() for t in tracker.all_tracks],
                )
                self._write_events(tracker, reader, meta.fps)

            if self.mode in (PipelineMode.GEOMETRY, PipelineMode.FULL):
                self._run_geometry(tracker, reader, roi_mask)

            if self.mode == PipelineMode.FULL:
                self._run_measurement(tracker)

            db = ResultsDatabase(self.output_dir / "results.db")
            db.insert_video(str(video_path), meta.__dict__)
            db.close()

            return {
                "video": str(video_path),
                "video_id": video_id,
                "output_dir": str(self.output_dir),
                "detections": len(all_detections),
                "tracks": len(tracker.all_tracks) if tracker else 0,
                "annotated_video": str(annotated_path),
            }

    def _write_events(self, tracker: PotholeTracker, reader: VideoReader, fps: float) -> None:
        event_writer = EventWriter(self.events_dir)
        for track in tracker.all_tracks:
            if track.frame_count < 1:
                continue
            best_det = track.latest_detection()
            best_frame = None
            if best_det:
                fd = reader.read_frame(best_det.frame_index)
                best_frame = fd.image if fd else None

            metadata = {
                "pothole_id": track.track_id,
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "detection_confidence": track.detection_confidence,
                "tracking_confidence": tracking_confidence_from_frames(track.frame_count),
                "status": track.status.value,
                "best_frames": track.best_frames,
            }
            mask = best_det.binary_mask if best_det else None
            event_writer.write_event(track, metadata, best_frame=best_frame, mask=mask)

    def _run_geometry(self, tracker: PotholeTracker | None, reader: VideoReader, roi_mask: np.ndarray) -> None:
        if not tracker:
            return
        recon_cfg = load_nested_config(self.config, "reconstruction", self.root)
        backend_name = recon_cfg.get("backend", "opencv")
        if backend_name == "colmap":
            backend = COLMAPBackend(recon_cfg)
        elif backend_name == "vggt":
            backend = VGGTBackend(recon_cfg)
        else:
            backend = OpenCVTriangulationBackend(recon_cfg)

        for track in tracker.all_tracks:
            if track.status.value == "REJECTED":
                continue
            indices = track.best_frames or [d.frame_index for d in track.detections[:5]]
            frames = []
            masks = []
            for idx in indices[:10]:
                fd = reader.read_frame(idx)
                if fd:
                    frames.append(fd.image)
                    masks.append(roi_mask)
            if len(frames) < 2:
                continue
            result = backend.reconstruct(frames, masks, self.camera)
            log_stage("GEOMETRY", f"{track.track_id} points={len(result.point_cloud_xyz)}")

            if self.debug and len(result.point_cloud_xyz) > 0:
                from pothole_vision.reconstruction.point_cloud import save_ply
                debug_dir = self.output_dir / "debug" / track.track_id
                debug_dir.mkdir(parents=True, exist_ok=True)
                save_ply(debug_dir / "point_cloud.ply", result.point_cloud_xyz, result.point_colors)

    def _run_measurement(self, tracker: PotholeTracker | None) -> None:
        if not tracker:
            return
        depth_cfg = load_nested_config(self.config, "depth", self.root)
        backend = depth_cfg.get("backend", "fallback")
        if backend == "video_depth_anything":
            estimator = VideoDepthAnythingEstimator(depth_cfg)
        else:
            estimator = FallbackDepthEstimator()

        event_writer = EventWriter(self.events_dir)
        for track in tracker.all_tracks:
            if track.frame_count < self.config.tracking.min_frames_for_depth:
                continue

            # Simplified measurement path for full mode
            measurement = measure_pothole(
                depths_perpendicular=np.array([0.05]),  # placeholder until geometry fused
                boundary_points=np.zeros((4, 2)),
                pixel_size_m=0.01,
                scale_confidence=0.0,
                geometry_confidence=0.0,
                surface_confidence=0.0,
                confidence_thresholds=self.config.confidence.model_dump(),
            )

            conf = compute_overall_confidence(
                detection=track.detection_confidence,
                tracking=tracking_confidence_from_frames(track.frame_count),
                geometry=0.0,
                scale=0.0,
                surface=0.0,
                depth=0.0,
            )

            meta = {
                "pothole_id": track.track_id,
                "metric_depth_available": measurement.metric_depth_available,
                "measurement_status": measurement.measurement_status,
                "reason": measurement.reason,
                **conf.to_dict(),
            }
            event_writer.write_event(track, meta, measurement=measurement.to_dict())
            self._print_summary(track, measurement, conf)

    def _print_summary(self, track, measurement, conf) -> None:
        print(f"\nPothole {track.track_id}")
        print("-" * 30)
        print(f"Detection        : {conf.detection_confidence:.2f}")
        print(f"Track frames     : {track.frame_count}")
        if measurement.metric_depth_available:
            print(f"Maximum depth    : {measurement.maximum_depth_cm:.1f} cm")
            print(f"Overall confidence: {conf.overall_confidence:.2f}")
            print(f"Status            : {measurement.measurement_status.upper()}")
        else:
            print("Depth            : NOT REPORTED")
            print(f"Status           : UNRELIABLE")
            print(f"Reason           : {measurement.reason}")
