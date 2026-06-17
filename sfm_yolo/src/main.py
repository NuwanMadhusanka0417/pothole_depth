"""End-to-end SfM + YOLO pothole detection & depth estimation pipeline.

Usage
-----
Run on a single video::

    python -m sfm_yolo.src.main process \
        --video data/pothole_video/pothole_video/test/rgb/0001.mp4 \
        --output sfm_yolo/outputs/runs/0001

Run on every test clip::

    python -m sfm_yolo.src.main batch \
        --root data/pothole_video/pothole_video --split test \
        --output sfm_yolo/outputs/runs/test_batch \
        --max-clips 10

Run on every video in a folder::

    python -m sfm_yolo.src.main folder \
        --input data/park \
        --output sfm_yolo/outputs/runs/park

Geometric-only mode (no COLMAP required)::

    python -m sfm_yolo.src.main process --video ... --no-sfm
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from .detection.yolo_detector import Detection, YOLODetector
from .fusion.depth_map_generator import (
    annotate_depth_overlay,
    colorize_depth,
    ground_plane_depth_map,
    write_image,
)
from .fusion.hybrid_estimator import HybridDepthEstimator, HybridDepthResult
from .geometry.geometric_depth import GeometricDepthEstimator
from .reconstruction.sfm_runner import SfMRunner
from .utils.camera_calibration import CameraIntrinsics, load_camera_calibration
from .utils.data_loader import MendeleyVideoDataset, mask_to_bboxes
from .utils.logging_utils import get_logger

_logger = get_logger("main")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _output_options(pipeline_cfg: dict) -> dict:
    """Resolve which artefacts to write from the pipeline config."""
    vis = pipeline_cfg.get("visualization", {})
    return {
        "save_overlay_video": vis.get("save_overlay_video", True),
        "save_depth_map": vis.get("save_depth_map", False),
        "save_pointcloud": vis.get("save_pointcloud", False),
        "save_summary": vis.get("save_summary", False),
    }


def _processing_options(pipeline_cfg: dict) -> dict:
    """Resolve frame-processing knobs from the pipeline config."""
    proc = pipeline_cfg.get("processing", {})
    scale = float(proc.get("scale", 0.5))
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"processing.scale must be in (0, 1], got {scale}")
    return {"process_scale": scale}


def build_components(
    pipeline_cfg_path: str | Path,
    *,
    use_sfm: bool = True,
    yolo_weights_override: Optional[str] = None,
) -> Tuple[YOLODetector | None, HybridDepthEstimator, dict]:
    """Construct YOLO + Hybrid estimator from the pipeline config."""
    pcfg = load_config(pipeline_cfg_path)
    cam_cfg = load_camera_calibration(pcfg["camera_config"])
    yolo_cfg = load_config(pcfg["yolo_config"])
    sfm_cfg = load_config(pcfg["sfm_config"]) if use_sfm else None

    geometric = GeometricDepthEstimator(cam_cfg)

    sfm_runner: SfMRunner | None = None
    if use_sfm and sfm_cfg is not None:
        sfm_runner = SfMRunner(
            cam_cfg,
            colmap_executable=sfm_cfg.get("colmap_executable", "colmap"),
            num_frames=sfm_cfg.get("num_frames_per_pothole", 8),
            frame_stride=sfm_cfg.get("frame_stride", 2),
            max_image_dim=sfm_cfg.get("image", {}).get("max_dim", 1600),
            single_camera=sfm_cfg.get("feature_extractor", {}).get("single_camera", True),
            camera_model=sfm_cfg.get("feature_extractor", {}).get("camera_model", "SIMPLE_RADIAL"),
            use_gpu=sfm_cfg.get("feature_extractor", {}).get("use_gpu", True),
            matcher_type=sfm_cfg.get("matcher", {}).get("type", "exhaustive"),
            enable_opencv_fallback=sfm_cfg.get("fallback", {}).get("enable_opencv_fallback", True),
            opencv_num_features=sfm_cfg.get("fallback", {}).get("num_features", 2000),
            opencv_ratio=sfm_cfg.get("fallback", {}).get("ratio_test", 0.75),
        )

    fusion_cfg = pcfg.get("fusion", {})
    hybrid = HybridDepthEstimator(
        geometric,
        sfm_runner=sfm_runner,
        geometric_weight=fusion_cfg.get("geometric_weight", 0.55),
        sfm_weight=fusion_cfg.get("sfm_weight", 0.45),
        disagreement_threshold=fusion_cfg.get("disagreement_threshold", 0.3),
        min_confidence=fusion_cfg.get("min_confidence", 0.35),
    )

    detector: YOLODetector | None = None
    weights_path = yolo_weights_override or pcfg.get("yolo_weights")
    if weights_path and Path(weights_path).exists():
        infer_cfg = yolo_cfg.get("inference", {})
        detector = YOLODetector(
            model_path=weights_path,
            conf_threshold=infer_cfg.get("conf_threshold", 0.4),
            iou_threshold=infer_cfg.get("iou_threshold", 0.5),
            max_detections=infer_cfg.get("max_detections", 30),
            class_names=yolo_cfg.get("dataset", {}).get("class_names", ("pothole",)),
        )
    else:
        _logger.warning(
            "No YOLO weights found at %s - falling back to mask-derived bboxes if available.",
            weights_path,
        )

    return detector, hybrid, pcfg


def _resize_frames(frames: List[np.ndarray], scale: float) -> List[np.ndarray]:
    """Downscale frames by ``scale`` (e.g. 0.5 = half width and height)."""
    if scale == 1.0:
        return frames
    resized: List[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized.append(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return resized


def _hybrid_for_frame_size(
    hybrid: HybridDepthEstimator,
    height: int,
    width: int,
) -> HybridDepthEstimator:
    """Return a hybrid estimator whose intrinsics match the frame size."""
    base = hybrid.geometric.intrinsics
    if (width, height) == base.image_size:
        return hybrid

    intr = base.for_image_size(width, height)
    geometric = GeometricDepthEstimator(intr)

    sfm_runner: SfMRunner | None = None
    if hybrid.sfm_runner is not None:
        old = hybrid.sfm_runner
        sfm_runner = SfMRunner(
            intr,
            colmap_executable=old.colmap_executable,
            num_frames=old.num_frames,
            frame_stride=old.frame_stride,
            max_image_dim=old.max_image_dim,
            single_camera=old.single_camera,
            camera_model=old.camera_model,
            use_gpu=old.use_gpu,
            matcher_type=old.matcher_type,
            enable_opencv_fallback=old.enable_opencv_fallback,
            opencv_num_features=old.opencv_num_features,
            opencv_ratio=old.opencv_ratio,
        )

    return HybridDepthEstimator(
        geometric,
        sfm_runner=sfm_runner,
        geometric_weight=hybrid.geometric_weight,
        sfm_weight=hybrid.sfm_weight,
        disagreement_threshold=hybrid.disagreement_threshold,
        min_confidence=hybrid.min_confidence,
    )


def discover_videos(
    folder: str | Path,
    *,
    recursive: bool = False,
) -> List[Path]:
    """Return all video files in ``folder``, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    videos = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos


def _overlay_output_path(
    video: Path,
    output_dir: Path,
    *,
    input_root: Path,
    recursive: bool,
) -> Path:
    """Build a flat overlay filename inside ``output_dir``."""
    if recursive:
        rel = video.relative_to(input_root)
        stem = rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
    else:
        stem = video.stem
    return output_dir / f"{stem}_overlay.mp4"


# ---------------------------------------------------------------------------
# Frame source helpers
# ---------------------------------------------------------------------------
def _load_video_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _load_paired_masks(mask_video: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(mask_video))
    if not cap.isOpened():
        return []
    masks: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            masks.append((gray > 16).astype(np.uint8) * 255)
    finally:
        cap.release()
    return masks


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------
def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    detector: Optional[YOLODetector],
    hybrid: HybridDepthEstimator,
    pipeline_cfg: dict,
    fallback_mask_video: Optional[str | Path] = None,
    save_depth_map: bool = False,
    save_overlay_video: bool = True,
    save_pointcloud: bool = False,
    save_summary: bool = False,
    overlay_path: Optional[str | Path] = None,
    process_scale: float = 0.5,
    every_nth_detect: int = 1,
) -> Dict:
    """Run the full pipeline on a single clip and dump artefacts."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    src_h, src_w = frames[0].shape[:2]
    if process_scale != 1.0:
        frames = _resize_frames(frames, process_scale)
        proc_h, proc_w = frames[0].shape[:2]
        _logger.info(
            "Loaded %d frames from %s, processing at %dx%d (scale=%.2f, source %dx%d)",
            len(frames), video_path, proc_w, proc_h, process_scale, src_w, src_h,
        )
    else:
        proc_h, proc_w = src_h, src_w
        _logger.info("Loaded %d frames (%dx%d) from %s", len(frames), proc_w, proc_h, video_path)

    active_hybrid = _hybrid_for_frame_size(hybrid, proc_h, proc_w)

    # ------------------------------------------------------------------
    # 1. Detect potholes per frame
    # ------------------------------------------------------------------
    per_frame_dets: List[List[Detection]] = []
    if detector is not None:
        for i, frame in enumerate(frames):
            dets = detector.detect(frame) if (i % every_nth_detect == 0) else []
            per_frame_dets.append(dets)
    elif fallback_mask_video is not None and Path(fallback_mask_video).exists():
        masks = _load_paired_masks(Path(fallback_mask_video))
        for i, frame in enumerate(frames):
            mask = masks[i] if i < len(masks) else np.zeros(frame.shape[:2], dtype=np.uint8)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            dets: List[Detection] = []
            for box in mask_to_bboxes(mask, min_pixels=max(1, int(80 * process_scale * process_scale))):
                dets.append(Detection(bbox=tuple(map(float, box)), confidence=1.0))
            per_frame_dets.append(dets)
    else:
        raise RuntimeError(
            "No YOLO weights and no fallback mask video provided - cannot detect potholes."
        )

    n_dets = sum(len(d) for d in per_frame_dets)
    _logger.info("Total detections: %d (avg %.2f per frame)", n_dets, n_dets / max(1, len(frames)))

    # ------------------------------------------------------------------
    # 2. Cross-frame association (greedy IoU)
    # ------------------------------------------------------------------
    tracker_cfg = pipeline_cfg.get("tracker", {})
    iou_thr = float(tracker_cfg.get("iou_threshold", 0.3))
    min_track = int(tracker_cfg.get("min_track_length", 3))
    max_age = int(tracker_cfg.get("max_age", 5))

    if detector is not None:
        frame_det_pairs = list(enumerate(per_frame_dets))
        tracks = detector.track_across_frames(
            frame_det_pairs, iou_threshold=iou_thr, max_age=max_age
        )
    else:
        # cheap stand-in: use per-frame detections as if each were its own track
        tracks = [d for f in per_frame_dets for d in [[x] for x in f]]

    confirmed = [t for t in tracks if len(t) >= min_track]
    _logger.info("Tracks: total=%d, confirmed (>= %d frames)=%d",
                 len(tracks), min_track, len(confirmed))

    # ------------------------------------------------------------------
    # 3. Per-track depth estimation
    # ------------------------------------------------------------------
    track_results: List[Dict] = []

    for tid, track in enumerate(confirmed):
        bboxes = [tuple(d.bbox) for d in track]
        # Approximate frame-window for SfM = first to last detection
        frame_indices = [
            _find_frame_index(per_frame_dets, det) for det in track
        ]
        frame_indices = [i for i in frame_indices if i is not None]
        if frame_indices:
            sfm_window = (min(frame_indices), max(frame_indices))
        else:
            sfm_window = (0, len(frames) - 1)

        ply_path: Optional[Path] = None
        if save_pointcloud:
            ply_path = output_dir / f"track_{tid:03d}.ply"

        sfm_work_dir: Optional[Path] = None
        if active_hybrid.sfm_runner is not None:
            if save_pointcloud:
                sfm_work_dir = output_dir / f"track_{tid:03d}_sfm"
            else:
                sfm_work_dir = Path(tempfile.mkdtemp(prefix=f"track_{tid:03d}_sfm_"))

        try:
            result: HybridDepthResult = active_hybrid.estimate_depth(
                bboxes=bboxes,
                sfm_frames=frames,
                sfm_window=sfm_window,
                ref_bbox_for_scale=bboxes[len(bboxes) // 2],
                save_pointcloud=ply_path,
                sfm_work_dir=sfm_work_dir,
            )
        except Exception as exc:  # pragma: no cover
            _logger.warning("Track %d failed: %s", tid, exc)
            continue
        finally:
            if sfm_work_dir is not None and not save_pointcloud:
                shutil.rmtree(sfm_work_dir, ignore_errors=True)

        track_results.append(
            {
                "track_id": tid,
                "n_frames": len(track),
                "frame_indices": frame_indices,
                "bbox_mid": list(bboxes[len(bboxes) // 2]),
                "result": result.as_dict(),
            }
        )

    # ------------------------------------------------------------------
    # 4. Save artefacts
    # ------------------------------------------------------------------
    summary = {
        "video": str(video_path),
        "frames": len(frames),
        "detections_total": n_dets,
        "tracks_total": len(tracks),
        "tracks_confirmed": len(confirmed),
        "tracks": track_results,
    }

    if save_overlay_video:
        out_overlay = Path(overlay_path) if overlay_path is not None else output_dir / "overlay.mp4"
        _write_overlay_video(
            frames=frames,
            per_frame_dets=per_frame_dets,
            track_results=track_results,
            confirmed_tracks=confirmed,
            out_path=out_overlay,
        )
        _logger.info("Wrote overlay video to %s", out_overlay)

    if save_depth_map and frames:
        depth_path = output_dir / "ground_plane_depth.png"
        depth = ground_plane_depth_map(frames[0].shape[:2], active_hybrid.geometric)
        write_image(colorize_depth(depth), depth_path)
        _logger.info("Wrote ground-plane depth map to %s", depth_path)

    if save_summary:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        _logger.info("Wrote summary to %s", summary_path)

    return summary


def _find_frame_index(
    per_frame_dets: Sequence[Sequence[Detection]],
    target: Detection,
) -> Optional[int]:
    for i, dets in enumerate(per_frame_dets):
        for d in dets:
            if d is target:
                return i
    return None


def _write_overlay_video(
    *,
    frames: Sequence[np.ndarray],
    per_frame_dets: Sequence[Sequence[Detection]],
    track_results: Sequence[dict],
    confirmed_tracks: Sequence[Sequence[Detection]],
    out_path: Path,
) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))

    # Map each detection identity to its track-level depth/confidence
    det_to_depth: Dict[int, Tuple[float, float]] = {}
    for tr, info in zip(confirmed_tracks, track_results):
        depth = info["result"]["depth_m"]
        conf = info["result"]["confidence"]
        for d in tr:
            det_to_depth[id(d)] = (depth, conf)

    try:
        for frame, dets in zip(frames, per_frame_dets):
            depths = [det_to_depth.get(id(d), (float("nan"), float("nan")))[0] for d in dets]
            confs = [det_to_depth.get(id(d), (float("nan"), float("nan")))[1] for d in dets]
            out = annotate_depth_overlay(frame, dets, depths, confs)
            writer.write(out)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Batch over a folder of videos
# ---------------------------------------------------------------------------
def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    detector: Optional[YOLODetector],
    hybrid: HybridDepthEstimator,
    pipeline_cfg: dict,
    recursive: bool = False,
    max_videos: Optional[int] = None,
    save_overlay_video: bool = True,
    save_depth_map: bool = False,
    save_pointcloud: bool = False,
    save_summary: bool = False,
    process_scale: float = 0.5,
) -> List[dict]:
    """Run the pipeline on every video file inside ``input_dir``.

    Overlay videos are written flat into ``output_dir`` as
    ``<video_stem>_overlay.mp4``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_dir, recursive=recursive)
    if not videos:
        raise FileNotFoundError(
            f"No videos found in {input_dir} "
            f"(supported: {', '.join(sorted(VIDEO_EXTENSIONS))})"
        )
    if max_videos is not None:
        videos = videos[:max_videos]

    _logger.info("Found %d video(s) in %s", len(videos), input_dir)

    summaries: List[dict] = []
    n_ok = 0
    for video_path in tqdm(videos, desc="videos", unit="video"):
        overlay_out = _overlay_output_path(
            video_path,
            output_dir,
            input_root=input_dir,
            recursive=recursive,
        )
        try:
            summary = process_video(
                video_path,
                output_dir,
                detector=detector,
                hybrid=hybrid,
                pipeline_cfg=pipeline_cfg,
                save_overlay_video=save_overlay_video,
                save_depth_map=save_depth_map,
                save_pointcloud=save_pointcloud,
                save_summary=save_summary,
                overlay_path=overlay_out,
                process_scale=process_scale,
            )
            n_ok += 1
            if save_summary:
                summaries.append(summary)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed on video %s: %s", video_path.name, exc)
            continue

    _logger.info(
        "Finished folder batch: %d/%d videos processed, outputs in %s",
        n_ok,
        len(videos),
        output_dir,
    )
    return summaries


# ---------------------------------------------------------------------------
# Batch over a Mendeley split
# ---------------------------------------------------------------------------
def process_split(
    root: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    detector: Optional[YOLODetector],
    hybrid: HybridDepthEstimator,
    pipeline_cfg: dict,
    max_clips: Optional[int] = None,
    use_mask_fallback: bool = True,
    save_overlay_video: bool = True,
    save_depth_map: bool = False,
    save_pointcloud: bool = False,
    save_summary: bool = False,
    process_scale: float = 0.5,
) -> List[dict]:
    """Run the pipeline on every clip in a given split.

    Returns a list of summaries (one per processed clip).
    """
    ds = MendeleyVideoDataset(root, split=split, frame_stride=1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    clips = ds.clips if max_clips is None else ds.clips[:max_clips]
    for clip_id, rgb_path, mask_path in tqdm(clips, desc=f"{split}", unit="clip"):
        clip_out = output_dir / clip_id
        try:
            summary = process_video(
                rgb_path,
                clip_out,
                detector=detector,
                hybrid=hybrid,
                pipeline_cfg=pipeline_cfg,
                fallback_mask_video=mask_path if use_mask_fallback else None,
                save_overlay_video=save_overlay_video,
                save_depth_map=save_depth_map,
                save_pointcloud=save_pointcloud,
                save_summary=save_summary,
                process_scale=process_scale,
            )
            if save_summary:
                summaries.append(summary)
        except Exception as exc:  # pragma: no cover
            _logger.error("Failed on clip %s: %s", clip_id, exc)
            continue

    if save_summary and summaries:
        summary_path = output_dir / "_index.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summaries": summaries, "n_clips": len(summaries)}, f, indent=2, default=str)
        _logger.info("Batch summary written to %s", summary_path)
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="sfm_yolo/configs/pipeline.yaml")
    p.add_argument("--weights", default=None, help="Override YOLO weights path")
    p.add_argument("--no-sfm", action="store_true", help="Disable SfM (geometric only)")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay video")
    p.add_argument("--summary", action="store_true", help="Write summary.json")
    p.add_argument("--depth-map", action="store_true", help="Write ground_plane_depth.png")
    p.add_argument("--pointcloud", action="store_true", help="Write per-track .ply point clouds")
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Process at this fraction of source resolution (default: 0.5 from config; 1.0 = full)",
    )


def main() -> None:
    parser = argparse.ArgumentParser("sfm_yolo pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_process = sub.add_parser("process", help="Process a single video")
    p_process.add_argument("--video", required=True, help="Input video path")
    p_process.add_argument("--output", required=True, help="Per-clip output directory")
    p_process.add_argument("--mask-video", default=None, help="Fallback mask video for bbox extraction")
    _common_args(p_process)

    p_batch = sub.add_parser("batch", help="Process every clip in a Mendeley split")
    p_batch.add_argument("--root", required=True,
                         help="Path to .../pothole_video/pothole_video")
    p_batch.add_argument("--split", default="test", choices=("train", "val", "test"))
    p_batch.add_argument("--output", required=True)
    p_batch.add_argument("--max-clips", type=int, default=None)
    p_batch.add_argument("--no-mask-fallback", action="store_true",
                         help="Don't use the dataset masks even if YOLO weights are missing")
    _common_args(p_batch)

    p_folder = sub.add_parser("folder", help="Process every video in a folder")
    p_folder.add_argument("--input", required=True, help="Folder containing input videos")
    p_folder.add_argument("--output", required=True, help="Folder for output overlay videos")
    p_folder.add_argument("--recursive", action="store_true",
                          help="Also search sub-folders for videos")
    p_folder.add_argument("--max-videos", type=int, default=None,
                          help="Process at most this many videos")
    _common_args(p_folder)

    args = parser.parse_args()

    detector, hybrid, pipeline_cfg = build_components(
        args.config,
        use_sfm=not args.no_sfm,
        yolo_weights_override=args.weights,
    )

    out_opts = _output_options(pipeline_cfg)
    proc_opts = _processing_options(pipeline_cfg)
    if args.no_overlay:
        out_opts["save_overlay_video"] = False
    if args.summary:
        out_opts["save_summary"] = True
    if args.depth_map:
        out_opts["save_depth_map"] = True
    if args.pointcloud:
        out_opts["save_pointcloud"] = True
    if args.scale is not None:
        if not (0.0 < args.scale <= 1.0):
            parser.error("--scale must be in (0, 1]")
        proc_opts["process_scale"] = args.scale

    run_kwargs = {**out_opts, **proc_opts}

    if args.cmd == "process":
        process_video(
            args.video,
            args.output,
            detector=detector,
            hybrid=hybrid,
            pipeline_cfg=pipeline_cfg,
            fallback_mask_video=args.mask_video,
            **run_kwargs,
        )
    elif args.cmd == "batch":
        process_split(
            args.root,
            args.split,
            args.output,
            detector=detector,
            hybrid=hybrid,
            pipeline_cfg=pipeline_cfg,
            max_clips=args.max_clips,
            use_mask_fallback=not args.no_mask_fallback,
            **run_kwargs,
        )
    elif args.cmd == "folder":
        process_folder(
            args.input,
            args.output,
            detector=detector,
            hybrid=hybrid,
            pipeline_cfg=pipeline_cfg,
            recursive=args.recursive,
            max_videos=args.max_videos,
            **run_kwargs,
        )


if __name__ == "__main__":
    main()
