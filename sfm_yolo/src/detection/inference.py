"""Real-time / offline pothole detection inference helpers.

Run from the project root::

    python -m sfm_yolo.src.detection.inference \
        --weights sfm_yolo/models/pothole_yolov8n.pt \
        --source data/pothole_video/pothole_video/test/rgb/0001.mp4 \
        --output sfm_yolo/outputs/inference_demo.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2

from ..utils.logging_utils import get_logger
from .yolo_detector import Detection, YOLODetector

_logger = get_logger("detection.inference")


def run_inference_on_video(
    weights: str | Path,
    source: str | Path,
    output_video: str | Path | None = None,
    *,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    every_nth: int = 1,
    device: str = "",
    save_json: str | Path | None = None,
) -> List[dict]:
    """Run pothole detection over a video and optionally render an overlay.

    Returns a list of per-frame results::

        [{"frame_index": 0, "detections": [Detection.as_dict(), ...]}, ...]
    """
    detector = YOLODetector(
        model_path=weights,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video is not None:
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    per_frame: List[dict] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets: List[Detection] = []
            if idx % every_nth == 0:
                dets = detector.detect(frame)
            per_frame.append(
                {
                    "frame_index": idx,
                    "detections": [d.as_dict() for d in dets],
                }
            )
            if writer is not None:
                rendered = detector.draw_detections(frame, dets)
                writer.write(rendered)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    n_dets = sum(len(f["detections"]) for f in per_frame)
    _logger.info("Processed %d frames, %d total detections", len(per_frame), n_dets)

    if save_json is not None:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(per_frame, f, indent=2)
        _logger.info("Saved per-frame detections to %s", save_json)

    return per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO pothole detection on a video.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Optional rendered output video")
    parser.add_argument("--json", default=None, help="Optional per-frame JSON output")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--every-nth", type=int, default=1)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    run_inference_on_video(
        weights=args.weights,
        source=args.source,
        output_video=args.output,
        save_json=args.json,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        every_nth=args.every_nth,
        device=args.device,
    )


if __name__ == "__main__":
    main()
