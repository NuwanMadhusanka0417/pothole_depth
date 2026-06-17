"""YOLOv8 wrapper for pothole detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.logging_utils import get_logger

_logger = get_logger("detection.yolo")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One pothole detection in a single frame."""

    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    class_name: str = "pothole"
    track_id: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class YOLODetector:
    """Thin wrapper around ``ultralytics.YOLO`` for pothole detection."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        *,
        device: str = "",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        max_detections: int = 30,
        class_names: Sequence[str] = ("pothole",),
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "pip install ultralytics"
            ) from exc

        self._YOLO = YOLO
        self.model_path = str(model_path)
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_names = tuple(class_names)
        _logger.info("Loading YOLO weights: %s (device=%s)", self.model_path, device or "auto")
        self.model = YOLO(self.model_path)

    # ------------------------------------------------------------------
    # Frame-level detection
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        *,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Run detection on a single frame and return ``Detection`` items."""
        if frame is None or frame.size == 0:
            return []
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        kwargs = {
            "source": frame,
            "conf": conf,
            "iou": iou,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            return []
        return self._parse_result(results[0])

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        **kwargs,
    ) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        if not frames:
            return []
        conf = kwargs.get("conf_threshold", self.conf_threshold)
        iou = kwargs.get("iou_threshold", self.iou_threshold)
        predict_kwargs = {
            "source": list(frames),
            "conf": conf,
            "iou": iou,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        results = self.model.predict(**predict_kwargs)
        return [self._parse_result(r) for r in results]

    # ------------------------------------------------------------------
    # Video-level helpers
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str | Path,
        *,
        every_nth: int = 1,
    ) -> List[Tuple[int, List[Detection]]]:
        """Run detection on every Nth frame of a video.

        Returns a list of ``(frame_index, [detections])`` tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")

        out: List[Tuple[int, List[Detection]]] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % every_nth == 0:
                    dets = self.detect(frame)
                    out.append((idx, dets))
                idx += 1
        finally:
            cap.release()
        return out

    # ------------------------------------------------------------------
    # Lightweight IoU-based "tracker" across frames
    # ------------------------------------------------------------------
    def track_across_frames(
        self,
        frame_detections: Iterable[Tuple[int, List[Detection]]],
        *,
        iou_threshold: float = 0.3,
        max_age: int = 5,
    ) -> List[List[Detection]]:
        """Greedy IoU association across frames.

        Each returned inner list is a *track*: detections of the same
        physical pothole across consecutive frames.
        """
        from ..utils.evaluation_metrics import iou_xyxy

        tracks: List[List[Detection]] = []
        last_seen: List[int] = []
        next_id = 0

        for frame_idx, dets in frame_detections:
            unmatched = list(range(len(dets)))
            for ti, track in enumerate(tracks):
                if frame_idx - last_seen[ti] > max_age:
                    continue
                last = track[-1]
                best_iou, best_j = 0.0, -1
                for j in unmatched:
                    iou = iou_xyxy(last.bbox, dets[j].bbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_threshold and best_j >= 0:
                    det = dets[best_j]
                    det.track_id = ti
                    track.append(det)
                    last_seen[ti] = frame_idx
                    unmatched.remove(best_j)

            for j in unmatched:
                det = dets[j]
                det.track_id = next_id
                tracks.append([det])
                last_seen.append(frame_idx)
                next_id += 1

        return tracks

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: Sequence[Detection],
        *,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        label_format: str = "{name} {conf:.2f}",
    ) -> np.ndarray:
        """Return a copy of ``frame`` with bounding boxes / labels drawn."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = label_format.format(name=det.class_name, conf=det.confidence)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse_result(self, result) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

        # ultralytics names attribute is a dict {id: name}
        names = getattr(result, "names", None) or {}

        out: List[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            cls_id = int(cls)
            name = names.get(cls_id) if isinstance(names, dict) else None
            if name is None:
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            out.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=str(name),
                )
            )
        return out
