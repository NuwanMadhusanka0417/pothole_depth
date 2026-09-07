"""Ultralytics YOLO segmentation backend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pothole_vision.detection.detector import PotholeDetector
from pothole_vision.detection.postprocess import filter_detections, refine_mask
from pothole_vision.models import FrameData, PotholeDetection
from pothole_vision.utils.logging import log_tag


class YOLOSegDetector(PotholeDetector):
    """YOLO instance segmentation with coordinate preservation."""

    def __init__(self, config: dict) -> None:
        self.config = config.get("detection", config)
        self.post_cfg = config.get("postprocess", {})
        self.model_path = Path(self.config.get("model_path", "data/models/pothole_seg.pt"))
        self.fallback = self.config.get("fallback_model", "yolov8n-seg.pt")
        self.conf = float(self.config.get("confidence_threshold", 0.35))
        self.iou = float(self.config.get("iou_threshold", 0.45))
        self.inference_size = int(self.config.get("inference_size", 640))
        self.device = self.config.get("device", "auto")
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for detection. pip install ultralytics"
            ) from e

        model_file = self.model_path
        if not model_file.exists():
            log_tag(
                "DETECT",
                f"Custom weights not found at {model_file}, using fallback {self.fallback}",
            )
            model_file = self.fallback

        self._model = YOLO(str(model_file))
        log_tag("DETECT", f"Loaded model: {model_file}")

    def predict(self, frame: FrameData, roi_mask: np.ndarray) -> list[PotholeDetection]:
        assert self._model is not None
        h, w = frame.height, frame.width

        results = self._model.predict(
            source=frame.image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.inference_size,
            device=self.device,
            verbose=False,
        )

        detections: list[PotholeDetection] = []
        if not results:
            return detections

        result = results[0]
        if result.masks is None:
            return detections

        masks_data = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            box = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            mask_small = masks_data[i]
            full_mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            full_mask = full_mask > 0.5
            full_mask = refine_mask(
                full_mask, self.post_cfg.get("morph_kernel_size", 3)
            )

            # Extract contour polygon
            contours, _ = cv2.findContours(
                full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                polygon = contours[0].reshape(-1, 2).astype(np.float64)
            else:
                polygon = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
                )

            ys, xs = np.where(full_mask)
            if len(xs) == 0:
                continue
            centroid = (float(np.mean(xs)), float(np.mean(ys)))
            area = int(np.sum(full_mask))

            detections.append(
                PotholeDetection(
                    bbox=(x1, y1, x2, y2),
                    polygon=polygon,
                    binary_mask=full_mask,
                    confidence=conf,
                    centroid=centroid,
                    area_pixels=area,
                    frame_index=frame.frame_index,
                )
            )

        return filter_detections(
            detections,
            roi_mask,
            min_area=self.post_cfg.get("min_area_pixels", 80),
            min_mask_ratio=self.post_cfg.get("min_mask_area_ratio", 0.3),
        )


def create_detector(config: dict) -> PotholeDetector:
    backend = config.get("detection", {}).get("backend", "yolo_seg")
    if backend == "yolo_seg":
        return YOLOSegDetector(config)
    raise ValueError(f"Unknown detection backend: {backend}")
