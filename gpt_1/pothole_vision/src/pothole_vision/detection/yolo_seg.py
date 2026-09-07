"""YOLO instance segmentation backend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pothole_vision.detection.detector import PotholeDetection, PotholeDetector
from pothole_vision.detection.postprocess import filter_detections_by_roi, mask_to_polygon
from pothole_vision.utils.logging import log_stage


class YOLOSegmentationDetector(PotholeDetector):
    """Ultralytics YOLO segmentation with coordinate preservation via full-frame inference."""

    def __init__(
        self,
        model_path: str | None = None,
        fallback_model: str = "yolov8n-seg.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        inference_size: int = 640,
        device: str = "auto",
        project_root: Path | None = None,
    ) -> None:
        from ultralytics import YOLO

        root = project_root or Path.cwd()
        path = Path(model_path) if model_path else None
        if path and not path.is_absolute():
            path = root / path

        if path and path.exists():
            self.model = YOLO(str(path))
            self.model_name = str(path)
            log_stage("DETECT", f"Loaded custom weights: {path}")
        else:
            if path:
                log_stage("DETECT", f"Custom model not found at {path}, using fallback {fallback_model}")
            self.model = YOLO(fallback_model)
            self.model_name = fallback_model

        self.confidence = confidence
        self.iou = iou
        self.inference_size = inference_size
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray | None = None,
        frame_index: int = 0,
    ) -> list[PotholeDetection]:
        h, w = image.shape[:2]

        # Full-frame inference — YOLO letterboxes internally; we map masks back
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.inference_size,
            device=self.device,
            verbose=False,
        )

        detections: list[PotholeDetection] = []
        if not results:
            return detections

        result = results[0]
        if result.masks is None or result.boxes is None:
            return detections

        masks_data = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            conf = float(confs[i])
            # Resize mask to original image size
            mask_small = (masks_data[i] > 0.5).astype(np.uint8) * 255
            mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

            if roi_mask is not None:
                mask_full = cv2.bitwise_and(mask_full, roi_mask)

            area = int(np.sum(mask_full > 0))
            if area < 50:
                continue

            x1, y1, x2, y2 = boxes[i]
            # Map bbox from letterboxed result coords to original
            # result.masks already aligned to orig via ultralytics
            bbox = (int(x1), int(y1), int(x2), int(y2))
            polygon = mask_to_polygon(mask_full)
            ys, xs = np.where(mask_full > 0)
            if len(xs) == 0:
                continue
            centroid = (float(np.mean(xs)), float(np.mean(ys)))

            detections.append(
                PotholeDetection(
                    bbox=bbox,
                    polygon=polygon,
                    binary_mask=mask_full,
                    confidence=conf,
                    centroid=centroid,
                    area_pixels=area,
                    frame_index=frame_index,
                )
            )

        if roi_mask is not None:
            detections = filter_detections_by_roi(detections, roi_mask)

        return detections
