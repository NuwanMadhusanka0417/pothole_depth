"""Frame quality assessment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    sharpness: float
    brightness: float
    contrast: float
    blur_score: float
    saturation_clip_ratio: float
    usable_for_detection: bool
    usable_for_depth: bool


def assess_frame_quality(image: np.ndarray, config: dict | None = None) -> FrameQuality:
    cfg = config or {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = lap_var

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Normalized blur score: higher = more blur
    blur_score = 1.0 / (1.0 + lap_var / 100.0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    clip_ratio = float(np.mean((sat > 250) | (val > 250) | (val < 5)))

    min_sharp = cfg.get("min_sharpness", 50.0)
    max_blur = cfg.get("max_blur_score", 0.65)
    min_bright = cfg.get("min_brightness", 30.0)
    max_bright = cfg.get("max_brightness", 230.0)
    min_contrast = cfg.get("min_contrast", 15.0)
    max_clip = cfg.get("max_saturation_clip_ratio", 0.15)

    usable_det = (
        sharpness >= min_sharp * 0.5
        and min_bright <= brightness <= max_bright
        and contrast >= min_contrast * 0.5
    )
    usable_depth = (
        sharpness >= min_sharp
        and blur_score <= max_blur
        and min_bright <= brightness <= max_bright
        and contrast >= min_contrast
        and clip_ratio <= max_clip
    )

    return FrameQuality(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        saturation_clip_ratio=clip_ratio,
        usable_for_detection=usable_det,
        usable_for_depth=usable_depth,
    )
