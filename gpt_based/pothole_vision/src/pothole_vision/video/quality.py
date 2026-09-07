"""Frame quality assessment."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.models import FrameData, FrameQuality


def assess_frame_quality(
    frame: FrameData,
    roi_mask: np.ndarray | None = None,
    config: dict | None = None,
) -> FrameQuality:
    """Compute sharpness, blur, brightness, contrast metrics."""
    cfg = config or {}
    min_sharp = cfg.get("min_sharpness", 50.0)
    max_blur = cfg.get("max_blur_score", 0.85)
    min_bright = cfg.get("min_brightness", 30.0)
    max_bright = cfg.get("max_brightness", 240.0)
    min_contrast = cfg.get("min_contrast", 15.0)
    max_clip = cfg.get("max_saturation_clip_ratio", 0.15)

    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    if roi_mask is not None:
        gray = gray.copy()
        gray[~roi_mask] = 0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Normalized blur score: higher = more blur
    blur_score = 1.0 / (1.0 + sharpness / 100.0)

    brightness = float(np.mean(gray[roi_mask] if roi_mask is not None else gray))
    contrast = float(np.std(gray[roi_mask] if roi_mask is not None else gray))

    hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    clip_ratio = float(np.mean((sat > 250) | (sat < 5)))

    usable_detection = (
        sharpness >= min_sharp
        and blur_score <= max_blur
        and min_bright <= brightness <= max_bright
        and contrast >= min_contrast
        and clip_ratio <= max_clip
    )

    usable_depth = usable_detection and sharpness >= min_sharp * 1.2

    return FrameQuality(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        saturation_clip_ratio=clip_ratio,
        usable_for_detection=usable_detection,
        usable_for_depth=usable_depth,
    )


def assess_pothole_visibility(
    mask: np.ndarray,
    roi_mask: np.ndarray,
    min_pixels: int = 100,
) -> tuple[float, int, bool]:
    """Return visible ratio, pixel count, sufficient flag."""
    pothole_pixels = int(np.sum(mask))
    roi_pixels = int(np.sum(roi_mask))
    visible_in_roi = int(np.sum(mask & roi_mask))
    ratio = visible_in_roi / max(pothole_pixels, 1)
    sufficient = visible_in_roi >= min_pixels and ratio >= 0.5
    return ratio, visible_in_roi, sufficient
