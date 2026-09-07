"""Feature detection and matching."""

from __future__ import annotations

import cv2
import numpy as np


def detect_and_match(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None = None,
    detector: str = "orb",
    max_features: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    if detector == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
    else:
        det = cv2.ORB_create(nfeatures=max_features)

    kp1, des1 = det.detectAndCompute(gray1, mask.astype(np.uint8) * 255 if mask is not None else None)
    kp2, des2 = det.detectAndCompute(gray2, mask.astype(np.uint8) * 255 if mask is not None else None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.array([]), np.array([]), np.array([])

    norm = cv2.NORM_HAMMING if detector == "orb" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2, np.array([m.distance for m in matches])
