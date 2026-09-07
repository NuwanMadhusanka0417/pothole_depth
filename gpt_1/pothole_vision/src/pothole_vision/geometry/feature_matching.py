"""Feature detection and matching."""

from __future__ import annotations

import cv2
import numpy as np


def create_detector(detector_type: str = "orb", max_features: int = 2000):
    if detector_type == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.ORB_create(nfeatures=max_features)


def detect_and_match(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    detector_type: str = "orb",
    max_features: int = 2000,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    detector = create_detector(detector_type, max_features)
    kp1, des1 = detector.detectAndCompute(gray1, mask1)
    kp2, des2 = detector.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2)), []

    if detector_type == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2, matches
