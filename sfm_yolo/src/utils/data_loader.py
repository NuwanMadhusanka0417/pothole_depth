"""Data loaders for the Mendeley pothole *video* dataset.

Dataset layout (after unzipping ``pothole_video.zip``)::

    data/pothole_video/pothole_video/
        train/{rgb,mask}/0001.mp4 .. 0373.mp4
        val/  {rgb,mask}/0001.mp4 .. 0124.mp4
        test/ {rgb,mask}/0001.mp4 .. 0124.mp4

Each ``rgb/NNNN.mp4`` is a short dash-cam style clip, paired with a
``mask/NNNN.mp4`` of identical length where pothole pixels are bright
and the background is dark. We use the masks to (a) derive bounding
boxes for YOLO fine-tuning, and (b) get a "ground truth" pothole
region when validating the geometric / SfM depth estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFramePair:
    """A single time-aligned (rgb, mask) frame from a video pair."""

    clip_id: str
    split: str
    frame_index: int
    rgb: np.ndarray   # (H, W, 3) uint8 BGR (OpenCV convention)
    mask: np.ndarray  # (H, W)    uint8, foreground > 0


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class MendeleyVideoDataset:
    """Iterates over clip pairs from the Mendeley pothole video dataset.

    Parameters
    ----------
    root : path-like
        Path to ``pothole_video/pothole_video`` (the directory containing
        ``train/``, ``val/`` and ``test/``).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    frame_stride : int
        Yield only every Nth frame from each clip (1 = every frame).
    binarize_threshold : int
        Pixel value above which the mask is considered foreground.
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        frame_stride: int = 1,
        binarize_threshold: int = 16,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.frame_stride = max(1, int(frame_stride))
        self.binarize_threshold = int(binarize_threshold)

        self.rgb_dir = self.root / split / "rgb"
        self.mask_dir = self.root / split / "mask"
        if not self.rgb_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected '{self.rgb_dir}' and '{self.mask_dir}' to exist."
            )

        self.clips: List[Tuple[str, Path, Path]] = self._index_clips()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _index_clips(self) -> List[Tuple[str, Path, Path]]:
        rgb_videos = {p.stem: p for p in sorted(self.rgb_dir.glob("*.mp4"))}
        mask_videos = {p.stem: p for p in sorted(self.mask_dir.glob("*.mp4"))}
        common = sorted(set(rgb_videos) & set(mask_videos))
        return [(stem, rgb_videos[stem], mask_videos[stem]) for stem in common]

    def __len__(self) -> int:
        return len(self.clips)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MendeleyVideoDataset(split={self.split!r}, n_clips={len(self.clips)}, "
            f"stride={self.frame_stride})"
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def iter_clips(self) -> Iterator[Tuple[str, Path, Path]]:
        """Yield (clip_id, rgb_path, mask_path) for each clip."""
        yield from self.clips

    def iter_frames(
        self,
        clip_indices: Optional[Sequence[int]] = None,
    ) -> Iterator[VideoFramePair]:
        """Yield :class:`VideoFramePair` instances across clips.

        Parameters
        ----------
        clip_indices : iterable of int, optional
            Restrict iteration to these clip positions (``range(len(self))``
            by default).
        """
        clip_indices = range(len(self)) if clip_indices is None else clip_indices
        for ci in clip_indices:
            yield from self.iter_clip_frames(ci)

    def iter_clip_frames(self, clip_index: int) -> Iterator[VideoFramePair]:
        """Yield :class:`VideoFramePair` items from a single clip."""
        clip_id, rgb_path, mask_path = self.clips[clip_index]
        rgb_cap = cv2.VideoCapture(str(rgb_path))
        mask_cap = cv2.VideoCapture(str(mask_path))
        try:
            if not rgb_cap.isOpened() or not mask_cap.isOpened():
                raise RuntimeError(f"Failed to open clip pair {clip_id!r}")

            idx = 0
            while True:
                rgb_ok, rgb_frame = rgb_cap.read()
                mask_ok, mask_frame = mask_cap.read()
                if not (rgb_ok and mask_ok):
                    break

                if idx % self.frame_stride == 0:
                    if mask_frame.ndim == 3:
                        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_gray = mask_frame
                    mask_bin = (mask_gray > self.binarize_threshold).astype(np.uint8) * 255

                    yield VideoFramePair(
                        clip_id=clip_id,
                        split=self.split,
                        frame_index=idx,
                        rgb=rgb_frame,
                        mask=mask_bin,
                    )
                idx += 1
        finally:
            rgb_cap.release()
            mask_cap.release()


# ---------------------------------------------------------------------------
# Mask -> bounding box utilities
# ---------------------------------------------------------------------------
def mask_to_bboxes(
    mask: np.ndarray,
    *,
    min_pixels: int = 80,
    pad: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Extract one bounding box per connected component of ``mask``.

    Returns a list of ``(x1, y1, x2, y2)`` tuples in pixel coordinates.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = mask.shape

    boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, n_labels):  # 0 is background
        x, y, bw, bh, area = stats[label]
        if area < min_pixels:
            continue
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw - 1 + pad)
        y2 = min(h - 1, y + bh - 1 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def bboxes_to_yolo(
    boxes: Sequence[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    class_id: int = 0,
) -> List[str]:
    """Convert pixel ``(x1,y1,x2,y2)`` boxes to YOLO label lines.

    YOLO format (per line):  ``class cx_norm cy_norm w_norm h_norm``
    """
    h, w = image_shape[:2]
    lines: List[str] = []
    for x1, y1, x2, y2 in boxes:
        cx = 0.5 * (x1 + x2) / w
        cy = 0.5 * (y1 + y2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Quick frame extraction
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    stride: int = 1,
    max_frames: Optional[int] = None,
    suffix: str = ".jpg",
    quality: int = 95,
) -> List[Path]:
    """Decode a video and dump frames to ``out_dir``.

    Returns the list of written file paths in temporal order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    paths: List[Path] = []
    idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                path = out_dir / f"{idx:06d}{suffix}"
                if suffix.lower() in (".jpg", ".jpeg"):
                    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(str(path), frame)
                paths.append(path)
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    return paths
