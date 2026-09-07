"""Video reading and frame data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from pothole_vision.video.quality import FrameQuality, assess_frame_quality


@dataclass
class FrameData:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    width: int
    height: int
    quality_metrics: FrameQuality | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str = ""


class VideoReader:
    """Read MP4 frames with computed timestamps (not burned-in overlay text)."""

    def __init__(self, video_path: Path | str) -> None:
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.metadata = self._read_metadata()

    def _read_metadata(self) -> VideoMetadata:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            codec=codec,
        )

    def iter_frames(
        self,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        duration_seconds: float | None = None,
        assess_quality: bool = True,
        quality_config: dict | None = None,
    ) -> Iterator[FrameData]:
        end_frame = self.metadata.frame_count
        if duration_seconds is not None:
            end_frame = min(end_frame, start_frame + int(duration_seconds * self.metadata.fps))
        if max_frames is not None:
            end_frame = min(end_frame, start_frame + max_frames * stride)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        yielded = 0

        while idx < end_frame:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts = idx / self.metadata.fps
            h, w = frame.shape[:2]
            quality = None
            if assess_quality:
                quality = assess_frame_quality(frame, quality_config or {})
            yield FrameData(
                frame_index=idx,
                timestamp_seconds=ts,
                image=frame,
                width=w,
                height=h,
                quality_metrics=quality,
            )
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
            for _ in range(stride - 1):
                self._cap.grab()
                idx += 1
            idx += 1

    def read_frame(self, index: int) -> FrameData | None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        return FrameData(
            frame_index=index,
            timestamp_seconds=index / self.metadata.fps,
            image=frame,
            width=w,
            height=h,
        )

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
