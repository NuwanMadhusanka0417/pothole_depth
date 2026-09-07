"""Video reading and metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from pothole_vision.models import FrameData


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    fourcc: str


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
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            fourcc=fourcc,
        )

    def iter_frames(
        self,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        duration_seconds: float | None = None,
    ) -> Iterator[FrameData]:
        end_frame = self.metadata.frame_count
        if duration_seconds is not None:
            end_frame = min(
                end_frame,
                start_frame + int(duration_seconds * self.metadata.fps),
            )
        if max_frames is not None:
            end_frame = min(end_frame, start_frame + max_frames * stride)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        index = start_frame
        yielded = 0

        while index < end_frame:
            ret, image = self._cap.read()
            if not ret:
                break
            if (index - start_frame) % stride == 0:
                timestamp = index / self.metadata.fps
                yield FrameData(
                    frame_index=index,
                    timestamp_seconds=timestamp,
                    image=image,
                    width=self.metadata.width,
                    height=self.metadata.height,
                )
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1

    def read_frame(self, frame_index: int) -> FrameData | None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, image = self._cap.read()
        if not ret:
            return None
        return FrameData(
            frame_index=frame_index,
            timestamp_seconds=frame_index / self.metadata.fps,
            image=image,
            width=self.metadata.width,
            height=self.metadata.height,
        )

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
