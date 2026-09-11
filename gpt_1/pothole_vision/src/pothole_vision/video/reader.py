"""Video reading and frame data models."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from pothole_vision.video.quality import FrameQuality, assess_frame_quality


def open_video_capture(path: Path) -> cv2.VideoCapture:
    """
    Open a video file for decoding via OpenCV.

    Prefer FFmpeg backend (handles many MP4/H.264 + MP3/AAC dashcam containers).
    Fall back to OpenCV default if FFmpeg cannot open the file.
    """
    path_str = str(path)
    cap = cv2.VideoCapture(path_str, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()

    cap = cv2.VideoCapture(path_str)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap
        cap.release()

    raise RuntimeError("opencv_capture_failed")


@dataclass
class FrameData:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    width: int
    height: int
    quality_metrics: Optional[FrameQuality] = None

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
    backend: str = "opencv"


class _FfmpegFrameDecoder:
    """Decode frames by streaming raw BGR24 from ffmpeg."""

    def __init__(self, path: Path, metadata: VideoMetadata) -> None:
        self.path = path
        self.metadata = metadata
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = metadata.width * metadata.height * 3

    def _ensure_pipe(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 4,
        )
        return self._proc

    def read_next_frame(self) -> Optional[np.ndarray]:
        proc = self._ensure_pipe()
        if proc.stdout is None:
            return None
        raw = proc.stdout.read(self._frame_bytes)
        if not raw or len(raw) < self._frame_bytes:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.metadata.height, self.metadata.width, 3)
        )
        return frame.copy()

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait(timeout=5)
            self._proc = None


class VideoReader:
    """Read MP4 frames with computed timestamps (not burned-in overlay text)."""

    def __init__(self, video_path: Path | str) -> None:
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        from pothole_vision.video.ffprobe_io import (
            ffprobe_metadata,
            file_container_hint,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg: Optional[_FfmpegFrameDecoder] = None
        self._opencv_frame_index = 0

        hint = file_container_hint(self.path)
        try:
            self._cap = open_video_capture(self.path)
            self.metadata = self._read_metadata_opencv()
            self.metadata.backend = "opencv"
            if self.metadata.width <= 0 or self.metadata.height <= 0:
                raise RuntimeError("invalid_opencv_dimensions")
        except RuntimeError:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            meta_dict, ff_err = ffprobe_metadata(self.path)
            if meta_dict is None:
                raise RuntimeError(
                    f"Cannot open video: {self.path}\n"
                    f"  On disk: {hint}\n"
                    f"  OpenCV: could not decode first frame\n"
                    f"  ffprobe: {ff_err}\n"
                    "If files were copied from OneDrive/Windows, re-copy full MP4s (not placeholders). "
                    "On Gadi: module load ffmpeg. Fix with: "
                    f"ffmpeg -i {self.path.name} -c:v libx264 -c:a aac fixed_{self.path.name}"
                ) from None

            self.metadata = VideoMetadata(backend="ffmpeg", **meta_dict)
            self._ffmpeg = _FfmpegFrameDecoder(self.path, self.metadata)

    def _read_metadata_opencv(self) -> VideoMetadata:
        assert self._cap is not None
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
        max_frames: Optional[int] = None,
        stride: int = 1,
        duration_seconds: Optional[float] = None,
        assess_quality: bool = True,
        quality_config: Optional[dict] = None,
    ) -> Iterator[FrameData]:
        end_frame = self.metadata.frame_count or 10**9
        if self.metadata.frame_count > 0:
            end_frame = self.metadata.frame_count
        if duration_seconds is not None:
            end_frame = min(end_frame, start_frame + int(duration_seconds * self.metadata.fps))
        if max_frames is not None:
            end_frame = min(end_frame, start_frame + max_frames * stride)

        if self._cap is not None:
            yield from self._iter_frames_opencv(
                start_frame, end_frame, max_frames, stride,
                assess_quality, quality_config or {},
            )
        elif self._ffmpeg is not None:
            yield from self._iter_frames_ffmpeg(
                start_frame, end_frame, max_frames, stride,
                assess_quality, quality_config or {},
            )

    def _iter_frames_opencv(
        self,
        start_frame: int,
        end_frame: int,
        max_frames: Optional[int],
        stride: int,
        assess_quality: bool,
        quality_config: dict,
    ) -> Iterator[FrameData]:
        assert self._cap is not None
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
                quality = assess_frame_quality(frame, quality_config)
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

    def _iter_frames_ffmpeg(
        self,
        start_frame: int,
        end_frame: int,
        max_frames: Optional[int],
        stride: int,
        assess_quality: bool,
        quality_config: dict,
    ) -> Iterator[FrameData]:
        assert self._ffmpeg is not None
        idx = 0
        yielded = 0
        while idx < end_frame:
            frame = self._ffmpeg.read_next_frame()
            if frame is None:
                break
            if idx >= start_frame and ((idx - start_frame) % stride == 0):
                ts = idx / self.metadata.fps
                quality = None
                if assess_quality:
                    quality = assess_frame_quality(frame, quality_config)
                yield FrameData(
                    frame_index=idx,
                    timestamp_seconds=ts,
                    image=frame,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    quality_metrics=quality,
                )
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1

    def read_frame(self, index: int) -> Optional[FrameData]:
        if self._cap is not None:
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

        if self._ffmpeg is not None:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(index / self.metadata.fps),
                "-i",
                str(self.path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
            raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            expected = self.metadata.width * self.metadata.height * 3
            if len(raw) < expected:
                return None
            frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(
                (self.metadata.height, self.metadata.width, 3)
            )
            return FrameData(
                frame_index=index,
                timestamp_seconds=index / self.metadata.fps,
                image=frame.copy(),
                width=self.metadata.width,
                height=self.metadata.height,
            )
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ffmpeg is not None:
            self._ffmpeg.close()
            self._ffmpeg = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
