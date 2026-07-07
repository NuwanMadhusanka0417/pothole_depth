"""Loader for the *misensorkit* phone capture format (RGB + IMU only).

A misensorkit recording is a folder with one file per frame in parallel
sub-directories::

    misensorkit_YYYYMMDD_HHMMSS/
        rgb/      frame_000001.jpg ...   (RGB image)
        imu/      frame_000001.json ...  (gravity, attitude, gyro, accel)
        camera/   frame_000001.json ...  (intrinsics + extrinsics)
        depth/    frame_000001.bin ...   (metric depth - NOT used here)
        gps/      frame_000001.json ...  (lat/lon - NOT used here)
        metadata/ frame_000001.json ...

This loader deliberately reads **only the RGB and IMU streams**, plus the
camera *intrinsics* (focal length / principal point) which are a fixed
calibration property, not a per-frame sensor measurement. It does **not**
read the depth maps, GPS, or camera *extrinsics* (the ARKit visual-inertial
pose) -- the whole point of the RGB+IMU pipeline is to estimate depth
without those.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .camera_calibration import CameraIntrinsics
from .logging_utils import get_logger

_logger = get_logger("misensorkit")

_FRAME_RE = re.compile(r"frame_(\d+)")


def _frame_id(path: Path) -> Optional[str]:
    m = _FRAME_RE.search(path.stem)
    return m.group(1) if m else None


def _normalize_name(name: str) -> str:
    """Collapse every kind of whitespace to a single ASCII space, lowercased.

    misensorkit exports name folders with a narrow no-break space (U+202F)
    before 'am'/'pm'. Typing a normal space on the command line then fails
    to match. Normalising any Unicode space separator (category ``Zs``) and
    the ASCII whitespace controls makes folder lookup robust to this.
    """
    chars = [
        " " if (unicodedata.category(c) == "Zs" or c in "\t\n\r\f\v") else c
        for c in name
    ]
    collapsed = re.sub(r" +", " ", "".join(chars)).strip().lower()
    # Also drop a trailing am/pm token: the export sometimes keeps it
    # (separated by U+202F) and sometimes the folder gets renamed without it.
    return re.sub(r"\s*(am|pm)$", "", collapsed)


def _resolve_recording_folder(folder: Path) -> Path:
    """Return ``folder`` if it exists, else find a whitespace-equivalent sibling."""
    if folder.is_dir():
        return folder
    parent = folder.parent
    if parent.is_dir():
        target = _normalize_name(folder.name)
        matches = [
            p for p in parent.iterdir()
            if p.is_dir() and _normalize_name(p.name) == target
        ]
        if len(matches) == 1:
            _logger.warning(
                "Resolved %r -> %r (whitespace/encoding mismatch in folder name)",
                folder.name, matches[0].name,
            )
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous recording folder {folder.name!r}; candidates: "
                f"{[m.name for m in matches]}"
            )
    raise FileNotFoundError(f"Recording folder not found: {folder}")


@dataclass
class MiSensorKitFrame:
    """One time-aligned RGB + IMU frame."""

    index: int          # 0-based position in the sequence
    frame_id: str       # zero-padded id shared across streams, e.g. "000001"
    rgb_path: Path
    imu: dict           # raw IMU record (gravity_*, attitude_*, gyro_*, ...)
    intrinsics: Optional[dict] = None   # raw camera-JSON "intrinsics" block

    def load_rgb(self) -> np.ndarray:
        """Load the RGB frame as an OpenCV BGR uint8 array."""
        img = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read RGB frame: {self.rgb_path}")
        return img


class MiSensorKitDataset:
    """Index and load the RGB + IMU streams of a misensorkit recording.

    Parameters
    ----------
    folder : path-like
        The recording directory (the one containing ``rgb/`` and ``imu/``).
    require_intrinsics : bool
        If True, also index the ``camera/`` intrinsics. If the folder has
        no ``camera/`` stream, :meth:`camera_intrinsics` will require an
        explicit fallback instead.
    """

    def __init__(self, folder: str | Path, *, require_intrinsics: bool = True) -> None:
        self.folder = _resolve_recording_folder(Path(folder))
        self.rgb_dir = self.folder / "rgb"
        self.imu_dir = self.folder / "imu"
        self.camera_dir = self.folder / "camera"

        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing rgb/ directory in {self.folder}")
        if not self.imu_dir.is_dir():
            raise FileNotFoundError(f"Missing imu/ directory in {self.folder}")

        self._has_camera = self.camera_dir.is_dir()
        if require_intrinsics and not self._has_camera:
            _logger.warning(
                "No camera/ directory in %s - intrinsics must be supplied explicitly.",
                self.folder,
            )

        self.frames: List[MiSensorKitFrame] = self._index_frames()
        if not self.frames:
            raise RuntimeError(f"No paired RGB + IMU frames found in {self.folder}")
        _logger.info("Indexed %d RGB+IMU frames from %s", len(self.frames), self.folder)

    # ------------------------------------------------------------------
    def _index_frames(self) -> List[MiSensorKitFrame]:
        rgb = {fid: p for p in sorted(self.rgb_dir.glob("frame_*.jpg"))
               if (fid := _frame_id(p)) is not None}
        imu = {fid: p for p in sorted(self.imu_dir.glob("frame_*.json"))
               if (fid := _frame_id(p)) is not None}
        cam = {}
        if self._has_camera:
            cam = {fid: p for p in sorted(self.camera_dir.glob("frame_*.json"))
                   if (fid := _frame_id(p)) is not None}

        common = sorted(set(rgb) & set(imu))
        frames: List[MiSensorKitFrame] = []
        for i, fid in enumerate(common):
            with imu[fid].open("r", encoding="utf-8") as f:
                imu_rec = json.load(f)
            intr = None
            if fid in cam:
                with cam[fid].open("r", encoding="utf-8") as f:
                    intr = json.load(f).get("intrinsics")
            frames.append(
                MiSensorKitFrame(
                    index=i,
                    frame_id=fid,
                    rgb_path=rgb[fid],
                    imu=imu_rec,
                    intrinsics=intr,
                )
            )
        return frames

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[MiSensorKitFrame]:
        return iter(self.frames)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MiSensorKitDataset(folder={self.folder.name!r}, n_frames={len(self.frames)})"

    # ------------------------------------------------------------------
    def camera_intrinsics(
        self,
        camera_height_m: float,
        *,
        pitch_deg: float = 0.0,
        fallback: Optional[CameraIntrinsics] = None,
    ) -> CameraIntrinsics:
        """Build :class:`CameraIntrinsics` for the RGB frames.

        Reads ``fx, fy, cx, cy`` and the image size from the first
        available camera JSON (a fixed calibration property). The metric
        ``camera_height_m`` -- the scale anchor for the whole pipeline --
        must be supplied by the caller; it cannot be read from the data.

        ``pitch_deg`` is left at 0 by default because the per-frame pitch
        is taken from the IMU at run time, not from this static value.
        """
        intr = next((f.intrinsics for f in self.frames if f.intrinsics), None)
        if intr is None:
            if fallback is not None:
                return fallback
            raise RuntimeError(
                "No camera intrinsics found in the recording and no fallback "
                "supplied. Provide a camera_calib.yaml fallback."
            )
        width = int(intr.get("image_width"))
        height = int(intr.get("image_height"))
        focal = float(0.5 * (float(intr["fx"]) + float(intr["fy"])))
        return CameraIntrinsics(
            camera_height_m=float(camera_height_m),
            focal_length_px=focal,
            principal_point=(float(intr["cx"]), float(intr["cy"])),
            image_size=(width, height),
            pitch_deg=float(pitch_deg),
        )

    def camera_family_hint(self) -> Optional[str]:
        """Best-effort read of the recording's camera ('frontCamera'/'backCamera').

        Reads ``metadata/`` if present. This is only a *hint* used to warn
        the user about the optical-axis convention; it is not part of the
        RGB+IMU estimation path.
        """
        meta_dir = self.folder / "metadata"
        if not meta_dir.is_dir():
            return None
        first = next(iter(sorted(meta_dir.glob("frame_*.json"))), None)
        if first is None:
            return None
        try:
            with first.open("r", encoding="utf-8") as f:
                return json.load(f).get("camera_family")
        except Exception:  # pragma: no cover
            return None
