"""Structure-from-Motion runner.

Two implementations live behind one interface:

  * **COLMAP** via subprocess. Highest quality. Requires the ``colmap``
    binary to be installed and in PATH (or configurable in
    ``sfm_config.yaml``).
  * **OpenCV** two-view fallback. Activates automatically when COLMAP
    is unavailable. Much less accurate but lets the rest of the
    pipeline run end-to-end.

In both cases we return a unified :class:`SfMResult` containing 3D
points (in an arbitrary scale) and per-image extrinsics so the scale
recovery module can later turn things into metres.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.camera_calibration import CameraIntrinsics
from ..utils.logging_utils import get_logger
from . import feature_tracking as ft

_logger = get_logger("reconstruction.sfm")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CameraPose:
    """Camera pose in world coordinates (R | t)."""

    image_id: int
    name: str
    R: np.ndarray   # (3, 3)
    t: np.ndarray   # (3,)

    @property
    def position(self) -> np.ndarray:
        """Camera center in world coordinates: ``-R^T t``."""
        return -self.R.T @ self.t


@dataclass
class SfMResult:
    """Output of an SfM reconstruction."""

    points_3d: np.ndarray                       # (N, 3)
    point_colors: Optional[np.ndarray] = None   # (N, 3) uint8
    poses: List[CameraPose] = field(default_factory=list)
    K: Optional[np.ndarray] = None              # used intrinsics
    image_paths: List[Path] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None  # (W, H)
    method: str = "unknown"                     # 'colmap' | 'opencv-two-view' | ...
    notes: str = ""

    @property
    def num_points(self) -> int:
        return int(self.points_3d.shape[0]) if self.points_3d.size else 0

    def save_ply(self, path: str | Path) -> Path:
        """Write a simple ASCII PLY of the point cloud."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self.num_points
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        colors = self.point_colors
        if colors is None or len(colors) != n:
            colors = np.full((n, 3), 200, dtype=np.uint8)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
            for (x, y, z), (r, g, b) in zip(self.points_3d, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        return path


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class SfMRunner:
    """Build an :class:`SfMResult` from a list of images / a video clip."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        colmap_executable: str = "colmap",
        num_frames: int = 8,
        frame_stride: int = 2,
        max_image_dim: int = 1600,
        single_camera: bool = True,
        camera_model: str = "SIMPLE_RADIAL",
        use_gpu: bool = True,
        matcher_type: str = "exhaustive",
        enable_opencv_fallback: bool = True,
        opencv_num_features: int = 2000,
        opencv_ratio: float = 0.75,
    ) -> None:
        self.intrinsics = intrinsics
        self.colmap_executable = colmap_executable
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.max_image_dim = int(max_image_dim)
        self.single_camera = bool(single_camera)
        self.camera_model = camera_model
        self.use_gpu = bool(use_gpu)
        self.matcher_type = matcher_type
        self.enable_opencv_fallback = bool(enable_opencv_fallback)
        self.opencv_num_features = int(opencv_num_features)
        self.opencv_ratio = float(opencv_ratio)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def run_from_video(
        self,
        video_path: str | Path,
        *,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        work_dir: str | Path | None = None,
    ) -> SfMResult:
        """Extract frames from a video clip and run SfM on them."""
        frames = self._extract_frames(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        return self.run_from_frames(frames, work_dir=work_dir)

    def run_from_frames(
        self,
        frames: Sequence[np.ndarray] | Sequence[Path],
        work_dir: str | Path | None = None,
    ) -> SfMResult:
        """Run SfM from in-memory frames or pre-saved image paths."""
        cleanup_dir: Optional[Path] = None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="sfm_yolo_"))
            cleanup_dir = work_dir
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            image_paths = self._save_frames(frames, work_dir / "images")

            if self._colmap_available():
                _logger.info("Running COLMAP on %d images", len(image_paths))
                try:
                    return self._run_colmap(image_paths, work_dir)
                except Exception as exc:  # pragma: no cover - depends on env
                    _logger.warning("COLMAP failed (%s); falling back to OpenCV", exc)
                    if not self.enable_opencv_fallback:
                        raise
            elif not self.enable_opencv_fallback:
                raise RuntimeError(
                    "COLMAP not found and OpenCV fallback disabled in config."
                )
            else:
                _logger.info("COLMAP not available - using OpenCV two-view fallback")

            return self._run_opencv_fallback(image_paths)
        finally:
            if cleanup_dir is not None:
                # keep results around for debugging unless the caller
                # explicitly asked for a temp dir
                pass

    # ------------------------------------------------------------------
    # Helpers - frame extraction / saving
    # ------------------------------------------------------------------
    def _extract_frames(
        self,
        video_path: str | Path,
        *,
        start_frame: int,
        end_frame: Optional[int],
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = total - 1 if end_frame is None else min(end_frame, total - 1)
        end_frame = max(end_frame, start_frame)

        # Pick `num_frames` evenly spaced indices inside the window
        if self.num_frames >= (end_frame - start_frame + 1):
            chosen = list(range(start_frame, end_frame + 1))
        else:
            chosen = list(
                np.linspace(start_frame, end_frame, self.num_frames).round().astype(int)
            )

        frames: List[np.ndarray] = []
        try:
            for idx in chosen:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        finally:
            cap.release()
        return frames

    def _save_frames(
        self,
        frames: Sequence[np.ndarray] | Sequence[Path],
        out_dir: Path,
    ) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise ValueError("no frames provided to SfMRunner")
        paths: List[Path] = []
        if isinstance(frames[0], (str, Path)):
            for i, p in enumerate(frames):
                src = Path(p)
                dst = out_dir / f"frame_{i:04d}{src.suffix or '.jpg'}"
                shutil.copyfile(src, dst)
                paths.append(dst)
        else:
            for i, frame in enumerate(frames):
                resized = self._maybe_resize(frame)
                dst = out_dir / f"frame_{i:04d}.jpg"
                cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                paths.append(dst)
        return paths

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= self.max_image_dim:
            return frame
        scale = self.max_image_dim / long_side
        new_size = (int(round(w * scale)), int(round(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # COLMAP back-end
    # ------------------------------------------------------------------
    def _colmap_available(self) -> bool:
        if shutil.which(self.colmap_executable) is not None:
            return True
        # Allow absolute paths even if not in PATH
        return Path(self.colmap_executable).is_file()

    def _run_colmap(self, image_paths: List[Path], work_dir: Path) -> SfMResult:
        db_path = work_dir / "database.db"
        sparse_dir = work_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        image_dir = image_paths[0].parent
        gpu = "1" if self.use_gpu else "0"

        # 1) Feature extraction
        self._run_subprocess(
            [
                self.colmap_executable, "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
                "--ImageReader.camera_model", self.camera_model,
                "--ImageReader.single_camera", "1" if self.single_camera else "0",
                "--SiftExtraction.use_gpu", gpu,
            ]
        )

        # 2) Matching
        matcher_cmd = (
            "exhaustive_matcher" if self.matcher_type == "exhaustive"
            else "sequential_matcher"
        )
        self._run_subprocess(
            [
                self.colmap_executable, matcher_cmd,
                "--database_path", str(db_path),
                "--SiftMatching.use_gpu", gpu,
            ]
        )

        # 3) Mapping
        self._run_subprocess(
            [
                self.colmap_executable, "mapper",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
            ]
        )

        # COLMAP creates sparse/0/ (and possibly 1, 2, ...). Pick the largest.
        models = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
        if not models:
            raise RuntimeError("COLMAP produced no sparse model")
        chosen = max(models, key=lambda p: _safe_filesize(p / "points3D.bin"))
        return self._parse_colmap_model(chosen, image_paths)

    def _run_subprocess(self, cmd: List[str]) -> None:
        _logger.debug("COLMAP $ %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"COLMAP command failed ({completed.returncode}): {' '.join(cmd)}\n"
                f"stderr:\n{completed.stderr[-1000:]}"
            )

    # ------------------------------------------------------------------
    # COLMAP model parsing (binary OR text)
    # ------------------------------------------------------------------
    def _parse_colmap_model(
        self,
        model_dir: Path,
        image_paths: List[Path],
    ) -> SfMResult:
        if (model_dir / "cameras.bin").exists():
            cameras = _read_cameras_binary(model_dir / "cameras.bin")
            images = _read_images_binary(model_dir / "images.bin")
            points = _read_points3d_binary(model_dir / "points3D.bin")
        else:
            cameras = _read_cameras_text(model_dir / "cameras.txt")
            images = _read_images_text(model_dir / "images.txt")
            points = _read_points3d_text(model_dir / "points3D.txt")

        cam = next(iter(cameras.values()))
        K = _camera_to_K(cam)

        poses: List[CameraPose] = []
        name_to_path = {p.name: p for p in image_paths}
        for img in images.values():
            R = _qvec_to_rotmat(img["qvec"])
            t = np.asarray(img["tvec"], dtype=np.float64)
            poses.append(
                CameraPose(
                    image_id=int(img["image_id"]),
                    name=str(img["name"]),
                    R=R,
                    t=t,
                )
            )
        poses.sort(key=lambda p: p.name)

        if points:
            xyz = np.array([p["xyz"] for p in points.values()], dtype=np.float64)
            rgb = np.array([p["rgb"] for p in points.values()], dtype=np.uint8)
        else:
            xyz = np.zeros((0, 3), dtype=np.float64)
            rgb = np.zeros((0, 3), dtype=np.uint8)

        result = SfMResult(
            points_3d=xyz,
            point_colors=rgb,
            poses=poses,
            K=K,
            image_paths=[name_to_path.get(p.name, Path(p.name)) for p in poses],
            method="colmap",
            notes=f"model_dir={model_dir}",
        )
        if poses and poses[0].name in name_to_path:
            img = cv2.imread(str(name_to_path[poses[0].name]))
            if img is not None:
                result.image_size = (img.shape[1], img.shape[0])
        return result

    # ------------------------------------------------------------------
    # OpenCV fallback (two-view triangulation)
    # ------------------------------------------------------------------
    def _run_opencv_fallback(self, image_paths: List[Path]) -> SfMResult:
        if len(image_paths) < 2:
            raise RuntimeError("OpenCV fallback needs >= 2 images")

        # Use the first and last image (largest baseline)
        img_a = cv2.imread(str(image_paths[0]))
        img_b = cv2.imread(str(image_paths[-1]))
        if img_a is None or img_b is None:
            raise RuntimeError("Failed to read fallback images")

        K = self._scaled_K_for_image(img_a)

        feats_a = ft.detect_features(img_a, detector="sift", max_features=self.opencv_num_features)
        feats_b = ft.detect_features(img_b, detector="sift", max_features=self.opencv_num_features)
        pts_a, pts_b = ft.match_features(feats_a, feats_b, ratio=self.opencv_ratio)
        if pts_a.shape[0] < 16:
            _logger.warning("Only %d matches - SfM result will be unreliable", pts_a.shape[0])

        pts3, R, t = ft.triangulate_two_views(pts_a, pts_b, K)
        if pts3 is None:
            return SfMResult(
                points_3d=np.zeros((0, 3)),
                K=K,
                image_paths=image_paths,
                image_size=(img_a.shape[1], img_a.shape[0]),
                method="opencv-two-view",
                notes="triangulation failed",
            )

        # Sample colours from the first image at the matched pixel locations
        colors = []
        for x, y in pts_a:
            xi = int(np.clip(round(x), 0, img_a.shape[1] - 1))
            yi = int(np.clip(round(y), 0, img_a.shape[0] - 1))
            b, g, r = img_a[yi, xi]
            colors.append((int(r), int(g), int(b)))
        colors = np.asarray(colors[: pts3.shape[0]], dtype=np.uint8)

        poses = [
            CameraPose(image_id=0, name=image_paths[0].name, R=np.eye(3), t=np.zeros(3)),
            CameraPose(image_id=1, name=image_paths[-1].name, R=R, t=t.reshape(3)),
        ]
        return SfMResult(
            points_3d=pts3,
            point_colors=colors,
            poses=poses,
            K=K,
            image_paths=[image_paths[0], image_paths[-1]],
            image_size=(img_a.shape[1], img_a.shape[0]),
            method="opencv-two-view",
            notes=f"matches={pts_a.shape[0]}, points={pts3.shape[0]}",
        )

    def _scaled_K_for_image(self, img: np.ndarray) -> np.ndarray:
        K = self.intrinsics.K().copy()
        h, w = img.shape[:2]
        ow, oh = self.intrinsics.width, self.intrinsics.height
        if (w, h) != (ow, oh):
            sx, sy = w / ow, h / oh
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
        return K


# ---------------------------------------------------------------------------
# COLMAP file readers
# ---------------------------------------------------------------------------
def _safe_filesize(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _camera_to_K(cam: dict) -> np.ndarray:
    """COLMAP camera dict -> 3x3 intrinsic matrix.

    Supports the common SIMPLE_RADIAL / PINHOLE models.
    """
    model = cam["model"]
    p = cam["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    if model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


# --- text format -----------------------------------------------------------
def _read_cameras_text(path: Path) -> Dict[int, dict]:
    cameras: Dict[int, dict] = {}
    for line in _iter_text_lines(path):
        parts = line.split()
        cid = int(parts[0])
        cameras[cid] = {
            "id": cid,
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(x) for x in parts[4:]],
        }
    return cameras


def _read_images_text(path: Path) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    lines = list(_iter_text_lines(path))
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]
        images[image_id] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
        }
        i += 2  # skip the points2D line
    return images


def _read_points3d_text(path: Path) -> Dict[int, dict]:
    points: Dict[int, dict] = {}
    for line in _iter_text_lines(path):
        parts = line.split()
        pid = int(parts[0])
        xyz = [float(x) for x in parts[1:4]]
        rgb = [int(x) for x in parts[4:7]]
        error = float(parts[7])
        points[pid] = {"id": pid, "xyz": xyz, "rgb": rgb, "error": error}
    return points


def _iter_text_lines(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


# --- binary format ---------------------------------------------------------
def _read_next_bytes(fp, n_bytes: int, fmt: str):
    return struct.unpack(fmt, fp.read(n_bytes))


def _read_cameras_binary(path: Path) -> Dict[int, dict]:
    cameras: Dict[int, dict] = {}
    with open(path, "rb") as fp:
        n = _read_next_bytes(fp, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id, width, height = _read_next_bytes(fp, 24, "iiQQ")
            num_params = _COLMAP_CAMERA_MODEL_NUM_PARAMS.get(model_id, 0)
            params = list(_read_next_bytes(fp, 8 * num_params, "d" * num_params))
            cameras[cam_id] = {
                "id": cam_id,
                "model": _COLMAP_CAMERA_MODEL_NAMES.get(model_id, str(model_id)),
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_images_binary(path: Path) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    with open(path, "rb") as fp:
        n = _read_next_bytes(fp, 8, "Q")[0]
        for _ in range(n):
            data = _read_next_bytes(fp, 64, "idddddddi")
            image_id = data[0]
            qvec = list(data[1:5])
            tvec = list(data[5:8])
            camera_id = data[8]
            # null-terminated name
            chars: List[bytes] = []
            while True:
                ch = fp.read(1)
                if ch == b"\x00":
                    break
                chars.append(ch)
            name = b"".join(chars).decode("utf-8")
            num_p2d = _read_next_bytes(fp, 8, "Q")[0]
            fp.read(num_p2d * 24)  # skip 2D-3D associations
            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def _read_points3d_binary(path: Path) -> Dict[int, dict]:
    points: Dict[int, dict] = {}
    with open(path, "rb") as fp:
        n = _read_next_bytes(fp, 8, "Q")[0]
        for _ in range(n):
            data = _read_next_bytes(fp, 43, "QdddBBBd")
            pid = data[0]
            xyz = [data[1], data[2], data[3]]
            rgb = [data[4], data[5], data[6]]
            error = data[7]
            track_len = _read_next_bytes(fp, 8, "Q")[0]
            fp.read(track_len * 8)
            points[pid] = {"id": pid, "xyz": xyz, "rgb": rgb, "error": error}
    return points


_COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}
_COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
}
