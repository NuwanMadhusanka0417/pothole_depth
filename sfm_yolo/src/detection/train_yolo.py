"""Build a YOLO-format dataset from the Mendeley pothole *video* dataset
and fine-tune YOLOv8 on it.

The Mendeley dataset ships as paired video clips, where each ``rgb`` clip
has a matching ``mask`` clip. We turn this into the standard YOLO layout:

    yolo_root/
        dataset.yaml
        images/{train,val,test}/<clip>_<frame>.jpg
        labels/{train,val,test}/<clip>_<frame>.txt

Run from the project root::

    python -m sfm_yolo.src.detection.train_yolo --build
    python -m sfm_yolo.src.detection.train_yolo --train
    python -m sfm_yolo.src.detection.train_yolo --build --train
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import yaml
from tqdm import tqdm

from ..utils.data_loader import (
    MendeleyVideoDataset,
    bboxes_to_yolo,
    mask_to_bboxes,
)
from ..utils.logging_utils import get_logger

_logger = get_logger("detection.train")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_yolo_dataset(
    source_root: str | Path,
    yolo_root: str | Path,
    *,
    frame_stride: int = 5,
    min_mask_pixels: int = 80,
    class_names: Iterable[str] = ("pothole",),
    image_quality: int = 92,
    overwrite: bool = False,
) -> Path:
    """Convert the Mendeley video dataset to YOLO format on disk.

    Returns the path of the generated ``dataset.yaml``.
    """
    source_root = Path(source_root)
    yolo_root = Path(yolo_root)
    class_names = list(class_names)

    if yolo_root.exists() and overwrite:
        _logger.warning("Removing existing YOLO dataset at %s", yolo_root)
        shutil.rmtree(yolo_root)

    splits_present = [s for s in ("train", "val", "test") if (source_root / s).is_dir()]
    if not splits_present:
        raise FileNotFoundError(
            f"Source root {source_root!s} contains no train/val/test sub-folders"
        )

    yolo_root.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for split in splits_present:
        img_dir = yolo_root / "images" / split
        lbl_dir = yolo_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ds = MendeleyVideoDataset(
            source_root, split=split, frame_stride=frame_stride
        )
        _logger.info("Building split=%s, n_clips=%d", split, len(ds))

        n_frames = 0
        n_with_box = 0
        for frame in tqdm(ds.iter_frames(), desc=f"{split}", unit="frame"):
            stem = f"{frame.clip_id}_{frame.frame_index:06d}"
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(
                str(img_path),
                frame.rgb,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(image_quality)],
            )

            boxes = mask_to_bboxes(frame.mask, min_pixels=min_mask_pixels)
            lines = bboxes_to_yolo(boxes, frame.rgb.shape, class_id=0)
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            n_frames += 1
            if lines:
                n_with_box += 1
        counts[split] = n_frames
        _logger.info(
            "  split=%s wrote %d frames (%d with at least one bbox, %.1f%%)",
            split,
            n_frames,
            n_with_box,
            (100.0 * n_with_box / max(1, n_frames)),
        )

    dataset_yaml = yolo_root / "dataset.yaml"
    payload = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val" if "val" in splits_present else "images/train",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    if "test" in splits_present:
        payload["test"] = "images/test"
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _logger.info("Wrote %s", dataset_yaml)
    _logger.info("Per-split counts: %s", counts)
    return dataset_yaml


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_yolo(
    dataset_yaml: str | Path,
    *,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    patience: int = 15,
    device: str = "",
    workers: int = 4,
    seed: int = 42,
    project_dir: str | Path = "outputs/yolo_runs",
    run_name: str = "pothole_yolov8n",
    augment: dict | None = None,
) -> Path:
    """Fine-tune YOLOv8 on the YOLO-format pothole dataset.

    Returns the path to the best weights (``best.pt``).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required to train YOLO models. pip install ultralytics"
        ) from exc

    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(dataset_yaml)

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("Initialising YOLO from %s", base_model)
    model = YOLO(str(base_model))

    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        seed=seed,
        workers=workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    if device:
        train_kwargs["device"] = device
    if augment:
        train_kwargs.update(
            {
                k: v
                for k, v in augment.items()
                if k in {"hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                         "scale", "shear", "fliplr", "mosaic"}
            }
        )

    _logger.info("Starting training: %s", train_kwargs)
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", project_dir / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():  # fallback - some ultralytics versions
        cand = list((save_dir / "weights").glob("*.pt"))
        if cand:
            best = cand[0]
    _logger.info("Best weights: %s", best)
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_yolo_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & train YOLO on the Mendeley pothole dataset.")
    parser.add_argument("--config", default="sfm_yolo/configs/yolo_config.yaml")
    parser.add_argument("--build", action="store_true", help="Build the YOLO-format dataset")
    parser.add_argument("--train", action="store_true", help="Run YOLO training")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing YOLO dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--device", default=None, help="Override device, e.g. 'cpu' or '0'")
    args = parser.parse_args()

    cfg = _load_yolo_config(args.config)
    ds_cfg = cfg["dataset"]

    if not (args.build or args.train):
        parser.error("Specify at least --build or --train")

    dataset_yaml = Path(ds_cfg["yolo_root"]) / "dataset.yaml"

    if args.build:
        dataset_yaml = build_yolo_dataset(
            ds_cfg["source_root"],
            ds_cfg["yolo_root"],
            frame_stride=ds_cfg.get("frame_stride", 5),
            min_mask_pixels=ds_cfg.get("min_mask_pixels", 80),
            class_names=ds_cfg.get("class_names", ["pothole"]),
            overwrite=args.overwrite,
        )

    if args.train:
        train_yolo(
            dataset_yaml,
            base_model=cfg.get("base_model", "yolov8n.pt"),
            epochs=args.epochs or cfg.get("epochs", 50),
            batch_size=cfg.get("batch_size", 16),
            img_size=cfg.get("img_size", 640),
            patience=cfg.get("patience", 15),
            device=args.device if args.device is not None else cfg.get("device", ""),
            workers=cfg.get("workers", 4),
            seed=cfg.get("seed", 42),
            project_dir=cfg.get("project_dir", "outputs/yolo_runs"),
            run_name=cfg.get("run_name", "pothole_yolov8n"),
            augment=cfg.get("augment"),
        )


if __name__ == "__main__":
    main()
