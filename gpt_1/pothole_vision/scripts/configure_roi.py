#!/usr/bin/env python3
"""Interactive normalized ROI configuration tool."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.geometry.coordinates import pixel_to_normalized
from pothole_vision.roi.fixed_roi import ROIConfig, load_roi_config, save_roi_config
from pothole_vision.video.reader import VideoReader


class ROIConfigurator:
    def __init__(self, image: np.ndarray, existing: ROIConfig | None = None) -> None:
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[float, float]] = list(existing.road_polygon) if existing else []
        self.window = "Configure ROI — click points, r=reset, s=save, q=quit"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.image.shape[:2]
            self.points.append((x / w, y / h))
            self._redraw()

    def _redraw(self) -> None:
        self.display = self.image.copy()
        h, w = self.display.shape[:2]
        if self.points:
            pts = np.array([[int(x * w), int(y * h)] for x, y in self.points], dtype=np.int32)
            for p in pts:
                cv2.circle(self.display, tuple(p), 5, (0, 255, 255), -1)
            if len(pts) >= 2:
                cv2.polylines(self.display, [pts], len(self.points) >= 3, (0, 255, 255), 2)
        cv2.imshow(self.window, self.display)

    def run(self) -> list[tuple[float, float]]:
        self._redraw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.points.clear()
                self._redraw()
            elif key == ord("s"):
                break
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        return self.points


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure normalized road ROI")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", default="configs/roi.yaml", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    existing = load_roi_config(output_path) if output_path.exists() else None

    with VideoReader(args.video) as reader:
        frame = reader.read_frame(args.frame)
        if frame is None:
            raise RuntimeError(f"Cannot read frame {args.frame}")

    print("Click polygon vertices. Keys: r=reset, s=save, q=quit")
    configurator = ROIConfigurator(frame.image, existing)
    points = configurator.run()

    if len(points) < 3:
        print("Need at least 3 points. Keeping existing config.")
        return

    config = ROIConfig(road_polygon=points)
    if existing:
        config.bonnet_y = existing.bonnet_y
        config.overlay_top_y = existing.overlay_top_y
        config.manual_exclusions = existing.manual_exclusions

    save_roi_config(output_path, config)
    print(f"Saved {len(points)} points to {output_path}")


if __name__ == "__main__":
    main()
