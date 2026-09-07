#!/usr/bin/env python3
"""Interactive ROI configuration tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pothole_vision.roi.fixed_roi import ROIConfig
from pothole_vision.utils.math import pixel_to_normalized
from pothole_vision.video.reader import VideoReader


class ROIConfigurator:
    def __init__(self, image, output_path: Path) -> None:
        self.image = image.copy()
        self.display = image.copy()
        self.points: list[tuple[int, int]] = []
        self.output_path = output_path
        self.window = "Configure ROI - click points, r=reset, s=save, q=quit"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse)

    def _mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._redraw()

    def _redraw(self) -> None:
        self.display = self.image.copy()
        if self.points:
            pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(self.display, [pts], False, (0, 255, 255), 2)
            for p in self.points:
                cv2.circle(self.display, p, 4, (0, 0, 255), -1)
        cv2.imshow(self.window, self.display)

    def run(self) -> None:
        cv2.imshow(self.window, self.display)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.points.clear()
                self._redraw()
            elif key == ord("s") and len(self.points) >= 3:
                h, w = self.image.shape[:2]
                norm = pixel_to_normalized(np.array(self.points, dtype=float), w, h)
                cfg = ROIConfig(road_polygon=norm.tolist())
                cfg.save(self.output_path)
                print(f"Saved {len(self.points)} points to {self.output_path}")
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure normalized road ROI")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "roi.yaml")
    args = parser.parse_args()

    with VideoReader(args.video) as reader:
        frame = reader.read_frame(args.frame)
        if frame is None:
            raise RuntimeError(f"Cannot read frame {args.frame}")

    ROIConfigurator(frame.image, args.output).run()


if __name__ == "__main__":
    main()
