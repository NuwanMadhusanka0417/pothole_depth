"""Shared CLI arguments for single-video or batch input."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from pothole_vision.utils.config import AppConfig, resolve_path
from pothole_vision.utils.video_paths import collect_videos_from_args


def add_video_input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", type=Path, help="Process one video file")
    group.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of videos (default: configs paths.input_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (per-video subfolders created inside)",
    )


def resolve_videos(args: argparse.Namespace, config: AppConfig) -> list[Path]:
    """Build video list from parsed args and config."""
    input_dir = getattr(args, "input_dir", None)
    video = getattr(args, "video", None)
    # When neither --video nor --input-dir: process all videos in config input_dir
    if video is None and input_dir is None:
        return collect_videos_from_args(
            PROJECT_ROOT,
            video=None,
            input_dir=None,
            config_input_dir=config.paths.input_dir,
            config_glob=config.paths.input_glob,
            all_videos=True,
        )
    if video is not None:
        return collect_videos_from_args(
            PROJECT_ROOT,
            video=video,
            input_dir=None,
            config_input_dir=config.paths.input_dir,
            config_glob=config.paths.input_glob,
        )
    return collect_videos_from_args(
        PROJECT_ROOT,
        video=None,
        input_dir=input_dir,
        config_input_dir=config.paths.input_dir,
        config_glob=config.paths.input_glob,
        all_videos=True,
    )


def apply_output_dir_override(config: AppConfig, output_dir: Path | None, project_root: Path) -> None:
    if output_dir is not None:
        config.paths.output_dir = str(
            output_dir if output_dir.is_absolute() else (project_root / output_dir).resolve()
        )
        config.paths.cache_dir = str(Path(config.paths.output_dir) / "cache")
