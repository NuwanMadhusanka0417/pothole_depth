"""Tests for video discovery and per-video paths."""

from pathlib import Path

import pytest

from pothole_vision.utils.video_paths import discover_videos, paths_for_video, resolve_video_path


def test_paths_for_video():
    video = Path("data/input/my_clip.mp4")
    run = paths_for_video(
        video,
        Path("data/output"),
        Path("data/events"),
        Path("data/output/cache"),
    )
    assert run.video_id == "my_clip"
    assert run.output_dir == Path("data/output/my_clip")
    assert run.events_dir == Path("data/events/my_clip")
    assert run.annotated_video == Path("data/output/my_clip/annotated.mp4")


def test_discover_videos_empty(tmp_path: Path):
    assert discover_videos(tmp_path) == []


def test_discover_videos_finds_mp4(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.MP4").write_bytes(b"")
    found = discover_videos(tmp_path)
    assert len(found) == 2


def test_resolve_video_path_relative(project_root=None):
    root = Path(__file__).resolve().parents[1]
    p = resolve_video_path(Path("configs/default.yaml"), root)
    assert p.is_absolute()
    assert p.exists()
