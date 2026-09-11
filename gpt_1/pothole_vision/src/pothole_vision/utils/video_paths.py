"""Input discovery and per-video output path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIDEO_GLOBS = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV")


def resolve_video_path(path: Path | str, project_root: Path) -> Path:
    """Resolve a video path relative to project_root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def discover_videos(
    input_dir: Path,
    patterns: tuple[str, ...] | list[str] | None = None,
) -> list[Path]:
    """List video files in input_dir (sorted by name)."""
    if not input_dir.is_dir():
        return []
    globs = tuple(patterns) if patterns else DEFAULT_VIDEO_GLOBS
    seen: set[Path] = set()
    videos: list[Path] = []
    for pattern in globs:
        for p in sorted(input_dir.glob(pattern)):
            resolved = p.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                videos.append(resolved)
    return sorted(videos, key=lambda x: x.name.lower())


@dataclass(frozen=True)
class VideoRunPaths:
    """Output locations for one input video."""

    video_id: str
    output_dir: Path
    events_dir: Path
    cache_dir: Path

    @property
    def annotated_video(self) -> Path:
        return self.output_dir / "annotated.mp4"


def paths_for_video(
    video_path: Path,
    base_output: Path,
    base_events: Path,
    base_cache: Path,
) -> VideoRunPaths:
    """Per-video subdirectories under base output/events/cache."""
    video_id = video_path.stem
    return VideoRunPaths(
        video_id=video_id,
        output_dir=base_output / video_id,
        events_dir=base_events / video_id,
        cache_dir=base_cache / video_id,
    )


def collect_videos_from_args(
    project_root: Path,
    video: Path | None,
    input_dir: Path | None,
    config_input_dir: str,
    config_glob: str | None = None,
    all_videos: bool = False,
) -> list[Path]:
    """
    Resolve which videos to process from CLI args.

    - If ``video`` is set: process that file only.
    - Else: scan ``input_dir`` or config ``input_dir``; use globs from config or defaults.
    """
    if video is not None:
        p = resolve_video_path(video, project_root)
        if not p.is_file():
            raise FileNotFoundError(f"Video not found: {p}")
        return [p]

    root_in = input_dir or resolve_video_path(config_input_dir, project_root)
    patterns: tuple[str, ...]
    if config_glob:
        patterns = (config_glob,)
    else:
        patterns = DEFAULT_VIDEO_GLOBS

    found = discover_videos(root_in, patterns)
    if not found:
        raise FileNotFoundError(
            f"No videos found in {root_in} (patterns: {', '.join(patterns)}). "
            "Place .mp4 files in data/input/ or pass --video PATH."
        )
    if all_videos or video is None:
        return found
    return found
