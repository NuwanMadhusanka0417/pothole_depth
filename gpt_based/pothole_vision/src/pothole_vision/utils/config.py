"""Configuration loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load default config and merge included sub-configs."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    config_path = Path(config_path)
    config_dir = config_path.parent

    config = load_yaml(config_path)

    includes = config.pop("includes", [])
    for include in includes:
        include_path = config_dir / include
        if include_path.exists():
            sub = load_yaml(include_path)
            config = _deep_merge(config, sub)

    return config


def resolve_path(config: dict[str, Any], key: str, project_root: Path | None = None) -> Path:
    """Resolve a path from config paths section."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    paths = config.get("paths", {})
    return project_root / paths.get(key, key)
