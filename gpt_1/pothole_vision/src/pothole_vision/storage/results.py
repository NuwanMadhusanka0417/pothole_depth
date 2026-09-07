"""Results export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_detections_json(path: Path, detections: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"detections": detections}, f, indent=2)


def save_tracks_json(path: Path, tracks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"tracks": tracks}, f, indent=2)


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
