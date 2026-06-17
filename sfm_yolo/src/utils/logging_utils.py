"""Centralised logging configuration.

The whole package uses a single named logger ``sfm_yolo`` so that all
sub-modules share the same handlers and formatting.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "sfm_yolo"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)s | %(message)s"


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        env = os.environ.get("SFM_YOLO_LOG_LEVEL")
        return getattr(logging, env.upper(), logging.INFO) if env else logging.INFO
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def get_logger(
    name: Optional[str] = None,
    *,
    level: int | str | None = None,
    log_file: Optional[Path | str] = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str, optional
        Sub-logger name. The full logger name will be ``sfm_yolo.<name>``.
        Pass ``None`` to get the root project logger.
    level : int or str, optional
        Logging level (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
        Defaults to ``$SFM_YOLO_LOG_LEVEL`` or ``INFO``.
    log_file : path-like, optional
        If provided, a rotating-free file handler is attached.
    """
    root = logging.getLogger(_LOGGER_NAME)

    if not getattr(root, "_sfm_yolo_configured", False):
        root.setLevel(_resolve_level(level))
        root.propagate = False

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(stream)
        root._sfm_yolo_configured = True  # type: ignore[attr-defined]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
            for h in root.handlers
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(fh)

    if level is not None:
        root.setLevel(_resolve_level(level))

    return root if name is None else root.getChild(name)
