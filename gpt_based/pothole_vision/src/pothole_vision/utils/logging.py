"""Logging setup."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    logger.remove()
    if structured:
        fmt = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[tag]}</cyan> | {message}"
        )
    else:
        fmt = "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"

    logger.add(sys.stderr, format=fmt, level=level.upper(), filter=_inject_tag)
    logger.configure(extra={"tag": "SYSTEM"})


def _inject_tag(record: dict[str, Any]) -> bool:
    record["extra"].setdefault("tag", "SYSTEM")
    return True


def log_tag(tag: str, message: str, **kwargs: Any) -> None:
    logger.bind(tag=tag).info(message, **kwargs)
