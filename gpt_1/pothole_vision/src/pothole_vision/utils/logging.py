"""Logging setup."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level.upper(),
        colorize=True,
    )


def log_stage(stage: str, message: str) -> None:
    logger.info(f"[{stage}] {message}")
