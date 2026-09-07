#!/usr/bin/env python3
"""Placeholder for depth-only processing."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
print("Use run_pipeline.py --mode full for depth processing (Phase 4+)")
