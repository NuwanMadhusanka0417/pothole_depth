#!/usr/bin/env python3
"""Depth evaluation stub."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    print("Depth evaluation — implement in Phase 7")

if __name__ == "__main__":
    main()
