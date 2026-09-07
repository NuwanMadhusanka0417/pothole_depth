#!/usr/bin/env python3
"""Detection evaluation stub."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    args = parser.parse_args()
    print("Evaluation script — implement in Phase 7")
    print(f"Predictions: {args.predictions}, GT: {args.ground_truth}")

if __name__ == "__main__":
    main()
