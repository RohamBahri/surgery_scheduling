#!/usr/bin/env python3
"""Command-line entry point for the estimation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.estimation.serialization import DEFAULT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the surgery duration estimation pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override path to the warm-up training dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to write estimation artifacts (default: {DEFAULT_PATH}).",
    )
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Skip response profile generation.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap confidence interval estimation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
