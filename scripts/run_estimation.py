#!/usr/bin/env python3
"""Command-line entry point for the estimation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.data.loader import load_data
from src.data.splits import split_warmup_pool
from src.estimation.orchestrator import fit_estimation_pipeline
from src.estimation.serialization import DEFAULT_PATH, save_estimation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the surgery duration estimation pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override path to the raw Excel dataset.",
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
        "--no-save",
        action="store_true",
        help="Run estimation without writing artifacts to disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config()
    if args.data is not None:
        config.data.excel_file_path = str(args.data)

    logging.info("Loading and splitting data...")
    df = load_data(config)
    df_train, _, _ = split_warmup_pool(df, config)

    logging.info("Fitting estimation pipeline on %d warm-up rows...", len(df_train))
    result = fit_estimation_pipeline(
        df_train,
        config,
        skip_profiles=args.skip_profiles,
        quiet=args.quiet,
        skip_bootstrap=True,
    )

    if not args.no_save:
        path = save_estimation(result, args.output)
        logging.info("Saved estimation artifacts to %s", path)
    else:
        logging.info("Skipping artifact save (--no-save).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
