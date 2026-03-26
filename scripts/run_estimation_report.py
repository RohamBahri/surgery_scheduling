#!/usr/bin/env python3
"""Generate estimation diagnostics/reports from saved estimation artifacts."""

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
from src.diagnostics.estimation_report import (
    export_bootstrap_confidence_intervals,
    export_critical_ratio_distribution,
    export_profile_summary,
    export_quantile_model_quality,
    export_response_parameters,
    run_specification_checks,
)
from src.estimation.serialization import DEFAULT_PATH, load_estimation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate diagnostics files from saved estimation artifacts.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to estimation artifact file (default: {DEFAULT_PATH}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override path to the raw Excel dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/estimation_diagnostics"),
        help="Directory to write diagnostics files.",
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

    logging.info("Loading estimation artifacts from %s", args.artifact)
    result = load_estimation(args.artifact)

    cfg = Config()
    if args.data is not None:
        cfg.data.excel_file_path = str(args.data)

    logging.info("Loading and splitting data...")
    df = load_data(cfg)
    df_train, _, _ = split_warmup_pool(df, cfg)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Exporting diagnostics to %s", out_dir)

    paths: list[Path] = []
    paths.append(export_critical_ratio_distribution(result, out_dir))
    paths.append(export_response_parameters(result, out_dir))
    paths.append(run_specification_checks(result, out_dir))
    paths.append(export_quantile_model_quality(result, df_train, out_dir))

    profile_path = export_profile_summary(result, out_dir)
    if profile_path is not None:
        paths.append(profile_path)

    bootstrap_path = export_bootstrap_confidence_intervals(result, out_dir)
    if bootstrap_path is not None:
        paths.append(bootstrap_path)

    print("\nGenerated diagnostics files:")
    for path in paths:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())