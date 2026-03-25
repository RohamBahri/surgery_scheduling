#!/usr/bin/env python
"""Run the rolling-horizon out-of-sample experiment.

Usage
-----
    python run_experiment.py
    python run_experiment.py --data path/to/data.xlsx --horizons 53
    python run_experiment.py --quick          # 3 horizons, 60 s solver limit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so "src.*" imports work when
# invoking this script directly (i.e. without pip-installing the package).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import CONFIG
from src.methods.booked import BookedTimeMethod
from src.methods.oracle import OracleMethod
from src.methods.behavioral_ccg import BehavioralCCGMethod
from src.methods.registry import MethodRegistry
from src.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incentive-aware OR scheduling — out-of-sample experiment",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to the UHN Excel dataset (overrides config default).",
    )
    parser.add_argument(
        "--horizons", type=int, default=None,
        help="Number of weekly horizons to evaluate.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Directory for result files (default: outputs/).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick-test mode: 3 horizons, 60 s solver limit.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show Gurobi solver output.",
    )
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    # ── Apply CLI overrides to config ────────────────────────────────────
    if args.data:
        CONFIG.data.excel_file_path = args.data
    if args.horizons:
        CONFIG.data.num_horizons = args.horizons
    if args.quick:
        CONFIG.data.num_horizons = 3
        CONFIG.solver.time_limit_seconds = 60
    if args.verbose:
        CONFIG.solver.verbose = True

    # ── Build method registry ────────────────────────────────────────────
    registry = MethodRegistry()
    registry.register(BookedTimeMethod(CONFIG))
    registry.register(OracleMethod(CONFIG))
    registry.register(BehavioralCCGMethod(CONFIG))
    # Future methods are added here:
    # registry.register(LassoMethod(CONFIG))

    # ── Run ──────────────────────────────────────────────────────────────
    run_experiment(registry, CONFIG, output_dir=args.output)


if __name__ == "__main__":
    main()
