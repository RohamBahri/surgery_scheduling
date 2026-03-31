from __future__ import annotations

import argparse
import logging

from src.core.config import Config
from src.core.paths import ArtifactManager
from src.experiments.runner import run_experiment
from src.methods.booked import BookedTimeMethod
from src.methods.oracle import OracleMethod
from src.methods.registry import MethodRegistry
from src.methods.vfcg import VFCGMethod


def main() -> int:
    parser = argparse.ArgumentParser(description="Run out-of-sample scheduling experiment.")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--horizons", type=int, default=None)
    parser.add_argument("--artifact-root", type=str, default="artifacts")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    config = Config()
    if args.data:
        config.data.excel_file_path = args.data
    if args.horizons:
        config.data.num_horizons = args.horizons
    if args.quick:
        config.data.num_horizons = 3
        config.solver.time_limit_seconds = 60
    if args.verbose:
        config.solver.verbose = True

    registry = MethodRegistry()
    registry.register(BookedTimeMethod(config))
    registry.register(OracleMethod(config))
    registry.register(VFCGMethod(config))

    artifact_run = ArtifactManager(args.artifact_root).run("experiments", args.run_label)
    run_experiment(registry, config, artifact_run=artifact_run)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
