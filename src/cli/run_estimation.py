from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.config import Config
from src.core.paths import ArtifactManager
from src.data.loader import load_data
from src.data.splits import split_warmup_pool
from src.estimation.orchestrator import fit_estimation_pipeline
from src.estimation.serialization import save_estimation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the surgery duration estimation pipeline.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=None)
    parser.add_argument("--bootstrap-jobs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = Config()
    if args.data is not None:
        config.data.excel_file_path = str(args.data)
    if args.bootstrap_iters is not None:
        config.estimation.bootstrap.n_bootstrap = int(args.bootstrap_iters)
    if args.bootstrap_jobs is not None:
        config.estimation.bootstrap.n_jobs = int(args.bootstrap_jobs)

    df = load_data(config)
    df_train, _, _ = split_warmup_pool(df, config)

    result = fit_estimation_pipeline(
        df_train,
        config,
        skip_profiles=args.skip_profiles,
        quiet=args.quiet,
        skip_bootstrap=not args.bootstrap,
        run_bootstrap=args.bootstrap,
    )

    if not args.no_save:
        artifact_run = ArtifactManager(args.artifact_root).run("estimation", args.run_label)
        path = save_estimation(result, artifact_run.path("estimation_artifacts.joblib"))
        logging.info("Saved estimation artifacts to %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
