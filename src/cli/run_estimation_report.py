from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.config import Config
from src.core.paths import ArtifactManager
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
from src.estimation.serialization import load_estimation


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diagnostics files from saved estimation artifacts.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = load_estimation(args.artifact)
    cfg = Config()
    if args.data is not None:
        cfg.data.excel_file_path = str(args.data)

    df = load_data(cfg)
    df_train, _, _ = split_warmup_pool(df, cfg)

    artifact_run = ArtifactManager(args.artifact_root).run("estimation", args.run_label)
    out_dir = artifact_run.directory("diagnostics")

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

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
