#!/usr/bin/env python3
"""Training-only gate for the final paper experiment.

Run this first. It executes the complete 72-week training pipeline (including
RA, OS, VF and the optional saturation diagnostic), writes the frozen policies
and training artifacts, and exits immediately before any holdout optimization
or policy evaluation.

After the training artifacts are reviewed and accepted, run
``run_final_paper_experiment.py`` once for the final 22-week holdout analysis.
"""

from __future__ import annotations

from pathlib import Path

import run_final_paper_experiment as final
import run_final_vf_experiment as base


def main() -> None:
    final.install_final_adapter()
    args = base.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    original = base.build_bundles

    def training_only_build_bundles(*args_, **kwargs_):
        offset = int(kwargs_.get("offset", 0))
        if offset >= 72:
            payload = {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                "script_version": final.SCRIPT_VERSION,
                "git_head": base.git_head(),
                "holdout_evaluation_run": False,
                "next_step": "Review training artifacts, then run run_final_paper_experiment.py exactly once for the final holdout.",
            }
            base.write_json(artifact_root / "TRAINING_FREEZE.json", payload)
            base.write_json(
                artifact_root / "RUN_STATUS.json",
                {
                    "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                    "holdout_consumed": False,
                },
            )
            base.LOG.info("[HOLDOUT LOCK] training complete; stopping before holdout optimization/evaluation")
            raise SystemExit(0)
        return original(*args_, **kwargs_)

    base.build_bundles = training_only_build_bundles
    base.main()


if __name__ == "__main__":
    main()
