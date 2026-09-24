#!/usr/bin/env python3
"""Training-only gate for the final two-site paper experiment.

This is Stage 1. It executes the complete 72-week training pipeline under the
frozen scientific specification, fingerprints the accepted policy bundle, and
exits before any holdout instance is materialized or optimized.

Stage 2 must use ``run_final_paper_evaluation.py`` to load and evaluate exactly
this accepted bundle. It must never retrain the accepted policies.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def _tracked_tree_is_dirty() -> bool:
    """Ignore untracked data/artifacts, but reject modified tracked source."""

    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        return unstaged != 0 or staged != 0
    except Exception:
        return False


def _settings_from_args(args, artifact_root: Path) -> science.ScientificFinalSettings:
    return science.ScientificFinalSettings(
        data=args.data,
        artifact_root=str(artifact_root),
        cores=args.cores,
        max_wall_minutes=args.max_wall_minutes,
        final_reserve_minutes=args.final_reserve_minutes,
        verbose=args.verbose,
        vf_outer=args.outer,
        oracle_seconds=args.oracle_seconds,
        train_planner_seconds=args.train_planner_seconds,
        train_planner_gap=args.train_planner_gap,
        saturation_draws_per_week=args.saturation_draws,
        saturation_seconds=args.saturation_seconds,
        final_planner_seconds=args.final_planner_seconds,
        final_planner_gap=args.final_planner_gap,
        strict_data_freeze=not args.allow_data_mismatch,
    )


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()

    args = base.parse_args()
    artifact_root = Path(args.artifact_root).resolve()

    # A final scientific run must not silently mix new outputs with stale files.
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise RuntimeError(
            f"Artifact root is not empty: {artifact_root}. Use a fresh directory for each final run."
        )
    if _tracked_tree_is_dirty():
        raise RuntimeError(
            "Tracked repository files are modified. Commit or revert them before the final training run."
        )

    frozen_settings = _settings_from_args(args, artifact_root)
    frozen_settings.validate()
    base.write_json(artifact_root / "FROZEN_SETTINGS.json", asdict(frozen_settings))
    base.write_json(
        artifact_root / "SCIENTIFIC_SPEC.json",
        {
            "version": science.SCIENTIFIC_SPEC_VERSION,
            "cohort_rule": "all canonical quality rules; 0 < booked duration <= 480 minutes; no upper cap on positive realized room/surgical duration",
            "primary_roster": science.PRIMARY_ROSTER,
            "feature_dimension": science.EXPECTED_FEATURES,
            "calendar_features": "none",
            "coefficient_bound": science.COEFFICIENT_BOUND,
            "l1_rule": "lambda_m = eta * L_m(0) / p on each method's own unregularized objective scale",
            "vf_library": "per-site TGH/TWH surfaces with implicit Cartesian-product minimum",
            "planning_day_rule": "weekly day-flexible assignment",
            "elective_rule": "use recorded patient-type emergency label; no after-hours exclusion",
            "turnover_minutes": final.PRIMARY_TURNOVER,
            "deployment_tie_rule": science.DEPLOYMENT_TIE_RULE,
            "expected_cleaned_cases": science.EXPECTED_CLEANED_CASES,
            "expected_scoped_cases": science.EXPECTED_SCOPED_CASES,
            "expected_train_cases": science.EXPECTED_TRAIN_CASES,
            "expected_holdout_cases": science.EXPECTED_HOLDOUT_CASES,
        },
    )

    original = base.build_bundles

    def training_only_build_bundles(*args_, **kwargs_):
        offset = int(kwargs_.get("offset", 0))
        if offset >= 72:
            base.write_json(
                artifact_root / "REGULARIZATION.json",
                science.REGULARIZATION_AUDIT,
            )
            required = [
                artifact_root / "POLICIES.npz",
                artifact_root / "FEATURE_MANIFEST.json",
                artifact_root / "DATA_FREEZE.json",
                artifact_root / "FROZEN_SETTINGS.json",
                artifact_root / "SCIENTIFIC_SPEC.json",
                artifact_root / "REGULARIZATION.json",
            ]
            missing = [str(p.name) for p in required if not p.exists()]
            if missing:
                raise RuntimeError(
                    f"Training reached the holdout lock without required artifacts: {missing}"
                )

            fingerprints = {p.name: base.sha256_file(p) for p in required}
            payload = {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                "script_version": final.SCRIPT_VERSION,
                "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
                "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
                "git_head": base.git_head(),
                "holdout_evaluation_run": False,
                "artifact_fingerprints": fingerprints,
                "next_step": (
                    "Review the complete training bundle, then run run_final_paper_evaluation.py. "
                    "Stage 2 must load these exact policy/feature/data/settings fingerprints and must not retrain."
                ),
            }
            base.write_json(artifact_root / "TRAINING_FREEZE.json", payload)
            base.write_json(
                artifact_root / "RUN_STATUS.json",
                {
                    "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                    "holdout_consumed": False,
                    "script_version": final.SCRIPT_VERSION,
                    "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
                    "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
                    "git_head": base.git_head(),
                    "artifact_fingerprints": fingerprints,
                },
            )
            base.LOG.info(
                "[HOLDOUT LOCK] training complete; stopping before holdout optimization/evaluation"
            )
            raise SystemExit(0)
        return original(*args_, **kwargs_)

    base.build_bundles = training_only_build_bundles
    base.main()


if __name__ == "__main__":
    main()
