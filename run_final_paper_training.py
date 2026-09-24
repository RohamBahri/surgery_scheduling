#!/usr/bin/env python3
"""Stage 1: train and freeze the final two-site paper policies.

This entry point owns the complete training-only pipeline.  It deliberately does
not call the legacy experiment ``main()`` and therefore cannot materialize or
optimize the holdout.  A successful run fingerprints the exact accepted policy,
feature, data, environment, optimization-history and schedule-library artifacts.

Stage 2 must use ``run_final_paper_evaluation.py`` from the same clean Git commit
and must load these exact policies rather than retraining them.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

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


def _environment_payload() -> dict:
    packages = {}
    for dist in ("numpy", "pandas", "scipy", "scikit-learn", "gurobipy"):
        try:
            packages[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            packages[dist] = None
    try:
        gurobi_runtime = ".".join(map(str, base.gp.gurobi.version()))
    except Exception:
        gurobi_runtime = None
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except Exception:
        pip_freeze = []
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "gurobi_runtime": gurobi_runtime,
        "pip_freeze": pip_freeze,
    }


def _training_tie_seed_audit(train_weeks, train_a, s) -> list[dict]:
    """Small pre-holdout diagnostic of solver-selection variability across seeds."""
    weeks = list(train_weeks[: min(2, len(train_weeks))])
    rows = []
    for seed in (41, 42, 43):
        dm = {w.position: train_a.booked[train_a.week_slices[w.position]] for w in weeks}
        plans = science.deterministic_eval_solve_batch(
            weeks,
            dm,
            s,
            work_limit=min(60.0, float(s.sensitivity_planner_work_limit)),
            wall_seconds=max(300, int(s.train_planner_seconds)),
            gap=0.0,
            label=f"training_tie_seed_{seed}",
            seed=seed,
        )
        for w in weeks:
            idx = train_a.week_slices[w.position]
            plan = plans[w.position]
            realized = float(
                plan.column.compute_cost(
                    train_a.actual[idx], final.final_cost_cfg(s), final.PRIMARY_TURNOVER
                )
            )
            rows.append(
                {
                    "week": w.position,
                    "week_start": str(w.start.date()),
                    "seed": seed,
                    "planning_objective_phi": plan.objective,
                    "planning_bound_phi": plan.bound,
                    "planning_gap_native_psi": plan.gap,
                    "status": plan.status,
                    "realized_cost": realized,
                }
            )
    return rows


def _site_shift_policy(train_a, enc, s, lam_common, initial_full):
    reduced, site_idx = science.site_shift_reduced_arrays(train_a, enc.feature_names)
    initial_small = np.array([initial_full[0], initial_full[site_idx]], dtype=float)
    spec = base.FixedSpec(
        "SITE_SHIFT",
        reduced,
        np.full(len(reduced.error), s.overtime),
        np.full(len(reduced.error), s.idle),
        None,
        lam_common,
        s,
    )
    w_small, hist = base.run_pdca(
        spec, initial_small, s, max_iterations=s.pdca_fixed_max_iterations
    )
    return science.expand_site_shift_policy(w_small, train_a.p, site_idx), hist


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    science.REGULARIZATION_AUDIT.clear()

    args = base.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise RuntimeError(
            f"Artifact root is not empty: {artifact_root}. Use a fresh directory for each final run."
        )
    if _tracked_tree_is_dirty():
        raise RuntimeError(
            "Tracked repository files are modified. Commit or revert them before the final training run."
        )

    s = _settings_from_args(args, artifact_root)
    s.validate()
    base.setup_logging(artifact_root, s.verbose)
    start = time.monotonic()
    deadline = start + 60.0 * float(s.max_wall_minutes)
    base.write_json(
        artifact_root / "RUN_STATUS.json",
        {"status": "STARTED", "git_head": base.git_head(), "settings": asdict(s)},
    )

    try:
        base.write_json(artifact_root / "FROZEN_SETTINGS.json", asdict(s))
        base.write_json(
            artifact_root / "SCIENTIFIC_SPEC.json",
            {
                "version": science.SCIENTIFIC_SPEC_VERSION,
                "cohort_rule": (
                    "canonical noncancelled/nonemergency OR cases with ordered positive timestamps; "
                    "0 < booked duration <= 480 minutes; room duration <= 24 hours as a timestamp-validity rule; "
                    "no upper cap on otherwise valid realized surgical/room duration"
                ),
                "primary_roster": science.PRIMARY_ROSTER,
                "required_capacity_sensitivity": science.REQUIRED_CAPACITY_SENSITIVITY,
                "feature_dimension": science.EXPECTED_FEATURES,
                "calendar_features": "none",
                "coefficient_bound": science.COEFFICIENT_BOUND,
                "l1_rule": (
                    "Naive: eta times zero-policy LAD minutes divided by p; "
                    "RA/RA_FULL/OS/VF: one common eta times zero-policy full-weight case cost divided by p"
                ),
                "vf_library": "per-site TGH/TWH surfaces with implicit Cartesian-product minimum",
                "planning_day_rule": "weekly day-flexible assignment",
                "elective_rule": "recorded patient-type emergency label only; no after-hours exclusion",
                "turnover_minutes": final.PRIMARY_TURNOVER,
                "required_turnover_sensitivity_minutes": science.REQUIRED_TURNOVER_SENSITIVITY_MINUTES,
                "deployment_tie_rule": science.DEPLOYMENT_TIE_RULE,
                "response_sensitivity_scenarios": science.RESPONSE_SENSITIVITY_SCENARIOS,
                "expected_cleaned_cases": science.EXPECTED_CLEANED_CASES,
                "expected_scoped_cases": science.EXPECTED_SCOPED_CASES,
                "expected_train_cases": science.EXPECTED_TRAIN_CASES,
                "expected_holdout_cases": science.EXPECTED_HOLDOUT_CASES,
            },
        )
        base.write_json(
            artifact_root / "SENSITIVITY_PLAN.json",
            {
                "primary_capacity": science.PRIMARY_ROSTER,
                "required_capacity_sensitivity": science.REQUIRED_CAPACITY_SENSITIVITY,
                "primary_turnover_minutes": final.PRIMARY_TURNOVER,
                "required_turnover_sensitivity_minutes": science.REQUIRED_TURNOVER_SENSITIVITY_MINUTES,
                "response_scenarios": science.RESPONSE_SENSITIVITY_SCENARIOS,
                "note": "Sensitivity analyses are secondary and do not redefine the frozen primary holdout result.",
            },
        )
        base.write_json(artifact_root / "ENVIRONMENT.json", _environment_payload())

        cfg = base.build_config(s)
        data_path = Path(s.data).resolve()
        base.LOG.info("Loading final scientific cohort from %s", data_path)
        df = base.load_data(cfg)
        scoped, scope_summary = base.apply_experiment_scope(df, cfg)
        if len(scoped) != science.EXPECTED_SCOPED_CASES:
            raise RuntimeError(
                f"Scoped cohort changed: {len(scoped)} != {science.EXPECTED_SCOPED_CASES}"
            )
        train_starts, hold_starts, split_audit = base.choose_split(scoped, s)
        split_audit.to_csv(artifact_root / "WEEK_SPLIT.csv", index=False)
        ws = base.week_start_series(scoped)
        train_df = scoped[ws.isin(train_starts)].copy()
        hold_count = int(ws.isin(hold_starts).sum())
        if len(train_df) != science.EXPECTED_TRAIN_CASES or hold_count != science.EXPECTED_HOLDOUT_CASES:
            raise RuntimeError(
                f"DATA FREEZE MISMATCH: train={len(train_df)} holdout={hold_count}"
            )
        train_site_counts = {str(k): int(v) for k, v in train_df[base.Col.SITE].value_counts().to_dict().items()}
        if train_site_counts != science.EXPECTED_TRAIN_SITE_COUNTS:
            raise RuntimeError(f"Training site counts changed: {train_site_counts}")

        hold_start = min(hold_starts)
        hist = scoped[ws < hold_start].copy()
        pools = base.build_candidate_pools(hist, cfg)
        elig = base.build_eligibility_maps(hist, cfg)
        train_weeks = base.build_bundles(
            scoped,
            train_starts,
            cfg=cfg,
            candidate_pools=pools,
            eligibility_maps=elig,
            offset=0,
        )
        enc = base.FrozenFeatureEncoder().fit(train_df)
        train_a = base.build_arrays(train_weeks, enc)
        if train_a.p != science.EXPECTED_FEATURES or train_a.p != len(enc.feature_names):
            raise AssertionError(
                f"Final feature dimension changed: arrays={train_a.p}, manifest={len(enc.feature_names)}, "
                f"expected={science.EXPECTED_FEATURES}"
            )

        data_freeze = {
            "script_version": final.SCRIPT_VERSION,
            "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
            "git_head": base.git_head(),
            "input_sha256": base.sha256_file(data_path),
            "cleaned_cases": len(df),
            "scoped_cases": len(scoped),
            "train_cases": len(train_df),
            "holdout_cases": hold_count,
            "train_site_counts": train_site_counts,
            "train_first": str(train_starts[0].date()),
            "train_last": str(train_starts[-1].date()),
            "holdout_first": str(hold_starts[0].date()),
            "holdout_last": str(hold_starts[-1].date()),
            "train_case_ids_sha256": base.sha256_text(
                ",".join(map(str, sorted(train_a.case_ids.tolist())))
            ),
            "feature_manifest": enc.manifest(),
            "scope_summary": asdict(scope_summary),
        }
        base.write_json(artifact_root / "DATA_FREEZE.json", data_freeze)
        base.LOG.info(
            "[DATA] train=%d holdout=%d weeks=%d/%d p=%d",
            len(train_df),
            hold_count,
            len(train_starts),
            len(hold_starts),
            train_a.p,
        )

        audit = base.stable_solver_audit(train_weeks, s)
        base.write_csv(artifact_root / "STABLE_SOLVER_AUDIT.csv", audit)
        if audit and not all(r["passed"] for r in audit):
            raise AssertionError("Reduced Psi solver does not match direct Phi audit")

        week_lookup = {w.position: w for w in train_weeks}
        lib = base.ScheduleLibrary(week_lookup)
        actual_map = {wk: train_a.actual[idx] for wk, idx in train_a.week_slices.items()}
        train_oracle = base.solve_oracle_batch(train_weeks, actual_map, s, label="oracle_train")
        base.write_csv(
            artifact_root / "ORACLE_TRAIN.csv",
            [
                {
                    "week": w,
                    "objective_phi": r.objective,
                    "bound_phi": r.bound,
                    "gap_native_psi": r.gap,
                    "status": r.status,
                    "exact": r.exact,
                    "seconds": r.solve_seconds,
                }
                for w, r in sorted(train_oracle.items())
            ],
        )
        oracle_lb = {w: r.bound for w, r in train_oracle.items()}
        lib.add_plans(train_oracle, "REALIZED_ORACLE", pin=True)

        booked_map = {wk: train_a.booked[idx] for wk, idx in train_a.week_slices.items()}
        booked_plans = base.solve_batch(
            train_weeks,
            booked_map,
            s,
            seconds=s.train_planner_seconds,
            gap=s.train_planner_gap,
            label="booked_train",
        )
        lib.add_plans(booked_plans, "BOOKED")
        labels = base.exposure_labels(booked_plans, train_a, week_lookup, s)
        pi, pi_metrics = base.crossfit_pi(train_a, labels, s)
        base.write_json(artifact_root / "RA_PI_METRICS.json", pi_metrics)

        zero = np.zeros(train_a.p, dtype=float)
        lam_common = science.common_cost_lambda(train_a, s)
        base.LOG.info("[REG] common RA/RA_FULL/OS/VF lambda=%.6f", lam_common)
        naive = base.train_naive(train_a, s, lam_common)

        ra_spec = base.FixedSpec(
            "RA",
            train_a,
            s.overtime * pi,
            s.idle * (1.0 - pi),
            None,
            lam_common,
            s,
        )
        ra, ra_hist = base.run_pdca(
            ra_spec, naive, s, max_iterations=s.pdca_fixed_max_iterations
        )

        full_plus = np.full(len(train_a.error), s.overtime)
        full_minus = np.full(len(train_a.error), s.idle)
        ra_full_spec = base.FixedSpec(
            "RA_FULL", train_a, full_plus, full_minus, None, lam_common, s
        )
        ra_full, ra_full_hist = base.run_pdca(
            ra_full_spec, naive, s, max_iterations=s.pdca_fixed_max_iterations
        )

        site_shift, site_shift_hist = _site_shift_policy(
            train_a, enc, s, lam_common, ra_full
        )

        oracle_cols = {w: r.column for w, r in train_oracle.items()}
        os_spec = base.FixedSpec(
            "OS", train_a, full_plus, full_minus, oracle_cols, lam_common, s
        )
        os_w, os_hist = base.run_pdca(
            os_spec, ra_full, s, max_iterations=s.pdca_fixed_max_iterations
        )

        base.write_csv(artifact_root / "RA_PDCA.csv", ra_hist)
        base.write_csv(artifact_root / "RA_FULL_PDCA.csv", ra_full_hist)
        base.write_csv(artifact_root / "SITE_SHIFT_PDCA.csv", site_shift_hist)
        base.write_csv(artifact_root / "OS_PDCA.csv", os_hist)

        policies0 = {
            "ZERO": zero,
            "NAIVE": naive,
            "RA": ra,
            "RA_FULL": ra_full,
            "SITE_SHIFT": site_shift,
            "OS": os_w,
        }
        for name, w in policies0.items():
            if name == "ZERO":
                continue
            plans = base.plan_policy_training(
                w,
                train_a,
                train_weeks,
                lib,
                s,
                seconds=s.seed_planner_seconds,
                gap=s.seed_planner_gap,
                label=f"seed_{name}",
            )
            lib.add_plans(plans, f"SEED_{name}")

        start_rows = []
        for name, w in policies0.items():
            start_rows.append(
                {"method": name, **base.library_metrics(w, train_a, lib, oracle_lb, s, lam_common)}
            )
        pd.DataFrame(start_rows).to_csv(artifact_root / "START_DECOMPOSITION.csv", index=False)

        # OS is the designated full-weight structural predecessor of VF.
        vf_deadline = max(time.monotonic() + 120.0, deadline - 600.0)
        vf_w, _ = base.train_vf(
            os_w,
            train_a,
            train_weeks,
            lib,
            oracle_lb,
            s,
            lam_common,
            artifact_root,
            search_deadline=vf_deadline,
        )
        policies = {
            "NAIVE": naive,
            "RA": ra,
            "RA_FULL": ra_full,
            "SITE_SHIFT": site_shift,
            "OS": os_w,
            "VF": vf_w,
        }
        np.savez(artifact_root / "POLICIES.npz", **policies)
        base.write_json(artifact_root / "FEATURE_MANIFEST.json", enc.manifest())

        decomp = []
        for name, w in {"ZERO": zero, **policies}.items():
            metrics = base.library_metrics(w, train_a, lib, oracle_lb, s, lam_common)
            delta, corr, _ = base.correction_and_planning(w, train_a, s)
            decomp.append(
                {
                    "method": name,
                    **metrics,
                    "raw_mae": float(np.mean(np.abs(delta - train_a.error))),
                    "implemented_mae": float(np.mean(np.abs(corr - train_a.error))),
                    "training_lambda": (
                        science.REGULARIZATION_AUDIT.get("NAIVE", {}).get("lambda")
                        if name == "NAIVE"
                        else lam_common
                    ),
                }
            )
        pd.DataFrame(decomp).to_csv(artifact_root / "TRAIN_DECOMPOSITION.csv", index=False)
        base.write_csv(artifact_root / "LIBRARY_SUMMARY.csv", lib.summary_rows())
        if not lib.check_pins():
            raise AssertionError("Pinned oracle schedule disappeared from library")

        # Small training-only seed audit: records whether equal/near-equal planning
        # objectives can map to materially different realized schedules.
        tie_rows = _training_tie_seed_audit(train_weeks, train_a, s)
        base.write_csv(artifact_root / "TIE_SEED_AUDIT.csv", tie_rows)

        # Optional training-only saturation stress.  It may stop early at the
        # overall Stage-1 deadline; the lock is still written afterwards.
        sat_summary = {"skipped": True, "reason": "user flag or insufficient wall time"}
        if not args.skip_saturation and base.remaining(deadline) > 900:
            _, sat_summary = base.saturation_test(
                train_weeks, train_a, lib, s, artifact_root, deadline - 300.0
            )
            base.write_csv(
                artifact_root / "LIBRARY_SUMMARY_AFTER_SATURATION.csv", lib.summary_rows()
            )

        # Serialize the final library after all training-only enrichment.
        base.write_csv(artifact_root / "TRAIN_LIBRARY_SURFACES.csv", lib.surface_rows())
        base.write_json(artifact_root / "REGULARIZATION.json", science.REGULARIZATION_AUDIT)
        base.write_json(artifact_root / "SATURATION_STATUS.json", sat_summary)

        required = [
            artifact_root / "POLICIES.npz",
            artifact_root / "FEATURE_MANIFEST.json",
            artifact_root / "DATA_FREEZE.json",
            artifact_root / "FROZEN_SETTINGS.json",
            artifact_root / "SCIENTIFIC_SPEC.json",
            artifact_root / "SENSITIVITY_PLAN.json",
            artifact_root / "REGULARIZATION.json",
            artifact_root / "ENVIRONMENT.json",
            artifact_root / "RA_PDCA.csv",
            artifact_root / "RA_FULL_PDCA.csv",
            artifact_root / "SITE_SHIFT_PDCA.csv",
            artifact_root / "OS_PDCA.csv",
            artifact_root / "VF_TRAJECTORY.csv",
            artifact_root / "TRAIN_LIBRARY_SURFACES.csv",
            artifact_root / "TIE_SEED_AUDIT.csv",
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Training freeze missing required artifacts: {missing}")
        fingerprints = {p.name: base.sha256_file(p) for p in required}
        freeze = {
            "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
            "script_version": final.SCRIPT_VERSION,
            "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
            "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
            "git_head": base.git_head(),
            "holdout_evaluation_run": False,
            "policy_names": sorted(policies),
            "artifact_fingerprints": fingerprints,
            "elapsed_seconds": time.monotonic() - start,
            "next_step": (
                "Review all Stage-1 diagnostics. Then evaluate exactly this bundle once using "
                "run_final_paper_evaluation.py from a clean git worktree at the frozen commit."
            ),
        }
        base.write_json(artifact_root / "TRAINING_FREEZE.json", freeze)
        base.write_json(
            artifact_root / "RUN_STATUS.json",
            {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                "holdout_consumed": False,
                "git_head": base.git_head(),
                "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
                "artifact_fingerprints": fingerprints,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        base.LOG.info("[HOLDOUT LOCK] Stage 1 complete; no holdout instance was materialized")
    except Exception as exc:
        base.write_json(
            artifact_root / "RUN_STATUS.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        base.LOG.exception("Stage 1 failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
