#!/usr/bin/env python3
"""Training-only calibration of the deterministic Stage-2 planner path.

This is a local execution check, not a scientific experiment.  It builds the
frozen 72-week training cohort, selects the largest training weeks, and runs the
same deterministic WorkLimit/MIPGap policy planner used by Stage 2.  No holdout
weekly instance is materialized and no holdout outcome is evaluated.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric_guard
import final_paper_release_guard as release_guard
import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument("--cores", type=int, default=15)
    p.add_argument("--weeks", type=int, default=15)
    p.add_argument(
        "--work-limit",
        type=float,
        default=None,
        help="Default: frozen Stage-2 final_planner_work_limit (currently 1200).",
    )
    return p.parse_args()


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    science._deterministic_site_solve = hardening.robust_deterministic_site_solve
    # Exercise the exact all-path worker installation and accounting tolerance
    # used by reviewed Stage 2, including inside macOS spawned child processes.
    release_guard.install_reviewed_guards()
    args = parse_args()

    s = science.ScientificFinalSettings(
        data=args.data,
        artifact_root="artifacts/deterministic_calibration_not_written",
        cores=args.cores,
        max_wall_minutes=720,
    )
    s.validate()
    work_limit = float(
        s.final_planner_work_limit if args.work_limit is None else args.work_limit
    )
    wall_seconds = max(
        int(s.final_planner_seconds),
        int(np.ceil(hardening.EMERGENCY_WALL_TO_WORK_MULTIPLIER * work_limit)),
    )

    cfg = base.build_config(s)
    df = base.load_data(cfg)
    scoped, _ = base.apply_experiment_scope(df, cfg)
    train_starts, hold_starts, _ = base.choose_split(scoped, s)
    ws = base.week_start_series(scoped)
    hold_start = min(hold_starts)
    hist = scoped[ws < hold_start].copy()
    context = base.build_candidate_pools(hist, cfg)
    train_weeks = base.build_bundles(
        scoped,
        train_starts,
        cfg=cfg,
        candidate_pools=context,
        eligibility_maps=base.build_eligibility_maps(hist, cfg),
        offset=0,
    )
    train_df = scoped[ws.isin(train_starts)].copy()
    enc = science.ScientificFeatureEncoder().fit(train_df)
    arrays = base.build_arrays(train_weeks, enc)

    # Stress the largest training weeks rather than taking an arbitrary prefix.
    ranked = sorted(train_weeks, key=lambda w: w.instance.num_cases, reverse=True)
    subset = ranked[: min(max(1, int(args.weeks)), len(ranked))]
    dm = {
        w.position: arrays.booked[arrays.week_slices[w.position]]
        for w in subset
    }

    t0 = time.perf_counter()
    plans = science.deterministic_eval_solve_batch(
        subset,
        dm,
        s,
        work_limit=work_limit,
        wall_seconds=wall_seconds,
        gap=s.final_planner_gap,
        label="training_deterministic_calibration",
    )
    elapsed = time.perf_counter() - t0
    report = {
        "status": "DETERMINISTIC_CALIBRATION_OK",
        "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
        "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
        "release_guard_version": release_guard.RELEASE_GUARD_VERSION,
        "phi_accounting_atol": numeric_guard.PHI_ACCOUNTING_ATOL,
        "phi_accounting_rtol": numeric_guard.PHI_ACCOUNTING_RTOL,
        "training_weeks_tested": len(subset),
        "training_week_positions": [int(w.position) for w in subset],
        "training_week_case_counts": [int(w.instance.num_cases) for w in subset],
        "cores": int(s.cores),
        "work_limit_per_site": work_limit,
        "initial_emergency_wall_seconds_per_site": wall_seconds,
        "mip_gap": float(s.final_planner_gap),
        "batch_wall_seconds": elapsed,
        "max_week_solve_seconds": float(max(p.solve_seconds for p in plans.values())),
        "mean_week_solve_seconds": float(np.mean([p.solve_seconds for p in plans.values()])),
        "max_native_psi_gap": float(max(p.gap for p in plans.values())),
        "statuses": {str(w): str(p.status) for w, p in sorted(plans.items())},
        "note": "Training data only; no holdout weekly instance or holdout outcome was evaluated.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
