#!/usr/bin/env python3
"""Local preflight for the frozen final-paper experiment.

Default mode is solver-light: it uses the real workbook, verifies the frozen
cohort/split, builds all 72 training instances, checks the median-count roster
and eligibility, reconstructs the 107-feature predecision encoder, and performs
no holdout optimization or holdout-outcome summaries.

With ``--solver-check`` it also exercises the local Gurobi setup on training
only: 15 concurrent process-isolated weekly jobs and one full-size RA_FULL pDCA
convex subproblem with a short time cap.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.types import Col
from src.planning.roster import build_fixed_roster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument(
        "--solver-check",
        action="store_true",
        help="Run short training-only Gurobi concurrency and pDCA-size checks.",
    )
    p.add_argument("--cores", type=int, default=15)
    return p.parse_args()


def _solver_checks(train_weeks, train_arrays, s) -> dict:
    # 1) Exercise the same process-isolated concurrent weekly path used by Stage 1.
    subset = list(train_weeks[: min(15, len(train_weeks))])
    dm = {w.position: train_arrays.booked[train_arrays.week_slices[w.position]] for w in subset}
    t0 = time.perf_counter()
    plans = base.solve_batch(
        subset,
        dm,
        s,
        seconds=30,
        gap=0.20,
        label="preflight_concurrency",
    )
    concurrency_seconds = time.perf_counter() - t0

    # 2) Build the full 72-week RA_FULL convex pDCA subproblem and ask Gurobi for
    # one short solve.  This checks model size/license compatibility before the
    # expensive run without consuming any holdout information.
    ss = copy.copy(s)
    ss.pdca_convex_seconds = 30
    lam = science.common_cost_lambda(train_arrays, ss)
    spec = base.FixedSpec(
        "RA_FULL",
        train_arrays,
        np.full(len(train_arrays.error), ss.overtime),
        np.full(len(train_arrays.error), ss.idle),
        None,
        lam,
        ss,
    )
    zero = np.zeros(train_arrays.p, dtype=float)
    q = spec.h_subgradient_w(zero)
    t1 = time.perf_counter()
    solver = base.ConvexPDCASubproblem(spec, ss, "preflight_full_pdca")
    try:
        candidate = solver.solve(zero, q, ss.pdca_initial_gamma)
    finally:
        solver.dispose()
    pdca_seconds = time.perf_counter() - t1
    return {
        "concurrent_weekly_jobs": len(plans),
        "concurrency_wall_seconds": concurrency_seconds,
        "concurrency_max_gap_native_psi": float(max(p.gap for p in plans.values())),
        "full_pdca_variables": int(train_arrays.p),
        "full_pdca_wall_seconds": pdca_seconds,
        "full_pdca_candidate_finite": bool(np.all(np.isfinite(candidate))),
    }


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    args = parse_args()

    s = science.ScientificFinalSettings(
        data=args.data,
        artifact_root="artifacts/preflight_not_written",
        cores=args.cores,
        max_wall_minutes=720,
    )
    s.validate()
    cfg = base.build_config(s)

    df = base.load_data(cfg)
    scoped, _ = base.apply_experiment_scope(df, cfg)
    if len(df) != science.EXPECTED_CLEANED_CASES:
        raise RuntimeError(f"cleaned={len(df)} expected={science.EXPECTED_CLEANED_CASES}")
    if len(scoped) != science.EXPECTED_SCOPED_CASES:
        raise RuntimeError(f"scoped={len(scoped)} expected={science.EXPECTED_SCOPED_CASES}")

    train_starts, hold_starts, _ = base.choose_split(scoped, s)
    ws = base.week_start_series(scoped)
    train_mask = ws.isin(train_starts)
    hold_mask = ws.isin(hold_starts)
    train_df = scoped[train_mask].copy()
    hold_count = int(hold_mask.sum())
    if len(train_df) != science.EXPECTED_TRAIN_CASES:
        raise RuntimeError(f"train={len(train_df)} expected={science.EXPECTED_TRAIN_CASES}")
    if hold_count != science.EXPECTED_HOLDOUT_CASES:
        raise RuntimeError(f"holdout count={hold_count} expected={science.EXPECTED_HOLDOUT_CASES}")

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
    if len(train_weeks) != 72:
        raise RuntimeError(f"built {len(train_weeks)} training weeks, expected 72")

    roster = build_fixed_roster(
        pools.train, pd.Timestamp(train_starts[0]), cfg, science.PRIMARY_ROSTER
    )
    roster_counts = {
        site: sum(b.site == site for b in roster.calendar.candidates)
        for site in final.PRIMARY_SITES
    }

    enc = science.ScientificFeatureEncoder().fit(train_df)
    train_arrays = base.build_arrays(train_weeks, enc)
    if train_arrays.p != science.EXPECTED_FEATURES:
        raise RuntimeError(f"p={train_arrays.p} expected={science.EXPECTED_FEATURES}")

    missing = sum(
        int(not week.instance.case_eligible_blocks.get(i))
        for week in train_weeks
        for i in range(week.instance.num_cases)
    )
    if missing:
        raise RuntimeError(f"{missing} training cases have no fixed-roster eligible block")

    report = {
        "status": "PREFLIGHT_OK",
        "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
        "cleaned_cases": len(df),
        "two_site_weekday_cases": len(scoped),
        "train_cases": len(train_df),
        "holdout_cases_count_only": hold_count,
        "train_weeks": len(train_starts),
        "holdout_weeks_count_only": len(hold_starts),
        "holdout_first": str(hold_starts[0].date()),
        "feature_dimension": train_arrays.p,
        "primary_roster": science.PRIMARY_ROSTER,
        "required_capacity_sensitivity": science.REQUIRED_CAPACITY_SENSITIVITY,
        "roster_blocks_total": len(roster.calendar.candidates),
        "roster_blocks_by_site": roster_counts,
        "missing_training_eligibility": missing,
        # Training-only descriptive check; no holdout outcome is summarized here.
        "training_long_valid_realized_outcomes_gt480": int(
            (
                (pd.to_numeric(train_df[Col.ROOM_DURATION], errors="coerce") > science.PLANNING_CASE_LIMIT_MINUTES)
                | (pd.to_numeric(train_df[Col.SURGICAL_DURATION], errors="coerce") > science.PLANNING_CASE_LIMIT_MINUTES)
            ).sum()
        ),
        "solver_check_requested": bool(args.solver_check),
        "note": "No holdout weekly instance was materialized or optimized.",
    }
    if args.solver_check:
        report["solver_check"] = _solver_checks(train_weeks, train_arrays, s)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
