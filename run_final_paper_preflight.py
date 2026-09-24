#!/usr/bin/env python3
"""Cheap local preflight for the frozen final-paper experiment.

This command uses the real workbook but performs no long optimization and does
not materialize or optimize holdout weekly instances. It verifies the revised
booked-only cohort, the 72/22 calendar split, the training-only median-count
roster, fixed-roster eligibility on all 72 training weeks, and the 107-feature
predecision encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return p.parse_args()


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    args = parse_args()

    s = science.ScientificFinalSettings(
        data=args.data,
        artifact_root="artifacts/preflight_not_written",
        cores=1,
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
    train_df = scoped[ws.isin(train_starts)].copy()
    hold_df = scoped[ws.isin(hold_starts)].copy()
    if len(train_df) != science.EXPECTED_TRAIN_CASES:
        raise RuntimeError(f"train={len(train_df)} expected={science.EXPECTED_TRAIN_CASES}")
    if len(hold_df) != science.EXPECTED_HOLDOUT_CASES:
        raise RuntimeError(f"holdout={len(hold_df)} expected={science.EXPECTED_HOLDOUT_CASES}")

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

    # The primary median-count roster is constant across evaluation weeks because
    # it is fitted entirely from the frozen 72-week training cohort.
    roster = build_fixed_roster(pools.train, pd.Timestamp(train_starts[0]), cfg, science.PRIMARY_ROSTER)
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
        "holdout_cases_count_only": len(hold_df),
        "train_weeks": len(train_starts),
        "holdout_weeks_count_only": len(hold_starts),
        "holdout_first": str(hold_starts[0].date()),
        "feature_dimension": train_arrays.p,
        "primary_roster": science.PRIMARY_ROSTER,
        "roster_blocks_total": len(roster.calendar.candidates),
        "roster_blocks_by_site": roster_counts,
        "missing_training_eligibility": missing,
        "long_realized_outcomes_retained": int(
            (
                (pd.to_numeric(scoped[Col.ROOM_DURATION], errors="coerce") > science.PLANNING_CASE_LIMIT_MINUTES)
                | (pd.to_numeric(scoped[Col.SURGICAL_DURATION], errors="coerce") > science.PLANNING_CASE_LIMIT_MINUTES)
            ).sum()
        ),
        "note": "No long MILP/pDCA run and no holdout weekly instance was materialized or optimized.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
