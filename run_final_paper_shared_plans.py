#!/usr/bin/env python3
"""Solve behavior-independent training weekly plans once and reuse them.

This runner builds the exact frozen 72-week training planning instances, solves
REALIZED_ORACLE and/or BOOKED, saves complete assignments/bounds/statuses, and
keeps one Gurobi log per site solve.  A later invocation may load a previous
shared-plan root, use those assignments as MIP starts, and spend more time only
on weeks whose native-Psi gap still exceeds the requested target.

Examples
--------
Initial shared backbone::

    python run_final_paper.py shared-plans \
      --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
      --artifact-root artifacts/final_paper_shared_plans_v1 \
      --cores 15 --kind both --verbose

Refine the same solved weeks with longer limits, using saved schedules as starts::

    python run_final_paper.py shared-plans \
      --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
      --artifact-root artifacts/final_paper_shared_plans_v2 \
      --warm-start-root artifacts/final_paper_shared_plans_v1 \
      --cores 15 --kind both --oracle-seconds 3600 --booked-seconds 900 --verbose
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import final_paper_release_guard as release
import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import final_paper_shared_plans as shared
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--warm-start-root", default=None)
    p.add_argument("--cores", type=int, default=15)
    p.add_argument("--kind", choices=("oracle", "booked", "both"), default="both")
    p.add_argument("--oracle-seconds", type=int, default=1800)
    p.add_argument("--booked-seconds", type=int, default=300)
    p.add_argument("--booked-gap", type=float, default=0.01)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-data-mismatch", action="store_true")
    return p.parse_args()


def _build_training_weeks(s):
    cfg = base.build_config(s)
    df = base.load_data(cfg)
    scoped, scope_summary = base.apply_experiment_scope(df, cfg)
    train_starts, hold_starts, split_audit = base.choose_split(scoped, s)
    ws = base.week_start_series(scoped)
    train_df = scoped[ws.isin(train_starts)].copy()
    hold_count = int(ws.isin(hold_starts).sum())
    if s.strict_data_freeze:
        if len(df) != science.EXPECTED_CLEANED_CASES:
            raise RuntimeError(f"Cleaned cohort changed: {len(df)} != {science.EXPECTED_CLEANED_CASES}")
        if len(scoped) != science.EXPECTED_SCOPED_CASES:
            raise RuntimeError(f"Scoped cohort changed: {len(scoped)} != {science.EXPECTED_SCOPED_CASES}")
        if len(train_df) != science.EXPECTED_TRAIN_CASES or hold_count != science.EXPECTED_HOLDOUT_CASES:
            raise RuntimeError(
                f"Training split changed: train={len(train_df)} holdout={hold_count}"
            )
        site_counts = {
            str(k): int(v)
            for k, v in train_df[base.Col.SITE].value_counts().to_dict().items()
        }
        if site_counts != science.EXPECTED_TRAIN_SITE_COUNTS:
            raise RuntimeError(f"Training site counts changed: {site_counts}")

    hold_start = min(hold_starts)
    hist = scoped[ws < hold_start].copy()
    pools = base.build_candidate_pools(hist, cfg)
    elig = base.build_eligibility_maps(hist, cfg)
    weeks = base.build_bundles(
        scoped,
        train_starts,
        cfg=cfg,
        candidate_pools=pools,
        eligibility_maps=elig,
        offset=0,
    )
    return weeks, train_starts, hold_starts, split_audit, scope_summary


def _refine(
    kind: str,
    old,
    weeks,
    durations,
    s,
    *,
    seconds: int,
    gap: float,
):
    targets = [
        w
        for w in weeks
        if (not old[w.position].exact) and float(old[w.position].gap) > float(gap)
    ]
    if not targets:
        base.LOG.info("[SHARED] %s already meets target on all weeks; no refinement solves", kind)
        return dict(old)

    base.LOG.info(
        "[SHARED] refining %s on %d/%d weeks using saved assignments as MIP starts | limit=%ss gap=%.4g",
        kind,
        len(targets),
        len(weeks),
        seconds,
        gap,
    )
    target_map = {w.position: durations[w.position] for w in targets}
    warm = {w.position: old[w.position].column for w in targets}
    new = base.solve_batch(
        targets,
        target_map,
        s,
        seconds=int(seconds),
        gap=float(gap),
        label=f"shared_{kind}_refine",
        warm_by_week=warm,
    )
    merged_subset = shared.merge_refinement(
        {w.position: old[w.position] for w in targets},
        new,
        targets,
        target_map,
        s,
    )
    out = dict(old)
    out.update(merged_subset)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    warm_root = None if args.warm_start_root is None else Path(args.warm_start_root).resolve()

    # Install the identical reviewed fixed-capacity/numeric stack used by Stage 1,
    # plus spawn-safe persistent weekly Gurobi logging.
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    shared.install_flexible_behavior_validation()
    shared.install_spawn_safe_weekly_logging()
    release.install_reviewed_guards()
    shared.install_weekly_logging()

    s = science.ScientificFinalSettings(
        data=args.data,
        artifact_root=str(root),
        cores=int(args.cores),
        verbose=bool(args.verbose),
        oracle_seconds=int(args.oracle_seconds),
        train_planner_seconds=int(args.booked_seconds),
        train_planner_gap=float(args.booked_gap),
        strict_data_freeze=not bool(args.allow_data_mismatch),
    )
    s.validate()
    base.setup_logging(root, s.verbose)

    if root.exists() and any(root.iterdir()) and warm_root is None:
        raise RuntimeError(
            f"Artifact root is not empty: {root}. Use --warm-start-root for an explicit refinement run."
        )
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    base.write_json(
        root / "SHARED_RUN_STATUS.json",
        {
            "status": "STARTED",
            "version": shared.SHARED_PLANS_VERSION,
            "git_head": base.git_head(),
            "kind": args.kind,
            "warm_start_root": None if warm_root is None else str(warm_root),
            "oracle_seconds": int(args.oracle_seconds),
            "booked_seconds": int(args.booked_seconds),
            "booked_gap": float(args.booked_gap),
        },
    )

    try:
        weeks, train_starts, hold_starts, split_audit, scope_summary = _build_training_weeks(s)
        split_audit.to_csv(root / "WEEK_SPLIT.csv", index=False)
        actual = {
            w.position: np.asarray(w.instance.actual_durations(), dtype=float)
            for w in weeks
        }
        booked = {
            w.position: np.asarray(w.instance.booked_durations(), dtype=float)
            for w in weeks
        }

        requested = {args.kind} if args.kind != "both" else {"oracle", "booked"}
        if "oracle" in requested:
            if warm_root is None:
                base.LOG.info("[SHARED] solving behavior-independent realized oracle once")
                oracle = base.solve_oracle_batch(weeks, actual, s, label="shared_oracle")
            else:
                old = shared.load_plan_set(warm_root, "oracle", weeks, actual, s)
                oracle = _refine(
                    "oracle",
                    old,
                    weeks,
                    actual,
                    s,
                    seconds=int(args.oracle_seconds),
                    gap=float(s.oracle_gap),
                )
            shared.save_plan_set(root, "oracle", oracle, weeks, actual, s)

        if "booked" in requested:
            if warm_root is None:
                base.LOG.info("[SHARED] solving behavior-independent BOOKED plans once")
                booked_plans = base.solve_batch(
                    weeks,
                    booked,
                    s,
                    seconds=int(args.booked_seconds),
                    gap=float(args.booked_gap),
                    label="shared_booked",
                )
            else:
                old = shared.load_plan_set(warm_root, "booked", weeks, booked, s)
                booked_plans = _refine(
                    "booked",
                    old,
                    weeks,
                    booked,
                    s,
                    seconds=int(args.booked_seconds),
                    gap=float(args.booked_gap),
                )
            shared.save_plan_set(root, "booked", booked_plans, weeks, booked, s)

        manifest = json.loads((root / "SHARED_PLANS_MANIFEST.json").read_text(encoding="utf-8"))
        base.write_json(
            root / "SHARED_RUN_STATUS.json",
            {
                "status": "COMPLETE",
                "version": shared.SHARED_PLANS_VERSION,
                "git_head": base.git_head(),
                "artifact_root": str(root),
                "elapsed_seconds": time.monotonic() - start,
                "train_weeks": len(weeks),
                "train_first": str(train_starts[0].date()),
                "train_last": str(train_starts[-1].date()),
                "holdout_first": str(hold_starts[0].date()),
                "holdout_last": str(hold_starts[-1].date()),
                "plan_sets": manifest.get("plan_sets", {}),
                "gurobi_logs": str(root / "gurobi_logs"),
                "note": (
                    "These plan sets are behavior-independent and are intended to be reused by "
                    "all alpha/h training scenarios."
                ),
            },
        )
        print(
            json.dumps(
                {
                    "status": "SHARED_PLANS_COMPLETE",
                    "artifact_root": str(root),
                    "plan_sets": manifest.get("plan_sets", {}),
                    "gurobi_logs": str(root / "gurobi_logs"),
                },
                indent=2,
            )
        )
    except Exception as exc:
        base.write_json(
            root / "SHARED_RUN_STATUS.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        base.LOG.exception("Shared-plan solve failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
