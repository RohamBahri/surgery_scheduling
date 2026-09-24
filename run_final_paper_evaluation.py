#!/usr/bin/env python3
"""Stage 2: one-shot evaluation of the accepted Stage-1 policy bundle.

This program never trains a policy.  It verifies the frozen Stage-1 artifacts,
requires the exact clean Git commit used for training, writes a consumption
marker before loading/materializing holdout data, then evaluates the exact frozen
policies once.

Final policy schedules use deterministic Gurobi WorkLimit with one solver thread,
a fixed seed and predecision model ordering.  A wall-clock TimeLimit is retained
only as an emergency guard; if that guard fires first, the evaluation fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.solvers.fixed_capacity import schedule_metrics


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tracked_tree_is_dirty() -> bool:
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
        return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument("--training-artifact-root", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--cores", type=int, default=None, help="Default: use the frozen Stage-1 value")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--run-response-sensitivities",
        action="store_true",
        help="Also run the three predeclared response-misspecification scenarios in this one-shot evaluation.",
    )
    return p.parse_args()


def _verify_training_bundle(root: Path) -> dict:
    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError(f"Missing Stage-1 freeze: {freeze_path}")
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "TRAINING_COMPLETE_HOLDOUT_LOCKED":
        raise RuntimeError("Stage-1 bundle is not an accepted holdout-locked training run")
    if freeze.get("scientific_spec_version") != science.SCIENTIFIC_SPEC_VERSION:
        raise RuntimeError("Training scientific-spec version does not match this evaluator")
    if freeze.get("runtime_fixes_version") != fixes.RUNTIME_FIXES_VERSION:
        raise RuntimeError("Training runtime-fix version does not match this evaluator")
    if freeze.get("git_head") != base.git_head():
        raise RuntimeError(
            f"Git HEAD differs from frozen training code: current={base.git_head()} "
            f"frozen={freeze.get('git_head')}. Run Stage 2 from a git worktree at the frozen commit."
        )
    if _tracked_tree_is_dirty():
        raise RuntimeError("Tracked evaluation source is modified; Stage 2 requires a clean frozen worktree")
    for name, expected in dict(freeze.get("artifact_fingerprints", {})).items():
        path = root / name
        if not path.exists():
            raise RuntimeError(f"Frozen training artifact is missing: {name}")
        if base.sha256_file(path) != expected:
            raise RuntimeError(f"Frozen training artifact hash changed: {name}")
    for marker in ("HOLDOUT_EVALUATION_STARTED.json", "HOLDOUT_EVALUATED.json"):
        path = root / marker
        if path.exists():
            raise RuntimeError(
                f"This Stage-1 bundle has already started/finished holdout consumption: {path}"
            )
    return freeze


def _load_settings(training_root: Path, eval_root: Path, args: argparse.Namespace):
    payload = _read_json(training_root / "FROZEN_SETTINGS.json")
    payload["data"] = args.data
    payload["artifact_root"] = str(eval_root)
    if args.cores is not None:
        payload["cores"] = int(args.cores)
    if args.verbose:
        payload["verbose"] = True
    s = science.ScientificFinalSettings(**payload)
    s.validate()
    return s


def _policy_meta(name: str, w, a: base.Arrays, s) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    dm, meta = base.policy_duration_map(name, w, a, s)
    if w is None or name in {"BOOKED", "ORACLE"}:
        return dm, {**meta, "display_clip_fraction": math.nan, "floor_clip_fraction": math.nan}
    raw = np.asarray(a.X @ np.asarray(w, float), float).reshape(-1)
    delta, _, _ = base.correction_and_planning(np.asarray(w, float), a, s)
    lower = np.maximum(-float(s.display_cap), final.MIN_RECOMMENDED_DURATION - a.booked)
    return dm, {
        **meta,
        "display_clip_fraction": float(np.mean(np.abs(raw - delta) > 1e-9)),
        "floor_clip_fraction": float(np.mean(raw < lower - 1e-9)),
    }


def _assignment_rows(method: str, week: base.WeekBundle, plan: base.PlanResult) -> list[dict]:
    rows = []
    for (i, bid), value in plan.column.z_assign.items():
        if value <= 0.5:
            continue
        case = week.instance.cases[int(i)]
        rows.append(
            {
                "method": method,
                "week": week.position,
                "week_start": str(week.start.date()),
                "local_case_index": int(i),
                "case_id": int(case.case_id),
                "site": str(case.site),
                "booked_minutes": float(case.booked_duration_min),
                "actual_minutes": float(case.actual_duration_min),
                "block_day_index": int(bid.day_index),
                "block_site": str(bid.site),
                "block_room": str(bid.room),
            }
        )
    return rows


def _site_metric_rows(method, week, plan, actual, s) -> list[dict]:
    rows = []
    for site in final.PRIMARY_SITES:
        view = final._site_view(week, site)
        local = final._localize_warm(plan.column, view)
        if local is None:
            raise RuntimeError("Failed to localize final schedule")
        m = schedule_metrics(
            local,
            np.asarray(actual, float)[view.global_indices],
            final.final_cost_cfg(s),
            final.PRIMARY_TURNOVER,
        )
        rows.append(
            {
                "method": method,
                "week": week.position,
                "week_start": str(week.start.date()),
                "site": site,
                "realized_cost": float(m["phi"]),
                "overtime_minutes": float(m["overtime_minutes"]),
                "idle_minutes": float(m["idle_minutes"]),
                "turnover_minutes": float(m["turnover_minutes"]),
                "occupied_blocks": int(m["occupied_blocks"]),
            }
        )
    return rows


def _evaluate(
    weeks,
    a: base.Arrays,
    policies: Mapping[str, np.ndarray | None],
    oracle_plans,
    s,
    root: Path,
):
    week_lookup = {w.position: w for w in weeks}
    oracle_lb = {w: r.bound for w, r in oracle_plans.items()}
    oracle_ub = {w: r.objective for w, r in oracle_plans.items()}
    all_results = {"ORACLE": dict(oracle_plans)}
    metas = {
        "ORACLE": {
            "raw_mae": math.nan,
            "implemented_mae": 0.0,
            "acceptance": math.nan,
            "decay": math.nan,
            "discard": math.nan,
            "display_clip_fraction": math.nan,
            "floor_clip_fraction": math.nan,
            "retrospective_benchmark": True,
        }
    }

    for name, w in policies.items():
        dm, meta = _policy_meta(name, w, a, s)
        metas[name] = {**meta, "retrospective_benchmark": False}
        all_results[name] = science.deterministic_eval_solve_batch(
            weeks,
            dm,
            s,
            work_limit=s.final_planner_work_limit,
            wall_seconds=s.final_planner_seconds,
            gap=s.final_planner_gap,
            label=f"final_holdout_{name}",
        )

    # Retrospective behavioral ceiling: realized correction projected onto the
    # implementable interval, then passed through the same deterministic planner.
    impl_planning, impl_corr = science.implementable_oracle_planning(a, s)
    impl_dm = {wk: impl_planning[idx] for wk, idx in a.week_slices.items()}
    all_results["IMPLEMENTABLE_ORACLE"] = science.deterministic_eval_solve_batch(
        weeks,
        impl_dm,
        s,
        work_limit=s.final_planner_work_limit,
        wall_seconds=s.final_planner_seconds,
        gap=s.final_planner_gap,
        label="final_holdout_implementable_oracle",
    )
    metas["IMPLEMENTABLE_ORACLE"] = {
        "raw_mae": math.nan,
        "implemented_mae": float(np.mean(np.abs(a.error - impl_corr))),
        "acceptance": math.nan,
        "decay": math.nan,
        "discard": math.nan,
        "display_clip_fraction": math.nan,
        "floor_clip_fraction": math.nan,
        "retrospective_benchmark": True,
    }

    realized_candidates = {wk: [] for wk in a.week_slices}
    for name, plans in all_results.items():
        if name == "ORACLE":
            continue
        for wk, idx in a.week_slices.items():
            realized_candidates[wk].append(
                float(
                    plans[wk].column.compute_cost(
                        a.actual[idx], final.final_cost_cfg(s), final.PRIMARY_TURNOVER
                    )
                )
            )
    effective_oracle_ub = {
        wk: min([float(oracle_ub[wk])] + realized_candidates[wk]) for wk in a.week_slices
    }

    weekly_rows = []
    summary_rows = []
    assignments = []
    site_rows = []
    for name, plans in all_results.items():
        total_real = total_lo = total_hi = 0.0
        max_gap = 0.0
        exact_count = 0
        for wk, idx in a.week_slices.items():
            plan = plans[wk]
            rc = (
                float(plan.objective)
                if name == "ORACLE"
                else float(
                    plan.column.compute_cost(
                        a.actual[idx], final.final_cost_cfg(s), final.PRIMARY_TURNOVER
                    )
                )
            )
            lo = max(0.0, rc - effective_oracle_ub[wk])
            hi = max(0.0, rc - float(oracle_lb[wk]))
            total_real += rc
            total_lo += lo
            total_hi += hi
            max_gap = max(max_gap, float(plan.gap))
            exact_count += int(plan.exact)
            weekly_rows.append(
                {
                    "split": "holdout",
                    "method": name,
                    "week": wk,
                    "week_start": str(week_lookup[wk].start.date()),
                    "planning_obj_phi": float(plan.objective),
                    "planning_bound_phi": float(plan.bound),
                    "planning_gap_native_psi": float(plan.gap),
                    "planning_exact": bool(plan.exact),
                    "planning_status": str(plan.status),
                    "planning_seconds": float(plan.solve_seconds),
                    "stopping_rule": (
                        "oracle time/gap certificate"
                        if name == "ORACLE"
                        else "deterministic WorkLimit or MIPGap; TimeLimit is fail-only"
                    ),
                    "realized_cost": rc,
                    "oracle_lb_phi": float(oracle_lb[wk]),
                    "oracle_ub_effective_phi": float(effective_oracle_ub[wk]),
                    "regret_lower": lo,
                    "regret_upper": hi,
                }
            )
            assignments.extend(_assignment_rows(name, week_lookup[wk], plan))
            site_rows.extend(_site_metric_rows(name, week_lookup[wk], plan, a.actual[idx], s))

        meta = metas[name]
        n = a.n_weeks
        summary_rows.append(
            {
                "split": "holdout",
                "method": name,
                "avg_realized_cost": total_real / n,
                "avg_regret_lower": total_lo / n,
                "avg_regret_upper": total_hi / n,
                "avg_regret_mid": 0.5 * (total_lo + total_hi) / n,
                "max_planning_gap_native_psi": max_gap,
                "planning_exact_weeks": exact_count,
                **meta,
            }
        )

    weekly = pd.DataFrame(weekly_rows)
    summary = pd.DataFrame(summary_rows)
    booked = summary[summary.method == "BOOKED"].iloc[0]
    blo = max(1e-9, float(booked.avg_regret_lower))
    bhi = max(1e-9, float(booked.avg_regret_upper))
    summary["gap_closed_lower_pct"] = [
        100.0 * (1.0 - float(r.avg_regret_upper) / blo) if blo > 1e-8 else math.nan
        for _, r in summary.iterrows()
    ]
    summary["gap_closed_upper_pct"] = [
        100.0 * (1.0 - float(r.avg_regret_lower) / bhi) if bhi > 1e-8 else math.nan
        for _, r in summary.iterrows()
    ]

    weekly.to_csv(root / "FINAL_HOLDOUT_WEEKLY.csv", index=False)
    summary.to_csv(root / "FINAL_HOLDOUT_SUMMARY.csv", index=False)
    pd.DataFrame(assignments).to_csv(root / "FINAL_HOLDOUT_ASSIGNMENTS.csv", index=False)
    pd.DataFrame(site_rows).to_csv(root / "FINAL_HOLDOUT_SITE_WEEKLY.csv", index=False)
    return weekly, summary


def _run_response_sensitivities(weeks, a, policies, oracle_plans, s, root: Path) -> None:
    rows = []
    oracle_lb = {w: r.bound for w, r in oracle_plans.items()}
    for scenario, (alpha, h) in science.RESPONSE_SENSITIVITY_SCENARIOS.items():
        ss = copy.copy(s)
        ss.alpha = float(alpha)
        ss.h = float(h)
        for name, w in policies.items():
            dm, _ = _policy_meta(name, w, a, ss)
            plans = science.deterministic_eval_solve_batch(
                weeks,
                dm,
                ss,
                work_limit=s.sensitivity_planner_work_limit,
                wall_seconds=s.final_planner_seconds,
                gap=max(s.final_planner_gap, 0.002),
                label=f"response_{scenario}_{name}",
            )
            for wk, idx in a.week_slices.items():
                plan = plans[wk]
                rc = float(
                    plan.column.compute_cost(
                        a.actual[idx], final.final_cost_cfg(ss), final.PRIMARY_TURNOVER
                    )
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "alpha": alpha,
                        "h": h,
                        "method": name,
                        "week": wk,
                        "realized_cost": rc,
                        "planning_gap_native_psi": plan.gap,
                        "planning_status": plan.status,
                        "regret_upper": max(0.0, rc - float(oracle_lb[wk])),
                    }
                )
    pd.DataFrame(rows).to_csv(root / "RESPONSE_SENSITIVITY_WEEKLY.csv", index=False)
    if rows:
        (
            pd.DataFrame(rows)
            .groupby(["scenario", "alpha", "h", "method"], as_index=False)
            .agg(
                avg_realized_cost=("realized_cost", "mean"),
                avg_regret_upper=("regret_upper", "mean"),
                max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
            )
            .to_csv(root / "RESPONSE_SENSITIVITY_SUMMARY.csv", index=False)
        )


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    args = _parse_args()

    training_root = Path(args.training_artifact_root).resolve()
    eval_root = Path(args.artifact_root).resolve()
    if not training_root.exists():
        raise RuntimeError(f"Training artifact root does not exist: {training_root}")
    if eval_root.exists() and any(eval_root.iterdir()):
        raise RuntimeError(f"Evaluation artifact root is not empty: {eval_root}")

    freeze = _verify_training_bundle(training_root)
    s = _load_settings(training_root, eval_root, args)
    data_path = Path(s.data).resolve()
    data_freeze = _read_json(training_root / "DATA_FREEZE.json")
    if base.sha256_file(data_path) != data_freeze["input_sha256"]:
        raise RuntimeError("Evaluation workbook hash differs from Stage-1 workbook")

    eval_root.mkdir(parents=True, exist_ok=True)
    base.setup_logging(eval_root, s.verbose)
    base.write_json(eval_root / "EVALUATION_SETTINGS.json", asdict(s))
    base.write_json(
        eval_root / "TIE_RULE.json",
        {
            "rule": science.DEPLOYMENT_TIE_RULE,
            "secondary_mip": False,
            "solver_threads_per_site": 1,
            "seed": int(s.random_seed),
            "work_limit_per_site": float(s.final_planner_work_limit),
            "emergency_wall_limit_seconds_per_site": int(s.final_planner_seconds),
            "case_order": "predecision lexical order used by Stage 1/Stage 2 instance builder",
        },
    )

    # Consumption starts here, before the workbook is loaded or any holdout week
    # is materialized.  A failed run leaves this marker in place deliberately.
    started_marker = training_root / "HOLDOUT_EVALUATION_STARTED.json"
    base.write_json(
        started_marker,
        {
            "status": "HOLDOUT_CONSUMPTION_STARTED",
            "git_head": base.git_head(),
            "evaluation_artifact_root": str(eval_root),
            "training_freeze_sha256": base.sha256_file(training_root / "TRAINING_FREEZE.json"),
            "run_response_sensitivities": bool(args.run_response_sensitivities),
            "started_unix_time": time.time(),
        },
    )

    start = time.monotonic()
    try:
        cfg = base.build_config(s)
        df = base.load_data(cfg)
        scoped, scope_summary = base.apply_experiment_scope(df, cfg)
        if len(scoped) != science.EXPECTED_SCOPED_CASES:
            raise RuntimeError(
                f"Scoped cohort changed: {len(scoped)} != {science.EXPECTED_SCOPED_CASES}"
            )
        train_starts, hold_starts, split_audit = base.choose_split(scoped, s)
        split_audit.to_csv(eval_root / "WEEK_SPLIT.csv", index=False)
        ws = base.week_start_series(scoped)
        train_count = int(ws.isin(train_starts).sum())
        hold_count = int(ws.isin(hold_starts).sum())
        if train_count != science.EXPECTED_TRAIN_CASES or hold_count != science.EXPECTED_HOLDOUT_CASES:
            raise RuntimeError(
                f"Evaluation split changed: train={train_count}, holdout={hold_count}"
            )
        hold_df = scoped[ws.isin(hold_starts)].copy()
        hold_sites = {str(k): int(v) for k, v in hold_df[base.Col.SITE].value_counts().to_dict().items()}
        if hold_sites != science.EXPECTED_HOLDOUT_SITE_COUNTS:
            raise RuntimeError(f"Holdout site counts changed: {hold_sites}")

        hold_start = min(hold_starts)
        hist = scoped[ws < hold_start].copy()
        pools = base.build_candidate_pools(hist, cfg)
        elig = base.build_eligibility_maps(hist, cfg)
        hold_weeks = base.build_bundles(
            scoped,
            hold_starts,
            cfg=cfg,
            candidate_pools=pools,
            eligibility_maps=elig,
            offset=s.train_weeks,
        )

        enc = science.ScientificFeatureEncoder.from_manifest(
            _read_json(training_root / "FEATURE_MANIFEST.json")
        )
        hold_a = base.build_arrays(hold_weeks, enc)
        if hold_a.p != science.EXPECTED_FEATURES:
            raise RuntimeError("Holdout feature dimension differs from frozen training encoder")

        policy_file = np.load(training_root / "POLICIES.npz")
        required = ("NAIVE", "RA", "RA_FULL", "SITE_SHIFT", "OS", "VF")
        policies = {name: np.asarray(policy_file[name], float) for name in required}
        for name, w in policies.items():
            if w.shape != (science.EXPECTED_FEATURES,):
                raise RuntimeError(f"Frozen policy {name} has unexpected shape {w.shape}")

        hold_actual = {wk: hold_a.actual[idx] for wk, idx in hold_a.week_slices.items()}
        hold_oracle = base.solve_oracle_batch(hold_weeks, hold_actual, s, label="oracle_holdout")
        base.write_csv(
            eval_root / "ORACLE_HOLDOUT.csv",
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
                for w, r in sorted(hold_oracle.items())
            ],
        )

        eval_policies = {"BOOKED": None, **policies}
        _, summary = _evaluate(
            hold_weeks, hold_a, eval_policies, hold_oracle, s, eval_root
        )
        if args.run_response_sensitivities:
            _run_response_sensitivities(
                hold_weeks, hold_a, eval_policies, hold_oracle, s, eval_root
            )

        payload = {
            "status": "HOLDOUT_EVALUATION_COMPLETE",
            "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
            "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
            "git_head": base.git_head(),
            "training_freeze_sha256": base.sha256_file(training_root / "TRAINING_FREEZE.json"),
            "holdout_cases": hold_count,
            "holdout_weeks": len(hold_weeks),
            "max_oracle_gap_native_psi": float(max(r.gap for r in hold_oracle.values())),
            "max_policy_gap_native_psi": float(
                summary.loc[
                    ~summary.method.isin(["ORACLE"]), "max_planning_gap_native_psi"
                ].max()
            ),
            "deployment_tie_rule": science.DEPLOYMENT_TIE_RULE,
            "response_sensitivities_run": bool(args.run_response_sensitivities),
            "scope_summary": asdict(scope_summary),
            "elapsed_seconds": time.monotonic() - start,
        }
        base.write_json(eval_root / "FINAL_DECISION.json", payload)
        base.write_json(eval_root / "RUN_STATUS.json", payload)
        base.write_json(
            training_root / "HOLDOUT_EVALUATED.json",
            {
                "status": "HOLDOUT_CONSUMED",
                "evaluation_artifact_root": str(eval_root),
                "evaluation_final_decision_sha256": base.sha256_file(
                    eval_root / "FINAL_DECISION.json"
                ),
                "git_head": base.git_head(),
            },
        )
        base.LOG.info("[DONE] frozen-policy holdout evaluation complete")
    except Exception as exc:
        base.write_json(
            eval_root / "RUN_STATUS.json",
            {
                "status": "FAILED_AFTER_HOLDOUT_CONSUMPTION_STARTED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        base.LOG.exception("Stage 2 failed after the consumption marker was written: %s", exc)
        raise


if __name__ == "__main__":
    main()
