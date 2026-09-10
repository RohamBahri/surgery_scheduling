#!/usr/bin/env python3
"""Training-only audit of the fixed-capacity weekly planner.

PREREGISTRATION (MODEL_SPEC below is written before loading any data):
  TGH, Monday-Friday cohort; day FLEXIBLE; no deferral or capacity opening.
  Raw service-room compatibility: 3 historical training weeks; audit 1,2,3,5.
  Turnover: 30 minutes; sensitivity 0,20,30,40. OT=15, idle=10, H=480.
  No hard overtime bound, surgeon-room-count rule, VF training, or holdout solves.
  The final 94 eligible weeks define the frozen 72/22 split; minimum 50 cases.
  Templates and compatibility use the 72 training weeks (in-sample audit).
  Plausible calibration gaps are preregistered as [0,240] minutes.
  Quick weeks are the first, middle (index 35), and last training weeks.

Phi/Psi-shifted/Psi are independent cold solves with matching solver settings.
Psi-shifted uses the Psi model plus K; only the objective constant changes.
An exact attempt requests BOTH gaps zero; its termination is still reported.
Quick mode shortens budgets, not the model. An unrestricted Gurobi license is
needed for dataset-sized models. Solver failures produce incomplete artifacts
and a nonzero exit code, never fabricated bounds or an OPTIMAL label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import Config, SolverConfig
from src.core.types import Col
from src.data.loader import load_data
from src.data.scope import apply_experiment_scope
from src.planning.eligibility import (
    CapacityModelError,
    fit_service_room_history,
    resolve_eligibility,
)
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.retrospective import (
    eligibility_components,
    historical_assignment,
    historical_baseline,
    reassignment_metrics,
    turnover_audit,
)
from src.planning.roster import ROSTER_SOURCES, build_fixed_roster, week_starts
from src.solvers.fixed_capacity import schedule_metrics, solve_fixed_capacity_assignment
from src.solvers.result import absolute_gap, relative_gap

LOG = logging.getLogger("weekly_planner_audit")
MODEL_SPEC = {
    "version": 2,
    "site": "TGH",
    "cohort_weekdays": [0, 1, 2, 3, 4],
    "day": "flexible",
    "deferral": "forbidden",
    "opening_decision": "none",
    "capacity_minutes": 480,
    "overtime_per_minute": 15,
    "idle_per_minute": 10,
    "hard_overtime_limit": None,
    "hard_surgeon_room_count": None,
    "eligibility_source": "case_service_raw, same site, distinct training weeks, no weekday condition",
    "eligibility_weeks": 3,
    "eligibility_sensitivity": [1, 2, 3, 5],
    "fallback": "empty service-compatible set -> all allocated blocks at same site; no site capacity -> failure",
    "turnover_minutes": 30,
    "turnover_sensitivity": [0, 20, 30, 40],
    "turnover_plausible_gap_range": [0, 240],
    "roster_sources": list(ROSTER_SOURCES),
    "regular_min_activation_rate": 0.25,
    "median_rounding": "floor(median + 0.5); zero inactive weeks; lexicographic room ties",
    "roster_and_eligibility_history": "all 72 training weeks; retrospective in-sample audit",
    "eligibility_crossfit": "leave one week out of compatibility only; all 72 weeks even in quick mode; rosters unchanged",
    "objective_modes": ["phi", "psi_shifted", "psi"],
    "objective_scaling_control": "psi_shifted = K + psi with identical reduced variables and constraints",
    "solver_seed": 0,
    "train_weeks": 72,
    "train_cases": 9289,
    "holdout_weeks_for_boundary_only": 22,
    "holdout_boundary": "2013-01-28",
    "min_cases_per_week": 50,
    "quick_training_indices": [0, 35, 71],
    "symmetry_key": ["site", "capacity", "eligible_case_set"],
    "reassignment": "literal block-ID changes and minimum changes under valid within-group bijections",
    "assignment_optimization_value": "C_H - C_B (not recommendation value)",
    "recommendation_value_later": "C_B - C_R, only after policy retraining",
    "legacy_epsilon_lib_reference": 4602,
}
TABLES = {
    "CAPACITY_AUDIT": [],
    "ELIGIBILITY_AUDIT": [],
    "ELIGIBILITY_CROSSFIT_AUDIT": [],
    "COMPONENT_STRUCTURE": [],
    "PSI_SCALE_AUDIT": [],
    "FIXED_CAPACITY_EXACTNESS": [],
    "HISTORICAL_BASELINE": [],
    "CALIBRATION": [],
    "TURNOVER_SENSITIVITY": [],
    "SOLVE_RESULTS": [],
    "CASE_ELIGIBILITY": [],
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_table(path, rows):
    cooked = [
        {
            k: (
                json.dumps(json_safe(v), sort_keys=True)
                if isinstance(v, (dict, list, tuple))
                else v
            )
            for k, v in row.items()
        }
        for row in rows
    ]
    pd.DataFrame(
        cooked, columns=None if cooked else ["week_start", "roster", "status"]
    ).to_csv(path, index=False)


def freeze_training(
    scoped: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp], dict]:
    """Only aggregate week counts beyond training are used to preserve the split."""
    starts = week_starts(scoped)
    counts = scoped.groupby(starts).size().sort_index()
    eligible = counts[counts >= MODEL_SPEC["min_cases_per_week"]]
    if len(eligible) < 94:
        raise RuntimeError(
            f"DATA FREEZE MISMATCH: need 94 eligible weeks, found {len(eligible)}"
        )
    selected = eligible.iloc[-94:]
    train_starts = [pd.Timestamp(w) for w in selected.index[:72]]
    boundary = pd.Timestamp(selected.index[72])
    if boundary != pd.Timestamp(MODEL_SPEC["holdout_boundary"]):
        raise RuntimeError(f"DATA FREEZE MISMATCH: holdout boundary {boundary}")
    train = scoped[starts.isin(train_starts)].copy()
    if len(train) != MODEL_SPEC["train_cases"]:
        raise RuntimeError(
            f"DATA FREEZE MISMATCH: {len(train)} training cases, expected 9289"
        )
    ids = ",".join(map(str, sorted(train[Col.CASE_UID].astype(int))))
    return (
        train,
        train_starts,
        {
            "train_cases": len(train),
            "train_weeks": len(train_starts),
            "train_first": str(train_starts[0].date()),
            "train_last": str(train_starts[-1].date()),
            "train_starts": [str(w.date()) for w in train_starts],
            "holdout_boundary": str(boundary.date()),
            "holdout_instances_materialized": False,
            "holdout_case_level_eligibility_inspected": False,
            "train_case_ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
        },
    )


def run_solve(instance, durations, cfg, solver, mode, tau, key, all_rows, errors):
    import gurobipy as gp

    row = {
        **key,
        "n_cases": instance.num_cases,
        "n_blocks": instance.calendar.total_candidates,
        "n_assignment_pairs": sum(
            len(bids) for bids in instance.case_eligible_blocks.values()
        ),
        "objective_mode": mode,
        "turnover": tau,
        "requested_time_limit": solver.time_limit_seconds,
        "requested_mip_gap": solver.mip_gap,
        "requested_mip_gap_abs": solver.mip_gap_abs,
        "requested_threads": solver.threads,
        "requested_seed": solver.seed,
        "K": cfg.costs.idle_per_minute
        * (
            sum(b.capacity_minutes for b in instance.calendar.candidates)
            - float(np.sum(durations))
            - tau * instance.num_cases
        ),
    }
    try:
        result = solve_fixed_capacity_assignment(
            instance, durations, cfg.costs, tau, solver, mode
        )
    except gp.GurobiError as exc:
        row.update(
            status="SOLVER_ERROR",
            sol_count=0,
            proven_optimal=False,
            error_code=exc.errno,
            error=str(exc),
            obj_val=None,
            obj_bound=None,
            absolute_gap=float("inf"),
            relative_gap=float("inf"),
            runtime_seconds=None,
        )
        errors.append({**key, "objective_mode": mode, "error": str(exc)})
        all_rows.append(row)
        LOG.error("%s %s %s: %s", key["week_start"], key["roster"], mode, exc)
        return None, row
    row.update(
        result.diagnostics.as_dict(),
        K=result.K,
        phi_ub=result.phi_ub,
        phi_lb=result.phi_lb,
        psi_ub=result.psi_ub,
        psi_lb=result.psi_lb,
        symmetry_groups=result.symmetry_groups,
    )
    if result.metrics:
        row.update({f"schedule_{k}": v for k, v in result.metrics.items()})
    else:
        errors.append(
            {
                **key,
                "objective_mode": mode,
                "error": f"No incumbent: {result.diagnostics.status}",
            }
        )
    row.update(
        phi_relative_gap=relative_gap(result.phi_ub, result.phi_lb),
        psi_relative_gap=relative_gap(result.psi_ub, result.psi_lb),
        native_relative_gap_allowance=(
            solver.mip_gap * abs(result.diagnostics.obj_val)
            if result.diagnostics.obj_val is not None
            else None
        ),
    )
    all_rows.append(row)
    LOG.info(
        "%s %s %s %s tau=%s status=%s UB=%s LB=%s abs_gap=%s",
        key["week_start"],
        key["roster"],
        key["durations"],
        mode,
        tau,
        row["status"],
        row.get("obj_val"),
        row.get("obj_bound"),
        row.get("absolute_gap"),
    )
    return result, row


def scale_comparison(key, mode_rows):
    row = dict(key)
    for prefix in MODEL_SPEC["objective_modes"]:
        result = mode_rows[prefix]
        for target, source in (
            ("obj_val", "obj_val"),
            ("obj_bound", "obj_bound"),
            ("abs_gap", "absolute_gap"),
            ("rel_gap", "relative_gap"),
            ("runtime", "runtime_seconds"),
            ("status", "status"),
            ("psi_ub", "psi_ub"),
            ("psi_lb", "psi_lb"),
            ("psi_rel_gap", "psi_relative_gap"),
            ("native_gap_allowance", "native_relative_gap_allowance"),
            ("identity_error", "schedule_identity_error"),
        ):
            row[f"{prefix}_{target}"] = result.get(source)
    phi_row, psi_row = mode_rows["phi"], mode_rows["psi"]
    row.update(
        K=psi_row.get("K"),
        K_plus_psi=psi_row.get("phi_ub"),
        identity_error=psi_row.get("schedule_identity_error"),
        phi_identity_error=phi_row.get("schedule_identity_error"),
        psi_phi_lb=psi_row.get("phi_lb"),
        requested_time_limit=psi_row["requested_time_limit"],
        requested_mip_gap=psi_row["requested_mip_gap"],
        requested_mip_gap_abs=psi_row["requested_mip_gap_abs"],
        requested_threads=psi_row["requested_threads"],
        requested_seed=psi_row["requested_seed"],
    )
    # Do not call a difference between time-limited incumbents an identity error.
    return row


def eligibility_crossfit_rows(train, starts, cfg, history):
    """Isolate compatibility self-inclusion while holding each roster fixed.

    This is not out-of-sample roster or cost validation. In particular, the
    observed-activity roster still uses the evaluated week's realized activity.
    """
    rows = []
    weeks = week_starts(train)
    for wi, start in enumerate(starts):
        frame = train[weeks == start]
        loo_history = fit_service_room_history(train[weeks != start])
        cases = list(
            frame[[Col.CASE_SERVICE_RAW, Col.SITE, Col.OPERATING_ROOM]].itertuples(
                index=False, name=None
            )
        )
        profiles = {(str(s), str(site)) for s, site, _ in cases}
        for source in ROSTER_SOURCES:
            roster = build_fixed_roster(
                train, start, cfg, source, allow_retrospective=True
            )
            for k in MODEL_SPEC["eligibility_sensitivity"]:
                fitted = []
                row = {
                    "week_start": str(start.date()),
                    "roster": source,
                    "threshold_weeks": k,
                    "number_cases": len(cases),
                    "in_sample_training_weeks": history.provenance["n_weeks"],
                    "loo_training_weeks": loo_history.provenance["n_weeks"],
                    "roster_refitted": False,
                }
                for prefix, h in (("in_sample", history), ("loo", loo_history)):
                    decisions = {
                        p: resolve_eligibility(*p, roster.calendar, h, k)
                        for p in profiles
                    }
                    ordered = [decisions[(str(s), str(site))] for s, site, _ in cases]
                    fitted.append(ordered)
                    row.update(
                        {
                            f"{prefix}_primary_coverage": float(
                                np.mean([not d.fallback_used for d in ordered])
                            ),
                            f"{prefix}_fallback_rate": float(
                                np.mean([d.fallback_used for d in ordered])
                            ),
                            f"{prefix}_mean_eligible_blocks": float(
                                np.mean([d.number_final_blocks for d in ordered])
                            ),
                            f"{prefix}_historical_room_compatibility_rate": float(
                                np.mean(
                                    [
                                        h.weeks_by_pair.get(
                                            (str(s), str(site), str(room)), 0
                                        )
                                        >= k
                                        for s, site, room in cases
                                    ]
                                )
                            ),
                        }
                    )
                for metric in (
                    "primary_coverage",
                    "fallback_rate",
                    "mean_eligible_blocks",
                    "historical_room_compatibility_rate",
                ):
                    row[f"delta_{metric}"] = (
                        row[f"loo_{metric}"] - row[f"in_sample_{metric}"]
                    )
                row["fraction_eligible_set_changed"] = float(
                    np.mean(
                        [
                            a.eligible_blocks != b.eligible_blocks
                            for a, b in zip(*fitted)
                        ]
                    )
                )
                rows.append(row)
        if (wi + 1) % 12 == 0 or wi + 1 == len(starts):
            LOG.info("Eligibility crossfit %d/%d training weeks", wi + 1, len(starts))
    return rows


def summarize_crossfit(rows):
    output = []
    for source in ROSTER_SOURCES:
        for k in MODEL_SPEC["eligibility_sensitivity"]:
            selected = [
                r for r in rows if r["roster"] == source and r["threshold_weeks"] == k
            ]
            if not selected:
                continue
            weights = [r["number_cases"] for r in selected]
            metrics = [
                name
                for name in selected[0]
                if name.startswith(("in_sample_", "loo_", "delta_"))
                and not name.endswith("training_weeks")
            ]
            output.append(
                {
                    "roster": source,
                    "threshold_weeks": k,
                    "n_weeks": len(selected),
                    "n_cases": sum(weights),
                    "aggregation": "case-weighted",
                    **{
                        name: float(
                            np.average([r[name] for r in selected], weights=weights)
                        )
                        for name in metrics + ["fraction_eligible_set_changed"]
                    },
                }
            )
    return output


def eligibility_row(instance, week_frame, history, k):
    decisions = list(instance.eligibility_diagnostics.values())
    final = [d["number_final_blocks"] for d in decisions]
    raw = week_frame[Col.CASE_SERVICE_RAW].tolist()
    room_compatible = [
        history.weeks_by_pair.get((str(raw[i]), c.site, c.operating_room), 0) >= k
        for i, c in enumerate(instance.cases)
    ]
    return {
        "threshold_weeks": k,
        "number_cases": instance.num_cases,
        "primary_coverage": float(
            np.mean([d["eligibility_tier"] == 1 for d in decisions])
        ),
        "fallback_rate": float(np.mean([d["fallback_used"] for d in decisions])),
        "mean_eligible_blocks": float(np.mean(final)),
        "median_eligible_blocks": float(np.median(final)),
        "max_eligible_blocks": int(max(final)),
        "historical_room_compatibility_rate": float(np.mean(room_compatible)),
    }


def calibration_row(key, instance, history, solutions, costs):
    output = {
        **key,
        "C_H": history["realized_cost"],
        "historical_overtime": history["overtime_minutes"],
        "historical_occupied_blocks": history["occupied_blocks"],
        "historical_assignment_coverage": history["historical_assignment_coverage"],
        "historical_feasible_under_eligibility": history[
            "historical_assignment_feasible_under_eligibility"
        ],
    }
    actual = instance.actual_durations()
    historical = historical_assignment(instance)
    for label, name in (("B", "booked"), ("O", "realized")):
        candidates = [
            r for r in solutions[name] if r is not None and r.column is not None
        ]
        if not candidates:
            output[f"C_{label}"] = None
            continue
        best = min(candidates, key=lambda r: r.metrics["phi"])
        metrics = schedule_metrics(best.column, actual, costs, 30)
        mapping = {i: bid for (i, bid), v in best.column.z_assign.items() if v > 0.5}
        output[f"C_{label}"] = metrics["phi"]
        output[f"{label}_status"] = best.diagnostics.status
        for field in (
            "overtime_minutes",
            "occupied_blocks",
            "cases_per_occupied_block",
        ):
            output[f"{label}_{field}"] = metrics[field]
        output.update(
            {
                f"{label}_{k}": v
                for k, v in reassignment_metrics(instance, mapping, historical).items()
            }
        )
        bounds = [
            r.phi_lb for r in solutions[name] if r is not None and r.phi_lb is not None
        ]
        lb = max(bounds) if bounds else None
        output[f"{label}_planning_phi_lb"] = lb
        output[f"{label}_planning_absolute_uncertainty"] = absolute_gap(
            best.metrics["phi"], lb
        )
    ch, cb = output.get("C_H"), output.get("C_B")
    output["assignment_optimization_value_C_H_minus_C_B"] = (
        ch - cb if ch is not None and cb is not None else None
    )
    output["attribution"] = (
        "assignment-optimization value; no recommendation policy evaluated"
    )
    return output


def summarize_scale(rows):
    output = []
    for roster in ROSTER_SOURCES:
        for durations in ("booked", "realized"):
            selected = [
                r for r in rows if r["roster"] == roster and r["durations"] == durations
            ]
            for mode in MODEL_SPEC["objective_modes"]:
                values = [
                    r[f"{mode}_psi_ub"]
                    for r in selected
                    if r.get(f"{mode}_psi_ub") is not None
                ]
                gaps = [r.get(f"{mode}_abs_gap") for r in selected]
                finite = len(values) == len(selected) and all(
                    g is not None and math.isfinite(g) for g in gaps
                )
                output.append(
                    {
                        "roster": roster,
                        "durations": durations,
                        "objective_mode": mode,
                        "n_requested": len(selected),
                        "n_incumbents": len(values),
                        "mean_weekly_psi": float(np.mean(values)) if values else None,
                        "mean_absolute_uncertainty": (
                            float(np.mean(gaps)) if finite and gaps else None
                        ),
                        "max_absolute_uncertainty": (
                            max(gaps) if finite and gaps else None
                        ),
                        "all_bounds_available": finite and bool(gaps),
                        "units": "weekly objective cost units",
                        "legacy_epsilon_lib_reference": 4602,
                    }
                )
    return output


def summarize_proxy_comparison(rows):
    """Paired normal-budget Psi intervals; no claim about policy invariance."""
    output = []
    for durations in ("booked", "realized"):
        by_source = {
            source: {
                r["week_start"]: r
                for r in rows
                if r["roster"] == source and r["durations"] == durations
            }
            for source in ROSTER_SOURCES
        }
        for a, b in combinations(ROSTER_SOURCES, 2):
            common = sorted(by_source[a].keys() & by_source[b].keys())
            paired, constants = [], []
            for week in common:
                left, right = by_source[a][week], by_source[b][week]
                constants.append(left["K"] - right["K"])
                intervals = []
                for r in (left, right):
                    ubs = [r.get(f"{m}_psi_ub") for m in MODEL_SPEC["objective_modes"]]
                    lbs = [r.get(f"{m}_psi_lb") for m in MODEL_SPEC["objective_modes"]]
                    ubs = [v for v in ubs if v is not None and math.isfinite(v)]
                    lbs = [v for v in lbs if v is not None and math.isfinite(v)]
                    if ubs and lbs:
                        intervals.append((max(lbs), min(ubs)))
                if len(intervals) == 2:
                    (al, au), (bl, bu) = intervals
                    paired.append((au - bu, al - bu, au - bl))
            complete = bool(common) and len(paired) == len(common)
            output.append(
                {
                    "roster_a": a,
                    "roster_b": b,
                    "durations": durations,
                    "n_common_weeks": len(common),
                    "n_paired_finite_intervals": len(paired),
                    "mean_K_difference_a_minus_b": (
                        float(np.mean(constants)) if constants else None
                    ),
                    "mean_psi_ub_difference_a_minus_b": (
                        float(np.mean([p[0] for p in paired])) if complete else None
                    ),
                    "mean_absolute_paired_psi_ub_difference": (
                        float(np.mean([abs(p[0]) for p in paired]))
                        if complete
                        else None
                    ),
                    "mean_optimal_psi_difference_lb": (
                        float(np.mean([p[1] for p in paired])) if complete else None
                    ),
                    "mean_optimal_psi_difference_ub": (
                        float(np.mean([p[2] for p in paired])) if complete else None
                    ),
                    "status": "AVAILABLE" if complete else "UNAVAILABLE",
                }
            )
    return output


def summarize_structure(rows):
    output = []
    for source in ROSTER_SOURCES:
        capacity = [
            r
            for r in rows["CAPACITY_AUDIT"]
            if r["roster"] == source and "historical_room_day_coverage" in r
        ]
        eligibility = [
            r
            for r in rows["ELIGIBILITY_AUDIT"]
            if r["roster"] == source and r["threshold_weeks"] == 3
        ]
        if not capacity or not eligibility:
            continue
        cases = sum(r["number_cases"] for r in eligibility)
        output.append(
            {
                "roster": source,
                "n_weeks": len(capacity),
                "blocks_min": min(r["number_available_blocks"] for r in capacity),
                "blocks_mean": float(
                    np.mean([r["number_available_blocks"] for r in capacity])
                ),
                "blocks_max": max(r["number_available_blocks"] for r in capacity),
                "mean_historical_room_day_coverage": float(
                    np.mean([r["historical_room_day_coverage"] for r in capacity])
                ),
                "historically_complete_weeks": sum(
                    r["historical_assignment_coverage"] == 1 for r in capacity
                ),
                "case_weighted_fallback_rate_k3": sum(
                    r["fallback_rate"] * r["number_cases"] for r in eligibility
                )
                / cases,
            }
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx"
    )
    parser.add_argument("--artifact-root", default="artifacts/weekly_planner_audit")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="First/middle/last training weeks; unchanged cohort/template history",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Seconds per normal solve; default 10, quick 2",
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=None,
        help="Seconds per zero-gap attempt; default 60, quick 10",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Run training data/roster/eligibility audits; explicitly omit all solves",
    )
    args = parser.parse_args(argv)
    normal_seconds = (
        args.time_limit if args.time_limit is not None else (2 if args.quick else 10)
    )
    exact_seconds = (
        args.exact_time_limit
        if args.exact_time_limit is not None
        else (10 if args.quick else 60)
    )
    if normal_seconds <= 0 or exact_seconds <= 0 or args.threads < 0:
        parser.error("Time limits must be positive and threads nonnegative")
    out = Path(args.artifact_root)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(
            "Use an empty artifact directory so an earlier audit cannot be overwritten"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")],
    )
    cfg = Config()
    cfg.data.excel_file_path = args.data
    cfg.data.horizon_days = 7
    cfg.scope.planning_sites = (MODEL_SPEC["site"],)
    cfg.scope.planning_weekdays = tuple(MODEL_SPEC["cohort_weekdays"])
    cfg.capacity.block_capacity_minutes = MODEL_SPEC["capacity_minutes"]
    cfg.capacity.activation_cost_per_block = 0
    cfg.capacity.min_activation_rate = MODEL_SPEC["regular_min_activation_rate"]
    cfg.capacity.turnover_minutes = MODEL_SPEC["turnover_minutes"]
    cfg.capacity.eligibility_min_weeks = MODEL_SPEC["eligibility_weeks"]
    cfg.costs.overtime_per_minute = MODEL_SPEC["overtime_per_minute"]
    cfg.costs.idle_per_minute = MODEL_SPEC["idle_per_minute"]
    normal = SolverConfig(
        normal_seconds,
        0.01,
        args.threads,
        False,
        mip_gap_abs=1e-10,
        seed=MODEL_SPEC["solver_seed"],
    )
    exact = SolverConfig(
        exact_seconds,
        0,
        args.threads,
        False,
        mip_gap_abs=0,
        seed=MODEL_SPEC["solver_seed"],
    )
    import gurobipy as gp

    spec = {
        **MODEL_SPEC,
        "run": {
            "quick": args.quick,
            "structure_only": args.structure_only,
            "normal_solver": asdict(normal),
            "exact_solver": asdict(exact),
            "gurobi_version": list(gp.gurobi.version()),
        },
    }
    write_json(out / "MODEL_SPEC.json", spec)
    try:
        # Loader filtering stays canonical; imputation cannot use holdout sites.
        cleaned = load_data(cfg, site_history_end=MODEL_SPEC["holdout_boundary"])
        scoped, _ = apply_experiment_scope(cleaned, cfg)
        train, starts, freeze = freeze_training(scoped)
        del cleaned, scoped  # all subsequent analysis has only the training cohort
        freeze["input_sha256"] = hashlib.sha256(
            Path(args.data).read_bytes()
        ).hexdigest()
        try:
            freeze["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            freeze["working_tree_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=ROOT,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            # Downloaded source archives have no .git directory.
            freeze.update(git_head=None, working_tree_dirty=None)
        implementation = [Path(__file__)] + [
            p
            for folder in ("core", "data", "planning", "solvers")
            for p in (ROOT / "src" / folder).glob("*.py")
        ]
        freeze["implementation_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(implementation)
        }
        selected = (
            [starts[i] for i in MODEL_SPEC["quick_training_indices"]]
            if args.quick
            else starts
        )
        freeze["audited_week_starts"] = [str(w.date()) for w in selected]
        freeze["eligibility_crossfit_week_starts"] = [str(w.date()) for w in starts]
        write_json(out / "DATA_FREEZE.json", freeze)
        LOG.info(
            "Frozen %d cases / %d training weeks; auditing %d weeks",
            len(train),
            len(starts),
            len(selected),
        )
        gaps, turnover_summary = turnover_audit(train)
        gaps.to_csv(out / "TURNOVER_AUDIT.csv", index=False)
        plausible = gaps.loc[gaps.plausible_nonnegative, "gap_minutes"]
        hist_counts, edges = np.histogram(plausible, bins=np.arange(0, 241, 10))
        pd.DataFrame(
            {
                "left_minutes": edges[:-1],
                "right_minutes": edges[1:],
                "count": hist_counts,
            }
        ).to_csv(out / "TURNOVER_HISTOGRAM.csv", index=False)
        write_json(out / "TURNOVER_SUMMARY.json", turnover_summary)
        history = fit_service_room_history(train)
        rows = {name: [] for name in TABLES}
        errors = []
        rows["ELIGIBILITY_CROSSFIT_AUDIT"] = eligibility_crossfit_rows(
            train, starts, cfg, history
        )
        write_table(
            out / "ELIGIBILITY_CROSSFIT_AUDIT.csv", rows["ELIGIBILITY_CROSSFIT_AUDIT"]
        )
        for start in selected:
            week_frame = train[week_starts(train) == start]
            for source in ROSTER_SOURCES:
                key = {"week_start": str(start.date()), "roster": source}
                roster = build_fixed_roster(
                    train, start, cfg, source, allow_retrospective=True
                )
                primary = None
                try:
                    for threshold in MODEL_SPEC["eligibility_sensitivity"]:
                        instance = build_weekly_instance_with_calendar(
                            train,
                            start,
                            starts.index(start),
                            cfg,
                            roster.calendar,
                            history,
                            threshold,
                        )
                        rows["ELIGIBILITY_AUDIT"].append(
                            {
                                **key,
                                **eligibility_row(
                                    instance, week_frame, history, threshold
                                ),
                            }
                        )
                        rows["CASE_ELIGIBILITY"].extend(
                            {
                                **key,
                                "threshold_weeks": threshold,
                                "case_id": instance.cases[i].case_id,
                                **d,
                            }
                            for i, d in instance.eligibility_diagnostics.items()
                        )
                        rows["COMPONENT_STRUCTURE"].extend(
                            {**key, "threshold_weeks": threshold, **r}
                            for r in eligibility_components(instance)
                        )
                        if threshold == 3:
                            primary = instance
                except CapacityModelError as exc:
                    errors.append({**key, "error": str(exc)})
                    rows["CAPACITY_AUDIT"].append(
                        {
                            **key,
                            **roster.provenance,
                            "status": "CAPACITY_MODEL_FAILURE",
                            "error": str(exc),
                        }
                    )
                    continue
                hist = historical_baseline(primary, cfg.costs, 30)
                rows["HISTORICAL_BASELINE"].append({**key, **hist})
                rows["CAPACITY_AUDIT"].append(
                    {
                        **key,
                        **roster.provenance,
                        "historical_room_day_coverage": hist[
                            "historical_room_day_coverage"
                        ],
                        "historical_assignment_coverage": hist[
                            "historical_assignment_coverage"
                        ],
                    }
                )
                if (
                    source == "observed_activity_proxy"
                    and hist["historical_assignment_coverage"] != 1
                ):
                    raise AssertionError(
                        "Observed-activity roster must cover every retained historical case"
                    )
                if args.structure_only:
                    continue
                solutions = {"booked": [], "realized": []}
                for duration_name, durations in (
                    ("booked", primary.booked_durations()),
                    ("realized", primary.actual_durations()),
                ):
                    solve_key = {**key, "durations": duration_name}
                    normal_rows = {}
                    for attempt, settings in (
                        ("normal", normal),
                        ("exact_attempt", exact),
                    ):
                        for mode in MODEL_SPEC["objective_modes"]:
                            result, row = run_solve(
                                primary,
                                durations,
                                cfg,
                                settings,
                                mode,
                                30,
                                {**solve_key, "attempt": attempt},
                                rows["SOLVE_RESULTS"],
                                errors,
                            )
                            solutions[duration_name].append(result)
                            if attempt == "normal":
                                normal_rows[mode] = row
                            else:
                                rows["FIXED_CAPACITY_EXACTNESS"].append(row)
                    rows["PSI_SCALE_AUDIT"].append(
                        scale_comparison(solve_key, normal_rows)
                    )
                    for tau in MODEL_SPEC["turnover_sensitivity"]:
                        if tau == 30:
                            row = normal_rows["psi"]
                        else:
                            _, row = run_solve(
                                primary,
                                durations,
                                cfg,
                                normal,
                                "psi",
                                tau,
                                {**solve_key, "attempt": "turnover_sensitivity"},
                                rows["SOLVE_RESULTS"],
                                errors,
                            )
                        rows["TURNOVER_SENSITIVITY"].append(row)
                rows["CALIBRATION"].append(
                    calibration_row(key, primary, hist, solutions, cfg.costs)
                )
                # Preserve completed work if the next week is interrupted.
                for name, table in rows.items():
                    write_table(out / f"{name}.csv", table)
        for name, table in rows.items():
            write_table(out / f"{name}.csv", table)
        scale = summarize_scale(rows["PSI_SCALE_AUDIT"])
        crossfit = summarize_crossfit(rows["ELIGIBILITY_CROSSFIT_AUDIT"])
        proxy_comparison = summarize_proxy_comparison(rows["PSI_SCALE_AUDIT"])
        write_table(out / "PSI_PROXY_COMPARISON.csv", proxy_comparison)
        structure = summarize_structure(rows)
        component_counts = (
            pd.DataFrame(rows["COMPONENT_STRUCTURE"])
            .query("threshold_weeks == 3 and n_cases > 0")
            .groupby(["week_start", "roster"])
            .size()
            .tolist()
            if rows["COMPONENT_STRUCTURE"]
            else []
        )
        summary = {
            "status": (
                "STRUCTURE_ONLY"
                if args.structure_only
                else ("INCOMPLETE" if errors else "COMPLETE")
            ),
            "training_only": True,
            "audited_weeks": len(selected),
            "solver_errors_or_missing_incumbents": errors,
            "turnover": turnover_summary,
            "psi_scale": scale,
            "eligibility_crossfit": crossfit,
            "psi_proxy_comparison": proxy_comparison,
            "case_containing_components_k3": {
                "n_week_rosters": len(component_counts),
                "minimum": min(component_counts) if component_counts else None,
                "maximum": max(component_counts) if component_counts else None,
                "interpretation": "One component gives no nontrivial decomposition by eligibility components; other exact methods are not excluded",
            },
            "structure": structure,
            "number_solves": len(rows["SOLVE_RESULTS"]),
            "number_gurobi_optimal": sum(
                r.get("proven_optimal", False) for r in rows["SOLVE_RESULTS"]
            ),
            "max_schedule_identity_error": max(
                (
                    r.get("schedule_identity_error", 0)
                    for r in rows["SOLVE_RESULTS"]
                    if r.get("sol_count", 0)
                ),
                default=None,
            ),
            "final_capacity_proxy_selected": False,
            "planner_replacement_authorized": False,
        }
        write_json(out / "AUDIT_SUMMARY.json", summary)
        report = [
            "# Weekly planner training audit",
            "",
            f"Status: **{summary['status']}**.",
            "",
            f"Frozen cohort: 9,289 cases / 72 training weeks. Audited weeks: {len(selected)}.",
            "No holdout weekly instances, policy training, or recommendation evaluation.",
            "",
            "## Objective scale and solver uncertainty",
            "",
            "The central optimization result is unavailable until dataset solves return finite bounds. Structural checks alone cannot explain the earlier VF certification failure.",
            (
                ""
                if not component_counts
                else f"Primary eligibility has {min(component_counts)}–{max(component_counts)} case-containing components per audited week/roster. A single component provides no useful split into independent eligibility components. It does not rule out other exact solution methods."
            ),
            "",
            "| Roster | Durations | Native objective | Solved/requested | Mean weekly Psi | Mean absolute uncertainty | Max absolute uncertainty |",
            "|---|---|---|---:|---:|---:|---:|",
        ]

        def fmt(x):
            return "unavailable" if x is None else f"{x:,.3f}"

        for row in scale:
            report.append(
                f"| {row['roster']} | {row['durations']} | {row['objective_mode']} | {row['n_incumbents']}/{row['n_requested']} | {fmt(row['mean_weekly_psi'])} | {fmt(row['mean_absolute_uncertainty'])} | {fmt(row['max_absolute_uncertainty'])} |"
            )
        report += [
            "",
            "All three scale columns use weekly objective cost units. The historical epsilon_lib reference is approximately 4,602; it is a reference, not a like-for-like replacement certificate.",
            "Mean Psi uses available incumbents; uncertainty is unavailable if any requested solve lacks a finite bound. Inspect coverage before comparing means.",
            "Psi-shifted and Psi differ only by K in the objective; Phi also changes the formulation. All three use equal budgets, gaps, threads, and seed. Compare native gaps and the shifted run's gap on the Psi scale in PSI_SCALE_AUDIT.csv. A shifted solve may satisfy its native relative tolerance while still having large Psi-relative uncertainty.",
            "The native relative-gap allowance is MIPGap times the absolute native incumbent, not a measured error. An objective shift leaves absolute uncertainty unchanged for the same bounds. Different solve runs need not follow identical search paths. See [Gurobi's gap definitions](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#mipgap).",
            "",
            "## Capacity proxy sensitivity",
            "",
            "K depends on roster capacity, but the roster also changes feasible assignments and potentially Psi. The paired table uses the tightest intervals from the three normal-budget modes on the same weeks. Means are unavailable if any common week lacks finite paired intervals.",
            "",
            "| Roster A minus B | Durations | Bounded/common weeks | Mean K difference | Mean Psi incumbent difference | Mean absolute paired Psi difference | Mean optimal Psi difference interval |",
            "|---|---|---:|---:|---:|---:|---|",
            *[
                f"| {r['roster_a']} − {r['roster_b']} | {r['durations']} | {r['n_paired_finite_intervals']}/{r['n_common_weeks']} | {fmt(r['mean_K_difference_a_minus_b'])} | {fmt(r['mean_psi_ub_difference_a_minus_b'])} | {fmt(r['mean_absolute_paired_psi_ub_difference'])} | [{fmt(r['mean_optimal_psi_difference_lb'])}, {fmt(r['mean_optimal_psi_difference_ub'])}] |"
                for r in proxy_comparison
            ],
            "",
            "Similar mean Psi values would support limited cost sensitivity on the measured duration vectors. They would not establish equal schedules, feasible sets, policy responses, or recommendation values. Signed means can mask week-level differences; inspect the absolute paired differences and uncertainty too.",
            "The observed-activity roster provides full historical coverage by construction, but remains retrospective even if Psi is similar across proxies. Similarity cannot establish exogenous capacity or resolve that identification issue.",
            "",
            "| Roster | Blocks min/mean/max | Mean historical room-day coverage | Complete historical weeks | Fallback at k=3 |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {r['roster']} | {r['blocks_min']}/{r['blocks_mean']:.2f}/{r['blocks_max']} | {100*r['mean_historical_room_day_coverage']:.2f}% | {r['historically_complete_weeks']}/{r['n_weeks']} | {100*r['case_weighted_fallback_rate_k3']:.2f}% |"
                for r in structure
            ],
            "",
            "C_H - C_B is assignment-optimization value. C_O is a realized-duration incumbent with its lower bound and uncertainty, not automatically an exact oracle.",
            "Historical cost is unavailable when any historical room-day is absent; no assignment is remapped. Historical eligibility feasibility is reported separately.",
            "Reassignment is reported as both raw block-ID changes and minimum changes modulo model symmetry. The latter uses maximum-overlap block matching within interchangeable groups; missing historical blocks cannot be matched. It is not evidence that actual days/rooms are clinically interchangeable.",
            "",
            "## Leave-one-week-out compatibility stability",
            "",
            "All 72 training weeks are cross-fitted, including in quick mode. Only compatibility is refitted on the other 71 weeks; rosters and the final all-training eligibility rule remain unchanged. This does not validate rosters or optimized costs out of sample. All rates below are case-weighted.",
            "",
            "| Roster | Threshold | Full/LOO primary coverage | Full/LOO fallback | Fallback delta | Eligible set changed | Historical room compatibility delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {r['roster']} | {r['threshold_weeks']} | {100*r['in_sample_primary_coverage']:.3f}% / {100*r['loo_primary_coverage']:.3f}% | {100*r['in_sample_fallback_rate']:.3f}% / {100*r['loo_fallback_rate']:.3f}% | {100*r['delta_fallback_rate']:.3f} pp | {100*r['fraction_eligible_set_changed']:.3f}% | {100*r['delta_historical_room_compatibility_rate']:.3f} pp |"
                for r in crossfit
            ],
            "",
            f"Turnover calibration: n={turnover_summary['sample_size']}, median={turnover_summary['median']}, Q25={turnover_summary['q25']}, Q75={turnover_summary['q75']} minutes.",
            "Consecutive retained cases can have omitted activity between them. Primary turnover stays 30 minutes.",
            "",
            "No capacity proxy has been selected. Review provenance, coverage, and solver uncertainty before replacing the legacy planner.",
        ]
        if errors:
            report += [
                "",
                "## Unresolved execution failures",
                "",
                *sorted({f"- {e['error']}" for e in errors}),
            ]
        (out / "REPORT.md").write_text("\n".join(report) + "\n")
        LOG.info("Audit status=%s; outputs=%s", summary["status"], out)
        return 2 if errors else 0
    except Exception:
        LOG.exception("Audit aborted")
        write_json(
            out / "AUDIT_SUMMARY.json", {"status": "ABORTED", "training_only": True}
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
