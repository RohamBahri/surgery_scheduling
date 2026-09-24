"""Final pre-freeze hardening for the supported paper experiment.

This module contains only the remaining changes judged material after the
2026-09-24 repository reviews.  It is installed by ``run_final_paper.py`` on top
of the existing final adapter/runtime/scientific layers.

The fixes are intentionally narrow:

* fit the SITE_SHIFT comparator with its *deployed clipped objective* instead of
  passing an infeasible reduced warm start to the pDCA QP;
* refuse to freeze Stage 1 if VF never attempts an outer update, and record an
  explicit VF termination status;
* make deterministic Stage-2 scheduling robust to a too-short emergency wall
  cap while preserving WorkLimit/MIPGap as the actual stopping rule;
* cache BOOKED in response sensitivities because it is response-independent;
* add scenario-specific realized oracles/regret brackets to the two required
  structural sensitivities;
* record that IMPLEMENTABLE_ORACLE is a projected hindsight benchmark, not a
  scheduling-performance ceiling.

The exact Git commit plus ``FINALIZATION_FIXES.json`` is part of the frozen
Stage-1 bundle.  The underlying scientific model is otherwise unchanged.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import final_paper_required_sensitivities as sensitivity
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.config import SolverConfig
from src.solvers.fixed_capacity import schedule_metrics, solve_fixed_capacity_assignment


FINALIZATION_FIXES_VERSION = "final_paper_finalization_2026_09_24_v1"
EMERGENCY_WALL_TO_WORK_MULTIPLIER = 4.0
EMERGENCY_RETRY_MULTIPLIER = 2.0
SITE_SHIFT_COARSE_STEP = 0.10
SITE_SHIFT_FINE_STEP = 0.01
SITE_SHIFT_FINE_RADIUS = 0.25

# Save the unwrapped implementation once.  run_final_paper.py installs the
# wrapper after imports and before the Stage-1 main function is called.
_ORIGINAL_TRAIN_VF = base.train_vf


# ---------------------------------------------------------------------------
# Repository/freeze helpers
# ---------------------------------------------------------------------------

def strict_tracked_tree_is_dirty() -> bool:
    """Fail closed if Git cannot verify the tracked tree."""

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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stamp_training_bundle(root: Path) -> None:
    """Add this hardening layer to the accepted Stage-1 fingerprint set."""

    root = Path(root).resolve()
    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError("Stage 1 returned without TRAINING_FREEZE.json")
    vf_status_path = root / "VF_STATUS.json"
    if not vf_status_path.exists():
        raise RuntimeError("Stage 1 returned without VF_STATUS.json")
    vf_status = _read_json(vf_status_path)
    if not vf_status.get("search_attempted", False):
        raise RuntimeError("Refusing to stamp a Stage-1 bundle with an unattempted VF search")

    note_path = root / "FINALIZATION_FIXES.json"
    _write_json(
        note_path,
        {
            "version": FINALIZATION_FIXES_VERSION,
            "site_shift_fit": (
                "deterministic two-site grid/refinement on the deployed clipped full-weight "
                "case-loss objective; no infeasible pDCA warm start"
            ),
            "vf_freeze_rule": "at least one VF outer iteration must be attempted",
            "deterministic_evaluation": {
                "stopping_rule": "WorkLimit or MIPGap, Threads=1, fixed seed",
                "initial_emergency_wall_to_work_multiplier": EMERGENCY_WALL_TO_WORK_MULTIPLIER,
                "time_limit_retry_multiplier": EMERGENCY_RETRY_MULTIPLIER,
            },
            "implementable_oracle_interpretation": (
                "retrospective case-wise projected-duration benchmark; not a guaranteed "
                "scheduling-performance ceiling or lower bound"
            ),
            "structural_sensitivity_oracle": "scenario-specific realized oracle and regret brackets",
            "response_sensitivity_booked": "primary BOOKED schedule/cost reused; no redundant re-solves",
        },
    )

    freeze = _read_json(freeze_path)
    freeze["finalization_fixes_version"] = FINALIZATION_FIXES_VERSION
    fps = dict(freeze.get("artifact_fingerprints", {}))
    fps[note_path.name] = base.sha256_file(note_path)
    fps[vf_status_path.name] = base.sha256_file(vf_status_path)
    freeze["artifact_fingerprints"] = fps
    _write_json(freeze_path, freeze)

    status_path = root / "RUN_STATUS.json"
    if status_path.exists():
        status = _read_json(status_path)
        status["finalization_fixes_version"] = FINALIZATION_FIXES_VERSION
        _write_json(status_path, status)


def verify_training_finalization(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    freeze = _read_json(root / "TRAINING_FREEZE.json")
    if freeze.get("finalization_fixes_version") != FINALIZATION_FIXES_VERSION:
        raise RuntimeError(
            "Training bundle predates the final review hardening. Re-run Stage 1 with run_final_paper.py."
        )
    note = root / "FINALIZATION_FIXES.json"
    expected = dict(freeze.get("artifact_fingerprints", {})).get(note.name)
    if not note.exists() or not expected or base.sha256_file(note) != expected:
        raise RuntimeError("Frozen FINALIZATION_FIXES.json is missing or changed")
    return freeze


# ---------------------------------------------------------------------------
# SITE_SHIFT: deployment-consistent two-dimensional comparator
# ---------------------------------------------------------------------------

def _constant_site_loss(
    grid: np.ndarray,
    *,
    booked: np.ndarray,
    error: np.ndarray,
    n_weeks: int,
    s,
) -> np.ndarray:
    """Full-weight case loss for a constant raw display shift on one site.

    The deployed policy clips case by case to the one-minute recommendation
    floor.  This is exactly the behavior we evaluate here, so 14-minute cases do
    not impose their raw-score floor on every other case at the site.
    """

    lower = np.maximum(-float(s.display_cap), final.MIN_RECOMMENDED_DURATION - booked)
    out = np.empty(len(grid), dtype=float)
    for k, raw_shift in enumerate(grid):
        delta = np.maximum(np.minimum(float(raw_shift), float(s.display_cap)), lower)
        corr = base.response_value(delta, s)
        out[k] = float(
            np.sum(base.theta(error - corr, float(s.overtime), float(s.idle))) / n_weeks
        )
    return out


def _solve_site_shift_grid(reduced: base.Arrays, s, lam: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    xsite = np.asarray(reduced.X[:, 1].toarray(), dtype=float).reshape(-1)
    mask0 = xsite < 0.5
    mask1 = ~mask0
    if not mask0.any() or not mask1.any():
        raise RuntimeError("SITE_SHIFT requires observations from both sites")

    cap = float(s.display_cap)

    def solve_on_grids(g0: np.ndarray, g1: np.ndarray, stage: str):
        l0 = _constant_site_loss(
            g0,
            booked=reduced.booked[mask0],
            error=reduced.error[mask0],
            n_weeks=reduced.n_weeks,
            s=s,
        )
        l1 = _constant_site_loss(
            g1,
            booked=reduced.booked[mask1],
            error=reduced.error[mask1],
            n_weeks=reduced.n_weeks,
            s=s,
        )
        # Parameterization is bias=d0, site coefficient=d1-d0.  Intercept is
        # unpenalized, hence the common L1 contributes lam*|d1-d0|.
        obj = l0[:, None] + l1[None, :] + float(lam) * np.abs(g1[None, :] - g0[:, None])
        ii, jj = np.unravel_index(int(np.argmin(obj)), obj.shape)
        d0, d1 = float(g0[ii]), float(g1[jj])
        w = np.array([d0, d1 - d0], dtype=float)
        return w, {
            "stage": stage,
            "fit_method": "deterministic_deployment_clipped_grid",
            "grid0_points": int(len(g0)),
            "grid1_points": int(len(g1)),
            "reference_site_raw_shift": d0,
            "dummy_site_raw_shift": d1,
            "site_coefficient": float(w[1]),
            "regularized_objective": float(obj[ii, jj]),
            "accepted": True,
        }

    coarse = np.arange(-cap, cap + 0.5 * SITE_SHIFT_COARSE_STEP, SITE_SHIFT_COARSE_STEP)
    w0, row0 = solve_on_grids(coarse, coarse, "coarse")
    d0, d1 = float(w0[0]), float(w0[0] + w0[1])
    g0 = np.arange(
        max(-cap, d0 - SITE_SHIFT_FINE_RADIUS),
        min(cap, d0 + SITE_SHIFT_FINE_RADIUS) + 0.5 * SITE_SHIFT_FINE_STEP,
        SITE_SHIFT_FINE_STEP,
    )
    g1 = np.arange(
        max(-cap, d1 - SITE_SHIFT_FINE_RADIUS),
        min(cap, d1 + SITE_SHIFT_FINE_RADIUS) + 0.5 * SITE_SHIFT_FINE_STEP,
        SITE_SHIFT_FINE_STEP,
    )
    w1, row1 = solve_on_grids(g0, g1, "refine")

    if np.any(np.abs(w1) > float(s.coefficient_bound) + 1e-9):
        raise AssertionError("SITE_SHIFT refined coefficients exceed the frozen coefficient box")

    raw = np.asarray(reduced.X @ w1, dtype=float).reshape(-1)
    lower = np.maximum(-float(s.display_cap), final.MIN_RECOMMENDED_DURATION - reduced.booked)
    _, _, planning = base.correction_and_planning(w1, reduced, s)
    if np.any(planning <= 0) or np.any(~np.isfinite(planning)):
        raise AssertionError("SITE_SHIFT deployment clipping produced invalid planning durations")
    row1["training_cases_clipped_by_duration_floor"] = int(np.sum(raw < lower - 1e-9))
    row1["training_cases_clipped_by_display_cap"] = int(np.sum(raw > float(s.display_cap) + 1e-9))
    row1["deployment_clipping_is_part_of_fit_objective"] = True
    row1["unregularized_case_loss"] = float(base.case_envelope(w1, reduced, s))
    row1["l1_penalty"] = float(lam) * abs(float(w1[1]))
    return w1, [row0, row1]


def fit_site_shift_policy(train_a, enc, s, lam_common, initial_full):
    """Fit the two-parameter site-shift comparator without a QP-domain mismatch."""

    del initial_full  # SITE_SHIFT is fit independently; no full-policy warm start.
    reduced, site_idx = science.site_shift_reduced_arrays(train_a, enc.feature_names)
    w_small, hist = _solve_site_shift_grid(reduced, s, float(lam_common))
    hist[-1]["site_dummy_feature"] = str(enc.feature_names[site_idx])
    expanded = science.expand_site_shift_policy(w_small, train_a.p, site_idx)
    # Final deployment-level validity check on the full training arrays.
    _, _, planning = base.correction_and_planning(expanded, train_a, s)
    if np.any(planning <= 0) or np.any(~np.isfinite(planning)):
        raise AssertionError("Expanded SITE_SHIFT policy is not deployment-feasible")
    return expanded, hist


# ---------------------------------------------------------------------------
# VF: distinguish attempted search from budget exhaustion
# ---------------------------------------------------------------------------

def guarded_train_vf(*args, **kwargs):
    root = Path(args[7] if len(args) > 7 else kwargs["root"])
    s = args[5] if len(args) > 5 else kwargs["s"]
    start_w = np.asarray(args[0] if args else kwargs["start_w"], dtype=float)
    search_deadline = kwargs.get("search_deadline")
    t0 = time.monotonic()
    w, traj = _ORIGINAL_TRAIN_VF(*args, **kwargs)

    if not traj:
        payload = {
            "status": "VF_NOT_ATTEMPTED_BUDGET_EXHAUSTED",
            "search_attempted": False,
            "attempted_outer_iterations": 0,
            "same_as_start": bool(np.allclose(w, start_w, atol=1e-10, rtol=0.0)),
            "elapsed_seconds": time.monotonic() - t0,
            "remaining_search_seconds": (
                None if search_deadline is None else float(base.remaining(search_deadline))
            ),
        }
        base.write_json(root / "VF_STATUS.json", payload)
        raise RuntimeError(
            "VF anchor planning exhausted the search reserve before outer iteration 1. "
            "Stage 1 is incomplete; increase the wall budget or reduce earlier training cost."
        )

    if len(traj) >= int(s.vf_outer):
        reason = "MAX_OUTER_ITERATIONS"
    elif search_deadline is not None and base.remaining(search_deadline) < 120:
        reason = "BUDGET_GUARD_AFTER_ATTEMPTED_SEARCH"
    else:
        reason = "STAGNATION_OR_STOP_RULE"

    payload = {
        "status": "VF_SEARCH_ATTEMPTED",
        "search_attempted": True,
        "attempted_outer_iterations": int(len(traj)),
        "accepted_outer_iterations": int(sum(bool(r.get("accepted", False)) for r in traj)),
        "termination_reason": reason,
        "same_as_start": bool(np.allclose(w, start_w, atol=1e-10, rtol=0.0)),
        "elapsed_seconds": time.monotonic() - t0,
        "remaining_search_seconds": (
            None if search_deadline is None else float(base.remaining(search_deadline))
        ),
    }
    base.write_json(root / "VF_STATUS.json", payload)
    return w, traj


# ---------------------------------------------------------------------------
# Deterministic Stage-2 scheduling: WorkLimit is primary; TimeLimit is fail-safe
# ---------------------------------------------------------------------------

def _fixed_site_once(
    week,
    durations,
    s,
    site: str,
    *,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    seed: int,
    turnover: float,
):
    view = final._site_view(week, site)
    local_d = np.asarray(durations, dtype=float)[view.global_indices]
    cfg = SolverConfig(
        time_limit_seconds=max(1, int(wall_seconds)),
        work_limit=float(work_limit),
        mip_gap=max(0.0, float(mip_gap)),
        threads=1,
        verbose=bool(s.verbose),
        mip_gap_abs=1e-10,
        seed=int(seed),
    )
    result = solve_fixed_capacity_assignment(
        view.instance,
        local_d,
        final.final_cost_cfg(s),
        float(turnover),
        cfg,
        objective_mode=final.PRIMARY_OBJECTIVE,
        symmetry_breaking=True,
    )
    if result.column is None or any(
        x is None for x in (result.phi_ub, result.phi_lb, result.psi_ub, result.psi_lb)
    ):
        raise RuntimeError(
            f"Week {week.position} site {site}: deterministic planner returned no incumbent/bound "
            f"({result.diagnostics.status})"
        )
    return view, result


def robust_deterministic_site_solve(
    week,
    durations,
    s,
    site: str,
    *,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    seed: int,
):
    initial_wall = max(
        int(wall_seconds),
        int(math.ceil(EMERGENCY_WALL_TO_WORK_MULTIPLIER * float(work_limit))),
    )
    view, result = _fixed_site_once(
        week,
        durations,
        s,
        site,
        work_limit=work_limit,
        wall_seconds=initial_wall,
        mip_gap=mip_gap,
        seed=seed,
        turnover=final.PRIMARY_TURNOVER,
    )
    if result.diagnostics.status != "TIME_LIMIT":
        return view, result

    retry_wall = max(
        initial_wall + 1,
        int(math.ceil(initial_wall * EMERGENCY_RETRY_MULTIPLIER)),
    )
    base.LOG.warning(
        "[DET-PLANNER] week=%s site=%s emergency wall cap %ss fired; retrying from scratch "
        "with %ss while preserving WorkLimit=%.3g",
        week.position,
        site,
        initial_wall,
        retry_wall,
        work_limit,
    )
    view, result = _fixed_site_once(
        week,
        durations,
        s,
        site,
        work_limit=work_limit,
        wall_seconds=retry_wall,
        mip_gap=mip_gap,
        seed=seed,
        turnover=final.PRIMARY_TURNOVER,
    )
    if result.diagnostics.status == "TIME_LIMIT":
        raise RuntimeError(
            f"Week {week.position} site {site}: emergency TimeLimit fired twice before the "
            "deterministic WorkLimit/MIPGap rule. Recalibrate the wall cap on this machine."
        )
    return view, result


def reviewed_evaluation_settings_loader(original_loader):
    def load(training_root, eval_root, args):
        s = original_loader(training_root, eval_root, args)
        # Wall time is only a safety guard.  It is deliberately several times
        # the deterministic work budget and therefore does not define the
        # selected incumbent under normal execution.
        s.final_planner_seconds = max(
            int(s.final_planner_seconds),
            int(math.ceil(EMERGENCY_WALL_TO_WORK_MULTIPLIER * s.final_planner_work_limit)),
        )
        return s

    return load


# ---------------------------------------------------------------------------
# Response sensitivities: BOOKED is response-independent, so reuse it
# ---------------------------------------------------------------------------

def efficient_response_sensitivities(eval_module, weeks, a, policies, oracle_plans, s, root: Path) -> None:
    rows: list[dict[str, Any]] = []
    oracle_lb = {w: r.bound for w, r in oracle_plans.items()}
    primary = pd.read_csv(Path(root) / "FINAL_HOLDOUT_WEEKLY.csv")
    booked_primary = primary[primary["method"] == "BOOKED"].copy()
    if len(booked_primary) != a.n_weeks:
        raise RuntimeError("Cannot reuse BOOKED response sensitivity: primary BOOKED rows missing")

    for scenario, (alpha, h) in science.RESPONSE_SENSITIVITY_SCENARIOS.items():
        # BOOKED planning durations and realized cost do not depend on alpha/h.
        for _, r in booked_primary.iterrows():
            wk = int(r["week"])
            rc = float(r["realized_cost"])
            rows.append(
                {
                    "scenario": scenario,
                    "alpha": float(alpha),
                    "h": float(h),
                    "method": "BOOKED",
                    "week": wk,
                    "realized_cost": rc,
                    "planning_gap_native_psi": float(r["planning_gap_native_psi"]),
                    "planning_status": str(r["planning_status"]),
                    "regret_upper": max(0.0, rc - float(oracle_lb[wk])),
                    "booked_result_reused": True,
                }
            )

        ss = __import__("copy").copy(s)
        ss.alpha = float(alpha)
        ss.h = float(h)
        for name, w in policies.items():
            if name == "BOOKED":
                continue
            dm, _ = eval_module._policy_meta(name, w, a, ss)
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
                        "alpha": float(alpha),
                        "h": float(h),
                        "method": name,
                        "week": wk,
                        "realized_cost": rc,
                        "planning_gap_native_psi": float(plan.gap),
                        "planning_status": str(plan.status),
                        "regret_upper": max(0.0, rc - float(oracle_lb[wk])),
                        "booked_result_reused": False,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(Path(root) / "RESPONSE_SENSITIVITY_WEEKLY.csv", index=False)
    if len(frame):
        (
            frame.groupby(["scenario", "alpha", "h", "method"], as_index=False)
            .agg(
                avg_realized_cost=("realized_cost", "mean"),
                avg_regret_upper=("regret_upper", "mean"),
                max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
            )
            .to_csv(Path(root) / "RESPONSE_SENSITIVITY_SUMMARY.csv", index=False)
        )
    base.write_json(
        Path(root) / "RESPONSE_SENSITIVITY_NOTE.json",
        {
            "booked": "reused from the primary evaluation because BOOKED is response-independent",
            "projected_hindsight_benchmark": (
                "not rerun here; it changes with alpha/h and is not a performance ceiling. "
                "These scenarios are robustness checks for the frozen deployable policies."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Structural sensitivities: add scenario-specific oracle/regret brackets
# ---------------------------------------------------------------------------

def robust_sensitivity_worker(
    week,
    durations,
    s,
    *,
    turnover: float,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    label: str,
):
    d = np.asarray(durations, dtype=float)
    t0 = time.perf_counter()
    parts: list[tuple[Any, ScheduleColumn]] = []
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_optimal = True
    initial_wall = max(
        int(wall_seconds),
        int(math.ceil(EMERGENCY_WALL_TO_WORK_MULTIPLIER * float(work_limit))),
    )

    for site in final.PRIMARY_SITES:
        view, result = _fixed_site_once(
            week,
            d,
            s,
            site,
            work_limit=work_limit,
            wall_seconds=initial_wall,
            mip_gap=mip_gap,
            seed=int(s.random_seed),
            turnover=float(turnover),
        )
        if result.diagnostics.status == "TIME_LIMIT":
            retry_wall = int(math.ceil(initial_wall * EMERGENCY_RETRY_MULTIPLIER))
            base.LOG.warning(
                "[SENS] %s week=%s site=%s wall cap fired; retrying with %ss",
                label,
                week.position,
                site,
                retry_wall,
            )
            view, result = _fixed_site_once(
                week,
                d,
                s,
                site,
                work_limit=work_limit,
                wall_seconds=retry_wall,
                mip_gap=mip_gap,
                seed=int(s.random_seed),
                turnover=float(turnover),
            )
            if result.diagnostics.status == "TIME_LIMIT":
                raise RuntimeError(
                    f"Sensitivity {label}, week {week.position}, site {site}: wall cap fired twice"
                )
        parts.append((view, result.column))
        phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    metrics = schedule_metrics(column, d, final.final_cost_cfg(s), float(turnover))
    if abs(float(metrics["phi"]) - phi_ub) > 1e-5:
        raise AssertionError("Sensitivity decomposed planner Phi accounting mismatch")
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=float(phi_ub),
        bound=float(phi_lb),
        gap=float(base.rel_gap(psi_ub, psi_lb)),
        status="OPTIMAL" if all_optimal else "|".join(statuses),
        solve_seconds=time.perf_counter() - t0,
        exact=bool(all_optimal and abs(psi_ub - psi_lb) <= 1e-6),
        tiebreak_used=False,
    )


def _structural_scenario_rows(
    scenario: str,
    weeks: Sequence[base.WeekBundle],
    arrays: base.Arrays,
    policies: Mapping[str, np.ndarray | None],
    s,
    *,
    roster_name: str,
    turnover: float,
) -> list[dict[str, Any]]:
    actual_map = {wk: arrays.actual[idx] for wk, idx in arrays.week_slices.items()}
    oracle = sensitivity.solve_batch(
        weeks,
        actual_map,
        s,
        turnover=float(turnover),
        label=f"{scenario}_ORACLE",
    )

    selected = {name: policies[name] for name in sensitivity.SENSITIVITY_METHODS}
    method_plans: dict[str, dict[int, base.PlanResult]] = {}
    realized: dict[str, dict[int, float]] = {}
    for name, w in selected.items():
        dm, _ = base.policy_duration_map(name, w, arrays, s)
        plans = sensitivity.solve_batch(
            weeks,
            dm,
            s,
            turnover=float(turnover),
            label=f"{scenario}_{name}",
        )
        method_plans[name] = plans
        realized[name] = {}
        for wk, idx in arrays.week_slices.items():
            realized[name][wk] = float(
                plans[wk].column.compute_cost(
                    arrays.actual[idx], final.final_cost_cfg(s), float(turnover)
                )
            )

    effective_ub = {
        wk: min(
            [float(oracle[wk].objective)]
            + [float(realized[name][wk]) for name in selected]
        )
        for wk in arrays.week_slices
    }
    oracle_lb = {wk: float(oracle[wk].bound) for wk in arrays.week_slices}

    rows: list[dict[str, Any]] = []
    for wk in arrays.week_slices:
        op = oracle[wk]
        rc = float(op.objective)
        rows.append(
            {
                "scenario": scenario,
                "roster": roster_name,
                "turnover_minutes": float(turnover),
                "method": "ORACLE",
                "week": wk,
                "realized_cost": rc,
                "planning_gap_native_psi": float(op.gap),
                "planning_status": str(op.status),
                "planning_seconds": float(op.solve_seconds),
                "oracle_lb_phi": oracle_lb[wk],
                "oracle_ub_effective_phi": effective_ub[wk],
                "regret_lower": max(0.0, rc - effective_ub[wk]),
                "regret_upper": max(0.0, rc - oracle_lb[wk]),
                "frozen_policy_environment_robustness": False,
            }
        )

    for name in selected:
        for wk in arrays.week_slices:
            p = method_plans[name][wk]
            rc = realized[name][wk]
            rows.append(
                {
                    "scenario": scenario,
                    "roster": roster_name,
                    "turnover_minutes": float(turnover),
                    "method": name,
                    "week": wk,
                    "realized_cost": rc,
                    "planning_gap_native_psi": float(p.gap),
                    "planning_status": str(p.status),
                    "planning_seconds": float(p.solve_seconds),
                    "oracle_lb_phi": oracle_lb[wk],
                    "oracle_ub_effective_phi": effective_ub[wk],
                    "regret_lower": max(0.0, rc - effective_ub[wk]),
                    "regret_upper": max(0.0, rc - oracle_lb[wk]),
                    "frozen_policy_environment_robustness": True,
                }
            )
    return rows


def run_required_sensitivities_with_oracle(
    *,
    scoped: pd.DataFrame,
    hold_starts,
    cfg,
    context,
    primary_weeks,
    primary_arrays,
    encoder,
    policies,
    s,
    root,
):
    rows: list[dict[str, Any]] = []
    regular_weeks = sensitivity.build_bundles_with_roster(
        scoped,
        hold_starts,
        cfg=cfg,
        context=context,
        roster_name=science.REQUIRED_CAPACITY_SENSITIVITY,
        offset=s.train_weeks,
    )
    regular_arrays = base.build_arrays(regular_weeks, encoder)
    if not np.array_equal(regular_arrays.case_ids, primary_arrays.case_ids):
        raise AssertionError("Capacity sensitivity changed holdout case identity/order")

    rows.extend(
        _structural_scenario_rows(
            "regular_template_capacity",
            regular_weeks,
            regular_arrays,
            policies,
            s,
            roster_name=science.REQUIRED_CAPACITY_SENSITIVITY,
            turnover=final.PRIMARY_TURNOVER,
        )
    )
    rows.extend(
        _structural_scenario_rows(
            "zero_turnover",
            primary_weeks,
            primary_arrays,
            policies,
            s,
            roster_name=science.PRIMARY_ROSTER,
            turnover=science.REQUIRED_TURNOVER_SENSITIVITY_MINUTES,
        )
    )

    weekly = pd.DataFrame(rows)
    summary = (
        weekly.groupby(["scenario", "roster", "turnover_minutes", "method"], as_index=False)
        .agg(
            avg_realized_cost=("realized_cost", "mean"),
            avg_regret_lower=("regret_lower", "mean"),
            avg_regret_upper=("regret_upper", "mean"),
            max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
            mean_planning_seconds=("planning_seconds", "mean"),
        )
    )
    for scenario_name in summary["scenario"].unique():
        mask = summary["scenario"] == scenario_name
        booked = summary[mask & (summary["method"] == "BOOKED")].iloc[0]
        blo = max(1e-9, float(booked["avg_regret_lower"]))
        bhi = max(1e-9, float(booked["avg_regret_upper"]))
        summary.loc[mask, "gap_closed_lower_pct"] = [
            100.0 * (1.0 - float(v) / blo)
            for v in summary.loc[mask, "avg_regret_upper"]
        ]
        summary.loc[mask, "gap_closed_upper_pct"] = [
            100.0 * (1.0 - float(v) / bhi)
            for v in summary.loc[mask, "avg_regret_lower"]
        ]

    root = Path(root)
    weekly.to_csv(root / "REQUIRED_SENSITIVITY_WEEKLY.csv", index=False)
    summary.to_csv(root / "REQUIRED_SENSITIVITY_SUMMARY.csv", index=False)
    base.write_json(
        root / "REQUIRED_SENSITIVITY_INTERPRETATION.json",
        {
            "policy_training": "all deployable policies are frozen from the primary training specification",
            "meaning": "deployment robustness under changed capacity/turnover, not retrained alternative-specification performance",
            "oracle": "scenario-specific realized-duration oracle solved at the same sensitivity planning budget",
        },
    )
    return weekly, summary


# ---------------------------------------------------------------------------
# Installation hooks used by run_final_paper.py
# ---------------------------------------------------------------------------

def install_training_fixes(training_module) -> None:
    training_module._site_shift_policy = fit_site_shift_policy
    training_module._tracked_tree_is_dirty = strict_tracked_tree_is_dirty
    base.train_vf = guarded_train_vf


def install_evaluation_fixes(evaluation_module) -> None:
    science._deterministic_site_solve = robust_deterministic_site_solve
    evaluation_module._load_settings = reviewed_evaluation_settings_loader(
        evaluation_module._load_settings
    )
    evaluation_module._run_response_sensitivities = (
        lambda weeks, a, policies, oracle_plans, s, root: efficient_response_sensitivities(
            evaluation_module, weeks, a, policies, oracle_plans, s, root
        )
    )


def install_sensitivity_fixes(sensitivity_runner_module) -> None:
    del sensitivity_runner_module  # runner calls the helper module below.
    sensitivity._worker = robust_sensitivity_worker
    sensitivity.run_required_sensitivities = run_required_sensitivities_with_oracle


def write_benchmark_interpretation(eval_root: Path) -> None:
    base.write_json(
        Path(eval_root) / "PROJECTED_HINDSIGHT_BENCHMARK.json",
        {
            "method": "IMPLEMENTABLE_ORACLE",
            "preferred_label": "projected hindsight benchmark",
            "definition": "case-wise realized correction projected onto the implementable correction interval",
            "not_a_claim": [
                "not a scheduling-performance ceiling",
                "not a lower bound on achievable realized cost",
                "not a guaranteed nonnegative behavioral-loss decomposition",
            ],
        },
    )
