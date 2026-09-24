"""Last-mile release guard for the reviewed final-paper pipeline.

This layer closes defects that only appear when the wrapper combines the
training, evaluation, and sensitivity modules:

* install every numerical-accounting worker in every reviewed stage, so the
  late Stage-1 seed audit and Stage-2 holdout oracle cannot fall back to the
  legacy absolute 1e-5 accounting check;
* tighten Gurobi IntFeasTol for the final fixed-capacity planner;
* enforce the SITE_SHIFT coefficient box during grid optimization, rather than
  only checking it after an unconstrained minimization;
* make structural gap-closed percentages undefined when BOOKED has zero regret;
* solve BOOKED once at the response-sensitivity planner budget and reuse that
  result across response scenarios, avoiding mixed-budget comparisons.

Only ``run_final_paper.py`` should install this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric
import final_paper_required_sensitivities as sensitivity
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
import src.solvers.deterministic as deterministic_solver


RELEASE_GUARD_VERSION = "final_paper_release_guard_2026_09_24_v1"
INT_FEAS_TOL = 1e-9
ZERO_REGRET_TOL = 1e-8

_ORIGINAL_APPLY_SOLVER_PARAMS = deterministic_solver._apply_solver_params
_ORIGINAL_REQUIRED_SENSITIVITIES = hardening.run_required_sensitivities_with_oracle
_SOLVER_PARAMS_INSTALLED = False


def _reviewed_apply_solver_params(model, cfg) -> None:
    """Apply normal parameters, then tighten binary/integer feasibility."""

    _ORIGINAL_APPLY_SOLVER_PARAMS(model, cfg)
    model.Params.IntFeasTol = float(INT_FEAS_TOL)


def _install_solver_numeric_params() -> None:
    global _SOLVER_PARAMS_INSTALLED
    if _SOLVER_PARAMS_INSTALLED:
        return
    deterministic_solver._apply_solver_params = _reviewed_apply_solver_params
    _SOLVER_PARAMS_INSTALLED = True


def runtime_worker(*args, **kwargs):
    """Spawn-safe runtime worker that also tightens IntFeasTol in the child."""

    _install_solver_numeric_params()
    return numeric.training_process_task(*args, **kwargs)


def deterministic_worker(*args, **kwargs):
    """Spawn-safe deterministic worker that installs child-process solver params."""

    _install_solver_numeric_params()
    return numeric.deterministic_eval_worker(*args, **kwargs)


def sensitivity_worker(*args, **kwargs):
    """Spawn-safe sensitivity worker that installs child-process solver params."""

    _install_solver_numeric_params()
    return numeric.sensitivity_worker(*args, **kwargs)


def constrained_site_shift_grid(
    reduced: base.Arrays, s, lam: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Solve SITE_SHIFT over the actual two-coefficient feasible set."""

    xsite = np.asarray(reduced.X[:, 1].toarray(), dtype=float).reshape(-1)
    mask0 = xsite < 0.5
    mask1 = ~mask0
    if not mask0.any() or not mask1.any():
        raise RuntimeError("SITE_SHIFT requires observations from both sites")

    cap = float(s.display_cap)
    coef_bound = float(s.coefficient_bound)

    def solve_on_grids(g0: np.ndarray, g1: np.ndarray, stage: str):
        l0 = hardening._constant_site_loss(
            g0,
            booked=reduced.booked[mask0],
            error=reduced.error[mask0],
            n_weeks=reduced.n_weeks,
            s=s,
        )
        l1 = hardening._constant_site_loss(
            g1,
            booked=reduced.booked[mask1],
            error=reduced.error[mask1],
            n_weeks=reduced.n_weeks,
            s=s,
        )

        # Parameterization is w=[d0, d1-d0].  The frozen coefficient box is on
        # w itself, not on the two site-level raw shifts independently.
        d0_grid = g0[:, None]
        d1_grid = g1[None, :]
        valid = (
            (np.abs(d0_grid) <= coef_bound + 1e-12)
            & (np.abs(d1_grid - d0_grid) <= coef_bound + 1e-12)
        )
        obj = (
            l0[:, None]
            + l1[None, :]
            + float(lam) * np.abs(d1_grid - d0_grid)
        )
        obj = np.where(valid, obj, np.inf)
        if not np.isfinite(obj).any():
            raise RuntimeError(
                f"SITE_SHIFT {stage} grid contains no coefficient-feasible pair"
            )

        ii, jj = np.unravel_index(int(np.argmin(obj)), obj.shape)
        d0, d1 = float(g0[ii]), float(g1[jj])
        w = np.array([d0, d1 - d0], dtype=float)
        if np.any(np.abs(w) > coef_bound + 1e-9):
            raise AssertionError("SITE_SHIFT grid returned an infeasible coefficient")
        return w, {
            "stage": stage,
            "fit_method": "deterministic_deployment_clipped_grid_coefficient_feasible",
            "grid0_points": int(len(g0)),
            "grid1_points": int(len(g1)),
            "coefficient_bound": coef_bound,
            "reference_site_raw_shift": d0,
            "dummy_site_raw_shift": d1,
            "site_coefficient": float(w[1]),
            "regularized_objective": float(obj[ii, jj]),
            "accepted": True,
        }

    coarse = np.arange(
        -cap,
        cap + 0.5 * hardening.SITE_SHIFT_COARSE_STEP,
        hardening.SITE_SHIFT_COARSE_STEP,
    )
    w0, row0 = solve_on_grids(coarse, coarse, "coarse")
    d0, d1 = float(w0[0]), float(w0[0] + w0[1])

    g0 = np.arange(
        max(-cap, d0 - hardening.SITE_SHIFT_FINE_RADIUS),
        min(cap, d0 + hardening.SITE_SHIFT_FINE_RADIUS)
        + 0.5 * hardening.SITE_SHIFT_FINE_STEP,
        hardening.SITE_SHIFT_FINE_STEP,
    )
    g1 = np.arange(
        max(-cap, d1 - hardening.SITE_SHIFT_FINE_RADIUS),
        min(cap, d1 + hardening.SITE_SHIFT_FINE_RADIUS)
        + 0.5 * hardening.SITE_SHIFT_FINE_STEP,
        hardening.SITE_SHIFT_FINE_STEP,
    )
    w1, row1 = solve_on_grids(g0, g1, "refine")

    raw = np.asarray(reduced.X @ w1, dtype=float).reshape(-1)
    lower = np.maximum(
        -float(s.display_cap),
        final.MIN_RECOMMENDED_DURATION - reduced.booked,
    )
    _, _, planning = base.correction_and_planning(w1, reduced, s)
    if np.any(planning <= 0) or np.any(~np.isfinite(planning)):
        raise AssertionError("SITE_SHIFT deployment clipping produced invalid planning durations")

    row1["training_cases_clipped_by_duration_floor"] = int(
        np.sum(raw < lower - 1e-9)
    )
    row1["training_cases_clipped_by_display_cap"] = int(
        np.sum(raw > float(s.display_cap) + 1e-9)
    )
    row1["deployment_clipping_is_part_of_fit_objective"] = True
    row1["unregularized_case_loss"] = float(base.case_envelope(w1, reduced, s))
    row1["l1_penalty"] = float(lam) * abs(float(w1[1]))
    return w1, [row0, row1]


def _safe_gap_closed_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Use NaN, not an epsilon denominator, when BOOKED regret is zero."""

    out = summary.copy()
    out["gap_closed_lower_pct"] = np.nan
    out["gap_closed_upper_pct"] = np.nan
    out["gap_closed_lower_status"] = ""
    out["gap_closed_upper_status"] = ""

    for scenario_name in out["scenario"].unique():
        mask = out["scenario"] == scenario_name
        booked_rows = out[mask & (out["method"] == "BOOKED")]
        if len(booked_rows) != 1:
            raise RuntimeError(
                f"Expected one BOOKED summary row for structural scenario {scenario_name}"
            )
        booked = booked_rows.iloc[0]
        blo = float(booked["avg_regret_lower"])
        bhi = float(booked["avg_regret_upper"])

        if blo > ZERO_REGRET_TOL:
            out.loc[mask, "gap_closed_lower_pct"] = [
                100.0 * (1.0 - float(v) / blo)
                for v in out.loc[mask, "avg_regret_upper"]
            ]
            out.loc[mask, "gap_closed_lower_status"] = "DEFINED"
        else:
            out.loc[mask, "gap_closed_lower_status"] = "UNDEFINED_ZERO_BOOKED_REGRET"

        if bhi > ZERO_REGRET_TOL:
            out.loc[mask, "gap_closed_upper_pct"] = [
                100.0 * (1.0 - float(v) / bhi)
                for v in out.loc[mask, "avg_regret_lower"]
            ]
            out.loc[mask, "gap_closed_upper_status"] = "DEFINED"
        else:
            out.loc[mask, "gap_closed_upper_status"] = "UNDEFINED_ZERO_BOOKED_REGRET"

    return out


def required_sensitivities_zero_safe(**kwargs):
    """Run the reviewed sensitivities, then correct zero-denominator summaries."""

    weekly, summary = _ORIGINAL_REQUIRED_SENSITIVITIES(**kwargs)
    summary = _safe_gap_closed_columns(summary)
    root = Path(kwargs["root"])
    summary.to_csv(root / "REQUIRED_SENSITIVITY_SUMMARY.csv", index=False)

    note_path = root / "REQUIRED_SENSITIVITY_INTERPRETATION.json"
    note = {}
    if note_path.exists():
        note = json.loads(note_path.read_text(encoding="utf-8"))
    note["gap_closed_zero_baseline_rule"] = (
        "If the relevant BOOKED regret denominator is <=1e-8, gap-closed is "
        "undefined (NaN), not 100%."
    )
    base.write_json(note_path, note)
    return weekly, summary


def fair_response_sensitivities(
    eval_module,
    weeks,
    a,
    policies,
    oracle_plans,
    s,
    root: Path,
) -> None:
    """Use one sensitivity-budget BOOKED solve for all response scenarios."""

    rows: list[dict[str, Any]] = []
    oracle_lb = {w: r.bound for w, r in oracle_plans.items()}
    sens_gap = max(float(s.final_planner_gap), 0.002)

    booked_dm = {
        wk: np.asarray(a.booked[idx], dtype=float)
        for wk, idx in a.week_slices.items()
    }
    booked_plans = science.deterministic_eval_solve_batch(
        weeks,
        booked_dm,
        s,
        work_limit=float(s.sensitivity_planner_work_limit),
        wall_seconds=int(s.final_planner_seconds),
        gap=sens_gap,
        label="response_sensitivity_BOOKED_once",
    )
    booked_rows: dict[int, dict[str, Any]] = {}
    for wk, idx in a.week_slices.items():
        plan = booked_plans[wk]
        rc = float(
            plan.column.compute_cost(
                a.actual[idx],
                final.final_cost_cfg(s),
                final.PRIMARY_TURNOVER,
            )
        )
        booked_rows[wk] = {
            "week": wk,
            "realized_cost": rc,
            "planning_gap_native_psi": float(plan.gap),
            "planning_status": str(plan.status),
            "regret_upper": max(0.0, rc - float(oracle_lb[wk])),
        }

    for scenario, (alpha, h) in science.RESPONSE_SENSITIVITY_SCENARIOS.items():
        for wk, r in booked_rows.items():
            rows.append(
                {
                    "scenario": scenario,
                    "alpha": float(alpha),
                    "h": float(h),
                    "method": "BOOKED",
                    **r,
                    "booked_result_reused": True,
                    "planner_work_limit": float(s.sensitivity_planner_work_limit),
                    "planner_gap_target": sens_gap,
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
                work_limit=float(s.sensitivity_planner_work_limit),
                wall_seconds=int(s.final_planner_seconds),
                gap=sens_gap,
                label=f"response_{scenario}_{name}",
            )
            for wk, idx in a.week_slices.items():
                plan = plans[wk]
                rc = float(
                    plan.column.compute_cost(
                        a.actual[idx],
                        final.final_cost_cfg(ss),
                        final.PRIMARY_TURNOVER,
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
                        "planner_work_limit": float(s.sensitivity_planner_work_limit),
                        "planner_gap_target": sens_gap,
                    }
                )

    frame = pd.DataFrame(rows)
    root = Path(root)
    frame.to_csv(root / "RESPONSE_SENSITIVITY_WEEKLY.csv", index=False)
    if len(frame):
        (
            frame.groupby(["scenario", "alpha", "h", "method"], as_index=False)
            .agg(
                avg_realized_cost=("realized_cost", "mean"),
                avg_regret_upper=("regret_upper", "mean"),
                max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
            )
            .to_csv(root / "RESPONSE_SENSITIVITY_SUMMARY.csv", index=False)
        )
    base.write_json(
        root / "RESPONSE_SENSITIVITY_NOTE.json",
        {
            "booked": (
                "solved once at the same sensitivity WorkLimit/gap used by the "
                "learned policies, then reused across alpha/h scenarios"
            ),
            "planner_work_limit": float(s.sensitivity_planner_work_limit),
            "planner_gap_target": sens_gap,
            "projected_hindsight_benchmark": (
                "not rerun here; it changes with alpha/h and is not a performance "
                "ceiling. These scenarios are robustness checks for the frozen "
                "deployable policies."
            ),
        },
    )


def install_reviewed_guards(
    *,
    evaluation_module=None,
    sensitivity_runner_module=None,
) -> None:
    """Install all last-mile fixes after the stage-specific hardening hooks."""

    _install_solver_numeric_params()

    # Every stage gets every numeric worker. This deliberately covers the late
    # Stage-1 deterministic seed audit and the Stage-2 runtime oracle path.
    numeric.install_all_guards()
    # Use release-guard wrappers so macOS spawned children also install the
    # tightened IntFeasTol before creating Gurobi models.
    import final_paper_runtime_fixes as runtime
    runtime._solve_process_task = runtime_worker
    science.deterministic_eval_worker = deterministic_worker
    sensitivity._worker = sensitivity_worker

    # SITE_SHIFT's training hook resolves this module-global helper at call time.
    hardening._solve_site_shift_grid = constrained_site_shift_grid

    # Structural sensitivity runner may already hold a reference assigned by
    # hardening.install_sensitivity_fixes, so replace both names.
    hardening.run_required_sensitivities_with_oracle = required_sensitivities_zero_safe
    sensitivity.run_required_sensitivities = required_sensitivities_zero_safe

    if evaluation_module is not None:
        evaluation_module._run_response_sensitivities = (
            lambda weeks, a, policies, oracle_plans, s, root: fair_response_sensitivities(
                evaluation_module, weeks, a, policies, oracle_plans, s, root
            )
        )

    if sensitivity_runner_module is not None:
        # The runner dispatches through final_paper_required_sensitivities.
        sensitivity._worker = sensitivity_worker


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stamp_training_bundle(root: Path) -> None:
    """Fingerprint this release guard into the accepted Stage-1 bundle."""

    root = Path(root).resolve()
    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError("Stage 1 returned without TRAINING_FREEZE.json")

    note_path = root / "RELEASE_GUARD.json"
    _write_json(
        note_path,
        {
            "version": RELEASE_GUARD_VERSION,
            "numeric_worker_coverage": (
                "runtime, deterministic-evaluation, and sensitivity workers "
                "installed in every reviewed stage"
            ),
            "phi_accounting_atol": float(numeric.PHI_ACCOUNTING_ATOL),
            "phi_accounting_rtol": float(numeric.PHI_ACCOUNTING_RTOL),
            "gurobi_int_feas_tol": float(INT_FEAS_TOL),
            "site_shift_grid": "coefficient-feasible before minimization",
            "structural_gap_closed_zero_baseline": "NaN with explicit status",
            "response_sensitivity_booked": (
                "one solve at sensitivity budget, reused across response scenarios"
            ),
        },
    )

    freeze = _read_json(freeze_path)
    freeze["release_guard_version"] = RELEASE_GUARD_VERSION
    fps = dict(freeze.get("artifact_fingerprints", {}))
    fps[note_path.name] = base.sha256_file(note_path)
    freeze["artifact_fingerprints"] = fps
    _write_json(freeze_path, freeze)


def verify_training_bundle(root: Path) -> None:
    root = Path(root).resolve()
    freeze = _read_json(root / "TRAINING_FREEZE.json")
    if freeze.get("release_guard_version") != RELEASE_GUARD_VERSION:
        raise RuntimeError(
            "Training bundle predates the final all-path release guard; rerun Stage 1"
        )
    note = root / "RELEASE_GUARD.json"
    expected = dict(freeze.get("artifact_fingerprints", {})).get(note.name)
    if not note.exists() or not expected or base.sha256_file(note) != expected:
        raise RuntimeError("Frozen RELEASE_GUARD.json is missing or changed")
