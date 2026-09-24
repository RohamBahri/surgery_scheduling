"""Numerical-accounting guard for the supported final-paper experiment.

Gurobi's reported objective and an independently recomputed schedule cost can
occasionally differ by a few 1e-4 cost units on objectives around 1e5 because
of floating-point / feasibility tolerances.  That is not a scientific or
combinatorial discrepancy.  The former absolute 1e-5 assertion was therefore
far tighter than the scale of the solved models and caused an eight-hour Stage-1
run to abort on a relative discrepancy of about 1.15e-9.

This module keeps the accounting check strict, but scale-aware:

    |recomputed - solver_sum| <= max(1e-5, 1e-8 * max(1, |values|)).

A genuine accounting/model mismatch still fails loudly.  For accepted
round-off-level discrepancies, the independently recomputed feasible schedule
cost is used as the reported Phi incumbent, so the reported upper bound cannot
be made artificially optimistic by solver-objective rounding.

The module also removes the deprecated explicit ``penalty='l2'`` argument from
the supported RA cross-fitting path.  Leaving the penalty at scikit-learn's
default preserves L2 logistic regression while avoiding the 1.8/1.10
FutureWarning.

The reviewed entry point installs the same accounting guard in training,
deterministic holdout evaluation, and structural-sensitivity workers, including
under macOS ``spawn``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.solvers.fixed_capacity import schedule_metrics


NUMERIC_GUARD_VERSION = "final_paper_numeric_guard_2026_09_24_v2"
PHI_ACCOUNTING_ATOL = 1e-5
PHI_ACCOUNTING_RTOL = 1e-8


def phi_accounting_tolerance(a: float, b: float) -> float:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return max(float(PHI_ACCOUNTING_ATOL), float(PHI_ACCOUNTING_RTOL) * scale)


def assert_phi_accounting_close(recomputed: float, solver_sum: float, *, context: str) -> None:
    """Reject material accounting errors while tolerating solver-scale roundoff."""

    a = float(recomputed)
    b = float(solver_sum)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise AssertionError(f"{context}: non-finite Phi accounting values {a!r}, {b!r}")
    err = abs(a - b)
    tol = phi_accounting_tolerance(a, b)
    if err > tol:
        rel = err / max(1.0, abs(a), abs(b))
        raise AssertionError(
            f"{context}: decomposed Phi mismatch recomputed={a:.12g} solver_sum={b:.12g} "
            f"abs_error={err:.6g} relative_error={rel:.3g} tolerance={tol:.6g}"
        )
    if err > PHI_ACCOUNTING_ATOL:
        rel = err / max(1.0, abs(a), abs(b))
        base.LOG.warning(
            "[NUMERIC-AUDIT] %s | recomputed Phi and summed site solver Phi differ only at "
            "solver-scale roundoff: abs=%.6g rel=%.3g tol=%.6g",
            context,
            err,
            rel,
            tol,
        )


def _psi_constant(week: base.WeekBundle, durations: np.ndarray, s, turnover: float) -> float:
    d = np.asarray(durations, dtype=float)
    total_capacity = float(
        sum(float(b.capacity_minutes) for b in week.instance.calendar.candidates)
    )
    return float(s.idle) * (
        total_capacity - float(d.sum()) - float(turnover) * week.instance.num_cases
    )


def _accounted_native_gap(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    *,
    turnover: float,
    recomputed_phi_ub: float,
    psi_lb: float,
) -> float:
    # Use the independently recomputed feasible Phi as the incumbent side of
    # the reduced-Psi gap.  This is conservative if solver reporting is lower by
    # a few floating-point units.
    psi_ub = float(recomputed_phi_ub) - _psi_constant(week, durations, s, turnover)
    return float(base.rel_gap(psi_ub, float(psi_lb)))


def warning_free_crossfit_pi(
    a: base.Arrays, labels: np.ndarray, s
) -> tuple[np.ndarray, dict[str, Any]]:
    """Supported exposure cross-fit without deprecated sklearn penalty syntax."""

    X = sparse.csr_matrix(a.X, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(a.week_ids)
    if X.shape[0] != len(y) or len(groups) != len(y):
        raise ValueError("crossfit_pi input lengths do not agree")

    unique_groups = np.unique(groups)
    pred = np.zeros(len(y), dtype=float)
    if len(unique_groups) < 2:
        pred[:] = float(np.mean(y)) if len(y) else 0.5
    else:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
            if np.unique(y[tr]).size < 2:
                pred[te] = float(np.mean(y[tr]))
            else:
                # L2 is LogisticRegression's default.  Do not pass the
                # deprecated ``penalty`` parameter explicitly (sklearn >=1.8).
                mdl = LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    fit_intercept=False,
                    random_state=int(s.random_seed) + fold,
                )
                mdl.fit(X[tr], y[tr])
                pred[te] = mdl.predict_proba(X[te])[:, 1]

    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    prevalence = float(np.mean(y)) if len(y) else math.nan
    metrics = {
        "prevalence": prevalence,
        "pi_mean": float(np.mean(pred)) if len(pred) else math.nan,
        "brier": float(brier_score_loss(y, pred)) if len(y) else math.nan,
        "constant_brier": float(np.mean((y - prevalence) ** 2)) if len(y) else math.nan,
        "auc": float(roc_auc_score(y, pred)) if np.unique(y).size == 2 else math.nan,
    }
    return pred, metrics


def reviewed_final_solve_week(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int = 1,
    warm: ScheduleColumn | None = None,
    label: str = "plan",
    deterministic_tiebreak: bool = False,
    tiebreak_seconds: int | None = None,
) -> base.PlanResult:
    """Two-site weekly solve with the reviewed scale-aware Phi audit."""

    del deterministic_tiebreak, tiebreak_seconds
    d = np.asarray(durations, dtype=float)
    if d.shape != (week.instance.num_cases,):
        raise ValueError("duration length mismatch")

    t0 = time.perf_counter()
    site_parts: list[tuple[Any, ScheduleColumn]] = []
    solver_phi_ub = phi_lb = solver_psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_proven_optimal = True

    for site in final.PRIMARY_SITES:
        view, _, result = final._solve_site_assignment(
            week,
            d,
            s,
            site,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            warm=warm,
            objective_mode=final.PRIMARY_OBJECTIVE,
        )
        if (
            result.column is None
            or result.phi_ub is None
            or result.phi_lb is None
            or result.psi_ub is None
            or result.psi_lb is None
        ):
            raise RuntimeError(
                f"Week {week.position} {label} site {site}: fixed planner returned no incumbent/bound"
            )
        site_parts.append((view, result.column))
        solver_phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        solver_psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_proven_optimal = all_proven_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, site_parts)
    combined_metrics = schedule_metrics(
        column, d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER
    )
    recomputed_phi = float(combined_metrics["phi"])
    assert_phi_accounting_close(
        recomputed_phi,
        float(solver_phi_ub),
        context=f"Week {week.position} {label}",
    )

    native_gap = _accounted_native_gap(
        week,
        d,
        s,
        turnover=float(final.PRIMARY_TURNOVER),
        recomputed_phi_ub=recomputed_phi,
        psi_lb=psi_lb,
    )
    # Proven-optimal status is judged on the solver's own incumbent/bound pair;
    # the independently recomputed feasible objective is used for reporting.
    exact = bool(all_proven_optimal and abs(solver_psi_ub - psi_lb) <= 1e-6)
    status = "OPTIMAL" if all_proven_optimal else "|".join(statuses)
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=recomputed_phi,
        bound=float(phi_lb),
        gap=native_gap,
        status=status,
        solve_seconds=time.perf_counter() - t0,
        exact=exact,
        tiebreak_used=False,
    )


def training_process_task(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    *,
    seconds: int,
    gap: float,
    warm: ScheduleColumn | None,
    label: str,
    deterministic_tiebreak: bool,
    tiebreak_seconds: int | None,
) -> base.PlanResult:
    """Spawn-safe Stage-1 worker using the reviewed accounting tolerance."""

    return reviewed_final_solve_week(
        week,
        np.asarray(durations, dtype=float),
        s,
        time_limit=seconds,
        mip_gap=gap,
        threads=1,
        warm=warm,
        label=label,
        deterministic_tiebreak=deterministic_tiebreak,
        tiebreak_seconds=tiebreak_seconds,
    )


def deterministic_eval_worker(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    *,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    label: str,
    seed: int,
) -> base.PlanResult:
    """Spawn-safe deterministic Stage-2 worker with the same numeric audit."""

    import final_paper_finalization_fixes as hardening

    d = np.asarray(durations, dtype=float)
    t0 = time.perf_counter()
    parts: list[tuple[Any, ScheduleColumn]] = []
    solver_phi_ub = phi_lb = solver_psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_optimal = True

    for site in final.PRIMARY_SITES:
        view, result = hardening.robust_deterministic_site_solve(
            week,
            d,
            s,
            site,
            work_limit=work_limit,
            wall_seconds=wall_seconds,
            mip_gap=mip_gap,
            seed=seed,
        )
        if any(x is None for x in (result.phi_ub, result.phi_lb, result.psi_ub, result.psi_lb)):
            raise RuntimeError(f"Week {week.position} {label} {site}: missing bound/incumbent")
        parts.append((view, result.column))
        solver_phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        solver_psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    metrics = schedule_metrics(column, d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
    recomputed_phi = float(metrics["phi"])
    assert_phi_accounting_close(
        recomputed_phi,
        float(solver_phi_ub),
        context=f"Deterministic week {week.position} {label}",
    )
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=recomputed_phi,
        bound=float(phi_lb),
        gap=_accounted_native_gap(
            week,
            d,
            s,
            turnover=float(final.PRIMARY_TURNOVER),
            recomputed_phi_ub=recomputed_phi,
            psi_lb=psi_lb,
        ),
        status="OPTIMAL" if all_optimal else "|".join(statuses),
        solve_seconds=time.perf_counter() - t0,
        exact=bool(all_optimal and abs(solver_psi_ub - psi_lb) <= 1e-6),
        tiebreak_used=False,
    )


def sensitivity_worker(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    *,
    turnover: float,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    label: str,
) -> base.PlanResult:
    """Spawn-safe structural-sensitivity worker with reviewed Phi accounting."""

    import final_paper_finalization_fixes as hardening

    d = np.asarray(durations, dtype=float)
    t0 = time.perf_counter()
    parts: list[tuple[Any, ScheduleColumn]] = []
    solver_phi_ub = phi_lb = solver_psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_optimal = True
    initial_wall = max(
        int(wall_seconds),
        int(math.ceil(hardening.EMERGENCY_WALL_TO_WORK_MULTIPLIER * float(work_limit))),
    )

    for site in final.PRIMARY_SITES:
        view, result = hardening._fixed_site_once(
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
            retry_wall = int(math.ceil(initial_wall * hardening.EMERGENCY_RETRY_MULTIPLIER))
            base.LOG.warning(
                "[SENS] %s week=%s site=%s wall cap fired; retrying with %ss",
                label,
                week.position,
                site,
                retry_wall,
            )
            view, result = hardening._fixed_site_once(
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

        if any(x is None for x in (result.phi_ub, result.phi_lb, result.psi_ub, result.psi_lb)):
            raise RuntimeError(f"Sensitivity {label}, week {week.position}, site {site}: missing bound/incumbent")
        parts.append((view, result.column))
        solver_phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        solver_psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    metrics = schedule_metrics(column, d, final.final_cost_cfg(s), float(turnover))
    recomputed_phi = float(metrics["phi"])
    assert_phi_accounting_close(
        recomputed_phi,
        float(solver_phi_ub),
        context=f"Sensitivity week {week.position} {label}",
    )
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=recomputed_phi,
        bound=float(phi_lb),
        gap=_accounted_native_gap(
            week,
            d,
            s,
            turnover=float(turnover),
            recomputed_phi_ub=recomputed_phi,
            psi_lb=psi_lb,
        ),
        status="OPTIMAL" if all_optimal else "|".join(statuses),
        solve_seconds=time.perf_counter() - t0,
        exact=bool(all_optimal and abs(solver_psi_ub - psi_lb) <= 1e-6),
        tiebreak_used=False,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stamp_training_bundle(root: Path) -> None:
    """Fingerprint the numeric guard into the accepted Stage-1 bundle."""

    root = Path(root).resolve()
    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError("Stage 1 returned without TRAINING_FREEZE.json")
    note = root / "NUMERIC_GUARD.json"
    _write_json(
        note,
        {
            "version": NUMERIC_GUARD_VERSION,
            "phi_accounting_atol": PHI_ACCOUNTING_ATOL,
            "phi_accounting_rtol": PHI_ACCOUNTING_RTOL,
            "reported_incumbent": "independently recomputed feasible schedule Phi",
            "known_2026_09_24_failure": {
                "recomputed_phi": 154909.6444510882,
                "summed_solver_phi": 154909.64427304053,
                "absolute_difference": 0.00017804767481982708,
                "interpretation": "accepted solver-scale numerical roundoff; not an assignment/accounting failure",
            },
            "crossfit_logistic_regression": "default L2 penalty; deprecated explicit penalty argument omitted",
        },
    )
    freeze = _read_json(freeze_path)
    freeze["numeric_guard_version"] = NUMERIC_GUARD_VERSION
    fps = dict(freeze.get("artifact_fingerprints", {}))
    fps[note.name] = base.sha256_file(note)
    freeze["artifact_fingerprints"] = fps
    _write_json(freeze_path, freeze)


def verify_training_bundle(root: Path) -> None:
    root = Path(root).resolve()
    freeze = _read_json(root / "TRAINING_FREEZE.json")
    if freeze.get("numeric_guard_version") != NUMERIC_GUARD_VERSION:
        raise RuntimeError("Training bundle predates the reviewed numeric-accounting guard")
    note = root / "NUMERIC_GUARD.json"
    expected = dict(freeze.get("artifact_fingerprints", {})).get(note.name)
    if not note.exists() or not expected or base.sha256_file(note) != expected:
        raise RuntimeError("Frozen NUMERIC_GUARD.json is missing or changed")


def install_training_guard() -> None:
    # ``apply_runtime_fixes`` later assigns ``base.crossfit_pi`` from this module
    # attribute, so replacing the attribute here survives the Stage-1 install
    # sequence and removes the sklearn FutureWarning in spawned/local execution.
    runtime.safe_crossfit_pi = warning_free_crossfit_pi
    # safe_solve_batch resolves this module global at submit time, so the
    # submitted callable is this top-level spawn-safe function.
    runtime._solve_process_task = training_process_task


def install_evaluation_guard() -> None:
    # deterministic_eval_solve_batch resolves its worker global before submit.
    science.deterministic_eval_worker = deterministic_eval_worker


def install_sensitivity_guard() -> None:
    # final_paper_required_sensitivities.solve_batch likewise resolves _worker.
    sensitivity._worker = sensitivity_worker
