"""Numerical-accounting guard for the supported final-paper experiment.

Gurobi's reported objective and an independently recomputed schedule cost can
occasionally differ by a few 1e-4 cost units on objectives around 1e5 because
of floating-point / feasibility tolerances.  That is not a scientific or
combinatorial discrepancy.  The former absolute 1e-5 assertion was therefore
far tighter than the scale of the solved models and caused an eight-hour Stage-1
run to abort on a relative discrepancy of about 1.15e-9.

This module keeps the accounting check strict, but scale-aware:

    |recomputed - solver_sum| <= max(1e-5, 1e-8 * max(1, |values|)).

A genuine accounting/model mismatch still fails loudly.  The reviewed entry
point installs the same guard in training, deterministic holdout evaluation,
and structural-sensitivity workers, including under macOS ``spawn``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.solvers.fixed_capacity import schedule_metrics


NUMERIC_GUARD_VERSION = "final_paper_numeric_guard_2026_09_24_v1"
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
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
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
        phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_proven_optimal = all_proven_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, site_parts)
    combined_metrics = schedule_metrics(
        column, d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER
    )
    assert_phi_accounting_close(
        float(combined_metrics["phi"]),
        float(phi_ub),
        context=f"Week {week.position} {label}",
    )

    native_gap = base.rel_gap(psi_ub, psi_lb)
    exact = bool(all_proven_optimal and abs(psi_ub - psi_lb) <= 1e-6)
    status = "OPTIMAL" if all_proven_optimal else "|".join(statuses)
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=float(phi_ub),
        bound=float(phi_lb),
        gap=float(native_gap),
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

    # Import lazily to avoid an import cycle at module load time.
    import final_paper_finalization_fixes as hardening

    d = np.asarray(durations, dtype=float)
    t0 = time.perf_counter()
    parts: list[tuple[Any, ScheduleColumn]] = []
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
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
        phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    metrics = schedule_metrics(column, d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
    assert_phi_accounting_close(
        float(metrics["phi"]),
        float(phi_ub),
        context=f"Deterministic week {week.position} {label}",
    )
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
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
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

        parts.append((view, result.column))
        phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    metrics = schedule_metrics(column, d, final.final_cost_cfg(s), float(turnover))
    assert_phi_accounting_close(
        float(metrics["phi"]),
        float(phi_ub),
        context=f"Sensitivity week {week.position} {label}",
    )
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


def install_training_guard() -> None:
    # safe_solve_batch resolves this module global at submit time, so the
    # submitted callable is this top-level spawn-safe function.
    runtime._solve_process_task = training_process_task


def install_evaluation_guard() -> None:
    # deterministic_eval_solve_batch resolves its worker global before submit.
    science.deterministic_eval_worker = deterministic_eval_worker


def install_sensitivity_guard() -> None:
    # final_paper_required_sensitivities.solve_batch likewise resolves _worker.
    sensitivity._worker = sensitivity_worker
