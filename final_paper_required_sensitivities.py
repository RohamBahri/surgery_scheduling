"""Prespecified secondary holdout sensitivities for the final paper.

These analyses are intentionally separate from the primary result but are run
inside the same one-shot Stage-2 invocation, before the holdout bundle is marked
complete.  They use the already frozen policies; nothing is retrained.

Required scenarios
------------------
1. ``regular_template`` capacity instead of the primary median-count roster,
   with the primary 30-minute turnover assumption.
2. zero turnover under the primary median-count roster.

To contain runtime, the secondary planner uses the frozen reduced WorkLimit and
0.2% MIP-gap target.  The reported quantity is realized weekly planning cost;
these robustness tables do not redefine the primary oracle/regret brackets.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.config import SolverConfig
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.roster import build_fixed_roster
from src.solvers.fixed_capacity import schedule_metrics, solve_fixed_capacity_assignment


SENSITIVITY_METHODS = ("BOOKED", "RA", "RA_FULL", "OS", "VF")
SENSITIVITY_GAP = 0.002


def build_bundles_with_roster(
    df_scoped: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    *,
    cfg,
    context: final.PlanningContext,
    roster_name: str,
    offset: int,
) -> list[base.WeekBundle]:
    """Build prospective weeks with a training-only alternative fixed roster."""
    ordered = science._predecision_ordered_pool(df_scoped)
    out: list[base.WeekBundle] = []
    for j, start in enumerate(starts):
        start = pd.Timestamp(start).normalize()
        roster = build_fixed_roster(context.train, start, cfg, roster_name)
        inst = build_weekly_instance_with_calendar(
            ordered,
            start,
            offset + j,
            cfg,
            roster.calendar,
            context.history,
            final.PRIMARY_ELIGIBILITY_WEEKS,
        )
        if inst.num_cases == 0:
            raise RuntimeError(f"Sensitivity roster produced empty week {start.date()}")
        missing = [i for i in range(inst.num_cases) if not inst.case_eligible_blocks.get(i)]
        if missing:
            raise RuntimeError(
                f"Sensitivity roster {roster_name} leaves {len(missing)} cases in week "
                f"{start.date()} without an eligible block"
            )
        for i, case in enumerate(inst.cases):
            if any(bid.site != case.site for bid in inst.case_eligible_blocks[i]):
                raise AssertionError("Cross-site eligibility in sensitivity instance")
        out.append(base.WeekBundle(offset + j, start, inst))
    return out


def _worker(
    week: base.WeekBundle,
    durations: np.ndarray,
    s: science.ScientificFinalSettings,
    *,
    turnover: float,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    label: str,
) -> base.PlanResult:
    d = np.asarray(durations, dtype=float)
    t0 = time.perf_counter()
    parts: list[tuple[final.SiteView, ScheduleColumn]] = []
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_optimal = True

    for site in final.PRIMARY_SITES:
        view = final._site_view(week, site)
        local_d = d[view.global_indices]
        cfg = SolverConfig(
            time_limit_seconds=max(1, int(wall_seconds)),
            work_limit=float(work_limit),
            mip_gap=float(mip_gap),
            threads=1,
            verbose=bool(s.verbose),
            mip_gap_abs=1e-10,
            seed=int(s.random_seed),
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
                f"Sensitivity {label}, week {week.position}, site {site}: no incumbent/bound"
            )
        if result.diagnostics.status == "TIME_LIMIT":
            raise RuntimeError(
                f"Sensitivity {label}, week {week.position}, site {site}: emergency wall "
                "TimeLimit fired before deterministic WorkLimit/gap stopping"
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
    native_gap = base.rel_gap(psi_ub, psi_lb)
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=phi_ub,
        bound=phi_lb,
        gap=float(native_gap),
        status="OPTIMAL" if all_optimal else "|".join(statuses),
        solve_seconds=time.perf_counter() - t0,
        exact=bool(all_optimal and abs(psi_ub - psi_lb) <= 1e-6),
        tiebreak_used=False,
    )


def solve_batch(
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s: science.ScientificFinalSettings,
    *,
    turnover: float,
    label: str,
) -> dict[int, base.PlanResult]:
    workers = min(int(s.cores), len(weeks))
    out: dict[int, base.PlanResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {
            ex.submit(
                _worker,
                week,
                np.asarray(duration_by_week[week.position], dtype=float),
                s,
                turnover=float(turnover),
                work_limit=float(s.sensitivity_planner_work_limit),
                wall_seconds=int(s.final_planner_seconds),
                mip_gap=SENSITIVITY_GAP,
                label=label,
            ): week.position
            for week in weeks
        }
        for f in as_completed(fut):
            out[fut[f]] = f.result()
    return out


def _evaluate_scenario(
    scenario: str,
    weeks: Sequence[base.WeekBundle],
    arrays: base.Arrays,
    policies: Mapping[str, np.ndarray | None],
    s: science.ScientificFinalSettings,
    *,
    roster_name: str,
    turnover: float,
) -> list[dict]:
    rows: list[dict] = []
    selected = {name: policies[name] for name in SENSITIVITY_METHODS}
    for name, w in selected.items():
        dm, _ = base.policy_duration_map(name, w, arrays, s)
        plans = solve_batch(
            weeks,
            dm,
            s,
            turnover=turnover,
            label=f"{scenario}_{name}",
        )
        for wk, idx in arrays.week_slices.items():
            plan = plans[wk]
            actual = arrays.actual[idx]
            realized = float(
                plan.column.compute_cost(actual, final.final_cost_cfg(s), float(turnover))
            )
            rows.append(
                {
                    "scenario": scenario,
                    "roster": roster_name,
                    "turnover_minutes": float(turnover),
                    "method": name,
                    "week": wk,
                    "realized_cost": realized,
                    "planning_gap_native_psi": float(plan.gap),
                    "planning_status": str(plan.status),
                    "planning_seconds": float(plan.solve_seconds),
                }
            )
    return rows


def run_required_sensitivities(
    *,
    scoped: pd.DataFrame,
    hold_starts: Sequence[pd.Timestamp],
    cfg,
    context: final.PlanningContext,
    primary_weeks: Sequence[base.WeekBundle],
    primary_arrays: base.Arrays,
    encoder: science.ScientificFeatureEncoder,
    policies: Mapping[str, np.ndarray | None],
    s: science.ScientificFinalSettings,
    root,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both prespecified structural sensitivities using frozen policies."""
    rows: list[dict] = []

    regular_weeks = build_bundles_with_roster(
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
        _evaluate_scenario(
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
        _evaluate_scenario(
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
            max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
            mean_planning_seconds=("planning_seconds", "mean"),
        )
    )
    weekly.to_csv(root / "REQUIRED_SENSITIVITY_WEEKLY.csv", index=False)
    summary.to_csv(root / "REQUIRED_SENSITIVITY_SUMMARY.csv", index=False)
    return weekly, summary
