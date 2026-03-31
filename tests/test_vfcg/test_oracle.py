from __future__ import annotations

from datetime import datetime

import numpy as np

from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock, CaseRecord
from src.solvers.deterministic import solve_deterministic, solve_weekly_optimistic


def _single_case() -> list[CaseRecord]:
    return [
        CaseRecord(
            case_id=1,
            procedure_id="P1",
            surgeon_code="S1",
            service="Svc",
            patient_type="Elective",
            operating_room="OR1",
            booked_duration_min=100.0,
            actual_duration_min=100.0,
            actual_start=datetime(2024, 1, 1, 8, 0, 0),
            week_of_year=1,
            month=1,
            year=2024,
            site="TGH",
        )
    ]


def test_optimistic_equals_plain_in_no_tie_case() -> None:
    cases = _single_case()
    only_block = CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=100.0, activation_cost=0.0)
    calendar = BlockCalendar([only_block])
    eligible = {0: [only_block.id]}

    costs = CostConfig(overtime_per_minute=1.0, idle_per_minute=1.0, deferral_per_case=1000.0)
    solver_cfg = SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False)

    plain = solve_deterministic(
        cases=cases,
        durations=np.array([110.0], dtype=float),
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=eligible,
        turnover=0.0,
        model_name="plain_no_tie",
    )
    opt_col, pred_cost, real_cost, status, _ = solve_weekly_optimistic(
        cases=cases,
        planning_durations=np.array([110.0], dtype=float),
        realized_durations=np.array([90.0], dtype=float),
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=eligible,
        turnover=0.0,
        model_name="optimistic_no_tie",
    )

    assert status in {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"}
    assert opt_col is not None
    assert plain.objective_value is not None
    assert np.isclose(pred_cost, plain.objective_value)
    assert np.isclose(real_cost, 10.0)


def test_optimistic_breaks_tie_using_realized_cost() -> None:
    cases = _single_case()
    block_a = CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=100.0, activation_cost=0.0)
    block_b = CandidateBlock(day_index=1, site="TGH", room="OR2", capacity_minutes=120.0, activation_cost=0.0)
    calendar = BlockCalendar([block_a, block_b])
    eligible = {0: [block_a.id, block_b.id]}

    costs = CostConfig(overtime_per_minute=1.0, idle_per_minute=1.0, deferral_per_case=1000.0)
    solver_cfg = SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False)

    col, pred_cost, real_cost, status, _ = solve_weekly_optimistic(
        cases=cases,
        planning_durations=np.array([110.0], dtype=float),
        realized_durations=np.array([100.0], dtype=float),
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=eligible,
        turnover=0.0,
        model_name="optimistic_tie",
    )

    assert status in {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"}
    assert col is not None
    assigned_block = next(iter(col.z_assign))[1]
    assert assigned_block == BlockId(0, "TGH", "OR1")
    assert np.isclose(pred_cost, 10.0)
    assert np.isclose(real_cost, 0.0)
