from __future__ import annotations

from datetime import date, datetime

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CostConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock, CaseRecord, WeeklyInstance
from src.planning.evaluation import evaluate


def _sample_cases() -> list[CaseRecord]:
    ts = datetime(2025, 1, 6, 8, 0)
    return [
        CaseRecord(1, "P1", "S1", "Svc", "Elective", "OR1", 100.0, 100.0, ts, 1, 1, 2025, site="TGH"),
        CaseRecord(2, "P2", "S1", "Svc", "Elective", "OR1", 110.0, 110.0, ts, 1, 1, 2025, site="TGH"),
        CaseRecord(3, "P3", "S2", "Svc", "Elective", "OR2", 90.0, 90.0, ts, 1, 1, 2025, site="TGH"),
    ]


def _calendar() -> BlockCalendar:
    return BlockCalendar(
        candidates=[
            CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=240.0, activation_cost=50.0),
            CandidateBlock(day_index=0, site="TGH", room="OR2", capacity_minutes=240.0, activation_cost=50.0),
        ]
    )


def test_cost_matches_evaluation() -> None:
    cases = _sample_cases()
    calendar = _calendar()
    b1 = BlockId(0, "TGH", "OR1")
    b2 = BlockId(0, "TGH", "OR2")
    column = ScheduleColumn(
        z_assign={(0, b1): 1.0, (1, b1): 1.0, (2, b2): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1, b2}),
        y_used=frozenset({b1, b2}),
        n_cases=3,
        block_capacities={b1: 240.0, b2: 240.0},
        block_activation_costs={b1: 50.0, b2: 50.0},
    )

    costs = CostConfig(overtime_per_minute=2.0, idle_per_minute=1.0, deferral_per_case=500.0)
    turnover = 10.0
    durations = np.array([100.0, 110.0, 90.0])

    schedule = column.to_schedule_result(cases)
    instance = WeeklyInstance(
        week_index=0,
        start_date=date(2025, 1, 6),
        end_date=date(2025, 1, 12),
        cases=cases,
        calendar=calendar,
        case_eligible_blocks={0: [b1], 1: [b1], 2: [b2]},
    )

    col_cost = column.compute_cost(durations=durations, costs=costs, turnover=turnover)
    eval_cost = evaluate(instance, schedule, costs=costs, turnover=turnover).total_cost
    assert col_cost == eval_cost


def test_deferred_column_cost() -> None:
    b1 = BlockId(0, "TGH", "OR1")
    column = ScheduleColumn(
        z_assign={},
        z_defer=frozenset({0, 1, 2}),
        v_open=frozenset(),
        y_used=frozenset(),
        n_cases=3,
        block_capacities={b1: 240.0},
        block_activation_costs={b1: 50.0},
    )
    costs = CostConfig(deferral_per_case=321.0)
    assert column.compute_cost(np.array([1.0, 2.0, 3.0]), costs=costs, turnover=10.0) == 3 * 321.0


def test_empty_block_no_idle() -> None:
    b1 = BlockId(0, "TGH", "OR1")
    b2 = BlockId(0, "TGH", "OR2")
    column = ScheduleColumn(
        z_assign={(0, b1): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1}),
        y_used=frozenset({b1}),
        n_cases=1,
        block_capacities={b1: 240.0, b2: 240.0},
        block_activation_costs={b1: 50.0, b2: 50.0},
    )
    costs = CostConfig(overtime_per_minute=0.0, idle_per_minute=1.0, deferral_per_case=0.0)
    # Only OR1 is opened, so OR2 contributes no idle penalty.
    assert column.compute_cost(np.array([100.0]), costs=costs, turnover=0.0) == 50.0 + 140.0


def test_turnover_computation() -> None:
    b1 = BlockId(0, "TGH", "OR1")
    column = ScheduleColumn(
        z_assign={(0, b1): 1.0, (1, b1): 1.0, (2, b1): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1}),
        y_used=frozenset({b1}),
        n_cases=3,
        block_capacities={b1: 1000.0},
        block_activation_costs={b1: 0.0},
    )
    loads = column.compute_block_load(np.array([10.0, 20.0, 30.0]), turnover=15.0)
    assert loads[b1] == 10.0 + 20.0 + 30.0 + 2 * 15.0


def test_hashable() -> None:
    b1 = BlockId(0, "TGH", "OR1")
    c1 = ScheduleColumn(
        z_assign={(0, b1): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1}),
        y_used=frozenset({b1}),
        n_cases=1,
        block_capacities={b1: 240.0},
        block_activation_costs={b1: 50.0},
    )
    c2 = ScheduleColumn(
        z_assign={(0, b1): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1}),
        y_used=frozenset({b1}),
        n_cases=1,
        block_capacities={b1: 240.0},
        block_activation_costs={b1: 50.0},
    )
    c3 = ScheduleColumn(
        z_assign={},
        z_defer=frozenset({0}),
        v_open=frozenset(),
        y_used=frozenset(),
        n_cases=1,
        block_capacities={b1: 240.0},
        block_activation_costs={b1: 50.0},
    )

    assert hash(c1) == hash(c2)
    assert c1 == c2
    assert hash(c1) != hash(c3)
