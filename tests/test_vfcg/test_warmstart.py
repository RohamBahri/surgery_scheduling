from __future__ import annotations

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import WeekRecommendationData
from src.vfcg.warmstart import generate_warmstart_references


class _DummyRecommendation:
    def predict_at_quantile(self, week_data: WeekRecommendationData, q: float) -> np.ndarray:
        _ = q
        return week_data.bookings.copy()


def _week(week_index: int) -> WeekRecommendationData:
    block = CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=480.0, activation_cost=20.0)
    return WeekRecommendationData(
        week_index=week_index,
        n_cases=1,
        features=np.zeros((1, 1), dtype=float),
        bookings=np.array([100.0], dtype=float),
        realized=np.array([110.0], dtype=float),
        L_bounds=np.array([80.0], dtype=float),
        U_bounds=np.array([140.0], dtype=float),
        surgeon_codes=["S1"],
        sos2_data=[],
        case_eligible_blocks={0: [block.id]},
        calendar=BlockCalendar([block]),
    )


def _column(block: CandidateBlock) -> ScheduleColumn:
    return ScheduleColumn(
        z_assign={(0, block.id): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({block.id}),
        y_used=frozenset({block.id}),
        n_cases=1,
        block_capacities={block.id: block.capacity_minutes},
        block_activation_costs={block.id: block.activation_cost},
    )


def test_generate_warmstart_references_each_week_has_reference(monkeypatch) -> None:
    week0 = _week(0)
    week1 = _week(1)

    def _fake_solve_pricing(**kwargs):
        calendar = kwargs["calendar"]
        return _column(calendar.candidates[0]), 1.0

    monkeypatch.setattr("src.vfcg.warmstart.solve_pricing", _fake_solve_pricing)

    refs = generate_warmstart_references(
        week_data_list=[week0, week1],
        recommendation_model=_DummyRecommendation(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(),
        turnover=0.0,
        n_vectors=3,
    )

    assert set(refs.keys()) == {0, 1}
    assert len(refs[0]) >= 1
    assert len(refs[1]) >= 1


def test_generate_warmstart_references_removes_duplicates(monkeypatch) -> None:
    week0 = _week(0)

    def _fake_solve_pricing(**kwargs):
        calendar = kwargs["calendar"]
        # Always return the same schedule for each candidate vector.
        return _column(calendar.candidates[0]), 1.0

    monkeypatch.setattr("src.vfcg.warmstart.solve_pricing", _fake_solve_pricing)

    refs = generate_warmstart_references(
        week_data_list=[week0],
        recommendation_model=_DummyRecommendation(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(),
        turnover=0.0,
        n_vectors=3,
    )

    assert len(refs[0]) == 1
