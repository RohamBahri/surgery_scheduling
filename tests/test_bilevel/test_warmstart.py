from __future__ import annotations

import numpy as np

from src.bilevel.warmstart import generate_warmstart_columns
from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import WeekRecommendationData


class DummyRecommendation:
    def predict_at_quantile(self, week_data: WeekRecommendationData, q: float) -> np.ndarray:
        _ = q
        return week_data.bookings + 1.0


def _week_data() -> WeekRecommendationData:
    bid = BlockId(0, "TGH", "OR1")
    cal = BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 50.0)])
    return WeekRecommendationData(
        week_index=0,
        n_cases=2,
        features=np.zeros((2, 1)),
        bookings=np.array([100.0, 80.0]),
        realized=np.array([95.0, 90.0]),
        L_bounds=np.array([80.0, 70.0]),
        U_bounds=np.array([140.0, 120.0]),
        surgeon_codes=["S1", "S2"],
        sos2_data=[],
        case_eligible_blocks={0: [bid], 1: [bid]},
        calendar=cal,
    )


def _column() -> ScheduleColumn:
    bid = BlockId(0, "TGH", "OR1")
    return ScheduleColumn(
        z_assign={(0, bid): 1.0, (1, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=2,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 50.0},
    )


def test_warmstart_produces_feasible_columns(monkeypatch) -> None:
    def fake_solve_pricing(**kwargs):
        _ = kwargs
        return _column(), 0.0

    monkeypatch.setattr("src.bilevel.warmstart.solve_pricing", fake_solve_pricing)
    cols = generate_warmstart_columns(_week_data(), DummyRecommendation(), CostConfig(), SolverConfig(), turnover=0.0)
    assert cols
    for col in cols:
        assert len(col.z_assign) + len(col.z_defer) == col.n_cases
        for (i, b), _ in col.z_assign.items():
            assert i in [0, 1]
            assert b in col.v_open


def test_warmstart_includes_status_quo(monkeypatch) -> None:
    calls = []

    def fake_solve_pricing(**kwargs):
        calls.append(tuple(np.asarray(kwargs["durations"]).tolist()))
        return _column(), 0.0

    monkeypatch.setattr("src.bilevel.warmstart.solve_pricing", fake_solve_pricing)
    wd = _week_data()
    _ = generate_warmstart_columns(wd, DummyRecommendation(), CostConfig(), SolverConfig(), turnover=0.0)
    assert tuple(wd.bookings.tolist()) in calls
