from __future__ import annotations

import numpy as np

from src.bilevel.pricing import run_pricing
from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import WeekRecommendationData


class DummyRecommendation:
    def compute_post_review(self, w: np.ndarray, week_data: WeekRecommendationData) -> np.ndarray:
        _ = w
        return week_data.bookings


def _wd() -> WeekRecommendationData:
    bid = BlockId(0, "TGH", "OR1")
    return WeekRecommendationData(
        week_index=0,
        n_cases=1,
        features=np.zeros((1, 1)),
        bookings=np.array([100.0]),
        realized=np.array([110.0]),
        L_bounds=np.array([80.0]),
        U_bounds=np.array([130.0]),
        surgeon_codes=["S1"],
        sos2_data=[],
        case_eligible_blocks={0: [bid]},
        calendar=BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 20.0)]),
    )


def _col() -> ScheduleColumn:
    bid = BlockId(0, "TGH", "OR1")
    return ScheduleColumn(
        z_assign={(0, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=1,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 20.0},
    )


def test_pricing_returns_none_when_no_improvement(monkeypatch) -> None:
    monkeypatch.setattr("src.bilevel.pricing.solve_pricing", lambda **kwargs: (_col(), 100.0))
    out = run_pricing(np.zeros(1), _wd(), DummyRecommendation(), current_phi=100.0, costs=CostConfig(), solver_cfg=SolverConfig(), turnover=0.0)
    assert out is None


def test_pricing_returns_column_when_improving(monkeypatch) -> None:
    monkeypatch.setattr("src.bilevel.pricing.solve_pricing", lambda **kwargs: (_col(), 80.0))
    out = run_pricing(np.zeros(1), _wd(), DummyRecommendation(), current_phi=100.0, costs=CostConfig(), solver_cfg=SolverConfig(), turnover=0.0)
    assert out is not None
