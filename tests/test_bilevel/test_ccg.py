from __future__ import annotations

import numpy as np

from src.bilevel.ccg import solve_bilevel_ccg
from src.bilevel.config import BilevelConfig
from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData


class DummyRecommendation:
    def compute_post_review(self, w: np.ndarray, week_data: WeekRecommendationData) -> np.ndarray:
        _ = w
        return week_data.bookings


def _wd() -> WeekRecommendationData:
    bid = BlockId(0, "TGH", "OR1")
    booking = 100.0
    lower = 80.0
    upper = 130.0
    knot_x = np.array([lower - booking, -7.0, 0.0, 5.0, upper - booking], dtype=float)
    knot_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    return WeekRecommendationData(
        week_index=0,
        n_cases=1,
        features=np.zeros((1, 1)),
        bookings=np.array([booking]),
        realized=np.array([105.0]),
        L_bounds=np.array([lower]),
        U_bounds=np.array([upper]),
        surgeon_codes=["S1"],
        sos2_data=[
            SOS2CaseData(
                case_index=0,
                profile_id=0,
                knot_x=knot_x,
                knot_y=knot_y,
                booking=booking,
                L_bound=lower,
                U_bound=upper,
            )
        ],
        case_eligible_blocks={0: [bid]},
        calendar=BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 10.0)]),
    )


def _col(defer: bool) -> ScheduleColumn:
    bid = BlockId(0, "TGH", "OR1")
    return ScheduleColumn(
        z_assign={} if defer else {(0, bid): 1.0},
        z_defer=frozenset({0}) if defer else frozenset(),
        v_open=frozenset() if defer else frozenset({bid}),
        y_used=frozenset() if defer else frozenset({bid}),
        n_cases=1,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 10.0},
    )


def test_ccg_terminates_on_tiny_instance(monkeypatch) -> None:
    monkeypatch.setattr("src.bilevel.ccg.generate_warmstart_columns", lambda *args, **kwargs: [_col(False)])
    monkeypatch.setattr("src.bilevel.ccg.run_pricing", lambda *args, **kwargs: None)
    res = solve_bilevel_ccg([_wd()], DummyRecommendation(), BilevelConfig(max_ccg_iterations=5), CostConfig(), SolverConfig(), turnover=0.0)
    assert res.n_iterations <= 5


def test_ccg_non_degradation(monkeypatch) -> None:
    monkeypatch.setattr("src.bilevel.ccg.generate_warmstart_columns", lambda *args, **kwargs: [_col(False), _col(True)])
    monkeypatch.setattr("src.bilevel.ccg.run_pricing", lambda *args, **kwargs: None)
    costs = CostConfig(deferral_per_case=5000.0)
    res = solve_bilevel_ccg([_wd()], DummyRecommendation(), BilevelConfig(max_ccg_iterations=2), costs, SolverConfig(), turnover=0.0)
    status_quo = _col(False).compute_cost(_wd().realized, costs, turnover=0.0)
    assert res.objective <= status_quo


def test_ccg_oracle_bound(monkeypatch) -> None:
    monkeypatch.setattr("src.bilevel.ccg.generate_warmstart_columns", lambda *args, **kwargs: [_col(False)])
    monkeypatch.setattr("src.bilevel.ccg.run_pricing", lambda *args, **kwargs: None)
    costs = CostConfig()
    res = solve_bilevel_ccg([_wd()], DummyRecommendation(), BilevelConfig(max_ccg_iterations=2), costs, SolverConfig(), turnover=0.0)
    oracle = _col(False).compute_cost(_wd().realized, costs, turnover=0.0)
    assert res.objective >= oracle
