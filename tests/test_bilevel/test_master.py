from __future__ import annotations

import numpy as np

from src.bilevel.config import BilevelConfig
from src.bilevel.master import solve_restricted_master
from src.core.column import ScheduleColumn
from src.core.config import CostConfig
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
        features=np.zeros((1, 2)),
        bookings=np.array([100.0]),
        realized=np.array([110.0]),
        L_bounds=np.array([80.0]),
        U_bounds=np.array([120.0]),
        surgeon_codes=["S1"],
        sos2_data=[],
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


def test_rmp_with_single_column_per_week() -> None:
    wd = _wd()
    res = solve_restricted_master([wd], {0: [_col(False)]}, DummyRecommendation(), BilevelConfig(), CostConfig(), turnover=0.0)
    assert res.selected_columns[0] == 0


def test_rmp_zero_weights_feasible() -> None:
    wd = _wd()
    res = solve_restricted_master([wd], {0: [_col(False)]}, DummyRecommendation(), BilevelConfig(), CostConfig(), turnover=0.0)
    np.testing.assert_allclose(res.w, np.zeros_like(res.w))


def test_rmp_selects_lower_realized_cost() -> None:
    wd = _wd()
    # defer column has higher realized cost due deferral penalty.
    res = solve_restricted_master([wd], {0: [_col(True), _col(False)]}, DummyRecommendation(), BilevelConfig(), CostConfig(deferral_per_case=5000.0), turnover=0.0)
    assert res.selected_columns[0] == 1
