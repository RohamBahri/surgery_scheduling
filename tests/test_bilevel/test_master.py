"""Tests for the restricted master problem MILP."""

from __future__ import annotations

import numpy as np

from src.bilevel.config import LegacyCCGConfig
from src.bilevel.master import _compute_mae_base, solve_restricted_master
from src.core.column import ScheduleColumn
from src.core.config import CostConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData


def _bid() -> BlockId:
    return BlockId(0, "TGH", "OR1")


def _calendar() -> BlockCalendar:
    return BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 10.0)])


def _col_assign() -> ScheduleColumn:
    bid = _bid()
    return ScheduleColumn(
        z_assign={(0, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=1,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 10.0},
    )


def _col_defer() -> ScheduleColumn:
    bid = _bid()
    return ScheduleColumn(
        z_assign={},
        z_defer=frozenset({0}),
        v_open=frozenset(),
        y_used=frozenset(),
        n_cases=1,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 10.0},
    )


def _sos2_data(booking: float = 100.0, lower: float = 80.0, upper: float = 130.0) -> SOS2CaseData:
    a, h_plus, h_minus = 0.5, 5.0, 7.0
    hm_eff = min(h_minus, max(0.0, booking - lower))
    hp_eff = min(h_plus, max(0.0, upper - booking))
    knot_x = np.array([lower - booking, -hm_eff, 0.0, hp_eff, upper - booking], dtype=float)
    knot_y = np.array(
        [
            a * ((lower - booking) + hm_eff),
            0.0,
            0.0,
            0.0,
            a * ((upper - booking) - hp_eff),
        ],
        dtype=float,
    )
    return SOS2CaseData(
        case_index=0,
        profile_id=0,
        knot_x=knot_x,
        knot_y=knot_y,
        booking=booking,
        L_bound=lower,
        U_bound=upper,
    )


def _wd(features: np.ndarray | None = None, booking: float = 100.0, realized: float = 105.0) -> WeekRecommendationData:
    if features is None:
        features = np.array([[1.0, 0.5]], dtype=float)
    bid = _bid()
    return WeekRecommendationData(
        week_index=0,
        n_cases=1,
        features=features,
        bookings=np.array([booking], dtype=float),
        realized=np.array([realized], dtype=float),
        L_bounds=np.array([80.0], dtype=float),
        U_bounds=np.array([130.0], dtype=float),
        surgeon_codes=["S1"],
        sos2_data=[_sos2_data(booking=booking)],
        case_eligible_blocks={0: [bid]},
        calendar=_calendar(),
    )


class DummyRecommendation:
    pass


def test_single_column_selection() -> None:
    wd = _wd()
    res = solve_restricted_master([wd], {0: [_col_assign()]}, DummyRecommendation(), LegacyCCGConfig(), CostConfig(), turnover=0.0)
    assert res.selected_columns[0] == 0
    assert res.status in {"OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"}


def test_zero_weights_feasible() -> None:
    wd = _wd()
    res = solve_restricted_master([wd], {0: [_col_assign()]}, DummyRecommendation(), LegacyCCGConfig(), CostConfig(), turnover=0.0)
    assert res.status != "INFEASIBLE"


def test_selects_lower_realized_cost() -> None:
    wd = _wd()
    costs = CostConfig(deferral_per_case=50000.0)
    res = solve_restricted_master(
        [wd],
        {0: [_col_assign(), _col_defer()]},
        DummyRecommendation(),
        LegacyCCGConfig(),
        costs,
        turnover=0.0,
    )
    assert res.selected_columns[0] == 0


def test_non_degradation() -> None:
    wd = _wd()
    costs = CostConfig()
    col = _col_assign()
    status_quo_cost = col.compute_cost(wd.realized, costs, turnover=0.0)

    res = solve_restricted_master([wd], {0: [col]}, DummyRecommendation(), LegacyCCGConfig(), costs, turnover=0.0)
    assert res.objective <= status_quo_cost + 1e-6


def test_mae_base_computation() -> None:
    wd = _wd(booking=100.0, realized=110.0)
    assert np.isclose(_compute_mae_base([wd]), 10.0)


def test_credibility_constraint_limits_w() -> None:
    wd = _wd(booking=100.0, realized=100.0)
    res = solve_restricted_master(
        [wd], {0: [_col_assign()]}, DummyRecommendation(), LegacyCCGConfig(credibility_eta=1.0), CostConfig(), turnover=0.0
    )
    assert res.status != "INFEASIBLE"
