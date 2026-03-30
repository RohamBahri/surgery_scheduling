"""Integration tests for CCG + real restricted master."""

from __future__ import annotations

import numpy as np

from src.bilevel.ccg import solve_bilevel_ccg
from src.bilevel.config import LegacyCCGConfig
from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData


def _bid() -> BlockId:
    return BlockId(0, "TGH", "OR1")


def _cal() -> BlockCalendar:
    return BlockCalendar([CandidateBlock(0, "TGH", "OR1", 480.0, 10.0)])


def _sos2(booking: float, lower: float, upper: float) -> SOS2CaseData:
    a, h_plus, h_minus = 0.5, 5.0, 7.0
    hm_eff = min(h_minus, max(0.0, booking - lower))
    hp_eff = min(h_plus, max(0.0, upper - booking))
    kx = np.array([lower - booking, -hm_eff, 0.0, hp_eff, upper - booking], dtype=float)
    ky = np.array([a * ((lower - booking) + hm_eff), 0.0, 0.0, 0.0, a * ((upper - booking) - hp_eff)], dtype=float)
    return SOS2CaseData(0, 0, kx, ky, booking, lower, upper)


def _wd(week_index: int, booking: float = 100.0, realized: float = 120.0) -> WeekRecommendationData:
    bid = _bid()
    return WeekRecommendationData(
        week_index=week_index,
        n_cases=1,
        features=np.array([[1.0, 0.5]], dtype=float),
        bookings=np.array([booking], dtype=float),
        realized=np.array([realized], dtype=float),
        L_bounds=np.array([80.0], dtype=float),
        U_bounds=np.array([150.0], dtype=float),
        surgeon_codes=["S1"],
        sos2_data=[_sos2(booking, 80.0, 150.0)],
        case_eligible_blocks={0: [bid]},
        calendar=_cal(),
    )


def _col(booking_week: WeekRecommendationData) -> ScheduleColumn:
    bid = _bid()
    _ = booking_week
    return ScheduleColumn(
        z_assign={(0, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=1,
        block_capacities={bid: 480.0},
        block_activation_costs={bid: 10.0},
    )


class DummyRecModel:
    def compute_post_review(self, w: np.ndarray, wd: WeekRecommendationData) -> np.ndarray:
        delta_rec = float(wd.features @ w)
        delta_rec = float(np.clip(delta_rec, wd.L_bounds[0] - wd.bookings[0], wd.U_bounds[0] - wd.bookings[0]))
        a, h_plus, h_minus = 0.5, 5.0, 7.0
        if delta_rec < -h_minus:
            delta_post = a * (delta_rec + h_minus)
        elif delta_rec > h_plus:
            delta_post = a * (delta_rec - h_plus)
        else:
            delta_post = 0.0
        return wd.bookings + np.array([delta_post], dtype=float)

    def predict_at_quantile(self, wd: WeekRecommendationData, q: float) -> np.ndarray:
        _ = q
        return wd.bookings


def test_ccg_terminates_with_real_master(monkeypatch) -> None:
    wds = [_wd(0), _wd(1, booking=90.0, realized=110.0)]

    monkeypatch.setattr("src.bilevel.ccg.generate_warmstart_columns", lambda wd, *args, **kwargs: [_col(wd)])
    monkeypatch.setattr("src.bilevel.ccg.run_pricing", lambda *args, **kwargs: None)

    res = solve_bilevel_ccg(
        wds,
        DummyRecModel(),
        LegacyCCGConfig(max_iterations=5),
        CostConfig(),
        SolverConfig(time_limit_seconds=30),
        turnover=0.0,
    )
    assert res.n_iterations <= 5
    assert res.w_optimal is not None
