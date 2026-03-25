from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from src.core.config import CostConfig
from src.core.types import BlockCalendar, BlockId, CandidateBlock, CaseRecord, WeeklyInstance
from src.estimation import EstimationResult
from src.estimation.recommendation import RecommendationModel


class DummyCriticalRatios:
    def __init__(self, ratios: dict[str, float]) -> None:
        self._ratios = ratios

    def get_ratio(self, surgeon_code: str) -> float:
        return self._ratios.get(surgeon_code, 0.5)


class DummyQuantileModel:
    def predict(self, df: pd.DataFrame, q: float) -> np.ndarray:
        b = df["booked_time_minutes"].to_numpy(dtype=float)
        if q <= 0.02:
            return b - 20.0
        return b + 30.0


class DummyParams:
    def __init__(self, a: float, h_plus: float, h_minus: float) -> None:
        self.a = a
        self.h_plus = h_plus
        self.h_minus = h_minus


class DummyResponseEstimator:
    def get_params(self, surgeon_code: str) -> DummyParams:
        _ = surgeon_code
        return DummyParams(a=0.5, h_plus=5.0, h_minus=7.0)


class DummyResponseProfiler:
    def get_profile_id(self, surgeon_code: str) -> int:
        return 1 if surgeon_code == "S1" else 2

    def get_profile(self, surgeon_code: str):
        if surgeon_code == "S1":
            return type("P", (), {"profile_id": 1, "a": 0.5, "h_plus": 5.0, "h_minus": 7.0})()
        return type("P", (), {"profile_id": 2, "a": 0.4, "h_plus": 3.0, "h_minus": 6.0})()

    def get_sos2_knots(self, profile_id: int, L_ti: float, U_ti: float, b_ti: float):
        p = self.get_profile("S1" if profile_id == 1 else "S2")
        h_minus = min(p.h_minus, max(0.0, b_ti - L_ti))
        h_plus = min(p.h_plus, max(0.0, U_ti - b_ti))
        x = np.array([L_ti - b_ti, -h_minus, 0.0, h_plus, U_ti - b_ti], dtype=float)
        y = np.array([p.a * ((L_ti - b_ti) + h_minus), 0.0, 0.0, 0.0, p.a * ((U_ti - b_ti) - h_plus)], dtype=float)
        return x, y


def _instance() -> WeeklyInstance:
    ts = datetime(2025, 1, 6, 8, 0)
    cases = [
        CaseRecord(1, "P1", "S1", "SvcA", "Elective", "OR1", 120.0, 130.0, ts, 2, 1, 2025, site="TGH"),
        CaseRecord(2, "P2", "S2", "SvcB", "Elective", "OR1", 90.0, 100.0, ts, 2, 1, 2025, site="TGH"),
    ]
    b1 = CandidateBlock(day_index=0, site="TGH", room="OR1", capacity_minutes=480.0, activation_cost=100.0)
    calendar = BlockCalendar(candidates=[b1])
    return WeeklyInstance(
        week_index=0,
        start_date=date(2025, 1, 6),
        end_date=date(2025, 1, 12),
        cases=cases,
        calendar=calendar,
        case_eligible_blocks={0: [BlockId(0, "TGH", "OR1")], 1: [BlockId(0, "TGH", "OR1")]},
    )


def _model() -> RecommendationModel:
    est = EstimationResult(
        quantile_model=DummyQuantileModel(),
        critical_ratios=DummyCriticalRatios({"S1": 0.7, "S2": 0.3}),
        response_estimator=DummyResponseEstimator(),
        response_profiler=DummyResponseProfiler(),
        bootstrap=None,
    )
    model = RecommendationModel(estimation_result=est, costs=CostConfig(overtime_per_minute=15.0, idle_per_minute=10.0), w_max=100.0)
    train_df = pd.DataFrame({"case_service": ["SvcA", "SvcB"], "booked_time_minutes": [120.0, 90.0]})
    model.prepare(train_df)
    return model


def test_zero_weights_give_zero_correction() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    w = np.zeros(model.feature_dim)
    d_post = model.compute_post_review(w, week)
    np.testing.assert_allclose(d_post, week.bookings)


def test_plausibility_clipping() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    w = np.ones(model.feature_dim) * 100.0
    delta = model.compute_corrections(w, week.features, week.bookings, week.L_bounds, week.U_bounds)
    np.testing.assert_allclose(delta, week.U_bounds - week.bookings)


def test_inaction_band() -> None:
    model = _model()
    delta_post = model.apply_response(np.array([2.0, -1.0]), ["S1", "S2"])
    np.testing.assert_allclose(delta_post, np.array([0.0, 0.0]))


def test_partial_acceptance() -> None:
    model = _model()
    delta_post = model.apply_response(np.array([20.0]), ["S1"])
    # a=0.5, h_plus=5 => 0.5 * (20-5)
    np.testing.assert_allclose(delta_post, np.array([7.5]))


def test_credibility_at_zero() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    w = np.zeros(model.feature_dim)
    cred = model.compute_credibility(w, week, realized=week.realized)
    expected = float(np.mean(np.abs(week.realized - week.bookings)))
    assert cred == expected


def test_sos2_knots_consistent_with_profiler() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    first = week.sos2_data[0]
    px, py = model._estimation.response_profiler.get_sos2_knots(first.profile_id, first.L_bound, first.U_bound, first.booking)
    np.testing.assert_allclose(first.knot_x, px)
    np.testing.assert_allclose(first.knot_y, py)


def test_feature_includes_q_hat() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    assert "q_hat_s" in model.feature_names
    q_idx = model.feature_names.index("q_hat_s")
    np.testing.assert_allclose(week.features[:, q_idx], np.array([0.7, 0.3]))
