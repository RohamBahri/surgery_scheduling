import numpy as np
import pandas as pd

from src.core.config import CostConfig
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, WeeklyInstance
from src.estimation.recommendation import RecommendationModel


class _CriticalRatios:
    def get_ratio(self, surgeon_code: str) -> float:
        return {"S1": 0.7, "S2": 0.3}.get(surgeon_code, 0.5)


class _QuantileModel:
    def predict(self, df, q):
        n = len(df)
        if q <= 0.5:
            return np.full(n, 40.0)
        return np.full(n, 120.0)


class _Profiler:
    def get_sos2_knots(self, profile_id, L, U, booking):
        return np.array([L, booking, U]), np.array([L, booking, U])

    def profile_cases(self, case_df):
        return np.zeros(len(case_df), dtype=int)


class _Estimation:
    critical_ratios = _CriticalRatios()
    quantile_model = _QuantileModel()
    response_profiler = _Profiler()


def _case(case_id, surgeon, service, procedure, booked, actual):
    return CaseRecord(
        case_id=case_id,
        procedure_id=procedure,
        surgeon_code=surgeon,
        service=service,
        patient_type="ELECTIVE",
        operating_room="OR1",
        booked_duration_min=booked,
        actual_duration_min=actual,
        actual_start=pd.Timestamp("2012-01-02 08:00").to_pydatetime(),
        week_of_year=1,
        month=1,
        year=2012,
        site="TGH",
    )


def _instance():
    cases = [
        _case(1, "S1", "SvcA", "P1", 60.0, 80.0),
        _case(2, "S2", "SvcB", "P2", 90.0, 70.0),
    ]
    block = CandidateBlock(0, "TGH", "OR1", 480.0, 0.0)
    return WeeklyInstance(
        week_index=0,
        start_date=pd.Timestamp("2012-01-02").date(),
        end_date=pd.Timestamp("2012-01-08").date(),
        cases=cases,
        calendar=BlockCalendar([block]),
        case_eligible_blocks={0: [block.id], 1: [block.id]},
    )


def _model():
    model = RecommendationModel(_Estimation(), CostConfig(), w_max=10.0)
    return model


def test_prepare_instance_shapes() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    assert week.n_cases == 2
    assert week.features.shape[0] == 2
    assert week.bookings.shape == (2,)
    assert week.realized.shape == (2,)


def test_post_review_zero_weights_returns_bookings() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    w = np.zeros(model.feature_dim)
    post = model.compute_post_review(w, week)
    np.testing.assert_allclose(post, week.bookings)


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


def test_feature_includes_critical_ratio_misalignment() -> None:
    model = _model()
    week = model.prepare_instance(_instance())
    assert "misalignment_s" in model.feature_names
    idx = model.feature_names.index("misalignment_s")
    # Target ratio = 15 / (15 + 10) = 0.6, so q_hat - target is
    # 0.7 - 0.6 = 0.1 and 0.3 - 0.6 = -0.3.
    np.testing.assert_allclose(week.features[:, idx], np.array([0.1, -0.3]))
