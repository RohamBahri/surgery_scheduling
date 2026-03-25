import numpy as np
import pandas as pd

from src.core.config import ResponseConfig
from src.core.types import Col, Domain
from src.estimation.response import ResponseEstimator


class DummyCriticalRatioEstimator:
    def __init__(self, mapping: dict[str, float]):
        self.mapping = mapping

    def get_ratio(self, surgeon_code: str) -> float:
        return self.mapping.get(surgeon_code, self.mapping.get(Domain.OTHER, 0.5))


class MeanFoldModel:
    def __init__(self, train_mean: float):
        self.train_mean = train_mean

    def predict(self, df: pd.DataFrame, q: float) -> np.ndarray:
        return np.full(len(df), self.train_mean + 15.0 * (q - 0.5), dtype=float)


class MeanExcludingQuantileModel:
    def fit_excluding(self, df: pd.DataFrame, exclude_mask: np.ndarray):
        mask = np.asarray(exclude_mask, dtype=bool)
        train_mean = float(df.loc[~mask, Col.PROCEDURE_DURATION].mean())
        return MeanFoldModel(train_mean)


def _make_rows(surgeon: str, service: str, n: int, base: float, q: float, start_day: int):
    t0 = pd.Timestamp("2025-01-01") + pd.Timedelta(days=start_day)
    times = [t0 + pd.Timedelta(days=i) for i in range(n)]
    idx = np.arange(n, dtype=float)
    duration = base + 8.0 * np.sin(idx / 3.0)
    booked = duration + 15.0 * (q - 0.5) + 2.0 * np.cos(idx / 4.0)
    return pd.DataFrame(
        {
            Col.SURGEON_CODE: surgeon,
            Col.PROCEDURE_ID: ["P1"] * n,
            Col.CASE_SERVICE: [service] * n,
            Col.ACTUAL_START: times,
            Col.ACTUAL_STOP: [t + pd.Timedelta(hours=2) for t in times],
            Col.PROCEDURE_DURATION: duration,
            Col.BOOKED_MINUTES: booked,
        }
    )


def _training_df() -> pd.DataFrame:
    parts = [
        _make_rows("S1", "SvcA", 24, 120.0, 0.9, 0),
        _make_rows("S2", "SvcA", 22, 130.0, 0.4, 30),
        _make_rows("S3", "SvcA", 6, 118.0, 0.6, 80),
        _make_rows("S5", "SvcA", 7, 122.0, 0.7, 95),
        _make_rows("S4", "SvcB", 20, 150.0, 0.5, 120),
        _make_rows(Domain.OTHER, "SvcA", 5, 125.0, 0.5, 170),
    ]
    return pd.concat(parts, ignore_index=True)


def _fit_estimator() -> ResponseEstimator:
    q_model = MeanExcludingQuantileModel()
    critical = DummyCriticalRatioEstimator(
        {"S1": 0.9, "S2": 0.4, "S3": 0.6, "S4": 0.5, "S5": 0.7, Domain.OTHER: 0.5}
    )
    config = ResponseConfig(
        n_folds=3,
        min_pairs=10,
        h_grid_max=20.0,
        h_grid_step=5.0,
        a_min=0.01,
        a_max=1.0,
        delta_max_days=60,
    )
    return ResponseEstimator(q_model, critical, config).fit(_training_df())


def test_profile_least_squares_returns_valid_range():
    est = _fit_estimator()
    X = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    Y = np.array([-10.0, -2.0, 0.0, 3.0, 9.0])

    a, hp, hm, ssr = est._profile_least_squares(X, Y)

    assert 0.01 <= a <= 1.0
    assert hp >= 0.0
    assert hm >= 0.0
    assert ssr >= 0.0


def test_response_fit_returns_params_for_all_surgeons():
    df = _training_df()
    est = _fit_estimator()

    surgeons = set(df[Col.SURGEON_CODE].unique()) | {Domain.OTHER}
    params = est.get_all_params()
    assert surgeons.issubset(set(params["surgeon_code"]))


def test_sparse_surgeons_are_pooled():
    est = _fit_estimator()
    p3 = est.get_params("S3")
    p5 = est.get_params("S5")

    assert not p3.is_individual
    assert not p5.is_individual
    assert np.isclose(p3.a, p5.a)
    assert np.isclose(p3.h_plus, p5.h_plus)
    assert np.isclose(p3.h_minus, p5.h_minus)


def test_other_has_fallback_params():
    est = _fit_estimator()
    p_other = est.get_params(Domain.OTHER)

    assert p_other.surgeon_code == Domain.OTHER
    assert not p_other.is_individual
    assert 0.01 <= p_other.a <= 1.0


def test_get_all_params_dataframe_columns():
    est = _fit_estimator()
    df_params = est.get_all_params()

    expected_cols = {"surgeon_code", "a", "h_plus", "h_minus", "n_pairs", "is_individual"}
    assert expected_cols.issubset(set(df_params.columns))


def test_acceptance_weights_in_range():
    est = _fit_estimator()
    params = est.get_all_params()

    assert (params["a"] > 0.0).all()
    assert (params["a"] <= 1.0).all()

