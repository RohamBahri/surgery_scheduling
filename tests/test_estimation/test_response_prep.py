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


class SpyFoldModel:
    def __init__(self, train_mean: float):
        self.train_mean = train_mean

    def predict(self, df: pd.DataFrame, q: float) -> np.ndarray:
        return np.full(len(df), self.train_mean + q, dtype=float)


class SpyQuantileModel:
    def __init__(self):
        self.exclude_masks: list[np.ndarray] = []

    def fit_excluding(self, df: pd.DataFrame, exclude_mask: np.ndarray):
        mask = np.asarray(exclude_mask, dtype=bool)
        self.exclude_masks.append(mask.copy())
        train_mean = float(df.loc[~mask, Col.PROCEDURE_DURATION].mean())
        return SpyFoldModel(train_mean)


def _make_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            Col.SURGEON_CODE: ["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", Domain.OTHER, "S1", "S2"],
            Col.PROCEDURE_ID: ["P1", "P1", "P2", "P1", "P1", Domain.OTHER, "P1", "P1", "P2", "P1", "P1", "P1"],
            Col.CASE_SERVICE: ["SvcA", "SvcA", "SvcA", "SvcB", "SvcB", "SvcB", "SvcC", "SvcC", "SvcC", "SvcA", "SvcA", "SvcB"],
            Col.ACTUAL_START: times,
            Col.ACTUAL_STOP: times + pd.to_timedelta(2, unit="h"),
            Col.BOOKED_MINUTES: np.linspace(60, 180, 12),
            Col.PROCEDURE_DURATION: np.linspace(55, 170, 12),
        }
    )


def _make_estimator(n_folds: int = 3):
    qmodel = SpyQuantileModel()
    critical = DummyCriticalRatioEstimator({"S1": 0.9, "S2": 0.5, "S3": 0.1, Domain.OTHER: 0.5})
    est = ResponseEstimator(qmodel, critical, ResponseConfig(n_folds=n_folds, delta_max_days=60))
    return est, qmodel


def test_pair_builder_creates_expected_columns():
    df = _make_df()
    est, _ = _make_estimator()
    pairs = est._build_consecutive_pairs(df)

    assert list(pairs.columns) == ["surgeon_code", "service", "curr_idx", "prev_idx", "gap_days"]
    assert len(pairs) > 0


def test_pair_builder_excludes_other():
    df = _make_df()
    est, _ = _make_estimator()
    pairs = est._build_consecutive_pairs(df)

    assert (pairs["surgeon_code"] != Domain.OTHER).all()


def test_fold_assignment_complete():
    df = _make_df()
    est, _ = _make_estimator(n_folds=4)
    folds = est._assign_case_folds(df)

    assert folds.shape == (len(df),)
    assert set(folds.tolist()).issubset({0, 1, 2, 3})
    assert np.all(folds >= 0)


def test_cross_fitted_residuals_shape():
    df = _make_df()
    est, _ = _make_estimator(n_folds=3)
    r_hat, u_hat = est._cross_fitted_residuals(df)

    assert r_hat.shape == (len(df),)
    assert u_hat.shape == (len(df),)


def test_cross_fitted_residuals_have_few_or_no_nans():
    df = _make_df()
    est, _ = _make_estimator(n_folds=3)
    r_hat, u_hat = est._cross_fitted_residuals(df)

    assert np.isnan(r_hat).sum() == 0
    assert np.isnan(u_hat).sum() == 0


def test_each_case_predicted_out_of_fold():
    df = _make_df()
    est, qmodel = _make_estimator(n_folds=3)

    folds = est._assign_case_folds(df)
    r_hat, _ = est._cross_fitted_residuals(df)

    assert len(qmodel.exclude_masks) == 3
    for fold_id, mask in enumerate(qmodel.exclude_masks):
        expected_mask = folds == fold_id
        assert np.array_equal(mask, expected_mask)

    pred = df[Col.BOOKED_MINUTES].to_numpy(dtype=float) - r_hat
    for i in range(len(df)):
        fold_id = folds[i]
        train_mean = float(df.loc[folds != fold_id, Col.PROCEDURE_DURATION].mean())
        q = est._critical.get_ratio(df.loc[i, Col.SURGEON_CODE])
        assert np.isclose(pred[i], train_mean + q)
