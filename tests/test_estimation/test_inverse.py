import numpy as np
import pandas as pd

from src.core.config import InverseConfig
from src.core.types import Col, Domain
from src.estimation.inverse import CriticalRatioEstimator


class FakeQuantileModel:
    def __init__(self) -> None:
        self.grid = np.array([0.1, 0.5, 0.9])

    def predict_grid(self, df: pd.DataFrame, q_grid=None):
        grid = self.grid if q_grid is None else np.array(q_grid)
        base = df["_base"].to_numpy(dtype=float)
        return {float(q): base + 100.0 * (float(q) - 0.5) for q in grid}


def _make_surgeon_rows(surgeon: str, service: str, n: int, q_true: float, base: float, seed: int):
    rng = np.random.default_rng(seed)
    base_vals = base + rng.normal(0.0, 3.0, size=n)
    booked = base_vals + 100.0 * (q_true - 0.5)
    return pd.DataFrame(
        {
            Col.SURGEON_CODE: surgeon,
            Col.CASE_SERVICE: service,
            Col.BOOKED_MINUTES: booked,
            "_base": base_vals,
        }
    )


def _fixture_df() -> pd.DataFrame:
    parts = [
        _make_surgeon_rows("A", "X", 60, 0.9, 120.0, 1),  # rich
        _make_surgeon_rows("B", "X", 55, 0.5, 110.0, 2),  # rich
        _make_surgeon_rows("C", "X", 12, 0.1, 105.0, 3),  # sparse >=10 pooled blend
        _make_surgeon_rows("D", "Y", 8, 0.9, 130.0, 4),   # sparse <10 -> service/grand
        _make_surgeon_rows(Domain.OTHER, "X", 6, 0.5, 115.0, 5),
    ]
    return pd.concat(parts, ignore_index=True)


def _fit_estimator() -> CriticalRatioEstimator:
    est = CriticalRatioEstimator(
        quantile_model=FakeQuantileModel(),
        config=InverseConfig(n_min=50, pooling_lambda=50.0),
    )
    return est.fit(_fixture_df())


def test_all_ratios_returned():
    df = _fixture_df()
    est = _fit_estimator()
    ratios = est.get_all_ratios()

    expected = set(df[Col.SURGEON_CODE].unique()) | {Domain.OTHER}
    assert expected.issubset(set(ratios.keys()))


def test_ratios_in_unit_interval():
    est = _fit_estimator()
    ratios = est.get_all_ratios()
    assert all(0.01 <= q <= 0.99 for q in ratios.values())


def test_unknown_surgeon_falls_back_to_other():
    est = _fit_estimator()
    assert est.get_ratio("not-seen") == est.get_ratio(Domain.OTHER)


def test_partial_pooling_for_sparse_surgeon():
    est = _fit_estimator()

    q_a = est.get_ratio("A")  # rich, should be individual near 0.9
    q_b = est.get_ratio("B")  # rich, should be individual near 0.5
    q_c = est.get_ratio("C")  # sparse pooled blend
    q_d = est.get_ratio("D")  # sparse < 10 -> target

    assert np.isclose(q_a, 0.9)
    assert np.isclose(q_b, 0.5)
    assert 0.1 < q_c < 0.9
    assert q_d > 0.5


def test_misalignment_formula():
    est = _fit_estimator()
    q_a = est.get_ratio("A")

    co, cu = 2.0, 3.0
    expected = q_a - (co / (co + cu))
    assert np.isclose(est.get_misalignment("A", co=co, cu=cu), expected)
