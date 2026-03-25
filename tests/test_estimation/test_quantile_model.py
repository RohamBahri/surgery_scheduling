import numpy as np
import pandas as pd

from src.core.config import QuantileModelConfig
from src.core.types import Col
from src.estimation.quantile_model import ConditionalQuantileModel


def _synthetic_df(n: int = 220, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    procedures = np.array(["P1", "P2", "P3", "P4"])
    surgeons = np.array(["S1", "S2", "S3"])
    services = np.array(["SvcA", "SvcB"])

    proc = rng.choice(procedures, size=n, p=[0.4, 0.3, 0.2, 0.1])
    surg = rng.choice(surgeons, size=n)
    svc = rng.choice(services, size=n)
    month = rng.integers(1, 13, size=n)

    proc_effect = {"P1": 70, "P2": 110, "P3": 150, "P4": 220}
    surg_effect = {"S1": -8, "S2": 0, "S3": 12}
    svc_effect = {"SvcA": 0, "SvcB": 10}

    duration = np.array([
        proc_effect[p] + surg_effect[s] + svc_effect[v] + rng.normal(0, 18)
        for p, s, v in zip(proc, surg, svc)
    ])
    duration = np.clip(duration, 30, 500)

    return pd.DataFrame(
        {
            Col.PROCEDURE_ID: proc,
            Col.SURGEON_CODE: surg,
            Col.CASE_SERVICE: svc,
            Col.MONTH: month,
            Col.PROCEDURE_DURATION: duration,
            Col.BOOKED_MINUTES: duration + rng.normal(0, 15, size=n),
        }
    )


def test_fit_and_predict_shapes():
    df = _synthetic_df()
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=15)).fit(df)
    preds = model.predict(df.head(25), q=0.5)

    assert model.is_fitted
    assert preds.shape == (25,)
    assert np.isfinite(preds).all()


def test_predict_grid_keys():
    df = _synthetic_df()
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=9)).fit(df)
    grid = np.array([0.1, 0.5, 0.9])
    pred_map = model.predict_grid(df.head(10), q_grid=grid)

    expected_keys = {model._snap_quantile(q) for q in grid}
    assert set(pred_map.keys()) == expected_keys
    for arr in pred_map.values():
        assert arr.shape == (10,)


def test_fit_excluding_returns_new_model():
    df = _synthetic_df()
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=7)).fit(df)
    original_models_id = id(model._models)

    exclude_mask = np.zeros(len(df), dtype=bool)
    exclude_mask[:20] = True
    new_model = model.fit_excluding(df, exclude_mask)

    assert new_model is not model
    assert new_model.is_fitted
    assert model.is_fitted
    assert id(model._models) == original_models_id


def test_no_booked_time_in_features():
    df = _synthetic_df()
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=5)).fit(df)

    names = model._preprocessor.get_feature_names_out().tolist()
    assert all("booked" not in name.lower() for name in names)
    assert Col.BOOKED_MINUTES not in model.feature_columns


def test_predictions_are_clipped():
    df = _synthetic_df()
    df.loc[:15, Col.PROCEDURE_DURATION] = 5000.0
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=11)).fit(df)

    preds = model.predict(df, q=0.99)
    assert (preds >= 30.0).all()
    assert (preds <= 1440.0).all()


def test_quantile_monotonicity_on_sample():
    df = _synthetic_df(n=260)
    model = ConditionalQuantileModel(QuantileModelConfig(q_grid_size=21)).fit(df)

    sample = df.sample(40, random_state=2)
    p10 = model.predict(sample, q=0.10)
    p50 = model.predict(sample, q=0.50)
    p90 = model.predict(sample, q=0.90)

    gross_violations = np.mean((p50 < p10 - 5.0) | (p90 < p50 - 5.0))
    assert gross_violations < 0.2
