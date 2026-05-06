from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.types import Col


@pytest.fixture
def synthetic_df_train() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 180

    surgeons = np.array(["S1", "S2", "S3"])
    services = {"S1": "SvcA", "S2": "SvcA", "S3": "SvcB"}
    procedures = np.array(["P1", "P2", "P3", "P4"])

    surgeon_col = rng.choice(surgeons, size=n, p=[0.4, 0.35, 0.25])
    proc_col = rng.choice(procedures, size=n)
    service_col = np.array([services[s] for s in surgeon_col])

    proc_base = {"P1": 90.0, "P2": 120.0, "P3": 150.0, "P4": 200.0}
    surg_bias = {"S1": +12.0, "S2": -8.0, "S3": +5.0}
    book_bias = {"S1": +10.0, "S2": -15.0, "S3": +4.0}

    start0 = pd.Timestamp("2024-01-01")
    day_offsets = np.sort(rng.integers(0, 220, size=n))
    actual_start = pd.to_datetime(start0 + pd.to_timedelta(day_offsets, unit="D"))

    duration = np.array([
        proc_base[p] + surg_bias[s] + rng.normal(0, 15)
        for p, s in zip(proc_col, surgeon_col)
    ])
    duration = np.clip(duration, 30, 400)

    booked = np.array([
        d + book_bias[s] + rng.normal(0, 8)
        for d, s in zip(duration, surgeon_col)
    ])
    booked = np.clip(booked, 30, 500)

    df = pd.DataFrame(
        {
            Col.CASE_UID: np.arange(n),
            Col.SURGEON_CODE: surgeon_col,
            Col.CASE_SERVICE: service_col,
            Col.PROCEDURE_ID: proc_col,
            Col.BOOKED_MINUTES: booked,
            Col.PROCEDURE_DURATION: duration,
            Col.ACTUAL_START: actual_start,
            Col.ACTUAL_STOP: actual_start + pd.to_timedelta(duration, unit="m"),
            Col.SITE: rng.choice(["TGH", "TWH"], size=n),
            Col.WEEK_OF_YEAR: pd.Series(actual_start).dt.isocalendar().week.astype(int).to_numpy(),
            Col.MONTH: pd.Series(actual_start).dt.month.to_numpy(),
            Col.YEAR: pd.Series(actual_start).dt.year.to_numpy(),
        }
    )
    return df.reset_index(drop=True)
