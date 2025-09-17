import numpy as np
import pandas as pd
import datetime as dt
import enum
from typing import Any


def to_json_safe(obj: Any) -> Any:
    # ─ numbers & enums ─
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, enum.Enum):
        return int(obj.value)

    # ─ time ─
    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)):
        return obj.isoformat()

    # ─ NumPy arrays (0-D or higher) ─
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())  # .tolist() may return scalar or nested list

    # ─ data frames / series ─
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict(orient="records")

    # ─ generic containers ─
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]

    # ─ already JSON-friendly ─
    return obj
