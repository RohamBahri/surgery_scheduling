"""Response-model data preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from src.core.config import ResponseConfig
from src.core.types import Col, Domain


@dataclass
class SurgeonResponseParams:
    surgeon_code: str
    a: float
    h_plus: float
    h_minus: float
    n_pairs: int
    is_individual: bool


class ResponseEstimator:
    """Prepare residualized pair data for surgeon response estimation."""

    def __init__(self, quantile_model, critical_ratio_estimator, config: ResponseConfig | None = None) -> None:
        self._qmodel = quantile_model
        self._critical = critical_ratio_estimator
        self.config = config or ResponseConfig()

    def _build_consecutive_pairs(self, df_train: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df_train)
        df = df_train.copy()
        df["_row_pos"] = np.arange(len(df), dtype=int)
        df["_completion_time"] = self._completion_time(df)

        sort_cols = [Col.SURGEON_CODE, Col.PROCEDURE_ID, "_completion_time"]
        df_sorted = df.sort_values(sort_cols)

        rows = []
        for (_, _), g in df_sorted.groupby([Col.SURGEON_CODE, Col.PROCEDURE_ID], sort=False):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                prev = g.iloc[i - 1]
                curr = g.iloc[i]
                if prev[Col.SURGEON_CODE] == Domain.OTHER or prev[Col.PROCEDURE_ID] == Domain.OTHER:
                    continue
                gap_days = (curr["_completion_time"] - prev["_completion_time"]).total_seconds() / 86400.0
                if gap_days < 0 or gap_days > self.config.delta_max_days:
                    continue
                rows.append(
                    {
                        "surgeon_code": curr[Col.SURGEON_CODE],
                        "service": curr[Col.CASE_SERVICE],
                        "curr_idx": int(curr["_row_pos"]),
                        "prev_idx": int(prev["_row_pos"]),
                        "gap_days": float(gap_days),
                    }
                )

        return pd.DataFrame(rows, columns=["surgeon_code", "service", "curr_idx", "prev_idx", "gap_days"])

    def _assign_case_folds(self, df_train: pd.DataFrame) -> np.ndarray:
        self._validate_columns(df_train)
        k = self.config.n_folds
        if k <= 0:
            raise ValueError("n_folds must be positive")

        df = df_train.copy()
        df["_row_pos"] = np.arange(len(df), dtype=int)
        df["_completion_time"] = self._completion_time(df)
        folds = np.full(len(df), -1, dtype=int)

        for _, g in df.sort_values([Col.SURGEON_CODE, "_completion_time"]).groupby(Col.SURGEON_CODE, sort=False):
            row_pos = g["_row_pos"].to_numpy(dtype=int)
            assigned = np.arange(len(g), dtype=int) % k
            folds[row_pos] = assigned

        if np.any(folds < 0):
            raise RuntimeError("Fold assignment failed for one or more rows")
        return folds

    def _cross_fitted_residuals(self, df_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._validate_columns(df_train)
        n = len(df_train)
        r_hat = np.full(n, np.nan, dtype=float)
        u_hat = np.full(n, np.nan, dtype=float)

        folds = self._assign_case_folds(df_train)
        for fold_id in range(self.config.n_folds):
            test_mask = folds == fold_id
            if not np.any(test_mask):
                continue

            qm_fold = self._qmodel.fit_excluding(df_train, test_mask)
            test_df = df_train.loc[test_mask]
            test_positions = np.flatnonzero(test_mask)

            surgeon_codes = test_df[Col.SURGEON_CODE].astype(str).to_numpy()
            q_vals = np.array([self._critical.get_ratio(s) for s in surgeon_codes], dtype=float)

            preds = np.full(len(test_df), np.nan, dtype=float)
            for q in np.unique(q_vals):
                local_mask = q_vals == q
                pred_local = qm_fold.predict(test_df.loc[local_mask], float(q))
                preds[local_mask] = pred_local

            r_hat[test_positions] = test_df[Col.BOOKED_MINUTES].to_numpy(dtype=float) - preds
            u_hat[test_positions] = test_df[Col.PROCEDURE_DURATION].to_numpy(dtype=float) - preds

        if np.isnan(r_hat).any() or np.isnan(u_hat).any():
            warnings.warn("NaNs remain in cross-fitted residuals.", RuntimeWarning)

        return r_hat, u_hat

    def _completion_time(self, df: pd.DataFrame) -> pd.Series:
        stop = pd.to_datetime(df[Col.ACTUAL_STOP], errors="coerce")
        start = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
        return stop.fillna(start)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [
            Col.SURGEON_CODE,
            Col.PROCEDURE_ID,
            Col.CASE_SERVICE,
            Col.ACTUAL_START,
            Col.ACTUAL_STOP,
            Col.BOOKED_MINUTES,
            Col.PROCEDURE_DURATION,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
