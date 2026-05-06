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

    def __init__(self, quantile_model, critical_ratios, config: ResponseConfig | None = None) -> None:
        self._qmodel = quantile_model
        self._critical = critical_ratios
        self.config = config or ResponseConfig()
        self._params: dict[str, SurgeonResponseParams] = {}
        self._residualized_pairs: pd.DataFrame | None = None
        self._r_hat: np.ndarray | None = None
        self._u_hat: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return len(self._params) > 0

    def fit(self, df_train: pd.DataFrame) -> "ResponseEstimator":
        self._validate_columns(df_train)
        pairs = self._build_consecutive_pairs(df_train)
        r_hat, u_hat = self._cross_fitted_residuals(df_train)
        residualized_pairs = self._build_residualized_pairs(pairs, r_hat, u_hat)

        self._r_hat = r_hat
        self._u_hat = u_hat
        self._residualized_pairs = residualized_pairs

        surgeon_service = (
            df_train.groupby(Col.SURGEON_CODE)[Col.CASE_SERVICE]
            .agg(lambda x: x.mode().iat[0])
            .to_dict()
        )

        individual_params: dict[str, SurgeonResponseParams] = {}
        for surgeon, g in residualized_pairs.groupby("surgeon_code", sort=False):
            if surgeon == Domain.OTHER or len(g) < self.config.min_pairs:
                continue
            a, hp, hm, _ = self._profile_least_squares(
                g["X"].to_numpy(dtype=float),
                g["Y"].to_numpy(dtype=float),
            )
            individual_params[surgeon] = SurgeonResponseParams(
                surgeon_code=surgeon,
                a=a,
                h_plus=hp,
                h_minus=hm,
                n_pairs=int(len(g)),
                is_individual=True,
            )

        service_params: dict[str, tuple[float, float, float]] = {}
        for service, g in residualized_pairs.groupby("service", sort=False):
            if len(g) < self.config.min_pairs:
                continue
            a, hp, hm, _ = self._profile_least_squares(
                g["X"].to_numpy(dtype=float),
                g["Y"].to_numpy(dtype=float),
            )
            service_params[service] = (a, hp, hm)

        if len(residualized_pairs) > 0:
            g_all = residualized_pairs
            a_g, hp_g, hm_g, _ = self._profile_least_squares(
                g_all["X"].to_numpy(dtype=float),
                g_all["Y"].to_numpy(dtype=float),
            )
        else:
            a_g, hp_g, hm_g = self.config.a_min, 0.0, 0.0

        global_default = SurgeonResponseParams(
            surgeon_code=Domain.OTHER,
            a=float(a_g),
            h_plus=float(hp_g),
            h_minus=float(hm_g),
            n_pairs=int(len(residualized_pairs)),
            is_individual=False,
        )

        params: dict[str, SurgeonResponseParams] = {}
        for surgeon in df_train[Col.SURGEON_CODE].astype(str).unique():
            if surgeon in individual_params:
                params[surgeon] = individual_params[surgeon]
                continue

            service = surgeon_service.get(surgeon, "")
            pooled = service_params.get(service, (global_default.a, global_default.h_plus, global_default.h_minus))
            n_pairs = int((residualized_pairs["surgeon_code"] == surgeon).sum()) if len(residualized_pairs) else 0
            params[surgeon] = SurgeonResponseParams(
                surgeon_code=surgeon,
                a=float(pooled[0]),
                h_plus=float(pooled[1]),
                h_minus=float(pooled[2]),
                n_pairs=n_pairs,
                is_individual=False,
            )

        params[Domain.OTHER] = global_default
        self._params = params
        return self

    def get_params(self, surgeon_code: str) -> SurgeonResponseParams:
        self._require_fitted()
        return self._params.get(surgeon_code, self._params[Domain.OTHER])

    def get_all_params(self) -> pd.DataFrame:
        self._require_fitted()
        rows = [
            {
                "surgeon_code": p.surgeon_code,
                "a": p.a,
                "h_plus": p.h_plus,
                "h_minus": p.h_minus,
                "n_pairs": p.n_pairs,
                "is_individual": p.is_individual,
            }
            for p in self._params.values()
        ]
        return pd.DataFrame(rows).sort_values("surgeon_code").reset_index(drop=True)

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
            q_vals = np.array([self._get_ratio(s) for s in surgeon_codes], dtype=float)

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

    def _build_residualized_pairs(
        self, pairs: pd.DataFrame, r_hat: np.ndarray, u_hat: np.ndarray
    ) -> pd.DataFrame:
        if len(pairs) == 0:
            return pd.DataFrame(columns=["surgeon_code", "service", "X", "Y"])

        curr_idx = pairs["curr_idx"].to_numpy(dtype=int)
        prev_idx = pairs["prev_idx"].to_numpy(dtype=int)

        out = pd.DataFrame(
            {
                "surgeon_code": pairs["surgeon_code"].to_numpy(),
                "service": pairs["service"].to_numpy(),
                "X": u_hat[prev_idx],
                "Y": r_hat[curr_idx] - r_hat[prev_idx],
            }
        )
        return out.dropna().reset_index(drop=True)

    def _profile_least_squares(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float, float, float]:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        h_grid = np.arange(0.0, self.config.h_grid_max + self.config.h_grid_step, self.config.h_grid_step)
        best = (self.config.a_min, 0.0, 0.0, np.inf)

        for hp in h_grid:
            for hm in h_grid:
                Z = np.where(X < -hm, X + hm, np.where(X > hp, X - hp, 0.0))
                denom = float(np.dot(Z, Z))
                if denom <= 1e-12:
                    a = self.config.a_min
                else:
                    a = float(np.dot(Y, Z) / denom)
                a = float(np.clip(a, self.config.a_min, self.config.a_max))
                resid = Y - a * Z
                ssr = float(np.dot(resid, resid))
                if ssr < best[3]:
                    best = (a, float(hp), float(hm), ssr)

        return best

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

    def _get_ratio(self, surgeon_code: str) -> float:
        if hasattr(self._critical, "get_ratio"):
            return float(self._critical.get_ratio(surgeon_code))
        if isinstance(self._critical, dict):
            return float(self._critical.get(surgeon_code, self._critical.get(Domain.OTHER, 0.5)))
        raise TypeError("critical_ratios must provide get_ratio or be a dict")

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("ResponseEstimator must be fitted before access")
