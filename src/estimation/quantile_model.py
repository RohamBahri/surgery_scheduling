"""Conditional quantile model for surgery duration estimation."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.config import QuantileModelConfig
from src.core.types import Col


class ConditionalQuantileModel:
    """Fits conditional quantile surfaces Q(x, s; q) over a fixed quantile grid."""

    def __init__(self, config: QuantileModelConfig | None = None) -> None:
        self.config = config or QuantileModelConfig()
        self.numeric_features = ["_proc_median_duration", "_proc_log_volume"]
        self.categorical_features = [
            Col.SURGEON_CODE,
            Col.CASE_SERVICE,
            Col.PROCEDURE_ID,
            "_month_str",
        ]
        self.feature_columns = self.numeric_features + self.categorical_features

        self._proc_median: dict[str, float] = {}
        self._proc_count: dict[str, float] = {}
        self._global_median: float = 0.0
        self._quantile_grid = np.linspace(0.01, 0.99, self.config.q_grid_size)
        self._preprocessor: ColumnTransformer | None = None
        self._models: Dict[float, QuantileRegressor] = {}

    @property
    def is_fitted(self) -> bool:
        return self._preprocessor is not None and len(self._models) > 0

    def fit(self, df: pd.DataFrame) -> "ConditionalQuantileModel":
        self._validate_columns(df)

        proc_stats = df.groupby(Col.PROCEDURE_ID)[Col.PROCEDURE_DURATION].agg(["median", "count"])
        self._proc_median = proc_stats["median"].to_dict()
        self._proc_count = proc_stats["count"].astype(float).to_dict()
        self._global_median = float(df[Col.PROCEDURE_DURATION].median())

        X_df = self._build_features(df)
        y = df[Col.PROCEDURE_DURATION].to_numpy(dtype=float)

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ]
        )
        X = self._preprocessor.fit_transform(X_df)

        self._models = {}
        for q in self._quantile_grid:
            model = QuantileRegressor(
                quantile=float(q),
                alpha=self.config.alpha,
                solver=self.config.solver,
                solver_options={"maxiter": self.config.max_iter},
            )
            model.fit(X, y)
            self._models[float(q)] = model

        return self

    def predict(self, df: pd.DataFrame, q: float) -> np.ndarray:
        self._require_fitted()
        q_snap = self._snap_quantile(float(q))
        X = self._transform_features(df)
        preds = self._models[q_snap].predict(X)
        return np.clip(preds, 30.0, 1440.0)

    def predict_grid(
        self, df: pd.DataFrame, q_grid: np.ndarray | None = None
    ) -> Dict[float, np.ndarray]:
        self._require_fitted()
        grid = self._quantile_grid if q_grid is None else np.asarray(q_grid, dtype=float)
        X = self._transform_features(df)

        out: Dict[float, np.ndarray] = {}
        for q in grid:
            q_snap = self._snap_quantile(float(q))
            out[q_snap] = np.clip(self._models[q_snap].predict(X), 30.0, 1440.0)
        return out

    def fit_excluding(self, df: pd.DataFrame, exclude_mask: np.ndarray) -> "ConditionalQuantileModel":
        mask = np.asarray(exclude_mask, dtype=bool)
        if len(mask) != len(df):
            raise ValueError("exclude_mask length must match dataframe length")
        new_model = ConditionalQuantileModel(config=replace(self.config))
        return new_model.fit(df.loc[~mask].copy())

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        procedure_series = df[Col.PROCEDURE_ID]
        out["_proc_median_duration"] = (
            procedure_series.map(self._proc_median).fillna(self._global_median).astype(float)
        )
        out["_proc_log_volume"] = np.log1p(procedure_series.map(self._proc_count).fillna(0.0)).astype(
            float
        )
        out[Col.SURGEON_CODE] = df[Col.SURGEON_CODE].astype(str)
        out[Col.CASE_SERVICE] = df[Col.CASE_SERVICE].astype(str)
        out[Col.PROCEDURE_ID] = procedure_series.astype(str)
        out["_month_str"] = df[Col.MONTH].astype(str)
        return out

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self._validate_columns(df)
        X_df = self._build_features(df)
        return self._preprocessor.transform(X_df)  # type: ignore[union-attr]

    def _snap_quantile(self, q: float) -> float:
        idx = int(np.argmin(np.abs(self._quantile_grid - q)))
        return float(self._quantile_grid[idx])

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("ConditionalQuantileModel must be fitted before prediction")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [
            Col.PROCEDURE_ID,
            Col.PROCEDURE_DURATION,
            Col.SURGEON_CODE,
            Col.CASE_SERVICE,
            Col.MONTH,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
