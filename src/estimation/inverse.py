"""Inverse critical-ratio estimation by quantile matching."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.core.config import InverseConfig
from src.core.types import Col, Domain


class CriticalRatioEstimator:
    """Estimate surgeon-specific critical ratios by inverse quantile matching."""

    def __init__(self, quantile_model, config: InverseConfig | None = None) -> None:
        self.quantile_model = quantile_model
        self.config = config or InverseConfig()
        self._ratios: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return len(self._ratios) > 0

    def fit(self, df_train: pd.DataFrame) -> "CriticalRatioEstimator":
        required = [Col.SURGEON_CODE, Col.CASE_SERVICE, Col.BOOKED_MINUTES]
        missing = [c for c in required if c not in df_train.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df_train.copy()
        grouped = df.groupby(Col.SURGEON_CODE)
        surgeon_counts = grouped.size().to_dict()
        surgeon_service = grouped[Col.CASE_SERVICE].agg(lambda x: x.mode().iat[0]).to_dict()

        individual_q: Dict[str, float] = {}
        for surgeon, g in grouped:
            if surgeon == Domain.OTHER:
                continue
            individual_q[surgeon] = self._estimate_individual_ratio(g)

        rich_surgeons = [
            s
            for s in individual_q
            if surgeon_counts.get(s, 0) >= self.config.n_min and s != Domain.OTHER
        ]

        service_logits: Dict[str, float] = {}
        for service, values in self._group_by_service(rich_surgeons, individual_q, surgeon_service).items():
            service_logits[service] = float(np.mean([self._logit(v) for v in values]))

        if rich_surgeons:
            grand_logit = float(np.mean([self._logit(individual_q[s]) for s in rich_surgeons]))
        else:
            grand_logit = self._logit(0.5)

        ratios: Dict[str, float] = {}
        for surgeon, n_cases in surgeon_counts.items():
            if surgeon == Domain.OTHER:
                continue

            if surgeon in rich_surgeons:
                q_hat = individual_q[surgeon]
            else:
                theta_target = service_logits.get(surgeon_service[surgeon], grand_logit)
                if n_cases < 10:
                    q_hat = self._sigmoid(theta_target)
                else:
                    theta_ind = self._logit(individual_q[surgeon])
                    w = n_cases / (n_cases + self.config.pooling_lambda)
                    q_hat = self._sigmoid(w * theta_ind + (1.0 - w) * theta_target)
            ratios[surgeon] = self._clip_ratio(q_hat)

        non_other = [q for s, q in ratios.items() if s != Domain.OTHER]
        if non_other:
            ratios[Domain.OTHER] = self._clip_ratio(float(np.median(non_other)))
        else:
            ratios[Domain.OTHER] = 0.5

        self._ratios = ratios
        return self

    def get_ratio(self, surgeon_code: str) -> float:
        self._require_fitted()
        return self._ratios.get(surgeon_code, self._ratios[Domain.OTHER])

    def get_all_ratios(self) -> Dict[str, float]:
        self._require_fitted()
        return dict(self._ratios)

    def get_misalignment(self, surgeon_code: str, co: float, cu: float) -> float:
        target = co / (co + cu)
        return self.get_ratio(surgeon_code) - target

    def _estimate_individual_ratio(self, surgeon_df: pd.DataFrame) -> float:
        pred_grid = self.quantile_model.predict_grid(surgeon_df)
        q_vals = np.array(sorted(pred_grid.keys()), dtype=float)
        booked = surgeon_df[Col.BOOKED_MINUTES].to_numpy(dtype=float)
        sse = np.array([
            np.square(booked - pred_grid[q]).sum() for q in q_vals
        ])
        best_idx = int(np.argmin(sse))
        return self._clip_ratio(float(q_vals[best_idx]))

    def _group_by_service(
        self,
        surgeons: list[str],
        individual_q: Dict[str, float],
        surgeon_service: Dict[str, str],
    ) -> Dict[str, list[float]]:
        grouped: Dict[str, list[float]] = {}
        for s in surgeons:
            svc = surgeon_service[s]
            grouped.setdefault(svc, []).append(individual_q[s])
        return grouped

    def _clip_ratio(self, q: float) -> float:
        return float(np.clip(q, 0.01, 0.99))

    def _logit(self, q: float) -> float:
        q_clip = self._clip_ratio(q)
        return float(np.log(q_clip / (1.0 - q_clip)))

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("CriticalRatioEstimator must be fitted before access")
