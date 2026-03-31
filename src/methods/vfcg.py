"""Exact VFCG method (piece 1 stub)."""

from __future__ import annotations

import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.methods.base import Method


class VFCGMethod(Method):
    def __init__(self, config: Config):
        super().__init__(name="VFCG", config=config)

    def fit(self, df_train: pd.DataFrame) -> None:
        _ = df_train
        raise NotImplementedError("VFCGMethod.fit is not implemented in piece 1.")

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        _ = instance
        raise NotImplementedError("VFCGMethod.plan is not implemented in piece 1.")
