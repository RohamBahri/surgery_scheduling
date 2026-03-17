"""Booked-time baseline.

Uses the surgeon's booked duration as the planning input.  This is the
status-quo benchmark: the planner trusts the booking without correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.methods.base import Method
from src.solvers.deterministic import solve_deterministic


class BookedTimeMethod(Method):
    """Schedule using raw booked durations."""

    def __init__(self, config: Config) -> None:
        super().__init__(name="Booked", config=config)

    def fit(self, df_train: pd.DataFrame) -> None:
        pass  # nothing to learn

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        durations = np.array(instance.booked_durations())
        return solve_deterministic(
            cases=instance.cases,
            durations=durations,
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            model_name="Booked",
        )
