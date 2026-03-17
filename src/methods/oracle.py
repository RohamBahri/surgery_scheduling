"""Oracle (clairvoyant) baseline.

Schedules using realized surgery durations — information that is NOT
available at planning time.  This produces a lower bound on achievable
cost: no implementable method can beat the oracle.

Used as the denominator in regret calculations (paper Section 4.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.methods.base import Method
from src.solvers.deterministic import solve_deterministic


class OracleMethod(Method):
    """Schedule using perfect-information (actual) durations."""

    def __init__(self, config: Config) -> None:
        super().__init__(name="Oracle", config=config)

    def fit(self, df_train: pd.DataFrame) -> None:
        pass  # nothing to learn

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        durations = np.array(instance.actual_durations())
        return solve_deterministic(
            cases=instance.cases,
            durations=durations,
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            model_name="Oracle",
        )
