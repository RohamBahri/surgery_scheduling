from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.methods.base import Method
from src.solvers.deterministic import solve_deterministic


class OracleMethod(Method):
    """Diagnostic lower bound (not implementable in operations)."""

    def __init__(self, config: Config) -> None:
        super().__init__(name="Oracle", config=config)

    def fit(self, df_train: pd.DataFrame) -> None:
        return None

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        durations = np.array(instance.actual_durations())
        return solve_deterministic(
            cases=instance.cases,
            durations=durations,
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            case_eligible_blocks=instance.case_eligible_blocks,
            eligibility=instance.eligibility,
            turnover=self.config.capacity.turnover_minutes,
            model_name=self.name,
        )
