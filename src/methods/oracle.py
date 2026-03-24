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
from src.data.eligibility import EligibilityMaps, build_eligibility_maps
from src.methods.base import Method
from src.methods.booked import _build_case_eligibility
from src.solvers.deterministic import solve_deterministic


class OracleMethod(Method):
    """Schedule using perfect-information (actual) durations."""

    def __init__(self, config: Config) -> None:
        super().__init__(name="Oracle", config=config)
        self._eligibility: EligibilityMaps | None = None

    def fit(self, df_train: pd.DataFrame) -> None:
        self._eligibility = build_eligibility_maps(df_train, self.config)

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        durations = np.array(instance.actual_durations())

        eligible = _build_case_eligibility(
            instance, self._eligibility, self.config
        )

        return solve_deterministic(
            cases=instance.cases,
            durations=durations,
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            model_name="Oracle",
            eligible_blocks=eligible,
            mean_turnover=self.config.capacity.mean_turnover_minutes,
        )