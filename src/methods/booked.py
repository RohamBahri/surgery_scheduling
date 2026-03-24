"""Booked-time baseline.

Uses the surgeon's booked duration as the planning input.  This is the
status-quo benchmark: the planner trusts the booking without correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance
from src.data.eligibility import EligibilityMaps, build_eligibility_maps
from src.methods.base import Method
from src.solvers.deterministic import solve_deterministic


class BookedTimeMethod(Method):
    """Schedule using raw booked durations."""

    def __init__(self, config: Config) -> None:
        super().__init__(name="Booked", config=config)
        self._eligibility: EligibilityMaps | None = None

    def fit(self, df_train: pd.DataFrame) -> None:
        self._eligibility = build_eligibility_maps(df_train, self.config)

    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        durations = np.array(instance.booked_durations())

        # Build per-case eligible block sets
        eligible = _build_case_eligibility(
            instance, self._eligibility, self.config
        )

        return solve_deterministic(
            cases=instance.cases,
            durations=durations,
            calendar=instance.calendar,
            costs=self.config.costs,
            solver_cfg=self.config.solver,
            model_name="Booked",
            eligible_blocks=eligible,
            mean_turnover=self.config.capacity.mean_turnover_minutes,
        )


def _build_case_eligibility(instance, eligibility, config):
    """Build the eligible_blocks dict for all cases in an instance."""
    if eligibility is None:
        return None

    from src.core.types import BlockId

    eligible = {}
    for i, case in enumerate(instance.cases):
        allowed_rooms = eligibility.eligible_rooms_for_case(
            service=case.service,
            surgeon_code=case.surgeon_code,
            operating_room=case.operating_room,
            config=config,
        )
        if allowed_rooms is not None:
            eligible[i] = [
                b.id for b in instance.calendar.candidates
                if b.room in allowed_rooms
            ]
    return eligible if eligible else None