"""Abstract base class for scheduling methods.

Every scheduling method — from a trivial booked-time baseline to the full
behavioral CCG — implements this interface.  The experiment runner calls
only :meth:`fit` and :meth:`plan`; it never knows method internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.core.config import Config
from src.core.types import ScheduleResult, WeeklyInstance


class Method(ABC):
    """Base class for all scheduling methods."""

    def __init__(self, name: str, config: Config) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> None:
        """Learn any required artifacts from training (warm-up) data.

        For simple baselines this may be a no-op.  For the paper method it
        includes inverse preference recovery, response estimation, and
        pre-computation of training-week columns.

        Parameters
        ----------
        df_train : DataFrame
            The warm-up (training) portion of the dataset.
        """

    @abstractmethod
    def plan(self, instance: WeeklyInstance) -> ScheduleResult:
        """Produce a schedule for one planning horizon.

        The method may read ``instance.cases[i].booked_duration_min`` and
        any booking-time features, but must NOT read
        ``instance.cases[i].actual_duration_min``.

        Parameters
        ----------
        instance : WeeklyInstance
            The candidate cases and block calendar for the week.

        Returns
        -------
        ScheduleResult
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
