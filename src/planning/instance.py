"""Weekly instance construction.

Converts a raw DataFrame slice for one planning horizon into a typed
:class:`WeeklyInstance` that every method receives identically.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from src.core.config import Config
from src.core.types import (
    CaseRecord,
    Col,
    Domain,
    WeeklyInstance,
)
from src.data.capacity import build_block_calendar

logger = logging.getLogger(__name__)


def build_weekly_instance(
    df_pool: pd.DataFrame,
    horizon_start: pd.Timestamp,
    week_index: int,
    config: Config,
    candidate_pools: Dict[int, List[str]],
) -> WeeklyInstance:
    """Build a :class:`WeeklyInstance` for one horizon window.

    Parameters
    ----------
    df_pool : DataFrame
        The full scheduling pool (test set).
    horizon_start : Timestamp
        First day of the horizon (normally a Monday).
    week_index : int
        Ordinal index of this horizon in the experiment.
    config : Config
        Provides ``data.horizon_days`` and capacity settings.
    candidate_pools : dict
        Weekday → room list, pre-computed from training data via
        :func:`build_candidate_pools`.

    Returns
    -------
    WeeklyInstance
    """
    horizon_days = config.data.horizon_days
    start = horizon_start.normalize()
    end = start + pd.Timedelta(days=horizon_days - 1)

    start_date = start.date()
    end_date = end.date()

    # Filter to cases in the horizon window
    actual = pd.to_datetime(df_pool[Col.ACTUAL_START])
    mask = (actual.dt.normalize().dt.date >= start_date) & (
        actual.dt.normalize().dt.date <= end_date
    )
    df_week = df_pool[mask]

    # Build block calendar from pre-computed candidate pools
    calendar = build_block_calendar(candidate_pools, start, config)

    # Convert rows to CaseRecord objects
    cases = _dataframe_to_cases(df_week)

    logger.info(
        "Week %d (%s – %s): %d cases, %d candidate blocks",
        week_index, start_date, end_date,
        len(cases), calendar.total_candidates,
    )
    return WeeklyInstance(
        week_index=week_index,
        start_date=start_date,
        end_date=end_date,
        cases=cases,
        calendar=calendar,
    )


def _dataframe_to_cases(df: pd.DataFrame) -> list[CaseRecord]:
    """Convert DataFrame rows to a list of CaseRecord objects."""
    records: list[CaseRecord] = []
    for idx, row in df.iterrows():
        ts = pd.to_datetime(row[Col.ACTUAL_START])
        records.append(
            CaseRecord(
                case_id=int(idx),
                procedure_id=str(row.get(Col.PROCEDURE_ID, Domain.UNKNOWN)),
                surgeon_code=str(row.get(Col.SURGEON_CODE, Domain.UNKNOWN)),
                service=str(row.get(Col.CASE_SERVICE, Domain.UNKNOWN)),
                patient_type=str(row.get(Col.PATIENT_TYPE, Domain.UNKNOWN)),
                operating_room=str(row.get(Col.OPERATING_ROOM, "")),
                booked_duration_min=float(row[Col.BOOKED_MINUTES]),
                actual_duration_min=float(row[Col.PROCEDURE_DURATION]),
                actual_start=ts.to_pydatetime(),
                week_of_year=int(ts.isocalendar().week),
                month=ts.month,
                year=ts.year,
            )
        )
    return records
