"""Data loading and cleaning for the UHN surgical dataset.

Reads the raw Excel file, normalises column names, filters to elective
completed OR cases, computes durations, recodes rare categories, and returns
a tidy DataFrame sorted by ``actual_start``.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import Col, Domain

logger = logging.getLogger(__name__)

# Columns we ask the Excel reader to load (original header names).
_EXCEL_COLUMNS = [
    "Patient_Type", "Case_Service", "Main_Procedure", "Main_Procedure_Id",
    "Operating_Room", "Booked Time (Minutes)",
    "Enter Room Date", "Enter Room Time",
    "Actual Start Date", "Actual Start Time",
    "Actual Stop Date", "Actual Stop Time",
    "Leave Room Date", "Leave Room Time",
    "Patient_ID", "Surgeon", "Surgeon_Code",
]


def load_data(config: Config) -> pd.DataFrame:
    """Read the raw Excel file and return a cleaned elective-surgery DataFrame.

    The returned DataFrame uses canonical snake_case column names defined in
    :class:`Col` and is sorted by ``actual_start``.

    Raises
    ------
    SystemExit
        If the Excel file cannot be found or read.
    """
    path = config.data.excel_file_path
    logger.info("Loading data from %s", path)

    try:
        df = pd.read_excel(path, usecols=_EXCEL_COLUMNS)
    except FileNotFoundError:
        sys.exit(f"Data file not found: {path}")
    except Exception as exc:
        sys.exit(f"Error reading data file: {exc}")

    # ── Normalise column names to snake_case ─────────────────────────────
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )

    # ── Row-level filters ────────────────────────────────────────────────
    df = df[df[Col.ACTUAL_START_DATE].notna()]

    # Keep only actual OR rooms (prefix "OR"), exclude emergency-designated rooms
    room_upper = df[Col.OPERATING_ROOM].fillna("").astype(str).str.upper()
    is_or = room_upper.str.startswith(Domain.OR_ROOM_PREFIX)
    is_emergency_room = room_upper.isin(
        [r.upper() for r in Domain.EMERGENCY_ROOMS])
    df = df[is_or & ~is_emergency_room]

    # Exclude emergency patients
    df = df[df[Col.PATIENT_TYPE] != Domain.EMERGENCY_PATIENT]

    # ── Parse timestamps ─────────────────────────────────────────────────
    df[Col.ACTUAL_START] = _combine_date_time(df, Col.ACTUAL_START_DATE,
                                               Col.ACTUAL_START_TIME)
    df[Col.ACTUAL_STOP] = _combine_date_time(df, Col.ACTUAL_STOP_DATE,
                                              Col.ACTUAL_STOP_TIME)
    df[Col.ENTER_ROOM] = _combine_date_time(df, Col.ENTER_ROOM_DATE,
                                             Col.ENTER_ROOM_TIME)
    df[Col.LEAVE_ROOM] = _combine_date_time(df, Col.LEAVE_ROOM_DATE,
                                             Col.LEAVE_ROOM_TIME)

    # Drop the raw date / time string columns
    _date_time_cols = [
        Col.ACTUAL_START_DATE, Col.ACTUAL_START_TIME,
        Col.ACTUAL_STOP_DATE, Col.ACTUAL_STOP_TIME,
        Col.ENTER_ROOM_DATE, Col.ENTER_ROOM_TIME,
        Col.LEAVE_ROOM_DATE, Col.LEAVE_ROOM_TIME,
    ]
    df = df.drop(columns=[c for c in _date_time_cols if c in df.columns])

    # ── Compute durations ────────────────────────────────────────────────
    df[Col.PROCEDURE_DURATION] = (
        (df[Col.ACTUAL_STOP] - df[Col.ACTUAL_START])
        .dt.total_seconds() / 60.0
    )
    df[Col.PREPARATION_DURATION] = (
        (df[Col.ACTUAL_START] - df[Col.ENTER_ROOM])
        .dt.total_seconds().clip(lower=0) / 60.0
    )

    # Drop cases with missing or non-positive procedure durations
    df = df[df[Col.PROCEDURE_DURATION] >= Domain.MIN_PROCEDURE_DURATION]

    # ── Recode rare categories ───────────────────────────────────────────
    df[Col.PROCEDURE_ID] = _recode_rare(
        df[Col.PROCEDURE_ID], config.data.min_samples_procedure)
    df[Col.SURGEON_CODE] = _recode_rare(
        df[Col.SURGEON_CODE], config.data.min_samples_surgeon)
    df[Col.CASE_SERVICE] = _recode_rare(
        df[Col.CASE_SERVICE], config.data.min_samples_service)

    # ── Calendar features ────────────────────────────────────────────────
    df = add_time_features(df)

    # ── Final sort and reset ─────────────────────────────────────────────
    df = df.sort_values(Col.ACTUAL_START).reset_index(drop=True)
    logger.info("Loaded %d elective cases after cleaning.", len(df))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``week_of_year``, ``month``, ``year`` columns from ``actual_start``."""
    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
    df[Col.WEEK_OF_YEAR] = dt.dt.isocalendar().week.astype(int)
    df[Col.MONTH] = dt.dt.month.astype(int)
    df[Col.YEAR] = dt.dt.year.astype(int)
    return df


# ── Private helpers ──────────────────────────────────────────────────────────

def _combine_date_time(df: pd.DataFrame, date_col: str,
                       time_col: str) -> pd.Series:
    """Merge a date column and a time-string column into a datetime Series."""
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    times = df[time_col].fillna("00:00:00").astype(str).str.strip()
    times = times.str.split(".").str[0]           # drop fractional seconds

    valid = dates.notna()
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if valid.any():
        result[valid] = pd.to_datetime(
            dates[valid].dt.strftime("%Y-%m-%d") + " " + times[valid],
            errors="coerce",
        )
    return result


def _recode_rare(series: pd.Series, threshold: int) -> pd.Series:
    """Replace values that appear fewer than *threshold* times with 'Other'."""
    series = series.astype(object)
    counts = series.value_counts()
    rare = counts[counts < threshold].index
    series = series.copy()
    series[series.isin(rare)] = Domain.OTHER
    return series
