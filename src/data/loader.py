"""Data loading and cleaning for the UHN surgical dataset."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.types import Col, Domain

logger = logging.getLogger(__name__)

_EXCEL_COLUMNS = [
    "Patient_Type", "Case_Service", "Main_Procedure", "Main_Procedure_Id",
    "Operating_Room", "Site", "Booked Time (Minutes)",
    "Enter Room Date", "Enter Room Time",
    "Actual Start Date", "Actual Start Time",
    "Actual Stop Date", "Actual Stop Time",
    "Leave Room Date", "Leave Room Time",
    "Patient_ID", "Surgeon", "Surgeon_Code",
    "Case_Cancelled_Reason", "Case Cancel Date", "Case Cancel Time",
]


def load_data(config: Config) -> pd.DataFrame:
    path = config.data.excel_file_path
    logger.info("Loading data from %s", path)
    try:
        try:
            df = pd.read_excel(path, usecols=_EXCEL_COLUMNS)
        except ValueError:
            df = pd.read_excel(path)
    except FileNotFoundError:
        sys.exit(f"Data file not found: {path}")
    except Exception as exc:
        sys.exit(f"Error reading data file: {exc}")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )

    room_compact = (
        df[Col.OPERATING_ROOM]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    is_or = room_compact.str.fullmatch(r"OR\d+")
    df = df[is_or].copy()
    df[Col.OPERATING_ROOM] = room_compact.loc[df.index]

    reason_flag = pd.Series(False, index=df.index)
    if "case_cancelled_reason" in df.columns:
        reason_text = df["case_cancelled_reason"].fillna("").astype(str).str.strip()
        reason_flag = (reason_text != "") & (reason_text.str.lower() != "nan")
    date_flag = pd.Series(False, index=df.index)
    if "case_cancel_date" in df.columns:
        date_flag = pd.to_datetime(df["case_cancel_date"], errors="coerce").notna()
    df = df[~(reason_flag | date_flag)].copy()

    patient_type = df[Col.PATIENT_TYPE].fillna("").astype(str).str.strip().str.upper()
    df = df[~patient_type.isin([Domain.EMERGENCY_PATIENT, "EMERGENCY"])].copy()

    df[Col.ACTUAL_START] = _combine_date_time(df, Col.ACTUAL_START_DATE, Col.ACTUAL_START_TIME)
    df[Col.ACTUAL_STOP] = _combine_date_time(df, Col.ACTUAL_STOP_DATE, Col.ACTUAL_STOP_TIME)
    df[Col.ENTER_ROOM] = _combine_date_time(df, Col.ENTER_ROOM_DATE, Col.ENTER_ROOM_TIME)
    df[Col.LEAVE_ROOM] = _combine_date_time(df, Col.LEAVE_ROOM_DATE, Col.LEAVE_ROOM_TIME)

    df = df.drop(columns=[
        Col.ACTUAL_START_DATE, Col.ACTUAL_START_TIME,
        Col.ACTUAL_STOP_DATE, Col.ACTUAL_STOP_TIME,
        Col.ENTER_ROOM_DATE, Col.ENTER_ROOM_TIME,
        Col.LEAVE_ROOM_DATE, Col.LEAVE_ROOM_TIME,
    ], errors="ignore")

    df[Col.SURGICAL_DURATION] = ((df[Col.ACTUAL_STOP] - df[Col.ACTUAL_START]).dt.total_seconds() / 60.0)
    df[Col.ROOM_DURATION] = ((df[Col.LEAVE_ROOM] - df[Col.ENTER_ROOM]).dt.total_seconds() / 60.0)

    has_all_four = (
        df[Col.ENTER_ROOM].notna()
        & df[Col.ACTUAL_START].notna()
        & df[Col.ACTUAL_STOP].notna()
        & df[Col.LEAVE_ROOM].notna()
    )
    df = df[has_all_four].copy()
    order_ok = (
        (df[Col.ENTER_ROOM] <= df[Col.ACTUAL_START])
        & (df[Col.ACTUAL_START] <= df[Col.ACTUAL_STOP])
        & (df[Col.ACTUAL_STOP] <= df[Col.LEAVE_ROOM])
    )
    df = df[order_ok].copy()

    positive_durations = (
        (pd.to_numeric(df[Col.BOOKED_MINUTES], errors="coerce") > 0)
        & (df[Col.ROOM_DURATION] > 0)
        & (df[Col.SURGICAL_DURATION] > 0)
    )
    df = df[positive_durations].copy()

    within_planning_limit = (
        (pd.to_numeric(df[Col.BOOKED_MINUTES], errors="coerce") <= Domain.MAX_PLANNING_CASE_MINUTES)
        & (df[Col.ROOM_DURATION] <= Domain.MAX_PLANNING_CASE_MINUTES)
        & (df[Col.SURGICAL_DURATION] <= Domain.MAX_PLANNING_CASE_MINUTES)
    )
    df = df[within_planning_limit].copy()

    df[Col.TIMESTAMP_VIOLATION] = False
    df[Col.OVERHEAD_CAPPED] = False
    df[Col.USED_ROOM_TIME] = True
    df[Col.FELL_BACK_SURGICAL] = False
    df[Col.PROCEDURE_DURATION] = df[Col.ROOM_DURATION]

    df[Col.PREPARATION_DURATION] = (
        (df[Col.ACTUAL_START] - df[Col.ENTER_ROOM]).dt.total_seconds().clip(lower=0) / 60.0
    )

    df[Col.SITE] = df.get(Col.SITE, "").fillna("").astype(str).str.strip().str.upper()
    room_site_nuniq = (
        df[df[Col.SITE] != ""]
        .groupby(Col.OPERATING_ROOM)[Col.SITE]
        .nunique()
    )
    unambiguous_rooms = set(room_site_nuniq[room_site_nuniq == 1].index)
    room_site_lookup = (
        df[(df[Col.OPERATING_ROOM].isin(unambiguous_rooms)) & (df[Col.SITE] != "")]
        .groupby(Col.OPERATING_ROOM)[Col.SITE]
        .first()
        .to_dict()
    )
    missing_site = df[Col.SITE] == ""
    df.loc[missing_site, Col.SITE] = df.loc[missing_site, Col.OPERATING_ROOM].map(room_site_lookup).fillna("")

    df[Col.PROCEDURE_ID] = _recode_rare(df[Col.PROCEDURE_ID], config.data.min_samples_procedure)
    df[Col.SURGEON_CODE] = _recode_rare(df[Col.SURGEON_CODE], config.data.min_samples_surgeon)
    df[Col.CASE_SERVICE] = _recode_rare(df[Col.CASE_SERVICE], config.data.min_samples_service)

    df = _canonicalize_identifier_columns(df)

    df = add_time_features(df)
    df = df.sort_values(Col.ACTUAL_START).reset_index(drop=True)
    df[Col.CASE_UID] = range(len(df))

    logger.info(
        "Quality: used_room_time=%d fallback=%d mild_timestamp_violations=%d overhead_capped=%d missing_site=%d",
        int(df[Col.USED_ROOM_TIME].sum()),
        int(df[Col.FELL_BACK_SURGICAL].sum()),
        int(df[Col.TIMESTAMP_VIOLATION].sum()),
        int(df[Col.OVERHEAD_CAPPED].sum()),
        int((df[Col.SITE] == "").sum()),
    )
    site_counts = df[Col.SITE].value_counts(dropna=False).to_dict()
    logger.info("Cases by site: %s", site_counts)
    logger.info("Loaded %d elective cases after cleaning.", len(df))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
    df[Col.WEEK_OF_YEAR] = dt.dt.isocalendar().week.astype(int)
    df[Col.MONTH] = dt.dt.month.astype(int)
    df[Col.YEAR] = dt.dt.year.astype(int)
    return df


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    times = (
        df[time_col]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "NaT": "", "None": "", "<NA>": ""})
        .str.split(".")
        .str[0]
    )
    valid = dates.notna() & (times != "")
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if valid.any():
        result[valid] = pd.to_datetime(
            dates[valid].dt.strftime("%Y-%m-%d") + " " + times[valid],
            errors="coerce",
        )
    return result


def _recode_rare(series: pd.Series, threshold: int) -> pd.Series:
    series = series.astype(object)
    counts = series.value_counts()
    rare = counts[counts < threshold].index
    series = series.copy()
    series[series.isin(rare)] = Domain.OTHER
    return series


def _canonicalize_id_value(x, default: str) -> str:
    if pd.isna(x):
        return default

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return default
        if float(x).is_integer():
            return str(int(x))
        return str(x).strip()

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return default
    return s


def _canonicalize_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if Col.SURGEON_CODE in out.columns:
        out[Col.SURGEON_CODE] = out[Col.SURGEON_CODE].map(lambda x: _canonicalize_id_value(x, Domain.OTHER))

    if Col.PROCEDURE_ID in out.columns:
        out[Col.PROCEDURE_ID] = out[Col.PROCEDURE_ID].map(lambda x: _canonicalize_id_value(x, Domain.OTHER))

    if Col.CASE_SERVICE in out.columns:
        out[Col.CASE_SERVICE] = out[Col.CASE_SERVICE].map(lambda x: _canonicalize_id_value(x, Domain.UNKNOWN))

    return out
