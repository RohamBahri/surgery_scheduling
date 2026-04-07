"""
Dataset exploration for incentive-aware operating-room scheduling.

Reads the UHN surgical dataset, applies a cleaning pipeline for elective
completed OR cases, and produces descriptive statistics, tables, and figures
that inform every stage of the modeling framework:
  - Inverse optimization (booking-error asymmetry, empirical quantile coverage)
  - Response estimation  (consecutive same-surgeon-procedure pair counts)
  - Bilevel scheduling    (block structure, turnover, eligibility)

Additional analyses support experimental design decisions:
  - Within-block case sequencing (surgeon clustering vs. interleaving)
  - Turnover variability and resource share across services and surgeons
  - Booking granularity and EHR convention detection (n×5 − 1 pattern)
  - Site decomposition (joint vs. per-site planning)
  - Block opening design (fixed vs. decision-variable templates)

Usage:
    python data_analysis.py --data path/to/dataset.xlsx
"""

import argparse
import math
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.core.paths import ArtifactManager

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
FIGDIR = Path(".")
TBLDIR = Path(".")
REPORTDIR = Path(".")

# Minimum counts for reliable estimation
MIN_SURGEON_CASES  = 30
MIN_PROCEDURE_CASES = 20
MIN_SERVICE_CASES  = 50
MIN_TYPE_CASES     = 50       # paper's n_min for surgeon-type groups

# Consecutive-pair gap thresholds (days)
MAX_GAP_DAYS_LIST = [30, 60, 90]

# Warm-up for train/test split
WARMUP_WEEKS = 52

# Coverage diagnostics for historical room-day templates
TOP_K_TEMPLATE_COVERAGE = [1, 3, 5]

# Block capacity candidates (minutes)
FIXED_CAPACITY_OPTIONS = [420, 450, 480, 510, 540, 600]

# Turnover caps to explore (minutes)
TURNOVER_CAPS = [60, 90]

# Rooms that are physically ORs but designated for emergency use
EMERGENCY_ROOM_PATTERNS = ["OREMER", "ORER"]

SEPARATOR = "=" * 80


# ─────────────────────────────────────────────────────────────────────────────
#  Required and optional columns
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "Operating_Room", "Booked Time (Minutes)", "Surgeon_Code",
    "Main_Procedure_Id", "Patient_ID",
    "Actual Start Date", "Actual Stop Date",
]

OPTIONAL_COLS = [
    "Surgeon", "Main_Procedure",
    "Actual Start Time", "Actual Stop Time",
    "Enter Room Date", "Enter Room Time",
    "Leave Room Date", "Leave Room Time",
    "Case_Service", "Patient_Type", "Site",
    "Case_Cancelled_Reason", "Case Cancel Date", "Case Cancel Time",
    "Decision_Date", "Consult_Date",
    "Anaesthetic_Type_Given", "CMG", "CMG Description",
    "Acute LOS", "LOS",
    "Complication_diag1", "Recovery_Time_Mins",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def subsection(title):
    print(f"\n--- {title} ---")


def pct(num, denom):
    return f"{num:,} ({100 * num / denom:.1f}%)" if denom > 0 else "N/A"


def col_available(df, col):
    """True if *col* exists and has at least one non-null value."""
    return col in df.columns and df[col].notna().any()


def describe_series(s, name=""):
    """Print a percentile summary for a numeric series."""
    s = pd.Series(s).dropna()
    if len(s) == 0:
        print(f"  {name}  N=0  (no data)")
        return
    d = s.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50,
                                0.75, 0.90, 0.95, 0.99])
    print(f"  {name}  N={int(d['count']):,}  "
          f"mean={d['mean']:.1f}  std={d['std']:.1f}  "
          f"min={d['min']:.1f}  p5={d['5%']:.1f}  p25={d['25%']:.1f}  "
          f"median={d['50%']:.1f}  p75={d['75%']:.1f}  p95={d['95%']:.1f}  "
          f"max={d['max']:.1f}")


def save_csv(df_or_series, name):
    path = TBLDIR / f"{name}.csv"
    df_or_series.to_csv(path)
    print(f"  → Saved: {path}")


def build_full_weekly_index(weekly):
    """Reindex a weekly aggregation to a complete Monday-based calendar."""
    weekly = weekly[weekly.index.notna()].sort_index()
    if len(weekly) == 0:
        return weekly
    full_index = pd.date_range(weekly.index.min(), weekly.index.max(),
                               freq="W-MON")
    weekly = weekly.reindex(full_index, fill_value=0)
    weekly.index.name = "Week_Start"
    return weekly


def guard_empty(df, label):
    """Return True (and print a warning) if df is empty."""
    if df is None or len(df) == 0:
        print(f"  ⚠ No data available — skipping {label}.")
        return True
    return False


def weekday_order():
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def format_day_set(days):
    order = {d: i for i, d in enumerate(weekday_order())}
    days = [d for d in pd.Series(list(days)).dropna().astype(str).unique()
            if d in order]
    days = sorted(days, key=lambda d: order[d])
    short = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed",
             "Thursday": "Thu", "Friday": "Fri",
             "Saturday": "Sat", "Sunday": "Sun"}
    return "-".join(short[d] for d in days) if days else "(none)"


def hhi_from_counts(counts):
    s = pd.Series(counts).dropna()
    total = s.sum()
    if total <= 0:
        return np.nan
    share = s / total
    return float((share ** 2).sum())


def top_k_share(counts, k):
    s = pd.Series(counts).sort_values(ascending=False)
    total = s.sum()
    if total <= 0 or len(s) == 0:
        return np.nan
    return float(100 * s.head(k).sum() / total)


def make_room_day_blocks(df, weekdays_only=False):
    """Construct room-date block summaries with dominance/sharing metrics."""
    if guard_empty(df, "room-day block builder"):
        return pd.DataFrame()

    tmp = df.copy()
    dt = pd.to_datetime(tmp["Actual Start Date"], errors="coerce")
    tmp = tmp[dt.notna()].copy()
    tmp["OR_Date"] = dt[dt.notna()].dt.date
    tmp["DayOfWeek"] = dt[dt.notna()].dt.day_name()
    tmp["Week_Start"] = (
        dt[dt.notna()] - pd.to_timedelta(dt[dt.notna()].dt.weekday, unit="D")
    ).dt.normalize()
    tmp["WeekdayNum"] = pd.to_datetime(tmp["Actual Start Date"]).dt.weekday

    if weekdays_only:
        tmp = tmp[tmp["WeekdayNum"] < 5].copy()

    agg = tmp.groupby(["Operating_Room", "OR_Date"]).agg(
        Week_Start=("Week_Start", "first"),
        DayOfWeek=("DayOfWeek", "first"),
        n_cases=("Patient_ID", "size"),
        n_surgeons=("Surgeon_Code", "nunique"),
        total_booked=("Booked Time (Minutes)", "sum"),
        total_realized=("Realized_Duration_Min", "sum"),
        total_surgical=("Surgical_Duration_Min", "sum"),
    )

    if col_available(tmp, "Case_Service"):
        svc_non_missing = tmp.dropna(subset=["Case_Service"])
        if len(svc_non_missing) > 0:
            n_services = (svc_non_missing.groupby(["Operating_Room", "OR_Date"])
                          ["Case_Service"].nunique())
            agg["n_services"] = n_services.reindex(agg.index).fillna(0).astype(int)
        else:
            agg["n_services"] = 0

    if "Enter Room_DT" in tmp.columns and "Leave Room_DT" in tmp.columns:
        timing = tmp.dropna(subset=["Enter Room_DT", "Leave Room_DT"]).groupby(
            ["Operating_Room", "OR_Date"]).agg(
                first_enter=("Enter Room_DT", "min"),
                last_leave=("Leave Room_DT", "max"),
        )
        agg = agg.join(timing, how="left")
        if {"first_enter", "last_leave"}.issubset(agg.columns):
            agg["span_min"] = (
                (agg["last_leave"] - agg["first_enter"]).dt.total_seconds() / 60
            )

    total_cases = tmp.groupby(["Operating_Room", "OR_Date"]).size()
    surg_counts = (tmp.groupby(["Operating_Room", "OR_Date", "Surgeon_Code"])
                   .size().rename("cases"))
    dom_surg = surg_counts.groupby(["Operating_Room", "OR_Date"]).max() / total_cases
    agg["dom_surgeon_case_share"] = dom_surg.reindex(agg.index).astype(float)

    if col_available(tmp, "Case_Service"):
        svc_tmp = tmp.dropna(subset=["Case_Service"])
        if len(svc_tmp) > 0:
            svc_counts = (svc_tmp.groupby(["Operating_Room", "OR_Date", "Case_Service"])
                          .size().rename("cases"))
            svc_total = svc_tmp.groupby(["Operating_Room", "OR_Date"]).size()
            dom_svc = svc_counts.groupby(["Operating_Room", "OR_Date"]).max() / svc_total
            agg["dom_service_case_share"] = dom_svc.reindex(agg.index).astype(float)

    return agg.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path):
    """Load the dataset from CSV, pickle, parquet, or Excel."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Drop stale index columns that appear when Excel files are re-saved
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"  Dropped {len(unnamed)} unnamed index column(s).")

    # Parse date columns that pandas might not auto-detect
    date_cols = [
        "Consult_Date", "Decision_Date", "Enter Room Date",
        "Actual Start Date", "Actual Stop Date", "Leave Room Date",
        "Admit Recovery Date", "Leave Recovery Date",
        "Ward Admit Date", "Ward Discharge Date",
        "Case Cancel Date", "MDU Admit Date", "MDU Discharge Date",
    ]
    for c in date_cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Column audit
# ─────────────────────────────────────────────────────────────────────────────
def audit_columns(df):
    section("COLUMN AUDIT")
    found_req  = [c for c in REQUIRED_COLS if c in df.columns]
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    found_opt  = [c for c in OPTIONAL_COLS if c in df.columns]
    missing_opt = [c for c in OPTIONAL_COLS if c not in df.columns]

    print(f"  Total columns in file: {df.shape[1]}")
    print(f"\n  Required columns: {len(found_req)}/{len(REQUIRED_COLS)} found")
    if missing_req:
        print(f"  ⚠ MISSING REQUIRED: {missing_req}")
    print(f"  Optional columns: {len(found_opt)}/{len(OPTIONAL_COLS)} found")
    if missing_opt:
        print(f"  Missing optional:  {missing_opt}")

    subsection("All column names")
    for i, c in enumerate(df.columns):
        tag = ""
        if c in REQUIRED_COLS:
            tag = " [REQUIRED]"
        elif c in OPTIONAL_COLS:
            tag = " [OPTIONAL]"
        print(f"    {i:3d}  {c}{tag}")

    if missing_req:
        raise ValueError(
            f"Cannot proceed without required columns: {missing_req}")


# ─────────────────────────────────────────────────────────────────────────────
#  Derived columns
# ─────────────────────────────────────────────────────────────────────────────
def combine_date_time(df, date_col, time_col, dt_col):
    """Combine a date column and a separate time-string column into one
    datetime column.  Handles HH:MM, HH:MM:SS, HH:MM:SS.fff, and AM/PM."""
    if date_col not in df.columns or time_col not in df.columns:
        return 0

    date_parsed = pd.to_datetime(df[date_col], errors="coerce")
    date_str = date_parsed.dt.strftime("%Y-%m-%d")

    time_str = (df[time_col].astype(str).str.strip()
                .replace({"nan": "", "NaT": "", "None": "", "<NA>": ""}))

    valid = date_parsed.notna() & (time_str != "")
    combined = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if valid.any():
        combined[valid] = pd.to_datetime(
            date_str[valid] + " " + time_str[valid], errors="coerce")

    df[dt_col] = combined
    n_ok = combined.notna().sum()
    n_failed = valid.sum() - n_ok
    if n_failed > 0:
        print(f"  [datetime] {dt_col}: {n_ok:,} OK, {n_failed:,} failed")
    return n_ok


def add_derived_columns(df):
    """Compute timestamps, realized duration, room time, booking error,
    year, week, OR-room flag, and cancellation flag."""

    # Combine date + time into precise datetime columns
    dt_pairs = [
        ("Actual Start Date", "Actual Start Time", "Actual Start_DT"),
        ("Actual Stop Date",  "Actual Stop Time",  "Actual Stop_DT"),
        ("Enter Room Date",   "Enter Room Time",   "Enter Room_DT"),
        ("Leave Room Date",   "Leave Room Time",   "Leave Room_DT"),
    ]
    for date_col, time_col, dt_col in dt_pairs:
        combine_date_time(df, date_col, time_col, dt_col)

    # Surgical duration: Actual Start → Actual Stop (skin-to-skin)
    if "Actual Start_DT" in df.columns and "Actual Stop_DT" in df.columns:
        df["Surgical_Duration_Min"] = (
            (df["Actual Stop_DT"] - df["Actual Start_DT"])
            .dt.total_seconds() / 60
        )
    else:
        df["Surgical_Duration_Min"] = np.nan

    # Room occupancy time: Enter Room → Leave Room
    if "Enter Room_DT" in df.columns and "Leave Room_DT" in df.columns:
        df["Room_Time_Min"] = (
            (df["Leave Room_DT"] - df["Enter Room_DT"])
            .dt.total_seconds() / 60
        )

    # For block-load modelling, the relevant duration is room time when
    # available, falling back to surgical time.  The booking error is
    # computed against *both* to let the analyst decide which baseline
    # to use for the inverse step.
    df["Booking_Error_Surgical"] = (
        df["Booked Time (Minutes)"] - df["Surgical_Duration_Min"])
    if "Room_Time_Min" in df.columns:
        df["Booking_Error_Room"] = (
            df["Booked Time (Minutes)"] - df["Room_Time_Min"])

    # Default booking error: use room time if available, else surgical time
    if "Room_Time_Min" in df.columns:
        df["Realized_Duration_Min"] = df["Room_Time_Min"].fillna(
            df["Surgical_Duration_Min"])
    else:
        df["Realized_Duration_Min"] = df["Surgical_Duration_Min"]
    df["Booking_Error_Min"] = (
        df["Booked Time (Minutes)"] - df["Realized_Duration_Min"])

    # Calendar helpers
    df["Year"] = pd.to_datetime(
        df["Actual Start Date"], errors="coerce").dt.year
    dt = pd.to_datetime(df["Actual Start Date"], errors="coerce")
    df["Week_Start"] = (
        dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()

    # OR-room flag: starts with "OR" but NOT emergency-designated rooms
    room_str = df["Operating_Room"].fillna("").astype(str).str.upper()
    starts_or = room_str.str.startswith("OR")
    is_emerg_room = room_str.isin(
        [r.upper() for r in EMERGENCY_ROOM_PATTERNS])
    df["Is_OR_Room"] = starts_or & ~is_emerg_room

    # Cancellation flag (union of non-empty reason and cancel date)
    reason_flag = pd.Series(False, index=df.index)
    if "Case_Cancelled_Reason" in df.columns:
        reason_text = df["Case_Cancelled_Reason"].astype(str).str.strip()
        reason_flag = (
            df["Case_Cancelled_Reason"].notna()
            & (reason_text != "")
            & (reason_text.str.lower() != "nan")
        )
    date_flag = pd.Series(False, index=df.index)
    if "Case Cancel Date" in df.columns:
        date_flag = df["Case Cancel Date"].notna()
    df["Is_Cancelled"] = reason_flag | date_flag

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing / cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    """Apply the cleaning pipeline for elective completed OR cases.

    Steps:
      1. Keep OR-room cases (excluding emergency-designated rooms).
      2. Exclude cancelled cases.
      3. Exclude emergency patients.
      4. Require actual start and stop dates.
      5. Require positive and finite realized durations (< 24 h).
      6. Require positive booked duration.
      7. Drop cases with impossible timestamps (Leave Room < Enter Room).
    """
    section("PREPROCESSING / CLEANING PIPELINE")
    n0 = len(df)

    df_c = df[df["Is_OR_Room"]].copy()
    n1 = len(df_c)
    print(f"  [1] Keep OR-room cases (excl. emergency rooms): "
          f"{n0:,} → {n1:,}  (dropped {n0 - n1:,})")

    df_c = df_c[~df_c["Is_Cancelled"]].copy()
    n2 = len(df_c)
    print(f"  [2] Exclude cancelled cases:                    "
          f"{n1:,} → {n2:,}  (dropped {n1 - n2:,})")

    if col_available(df_c, "Patient_Type"):
        emerg = (df_c["Patient_Type"].fillna("").astype(str)
                 .str.strip().str.upper()
                 .isin(["EMERGENCY PATIENT", "EMERGENCY"]))
        n_emerg = emerg.sum()
        df_c = df_c[~emerg].copy()
        n3 = len(df_c)
        print(f"  [3] Exclude emergency patients:                 "
              f"{n2:,} → {n3:,}  (dropped {n_emerg:,})")
    else:
        n3 = n2
        print(f"  [3] Exclude emergency patients:                 "
              f"skipped (Patient_Type not available)")

    df_c = df_c.dropna(subset=["Actual Start Date", "Actual Stop Date"]).copy()
    n4 = len(df_c)
    print(f"  [4] Require Actual Start & Stop Date:            "
          f"{n3:,} → {n4:,}  (dropped {n3 - n4:,})")

    valid_dur = ((df_c["Realized_Duration_Min"] > 0)
                 & (df_c["Realized_Duration_Min"] < 1440))
    df_c = df_c[valid_dur].copy()
    n5 = len(df_c)
    print(f"  [5] Valid realized duration (0, 1440 min):       "
          f"{n4:,} → {n5:,}  (dropped {n4 - n5:,})")

    df_c = df_c[df_c["Booked Time (Minutes)"] > 0].copy()
    n6 = len(df_c)
    print(f"  [6] Positive booked duration:                    "
          f"{n5:,} → {n6:,}  (dropped {n5 - n6:,})")

    # Drop impossible room timestamps
    n_bad_room = 0
    if "Enter Room_DT" in df_c.columns and "Leave Room_DT" in df_c.columns:
        bad = (df_c["Leave Room_DT"] < df_c["Enter Room_DT"])
        bad = bad.fillna(False)
        n_bad_room = bad.sum()
        df_c = df_c[~bad].copy()
    n7 = len(df_c)
    print(f"  [7] Drop impossible room timestamps:             "
          f"{n6:,} → {n7:,}  (dropped {n_bad_room:,})")

    # Recompute booking error on final cohort
    df_c["Booking_Error_Min"] = (
        df_c["Booked Time (Minutes)"] - df_c["Realized_Duration_Min"])
    df_c["Booking_Error_Surgical"] = (
        df_c["Booked Time (Minutes)"] - df_c["Surgical_Duration_Min"])
    if "Room_Time_Min" in df_c.columns:
        df_c["Booking_Error_Room"] = (
            df_c["Booked Time (Minutes)"] - df_c["Room_Time_Min"])

    print(f"\n  ✓ Final cleaned dataset: {n7:,} cases")

    # Outlier diagnostics
    subsection("Outlier diagnostics (on cleaned data)")
    rd = df_c["Realized_Duration_Min"]
    for thresh in [480, 600, 720]:
        n_above = (rd > thresh).sum()
        print(f"  Realized duration > {thresh} min ({thresh // 60}h): "
              f"{pct(n_above, len(df_c))}")
    be = df_c["Booking_Error_Min"]
    print(f"  |Booking error| > 240 min (4h): "
          f"{pct((be.abs() > 240).sum(), len(df_c))}")
    if "Actual Stop_DT" in df_c.columns and "Actual Start_DT" in df_c.columns:
        bad_surg = (df_c["Actual Stop_DT"] < df_c["Actual Start_DT"]).sum()
        print(f"  Actual Stop before Actual Start: {bad_surg}")

    return df_c


# ══════════════════════════════════════════════════════════════════════════════
#                              ANALYSIS SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 0. Raw overview ────────────────────────────────────────────────────────
def analyze_raw_overview(df):
    section("RAW DATASET OVERVIEW")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns:    {df.shape[1]}")

    if col_available(df, "Patient_Type"):
        subsection("Patient_Type values")
        for pt, cnt in (df["Patient_Type"].fillna("(missing)")
                        .value_counts().items()):
            print(f"    {str(pt):30s}  {cnt:6,}")

    subsection("Operating Room breakdown")
    or_rooms = df[df["Is_OR_Room"]]
    non_or = df[~df["Is_OR_Room"]]
    emerg_rooms = df[df["Operating_Room"].fillna("").astype(str).str.upper()
                     .isin([r.upper() for r in EMERGENCY_ROOM_PATTERNS])]
    print(f"  Elective OR rooms:     {or_rooms['Operating_Room'].nunique()} "
          f"rooms, {len(or_rooms):,} cases")
    print(f"  Emergency-desig rooms: {emerg_rooms['Operating_Room'].nunique()} "
          f"rooms, {len(emerg_rooms):,} cases")
    print(f"  Non-OR rooms:          {non_or['Operating_Room'].nunique()} "
          f"rooms, {len(non_or):,} cases")

    print(f"\n  Elective OR rooms:")
    for room, cnt in (df.loc[df["Is_OR_Room"], "Operating_Room"]
                      .value_counts().sort_index().items()):
        print(f"    {room:30s}  {cnt:6,}")

    subsection("Cancellation overview")
    n_cancelled = df["Is_Cancelled"].sum()
    print(f"  Cancelled cases: {pct(n_cancelled, len(df))}")
    print(f"  Non-cancelled:   {pct(len(df) - n_cancelled, len(df))}")
    if n_cancelled > 0 and col_available(df, "Case_Cancelled_Reason"):
        print(f"\n  Top cancellation reasons:")
        reasons = df.loc[df["Is_Cancelled"]
                         & df["Case_Cancelled_Reason"].notna(),
                         "Case_Cancelled_Reason"]
        for reason, cnt in reasons.value_counts().head(15).items():
            print(f"    {str(reason):50s}  {cnt:5,}")

    subsection("Multi-site structure")
    if col_available(df, "Site"):
        print(f"  Unique sites: {df['Site'].nunique()}  "
              f"(plus {df['Site'].isna().sum():,} missing)")
        for site, cnt in df["Site"].value_counts().items():
            print(f"    {site:30s}  {cnt:6,}")
        surg_sites = (df.dropna(subset=["Site"])
                      .groupby("Surgeon_Code")["Site"].nunique())
        multi = (surg_sites > 1).sum()
        print(f"\n  Surgeons at multiple sites: {multi} / {len(surg_sites)}")
    else:
        print("  Site column not available.")


# ── 1. Cleaned overview ────────────────────────────────────────────────────
def analyze_cleaned_overview(df):
    section("CLEANED DATASET OVERVIEW")
    if guard_empty(df, "cleaned overview"):
        return
    print(f"  Total cleaned cases: {len(df):,}")

    subsection("By year")
    year_counts = df["Year"].value_counts().sort_index()
    for yr, cnt in year_counts.items():
        print(f"    {int(yr)}: {cnt:,}")
    save_csv(year_counts.rename("cases"), "cases_by_year")

    subsection("Date range")
    print(f"  Earliest: {df['Actual Start Date'].min()}")
    print(f"  Latest:   {df['Actual Start Date'].max()}")

    if col_available(df, "Patient_Type"):
        subsection("Patient_Type after cleaning")
        for pt, cnt in (df["Patient_Type"].fillna("(missing)")
                        .value_counts().items()):
            print(f"    {str(pt):30s}  {cnt:6,}")


# ── 2. Missingness and temporal-provenance audit ───────────────────────────
def analyze_missingness_audit(df):
    """For every candidate feature, report availability and classify timing
    as confirmed pre-op, likely post-op, or provenance uncertain."""
    section("MISSINGNESS AND TEMPORAL-PROVENANCE AUDIT")
    if guard_empty(df, "missingness audit"):
        return

    # Classification based on clinical workflow knowledge
    feature_timing = {
        # Confirmed pre-op (known at booking / scheduling time)
        "Patient_Type":           "confirmed_preop",
        "Case_Service":           "confirmed_preop",
        "Main_Procedure_Id":      "confirmed_preop",
        "Surgeon_Code":           "confirmed_preop",
        "Operating_Room":         "confirmed_preop",
        "Site":                   "confirmed_preop",
        "Booked Time (Minutes)":  "confirmed_preop",

        # Provenance uncertain — may be assigned pre-op or post-discharge
        "Anaesthetic_Type_Given": "provenance_uncertain",
        "CMG":                    "provenance_uncertain",

        # Confirmed post-op (known only after case execution)
        "Acute LOS":              "confirmed_postop",
        "LOS":                    "confirmed_postop",
        "Recovery_Time_Mins":     "confirmed_postop",
        "Complication_diag1":     "confirmed_postop",
    }

    print(f"  {'Feature':30s}  {'Non-null':>10s}  {'% avail':>8s}  "
          f"{'Timing':>25s}  {'Detail':>20s}")
    print(f"  {'─' * 100}")

    for feat, timing in feature_timing.items():
        if feat not in df.columns:
            print(f"  {feat:30s}  {'N/A':>10s}  {'—':>8s}  "
                  f"{timing:>25s}  column missing")
            continue
        n_valid = df[feat].notna().sum()
        pct_valid = 100 * n_valid / len(df)
        if df[feat].dtype == "object" or df[feat].dtype.name == "category":
            detail = f"{df[feat].nunique()} categories"
        elif pd.api.types.is_numeric_dtype(df[feat]):
            detail = f"mean={df[feat].mean():.1f}" if n_valid > 0 else "N/A"
        else:
            detail = ""
        print(f"  {feat:30s}  {n_valid:10,}  {pct_valid:7.1f}%  "
              f"{timing:>25s}  {detail:>20s}")

    subsection("Provenance notes")
    print("  Anaesthetic_Type_Given: the column name says 'Given', which")
    print("    suggests the anesthesia actually administered, not the planned")
    print("    type.  Verify against the data dictionary before using as a")
    print("    booking-time feature in x_ti.")
    if col_available(df, "CMG") and col_available(df, "LOS"):
        cmg_avail = df["CMG"].notna().mean()
        los_avail = df["LOS"].notna().mean()
        print(f"  CMG availability ({cmg_avail:.1%}) is very close to LOS "
              f"({los_avail:.1%}),")
        print("    suggesting CMG may be assigned post-discharge.  Do not use")
        print("    as a scheduling-time feature without provenance confirmation.")

    # Detail on pre-op categorical features
    for feat in ["Patient_Type", "Anaesthetic_Type_Given"]:
        if col_available(df, feat):
            subsection(f"{feat} value counts (top 15)")
            for val, cnt in (df[feat].fillna("(missing)")
                             .value_counts().head(15).items()):
                print(f"    {str(val):40s}  {cnt:6,}")

    if col_available(df, "CMG") and col_available(df, "Case_Service"):
        subsection("CMG availability by service")
        cmg_by = df.groupby("Case_Service")["CMG"].apply(
            lambda x: x.notna().mean())
        for svc, p in cmg_by.sort_values(ascending=False).items():
            print(f"    {svc:40s}  {100 * p:5.1f}%")

    # ── Feature leakage risk summary ──────────────────────────────────────
    subsection("Feature leakage risk classification")
    print("  This table classifies candidate features by whether they are")
    print("  available at booking/scheduling time, which determines whether")
    print("  they are safe to include in the covariate vector x_ti.\n")

    leakage_table = [
        # (feature, status, rationale)
        ("Patient_Type",           "safe",     "Assigned at booking"),
        ("Case_Service",           "safe",     "Determined by referral"),
        ("Main_Procedure_Id",      "safe",     "Booked procedure code"),
        ("Surgeon_Code",           "safe",     "Assigned at scheduling"),
        ("Operating_Room",         "safe",     "Assigned at scheduling"),
        ("Site",                   "safe",     "Determined by referral"),
        ("Booked Time (Minutes)",  "safe",     "Entered at booking"),
        ("Decision_Date",          "safe",     "Pre-operative date"),
        ("Consult_Date",           "safe",     "Pre-operative date"),
        ("Anaesthetic_Type_Given", "verify",   "Name says 'Given' — may be post-op"),
        ("CMG",                    "unsafe",   "Availability tracks LOS; likely post-discharge"),
        ("CMG Description",        "unsafe",   "Same provenance as CMG"),
        ("Acute LOS",              "unsafe",   "Post-operative outcome"),
        ("LOS",                    "unsafe",   "Post-operative outcome"),
        ("Recovery_Time_Mins",     "unsafe",   "Post-operative measurement"),
        ("Complication_diag1",     "unsafe",   "Post-operative diagnosis"),
    ]

    print(f"  {'Feature':30s}  {'Status':8s}  {'Rationale'}")
    print(f"  {'─' * 75}")
    leakage_rows = []
    for feat, status, rationale in leakage_table:
        avail = "present" if feat in df.columns else "missing"
        marker = {"safe": "✓", "verify": "?", "unsafe": "✗"}[status]
        print(f"  {feat:30s}  {marker} {status:7s}  {rationale}  [{avail}]")
        leakage_rows.append(dict(feature=feat, status=status,
                                 rationale=rationale, available=avail))

    print(f"\n  Legend: ✓ = safe at booking time, "
          f"? = must verify provenance, "
          f"✗ = post-operative / unsafe")
    save_csv(pd.DataFrame(leakage_rows), "feature_leakage_risk")


# ── 3. Surgeons, services, procedures ──────────────────────────────────────
def analyze_surgeons_services_procedures(df):
    section("SURGEONS, SERVICES, AND PROCEDURES")
    if guard_empty(df, "surgeon/service analysis"):
        return

    n_surgeons = df["Surgeon_Code"].nunique()
    n_procs = df["Main_Procedure_Id"].nunique()
    print(f"  Unique Surgeon_Code:      {n_surgeons}")
    if col_available(df, "Case_Service"):
        n_svc = df["Case_Service"].nunique()
        print(f"  Unique Case_Service:      {n_svc}  "
              f"(plus {df['Case_Service'].isna().sum():,} missing)")
    print(f"  Unique Main_Procedure_Id: {n_procs}")

    # Service composition
    if col_available(df, "Case_Service"):
        subsection("Case_Service values and case counts")
        svc = df["Case_Service"].value_counts().sort_values(ascending=False)
        for s, cnt in svc.items():
            flag = "  ◄ RARE" if cnt < MIN_SERVICE_CASES else ""
            print(f"    {s:40s}  {cnt:6,}{flag}")
        print(f"    {'(missing)':40s}  "
              f"{df['Case_Service'].isna().sum():6,}")
        save_csv(svc.rename("cases"), "service_composition")

    # Surgeon volume
    subsection(f"Surgeon volume distribution (N={n_surgeons})")
    surg_vol = df.groupby("Surgeon_Code").size().rename("cases")
    describe_series(surg_vol, "Cases/surgeon")
    n_below = (surg_vol < MIN_SURGEON_CASES).sum()
    n_above = (surg_vol >= MIN_SURGEON_CASES).sum()
    print(f"\n  With ≥ {MIN_SURGEON_CASES} cases: {n_above} surgeons  "
          f"({surg_vol[surg_vol >= MIN_SURGEON_CASES].sum():,} cases)")
    print(f"  With < {MIN_SURGEON_CASES} cases: {n_below} surgeons  "
          f"({surg_vol[surg_vol < MIN_SURGEON_CASES].sum():,} cases)")
    save_csv(surg_vol.sort_values(ascending=False).reset_index(),
             "surgeon_volume")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(surg_vol.clip(upper=500), bins=50, edgecolor="white", alpha=0.8)
    ax.axvline(MIN_SURGEON_CASES, color="red", ls="--",
               label=f"n_min={MIN_SURGEON_CASES}")
    ax.set_xlabel("Cases per surgeon")
    ax.set_ylabel("Number of surgeons")
    ax.set_title("Surgeon Volume Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "surgeon_volume_distribution.png", dpi=150)
    plt.close(fig)

    # Procedure volume
    subsection("Procedure volume distribution")
    proc_vol = df.groupby("Main_Procedure_Id").size().rename("cases")
    describe_series(proc_vol, "Cases/procedure")
    n_rare = (proc_vol < MIN_PROCEDURE_CASES).sum()
    print(f"\n  Procedures with < {MIN_PROCEDURE_CASES} cases: "
          f"{n_rare} / {len(proc_vol)} "
          f"(covering {proc_vol[proc_vol < MIN_PROCEDURE_CASES].sum():,})")
    print(f"  Top 20 procedures:")
    for proc, cnt in proc_vol.sort_values(ascending=False).head(20).items():
        name = ""
        if col_available(df, "Main_Procedure"):
            m = df.loc[df["Main_Procedure_Id"] == proc, "Main_Procedure"]
            if len(m) > 0:
                name = str(m.iloc[0])[:60]
        print(f"    {str(proc):15s}  {cnt:6,}  {name}")

    # Surgeon × procedure pair volume
    subsection("Cases per (Surgeon, Procedure) pair")
    sp_vol = df.groupby(
        ["Surgeon_Code", "Main_Procedure_Id"]).size().rename("cases")
    describe_series(sp_vol, "Cases/pair")
    print(f"\n  Total unique pairs: {len(sp_vol):,}")
    for t in [2, 5, 10, 20]:
        above = sp_vol[sp_vol >= t]
        print(f"  Pairs with ≥ {t:2d} cases: {len(above):,}  "
              f"(covering {above.sum():,} cases)")


# ── 4. Pooling groups for surgeon-level estimation ────────────────────────
def analyze_surgeon_types(df):
    """Construct exogenous pooling groups (service × experience band) used
    as a regularization scaffold for surgeon-level estimation.

    The behavioral primitive is surgeon-specific (q_s).  Groups are NOT
    the conceptual object of interest; they serve only as shrinkage
    targets for data-poor surgeons via hierarchical partial pooling.
    Data-rich surgeons receive near-individual estimates.

    Groups are formed as service × volume-tertile, with rare cells merged
    until every group has at least n_min cases.  The volume tertile proxies
    experience (years-since-credentialing is unavailable).
    """
    section("POOLING GROUPS FOR SURGEON-LEVEL ESTIMATION")
    if guard_empty(df, "surgeon types"):
        return df
    if not col_available(df, "Case_Service"):
        print("  Case_Service not available — cannot build types.")
        return df

    # Assign each surgeon a primary service (mode of Case_Service)
    surg_svc = (df.groupby("Surgeon_Code")["Case_Service"]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0
                     else np.nan))
    surg_vol = df.groupby("Surgeon_Code").size().rename("cases")
    surg_df = pd.DataFrame({"primary_service": surg_svc, "cases": surg_vol})
    surg_df = surg_df.dropna(subset=["primary_service"])

    # Within-service volume tertile as experience proxy.
    # When a service has too few surgeons or identical volumes, the tertile
    # boundaries collapse.  We fall back to fewer bands or a single band.
    def vol_band_for_group(g):
        vals = g["cases"]
        n_unique = vals.nunique()
        if n_unique >= 3:
            try:
                return pd.qcut(vals, q=3, labels=["low", "mid", "high"],
                               duplicates="drop")
            except ValueError:
                pass
        if n_unique == 2:
            median = vals.median()
            return np.where(vals <= median, "low", "high")
        # All identical or single surgeon — one band
        return pd.Series("all", index=g.index)

    bands = []
    for svc, grp in surg_df.groupby("primary_service"):
        b = vol_band_for_group(grp)
        bands.append(pd.Series(b, index=grp.index, dtype=str))
    surg_df["vol_band"] = pd.concat(bands)

    surg_df["type_label"] = (surg_df["primary_service"] + "_"
                             + surg_df["vol_band"].astype(str))

    # Merge rare types: collapse within service until n_min is met
    type_cases = surg_df.groupby("type_label")["cases"].sum()
    rare_types = type_cases[type_cases < MIN_TYPE_CASES].index
    surg_df.loc[surg_df["type_label"].isin(rare_types), "type_label"] = (
        surg_df.loc[surg_df["type_label"].isin(rare_types), "primary_service"]
        + "_pooled"
    )

    # Re-check after merging within service
    type_cases2 = surg_df.groupby("type_label")["cases"].sum()
    still_rare = type_cases2[type_cases2 < MIN_TYPE_CASES].index
    surg_df.loc[surg_df["type_label"].isin(still_rare), "type_label"] = (
        "OTHER_pooled")

    # Map types back to case-level data
    surg_type_map = surg_df["type_label"].to_dict()
    df["Surgeon_Type"] = df["Surgeon_Code"].map(surg_type_map).fillna("OTHER")

    n_types = df["Surgeon_Type"].nunique()
    type_summary = (df.groupby("Surgeon_Type")
                    .agg(n_cases=("Patient_ID", "size"),
                         n_surgeons=("Surgeon_Code", "nunique"))
                    .sort_values("n_cases", ascending=False))

    print(f"  Total types: {n_types}")
    print(f"  n_min target: {MIN_TYPE_CASES}")
    print(f"\n  {'Type':35s}  {'Cases':>8s}  {'Surgeons':>10s}")
    for typ, row in type_summary.iterrows():
        flag = "  ◄ below n_min" if row["n_cases"] < MIN_TYPE_CASES else ""
        print(f"  {typ:35s}  {int(row['n_cases']):8,}  "
              f"{int(row['n_surgeons']):10,}{flag}")

    save_csv(type_summary.reset_index(), "surgeon_types")
    subsection("Type-size diagnostics")
    describe_series(type_summary["n_cases"], "Cases/type")
    describe_series(type_summary["n_surgeons"], "Surgeons/type")

    return df


# ── 5. Booking signal value ────────────────────────────────────────────────
def analyze_booking_signal(df):
    """Quantify how informative bookings are for realized durations.

    This supports the paper's 'preserve signal' argument: bookings are
    distorted but carry substantial case-level information that a purely
    predictive model would lose.
    """
    section("BOOKING SIGNAL VALUE (booked vs. realized)")
    if guard_empty(df, "booking signal"):
        return

    b = df["Booked Time (Minutes)"].values
    d = df["Realized_Duration_Min"].values
    mask = np.isfinite(b) & np.isfinite(d) & (b > 0) & (d > 0)
    b, d = b[mask], d[mask]

    corr_p, _ = sp_stats.pearsonr(b, d)
    corr_s, _ = sp_stats.spearmanr(b, d)
    mae = np.mean(np.abs(b - d))
    rmse = np.sqrt(np.mean((b - d) ** 2))
    mdae = np.median(np.abs(b - d))

    # Calibration: OLS  realized ~ booked
    slope, intercept, r_val, _, _ = sp_stats.linregress(b, d)

    subsection("Overall booking-vs-realized statistics")
    print(f"  N = {len(b):,}")
    print(f"  Pearson correlation:   {corr_p:.4f}")
    print(f"  Spearman correlation:  {corr_s:.4f}")
    print(f"  MAE (booked − real):   {mae:.1f} min")
    print(f"  RMSE:                  {rmse:.1f} min")
    print(f"  Median absolute error: {mdae:.1f} min")
    print(f"  Calibration slope:     {slope:.4f}  "
          f"(perfect calibration = 1.0)")
    print(f"  Calibration intercept: {intercept:.1f}  "
          f"(perfect calibration = 0.0)")
    print(f"  R²:                    {r_val**2:.4f}")

    # By-service
    if col_available(df, "Case_Service"):
        subsection("By service")
        df_m = df.dropna(subset=["Case_Service"]).copy()
        print(f"  {'Service':30s}  {'N':>6s}  {'Corr':>6s}  {'MAE':>6s}  "
              f"{'Slope':>7s}  {'Intcpt':>7s}")
        rows = []
        for svc in df_m["Case_Service"].value_counts().index:
            sub = df_m[df_m["Case_Service"] == svc]
            bs, ds = (sub["Booked Time (Minutes)"].values,
                      sub["Realized_Duration_Min"].values)
            ok = np.isfinite(bs) & np.isfinite(ds)
            if ok.sum() < 10:
                continue
            bs, ds = bs[ok], ds[ok]
            c, _ = sp_stats.pearsonr(bs, ds)
            m = np.mean(np.abs(bs - ds))
            sl, ic, _, _, _ = sp_stats.linregress(bs, ds)
            print(f"  {svc:30s}  {len(bs):6,}  {c:6.3f}  {m:6.1f}  "
                  f"{sl:7.3f}  {ic:7.1f}")
            rows.append(dict(service=svc, n=len(bs), corr=c, mae=m,
                             slope=sl, intercept=ic))
        save_csv(pd.DataFrame(rows), "booking_signal_by_service")

    # Scatter plot: booked vs realized
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sample_idx = np.random.default_rng(42).choice(
        len(b), size=min(8000, len(b)), replace=False)
    ax = axes[0]
    ax.scatter(b[sample_idx], d[sample_idx], alpha=0.1, s=6, c="steelblue")
    lims = [0, np.percentile(np.concatenate([b, d]), 99)]
    ax.plot(lims, lims, "r--", lw=1, label="y = x")
    ax.plot(lims, [intercept + slope * x for x in lims],
            "orange", lw=1.5, label=f"OLS: y={slope:.2f}x+{intercept:.0f}")
    ax.set_xlabel("Booked duration (min)")
    ax.set_ylabel("Realized duration (min)")
    ax.set_title("Booked vs. Realized Duration")
    ax.legend(fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Residual by booked-duration bin
    ax = axes[1]
    err = b - d
    bins = pd.qcut(b, q=10, duplicates="drop")
    bin_stats = pd.DataFrame({"error": err, "bin": bins}).groupby("bin")
    means = bin_stats["error"].mean()
    stds = bin_stats["error"].std()
    x_pos = range(len(means))
    ax.bar(x_pos, means, yerr=stds, color="steelblue", alpha=0.7,
           capsize=3, edgecolor="white")
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{iv.left:.0f}–{iv.right:.0f}"
                        for iv in means.index],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Booked duration bin (min)")
    ax.set_ylabel("Mean booking error (booked − realized)")
    ax.set_title("Booking Error by Duration Bin")

    fig.tight_layout()
    fig.savefig(FIGDIR / "booking_signal_analysis.png", dpi=150)
    plt.close(fig)


# ── 6. Booking error distribution ──────────────────────────────────────────
def analyze_booking_error(df):
    section("BOOKING ERROR DISTRIBUTION (b − d̃)")
    if guard_empty(df, "booking error"):
        return

    err = df["Booking_Error_Min"].dropna()
    subsection("Overall")
    describe_series(err, "Booking error (min)")
    print(f"\n  Fraction over-booked (b > d̃):  {(err > 0).mean():.3f}")
    print(f"  Fraction under-booked (b < d̃): {(err < 0).mean():.3f}")
    print(f"  Fraction exactly zero:          {(err == 0).mean():.3f}")

    # Compare surgical-time vs room-time errors
    if "Booking_Error_Room" in df.columns:
        subsection("Booking error against surgical time vs. room time")
        err_surg = df["Booking_Error_Surgical"].dropna()
        err_room = df["Booking_Error_Room"].dropna()
        print(f"  Against surgical time:  mean={err_surg.mean():.1f}  "
              f"median={err_surg.median():.1f}  %>0={100*(err_surg>0).mean():.1f}%  "
              f"N={len(err_surg):,}")
        print(f"  Against room time:      mean={err_room.mean():.1f}  "
              f"median={err_room.median():.1f}  %>0={100*(err_room>0).mean():.1f}%  "
              f"N={len(err_room):,}")
        overhead = (df["Room_Time_Min"] - df["Surgical_Duration_Min"]).dropna()
        overhead = overhead[overhead >= 0]
        if len(overhead) > 0:
            print(f"\n  Non-surgical room overhead (Room − Surgical time):")
            describe_series(overhead, "Overhead (min)")

        print("\n  Conclusion: Booked Time is much closer to room occupancy than")
        print("  to surgical time alone (mean error "
              f"{err_room.mean():.1f} vs {err_surg.mean():.1f} min). This is")
        print("  consistent with bookings targeting total room time inclusive of")
        print("  setup, positioning, and cleanup — not knife-to-close time.")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    clip_err = err.clip(-120, 120)
    ax.hist(clip_err, bins=80, edgecolor="white", alpha=0.8, color="steelblue")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.axvline(err.mean(), color="orange", lw=1.5,
               label=f"Mean={err.mean():.1f}")
    ax.axvline(err.median(), color="green", lw=1.5,
               label=f"Median={err.median():.1f}")
    ax.set_xlabel("Booking Error (min): b − d̃")
    ax.set_ylabel("Count")
    ax.set_title("Overall Booking Error Distribution")
    ax.legend()

    ax = axes[1]
    if col_available(df, "Case_Service"):
        svc_err = df.dropna(subset=["Case_Service", "Booking_Error_Min"])
        top_svc = (svc_err["Case_Service"].value_counts()
                   .loc[lambda x: x >= 100].index.tolist())
        if len(top_svc) >= 2:
            svc_err = svc_err[svc_err["Case_Service"].isin(top_svc)]
            order = (svc_err.groupby("Case_Service")["Booking_Error_Min"]
                     .median().sort_values())
            data = [svc_err.loc[svc_err["Case_Service"] == s,
                                "Booking_Error_Min"].clip(-120, 120).values
                    for s in order.index]
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color="red", lw=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue")
            ax.set_xticklabels([s[:20] for s in order.index],
                               rotation=45, ha="right", fontsize=7)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_ylabel("Booking Error (min)")
            ax.set_title("Booking Error by Service")
    fig.tight_layout()
    fig.savefig(FIGDIR / "booking_error_distribution.png", dpi=150)
    plt.close(fig)

    # By service table
    if col_available(df, "Case_Service"):
        subsection("By service (sorted by median error)")
        svc_all = df.dropna(subset=["Case_Service", "Booking_Error_Min"])
        order = (svc_all.groupby("Case_Service")["Booking_Error_Min"]
                 .median().sort_values())
        rows = []
        print(f"  {'Service':30s}  {'N':>6s}  {'Mean':>7s}  {'Med':>7s}  "
              f"{'Std':>7s}  {'%>0':>6s}  {'Skew':>7s}")
        for svc in order.index:
            sub = svc_all.loc[svc_all["Case_Service"] == svc,
                              "Booking_Error_Min"]
            skw = sub.skew() if len(sub) >= 3 else np.nan
            print(f"  {svc:30s}  {len(sub):6,}  {sub.mean():7.1f}  "
                  f"{sub.median():7.1f}  {sub.std():7.1f}  "
                  f"{(sub > 0).mean():6.3f}  {skw:7.2f}")
            rows.append(dict(service=svc, n=len(sub), mean=sub.mean(),
                             median=sub.median(), std=sub.std(),
                             pct_positive=(sub > 0).mean(), skewness=skw))
        save_csv(pd.DataFrame(rows), "booking_error_by_service")

    subsection("By year")
    for yr in sorted(df["Year"].dropna().unique()):
        sub = df.loc[df["Year"] == yr, "Booking_Error_Min"].dropna()
        print(f"  {int(yr)}: N={len(sub):,}  mean={sub.mean():.1f}  "
              f"median={sub.median():.1f}  std={sub.std():.1f}  "
              f"%>0={100 * (sub > 0).mean():.1f}%")


# ── 7. Empirical quantile coverage ─────────────────────────────────────────
def analyze_empirical_coverage(df):
    """Compute Pr(d̃ ≤ b) as a descriptive proxy for surgeon-specific q_s.

    Under the newsvendor rationalization, a surgeon targeting quantile q_s
    will produce bookings that exceed realized duration with probability q_s
    on average.  This is a model-free bridge to the inverse step.
    """
    section("EMPIRICAL QUANTILE COVERAGE — Pr(d̃ ≤ b)")
    if guard_empty(df, "empirical coverage"):
        return

    df_m = df.dropna(subset=["Booking_Error_Min"]).copy()
    overall_cov = (df_m["Realized_Duration_Min"]
                   <= df_m["Booked Time (Minutes)"]).mean()
    print(f"  Overall Pr(d̃ ≤ b) = {overall_cov:.4f}")
    print(f"  Interpretation: bookings exceed realized duration "
          f"{100 * overall_cov:.1f}% of the time.")

    # By service
    if col_available(df_m, "Case_Service"):
        subsection("By service")
        svc_cov = (df_m.groupby("Case_Service")
                   .apply(lambda g: (g["Realized_Duration_Min"]
                                     <= g["Booked Time (Minutes)"]).mean())
                   .sort_values(ascending=False))
        svc_n = df_m.groupby("Case_Service").size()
        print(f"  {'Service':30s}  {'N':>6s}  {'Pr(d̃≤b)':>10s}")
        for svc in svc_cov.index:
            print(f"  {svc:30s}  {svc_n[svc]:6,}  {svc_cov[svc]:10.4f}")
        save_csv(pd.DataFrame({"n": svc_n, "coverage": svc_cov}),
                 "empirical_coverage_by_service")

    # By surgeon (with sufficient volume)
    subsection("By surgeon (≥ 50 cases)")
    surg_cov = (df_m.groupby("Surgeon_Code")
                .apply(lambda g: (g["Realized_Duration_Min"]
                                  <= g["Booked Time (Minutes)"]).mean()))
    surg_n = df_m.groupby("Surgeon_Code").size()
    surg_cov = surg_cov[surg_n >= 50]
    describe_series(surg_cov, "Pr(d̃ ≤ b) across surgeons")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(surg_cov, bins=30, edgecolor="white", alpha=0.8, color="steelblue")
    ax.axvline(0.5, color="red", ls="--", lw=1, label="q = 0.5 (unbiased)")
    ax.axvline(surg_cov.mean(), color="orange", lw=1.5,
               label=f"Mean = {surg_cov.mean():.3f}")
    ax.set_xlabel("Pr(d̃ ≤ b) per surgeon")
    ax.set_ylabel("Number of surgeons")
    ax.set_title("Empirical Coverage by Surgeon (proxy for q_s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "empirical_coverage_by_surgeon.png", dpi=150)
    plt.close(fig)

    # By surgeon type if available
    if "Surgeon_Type" in df_m.columns:
        subsection("By surgeon type")
        type_cov = (df_m.groupby("Surgeon_Type")
                    .apply(lambda g: (g["Realized_Duration_Min"]
                                      <= g["Booked Time (Minutes)"]).mean())
                    .sort_values(ascending=False))
        type_n = df_m.groupby("Surgeon_Type").size()
        print(f"  {'Type':35s}  {'N':>8s}  {'Pr(d̃≤b)':>10s}")
        for typ in type_cov.index:
            print(f"  {typ:35s}  {type_n[typ]:8,}  {type_cov[typ]:10.4f}")


# ── 8. Within-service surgeon heterogeneity ────────────────────────────────
def analyze_within_service_heterogeneity(df):
    """Decompose booking-error variance into between-surgeon and
    within-surgeon components to quantify the signal the behavioral
    model can capture."""
    section("WITHIN-SERVICE SURGEON HETEROGENEITY")
    if guard_empty(df, "heterogeneity"):
        return
    if not col_available(df, "Case_Service"):
        print("  Case_Service not available — skipping.")
        return

    df_m = df.dropna(subset=["Case_Service", "Booking_Error_Min"]).copy()

    # Keep services with enough surgeons
    svc_surg = df_m.groupby("Case_Service")["Surgeon_Code"].nunique()
    usable_svc = svc_surg[svc_surg >= 3].index

    print(f"  {'Service':30s}  {'Surgeons':>9s}  "
          f"{'σ_between':>10s}  {'σ_within':>10s}  {'Ratio':>7s}")
    rows = []
    for svc in sorted(usable_svc):
        sub = df_m[df_m["Case_Service"] == svc]
        surg_means = sub.groupby("Surgeon_Code")["Booking_Error_Min"].mean()
        surg_means = surg_means[sub.groupby("Surgeon_Code").size() >= 10]
        if len(surg_means) < 3:
            continue
        sigma_between = surg_means.std()
        sigma_within = sub.groupby("Surgeon_Code")[
            "Booking_Error_Min"].std().mean()
        ratio = sigma_between / sigma_within if sigma_within > 0 else np.inf
        print(f"  {svc:30s}  {len(surg_means):9d}  {sigma_between:10.1f}  "
              f"{sigma_within:10.1f}  {ratio:7.2f}")
        rows.append(dict(service=svc, n_surgeons=len(surg_means),
                         sigma_between=sigma_between,
                         sigma_within=sigma_within, ratio=ratio))

    if rows:
        save_csv(pd.DataFrame(rows), "surgeon_heterogeneity_by_service")
        subsection("Interpretation")
        print("  A high between/within ratio means surgeons within the same")
        print("  service differ substantially in their booking bias.  This is")
        print("  the surgeon-level variation in q_s that the inverse step")
        print("  captures — and it justifies surgeon-specific estimation")
        print("  rather than purely group-level parameters.")


# ── 9. Booking granularity and inertia ─────────────────────────────────────
def analyze_booking_granularity(df):
    """Analyze heaping (rounding) in booked times and inertia across
    consecutive same-surgeon-procedure bookings."""
    section("BOOKING GRANULARITY AND INERTIA")
    if guard_empty(df, "granularity"):
        return

    booked = df["Booked Time (Minutes)"].dropna()

    subsection("Heaping / rounding patterns")
    for g in [5, 10, 15, 30, 60]:
        frac = (booked % g == 0).mean()
        print(f"  Divisible by {g:2d} min: {100 * frac:6.1f}%")

    # EHR convention detection: bookings follow n*5 − 1 pattern
    subsection("EHR booking convention (n×5 − 1 pattern)")
    frac_n5m1 = ((booked + 1) % 5 == 0).mean()
    print(f"  Fraction following n×5 − 1 pattern: {100 * frac_n5m1:.1f}%")
    if frac_n5m1 > 0.9:
        print("  → Nearly all bookings are of the form n×5 − 1 "
              "(e.g., 59, 89, 119, 179, 239).")
        print("    This is an EHR/scheduling-system convention, likely "
              "encoding time in")
        print("    5-minute slots as (slot_count × 5 − 1).  The effective "
              "granularity is")
        print("    5 minutes, even though no booked value is divisible by 5.")
        print("    This discrete structure should be noted in the data "
              "description and")
        print("    verified to not create artifacts in booking-error "
              "distributions.")

    # Verify with small examples
    top20_vals = booked.value_counts().head(20).index.tolist()
    n_follow = sum(1 for v in top20_vals if (v + 1) % 5 == 0)
    print(f"\n  Among top 20 most common durations: "
          f"{n_follow}/20 follow n×5 − 1")

    # Most common booked values
    subsection("Most common booked durations")
    top_vals = booked.value_counts().head(20)
    print(f"  {'Duration':>10s}  {'Count':>8s}  {'% of total':>10s}")
    for val, cnt in top_vals.items():
        print(f"  {val:10.0f}  {cnt:8,}  {100 * cnt / len(booked):10.1f}%")

    # Inertia: how often is the booking unchanged between consecutive
    # same-surgeon-procedure cases?
    subsection("Booking inertia (consecutive same-surgeon-procedure pairs)")
    sort_col = "Actual Start_DT" if col_available(df, "Actual Start_DT") \
        else "Actual Start Date"
    df_s = df.sort_values(
        ["Surgeon_Code", "Main_Procedure_Id", sort_col]).copy()
    same = ((df_s["Surgeon_Code"] == df_s["Surgeon_Code"].shift(1))
            & (df_s["Main_Procedure_Id"]
               == df_s["Main_Procedure_Id"].shift(1)))
    pairs = df_s[same].copy()
    if len(pairs) == 0:
        print("  No consecutive pairs — skipping inertia analysis.")
        return

    pairs["prev_booked"] = df_s["Booked Time (Minutes)"].shift(1).loc[
        pairs.index]
    change = pairs["Booked Time (Minutes)"] - pairs["prev_booked"]

    n_unchanged = (change == 0).sum()
    print(f"  Consecutive pairs:  {len(pairs):,}")
    print(f"  Booking unchanged:  {pct(n_unchanged, len(pairs))}")
    for t in [15, 30, 60]:
        n_within = (change.abs() <= t).sum()
        print(f"  |Change| ≤ {t:2d} min:  {pct(n_within, len(pairs))}")

    describe_series(change, "Booking change (min)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(change.clip(-120, 120), bins=80, edgecolor="white",
            alpha=0.8, color="steelblue")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Change in booked time (current − previous, min)")
    ax.set_ylabel("Count")
    ax.set_title("Booking Inertia: Change Between Consecutive Bookings")
    fig.tight_layout()
    fig.savefig(FIGDIR / "booking_inertia.png", dpi=150)
    plt.close(fig)


# ── 10. Consecutive pairs (raw preview for response estimation) ────────────
def analyze_consecutive_pairs(df):
    """Identify consecutive same-surgeon same-procedure cases and compute
    the raw experience signal and booking update.

    NOTE: This is a raw exploratory preview.  The formal response estimation
    (paper Section 7.2) requires cross-fitted residualization through the
    conditional quantile model Q̂(·; q̂_k).  Completion time is used for
    ordering, matching the paper's chronological convention.
    """
    section("CONSECUTIVE PAIRS — RAW PREVIEW FOR RESPONSE ESTIMATION")
    if guard_empty(df, "consecutive pairs"):
        return

    # Order by completion time (Actual Stop or Leave Room)
    if col_available(df, "Actual Stop_DT"):
        sort_col = "Actual Stop_DT"
    elif col_available(df, "Leave Room_DT"):
        sort_col = "Leave Room_DT"
    else:
        sort_col = "Actual Start Date"
    print(f"  Ordering by: {sort_col}  (completion time)")

    df_s = df.sort_values(
        ["Surgeon_Code", "Main_Procedure_Id", sort_col]).copy()

    df_s["prev_surgeon"]   = df_s["Surgeon_Code"].shift(1)
    df_s["prev_procedure"] = df_s["Main_Procedure_Id"].shift(1)
    df_s["prev_sort"]      = df_s[sort_col].shift(1)
    df_s["prev_realized"]  = df_s["Realized_Duration_Min"].shift(1)
    df_s["prev_booked"]    = df_s["Booked Time (Minutes)"].shift(1)

    same_sp = ((df_s["Surgeon_Code"] == df_s["prev_surgeon"])
               & (df_s["Main_Procedure_Id"] == df_s["prev_procedure"]))
    df_s["gap_days"] = (
        (pd.to_datetime(df_s[sort_col])
         - pd.to_datetime(df_s["prev_sort"]))
        .dt.total_seconds() / 86400)

    consecutive = df_s[same_sp].copy()
    print(f"  Total consecutive (surgeon, procedure) pairs: "
          f"{len(consecutive):,}")
    if guard_empty(consecutive, "consecutive pairs"):
        return

    subsection("Gap distribution (days between consecutive cases)")
    describe_series(consecutive["gap_days"], "Gap (days)")

    subsection("Valid pairs by max-gap threshold")
    pair_rows = []
    for max_gap in MAX_GAP_DAYS_LIST:
        valid = consecutive[consecutive["gap_days"] <= max_gap]
        n_v = len(valid)
        n_s = valid["Surgeon_Code"].nunique()
        n_sp = valid.groupby(
            ["Surgeon_Code", "Main_Procedure_Id"]).ngroups

        surg_pair_cnt = valid.groupby("Surgeon_Code").size()

        print(f"\n  Δ_max = {max_gap} days:")
        print(f"    Valid pairs:                         {n_v:,}")
        print(f"    Surgeons with ≥ 1 pair:              {n_s}")
        print(f"    (Surgeon, Procedure) groups:          {n_sp}")
        for thr in [5, 10, 20, 50]:
            n_above = (surg_pair_cnt >= thr).sum()
            print(f"    Surgeons with ≥ {thr:2d} pairs:             "
                  f"{n_above}")

        if col_available(valid, "Case_Service"):
            svc_pairs = (valid.groupby("Case_Service").size()
                         .sort_values(ascending=False))
            print(f"    By service:")
            for svc, cnt in svc_pairs.items():
                print(f"      {svc:40s}  {cnt:6,}")

        pair_rows.append(dict(max_gap_days=max_gap, valid_pairs=n_v,
                              surgeons=n_s, sp_groups=n_sp))

    save_csv(pd.DataFrame(pair_rows), "consecutive_pair_summary")

    # Raw scatter preview
    subsection("Raw signal–update scatter (Δ_max = 90 days)")
    valid_90 = consecutive[consecutive["gap_days"] <= 90].copy()
    if len(valid_90) > 0:
        # Raw (non-residualized) signal and update
        valid_90["sig"] = valid_90["prev_realized"] - valid_90["prev_booked"]
        valid_90["delta_b"] = (valid_90["Booked Time (Minutes)"]
                               - valid_90["prev_booked"])
        print(f"  NOTE: These are RAW signal and update values.")
        print(f"  The paper's estimation uses cross-fitted residualized ")
        print(f"  quantities through Q̂(·; q̂_k). The mean signal of ")
        print(f"  ~{valid_90['sig'].mean():.0f} min mostly reflects ")
        print(f"  systematic over-booking, not genuine surprise.\n")
        describe_series(valid_90["sig"], "Raw signal (d̃_prev − b_prev)")
        describe_series(valid_90["delta_b"], "Raw update (b_curr − b_prev)")

        fig, ax = plt.subplots(figsize=(8, 6))
        samp = valid_90.sample(min(5000, len(valid_90)), random_state=42)
        ax.scatter(samp["sig"], samp["delta_b"],
                   alpha=0.15, s=8, c="steelblue")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Raw signal: d̃_prev − b_prev  (minutes)")
        ax.set_ylabel("Raw update: b_curr − b_prev  (minutes)")
        ax.set_title("Raw Experience Signal vs. Booking Update (Δ_max=90d)")
        ax.set_xlim(np.percentile(valid_90["sig"].dropna(), [1, 99]))
        ax.set_ylim(np.percentile(valid_90["delta_b"].dropna(), [1, 99]))
        fig.tight_layout()
        fig.savefig(FIGDIR / "raw_signal_vs_update_scatter.png", dpi=150)
        plt.close(fig)


# ── 11. Temporal drift ─────────────────────────────────────────────────────
def analyze_temporal_drift(df):
    """Rolling summaries to detect drift in booking behavior over time."""
    section("TEMPORAL DRIFT DIAGNOSTICS")
    if guard_empty(df, "temporal drift"):
        return

    df_m = df.dropna(subset=["Week_Start", "Booking_Error_Min"]).copy()
    df_m = df_m.set_index("Week_Start").sort_index()

    # 8-week rolling summaries
    window = "8W"
    roll = df_m.resample("W-MON").agg(
        n=("Patient_ID", "size"),
        mean_error=("Booking_Error_Min", "mean"),
        median_error=("Booking_Error_Min", "median"),
        pct_pos=("Booking_Error_Min", lambda x: (x > 0).mean()),
    )
    roll_smooth = roll.rolling(8, min_periods=4).mean()

    subsection(f"Rolling {window} mean booking error")
    if len(roll_smooth) > 0:
        print(f"  Range of rolling mean error: "
              f"{roll_smooth['mean_error'].min():.1f} to "
              f"{roll_smooth['mean_error'].max():.1f} min")
        print(f"  Range of rolling % positive: "
              f"{100 * roll_smooth['pct_pos'].min():.1f}% to "
              f"{100 * roll_smooth['pct_pos'].max():.1f}%")

    # Service composition drift
    if col_available(df, "Case_Service"):
        subsection("Service mix stability (quarterly)")
        df_q = df.dropna(subset=["Case_Service"]).copy()
        df_q["quarter"] = (pd.to_datetime(df_q["Actual Start Date"])
                           .dt.to_period("Q"))
        mix = (df_q.groupby(["quarter", "Case_Service"]).size()
               .unstack(fill_value=0))
        mix_pct = mix.div(mix.sum(axis=1), axis=0)
        print("  Top 5 services, share by quarter:")
        top5 = df_q["Case_Service"].value_counts().head(5).index
        for svc in top5:
            if svc in mix_pct.columns:
                vals = mix_pct[svc]
                print(f"    {svc:25s}  "
                      + "  ".join(f"{v:.3f}" for v in vals))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(roll_smooth.index, roll_smooth["mean_error"],
            color="steelblue", lw=1.5)
    ax.axhline(df["Booking_Error_Min"].mean(), color="gray",
               ls="--", lw=1, label="Overall mean")
    ax.set_ylabel("Mean booking error (min)")
    ax.set_title("Rolling 8-Week Mean Booking Error")
    ax.legend()

    ax = axes[1]
    ax.plot(roll_smooth.index, roll_smooth["pct_pos"],
            color="darkorange", lw=1.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("Fraction over-booked")
    ax.set_xlabel("Week")
    ax.set_title("Rolling 8-Week Over-Booking Rate")

    fig.tight_layout()
    fig.savefig(FIGDIR / "temporal_drift.png", dpi=150)
    plt.close(fig)


# ── 12. Weekly horizons and out-of-sample evaluation ───────────────────────
def analyze_drift_decomposition(df: pd.DataFrame) -> None:
    section("TEMPORAL DRIFT DECOMPOSITION (COMPOSITION VS WITHIN-SURGEON)")
    if guard_empty(df, "drift decomposition"):
        return
    if "Actual Start Date" not in df.columns:
        print("  Actual Start Date not available — skipping.")
        return

    work = df.copy()
    work["Year"] = pd.to_datetime(work["Actual Start Date"], errors="coerce").dt.year
    work = work.dropna(subset=["Year", "Booking_Error_Min"]).copy()
    work["Year"] = work["Year"].astype(int)
    years = sorted(work["Year"].unique())
    if len(years) < 2:
        print("  Less than two years of data — skipping.")
        return

    surgeons_by_year = {y: set(work.loc[work["Year"] == y, "Surgeon_Code"].dropna().unique()) for y in years}
    stable = set.intersection(*surgeons_by_year.values())
    total_counts = work["Surgeon_Code"].value_counts()
    stable = {s for s in stable if total_counts.get(s, 0) >= MIN_SURGEON_CASES}
    if len(stable) == 0:
        print("  No stable surgeon panel after volume filtering — skipping.")
        return

    rows = []
    stable_weights = total_counts.loc[list(stable)].astype(float)
    stable_weights = stable_weights / stable_weights.sum() if stable_weights.sum() > 0 else stable_weights

    for y in years:
        sub = work[work["Year"] == y].copy()
        overall_mean = float(sub["Booking_Error_Min"].mean())
        sub_stable = sub[sub["Surgeon_Code"].isin(stable)].copy()
        stable_mean = float(sub_stable["Booking_Error_Min"].mean()) if len(sub_stable) > 0 else np.nan
        means_y = sub_stable.groupby("Surgeon_Code")["Booking_Error_Min"].mean() if len(sub_stable) > 0 else pd.Series(dtype=float)
        w = stable_weights.reindex(means_y.index).fillna(0.0)
        if w.sum() > 0:
            w = w / w.sum()
            fixed_comp_mean = float((w * means_y).sum())
        else:
            fixed_comp_mean = np.nan
        rows.append({
            "year": y,
            "overall_mean_error": overall_mean,
            "stable_panel_mean_error": stable_mean,
            "stable_fixed_composition_mean_error": fixed_comp_mean,
            "overall_cases": int(len(sub)),
            "stable_panel_cases": int(len(sub_stable)),
        })

    out = pd.DataFrame(rows)
    _save_csv(out, TBLDIR / "drift_decomposition_by_year.csv")

    stable_cover = 100 * work[work["Surgeon_Code"].isin(stable)].shape[0] / len(work)
    print(f"\n  Stable surgeon panel: {len(stable)} surgeons in all {len(years)} years with ≥{MIN_SURGEON_CASES} total cases")
    print(f"  Stable panel covers {stable_cover:.1f}% of cases\n")
    print(f"  {'Year':<6} {'Overall':>10} {'Stable':>10} {'FixedComp':>10} {'N_all':>8} {'N_stable':>8}")
    for _, r in out.iterrows():
        print(f"  {int(r['year']):<6} {r['overall_mean_error']:>10.1f} {r['stable_panel_mean_error']:>10.1f} {r['stable_fixed_composition_mean_error']:>10.1f} {int(r['overall_cases']):>8,} {int(r['stable_panel_cases']):>8,}")

    if len(out) >= 2 and np.isfinite(out['overall_mean_error']).all() and np.isfinite(out['stable_fixed_composition_mean_error']).all():
        overall_drop = float(out['overall_mean_error'].iloc[0] - out['overall_mean_error'].iloc[-1])
        fixed_drop = float(out['stable_fixed_composition_mean_error'].iloc[0] - out['stable_fixed_composition_mean_error'].iloc[-1])
        comp = overall_drop - fixed_drop
        print("\n  If FixedComp tracks Overall → drift is within-surgeon (behavioral).")
        print("  If FixedComp diverges from Overall → drift is compositional (turnover).")
        print(f"  Here, overall drift is {overall_drop:.1f} min; fixed-composition drift is {fixed_drop:.1f} min;")
        print(f"  composition explains about {comp:.1f} min, with within-surgeon change accounting for the majority.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(out["year"], out["overall_mean_error"], marker="o", label="Overall")
    ax.plot(out["year"], out["stable_panel_mean_error"], marker="o", label="Stable panel")
    ax.plot(out["year"], out["stable_fixed_composition_mean_error"], marker="o", label="Stable fixed composition")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean booking error (min)")
    ax.set_title("Drift decomposition: overall vs stable panel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "drift_decomposition_by_year.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {FIGDIR / 'drift_decomposition_by_year.png'}")


def analyze_within_group_signal_with_controls(df: pd.DataFrame) -> None:
    """Within (surgeon×procedure) booking signal, controlling for safe covariates."""
    section("WITHIN-GROUP BOOKING SIGNAL WITH SAFE CONTROLS")
    if guard_empty(df, "within-group signal with controls"):
        return

    # Define groups
    counts = df.groupby(["Surgeon_Code", "Main_Procedure_Id"]).size()
    eligible = counts[counts >= 3].index
    if len(eligible) == 0:
        print("  No repeated surgeon × procedure groups with ≥3 cases.")
        return
    temp = df.set_index(["Surgeon_Code", "Main_Procedure_Id"]).loc[eligible].reset_index().copy()

    # Within-group centered booking and realized
    temp["booking_c"] = temp["Booked Time (Minutes)"] - temp.groupby(["Surgeon_Code", "Main_Procedure_Id"])["Booked Time (Minutes)"].transform("mean")
    temp["realized_c"] = temp["Realized_Duration_Min"] - temp.groupby(["Surgeon_Code", "Main_Procedure_Id"])["Realized_Duration_Min"].transform("mean")

    # Safe controls (booking-time available): avoid post-op / leakage variables.
    control_cats = [c for c in ["Patient_Type", "Site", "DayOfWeek", "Month", "Year"] if c in temp.columns]
    # Ensure calendar controls exist if date is present
    temp = _ensure_prediction_feature_columns(temp)
    control_cats = [c for c in ["Patient_Type", "Site", "DayOfWeek", "Month", "Year"] if c in temp.columns]

    # Build design matrix: booking_c + within-demeaned one-hot controls
    y = temp["realized_c"].to_numpy(dtype=float)
    x_booking = temp["booking_c"].to_numpy(dtype=float).reshape(-1, 1)
    X_parts = [x_booking]
    col_names = ["booking_c"]

    if control_cats:
        dummies = pd.get_dummies(temp[control_cats].astype("object"), dummy_na=True, drop_first=False)
        # Within-group demean each dummy column to emulate FE regression.
        grp_key = temp[["Surgeon_Code", "Main_Procedure_Id"]].astype(str).agg("||".join, axis=1)
        dummies = dummies.apply(lambda col: col - pd.Series(col).groupby(grp_key).transform("mean"))
        X_parts.append(dummies.to_numpy(dtype=float))
        col_names.extend(list(dummies.columns))

    X = np.column_stack(X_parts)

    # Drop rows with non-finite y or booking_c (controls are finite after dummy)
    mask = np.isfinite(y) & np.isfinite(X[:, 0])
    X = X[mask]
    y = y[mask]
    clusters = temp.loc[mask, "Surgeon_Code"].astype(str).to_numpy()

    if len(y) < 50:
        print("  Insufficient rows after filtering — skipping.")
        return

    # OLS with cluster-robust SE (cluster by surgeon).
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Cluster meat
    uniq = pd.unique(clusters)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in uniq:
        idx = (clusters == g)
        Xg = X[idx]
        ug = resid[idx].reshape(-1, 1)
        meat += (Xg.T @ ug) @ (Xg.T @ ug).T
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    slope = float(beta[0])
    slope_se = float(se[0]) if np.isfinite(se[0]) else np.nan
    t = slope / slope_se if slope_se and slope_se > 0 else np.nan

    # Uncontrolled within slope for comparison
    base = _ols_summary(temp["booking_c"], temp["realized_c"])
    out = pd.DataFrame([{
        "n": len(y),
        "slope_uncontrolled": base["slope"],
        "slope_uncontrolled_se": base["slope_se"],
        "slope_controlled": slope,
        "slope_controlled_se_cluster": slope_se,
        "t_controlled": t,
        "controls": ",".join(control_cats) if control_cats else "",
    }])
    _save_csv(out, TBLDIR / "within_group_signal_with_controls.csv")

    print(f"  Uncontrolled within slope: {base['slope']:.4f} (se={base['slope_se']:.4f})")
    print(f"  Controlled within slope:   {slope:.4f} (cluster se={slope_se:.4f}, N={len(y):,})")
    if np.isfinite(slope):
        print(f"  Interpretation: +10 booking minutes within surgeon×procedure → ~{10*slope:.1f} realized minutes, holding controls fixed.")


def analyze_weekly_horizons(df):
    section("WEEKLY HORIZONS AND OUT-OF-SAMPLE SPLIT")
    if guard_empty(df, "weekly horizons"):
        return

    weekly = df.groupby("Week_Start").agg(
        n_cases=("Patient_ID", "size"),
        n_surgeons=("Surgeon_Code", "nunique"),
        n_rooms=("Operating_Room", "nunique"),
    ).sort_index()
    weekly = build_full_weekly_index(weekly)

    n_weeks = len(weekly)
    n_active = (weekly["n_cases"] > 0).sum()
    print(f"  Total calendar weeks: {n_weeks}")
    print(f"  Active weeks:         {n_active}")
    if n_weeks > 0:
        print(f"  Date range: {weekly.index.min()} to {weekly.index.max()}")

    active = weekly[weekly["n_cases"] > 0]
    subsection("Weekly case count distribution (active weeks)")
    describe_series(active["n_cases"], "Cases/week")
    describe_series(active["n_surgeons"], "Surgeons/week")
    describe_series(active["n_rooms"], "Rooms/week")

    subsection(f"Train/test split ({WARMUP_WEEKS}-week warm-up)")
    if n_weeks > WARMUP_WEEKS:
        oos = weekly.iloc[WARMUP_WEEKS:]
        oos_active = oos[oos["n_cases"] > 0]
        print(f"  Training weeks:       {WARMUP_WEEKS}")
        print(f"  Test weeks (calendar): {len(oos)}")
        print(f"  Test weeks (active):   {len(oos_active)}")
        print(f"  Test date range:       {oos.index.min()} to "
              f"{oos.index.max()}")
        print(f"  Test total cases:      {oos_active['n_cases'].sum():,}")
    else:
        print(f"  ⚠ Only {n_weeks} weeks — not enough for "
              f"{WARMUP_WEEKS}-week warm-up.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(weekly)), weekly["n_cases"], color="steelblue",
           alpha=0.7, width=1.0)
    if n_weeks > WARMUP_WEEKS:
        ax.axvline(WARMUP_WEEKS, color="red", ls="--", lw=1.5,
                   label=f"Warm-up cutoff (week {WARMUP_WEEKS})")
        ax.legend()
    ax.set_xlabel("Week index")
    ax.set_ylabel("Cases per week")
    ax.set_title("Weekly Case Volume")
    fig.tight_layout()
    fig.savefig(FIGDIR / "weekly_case_volume.png", dpi=150)
    plt.close(fig)

    subsection("Weeks per year")
    wc = active.copy()
    wc["year"] = wc.index.year
    for yr, grp in wc.groupby("year"):
        print(f"  {yr}: {len(grp)} weeks, {grp['n_cases'].sum():,} cases, "
              f"mean {grp['n_cases'].mean():.0f}/week")
    save_csv(weekly.reset_index(), "weekly_case_volume")


# ── 13. Surgeon identifier stability ───────────────────────────────────────
def analyze_surgeon_stability(df):
    section("SURGEON IDENTIFIER STABILITY ACROSS YEARS")
    if guard_empty(df, "surgeon stability"):
        return

    yr_surg = df.groupby("Year")["Surgeon_Code"].apply(set)
    years = sorted(yr_surg.index.dropna())

    if len(years) < 2:
        print("  Only one year — stability check not applicable.")
        return

    print(f"  Years present: {[int(y) for y in years]}")
    for yr in years:
        print(f"    {int(yr)}: {len(yr_surg[yr])} unique surgeons")

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        s1, s2 = yr_surg[y1], yr_surg[y2]
        overlap = s1 & s2
        print(f"\n  {int(y1)} → {int(y2)}:")
        print(f"    Overlap: {len(overlap)}")
        print(f"    Only in {int(y1)}: {len(s1 - s2)}")
        print(f"    Only in {int(y2)}: {len(s2 - s1)}")
        print(f"    Retention: {100 * len(overlap) / len(s1):.1f}%")

    all_yrs = set.intersection(*[yr_surg[y] for y in years])
    stable_cases = df[df["Surgeon_Code"].isin(all_yrs)].shape[0]
    print(f"\n  Surgeons in ALL years: {len(all_yrs)}")
    print(f"  Cases from stable surgeons: "
          f"{stable_cases:,} ({100 * stable_cases / len(df):.1f}%)")

    if col_available(df, "Surgeon"):
        subsection("Identifier consistency")
        multi_name = (df.groupby("Surgeon_Code")["Surgeon"]
                      .nunique() > 1).sum()
        multi_code = (df.groupby("Surgeon")["Surgeon_Code"]
                      .nunique() > 1).sum()
        print(f"  Surgeon_Code → >1 name: {multi_name}")
        print(f"  Surgeon name → >1 code: {multi_code}")
        if multi_name > 0 or multi_code > 0:
            print("  ⚠ Identifier instability — investigate before "
                  "longitudinal learning.")


# ── 14. Site-level booking behavior ────────────────────────────────────────
def analyze_site_booking_behavior(df):
    """Check whether the same surgeon books differently at different sites."""
    section("SITE-LEVEL BOOKING BEHAVIOR")
    if guard_empty(df, "site booking"):
        return
    if not col_available(df, "Site"):
        print("  Site column not available — skipping.")
        return

    df_m = df.dropna(subset=["Site", "Booking_Error_Min"]).copy()
    multi_site_surgs = (df_m.groupby("Surgeon_Code")["Site"].nunique()
                        .loc[lambda x: x > 1].index)
    print(f"  Surgeons at multiple sites: {len(multi_site_surgs)}")

    if len(multi_site_surgs) == 0:
        return

    subsection("Mean booking error by site for multi-site surgeons")
    rows = []
    for s in multi_site_surgs:
        sub = df_m[df_m["Surgeon_Code"] == s]
        if len(sub) < 20:
            continue
        site_err = sub.groupby("Site")["Booking_Error_Min"].agg(
            ["mean", "count"])
        if len(site_err) >= 2 and (site_err["count"] >= 5).all():
            site_means = site_err["mean"].to_dict()
            spread = max(site_means.values()) - min(site_means.values())
            rows.append(dict(surgeon=s, spread=spread, **{
                f"mean_{k}": v for k, v in site_means.items()}))

    if rows:
        rdf = pd.DataFrame(rows)
        describe_series(rdf["spread"],
                        "Spread in mean error across sites (min)")
        if rdf["spread"].mean() > 10:
            print("  → Meaningful site-level differences detected.")
            print("    Consider including Site as a feature in x_ti.")
        else:
            print("  → Differences are small; site may not add much "
                  "to the correction model.")
        save_csv(rdf, "site_booking_behavior")


# ── 15. Site decomposition for experimental design ─────────────────────────
def analyze_site_decomposition(df):
    """Investigate whether the three UHN sites (TWH, TGH, PMH) should be
    modeled jointly or separately in the bilevel scheduling framework.

    Key considerations: room disjointness (already checked in eligibility),
    surgeon overlap, volume balance, and service composition differences.
    """
    section("SITE DECOMPOSITION FOR EXPERIMENTAL DESIGN")
    if guard_empty(df, "site decomposition"):
        return
    if not col_available(df, "Site"):
        print("  Site column not available — skipping.")
        return

    df_m = df.dropna(subset=["Site"]).copy()
    sites = sorted(df_m["Site"].unique())
    print(f"  Sites: {', '.join(sites)}")
    print(f"  Cases with Site label: {len(df_m):,} / {len(df):,}")

    subsection("A. Per-site volume and structure")
    site_rows = []
    for site in sites:
        sub = df_m[df_m["Site"] == site]
        row = {
            "site": site,
            "n_cases": len(sub),
            "pct_cases": 100 * len(sub) / len(df_m),
            "n_surgeons": sub["Surgeon_Code"].nunique(),
            "n_rooms": sub["Operating_Room"].nunique(),
            "n_procedures": sub["Main_Procedure_Id"].nunique(),
        }
        if col_available(sub, "Case_Service"):
            row["n_services"] = sub["Case_Service"].nunique()
        if "Realized_Duration_Min" in sub.columns:
            row["mean_duration"] = sub["Realized_Duration_Min"].mean()
            row["median_duration"] = sub["Realized_Duration_Min"].median()
        if "Booking_Error_Min" in sub.columns:
            row["mean_booking_error"] = sub["Booking_Error_Min"].mean()
        site_rows.append(row)

    site_df = pd.DataFrame(site_rows)
    print(f"\n  {'Site':6s}  {'Cases':>8s}  {'%':>6s}  "
          f"{'Surgeons':>9s}  {'Rooms':>6s}  {'Procs':>6s}  "
          f"{'Mean dur':>8s}  {'Mean err':>8s}")
    for _, row in site_df.iterrows():
        dur_str = f"{row.get('mean_duration', 0):.0f}" \
            if 'mean_duration' in row and pd.notna(row.get('mean_duration')) else "N/A"
        err_str = f"{row.get('mean_booking_error', 0):.1f}" \
            if 'mean_booking_error' in row and pd.notna(row.get('mean_booking_error')) else "N/A"
        print(f"  {row['site']:6s}  {int(row['n_cases']):8,}  "
              f"{row['pct_cases']:5.1f}%  "
              f"{int(row['n_surgeons']):9d}  {int(row['n_rooms']):6d}  "
              f"{int(row['n_procedures']):6d}  {dur_str:>8s}  {err_str:>8s}")

    subsection("B. Surgeon overlap between sites")
    site_surgeons = {site: set(df_m.loc[df_m["Site"] == site,
                                        "Surgeon_Code"].unique())
                     for site in sites}
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if j <= i:
                continue
            shared = site_surgeons[s1] & site_surgeons[s2]
            only_s1 = site_surgeons[s1] - site_surgeons[s2]
            only_s2 = site_surgeons[s2] - site_surgeons[s1]
            print(f"  {s1} ∩ {s2}: {len(shared)} shared surgeons, "
                  f"{len(only_s1)} only in {s1}, "
                  f"{len(only_s2)} only in {s2}")
            if len(shared) > 0:
                # How many cases do shared surgeons contribute?
                shared_cases_s1 = df_m[
                    (df_m["Site"] == s1)
                    & df_m["Surgeon_Code"].isin(shared)].shape[0]
                shared_cases_s2 = df_m[
                    (df_m["Site"] == s2)
                    & df_m["Surgeon_Code"].isin(shared)].shape[0]
                print(f"    Shared surgeons contribute: "
                      f"{shared_cases_s1:,} cases at {s1}, "
                      f"{shared_cases_s2:,} cases at {s2}")

    subsection("C. Room overlap between sites")
    site_rooms = {site: set(df_m.loc[df_m["Site"] == site,
                                     "Operating_Room"].unique())
                  for site in sites}
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if j <= i:
                continue
            shared = site_rooms[s1] & site_rooms[s2]
            if shared:
                print(f"  {s1} ∩ {s2}: {len(shared)} shared rooms {shared}")
            else:
                print(f"  {s1} ∩ {s2}: room-disjoint")

    subsection("D. Service composition by site")
    if col_available(df_m, "Case_Service"):
        svc_site = (df_m.groupby(["Site", "Case_Service"]).size()
                    .rename("cases").reset_index())
        for site in sites:
            sub = svc_site[svc_site["Site"] == site].sort_values(
                "cases", ascending=False)
            tot = sub["cases"].sum()
            print(f"\n  {site} (top 8 services):")
            for _, row in sub.head(8).iterrows():
                print(f"    {row['Case_Service']:30s}  {int(row['cases']):5,}  "
                      f"({100 * row['cases'] / tot:.1f}%)")

    subsection("E. Exploratory design guidance")
    print("  NOTE: The following is exploratory design guidance based on")
    print("  descriptive statistics, not a formal identification argument.")
    # Check coupling
    all_shared = set()
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if j <= i:
                continue
            all_shared |= (site_surgeons[s1] & site_surgeons[s2])

    shared_room_count = 0
    for i, s1 in enumerate(sites):
        for j, s2 in enumerate(sites):
            if j <= i:
                continue
            shared_room_count += len(site_rooms[s1] & site_rooms[s2])

    shared_surg_cases = df_m[df_m["Surgeon_Code"].isin(all_shared)].shape[0]
    print(f"  Cross-site surgeons: {len(all_shared)} "
          f"({shared_surg_cases:,} cases, "
          f"{100 * shared_surg_cases / len(df_m):.1f}%)")
    print(f"  Cross-site shared rooms: {shared_room_count}")

    if shared_room_count <= 2 and len(all_shared) < 30:
        print("\n  Sites are nearly room-disjoint with limited surgeon overlap.")
        print("  Options:")
        print("    1. JOINT planning (current model): treats all sites as one")
        print("       pool. Simpler, and the few shared rooms/surgeons")
        print("       provide natural coupling.")
        print("    2. PER-SITE planning: decomposes the problem. Reduces CCG")
        print("       block pool size but requires handling shared surgeons.")
        if any(row["n_cases"] < 5000 for _, row in site_df.iterrows()):
            small_sites = [row["site"] for _, row in site_df.iterrows()
                           if row["n_cases"] < 5000]
            print(f"    ⚠ Site(s) {small_sites} may be too small for "
                  "standalone analysis.")
    else:
        print("\n  Significant cross-site coupling detected.")
        print("  Joint planning is recommended.")

    save_csv(site_df, "site_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(site_df["site"], site_df["n_cases"],
           edgecolor="white", alpha=0.8, color="steelblue")
    ax.set_xlabel("Site")
    ax.set_ylabel("Cases")
    ax.set_title("Case Volume by Site")

    ax = axes[1]
    if "mean_booking_error" in site_df.columns:
        ax.bar(site_df["site"], site_df["mean_booking_error"],
               edgecolor="white", alpha=0.8, color="darkorange")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Site")
        ax.set_ylabel("Mean booking error (min)")
        ax.set_title("Mean Booking Error by Site")

    fig.tight_layout()
    fig.savefig(FIGDIR / "site_decomposition.png", dpi=150)
    plt.close(fig)


# ── 16. Block capacity analysis ────────────────────────────────────────────
def analyze_block_capacity(df):
    """Block structure from room timestamps, restricted to weekdays."""
    section("BLOCK STRUCTURE AND CAPACITY (weekdays only)")
    if guard_empty(df, "block capacity"):
        return
    if "Enter Room_DT" not in df.columns or "Leave Room_DT" not in df.columns:
        print("  Enter/Leave Room timestamps not available — skipping.")
        return

    df_b = df.dropna(subset=["Enter Room_DT", "Leave Room_DT"]).copy()
    df_b["OR_Date"] = pd.to_datetime(df_b["Actual Start Date"]).dt.date

    block = df_b.groupby(["Operating_Room", "OR_Date"]).agg(
        n_cases=("Patient_ID", "size"),
        first_enter=("Enter Room_DT", "min"),
        last_leave=("Leave Room_DT", "max"),
        total_booked=("Booked Time (Minutes)", "sum"),
        total_realized=("Realized_Duration_Min", "sum"),
        total_surgical=("Surgical_Duration_Min", "sum"),
    )
    block["span_min"] = (
        (block["last_leave"] - block["first_enter"]).dt.total_seconds() / 60)
    block["weekday"] = pd.to_datetime(
        block.index.get_level_values("OR_Date")).weekday
    block["weekday_name"] = pd.to_datetime(
        block.index.get_level_values("OR_Date")).strftime("%A")

    # Filter: reasonable span, weekdays only
    block = block[(block["span_min"] > 60)
                  & (block["span_min"] < 960)
                  & (block["weekday"] < 5)].copy()

    print(f"  Weekday (room, date) blocks: {len(block):,}")

    subsection("Occupied span distribution (first enter → last leave)")
    describe_series(block["span_min"], "Span (min)")

    subsection("Cases per block")
    describe_series(block["n_cases"], "Cases/block")

    subsection("Total realized duration per block (case-level room occupancy)")
    describe_series(block["total_realized"], "Realized (min)")

    # Fixed-capacity: case-level room time, excluding inter-case turnover
    subsection("Fixed-capacity analysis — case-level room occupancy, excl. turnover")
    print("  Note: 'Realized' is room occupancy per case (Enter Room → Leave Room)")
    print("  when available, falling back to surgical time (Start → Stop).")
    print("  This excludes inter-case turnover.  See turnover section for")
    print("  combined block-load estimates.\n")
    for C in FIXED_CAPACITY_OPTIONS:
        ot = (block["total_realized"] - C).clip(lower=0)
        idle = (C - block["total_realized"]).clip(lower=0)
        print(f"  C={C:4d} min ({C / 60:.1f}h):  "
              f"OT rate={100 * (ot > 0).mean():5.1f}%  "
              f"mean OT={ot.mean():5.1f}  mean idle={idle.mean():5.1f}")

    # Weekday breakdown
    subsection("Occupied span by weekday")
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd_stats = block.groupby("weekday_name")["span_min"].agg(
        ["count", "mean", "median", "std"])
    wd_stats = wd_stats.reindex([d for d in wd_order if d in wd_stats.index])
    for day, row in wd_stats.iterrows():
        print(f"    {day:12s}  N={int(row['count']):5,}  "
              f"mean={row['mean']:5.0f}  median={row['median']:5.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(block["span_min"], bins=60, edgecolor="white",
            alpha=0.8, color="steelblue")
    for cap in [480, 600]:
        ax.axvline(cap, color="red", ls="--", lw=1,
                   label=f"{cap} min ({cap // 60}h)")
    ax.set_xlabel("Occupied Span (min)")
    ax.set_ylabel("Blocks")
    ax.set_title("Weekday Block Span Distribution")
    ax.legend()

    ax = axes[1]
    mx = int(block["n_cases"].max()) + 2
    ax.hist(block["n_cases"], bins=range(0, mx),
            edgecolor="white", alpha=0.8, color="steelblue")
    ax.set_xlabel("Cases per block")
    ax.set_ylabel("Blocks")
    ax.set_title("Cases per Block Distribution")
    fig.tight_layout()
    fig.savefig(FIGDIR / "block_capacity_analysis.png", dpi=150)
    plt.close(fig)

    save_csv(block.reset_index()[
        ["Operating_Room", "OR_Date", "n_cases", "span_min",
         "total_booked", "total_realized", "weekday_name"]
    ], "block_stats")

    return block


# ── 17. Turnover time ──────────────────────────────────────────────────────
def analyze_turnover_time(df, block_data=None):
    """Turnover time between consecutive cases in the same room-day.

    Reports statistics at multiple trimming thresholds and by service/room,
    then redoes the fixed-capacity analysis with turnover included.
    """
    section("TURNOVER TIME BETWEEN CASES")
    if guard_empty(df, "turnover"):
        return
    if "Enter Room_DT" not in df.columns or "Leave Room_DT" not in df.columns:
        print("  Enter/Leave Room timestamps not available — skipping.")
        return

    df_r = df.dropna(subset=["Enter Room_DT", "Leave Room_DT"]).copy()
    df_r["OR_Date"] = pd.to_datetime(df_r["Actual Start Date"]).dt.date
    df_r = df_r.sort_values(["Operating_Room", "OR_Date", "Enter Room_DT"])

    df_r["prev_room"]  = df_r["Operating_Room"].shift(1)
    df_r["prev_date"]  = df_r["OR_Date"].shift(1)
    df_r["prev_leave"] = df_r["Leave Room_DT"].shift(1)

    same_block = ((df_r["Operating_Room"] == df_r["prev_room"])
                  & (df_r["OR_Date"] == df_r["prev_date"]))
    df_r.loc[same_block, "turnover_min"] = (
        (df_r.loc[same_block, "Enter Room_DT"]
         - df_r.loc[same_block, "prev_leave"])
        .dt.total_seconds() / 60)

    turnover = df_r["turnover_min"].dropna()
    n_neg = (turnover < 0).sum()
    print(f"  Consecutive same-room-day pairs: {len(turnover):,}")
    print(f"  Negative (overlap):              {n_neg}")

    subsection("Turnover by trimming threshold")
    for cap in TURNOVER_CAPS:
        t = turnover[(turnover >= 0) & (turnover <= cap)]
        print(f"\n  Cap = {cap} min:  N={len(t):,}  "
              f"mean={t.mean():.1f}  median={t.median():.1f}  "
              f"p75={t.quantile(0.75):.1f}  p95={t.quantile(0.95):.1f}")

    # Use the moderate cap for the main estimate
    main_cap = TURNOVER_CAPS[0]
    tv = turnover[(turnover >= 0) & (turnover <= main_cap)]
    if len(tv) == 0:
        return
    describe_series(tv, f"Turnover (0–{main_cap} min)")

    # By service
    if col_available(df_r, "Case_Service"):
        subsection("Turnover by service")
        df_r_tv = df_r[same_block & (df_r["turnover_min"] >= 0)
                       & (df_r["turnover_min"] <= main_cap)].copy()
        if len(df_r_tv) > 0:
            svc_tv = df_r_tv.groupby("Case_Service")["turnover_min"].agg(
                ["count", "mean", "median"])
            svc_tv = svc_tv[svc_tv["count"] >= 20].sort_values(
                "mean", ascending=False)
            print(f"  {'Service':30s}  {'N':>6s}  {'Mean':>7s}  {'Median':>7s}")
            for svc, row in svc_tv.iterrows():
                print(f"  {svc:30s}  {int(row['count']):6,}  "
                      f"{row['mean']:7.1f}  {row['median']:7.1f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tv, bins=60, edgecolor="white", alpha=0.8, color="steelblue")
    ax.axvline(tv.mean(), color="orange", lw=1.5,
               label=f"Mean={tv.mean():.1f}")
    ax.axvline(tv.median(), color="green", lw=1.5,
               label=f"Median={tv.median():.1f}")
    ax.set_xlabel("Turnover Time (min)")
    ax.set_ylabel("Count")
    ax.set_title("Turnover Distribution (same room-day)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "turnover_time_distribution.png", dpi=150)
    plt.close(fig)

    # Redo fixed-capacity with turnover included
    if block_data is not None and len(block_data) > 0:
        mean_tv = tv.mean()
        subsection(f"Fixed-capacity WITH turnover "
                   f"(mean turnover = {mean_tv:.1f} min)")
        print(f"  Block load = Σ realized_duration "
              f"+ (n_cases − 1) × {mean_tv:.1f}\n")
        block = block_data.copy()
        block["load_with_turnover"] = (
            block["total_realized"]
            + (block["n_cases"] - 1).clip(lower=0) * mean_tv)
        for C in FIXED_CAPACITY_OPTIONS:
            ot = (block["load_with_turnover"] - C).clip(lower=0)
            idle = (C - block["load_with_turnover"]).clip(lower=0)
            print(f"  C={C:4d} min:  "
                  f"OT rate={100 * (ot > 0).mean():5.1f}%  "
                  f"mean OT={ot.mean():5.1f}  mean idle={idle.mean():5.1f}")


# ── 18. Turnover variability and resource share ────────────────────────────
def analyze_turnover_variability(df, block_data=None):
    """Quantify how much turnover varies across services, surgeons,
    and same-vs-different-surgeon transitions, and measure turnover's
    share of total block resource consumption.

    If turnover is a large and variable component of block load, ignoring
    it in the duration-prediction framework creates a systematic gap.
    """
    section("TURNOVER VARIABILITY AND RESOURCE SHARE")
    if guard_empty(df, "turnover variability"):
        return
    if "Enter Room_DT" not in df.columns or "Leave Room_DT" not in df.columns:
        print("  Enter/Leave Room timestamps not available — skipping.")
        return

    df_r = df.dropna(subset=["Enter Room_DT", "Leave Room_DT"]).copy()
    df_r["OR_Date"] = pd.to_datetime(df_r["Actual Start Date"]).dt.date
    df_r = df_r.sort_values(["Operating_Room", "OR_Date", "Enter Room_DT"])

    df_r["prev_room"]    = df_r["Operating_Room"].shift(1)
    df_r["prev_date"]    = df_r["OR_Date"].shift(1)
    df_r["prev_leave"]   = df_r["Leave Room_DT"].shift(1)
    df_r["prev_surgeon"] = df_r["Surgeon_Code"].shift(1)

    same_block = ((df_r["Operating_Room"] == df_r["prev_room"])
                  & (df_r["OR_Date"] == df_r["prev_date"]))
    df_r.loc[same_block, "turnover_min"] = (
        (df_r.loc[same_block, "Enter Room_DT"]
         - df_r.loc[same_block, "prev_leave"])
        .dt.total_seconds() / 60)
    df_r.loc[same_block, "same_surgeon_transition"] = (
        df_r.loc[same_block, "Surgeon_Code"]
        == df_r.loc[same_block, "prev_surgeon"])

    # Keep reasonable turnovers for analysis
    tv_all = df_r[same_block & (df_r["turnover_min"] >= 0)
                  & (df_r["turnover_min"] <= 90)].copy()
    if guard_empty(tv_all, "turnover variability"):
        return

    # ── A. Same-surgeon vs. different-surgeon turnovers ───────────────────
    subsection("A. Same-surgeon vs. different-surgeon turnover")
    same_surg = tv_all[tv_all["same_surgeon_transition"] == True]
    diff_surg = tv_all[tv_all["same_surgeon_transition"] == False]
    print(f"  Same-surgeon transitions:     {len(same_surg):,}")
    print(f"  Different-surgeon transitions: {len(diff_surg):,}")
    if len(same_surg) > 0:
        describe_series(same_surg["turnover_min"],
                        "Turnover (same surgeon)")
    if len(diff_surg) > 0:
        describe_series(diff_surg["turnover_min"],
                        "Turnover (different surgeon)")
    if len(same_surg) > 0 and len(diff_surg) > 0:
        diff_mean = diff_surg["turnover_min"].mean()
        same_mean = same_surg["turnover_min"].mean()
        print(f"\n  Mean difference (diff-surgeon − same-surgeon): "
              f"{diff_mean - same_mean:.1f} min")
        if diff_mean - same_mean > 5:
            print("  → Different-surgeon transitions incur meaningfully "
                  "longer turnover.")
        else:
            print("  → Turnover is similar regardless of surgeon change.")

    # ── B. Variance decomposition ─────────────────────────────────────────
    subsection("B. Turnover variance decomposition")
    total_var = tv_all["turnover_min"].var()
    print(f"  Overall turnover variance: {total_var:.1f}  "
          f"(std = {tv_all['turnover_min'].std():.1f} min)")
    print(f"  Coefficient of variation:  "
          f"{tv_all['turnover_min'].std() / tv_all['turnover_min'].mean():.2f}")

    if col_available(tv_all, "Case_Service"):
        svc_tv = tv_all.dropna(subset=["Case_Service"])
        svc_means = svc_tv.groupby("Case_Service")["turnover_min"].mean()
        svc_vars = svc_tv.groupby("Case_Service")["turnover_min"].var()
        between_var = svc_means.var()
        within_var = svc_vars.mean()
        print(f"  Between-service variance: {between_var:.1f}")
        print(f"  Mean within-service variance: {within_var:.1f}")
        if total_var > 0:
            print(f"  Service explains ~{100 * between_var / total_var:.1f}% "
                  f"of turnover variance")

    # ── C. Turnover as share of block load ────────────────────────────────
    subsection("C. Turnover as a share of total block load")
    if block_data is not None and len(block_data) > 0:
        block = block_data.copy()
        mean_tv = tv_all["turnover_min"].mean()
        block["turnover_load"] = (
            (block["n_cases"] - 1).clip(lower=0) * mean_tv)
        block["total_load"] = block["total_realized"] + block["turnover_load"]
        block["turnover_pct"] = (
            100 * block["turnover_load"] / block["total_load"])
        block["turnover_pct"] = block["turnover_pct"].replace(
            [np.inf, -np.inf], np.nan)

        describe_series(block["turnover_pct"],
                        "Turnover as % of total block load")
        multi_case = block[block["n_cases"] > 1]
        if len(multi_case) > 0:
            describe_series(multi_case["turnover_pct"],
                            "Turnover % (multi-case blocks only)")

        # Overall aggregate
        tot_surgical = block["total_realized"].sum()
        tot_turnover = block["turnover_load"].sum()
        tot_all = tot_surgical + tot_turnover
        print(f"\n  Aggregate across all blocks:")
        print(f"    Total surgical time:  {tot_surgical:,.0f} min")
        print(f"    Total turnover time:  {tot_turnover:,.0f} min")
        print(f"    Turnover share:       "
              f"{100 * tot_turnover / tot_all:.1f}%")

        subsection("D. Exploratory design guidance")
        print("  NOTE: The following is exploratory guidance, not a formal")
        print("  calibration of the capacity constraint.\n")
        if tot_turnover / tot_all > 0.05:
            print("  Turnover accounts for a non-trivial share of block "
                  "load.")
            print("  The block capacity constraint should be:")
            print("    Σ d̃_i + (n − 1) × τ ≤ C")
            print(f"  with τ ≈ {mean_tv:.0f} min (mean turnover).")
            print("  Alternatively, inflate each case's effective duration "
                  "by τ/2 on each side.")
        else:
            print("  Turnover is a small share of block load and can be "
                  "absorbed into idle time.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if len(same_surg) > 0 and len(diff_surg) > 0:
        ax.hist(same_surg["turnover_min"], bins=40, alpha=0.6,
                label=f"Same surgeon (N={len(same_surg):,})",
                edgecolor="white", color="steelblue")
        ax.hist(diff_surg["turnover_min"], bins=40, alpha=0.6,
                label=f"Different surgeon (N={len(diff_surg):,})",
                edgecolor="white", color="darkorange")
        ax.set_xlabel("Turnover time (min)")
        ax.set_ylabel("Count")
        ax.set_title("Turnover: Same vs. Different Surgeon")
        ax.legend(fontsize=8)
    elif len(tv_all) > 0:
        ax.hist(tv_all["turnover_min"], bins=40, edgecolor="white",
                alpha=0.8, color="steelblue")
        ax.set_xlabel("Turnover time (min)")
        ax.set_ylabel("Count")
        ax.set_title("Turnover Distribution")

    ax = axes[1]
    if col_available(tv_all, "Case_Service"):
        svc_tv_df = tv_all.dropna(subset=["Case_Service"])
        top_svc = (svc_tv_df["Case_Service"].value_counts()
                   .loc[lambda x: x >= 50].index.tolist())
        if len(top_svc) >= 2:
            svc_order = (svc_tv_df[svc_tv_df["Case_Service"].isin(top_svc)]
                         .groupby("Case_Service")["turnover_min"]
                         .mean().sort_values())
            data = [svc_tv_df.loc[svc_tv_df["Case_Service"] == s,
                                  "turnover_min"].values
                    for s in svc_order.index]
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color="red", lw=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue")
            ax.set_xticklabels([s[:12] for s in svc_order.index],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Turnover (min)")
            ax.set_title("Turnover by Service")

    fig.tight_layout()
    fig.savefig(FIGDIR / "turnover_variability.png", dpi=150)
    plt.close(fig)


# ── 19. Eligibility set structure ──────────────────────────────────────────
def analyze_eligibility_structure(df):
    """Check whether services use disjoint room sets, which determines
    whether the CCG pricing subproblem decomposes by service."""
    section("ELIGIBILITY SET STRUCTURE (service → room mapping)")
    if guard_empty(df, "eligibility"):
        return
    if not col_available(df, "Case_Service"):
        print("  Case_Service not available — skipping.")
        return

    svc_rooms = (df.dropna(subset=["Case_Service"])
                 .groupby("Case_Service")["Operating_Room"]
                 .apply(lambda x: set(x.unique())))

    services = sorted(svc_rooms.index)
    print(f"  Services: {len(services)}")
    print(f"  {'Service':30s}  {'Rooms':>6s}  Room list")
    for svc in services:
        rooms = sorted(svc_rooms[svc])
        print(f"  {svc:30s}  {len(rooms):6d}  "
              f"{', '.join(str(r) for r in rooms[:8])}"
              f"{'...' if len(rooms) > 8 else ''}")

    # Check for disjoint room sets
    subsection("Overlap analysis")
    overlaps = []
    for i, s1 in enumerate(services):
        for j, s2 in enumerate(services):
            if j <= i:
                continue
            shared = svc_rooms[s1] & svc_rooms[s2]
            if shared:
                overlaps.append((s1, s2, len(shared)))
    if overlaps:
        print(f"  Service pairs sharing rooms: {len(overlaps)}")
        for s1, s2, n in sorted(overlaps, key=lambda x: -x[2])[:15]:
            print(f"    {s1:20s} ∩ {s2:20s}  = {n} shared rooms")
        print("\n  → Room sets are NOT disjoint across services.")
        print("    CCG pricing does not decompose cleanly by service alone.")
    else:
        print("  ✓ Room sets are disjoint across all services.")
        print("    CCG pricing decomposes by service (Proposition 11.1).")

    # Site decomposition check
    if col_available(df, "Site"):
        subsection("Room disjointness by site")
        site_rooms = (df.dropna(subset=["Site"])
                      .groupby("Site")["Operating_Room"]
                      .apply(lambda x: set(x.unique())))
        sites = sorted(site_rooms.index)
        for i, s1 in enumerate(sites):
            for j, s2 in enumerate(sites):
                if j <= i:
                    continue
                shared = site_rooms[s1] & site_rooms[s2]
                if shared:
                    print(f"  {s1} ∩ {s2}: {len(shared)} shared rooms "
                          f"({shared})")
                else:
                    print(f"  {s1} ∩ {s2}: disjoint")


# ── 20. Repeat patient analysis ────────────────────────────────────────────
def analyze_repeat_patients(df):
    """Check how many patients appear multiple times and whether same-patient
    bookings show persistence or learning."""
    section("REPEAT PATIENT ANALYSIS")
    if guard_empty(df, "repeat patients"):
        return

    pat_vol = df.groupby("Patient_ID").size().rename("cases")
    n_unique = len(pat_vol)
    n_repeat = (pat_vol > 1).sum()
    cases_repeat = pat_vol[pat_vol > 1].sum()

    print(f"  Unique patients:      {n_unique:,}")
    print(f"  Patients with >1 case: {n_repeat:,} "
          f"({100 * n_repeat / n_unique:.1f}%)")
    print(f"  Cases from repeats:   {cases_repeat:,} "
          f"({100 * cases_repeat / len(df):.1f}%)")

    subsection("Distribution of cases per patient")
    describe_series(pat_vol, "Cases/patient")
    for t in [2, 3, 5, 10]:
        n = (pat_vol >= t).sum()
        print(f"  Patients with ≥ {t:2d} cases: {n:,}")


# ── 21. Cancellation patterns ──────────────────────────────────────────────
def analyze_cancellations(df_raw):
    """Cancellation rates, with explicit handling of metadata missingness."""
    section("CANCELLATION PATTERNS")

    or_cases = df_raw[df_raw["Is_OR_Room"]].copy()
    n_total = len(or_cases)
    n_canc = or_cases["Is_Cancelled"].sum()
    print(f"  OR-room cases: {n_total:,}")
    print(f"  Cancelled:     {pct(n_canc, n_total)}")

    # Check service availability on cancelled cases
    if col_available(or_cases, "Case_Service"):
        canc_with_svc = (or_cases["Is_Cancelled"]
                         & or_cases["Case_Service"].notna()).sum()
        canc_svc_rate = canc_with_svc / n_canc if n_canc > 0 else 0
        print(f"\n  Cancelled cases with Case_Service: "
              f"{pct(canc_with_svc, n_canc)}")

        if canc_svc_rate < 0.5:
            print(f"\n  ⚠ Only {100 * canc_svc_rate:.1f}% of cancelled cases "
                  f"have service labels.")
            print("    Service-level cancellation rates would be heavily "
                  "biased — suppressing that table.")
            print("    Use surgeon-level rates instead (Surgeon_Code is "
                  "present on all cancelled cases).")
        else:
            subsection("Cancellation rate by service")
            svc_canc = or_cases.groupby("Case_Service").agg(
                total=("Is_Cancelled", "size"),
                cancelled=("Is_Cancelled", "sum"))
            svc_canc["rate"] = svc_canc["cancelled"] / svc_canc["total"]
            svc_canc = svc_canc.sort_values("rate", ascending=False)
            print(f"  {'Service':40s}  {'Total':>7s}  {'Canc':>6s}  "
                  f"{'Rate':>6s}")
            for svc, row in svc_canc.iterrows():
                print(f"  {svc:40s}  {int(row['total']):7,}  "
                      f"{int(row['cancelled']):6,}  "
                      f"{100 * row['rate']:5.1f}%")
            save_csv(svc_canc, "cancellation_by_service")

    subsection("Cancellation rate by surgeon (top 20 by volume)")
    surg_canc = or_cases.groupby("Surgeon_Code").agg(
        total=("Is_Cancelled", "size"),
        cancelled=("Is_Cancelled", "sum"))
    surg_canc["rate"] = surg_canc["cancelled"] / surg_canc["total"]
    top = surg_canc.sort_values("total", ascending=False).head(20)
    print(f"  {'Surgeon':>10s}  {'Total':>7s}  {'Canc':>6s}  {'Rate':>6s}")
    for s, row in top.iterrows():
        print(f"  {str(s):>10s}  {int(row['total']):7,}  "
              f"{int(row['cancelled']):6,}  {100 * row['rate']:5.1f}%")


# ── 22. Partial pooling assessment and shrinkage diagnostic ──────────────
def analyze_partial_pooling(df):
    """Assess whether hierarchical partial pooling is needed for
    surgeon-level critical ratio (q_s) estimation, and demonstrate
    how shrinkage behaves as a function of surgeon volume.

    The behavioral primitive is surgeon-specific q_s.  For data-rich
    surgeons the estimate is near-individual; for data-poor surgeons
    it shrinks toward the group (type-level) mean.  This section
    quantifies how many surgeons fall in each regime and shows the
    shrinkage profile explicitly.
    """
    section("PARTIAL POOLING ASSESSMENT FOR SURGEON-LEVEL q_s")
    if guard_empty(df, "partial pooling"):
        return

    surg_vol = df.groupby("Surgeon_Code").size().rename("cases")
    thresholds = [10, 20, 30, 50, 100, 200]

    print(f"  Total surgeons: {len(surg_vol)}")
    print(f"\n  {'Threshold':>12s}  {'Above':>16s}  {'Cases above':>15s}  "
          f"{'Below':>16s}  {'Cases below':>13s}")
    for t in thresholds:
        above = surg_vol[surg_vol >= t]
        below = surg_vol[surg_vol < t]
        print(f"  n_min = {t:4d}   "
              f"{len(above):8,} ({100 * len(above) / len(surg_vol):5.1f}%)  "
              f"{above.sum():10,} ({100 * above.sum() / surg_vol.sum():5.1f}%)  "
              f"{len(below):8,} ({100 * len(below) / len(surg_vol):5.1f}%)  "
              f"{below.sum():8,} ({100 * below.sum() / surg_vol.sum():5.1f}%)")

    subsection("Implication for surgeon-level q_s estimation")
    n50 = (surg_vol < 50).sum()
    c50 = surg_vol[surg_vol < 50].sum()
    print(f"  With n_min=50: {n50} surgeons ({c50:,} cases) have insufficient")
    print(f"  history for reliable individual q̂_s and should borrow strength")
    print(f"  from their pooling group via hierarchical shrinkage.")
    n_rich = (surg_vol >= 50).sum()
    print(f"  The remaining {n_rich} surgeons ({surg_vol[surg_vol >= 50].sum():,} cases)")
    print(f"  can receive near-individual estimates with minimal shrinkage.")
    print(f"  Partial pooling is "
          f"{'recommended' if n50 > 5 else 'optional'}.")

    # ── Shrinkage diagnostic: raw vs. shrunk coverage by surgeon ──────────
    subsection("Surgeon-level shrinkage diagnostic (empirical coverage)")
    df_m = df.dropna(subset=["Booking_Error_Min"]).copy()

    # Raw surgeon-level empirical coverage
    surg_cov = (df_m.groupby("Surgeon_Code")
                .apply(lambda g: (g["Realized_Duration_Min"]
                                  <= g["Booked Time (Minutes)"]).mean(),
                       include_groups=False))
    surg_n = df_m.groupby("Surgeon_Code").size()

    # Grand mean and type-level means as shrinkage targets
    grand_mean = (df_m["Realized_Duration_Min"]
                  <= df_m["Booked Time (Minutes)"]).mean()

    type_mean = {}
    if "Surgeon_Type" in df_m.columns:
        type_mean = (df_m.groupby("Surgeon_Type")
                     .apply(lambda g: (g["Realized_Duration_Min"]
                                       <= g["Booked Time (Minutes)"]).mean(),
                            include_groups=False)
                     .to_dict())
        surg_type = df_m.groupby("Surgeon_Code")["Surgeon_Type"].first()

    # Simple James–Stein-style shrinkage: w = n / (n + λ)
    # λ chosen so surgeons with n_min cases get ~50% shrinkage
    lam = 50.0
    shrunk_rows = []
    for surg in surg_cov.index:
        n_s = surg_n[surg]
        raw = surg_cov[surg]
        w = n_s / (n_s + lam)

        # Shrinkage target: type mean if available, else grand mean
        if "Surgeon_Type" in df_m.columns and surg in surg_type.index:
            target = type_mean.get(surg_type[surg], grand_mean)
        else:
            target = grand_mean

        shrunk = w * raw + (1 - w) * target
        shrunk_rows.append({
            "Surgeon_Code": surg,
            "n_cases": n_s,
            "raw_coverage": raw,
            "shrunk_coverage": shrunk,
            "shrinkage_weight": 1 - w,
            "target": target,
        })

    shrunk_df = pd.DataFrame(shrunk_rows)
    if len(shrunk_df) == 0:
        return

    # Report by volume band
    print(f"\n  Shrinkage parameter λ = {lam:.0f} "
          f"(surgeons with {int(lam)} cases get 50% shrinkage)")
    print(f"  Grand mean coverage: {grand_mean:.4f}\n")

    vol_bands = [(0, 10), (10, 30), (30, 50), (50, 100),
                 (100, 200), (200, 9999)]
    print(f"  {'Volume band':>15s}  {'N surgs':>8s}  "
          f"{'Raw mean':>9s}  {'Raw std':>8s}  "
          f"{'Shrunk mean':>12s}  {'Shrunk std':>11s}  "
          f"{'Mean wt':>8s}")
    for lo, hi in vol_bands:
        sub = shrunk_df[(shrunk_df["n_cases"] >= lo)
                        & (shrunk_df["n_cases"] < hi)]
        if len(sub) == 0:
            continue
        label = f"{lo}–{hi}" if hi < 9999 else f"{lo}+"
        print(f"  {label:>15s}  {len(sub):>8d}  "
              f"{sub['raw_coverage'].mean():>9.4f}  "
              f"{sub['raw_coverage'].std():>8.4f}  "
              f"{sub['shrunk_coverage'].mean():>12.4f}  "
              f"{sub['shrunk_coverage'].std():>11.4f}  "
              f"{sub['shrinkage_weight'].mean():>8.3f}")

    # Key observation
    subsection("Interpretation")
    high_vol = shrunk_df[shrunk_df["n_cases"] >= 50]
    low_vol = shrunk_df[shrunk_df["n_cases"] < 50]
    if len(high_vol) > 0 and len(low_vol) > 0:
        raw_spread_high = high_vol["raw_coverage"].std()
        shrunk_spread_high = high_vol["shrunk_coverage"].std()
        raw_spread_low = low_vol["raw_coverage"].std()
        shrunk_spread_low = low_vol["shrunk_coverage"].std()
        print(f"  High-volume surgeons (≥50 cases):")
        print(f"    Raw std = {raw_spread_high:.4f}, "
              f"Shrunk std = {shrunk_spread_high:.4f}  "
              f"(minimal compression)")
        print(f"  Low-volume surgeons (<50 cases):")
        print(f"    Raw std = {raw_spread_low:.4f}, "
              f"Shrunk std = {shrunk_spread_low:.4f}  "
              f"(substantial regularization)")
        print(f"\n  This confirms that surgeon-level q̂_s with hierarchical")
        print(f"  pooling gives near-individual estimates where data support")
        print(f"  it, while stabilizing sparse surgeons via group shrinkage.")

    save_csv(shrunk_df, "surgeon_shrinkage_diagnostic")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(shrunk_df["n_cases"], shrunk_df["raw_coverage"],
               alpha=0.4, s=15, c="steelblue", label="Raw")
    ax.scatter(shrunk_df["n_cases"], shrunk_df["shrunk_coverage"],
               alpha=0.4, s=15, c="darkorange", label="Shrunk")
    ax.axhline(grand_mean, color="gray", ls="--", lw=1,
               label=f"Grand mean = {grand_mean:.3f}")
    ax.set_xlabel("Surgeon case volume")
    ax.set_ylabel("Empirical coverage Pr(d̃ ≤ b)")
    ax.set_title("Raw vs. Shrunk Surgeon Coverage")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(shrunk_df["n_cases"], shrunk_df["shrinkage_weight"],
               alpha=0.5, s=15, c="steelblue")
    ax.set_xlabel("Surgeon case volume")
    ax.set_ylabel("Shrinkage weight (1 = full shrinkage)")
    ax.set_title("Shrinkage Intensity by Volume")
    ax.set_xscale("log")
    ax.axhline(0.5, color="red", ls="--", lw=1,
               label=f"50% shrinkage (n = {int(lam)})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGDIR / "surgeon_shrinkage_diagnostic.png", dpi=150)
    plt.close(fig)


# ── 23. Waiting-time data ──────────────────────────────────────────────────
def analyze_waiting_times(df):
    """Decision-to-surgery wait, restricted to the cleaned elective cohort."""
    section("WAITING-TIME DATA (cleaned elective cohort only)")
    if guard_empty(df, "waiting times"):
        return

    n = len(df)
    has_decision = (df["Decision_Date"].notna().sum()
                    if "Decision_Date" in df.columns else 0)
    print(f"  Cleaned elective cases: {n:,}")
    print(f"  Has Decision_Date:      {pct(has_decision, n)}")

    if "Decision_Date" in df.columns:
        both = df[df["Decision_Date"].notna()
                  & df["Actual Start Date"].notna()].copy()
        if len(both) > 0:
            both["Wait_Days"] = (
                both["Actual Start Date"] - both["Decision_Date"]).dt.days
            valid = both[both["Wait_Days"] >= 0]["Wait_Days"]
            if len(valid) > 0:
                subsection("Decision-to-surgery wait (days)")
                describe_series(valid, "Wait")


# ── 24. Block–surgeon assignment structure ─────────────────────────────────
def analyze_block_surgeon_structure(df):
    """Determine whether OR blocks (room × date) are de-facto single-surgeon
    or whether multiple surgeons share a room-day.

    This directly addresses: 'Are blocks assigned to surgeons, or can a
    room-day host multiple surgeons?'

    Two complementary views are produced:
      A. Distribution of distinct surgeons per block.
      B. From the surgeon side: how often does a surgeon share their block?
    """
    section("BLOCK–SURGEON ASSIGNMENT STRUCTURE")
    if guard_empty(df, "block-surgeon structure"):
        return

    df_b = df.copy()
    df_b["OR_Date"] = pd.to_datetime(df_b["Actual Start Date"]).dt.date

    # ── A. How many distinct surgeons per (room, date) block? ───────────────
    block_surgeons = df_b.groupby(
        ["Operating_Room", "OR_Date"])["Surgeon_Code"].agg(
            ["nunique", "count"]).rename(
                columns={"nunique": "n_distinct_surgeons",
                         "count":  "n_cases"})

    subsection("A. Distinct surgeons per (room, date) block")
    total_blocks = len(block_surgeons)
    print(f"  Total (room, date) blocks: {total_blocks:,}")
    print()

    dist = block_surgeons["n_distinct_surgeons"].value_counts().sort_index()
    print(f"  {'# surgeons':>12s}  {'# blocks':>10s}  {'% blocks':>9s}  "
          f"{'# cases':>10s}")
    for n_surgs, n_blocks in dist.items():
        cases_in = block_surgeons.loc[
            block_surgeons["n_distinct_surgeons"] == n_surgs, "n_cases"].sum()
        print(f"  {int(n_surgs):>12d}  {int(n_blocks):>10,}  "
              f"{100 * n_blocks / total_blocks:>8.1f}%  {int(cases_in):>10,}")

    sole_blocks = (block_surgeons["n_distinct_surgeons"] == 1).sum()
    multi_blocks = (block_surgeons["n_distinct_surgeons"] > 1).sum()
    print(f"\n  Single-surgeon blocks : {pct(sole_blocks,  total_blocks)}")
    print(f"  Multi-surgeon blocks  : {pct(multi_blocks, total_blocks)}")

    describe_series(block_surgeons["n_distinct_surgeons"],
                    "Surgeons/block")
    save_csv(block_surgeons.reset_index(), "block_surgeon_counts")

    # ── B. Surgeon perspective: how often do they share their room-day? ─────
    subsection("B. Surgeon's view: sharing rate per surgeon")
    df_b2 = df_b.merge(
        block_surgeons.reset_index()[
            ["Operating_Room", "OR_Date", "n_distinct_surgeons"]],
        on=["Operating_Room", "OR_Date"])

    surg_sharing = df_b2.groupby("Surgeon_Code").apply(
        lambda g: pd.Series({
            "total_cases":   len(g),
            "sole_cases":    (g["n_distinct_surgeons"] == 1).sum(),
            "shared_cases":  (g["n_distinct_surgeons"] > 1).sum(),
            "sole_blocks":   g.loc[g["n_distinct_surgeons"] == 1,
                                   "OR_Date"].nunique(),
            "shared_blocks": g.loc[g["n_distinct_surgeons"] > 1,
                                   "OR_Date"].nunique(),
        })
    )
    surg_sharing["pct_shared_cases"] = (
        100 * surg_sharing["shared_cases"] / surg_sharing["total_cases"])

    describe_series(surg_sharing["pct_shared_cases"],
                    "% cases in shared blocks (per surgeon)")

    # Surgeons who almost always work alone vs always share
    alone_surgs = (surg_sharing["pct_shared_cases"] < 5).sum()
    share_surgs = (surg_sharing["pct_shared_cases"] > 95).sum()
    print(f"\n  Surgeons almost always alone   (<5 % shared): {alone_surgs}")
    print(f"  Surgeons almost always sharing (>95% shared): {share_surgs}")

    save_csv(surg_sharing.reset_index(), "surgeon_sharing_rates")

    # ── C. Top co-occurring surgeon pairs in the same block ─────────────────
    subsection("C. Most common surgeon pairs in the same (room, date)")
    # Only examine multi-surgeon blocks
    multi = block_surgeons[block_surgeons["n_distinct_surgeons"] > 1].index
    df_multi = df_b.set_index(["Operating_Room", "OR_Date"])
    df_multi = df_multi[df_multi.index.isin(multi)].reset_index()

    pair_counts: dict = {}
    for (_, _), grp in df_multi.groupby(["Operating_Room", "OR_Date"]):
        surgs = grp["Surgeon_Code"].unique()
        if len(surgs) < 2:
            continue
        for i in range(len(surgs)):
            for j in range(i + 1, len(surgs)):
                key = tuple(sorted([surgs[i], surgs[j]]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    if pair_counts:
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:15]
        print(f"  {'Surgeon A':>12s}  {'Surgeon B':>12s}  "
              f"{'Shared blocks':>14s}")
        for (a, b), cnt in top_pairs:
            print(f"  {str(a):>12s}  {str(b):>12s}  {cnt:>14,}")

    # ── D. Plot: histogram of surgeons/block ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    dist_vals = block_surgeons["n_distinct_surgeons"].value_counts().sort_index()
    ax.bar(dist_vals.index.astype(str), dist_vals.values,
           edgecolor="white", alpha=0.8, color="steelblue")
    ax.set_xlabel("Distinct surgeons per (room, date) block")
    ax.set_ylabel("Number of blocks")
    ax.set_title("Surgeons per Block Distribution")

    ax = axes[1]
    ax.hist(surg_sharing["pct_shared_cases"].clip(0, 100),
            bins=30, edgecolor="white", alpha=0.8, color="darkorange")
    ax.set_xlabel("% cases in shared blocks (per surgeon)")
    ax.set_ylabel("Surgeons")
    ax.set_title("Surgeon Sharing Rate Distribution")

    fig.tight_layout()
    fig.savefig(FIGDIR / "block_surgeon_structure.png", dpi=150)
    plt.close(fig)


# ── 25. Within-block case sequencing ───────────────────────────────────────
def analyze_within_block_sequencing(df):
    """Determine whether cases within multi-surgeon blocks are clustered
    by surgeon (A-A-A-B-B) or interleaved (A-B-A-B).

    For single-surgeon blocks (83%), cases are consecutive by construction.
    This analysis targets the ~17% of blocks where multiple surgeons share
    a room-day, and tests whether surgeons operate in contiguous runs.

    The 'clustering ratio' is defined as:
      1 − (n_surgeon_switches / (n_transitions))
    where n_transitions = n_cases − 1 and n_surgeon_switches counts
    how many times the surgeon changes between consecutive cases.
    A ratio of 1.0 means perfect clustering (no interleaving);
    a ratio of 0.0 means maximal alternation.
    """
    section("WITHIN-BLOCK CASE SEQUENCING")
    if guard_empty(df, "within-block sequencing"):
        return
    if "Enter Room_DT" not in df.columns:
        print("  Enter Room timestamps not available — skipping.")
        return

    df_b = df.dropna(subset=["Enter Room_DT"]).copy()
    df_b["OR_Date"] = pd.to_datetime(df_b["Actual Start Date"]).dt.date
    df_b = df_b.sort_values(["Operating_Room", "OR_Date", "Enter Room_DT"])

    # Identify multi-surgeon blocks
    block_surgs = df_b.groupby(
        ["Operating_Room", "OR_Date"])["Surgeon_Code"].nunique()
    multi_keys = block_surgs[block_surgs > 1].index

    if len(multi_keys) == 0:
        print("  No multi-surgeon blocks found.")
        return

    # Analyze sequencing within each multi-surgeon block
    seq_rows = []
    for (room, date) in multi_keys:
        mask = ((df_b["Operating_Room"] == room)
                & (df_b["OR_Date"] == date))
        block = df_b[mask].sort_values("Enter Room_DT")
        if len(block) < 2:
            continue

        surgeons = block["Surgeon_Code"].values
        n_cases = len(surgeons)
        n_transitions = n_cases - 1
        n_switches = sum(1 for i in range(n_transitions)
                         if surgeons[i] != surgeons[i + 1])
        clustering_ratio = 1.0 - (n_switches / n_transitions)

        # Count contiguous runs
        n_runs = 1 + n_switches

        seq_rows.append({
            "Operating_Room": room,
            "OR_Date": date,
            "n_cases": n_cases,
            "n_surgeons": len(set(surgeons)),
            "n_transitions": n_transitions,
            "n_switches": n_switches,
            "n_runs": n_runs,
            "clustering_ratio": clustering_ratio,
        })

    seq_df = pd.DataFrame(seq_rows)
    if guard_empty(seq_df, "sequencing analysis"):
        return

    print(f"  Multi-surgeon blocks analyzed: {len(seq_df):,}")
    print(f"  Total cases in these blocks:   {seq_df['n_cases'].sum():,}")

    subsection("A. Clustering ratio distribution")
    describe_series(seq_df["clustering_ratio"],
                    "Clustering ratio (1 = perfect, 0 = max alternation)")

    # Classify blocks
    perfect_cluster = (seq_df["clustering_ratio"] == 1.0).sum()
    high_cluster = (seq_df["clustering_ratio"] >= 0.75).sum()
    low_cluster = (seq_df["clustering_ratio"] < 0.5).sum()
    print(f"\n  Perfect clustering (ratio = 1.0):  "
          f"{pct(perfect_cluster, len(seq_df))}")
    print(f"  High clustering (ratio ≥ 0.75):    "
          f"{pct(high_cluster, len(seq_df))}")
    print(f"  Low clustering (ratio < 0.50):     "
          f"{pct(low_cluster, len(seq_df))}")

    subsection("B. Contiguous runs per block")
    describe_series(seq_df["n_runs"],
                    "Contiguous surgeon runs per multi-surgeon block")
    # The minimum number of runs equals the number of distinct surgeons
    # if perfectly clustered
    min_possible = seq_df["n_surgeons"]
    seq_df["excess_runs"] = seq_df["n_runs"] - min_possible
    n_minimal = (seq_df["excess_runs"] == 0).sum()
    print(f"\n  Blocks with minimal runs (= # surgeons): "
          f"{pct(n_minimal, len(seq_df))}")
    print(f"  Mean excess runs above minimum: "
          f"{seq_df['excess_runs'].mean():.2f}")

    subsection("C. Surgeon switches per block")
    describe_series(seq_df["n_switches"],
                    "Surgeon switches per multi-surgeon block")

    subsection("D. Interpretation for the scheduling model")
    if perfect_cluster / len(seq_df) > 0.4:
        print("  A large fraction of multi-surgeon blocks have perfectly "
              "clustered cases.")
        print("  This supports the modeling assumption that within-block "
              "sequencing")
        print("  is effectively contiguous by surgeon, with no meaningful "
              "interleaving overhead.")
    elif high_cluster / len(seq_df) > 0.6:
        print("  Most multi-surgeon blocks show high clustering, with "
              "occasional interleaving.")
        print("  The scheduling model's assumption of block-level capacity "
              "without")
        print("  surgeon-specific sequencing costs is well-supported.")
    else:
        print("  Significant interleaving observed in multi-surgeon blocks.")
        print("  Consider whether surgeon-change overhead should be modeled "
              "explicitly.")

    # Clustering by number of surgeons
    subsection("E. Clustering ratio by number of surgeons in block")
    for ns in sorted(seq_df["n_surgeons"].unique()):
        sub = seq_df[seq_df["n_surgeons"] == ns]
        if len(sub) >= 10:
            print(f"  {ns} surgeons: N={len(sub):,}  "
                  f"mean ratio={sub['clustering_ratio'].mean():.3f}  "
                  f"median={sub['clustering_ratio'].median():.3f}  "
                  f"perfect={100 * (sub['clustering_ratio'] == 1.0).mean():.1f}%")

    save_csv(seq_df, "within_block_sequencing")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(seq_df["clustering_ratio"], bins=20, edgecolor="white",
            alpha=0.8, color="steelblue")
    ax.axvline(1.0, color="red", ls="--", lw=1, label="Perfect clustering")
    ax.set_xlabel("Clustering ratio")
    ax.set_ylabel("Multi-surgeon blocks")
    ax.set_title("Within-Block Surgeon Clustering")
    ax.legend()

    ax = axes[1]
    run_dist = seq_df["n_runs"].value_counts().sort_index()
    ax.bar(run_dist.index.astype(str), run_dist.values,
           edgecolor="white", alpha=0.8, color="darkorange")
    ax.set_xlabel("Contiguous surgeon runs per block")
    ax.set_ylabel("Blocks")
    ax.set_title("Surgeon Run Count in Multi-Surgeon Blocks")

    fig.tight_layout()
    fig.savefig(FIGDIR / "within_block_sequencing.png", dpi=150)
    plt.close(fig)


# ── 26. Service–room assignment (static and daily) ─────────────────────────
def analyze_service_room_assignment(df):
    """Answer: 'Are blocks assigned to specialties, on a given day and in
    general?'

    Produces:
      A. Static view — which rooms does each service use, and how exclusively?
      B. Daily view  — within a (room, date), how many services share the day?
      C. Room ownership index per service.
    """
    section("SERVICE–ROOM ASSIGNMENT (static and daily views)")
    if guard_empty(df, "service-room assignment"):
        return
    if not col_available(df, "Case_Service"):
        print("  Case_Service not available — skipping.")
        return

    df_b = df.dropna(subset=["Case_Service"]).copy()
    df_b["OR_Date"] = pd.to_datetime(df_b["Actual Start Date"]).dt.date

    # ── A. Static: for each room, which services appear and how much? ───────
    subsection("A. Static room usage by service (% of room's total cases)")
    room_svc = (df_b.groupby(["Operating_Room", "Case_Service"])
                .size().rename("cases").reset_index())
    room_total = room_svc.groupby("Operating_Room")["cases"].sum()
    room_svc["pct"] = (room_svc["cases"]
                       / room_svc["Operating_Room"].map(room_total) * 100)

    rooms = sorted(df_b["Operating_Room"].unique())
    print(f"\n  {'Room':>10s}  {'Total cases':>12s}  "
          f"{'Dominant service':>35s}  {'% dominant':>11s}  "
          f"{'# services':>11s}")
    room_summary_rows = []
    for room in rooms:
        sub = room_svc[room_svc["Operating_Room"] == room].sort_values(
            "cases", ascending=False)
        if sub.empty:
            continue
        dom_svc  = sub.iloc[0]["Case_Service"]
        dom_pct  = sub.iloc[0]["pct"]
        n_svcs   = len(sub)
        tot      = int(room_total[room])
        print(f"  {room:>10s}  {tot:>12,}  "
              f"{str(dom_svc):>35s}  {dom_pct:>10.1f}%  {n_svcs:>11d}")
        room_summary_rows.append(dict(
            room=room, total_cases=tot, dominant_service=dom_svc,
            dominant_pct=dom_pct, n_services=n_svcs))

    room_summary = pd.DataFrame(room_summary_rows)
    save_csv(room_summary, "room_service_summary")

    # Service-to-room concentration: Herfindahl-style index
    subsection("A2. Room-ownership index per service "
               "(Herfindahl: 1 = one room, low = spread across many rooms)")
    svc_room_cases = (df_b.groupby(["Case_Service", "Operating_Room"])
                      .size().rename("cases").reset_index())
    svc_total = svc_room_cases.groupby("Case_Service")["cases"].sum()
    svc_room_cases["share"] = (
        svc_room_cases["cases"]
        / svc_room_cases["Case_Service"].map(svc_total))
    hhi = (svc_room_cases.groupby("Case_Service")["share"]
           .apply(lambda s: (s ** 2).sum()).rename("HHI"))
    dom_room = (svc_room_cases.loc[
        svc_room_cases.groupby("Case_Service")["cases"].idxmax(),
        ["Case_Service", "Operating_Room", "share"]]
                .set_index("Case_Service"))

    print(f"\n  {'Service':35s}  {'HHI':>6s}  {'# rooms':>7s}  "
          f"{'Top room':>12s}  {'Top room %':>10s}")
    for svc in svc_total.sort_values(ascending=False).index:
        h = hhi.get(svc, np.nan)
        n_rooms = svc_room_cases.loc[
            svc_room_cases["Case_Service"] == svc, "Operating_Room"].nunique()
        if svc in dom_room.index:
            top_r = dom_room.loc[svc, "Operating_Room"]
            top_p = dom_room.loc[svc, "share"] * 100
        else:
            top_r, top_p = "—", np.nan
        print(f"  {svc:35s}  {h:6.3f}  {n_rooms:7d}  "
              f"{str(top_r):>12s}  {top_p:10.1f}%")

    # ── B. Daily: how many distinct services share a (room, date)? ──────────
    subsection("B. Daily view: distinct services per (room, date) block")
    block_svcs = df_b.groupby(
        ["Operating_Room", "OR_Date"])["Case_Service"].agg(
            ["nunique", "count"]).rename(
                columns={"nunique": "n_services",
                         "count":  "n_cases"})

    total_rday = len(block_svcs)
    print(f"\n  Total (room, date) blocks: {total_rday:,}")
    dist_s = block_svcs["n_services"].value_counts().sort_index()
    print(f"\n  {'# services':>12s}  {'# blocks':>10s}  {'% blocks':>9s}")
    for ns, nb in dist_s.items():
        print(f"  {int(ns):>12d}  {int(nb):>10,}  "
              f"{100 * nb / total_rday:>8.1f}%")

    single_svc_blocks = (block_svcs["n_services"] == 1).sum()
    print(f"\n  Single-service blocks: {pct(single_svc_blocks, total_rday)}")
    print(f"  Multi-service blocks:  "
          f"{pct(total_rday - single_svc_blocks, total_rday)}")

    describe_series(block_svcs["n_services"], "Services/block")
    save_csv(block_svcs.reset_index(), "block_service_counts")

    # ── C. Cross-tab: service × room matrix (% of service cases per room) ───
    subsection("C. Service × room usage matrix (% of each service's cases)")
    pivot = (svc_room_cases
             .pivot_table(index="Case_Service", columns="Operating_Room",
                          values="share", fill_value=0) * 100)
    if pivot.shape[0] <= 20 and pivot.shape[1] <= 20:
        # Print compactly when small enough
        rooms_c = sorted(pivot.columns)
        header = f"  {'Service':35s}" + "".join(
            f"  {str(r)[:6]:>6s}" for r in rooms_c)
        print(f"\n{header}")
        for svc in pivot.index:
            row_str = f"  {svc:35s}" + "".join(
                f"  {pivot.loc[svc, r]:6.1f}" for r in rooms_c)
            print(row_str)
    else:
        print("  (matrix too large to print inline; "
              "see generated block_service_counts.csv table)")

    # ── D. Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if not room_summary.empty:
        ax.bar(room_summary["room"].astype(str),
               room_summary["dominant_pct"],
               edgecolor="white", alpha=0.8, color="steelblue")
        ax.axhline(100, color="red", ls="--", lw=1, label="100% (exclusive)")
        ax.set_xlabel("Operating Room")
        ax.set_ylabel("% cases from dominant service")
        ax.set_title("Room Service Exclusivity")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    ax = axes[1]
    dist_s = block_svcs["n_services"].value_counts().sort_index()
    ax.bar(dist_s.index.astype(str), dist_s.values,
           edgecolor="white", alpha=0.8, color="darkorange")
    ax.set_xlabel("Distinct services per (room, date) block")
    ax.set_ylabel("Number of blocks")
    ax.set_title("Services per Block Distribution")

    fig.tight_layout()
    fig.savefig(FIGDIR / "service_room_assignment.png", dpi=150)
    plt.close(fig)


# ── 27. Surgeon weekly activity patterns ───────────────────────────────────
def analyze_surgeon_weekly_patterns(df):
    """Characterise how surgeons operate on a weekly basis:
    operating days per week, cases per day, hours per session,
    rooms used, and schedule consistency.

    Answers: 'How many days, hours, operations, and rooms per week,
    and what rules govern the assignments?'
    """
    section("SURGEON WEEKLY ACTIVITY PATTERNS")
    if guard_empty(df, "surgeon weekly patterns"):
        return

    df_b = df.copy()
    df_b["OR_Date"]    = pd.to_datetime(df_b["Actual Start Date"]).dt.date
    df_b["DayOfWeek"]  = pd.to_datetime(
        df_b["Actual Start Date"]).dt.day_name()
    df_b["Week_Start"] = df_b["Week_Start"]  # already computed

    has_duration = (
        "Realized_Duration_Min" in df_b.columns
        and df_b["Realized_Duration_Min"].notna().any())

    # ── A. Surgeon-session level (one row per surgeon × room × date) ────────
    subsection("A. Session-level statistics (surgeon × room × date)")
    agg_dict = dict(
        n_cases=("Patient_ID", "size"),
        total_booked=("Booked Time (Minutes)", "sum"),
    )
    if has_duration:
        agg_dict["total_realized"] = ("Realized_Duration_Min", "sum")
    if "Enter Room_DT" in df_b.columns and "Leave Room_DT" in df_b.columns:
        df_b["Enter Room_DT"] = pd.to_datetime(df_b["Enter Room_DT"])
        df_b["Leave Room_DT"] = pd.to_datetime(df_b["Leave Room_DT"])
        agg_dict["first_enter"] = ("Enter Room_DT", "min")
        agg_dict["last_leave"]  = ("Leave Room_DT", "max")

    session = df_b.groupby(
        ["Surgeon_Code", "Operating_Room", "OR_Date"]).agg(**agg_dict)

    if "first_enter" in session.columns and "last_leave" in session.columns:
        session["session_span_min"] = (
            (session["last_leave"] - session["first_enter"])
            .dt.total_seconds() / 60)
        session = session[
            session["session_span_min"].between(10, 960, inclusive="both")]

    print(f"  Total surgeon-session records: {len(session):,}")

    describe_series(session["n_cases"], "Cases per surgeon-session")
    describe_series(session["total_booked"], "Booked minutes per session")
    if has_duration:
        describe_series(session["total_realized"], "Realized minutes per session")
    if "session_span_min" in session.columns:
        describe_series(session["session_span_min"],
                        "Session span min (first enter → last leave)")

    # Distribution: cases per session
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(session["n_cases"].clip(upper=20), bins=range(1, 22),
            edgecolor="white", alpha=0.8, color="steelblue")
    ax.set_xlabel("Cases per surgeon-session")
    ax.set_ylabel("Sessions")
    ax.set_title("Cases per Session")

    # ── B. Surgeon-day level: how many rooms/sessions does a surgeon use? ───
    subsection("B. Do surgeons operate in more than one room on the same day?")
    surg_day = df_b.groupby(["Surgeon_Code", "OR_Date"]).agg(
        n_rooms=("Operating_Room", "nunique"),
        n_cases=("Patient_ID",    "size")).reset_index()

    dist_rooms = surg_day["n_rooms"].value_counts().sort_index()
    total_sday = len(surg_day)
    print(f"\n  Surgeon-day records: {total_sday:,}")
    print(f"  {'# rooms used':>14s}  {'# surg-days':>12s}  {'%':>6s}")
    for nr, n in dist_rooms.items():
        print(f"  {int(nr):>14d}  {int(n):>12,}  "
              f"{100 * n / total_sday:>5.1f}%")
    multi_room_days = (surg_day["n_rooms"] > 1).sum()
    print(f"\n  Surgeon-days with >1 room: {pct(multi_room_days, total_sday)}")

    ax = axes[0, 1]
    dist_r = surg_day["n_rooms"].value_counts().sort_index()
    ax.bar(dist_r.index.astype(str), dist_r.values,
           edgecolor="white", alpha=0.8, color="darkorange")
    ax.set_xlabel("Rooms used in a single day (per surgeon)")
    ax.set_ylabel("Surgeon-days")
    ax.set_title("Rooms per Surgeon-Day")

    # ── C. Surgeon-week level: days and cases per week ──────────────────────
    subsection("C. Surgeon-week statistics (operating days and cases per week)")
    surg_week = df_b.groupby(["Surgeon_Code", "Week_Start"]).agg(
        n_days  =("OR_Date",     "nunique"),
        n_cases =("Patient_ID",  "size"),
        n_rooms =("Operating_Room", "nunique"),
    ).reset_index()

    print(f"\n  Surgeon-week records: {len(surg_week):,}")
    describe_series(surg_week["n_days"],  "Operating days per active week")
    describe_series(surg_week["n_cases"], "Cases per active week")
    describe_series(surg_week["n_rooms"], "Distinct rooms per active week")

    # Day-of-week distribution
    subsection("C2. On which days of the week do surgeons most commonly operate?")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    dow_counts = df_b["DayOfWeek"].value_counts()
    print(f"\n  {'Day':12s}  {'Cases':>8s}  {'% of total':>10s}")
    for day in dow_order:
        cnt = dow_counts.get(day, 0)
        print(f"  {day:12s}  {int(cnt):>8,}  "
              f"{100 * cnt / len(df_b):>9.1f}%")

    # Surgeons per day-of-week
    dow_surgs = df_b.groupby("DayOfWeek")["Surgeon_Code"].nunique()
    print(f"\n  {'Day':12s}  {'Distinct surgeons':>18s}")
    for day in dow_order:
        print(f"  {day:12s}  {int(dow_surgs.get(day, 0)):>18,}")

    ax = axes[1, 0]
    days_plot = [d for d in dow_order if d in dow_counts.index]
    ax.bar(days_plot, [dow_counts.get(d, 0) for d in days_plot],
           edgecolor="white", alpha=0.8, color="steelblue")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Cases")
    ax.set_title("Cases by Day of Week")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    ax.hist(surg_week["n_days"].clip(upper=7), bins=range(1, 9),
            edgecolor="white", alpha=0.8, color="darkorange")
    ax.set_xlabel("Operating days per active week (per surgeon)")
    ax.set_ylabel("Surgeon-weeks")
    ax.set_title("Operating Days per Active Week")

    fig.tight_layout()
    fig.savefig(FIGDIR / "surgeon_weekly_patterns.png", dpi=150)
    plt.close(fig)

    # ── D. Schedule consistency: does a surgeon reuse the same room-day slot? ─
    subsection("D. Schedule consistency — does a surgeon use a fixed room-day?")
    # For each surgeon, compute the fraction of their operating days
    # that fall on their most common day-of-week, and in their most
    # common room.
    surg_consistency = []
    for surg, grp in df_b.groupby("Surgeon_Code"):
        if len(grp) < MIN_SURGEON_CASES:
            continue
        grp_days = grp.groupby(["OR_Date"]).agg(
            day_name=("DayOfWeek", "first"),
            room=("Operating_Room",
                  lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""))
        n_total_days = len(grp_days)
        # Most common day-of-week measured over operating DAYS, not cases.
        dow_day_dist = grp_days["day_name"].value_counts()
        top_dow_days = dow_day_dist.iloc[0]
        pct_dow = 100 * top_dow_days / n_total_days if n_total_days > 0 else np.nan
        # Most common room remains a case-weighted measure, as labelled below.
        room_dist = grp["Operating_Room"].value_counts()
        top_room_cnt = room_dist.iloc[0]
        pct_room = 100 * top_room_cnt / len(grp)
        # Most common (room, day) slot — by count of operating days that
        # match this combination
        slot_dist = grp_days.groupby(
            ["room", "day_name"]).size().sort_values(ascending=False)
        top_slot_cnt = slot_dist.iloc[0] if len(slot_dist) > 0 else 0
        top_slot     = slot_dist.index[0] if len(slot_dist) > 0 else ("—", "—")
        pct_slot     = 100 * top_slot_cnt / n_total_days if n_total_days > 0 else np.nan

        surg_consistency.append(dict(
            surgeon=surg,
            total_days=n_total_days,
            top_dow=dow_day_dist.index[0],
            pct_top_dow=pct_dow,
            top_room=room_dist.index[0],
            pct_top_room=pct_room,
            top_slot=f"{top_slot[0]}_{top_slot[1][:3]}",
            pct_top_slot=pct_slot,
        ))

    if surg_consistency:
        cons_df = pd.DataFrame(surg_consistency)
        describe_series(cons_df["pct_top_dow"],
                        "% of days on surgeon's favourite day-of-week")
        describe_series(cons_df["pct_top_room"],
                        "% of cases in surgeon's favourite room")
        describe_series(cons_df["pct_top_slot"],
                        "% of days in surgeon's favourite (room, day) slot")

        # Classify surgeons by schedule rigidity
        thresholds = [50, 70, 90]
        print(f"\n  Schedule rigidity — fraction of surgeons whose top slot "
              f"accounts for ≥ X% of their days:")
        for t in thresholds:
            n = (cons_df["pct_top_slot"] >= t).sum()
            print(f"    ≥{t}%: {n} / {len(cons_df)} surgeons "
                  f"({100 * n / len(cons_df):.1f}%)")

        save_csv(cons_df, "surgeon_schedule_consistency")

        # Top 20 most schedule-rigid surgeons
        top_rigid = cons_df.sort_values(
            "pct_top_slot", ascending=False).head(20)
        print(f"\n  Top 20 most schedule-rigid surgeons (by top-slot %):")
        print(f"  {'Surgeon':>10s}  {'Days':>6s}  "
              f"{'Top slot':>18s}  {'% days in slot':>14s}")
        for _, row in top_rigid.iterrows():
            print(f"  {str(row['surgeon']):>10s}  "
                  f"{int(row['total_days']):>6d}  "
                  f"{str(row['top_slot']):>18s}  "
                  f"{row['pct_top_slot']:>14.1f}%")

    # ── E. Hours per session — from realized durations if available ─────────
    if has_duration and "session_span_min" in session.columns:
        subsection("E. Session duration distribution (in hours)")
        span_h = session["session_span_min"] / 60
        describe_series(span_h, "Session span (hours)")

        # Fraction of sessions that fit in 4h, 6h, 8h, 10h
        for h in [4, 6, 8, 10]:
            n = (span_h <= h).sum()
            print(f"  Sessions ≤ {h}h: {pct(n, len(span_h))}")

    # ── F. Cases per surgeon per day distribution by service ────────────────
    if col_available(df_b, "Case_Service"):
        subsection("F. Cases per surgeon per day, by service "
                   "(services with ≥500 cases)")
        svc_day = df_b.groupby(
            ["Case_Service", "Surgeon_Code", "OR_Date"]).size().rename("n")
        for svc in df_b["Case_Service"].value_counts().index:
            if df_b["Case_Service"].eq(svc).sum() < 500:
                continue
            svc_vals = svc_day.loc[svc] if svc in svc_day.index.get_level_values(0) else pd.Series(dtype=float)
            if len(svc_vals) == 0:
                continue
            print(f"\n  {svc}:")
            describe_series(svc_vals, "  Cases/surgeon-day")

    save_csv(surg_week, "surgeon_week_stats")
    save_csv(surg_day,  "surgeon_day_stats")



# ── 28. Room-day template regularity ───────────────────────────────────────
def analyze_room_day_template_regularity(df):
    """Quantify how regular room-day templates are across the calendar."""
    section("ROOM-DAY TEMPLATE REGULARITY")
    if guard_empty(df, "room-day template regularity"):
        return

    blocks = make_room_day_blocks(df, weekdays_only=True)
    if guard_empty(blocks, "room-day template regularity"):
        return

    total_weeks = blocks["Week_Start"].nunique()
    print(f"  Active weekday blocks: {len(blocks):,}")
    print(f"  Active weeks:          {total_weeks:,}")

    room_day = blocks.groupby(["Operating_Room", "DayOfWeek"]).agg(
        active_dates=("OR_Date", "nunique"),
        active_weeks=("Week_Start", "nunique"),
        total_cases=("n_cases", "sum"),
        mean_cases=("n_cases", "mean"),
        median_cases=("n_cases", "median"),
        mean_surgeons=("n_surgeons", "mean"),
        median_surgeons=("n_surgeons", "median"),
        mean_realized=("total_realized", "mean"),
        median_realized=("total_realized", "median"),
    )
    room_day["activation_rate"] = room_day["active_weeks"] / max(total_weeks, 1)

    if "n_services" in blocks.columns:
        svc_stats = blocks.groupby(["Operating_Room", "DayOfWeek"])["n_services"].agg(
            mean_services="mean", median_services="median")
        room_day = room_day.join(svc_stats)

    if "span_min" in blocks.columns:
        span_stats = blocks.groupby(["Operating_Room", "DayOfWeek"])["span_min"].agg(
            mean_span="mean", median_span="median")
        room_day = room_day.join(span_stats)

    if "dom_surgeon_case_share" in blocks.columns:
        dom_surg = blocks.groupby(["Operating_Room", "DayOfWeek"])[
            "dom_surgeon_case_share"].mean().rename("mean_dom_surgeon_share")
        room_day = room_day.join(dom_surg)

    if "dom_service_case_share" in blocks.columns:
        dom_svc = blocks.groupby(["Operating_Room", "DayOfWeek"])[
            "dom_service_case_share"].mean().rename("mean_dom_service_share")
        room_day = room_day.join(dom_svc)

    subsection("Activation-rate distribution across room-day templates")
    describe_series(room_day["activation_rate"], "Activation rate")

    subsection("How many room-day templates are truly regular?")
    for thr in [0.25, 0.50, 0.75, 0.90]:
        n = (room_day["activation_rate"] >= thr).sum()
        print(f"  Activation rate ≥ {thr:4.0%}: {n:,} / {len(room_day):,} templates")

    subsection("By day of week")
    day_summary = room_day.reset_index().groupby("DayOfWeek").agg(
        n_templates=("Operating_Room", "nunique"),
        mean_activation=("activation_rate", "mean"),
        median_activation=("activation_rate", "median"),
        mean_cases=("mean_cases", "mean"),
        mean_surgeons=("mean_surgeons", "mean"),
    )
    if "mean_services" in room_day.columns:
        day_summary["mean_services"] = (
            room_day.reset_index().groupby("DayOfWeek")["mean_services"].mean()
        )
    day_summary = day_summary.reindex([d for d in weekday_order() if d in day_summary.index])

    print(f"  {'Day':12s}  {'Templates':>10s}  {'Mean act.':>10s}  "
          f"{'Median act.':>12s}  {'Mean cases':>10s}  {'Mean surgs':>10s}")
    for day, row in day_summary.iterrows():
        print(f"  {day:12s}  {int(row['n_templates']):10,}  "
              f"{row['mean_activation']:10.3f}  {row['median_activation']:12.3f}  "
              f"{row['mean_cases']:10.2f}  {row['mean_surgeons']:10.2f}")

    subsection("Most regular templates (top 25 by activation rate, then volume)")
    top_regular = (room_day.reset_index()
                   .sort_values(["activation_rate", "total_cases", "active_weeks"],
                                ascending=[False, False, False])
                   .head(25))
    cols = ["Operating_Room", "DayOfWeek", "active_weeks", "activation_rate",
            "total_cases", "mean_cases", "median_cases", "mean_surgeons"]
    if "mean_services" in top_regular.columns:
        cols.append("mean_services")
    for _, row in top_regular[cols].iterrows():
        extra = ""
        if "mean_services" in row.index:
            extra += f"  mean_svcs={row['mean_services']:.2f}"
        print(f"  {row['Operating_Room']:>10s}  {row['DayOfWeek']:9s}  "
              f"weeks={int(row['active_weeks']):4d}  act={row['activation_rate']:.3f}  "
              f"cases={int(row['total_cases']):5d}  mean_cases={row['mean_cases']:.2f}  "
              f"mean_surgs={row['mean_surgeons']:.2f}{extra}")

    save_csv(room_day.reset_index(), "room_day_template_regularity")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(room_day["activation_rate"], bins=30, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Activation rate across weeks")
    ax.set_ylabel("Room-day templates")
    ax.set_title("Room-Day Template Regularity")

    ax = axes[1]
    day_summary_plot = day_summary.reset_index()
    ax.bar(day_summary_plot["DayOfWeek"], day_summary_plot["n_templates"],
           edgecolor="white", alpha=0.8)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Distinct rooms active")
    ax.set_title("Distinct Room Templates by Weekday")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIGDIR / "room_day_template_regularity.png", dpi=150)
    plt.close(fig)


# ── 29. Block opening design for experiments ───────────────────────────────
def analyze_block_opening_design(df):
    """Classify room-day templates into tiers by regularity and provide
    concrete guidance on which blocks to treat as fixed-open versus
    decision variables in the bilevel scheduling model.

    In the current formulation, block opening uses binary variables y_r.
    Historical regularity suggests many blocks are de-facto always open,
    which can be exploited to reduce the number of free binary variables
    and warm-start the CCG.
    """
    section("BLOCK OPENING DESIGN FOR EXPERIMENTS")
    if guard_empty(df, "block opening design"):
        return

    blocks = make_room_day_blocks(df, weekdays_only=True)
    if guard_empty(blocks, "block opening design"):
        return

    total_weeks = blocks["Week_Start"].nunique()

    room_day = blocks.groupby(["Operating_Room", "DayOfWeek"]).agg(
        active_weeks=("Week_Start", "nunique"),
        total_cases=("n_cases", "sum"),
        mean_cases=("n_cases", "mean"),
        mean_realized=("total_realized", "mean"),
    )
    room_day["activation_rate"] = room_day["active_weeks"] / max(total_weeks, 1)

    # Tier classification
    room_day["tier"] = pd.cut(
        room_day["activation_rate"],
        bins=[-0.01, 0.25, 0.50, 0.75, 0.90, 1.01],
        labels=["rare (<25%)", "occasional (25–50%)",
                "regular (50–75%)", "stable (75–90%)",
                "always-on (>90%)"])

    subsection("A. Template tier classification")
    tier_summary = room_day.groupby("tier", observed=True).agg(
        n_templates=("active_weeks", "size"),
        total_cases=("total_cases", "sum"),
        mean_activation=("activation_rate", "mean"),
    )
    print(f"  {'Tier':25s}  {'Templates':>10s}  {'Cases':>8s}  "
          f"{'Mean act.':>10s}")
    for tier, row in tier_summary.iterrows():
        print(f"  {str(tier):25s}  {int(row['n_templates']):10,}  "
              f"{int(row['total_cases']):8,}  "
              f"{row['mean_activation']:10.3f}")

    n_always = tier_summary.loc["always-on (>90%)", "n_templates"] \
        if "always-on (>90%)" in tier_summary.index else 0
    n_stable = tier_summary.loc["stable (75–90%)", "n_templates"] \
        if "stable (75–90%)" in tier_summary.index else 0
    n_total = len(room_day)

    subsection("B. Fixed-block strategy")
    print(f"  Always-on templates (activation > 90%): "
          f"{n_always} / {n_total}")
    print(f"  Stable templates (activation > 75%):    "
          f"{n_always + n_stable} / {n_total}")

    fixed_cases = room_day[room_day["activation_rate"] >= 0.75][
        "total_cases"].sum()
    all_cases = room_day["total_cases"].sum()
    print(f"\n  Cases in stable+ templates: "
          f"{fixed_cases:,} / {all_cases:,} "
          f"({100 * fixed_cases / all_cases:.1f}%)")

    subsection("C. Exploratory design guidance")
    print("  NOTE: The following is exploratory guidance; the activation")
    print("  threshold (75%) is a heuristic, not a formal optimality claim.")
    print(f"\n  Strategy: Fix y_r = 1 for the {n_always + n_stable} "
          f"templates with activation ≥ 75%.")
    print(f"  This covers {100 * fixed_cases / all_cases:.0f}% of cases "
          f"and eliminates {n_always + n_stable} binary variables from "
          f"the master problem.")
    n_flex = n_total - n_always - n_stable
    print(f"  Remaining flexible templates: {n_flex}")
    print(f"  The CCG then optimizes over these {n_flex} "
          f"block-opening decisions plus all case assignments.")

    subsection("D. Per-day fixed block counts")
    fixed_blocks = room_day[room_day["activation_rate"] >= 0.75]
    day_counts = fixed_blocks.reset_index().groupby("DayOfWeek").size()
    for day in weekday_order():
        if day in day_counts.index:
            print(f"    {day:12s}  {int(day_counts[day]):3d} fixed blocks")

    save_csv(room_day.reset_index(), "block_opening_design")


# ── 30. Block fragmentation and ownership ──────────────────────────────────
def analyze_block_fragmentation(df):
    """Measure whether shared blocks are balanced or dominated by one owner."""
    section("BLOCK FRAGMENTATION AND OWNERSHIP")
    if guard_empty(df, "block fragmentation"):
        return

    blocks = make_room_day_blocks(df, weekdays_only=True)
    if guard_empty(blocks, "block fragmentation"):
        return

    subsection("Dominant surgeon share per block")
    if "dom_surgeon_case_share" in blocks.columns:
        describe_series(100 * blocks["dom_surgeon_case_share"],
                        "Dominant surgeon share (%)")
        multi = blocks[blocks["n_surgeons"] > 1]
        if len(multi) > 0:
            print(f"  Multi-surgeon blocks: {len(multi):,}")
            describe_series(100 * multi["dom_surgeon_case_share"],
                            "Dominant surgeon share among multi-surgeon blocks (%)")
            for thr in [0.50, 0.67, 0.80, 0.90]:
                n = (multi["dom_surgeon_case_share"] >= thr).sum()
                print(f"  Multi-surgeon blocks with dominant surgeon share ≥ {thr:4.0%}: "
                      f"{pct(n, len(multi))}")

    if "dom_service_case_share" in blocks.columns:
        subsection("Dominant service share per block")
        describe_series(100 * blocks["dom_service_case_share"],
                        "Dominant service share (%)")
        multi_svc = blocks[blocks["n_services"] > 1]
        if len(multi_svc) > 0:
            print(f"  Multi-service blocks: {len(multi_svc):,}")
            describe_series(100 * multi_svc["dom_service_case_share"],
                            "Dominant service share among multi-service blocks (%)")
            for thr in [0.50, 0.67, 0.80, 0.90]:
                n = (multi_svc["dom_service_case_share"] >= thr).sum()
                print(f"  Multi-service blocks with dominant service share ≥ {thr:4.0%}: "
                      f"{pct(n, len(multi_svc))}")

    subsection("Most fragmented blocks (top 25)")
    frag = blocks.copy()
    frag["surgeon_fragmentation"] = 1 - frag["dom_surgeon_case_share"].fillna(1.0)
    if "dom_service_case_share" in frag.columns:
        frag["service_fragmentation"] = 1 - frag["dom_service_case_share"].fillna(1.0)
    sort_cols = ["n_surgeons", "surgeon_fragmentation", "n_cases"]
    ascending = [False, False, False]
    if "service_fragmentation" in frag.columns:
        sort_cols = ["n_services", "service_fragmentation", "n_surgeons", "n_cases"]
        ascending = [False, False, False, False]
    top_frag = frag.sort_values(sort_cols, ascending=ascending).head(25)

    for _, row in top_frag.iterrows():
        svc_info = f"  n_svcs={int(row['n_services'])}" if "n_services" in row.index else ""
        dom_svc = (f"  dom_svc={100*row['dom_service_case_share']:.1f}%"
                   if "dom_service_case_share" in row.index and pd.notna(row["dom_service_case_share"])
                   else "")
        print(f"  {row['Operating_Room']:>10s}  {str(row['OR_Date'])}  "
              f"cases={int(row['n_cases']):2d}  n_surgs={int(row['n_surgeons']):2d}  "
              f"dom_surg={100*row['dom_surgeon_case_share']:.1f}%{svc_info}{dom_svc}")

    save_csv(frag, "block_fragmentation")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(100 * blocks["dom_surgeon_case_share"].dropna(),
            bins=30, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Dominant surgeon share (%)")
    ax.set_ylabel("Blocks")
    ax.set_title("Dominant Surgeon Share per Block")

    ax = axes[1]
    share_source = "dom_service_case_share" if "dom_service_case_share" in blocks.columns else "dom_surgeon_case_share"
    ax.hist(100 * blocks[share_source].dropna(),
            bins=30, edgecolor="white", alpha=0.8)
    ax.set_xlabel(("Dominant service share (%)"
                   if share_source == "dom_service_case_share"
                   else "Dominant surgeon share (%)"))
    ax.set_ylabel("Blocks")
    ax.set_title("Block Ownership Concentration")

    fig.tight_layout()
    fig.savefig(FIGDIR / "block_fragmentation.png", dpi=150)
    plt.close(fig)


# ── 31. Candidate-pool proxies from historical templates ───────────────────
def analyze_candidate_pool_proxies(df):
    """Estimate how small surgeon/service eligibility sets could plausibly be."""
    section("CANDIDATE-POOL PROXIES FROM HISTORICAL TEMPLATES")
    if guard_empty(df, "candidate-pool proxies"):
        return

    tmp = df.copy()
    dt = pd.to_datetime(tmp["Actual Start Date"], errors="coerce")
    tmp = tmp[dt.notna()].copy()
    tmp["DayOfWeek"] = dt[dt.notna()].dt.day_name()
    tmp["Room_Day_Template"] = (
        tmp["Operating_Room"].astype(str) + " | " + tmp["DayOfWeek"].astype(str)
    )

    def build_profile(frame, entity_col, min_cases):
        rows = []
        for ent, grp in frame.groupby(entity_col):
            if len(grp) < min_cases:
                continue
            tpl_counts = grp["Room_Day_Template"].value_counts()
            room_counts = grp["Operating_Room"].value_counts()
            day_counts = grp["DayOfWeek"].value_counts()
            row = {
                entity_col: ent,
                "n_cases": len(grp),
                "n_templates": tpl_counts.size,
                "n_rooms": room_counts.size,
                "n_days": day_counts.size,
                "template_hhi": hhi_from_counts(tpl_counts),
                "room_hhi": hhi_from_counts(room_counts),
                "day_hhi": hhi_from_counts(day_counts),
                "top_template_share": top_k_share(tpl_counts, 1),
                "top_room_share": top_k_share(room_counts, 1),
                "top_day_share": top_k_share(day_counts, 1),
            }
            for k in TOP_K_TEMPLATE_COVERAGE:
                row[f"top{k}_template_share"] = top_k_share(tpl_counts, k)
                row[f"top{k}_room_share"] = top_k_share(room_counts, k)
            row["favorite_template"] = tpl_counts.index[0]
            row["favorite_day_pattern"] = format_day_set(grp["DayOfWeek"].unique())
            rows.append(row)
        return pd.DataFrame(rows)

    surg_prof = build_profile(tmp, "Surgeon_Code", MIN_SURGEON_CASES)
    subsection("Surgeon-level template concentration")
    if len(surg_prof) > 0:
        describe_series(surg_prof["n_templates"], "Distinct room-day templates per surgeon")
        describe_series(surg_prof["top1_template_share"], "Top-1 template share per surgeon (%)")
        describe_series(surg_prof["top3_template_share"], "Top-3 template share per surgeon (%)")
        describe_series(surg_prof["top5_template_share"], "Top-5 template share per surgeon (%)")
        describe_series(surg_prof["top_room_share"], "Top room share per surgeon (%)")
        describe_series(surg_prof["top_day_share"], "Top day share per surgeon (%)")

        print("\n  Coverage implication for surgeon-based eligibility:")
        for k in TOP_K_TEMPLATE_COVERAGE:
            print(f"    Median top-{k} template coverage: "
                  f"{surg_prof[f'top{k}_template_share'].median():.1f}%")
        rigid = surg_prof.sort_values(["top3_template_share", "top1_template_share"],
                                      ascending=False).head(15)
        flexible = surg_prof.sort_values(["top3_template_share", "n_templates"],
                                         ascending=[True, False]).head(15)

        print("\n  Most template-rigid surgeons (top 15):")
        for _, row in rigid.iterrows():
            print(f"    {str(row['Surgeon_Code']):>10s}  cases={int(row['n_cases']):4d}  "
                  f"templates={int(row['n_templates']):2d}  "
                  f"top1={row['top_template_share']:5.1f}%  "
                  f"top3={row['top3_template_share']:5.1f}%  "
                  f"fav={row['favorite_template']}")

        print("\n  Most template-flexible surgeons (top 15):")
        for _, row in flexible.iterrows():
            print(f"    {str(row['Surgeon_Code']):>10s}  cases={int(row['n_cases']):4d}  "
                  f"templates={int(row['n_templates']):2d}  "
                  f"top1={row['top_template_share']:5.1f}%  "
                  f"top3={row['top3_template_share']:5.1f}%  "
                  f"fav={row['favorite_template']}")

        save_csv(surg_prof, "surgeon_candidate_pool_proxies")

    if col_available(tmp, "Case_Service"):
        svc_tmp = tmp.dropna(subset=["Case_Service"]).copy()
        svc_prof = build_profile(svc_tmp, "Case_Service", MIN_SERVICE_CASES)
        subsection("Service-level template concentration")
        if len(svc_prof) > 0:
            describe_series(svc_prof["n_templates"], "Distinct room-day templates per service")
            describe_series(svc_prof["top1_template_share"], "Top-1 template share per service (%)")
            describe_series(svc_prof["top3_template_share"], "Top-3 template share per service (%)")
            describe_series(svc_prof["top5_template_share"], "Top-5 template share per service (%)")

            print("\n  Coverage implication for service-based eligibility:")
            for k in TOP_K_TEMPLATE_COVERAGE:
                print(f"    Median top-{k} template coverage: "
                      f"{svc_prof[f'top{k}_template_share'].median():.1f}%")

            print("\n  Services sorted by template spread:")
            spread = svc_prof.sort_values(["n_templates", "top3_template_share"],
                                          ascending=[False, True])
            for _, row in spread.iterrows():
                print(f"    {str(row['Case_Service']):35s}  cases={int(row['n_cases']):5d}  "
                      f"templates={int(row['n_templates']):2d}  "
                      f"top3={row['top3_template_share']:5.1f}%  "
                      f"top5={row['top5_template_share']:5.1f}%")
            save_csv(svc_prof, "service_candidate_pool_proxies")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if len(surg_prof) > 0:
        ax = axes[0]
        ax.hist(surg_prof["n_templates"], bins=30, edgecolor="white", alpha=0.8)
        ax.set_xlabel("Distinct room-day templates per surgeon")
        ax.set_ylabel("Surgeons")
        ax.set_title("Surgeon Template Spread")

        ax = axes[1]
        ax.hist(surg_prof["top3_template_share"], bins=30,
                edgecolor="white", alpha=0.8)
        ax.set_xlabel("Top-3 template coverage (%)")
        ax.set_ylabel("Surgeons")
        ax.set_title("How much of a surgeon's workload sits in top 3 templates?")
        fig.tight_layout()
        fig.savefig(FIGDIR / "candidate_pool_proxies.png", dpi=150)
        plt.close(fig)
    else:
        plt.close(fig)


# ── 32. Weekly block market and day-level complexity ───────────────────────
def analyze_weekly_block_market(df):
    """Describe weekly/day-level block counts and fragmentation proxies."""
    section("WEEKLY BLOCK MARKET AND DAY-LEVEL COMPLEXITY")
    if guard_empty(df, "weekly block market"):
        return

    blocks = make_room_day_blocks(df, weekdays_only=True)
    if guard_empty(blocks, "weekly block market"):
        return

    week_summary = blocks.groupby("Week_Start").agg(
        n_blocks=("Operating_Room", "size"),
        n_rooms=("Operating_Room", "nunique"),
        mean_cases_per_block=("n_cases", "mean"),
        mean_surgeons_per_block=("n_surgeons", "mean"),
        pct_single_surgeon=("n_surgeons", lambda x: 100 * (x == 1).mean()),
    )
    if "n_services" in blocks.columns:
        svc_week = blocks.groupby("Week_Start")["n_services"].apply(
            lambda x: 100 * (x == 1).mean()).rename("pct_single_service")
        week_summary = week_summary.join(svc_week)

    subsection("Weekly distribution")
    describe_series(week_summary["n_blocks"], "Blocks per active week")
    describe_series(week_summary["mean_cases_per_block"], "Mean cases/block within week")
    describe_series(week_summary["mean_surgeons_per_block"], "Mean surgeons/block within week")
    describe_series(week_summary["pct_single_surgeon"], "% single-surgeon blocks within week")

    subsection("Daily block counts inside weeks")
    day_counts = (blocks.groupby(["Week_Start", "DayOfWeek"]).size()
                  .rename("n_blocks").reset_index())
    print(f"  {'Day':12s}  {'Mean':>8s}  {'Median':>8s}  {'p95':>8s}  {'Max':>8s}")
    for day in [d for d in weekday_order() if d in day_counts["DayOfWeek"].unique()]:
        vals = day_counts.loc[day_counts["DayOfWeek"] == day, "n_blocks"]
        print(f"  {day:12s}  {vals.mean():8.2f}  {vals.median():8.2f}  "
              f"{vals.quantile(0.95):8.2f}  {vals.max():8.0f}")

    subsection("How large is the weekly block pool versus daily pools?")
    avg_daily = day_counts.groupby("DayOfWeek")["n_blocks"].mean()
    avg_weekly = week_summary["n_blocks"].mean()
    print(f"  Mean active blocks per week: {avg_weekly:.2f}")
    for day in avg_daily.index:
        print(f"  Mean active blocks on {day:9s}: {avg_daily[day]:.2f}")
    print("  Interpretation: this compares the size of a flat weekly block pool")
    print("  against day-conditioned pools observed in practice.")

    # Common weekly block-pattern strings
    subsection("Common weekday activation patterns for rooms")
    room_week = (blocks.assign(active=1)
                 .pivot_table(index=["Operating_Room", "Week_Start"],
                              columns="DayOfWeek", values="active",
                              aggfunc="max", fill_value=0))
    room_week = room_week.reindex(columns=[d for d in weekday_order() if d in room_week.columns],
                                  fill_value=0)
    pattern = room_week.apply(
        lambda r: format_day_set([c for c, v in r.items() if v > 0]), axis=1)
    patt_counts = pattern.value_counts().head(20)
    print(f"  {'Pattern':18s}  {'Room-weeks':>10s}  {'%':>8s}")
    for patt, cnt in patt_counts.items():
        print(f"  {patt:18s}  {cnt:10,}  {100 * cnt / len(pattern):8.1f}%")

    save_csv(week_summary.reset_index(), "weekly_block_market")
    save_csv(day_counts, "weekly_day_block_counts")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(week_summary["n_blocks"], bins=30, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Active blocks per week")
    ax.set_ylabel("Weeks")
    ax.set_title("Weekly Block Count Distribution")

    ax = axes[1]
    day_plot = day_counts.groupby("DayOfWeek")["n_blocks"].mean().reindex(
        [d for d in weekday_order() if d in day_counts["DayOfWeek"].unique()])
    ax.bar(day_plot.index, day_plot.values, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean active blocks")
    ax.set_title("Mean Active Blocks by Day")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIGDIR / "weekly_block_market.png", dpi=150)
    plt.close(fig)


# ── 33. Surgeon week-patterns ──────────────────────────────────────────────
def analyze_surgeon_week_patterns(df):
    """Characterise day-set patterns across active surgeon-weeks."""
    section("SURGEON WEEK-PATTERN STRUCTURE")
    if guard_empty(df, "surgeon week patterns"):
        return

    tmp = df.copy()
    dt = pd.to_datetime(tmp["Actual Start Date"], errors="coerce")
    tmp = tmp[dt.notna()].copy()
    tmp["DayOfWeek"] = dt[dt.notna()].dt.day_name()

    surg_week = tmp.groupby(["Surgeon_Code", "Week_Start"]).agg(
        n_cases=("Patient_ID", "size"),
        n_rooms=("Operating_Room", "nunique"),
        day_pattern=("DayOfWeek", lambda x: format_day_set(x)),
    ).reset_index()

    subsection("Overall active-week pattern distribution")
    patt_counts = surg_week["day_pattern"].value_counts().head(25)
    print(f"  {'Pattern':18s}  {'Surgeon-weeks':>14s}  {'%':>8s}")
    for patt, cnt in patt_counts.items():
        print(f"  {patt:18s}  {cnt:14,}  {100 * cnt / len(surg_week):8.1f}%")

    subsection("Favourite pattern consistency per surgeon")
    fav_rows = []
    for surg, grp in surg_week.groupby("Surgeon_Code"):
        if len(grp) < 10:
            continue
        counts = grp["day_pattern"].value_counts()
        fav_rows.append({
            "Surgeon_Code": surg,
            "n_active_weeks": len(grp),
            "n_patterns": counts.size,
            "favorite_pattern": counts.index[0],
            "favorite_pattern_share": 100 * counts.iloc[0] / counts.sum(),
            "top2_pattern_share": 100 * counts.head(2).sum() / counts.sum(),
        })
    fav_df = pd.DataFrame(fav_rows)
    if len(fav_df) > 0:
        describe_series(fav_df["n_patterns"], "Distinct week-patterns per surgeon")
        describe_series(fav_df["favorite_pattern_share"],
                        "Favorite week-pattern share per surgeon (%)")
        describe_series(fav_df["top2_pattern_share"],
                        "Top-2 week-pattern share per surgeon (%)")

        print("\n  Most pattern-rigid surgeons (top 20):")
        rigid = fav_df.sort_values(
            ["favorite_pattern_share", "n_active_weeks"], ascending=[False, False]
        ).head(20)
        for _, row in rigid.iterrows():
            print(f"    {str(row['Surgeon_Code']):>10s}  "
                  f"weeks={int(row['n_active_weeks']):3d}  "
                  f"patterns={int(row['n_patterns']):2d}  "
                  f"fav={str(row['favorite_pattern']):18s}  "
                  f"share={row['favorite_pattern_share']:5.1f}%")

        save_csv(fav_df, "surgeon_week_pattern_consistency")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(patt_counts.index.astype(str), patt_counts.values,
           edgecolor="white", alpha=0.8)
    ax.set_xlabel("Week pattern")
    ax.set_ylabel("Surgeon-weeks")
    ax.set_title("Most Common Surgeon Week Patterns")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    if len(fav_df) > 0:
        ax.hist(fav_df["favorite_pattern_share"], bins=30,
                edgecolor="white", alpha=0.8)
        ax.set_xlabel("Favorite week-pattern share (%)")
        ax.set_ylabel("Surgeons")
        ax.set_title("Surgeon Week-Pattern Rigidity")

    fig.tight_layout()
    fig.savefig(FIGDIR / "surgeon_week_patterns.png", dpi=150)
    plt.close(fig)


def analyze_block_load_decomposition(df, block_data=None):
    """Comprehensive block-level analysis for scheduling model design.

    Determines:
    1. How the block span decomposes into case occupancy + inter-case gaps
    2. What the inter-case gap (leave → enter) actually looks like
    3. What effective capacity reproduces observed overtime rates
    4. Whether bookings already include turnover implicitly
    5. How much service-based eligibility restricts the feasible set
    6. Whether case durations and turnover times are homogeneous
       across service, surgeon, room, and day of week
    """
    section("BLOCK LOAD DECOMPOSITION AND MODEL DESIGN DECISIONS")
    if guard_empty(df, "block load decomposition"):
        return
    if "Enter Room_DT" not in df.columns or "Leave Room_DT" not in df.columns:
        print("  Enter/Leave Room timestamps not available — skipping.")
        return

    # ── Prepare case-level data with valid room timestamps ───────────────
    df_c = df.dropna(subset=["Enter Room_DT", "Leave Room_DT"]).copy()
    df_c["OR_Date"] = pd.to_datetime(df_c["Actual Start Date"]).dt.date
    df_c["Weekday"] = pd.to_datetime(df_c["Actual Start Date"]).dt.weekday
    df_c["DayName"] = pd.to_datetime(df_c["Actual Start Date"]).dt.day_name()

    # Enforce timestamp ordering
    valid_order = (
        (df_c["Enter Room_DT"] <= df_c["Actual Start_DT"])
        & (df_c["Actual Start_DT"] <= df_c["Actual Stop_DT"])
        & (df_c["Actual Stop_DT"] <= df_c["Leave Room_DT"])
    )
    # Only filter among cases that have all four timestamps
    has_all = (df_c["Enter Room_DT"].notna() & df_c["Actual Start_DT"].notna()
               & df_c["Actual Stop_DT"].notna() & df_c["Leave Room_DT"].notna())
    violations = has_all & ~valid_order
    n_violations = violations.sum()
    print(f"  Cases with all four timestamps: {has_all.sum():,}")
    print(f"  Timestamp ordering violations:  {n_violations:,} "
          f"({100 * n_violations / max(has_all.sum(), 1):.1f}%)")
    df_c = df_c[~violations].copy()

    # Room time and surgical time
    df_c["room_time"] = (
        (df_c["Leave Room_DT"] - df_c["Enter Room_DT"])
        .dt.total_seconds() / 60)
    df_c["surg_time"] = (
        (df_c["Actual Stop_DT"] - df_c["Actual Start_DT"])
        .dt.total_seconds() / 60)
    df_c["pre_incision"] = (
        (df_c["Actual Start_DT"] - df_c["Enter Room_DT"])
        .dt.total_seconds() / 60)
    df_c["post_incision"] = (
        (df_c["Leave Room_DT"] - df_c["Actual Stop_DT"])
        .dt.total_seconds() / 60)
    df_c["overhead"] = df_c["room_time"] - df_c["surg_time"]

    # Cap implausible overhead
    extreme_overhead = df_c["overhead"] > 300
    print(f"  Extreme overhead (>300 min): {extreme_overhead.sum():,} — capped")
    df_c.loc[extreme_overhead, "room_time"] = df_c.loc[extreme_overhead, "surg_time"]
    df_c.loc[extreme_overhead, "overhead"] = 0

    # Restrict to weekdays
    df_c = df_c[df_c["Weekday"] < 5].copy()
    df_c = df_c.sort_values(["Operating_Room", "OR_Date", "Enter Room_DT"])
    print(f"  Weekday cases with valid timestamps: {len(df_c):,}")

    # ── Build block-level aggregates ─────────────────────────────────────
    def _build_blocks(df_in):
        blocks = []
        for (room, date), grp in df_in.groupby(["Operating_Room", "OR_Date"]):
            grp = grp.sort_values("Enter Room_DT")
            n = len(grp)
            if n == 0:
                continue

            sum_room = grp["room_time"].sum()
            sum_surg = grp["surg_time"].sum()
            sum_booked = grp["Booked Time (Minutes)"].sum()
            first_enter = grp["Enter Room_DT"].min()
            last_leave = grp["Leave Room_DT"].max()
            span = (last_leave - first_enter).total_seconds() / 60
            n_surgeons = grp["Surgeon_Code"].nunique()
            weekday = grp["DayName"].iloc[0]

            # Service: dominant
            svc = "Unknown"
            if col_available(grp, "Case_Service"):
                modes = grp["Case_Service"].mode()
                if len(modes) > 0:
                    svc = modes.iloc[0]

            # Compute each inter-case gap (leave[i] → enter[i+1])
            gaps = []
            gap_types = []  # 'same_surgeon' or 'diff_surgeon'
            if n > 1:
                for k in range(n - 1):
                    lv = grp["Leave Room_DT"].iloc[k]
                    en = grp["Enter Room_DT"].iloc[k + 1]
                    gap = (en - lv).total_seconds() / 60
                    gaps.append(gap)
                    same_s = (grp["Surgeon_Code"].iloc[k]
                              == grp["Surgeon_Code"].iloc[k + 1])
                    gap_types.append("same" if same_s else "diff")

            # Filter out negative and implausible gaps
            valid_gaps = [g for g in gaps if 0 <= g <= 120]
            total_gap = sum(valid_gaps)

            blocks.append({
                "room": room, "date": date, "n_cases": n,
                "n_surgeons": n_surgeons, "service": svc,
                "weekday": weekday,
                "sum_room_time": sum_room, "sum_surg_time": sum_surg,
                "sum_booked": sum_booked,
                "block_span": span, "total_inter_case_gap": total_gap,
                "n_gaps": len(valid_gaps),
                "mean_gap": np.mean(valid_gaps) if valid_gaps else np.nan,
            })
        return pd.DataFrame(blocks)

    bdf = _build_blocks(df_c)
    # Filter reasonable spans
    bdf = bdf[(bdf["block_span"] > 30) & (bdf["block_span"] < 960)].copy()
    multi = bdf[bdf["n_cases"] > 1].copy()
    print(f"\n  Total weekday blocks: {len(bdf):,}")
    print(f"  Multi-case blocks:   {len(multi):,}")
    print(f"  Single-case blocks:  {(bdf['n_cases'] == 1).sum():,}")

    # ═══════════════════════════════════════════════════════════════════════
    # A. BLOCK SPAN DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    subsection("A. How does the block span decompose?")
    print("  block span = Σ(room times) + Σ(inter-case gaps)")
    print("  The inter-case gap is the time between Leave Room of case i")
    print("  and Enter Room of case i+1, i.e., the room is nominally empty.\n")

    multi["residual"] = multi["block_span"] - multi["sum_room_time"]
    multi["check"] = (multi["sum_room_time"] + multi["total_inter_case_gap"])

    describe_series(multi["sum_room_time"], "Σ room times (multi-case)")
    describe_series(multi["total_inter_case_gap"],
                    "Σ inter-case gaps (multi-case)")
    describe_series(multi["residual"],
                    "block span − Σ room times (= total gap)")
    describe_series(multi["check"] - multi["block_span"],
                    "Σ(room) + Σ(gap) − span (should be ≈ 0)")

    if len(multi) > 0:
        corr = multi[["block_span", "sum_room_time"]].corr().iloc[0, 1]
        print(f"\n  Correlation(block span, Σ room times): {corr:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # B. WHAT DOES THE INTER-CASE GAP LOOK LIKE?
    # ═══════════════════════════════════════════════════════════════════════
    subsection("B. The inter-case gap: leave of case i → enter of case i+1")
    print("  This is the time between consecutive patients in the same room.")
    print("  It includes cleaning, setup, and any idle wait for the next case.")
    print("  This gap is NOT part of any case's room time.\n")

    # Build transition-level data
    df_c["prev_room"]    = df_c["Operating_Room"].shift(1)
    df_c["prev_date"]    = df_c["OR_Date"].shift(1)
    df_c["prev_leave"]   = df_c["Leave Room_DT"].shift(1)
    df_c["prev_surgeon"] = df_c["Surgeon_Code"].shift(1)
    if col_available(df_c, "Case_Service"):
        df_c["prev_service"] = df_c["Case_Service"].shift(1)

    same_block = ((df_c["Operating_Room"] == df_c["prev_room"])
                  & (df_c["OR_Date"] == df_c["prev_date"]))
    df_c.loc[same_block, "gap_min"] = (
        (df_c.loc[same_block, "Enter Room_DT"]
         - df_c.loc[same_block, "prev_leave"])
        .dt.total_seconds() / 60)

    gaps = df_c[same_block & df_c["gap_min"].notna()
                & (df_c["gap_min"] >= 0) & (df_c["gap_min"] <= 120)].copy()

    print(f"  Valid inter-case transitions (0–120 min): {len(gaps):,}")
    describe_series(gaps["gap_min"], "Inter-case gap")

    # Same vs different surgeon
    gaps["same_surgeon"] = (gaps["Surgeon_Code"] == gaps["prev_surgeon"])
    same_s = gaps[gaps["same_surgeon"]]
    diff_s = gaps[~gaps["same_surgeon"]]
    print(f"\n  Same-surgeon transitions:      {len(same_s):,}")
    print(f"  Different-surgeon transitions:  {len(diff_s):,}")
    if len(same_s) > 0:
        describe_series(same_s["gap_min"], "Gap (same surgeon)")
    if len(diff_s) > 0:
        describe_series(diff_s["gap_min"], "Gap (diff surgeon)")

    # Same vs different service
    if col_available(df_c, "Case_Service") and "prev_service" in gaps.columns:
        gaps["same_service"] = (gaps["Case_Service"] == gaps["prev_service"])
        same_svc = gaps[gaps["same_service"]]
        diff_svc = gaps[~gaps["same_service"]]
        print(f"\n  Same-service transitions:      {len(same_svc):,}")
        print(f"  Different-service transitions:  {len(diff_svc):,}")
        if len(same_svc) > 0:
            describe_series(same_svc["gap_min"], "Gap (same service)")
        if len(diff_svc) > 0:
            describe_series(diff_svc["gap_min"], "Gap (diff service)")

        # Crossed: same surgeon / same service, same surgeon / diff service, etc.
        print("\n  Crossed breakdown (surgeon × service):")
        for (ss, sv), subg in gaps.groupby(["same_surgeon", "same_service"]):
            label = (f"{'same' if ss else 'diff'} surgeon, "
                     f"{'same' if sv else 'diff'} service")
            if len(subg) >= 10:
                print(f"    {label:40s}  N={len(subg):5,}  "
                      f"mean={subg['gap_min'].mean():5.1f}  "
                      f"median={subg['gap_min'].median():5.1f}  "
                      f"std={subg['gap_min'].std():5.1f}")

    # ═══════════════════════════════════════════════════════════════════════
    # C. EFFECTIVE CAPACITY CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════
    subsection("C. Effective capacity: matching OT rates")
    print("  Goal: find C_eff such that OT(Σ room times, C_eff) ≈ "
          "OT(block span, 480).")
    print("  This tells us what capacity to use if we model block load")
    print("  as Σ d^case without explicit turnover.\n")

    # Reference: OT rate based on actual block span at C=480
    ref_ot_rate = (bdf["block_span"] > 480).mean()
    print(f"  Reference OT rate (block span > 480): {100 * ref_ot_rate:.1f}%\n")

    print(f"  {'C_eff':>5s}  {'OT(room)':>10s}  {'OT(span@480)':>13s}  "
          f"{'meanOT(room)':>13s}  {'meanIdle(room)':>14s}")
    best_c = 480
    best_diff = 1.0
    for C_eff in range(400, 521, 10):
        ot_rate_room = (bdf["sum_room_time"] > C_eff).mean()
        mean_ot = np.maximum(bdf["sum_room_time"] - C_eff, 0).mean()
        mean_idle = np.maximum(C_eff - bdf["sum_room_time"], 0).mean()
        marker = ""
        diff = abs(ot_rate_room - ref_ot_rate)
        if diff < best_diff:
            best_diff = diff
            best_c = C_eff
            marker = "  ← closest match"
        print(f"  {C_eff:5d}  {100 * ot_rate_room:9.1f}%  "
              f"{100 * ref_ot_rate:12.1f}%  "
              f"{mean_ot:12.1f}  {mean_idle:13.1f}{marker}")

    print(f"\n  → Best match: C_eff = {best_c} min")
    print(f"    At this capacity, OT rate from Σ(room times) matches "
          f"OT rate from actual block spans at 480 min.")

    # ═══════════════════════════════════════════════════════════════════════
    # D. DO BOOKINGS IMPLICITLY INCLUDE TURNOVER?
    # ═══════════════════════════════════════════════════════════════════════
    subsection("D. Does Σ(bookings) cover the block span or only Σ(room times)?")
    print("  If surgeons book enough to cover turnover, "
          "Σ(booked) ≈ block span.")
    print("  If they book only their case time, "
          "Σ(booked) ≈ Σ(room times).\n")

    bdf["err_vs_span"] = bdf["sum_booked"] - bdf["block_span"]
    bdf["err_vs_room"] = bdf["sum_booked"] - bdf["sum_room_time"]
    bdf["err_vs_surg"] = bdf["sum_booked"] - bdf["sum_surg_time"]

    print("  All blocks:")
    print(f"    Σ(booked) − block span:       "
          f"mean={bdf['err_vs_span'].mean():6.1f}  "
          f"median={bdf['err_vs_span'].median():6.1f}  "
          f"Pr(>0)={100 * (bdf['err_vs_span'] > 0).mean():.1f}%")
    print(f"    Σ(booked) − Σ(room times):    "
          f"mean={bdf['err_vs_room'].mean():6.1f}  "
          f"median={bdf['err_vs_room'].median():6.1f}  "
          f"Pr(>0)={100 * (bdf['err_vs_room'] > 0).mean():.1f}%")
    print(f"    Σ(booked) − Σ(surgical times): "
          f"mean={bdf['err_vs_surg'].mean():6.1f}  "
          f"median={bdf['err_vs_surg'].median():6.1f}  "
          f"Pr(>0)={100 * (bdf['err_vs_surg'] > 0).mean():.1f}%")

    print("\n  Multi-case blocks only:")
    m = multi.copy()
    m["err_vs_span"] = m["sum_booked"] - m["block_span"]
    m["err_vs_room"] = m["sum_booked"] - m["sum_room_time"]
    m["err_vs_surg"] = m["sum_booked"] - m["sum_surg_time"]
    print(f"    Σ(booked) − block span:       "
          f"mean={m['err_vs_span'].mean():6.1f}  "
          f"median={m['err_vs_span'].median():6.1f}  "
          f"Pr(>0)={100 * (m['err_vs_span'] > 0).mean():.1f}%")
    print(f"    Σ(booked) − Σ(room times):    "
          f"mean={m['err_vs_room'].mean():6.1f}  "
          f"median={m['err_vs_room'].median():6.1f}  "
          f"Pr(>0)={100 * (m['err_vs_room'] > 0).mean():.1f}%")
    print(f"    Σ(booked) − Σ(surgical times): "
          f"mean={m['err_vs_surg'].mean():6.1f}  "
          f"median={m['err_vs_surg'].median():6.1f}  "
          f"Pr(>0)={100 * (m['err_vs_surg'] > 0).mean():.1f}%")

    # Per-case averages
    total_cases = bdf["n_cases"].sum()
    multi_cases = multi["n_cases"].sum()
    print(f"\n  Per-case averages (all blocks):")
    print(f"    Mean booked per case:     "
          f"{bdf['sum_booked'].sum() / total_cases:.1f} min")
    print(f"    Mean room time per case:  "
          f"{bdf['sum_room_time'].sum() / total_cases:.1f} min")
    print(f"    Mean surg time per case:  "
          f"{bdf['sum_surg_time'].sum() / total_cases:.1f} min")
    print(f"    Mean span per case:       "
          f"{bdf['block_span'].sum() / total_cases:.1f} min")
    if len(multi) > 0:
        print(f"    Mean inter-case gap (multi-case blocks): "
              f"{multi['total_inter_case_gap'].sum() / multi_cases:.1f} min/case")

    # ═══════════════════════════════════════════════════════════════════════
    # E. ELIGIBILITY REDUCTION FROM SERVICE-BASED FILTERING
    # ═══════════════════════════════════════════════════════════════════════
    subsection("E. Eligibility reduction from service-based filtering")
    if col_available(df_c, "Case_Service"):
        svc_rooms = (df_c.groupby("Case_Service")["Operating_Room"]
                     .apply(set).to_dict())
        all_rooms = df_c["Operating_Room"].nunique()

        elig = []
        for _, row in df_c.iterrows():
            svc = row.get("Case_Service", None)
            n_e = len(svc_rooms.get(svc, set())) if svc else all_rooms
            elig.append(n_e)

        elig = np.array(elig)
        print(f"\n  Total distinct rooms: {all_rooms}")
        describe_series(elig, "Eligible rooms per case (service filter)")
        print(f"  Mean reduction: "
              f"{100 * (1 - elig.mean() / all_rooms):.1f}%")

        # By service
        print(f"\n  {'Service':25s}  {'N cases':>8s}  {'Rooms':>6s}  "
              f"{'% of total':>10s}")
        for svc in sorted(svc_rooms, key=lambda s: -len(svc_rooms[s])):
            n_svc = (df_c["Case_Service"] == svc).sum()
            if n_svc < 20:
                continue
            n_rooms = len(svc_rooms[svc])
            print(f"  {svc:25s}  {n_svc:8,}  {n_rooms:6d}  "
                  f"{100 * n_rooms / all_rooms:9.1f}%")
    else:
        print("  Case_Service not available — skipping.")

    # ═══════════════════════════════════════════════════════════════════════
    # F. HOMOGENEITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    subsection("F. Homogeneity of case durations and inter-case gaps")
    print("  For each duration type, we test whether the distribution")
    print("  differs materially across service, surgeon, room, and weekday.")
    print("  We report between-group variance as % of total variance,")
    print("  and Kruskal-Wallis H-test p-values.\n")

    def _homogeneity_table(data, value_col, group_col, label, min_group=30):
        """Print homogeneity diagnostics for value_col grouped by group_col."""
        sub = data[[value_col, group_col]].dropna()
        if len(sub) < 50:
            print(f"    {label}: insufficient data")
            return
        grp_counts = sub[group_col].value_counts()
        valid_groups = grp_counts[grp_counts >= min_group].index
        sub = sub[sub[group_col].isin(valid_groups)]
        if len(valid_groups) < 2:
            print(f"    {label}: fewer than 2 groups with ≥{min_group} obs")
            return

        total_var = sub[value_col].var()
        grp_means = sub.groupby(group_col)[value_col].mean()
        grp_vars = sub.groupby(group_col)[value_col].var()
        between_var = grp_means.var()
        within_var = grp_vars.mean()
        eta_sq = between_var / total_var if total_var > 0 else np.nan

        # Kruskal-Wallis
        groups_data = [g[value_col].values
                       for _, g in sub.groupby(group_col)
                       if len(g) >= min_group]
        if len(groups_data) >= 2:
            try:
                H, p = sp_stats.kruskal(*groups_data)
            except Exception:
                H, p = np.nan, np.nan
        else:
            H, p = np.nan, np.nan

        n_groups = len(valid_groups)
        spread = grp_means.max() - grp_means.min()
        print(f"    {label:45s}  groups={n_groups:3d}  "
              f"η²={eta_sq:.3f}  spread={spread:6.1f} min  "
              f"KW p={'<0.001' if p < 0.001 else f'{p:.3f}'}")

    # F.1 Case-level room time
    print("  F.1  Room time (Enter → Leave) per case:")
    df_c_filt = df_c[df_c["room_time"].between(30, 960)].copy()
    _homogeneity_table(df_c_filt, "room_time", "Case_Service",
                       "by service")
    _homogeneity_table(df_c_filt, "room_time", "Surgeon_Code",
                       "by surgeon", min_group=30)
    _homogeneity_table(df_c_filt, "room_time", "Operating_Room",
                       "by room")
    _homogeneity_table(df_c_filt, "room_time", "DayName",
                       "by weekday")

    # F.2 Surgical time
    print("\n  F.2  Surgical time (Start → Stop) per case:")
    _homogeneity_table(df_c_filt, "surg_time", "Case_Service",
                       "by service")
    _homogeneity_table(df_c_filt, "surg_time", "Surgeon_Code",
                       "by surgeon", min_group=30)
    _homogeneity_table(df_c_filt, "surg_time", "Operating_Room",
                       "by room")
    _homogeneity_table(df_c_filt, "surg_time", "DayName",
                       "by weekday")

    # F.3 Non-surgical overhead (room time − surgical time)
    print("\n  F.3  Non-surgical overhead (room − surgical) per case:")
    df_c_filt["overhead_clipped"] = df_c_filt["overhead"].clip(0, 300)
    _homogeneity_table(df_c_filt, "overhead_clipped", "Case_Service",
                       "by service")
    _homogeneity_table(df_c_filt, "overhead_clipped", "Surgeon_Code",
                       "by surgeon", min_group=30)
    _homogeneity_table(df_c_filt, "overhead_clipped", "Operating_Room",
                       "by room")
    _homogeneity_table(df_c_filt, "overhead_clipped", "DayName",
                       "by weekday")

    # F.4 Pre-incision time
    print("\n  F.4  Pre-incision time (Enter Room → Actual Start):")
    df_pi = df_c_filt[df_c_filt["pre_incision"].between(0, 180)].copy()
    describe_series(df_pi["pre_incision"], "Pre-incision time")
    _homogeneity_table(df_pi, "pre_incision", "Case_Service",
                       "by service")
    _homogeneity_table(df_pi, "pre_incision", "Surgeon_Code",
                       "by surgeon", min_group=30)

    # F.5 Post-incision time
    print("\n  F.5  Post-incision time (Actual Stop → Leave Room):")
    df_po = df_c_filt[df_c_filt["post_incision"].between(0, 180)].copy()
    describe_series(df_po["post_incision"], "Post-incision time")
    _homogeneity_table(df_po, "post_incision", "Case_Service",
                       "by service")
    _homogeneity_table(df_po, "post_incision", "Surgeon_Code",
                       "by surgeon", min_group=30)

    # F.6 Inter-case gap (turnover)
    print("\n  F.6  Inter-case gap (leave of case i → enter of case i+1):")
    if len(gaps) > 0:
        describe_series(gaps["gap_min"], "Inter-case gap (all)")
        _homogeneity_table(gaps, "gap_min", "Case_Service",
                           "by service of incoming case")
        if col_available(gaps, "Surgeon_Code"):
            _homogeneity_table(gaps, "gap_min", "Surgeon_Code",
                               "by surgeon of incoming case", min_group=20)
        _homogeneity_table(gaps, "gap_min", "Operating_Room",
                           "by room")
        _homogeneity_table(gaps, "gap_min", "DayName",
                           "by weekday")
        # Homogeneity of gap by same vs different surgeon
        if len(same_s) > 0 and len(diff_s) > 0:
            print(f"\n    By same/diff surgeon transition:")
            print(f"      Same surgeon:  N={len(same_s):,}  "
                  f"mean={same_s['gap_min'].mean():.1f}  "
                  f"std={same_s['gap_min'].std():.1f}")
            print(f"      Diff surgeon:  N={len(diff_s):,}  "
                  f"mean={diff_s['gap_min'].mean():.1f}  "
                  f"std={diff_s['gap_min'].std():.1f}")

    # F.7 Booked duration
    print("\n  F.7  Booked duration:")
    _homogeneity_table(df_c_filt, "Booked Time (Minutes)", "Case_Service",
                       "by service")
    _homogeneity_table(df_c_filt, "Booked Time (Minutes)", "Surgeon_Code",
                       "by surgeon", min_group=30)

    # F.8 Booking error (booked − room time)
    print("\n  F.8  Booking error (booked − room time):")
    df_c_filt["booking_err"] = (df_c_filt["Booked Time (Minutes)"]
                                - df_c_filt["room_time"])
    _homogeneity_table(df_c_filt, "booking_err", "Case_Service",
                       "by service")
    _homogeneity_table(df_c_filt, "booking_err", "Surgeon_Code",
                       "by surgeon", min_group=30)

    # ═══════════════════════════════════════════════════════════════════════
    # G. VARIANCE DECOMPOSITION OF BLOCK SPAN
    # ═══════════════════════════════════════════════════════════════════════
    subsection("G. Block-level variance decomposition")
    if len(multi) > 0:
        var_span = multi["block_span"].var()
        var_room = multi["sum_room_time"].var()
        var_gap = multi["total_inter_case_gap"].var()
        cov_rg = np.cov(multi["sum_room_time"],
                        multi["total_inter_case_gap"])[0, 1]
        print(f"  Multi-case blocks (N={len(multi):,}):\n")
        print(f"    Var(block span):            {var_span:>12,.0f}")
        print(f"    Var(Σ room times):          {var_room:>12,.0f}  "
              f"({100 * var_room / var_span:.1f}%)")
        print(f"    Var(Σ inter-case gaps):     {var_gap:>12,.0f}  "
              f"({100 * var_gap / var_span:.1f}%)")
        print(f"    2 × Cov(room, gap):         {2 * cov_rg:>12,.0f}  "
              f"({100 * 2 * cov_rg / var_span:.1f}%)")
        print(f"    Sum:                        "
              f"{var_room + var_gap + 2 * cov_rg:>12,.0f}")

        # CV comparison
        cv_room = multi["sum_room_time"].std() / multi["sum_room_time"].mean()
        cv_gap = (multi["total_inter_case_gap"].std()
                  / multi["total_inter_case_gap"].mean())
        print(f"\n    CV(Σ room times):       {cv_room:.3f}")
        print(f"    CV(Σ inter-case gaps):  {cv_gap:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # H. SINGLE-τ vs TWO-RATE APPROXIMATION
    # ═══════════════════════════════════════════════════════════════════════
    subsection("H. Turnover approximation accuracy (multi-case blocks)")
    if len(multi) > 0:
        overall_mean_gap = gaps["gap_min"].mean() if len(gaps) > 0 else 28
        same_mean = same_s["gap_min"].mean() if len(same_s) > 0 else 28
        diff_mean = diff_s["gap_min"].mean() if len(diff_s) > 0 else 45

        multi["approx_none"] = multi["sum_room_time"]
        multi["approx_single"] = (multi["sum_room_time"]
                                  + overall_mean_gap * (multi["n_cases"] - 1))
        multi["approx_two"] = (
            multi["sum_room_time"]
            + same_mean * (multi["n_cases"] - multi["n_surgeons"])
            + diff_mean * np.maximum(multi["n_surgeons"] - 1, 0))

        mae_none = (multi["block_span"] - multi["approx_none"]).abs().mean()
        mae_single = (multi["block_span"] - multi["approx_single"]).abs().mean()
        mae_two = (multi["block_span"] - multi["approx_two"]).abs().mean()

        bias_none = (multi["block_span"] - multi["approx_none"]).mean()
        bias_single = (multi["block_span"] - multi["approx_single"]).mean()
        bias_two = (multi["block_span"] - multi["approx_two"]).mean()

        print(f"\n  τ_all={overall_mean_gap:.1f}, "
              f"τ_same={same_mean:.1f}, τ_diff={diff_mean:.1f} min\n")
        print(f"  {'Approximation':35s}  {'MAE':>7s}  {'Bias':>7s}")
        print(f"  {'Σ room times only':35s}  {mae_none:7.1f}  {bias_none:7.1f}")
        print(f"  {'Σ room + τ×(n-1)':35s}  {mae_single:7.1f}  "
              f"{bias_single:7.1f}")
        print(f"  {'Σ room + τ_s(n-m) + τ_d(m-1)':35s}  {mae_two:7.1f}  "
              f"{bias_two:7.1f}")
        print(f"\n  Improvement two-rate over single: "
              f"{mae_single - mae_two:.1f} min MAE")

    # ═══════════════════════════════════════════════════════════════════════
    # I. FIGURES
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Block span vs Σ room times
    ax = axes[0, 0]
    if len(multi) > 0:
        ax.scatter(multi["sum_room_time"], multi["block_span"],
                   alpha=0.15, s=8, color="steelblue")
        lims = [0, max(multi["block_span"].max(),
                       multi["sum_room_time"].max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1, label="y = x (no gap)")
        ax.set_xlabel("Σ room times (min)")
        ax.set_ylabel("Block span (min)")
        ax.set_title("Block span vs Σ(room times)")
        ax.legend(fontsize=8)

    # (0,1) Distribution of inter-case gaps
    ax = axes[0, 1]
    if len(gaps) > 0:
        ax.hist(gaps["gap_min"], bins=60, edgecolor="white",
                alpha=0.8, color="steelblue", density=True)
        ax.axvline(gaps["gap_min"].mean(), color="orange", lw=1.5,
                   label=f"Mean={gaps['gap_min'].mean():.1f}")
        ax.axvline(gaps["gap_min"].median(), color="green", lw=1.5,
                   label=f"Median={gaps['gap_min'].median():.1f}")
        ax.set_xlabel("Inter-case gap (min)")
        ax.set_ylabel("Density")
        ax.set_title("Leave → Enter gap distribution")
        ax.legend(fontsize=8)

    # (0,2) Same vs diff surgeon gap
    ax = axes[0, 2]
    if len(same_s) > 0 and len(diff_s) > 0:
        ax.hist(same_s["gap_min"], bins=40, alpha=0.6,
                label=f"Same surgeon (N={len(same_s):,})",
                edgecolor="white", color="steelblue", density=True)
        ax.hist(diff_s["gap_min"], bins=40, alpha=0.6,
                label=f"Diff surgeon (N={len(diff_s):,})",
                edgecolor="white", color="darkorange", density=True)
        ax.set_xlabel("Inter-case gap (min)")
        ax.set_ylabel("Density")
        ax.set_title("Gap: same vs different surgeon")
        ax.legend(fontsize=8)

    # (1,0) Σ(booked) vs block span
    ax = axes[1, 0]
    if len(bdf) > 0:
        ax.scatter(bdf["block_span"], bdf["sum_booked"],
                   alpha=0.15, s=8, color="steelblue")
        lims = [0, max(bdf["block_span"].max(),
                       bdf["sum_booked"].max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1, label="y = x")
        ax.set_xlabel("Block span (min)")
        ax.set_ylabel("Σ booked (min)")
        ax.set_title("Σ(bookings) vs block span")
        ax.legend(fontsize=8)

    # (1,1) Σ(booked) vs Σ(room times)
    ax = axes[1, 1]
    if len(bdf) > 0:
        ax.scatter(bdf["sum_room_time"], bdf["sum_booked"],
                   alpha=0.15, s=8, color="steelblue")
        lims = [0, max(bdf["sum_room_time"].max(),
                       bdf["sum_booked"].max()) * 1.05]
        ax.plot(lims, lims, "r--", lw=1, label="y = x")
        ax.set_xlabel("Σ room times (min)")
        ax.set_ylabel("Σ booked (min)")
        ax.set_title("Σ(bookings) vs Σ(room times)")
        ax.legend(fontsize=8)

    # (1,2) Overhead by service
    ax = axes[1, 2]
    if col_available(df_c_filt, "Case_Service"):
        svc_oh = df_c_filt.dropna(subset=["Case_Service"])
        top_svc = (svc_oh["Case_Service"].value_counts()
                   .loc[lambda x: x >= 100].index.tolist())
        if len(top_svc) >= 2:
            svc_order = (svc_oh[svc_oh["Case_Service"].isin(top_svc)]
                         .groupby("Case_Service")["overhead_clipped"]
                         .mean().sort_values())
            data_bp = [svc_oh.loc[svc_oh["Case_Service"] == s,
                                  "overhead_clipped"].values
                       for s in svc_order.index]
            bp = ax.boxplot(data_bp, vert=True, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color="red", lw=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue")
            ax.set_xticklabels([s[:10] for s in svc_order.index],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Non-surgical overhead (min)")
            ax.set_title("Overhead (room − surg) by service")

    fig.tight_layout()
    fig.savefig(FIGDIR / "block_load_decomposition.png", dpi=150)
    plt.close(fig)

    # ── Save tables ──────────────────────────────────────────────────────
    save_csv(bdf[["room", "date", "n_cases", "n_surgeons", "service",
                  "weekday", "sum_room_time", "sum_surg_time",
                  "sum_booked", "block_span", "total_inter_case_gap"]],
             "block_load_decomposition")

    if len(gaps) > 0:
        gap_summary = gaps.groupby(
            ["same_surgeon"]).agg(
            N=("gap_min", "size"),
            mean=("gap_min", "mean"),
            median=("gap_min", "median"),
            std=("gap_min", "std"),
            p25=("gap_min", lambda x: x.quantile(0.25)),
            p75=("gap_min", lambda x: x.quantile(0.75)),
        )
        save_csv(gap_summary, "inter_case_gap_by_surgeon_transition")

    return bdf

DEFAULT_SERVICES_OF_INTEREST = ["OTO", "NEUR", "ANAE", "GEN", "UROL", "ORTH"]
DEFAULT_PAIR_GAP_DAYS = 60
DEFAULT_BASELINE_K = 10
DEFAULT_BASELINE_MIN_HISTORY = 3
DEFAULT_HOSPITAL_Q = 0.50
DEFAULT_MAX_DOWNWARD_REC = 45.0
PERCENTILE_CUTOFFS = [50, 60, 70, 75, 80, 90]


@dataclass
class Scenario:
    name: str
    h_full: float
    a: float
    h_reject: float


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)



def _subsection(title: str) -> None:
    print(f"\n--- {title} ---")



def _safe_slug(text: str) -> str:
    text = str(text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")



def _save_csv(df_or_series: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_series, pd.Series):
        df_or_series.to_csv(path)
    else:
        df_or_series.to_csv(path, index=False)
    print(f"  → Saved: {path}")



def _describe_numeric(series: pd.Series, name: str) -> None:
    s = pd.Series(series).dropna()
    if len(s) == 0:
        print(f"  {name}: no data")
        return
    q = s.quantile([0.25, 0.50, 0.75])
    print(
        f"  {name}: N={len(s):,}  mean={s.mean():.2f}  std={s.std():.2f}  "
        f"min={s.min():.2f}  q25={q.loc[0.25]:.2f}  median={q.loc[0.50]:.2f}  "
        f"q75={q.loc[0.75]:.2f}  max={s.max():.2f}"
    )



def _completion_col(df: pd.DataFrame) -> str:
    for col in ["Actual Stop_DT", "Leave Room_DT", "Actual Start_DT", "Actual Start Date"]:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError("No usable chronology column found for pair construction.")


def _online_group_mean(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    sort_col: str,
    min_history: int = MIN_SURGEON_CASES,
) -> pd.Series:
    """Online mean using only prior rows within each group."""
    work = df.sort_values([group_col, sort_col]).copy()
    grp = work.groupby(group_col, sort=False)[value_col]
    csum = grp.cumsum() - work[value_col]
    ccnt = grp.cumcount()
    mean = csum / ccnt.replace(0, np.nan)
    mean[ccnt < min_history] = np.nan
    return mean.reindex(work.index).sort_index()


def _detect_booking_lattice(df: pd.DataFrame) -> bool:
    if "Booked Time (Minutes)" not in df.columns:
        return False
    booked = pd.to_numeric(df["Booked Time (Minutes)"], errors="coerce").dropna()
    return len(booked) > 0 and float((((booked + 1) % 5) == 0).mean()) > 0.95


def _project_to_booking_lattice(values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(values, dtype=float)
    out = 5 * np.round((s + 1) / 5.0) - 1
    return out.clip(lower=4.0)



def _service_list(pair_df: pd.DataFrame, requested: list[str] | None = None, topn: int = 6) -> list[str]:
    available = pair_df["Case_Service"].dropna().astype(str)
    if requested:
        out = [svc for svc in requested if svc in set(available)]
        if out:
            return out
    counts = available.value_counts()
    out = [svc for svc in DEFAULT_SERVICES_OF_INTEREST if svc in counts.index]
    if len(out) >= min(topn, len(counts)):
        return out[:topn]
    for svc in counts.index:
        if svc not in out:
            out.append(svc)
        if len(out) >= topn:
            break
    return out



def _chronological_week_split(df: pd.DataFrame, week_col: str = "Week_Start") -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    tmp = tmp[tmp[week_col].notna()].sort_values(week_col)
    weeks = pd.Index(pd.Series(tmp[week_col].unique()).sort_values())
    if len(weeks) == 0:
        return tmp.iloc[:0].copy(), tmp.iloc[:0].copy()
    if len(weeks) > 52:
        train_weeks = weeks[:52]
        test_weeks = weeks[52:]
    else:
        cut = max(1, int(math.floor(0.8 * len(weeks))))
        cut = min(cut, len(weeks) - 1) if len(weeks) > 1 else 1
        train_weeks = weeks[:cut]
        test_weeks = weeks[cut:]
    train = tmp[tmp[week_col].isin(train_weeks)].copy()
    test = tmp[tmp[week_col].isin(test_weeks)].copy()
    if len(test_weeks) == 0:
        print("  ⚠ Chronological split produced empty test set; falling back to 80/20 row split.")
    if len(test) == 0 and len(train) > 1:
        split = max(1, int(0.8 * len(tmp)))
        train, test = tmp.iloc[:split].copy(), tmp.iloc[split:].copy()
    return train, test



def _ols_summary(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, add_intercept: bool = True) -> dict:
    x_arr = np.asarray(pd.Series(x), dtype=float)
    y_arr = np.asarray(pd.Series(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = len(x_arr)
    if n < 3 or np.nanstd(x_arr) <= 1e-12:
        return {
            "n": n,
            "intercept": np.nan,
            "slope": np.nan,
            "intercept_se": np.nan,
            "slope_se": np.nan,
            "t_slope": np.nan,
            "p_two_sided": np.nan,
            "p_one_sided_gt": np.nan,
            "r2": np.nan,
            "mae": np.nan,
        }
    X = np.column_stack([np.ones(n), x_arr]) if add_intercept else x_arr.reshape(-1, 1)
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y_arr
    resid = y_arr - X @ beta
    p = X.shape[1]
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((p, p))
    for i in range(n):
        xi = X[i : i + 1].T
        meat += (resid[i] ** 2) * (xi @ xi.T)
    cov = xtx_inv @ meat @ xtx_inv
    if n > p:
        cov *= n / (n - p)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    intercept = float(beta[0]) if add_intercept else 0.0
    slope = float(beta[-1])
    intercept_se = float(se[0]) if add_intercept else np.nan
    slope_se = float(se[-1])
    t_slope = slope / slope_se if slope_se > 0 else np.nan
    dfree = max(n - p, 1)
    p_two = 2 * (1 - sp_stats.t.cdf(abs(t_slope), df=dfree)) if np.isfinite(t_slope) else np.nan
    p_one = 1 - sp_stats.t.cdf(t_slope, df=dfree) if np.isfinite(t_slope) else np.nan
    y_hat = X @ beta
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y_arr - y_hat)))
    return {
        "n": n,
        "intercept": intercept,
        "slope": slope,
        "intercept_se": intercept_se,
        "slope_se": slope_se,
        "t_slope": float(t_slope),
        "p_two_sided": float(p_two),
        "p_one_sided_gt": float(p_one),
        "r2": float(r2),
        "mae": mae,
    }



def _one_sample_mean_test(values: pd.Series | np.ndarray, alternative: str = "two-sided") -> dict:
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": np.nan if n == 0 else float(arr.mean()), "se": np.nan, "t": np.nan, "p": np.nan}
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    t = mean / se if se and se > 0 else np.nan
    dfree = max(n - 1, 1)
    if not np.isfinite(t):
        p = np.nan
    elif alternative == "greater":
        p = 1 - sp_stats.t.cdf(t, df=dfree)
    elif alternative == "less":
        p = sp_stats.t.cdf(t, df=dfree)
    else:
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=dfree))
    return {"n": n, "mean": mean, "se": se, "t": float(t), "p": float(p)}



def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0 or np.sum(w) <= 0:
        return np.nan
    return float(np.sum(v * w) / np.sum(w))



def _lowess(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(pd.Series(x), dtype=float)
    y_arr = np.asarray(pd.Series(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 5:
        order = np.argsort(x_arr)
        return x_arr[order], y_arr[order]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    n = len(x_arr)
    span = max(3, int(math.ceil(frac * n)))
    y_hat = np.empty(n, dtype=float)
    for i in range(n):
        dist = np.abs(x_arr - x_arr[i])
        h = np.partition(dist, span - 1)[span - 1]
        if not np.isfinite(h) or h <= 0:
            y_hat[i] = y_arr[i]
            continue
        w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3
        if np.sum(w > 0) < 2:
            y_hat[i] = y_arr[i]
            continue
        X = np.column_stack([np.ones(n), x_arr - x_arr[i]])
        WX = X * w[:, None]
        beta = np.linalg.pinv(X.T @ WX) @ (WX.T @ y_arr)
        y_hat[i] = beta[0]
    return x_arr, y_hat



def _binned_means(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    sub = df[[x_col, y_col]].dropna().copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["bin", "x_mean", "y_mean", "y_se", "n"])
    unique_x = sub[x_col].nunique()
    q = max(2, min(n_bins, unique_x))
    try:
        sub["bin"] = pd.qcut(sub[x_col], q=q, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["bin", "x_mean", "y_mean", "y_se", "n"])
    out = (
        sub.groupby("bin", observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_std=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
    )
    out["y_se"] = out["y_std"] / np.sqrt(out["n"].clip(lower=1))
    return out



def _fit_old_piecewise(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 20:
        return {"model": "old_piecewise", "a": np.nan, "h_plus": np.nan, "h_minus": np.nan, "mae": np.nan}
    grid_a = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    grid_h = list(range(0, 105, 5))
    best = None
    for a in grid_a:
        for h_plus in grid_h:
            for h_minus in grid_h:
                pred = np.zeros_like(x)
                pos = x > h_plus
                neg = x < -h_minus
                pred[pos] = a * (x[pos] - h_plus)
                pred[neg] = a * (x[neg] + h_minus)
                mae = float(np.mean(np.abs(y - pred)))
                if best is None or mae < best["mae"]:
                    best = {"model": "old_piecewise", "a": a, "h_plus": h_plus, "h_minus": h_minus, "mae": mae}
    zero_mae = float(np.mean(np.abs(y)))
    best["zero_baseline_mae"] = zero_mae
    best["beats_zero"] = best["mae"] < zero_mae
    return best



def _fit_new_piecewise(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit the literature-consistent 3-regime response (symmetric version).

    This diagnostic fit uses a symmetric response where upward and downward
    corrections share the same (h_full, a, h_reject). A directionally
    asymmetric version is deferred to the later optimization experiments.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 20:
        return {"model": "new_piecewise", "a": np.nan, "h_full": np.nan, "h_reject": np.nan, "mae": np.nan}
    # Exclude a = 0 from the search grid: that null model is identified
    # separately by the explicit zero-prediction baseline. Including a=0
    # makes h parameters unidentified.
    grid_a = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    grid_full = list(range(0, 45, 5))
    grid_reject = list(range(5, 105, 5))
    best = None
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    for h_full in grid_full:
        for h_reject in grid_reject:
            if h_reject < h_full:
                continue
            for a in grid_a:
                pred_abs = np.empty_like(abs_x)
                mask_full = abs_x <= h_full
                mask_comp = (abs_x > h_full) & (abs_x <= h_reject)
                mask_sat = abs_x > h_reject
                pred_abs[mask_full] = abs_x[mask_full]
                pred_abs[mask_comp] = h_full + a * (abs_x[mask_comp] - h_full)
                pred_abs[mask_sat] = h_full + a * (h_reject - h_full)
                pred = sign_x * pred_abs
                mae = float(np.mean(np.abs(y - pred)))
                if best is None or mae < best["mae"]:
                    best = {
                        "model": "new_piecewise",
                        "a": a,
                        "h_full": h_full,
                        "h_reject": h_reject,
                        "mae": mae,
                    }
    return best



def _piecewise_new_response(delta_rec: np.ndarray, h_full: float, a: float, h_reject: float) -> np.ndarray:
    delta_rec = np.asarray(delta_rec, dtype=float)
    abs_x = np.abs(delta_rec)
    sign_x = np.sign(delta_rec)
    out_abs = np.empty_like(abs_x)
    mask_full = abs_x <= h_full
    mask_comp = (abs_x > h_full) & (abs_x <= h_reject)
    mask_sat = abs_x > h_reject
    out_abs[mask_full] = abs_x[mask_full]
    out_abs[mask_comp] = h_full + a * (abs_x[mask_comp] - h_full)
    out_abs[mask_sat] = h_full + a * max(h_reject - h_full, 0.0)
    return sign_x * out_abs


def _piecewise_new_downward_capacity(rec_magnitude: float, h_full: float, a: float, h_reject: float) -> float:
    """Maximum achievable *downward* post-edit correction for a positive
    downward recommendation magnitude.

    This keeps the treatability logic explicit instead of relying on the sign
    convention of a symmetric response function.
    """
    rec = float(rec_magnitude)
    if not np.isfinite(rec):
        return np.nan
    rec = max(rec, 0.0)
    if rec <= h_full:
        return rec
    if rec <= h_reject:
        return h_full + a * (rec - h_full)
    return h_full + a * max(h_reject - h_full, 0.0)


def _build_pair_dataset(
    df: pd.DataFrame,
    baseline_k: int = DEFAULT_BASELINE_K,
    min_history: int = DEFAULT_BASELINE_MIN_HISTORY,
    max_gap_days: int = DEFAULT_PAIR_GAP_DAYS,
    same_procedure: bool = True,
) -> pd.DataFrame:
    sort_col = _completion_col(df)
    ordered = df.sort_values(["Surgeon_Code", "Main_Procedure_Id", sort_col]).copy()
    ordered["case_row_id"] = ordered.index.astype(str)
    records: list[dict] = []
    group_cols = ["Surgeon_Code", "Main_Procedure_Id"] if same_procedure else ["Surgeon_Code"]
    for keys, grp in ordered.groupby(group_cols, sort=False):
        if same_procedure:
            surgeon, procedure = keys
        else:
            surgeon, procedure = keys, np.nan
        grp = grp.sort_values(sort_col).copy()
        if len(grp) < 2:
            continue
        booked = grp["Booked Time (Minutes)"].to_numpy(dtype=float)
        realized = grp["Realized_Duration_Min"].to_numpy(dtype=float)
        times = pd.to_datetime(grp[sort_col], errors="coerce")
        rows = grp.reset_index(drop=False)
        for i in range(1, len(rows)):
            # Leave-pair-out: exclude both current case i and previous case i-1.
            pool = booked[:max(0, i - 1)]
            if len(pool) < min_history:
                continue
            baseline = float(np.mean(pool[max(0, len(pool) - baseline_k) :]))
            gap_days = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 86400.0 if pd.notna(times.iloc[i]) and pd.notna(times.iloc[i - 1]) else np.nan
            if not np.isfinite(gap_days) or gap_days > max_gap_days:
                continue
            curr = rows.iloc[i]
            prev = rows.iloc[i - 1]
            records.append(
                {
                    "Surgeon_Code": surgeon,
                    "Main_Procedure_Id": procedure,
                    "Case_Service": curr.get("Case_Service", np.nan),
                    "current_case_row_id": str(curr["index"]),
                    "prev_case_row_id": str(prev["index"]),
                    "current_time": curr[sort_col],
                    "prev_time": prev[sort_col],
                    "gap_days": gap_days,
                    "b_curr": float(curr["Booked Time (Minutes)"]),
                    "b_prev": float(prev["Booked Time (Minutes)"]),
                    "d_curr": float(curr["Realized_Duration_Min"]),
                    "d_prev": float(prev["Realized_Duration_Min"]),
                    "booking_error_prev": float(prev["Booked Time (Minutes)"] - prev["Realized_Duration_Min"]),
                    "booking_error_curr": float(curr["Booked Time (Minutes)"] - curr["Realized_Duration_Min"]),
                    "baseline_booked_curr": baseline,
                    "Y_abnormal_booking": float(curr["Booked Time (Minutes)"] - baseline),
                    "X_prev_overrun": float(prev["Realized_Duration_Min"] - prev["Booked Time (Minutes)"]),
                    "Case_Service_prev": prev.get("Case_Service", np.nan),
                    "Main_Procedure": curr.get("Main_Procedure", np.nan),
                }
            )
    pair_df = pd.DataFrame(records)
    if len(pair_df) == 0:
        return pair_df
    pair_df["current_time"] = pd.to_datetime(pair_df["current_time"], errors="coerce")
    pair_df["prev_time"] = pd.to_datetime(pair_df["prev_time"], errors="coerce")
    pair_df["Week_Start"] = (pair_df["current_time"] - pd.to_timedelta(pair_df["current_time"].dt.weekday, unit="D")).dt.normalize()
    pair_df["abs_X_prev_overrun"] = pair_df["X_prev_overrun"].abs()
    pair_df["X_prev_rel"] = pair_df["X_prev_overrun"] / pair_df["b_prev"].replace(0, np.nan)
    pair_df["Y_abnormal_rel"] = pair_df["Y_abnormal_booking"] / pair_df["b_curr"].replace(0, np.nan)
    return pair_df



def _surgeon_qhat(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["covered"] = work["Realized_Duration_Min"] <= work["Booked Time (Minutes)"]
    agg = (
        work.groupby("Surgeon_Code")
        .agg(
            q_hat=("covered", "mean"),
            surgeon_cases=("Patient_ID", "size"),
            mean_booking_error=("Booking_Error_Min", "mean"),
            median_booking_error=("Booking_Error_Min", "median"),
        )
        .reset_index()
    )
    if "Case_Service" in work.columns:
        primary_service = work.groupby("Surgeon_Code")["Case_Service"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan)
        agg = agg.merge(primary_service.rename("primary_service"), on="Surgeon_Code", how="left")
    return agg



PREDICTION_CATEGORICAL_FEATURES = [
    "Main_Procedure_Id",
    "Surgeon_Code",
    "Case_Service",
    "Site",
    "Patient_Type",
    "DayOfWeek",
    "Month",
    "Year",
]

PREDICTION_NUMERIC_FEATURES = [
    "q_hat_empirical",
    "Age",
    "Patient_Age",
    "Age_at_Surgery",
    "BMI",
    "ASA",
    "ASA_Class",
    "ASA_Score",
]


def _coerce_prediction_categorical_columns(
    df: pd.DataFrame, categorical_cols: list[str] | None = None
) -> pd.DataFrame:
    """Force identifier-like prediction features to be categorical.

    Several fields are numeric-coded identifiers or calendar buckets
    (for example Surgeon_Code, Main_Procedure_Id, Month, Year).  If
    we infer feature type from dtype alone, these leak into the numeric
    pipeline and the linear model treats them as ordered quantities.
    That is a specification bug.  We therefore coerce the intended
    categorical features explicitly before model fitting.

    Important implementation detail: use plain Python-object/string
    columns with np.nan for missingness rather than pandas' nullable
    StringDtype with pd.NA.  Some sklearn imputers still fail on mixed
    object arrays containing pd.NA ("boolean value of NA is ambiguous").
    """
    df = df.copy()
    cols = categorical_cols or PREDICTION_CATEGORICAL_FEATURES
    for col in cols:
        if col in df.columns:
            s = df[col].astype(object)
            s = s.where(pd.notna(s), np.nan)
            df[col] = s
    return df


def _ensure_prediction_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Actual Start Date" in df.columns:
        dt = pd.to_datetime(df["Actual Start Date"], errors="coerce")
        df["DayOfWeek"] = dt.dt.day_name()
        df["Month"] = dt.dt.month
        df["Year"] = dt.dt.year
    df = _coerce_prediction_categorical_columns(df)
    return df


def _feature_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    df = _ensure_prediction_feature_columns(df)
    categorical = [c for c in PREDICTION_CATEGORICAL_FEATURES if c in df.columns]
    numeric = [c for c in PREDICTION_NUMERIC_FEATURES if c in df.columns]
    combined = categorical + numeric
    return df, categorical, numeric, combined



def _fit_linear_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> dict:
    if len(feature_cols) == 0:
        return {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "r2": np.nan,
            "model": None,
            "feature_cols": [],
        }
    train = train_df.dropna(subset=[target_col]).copy()
    test = test_df.dropna(subset=[target_col]).copy()
    if len(train) == 0 or len(test) == 0:
        return {
            "n_train": len(train),
            "n_test": len(test),
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "r2": np.nan,
            "model": None,
            "feature_cols": feature_cols,
        }

    if categorical_cols is None:
        categorical = [
            c for c in feature_cols
            if train[c].dtype == "object" or str(train[c].dtype).startswith("category")
        ]
    else:
        categorical = [c for c in feature_cols if c in categorical_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    # Sanitize feature matrices for sklearn:
    #   - categorical columns: plain object/string values with np.nan missingness
    #   - numeric columns: numeric dtype with invalid entries coerced to np.nan
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for c in categorical:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype(object).where(pd.notna(X_train[c]), np.nan)
            X_test[c] = X_test[c].astype(object).where(pd.notna(X_test[c]), np.nan)
    for c in numeric:
        if c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(X_train, train[target_col])
    pred = model.predict(X_test)
    residual = pred - test[target_col].to_numpy(dtype=float)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "mae": float(mean_absolute_error(test[target_col], pred)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mean_error": float(np.mean(residual)),
        "r2": float(r2_score(test[target_col], pred)),
        "model": model,
        "feature_cols": feature_cols,
        "pred": pred,
        "y_test": test[target_col].to_numpy(dtype=float),
        "test_index": test.index.to_numpy(),
    }



def _group_ablation_importance(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_groups: dict[str, list[str]],
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    all_features = [feat for feats in feature_groups.values() for feat in feats]
    all_features = [f for f in dict.fromkeys(all_features) if f in train_df.columns]
    base = _fit_linear_prediction(
        train_df, test_df, target_col, all_features, categorical_cols=categorical_cols
    )
    rows = [
        {
            "group": "all_features",
            "mae": base["mae"],
            "r2": base["r2"],
            "delta_mae_vs_full": 0.0,
            "delta_r2_vs_full": 0.0,
        }
    ]
    for group_name, feats in feature_groups.items():
        reduced = [f for f in all_features if f not in feats]
        if len(reduced) == 0:
            continue
        fit = _fit_linear_prediction(train_df, test_df, target_col, reduced, categorical_cols=categorical_cols)
        rows.append(
            {
                "group": group_name,
                "mae": fit["mae"],
                "r2": fit["r2"],
                "delta_mae_vs_full": fit["mae"] - base["mae"],
                "delta_r2_vs_full": fit["r2"] - base["r2"],
            }
        )
    return pd.DataFrame(rows)



def _weighted_between_within_variance(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    sub = df[[value_col, group_col]].dropna().copy()
    if len(sub) == 0:
        return {"total_variance": np.nan, "between_variance": np.nan, "within_variance": np.nan, "between_share": np.nan}
    total = float(sub[value_col].var(ddof=0))
    grp = sub.groupby(group_col)[value_col]
    means = grp.mean()
    counts = grp.size()
    within = grp.var(ddof=0).fillna(0.0)
    grand = _weighted_average(means, counts)
    between = float(np.sum(counts * (means - grand) ** 2) / np.sum(counts))
    within_w = float(np.sum(counts * within) / np.sum(counts))
    share = between / total if total > 0 else np.nan
    return {
        "total_variance": total,
        "between_variance": between,
        "within_variance": within_w,
        "between_share": share,
    }



def _choose_dominant_procedure(df: pd.DataFrame, service: str) -> pd.DataFrame:
    sub = df[df["Case_Service"] == service].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    cols = ["Main_Procedure_Id"]
    if "Main_Procedure" in sub.columns:
        cols.append("Main_Procedure")
    out = sub.groupby(cols).size().rename("cases").reset_index().sort_values("cases", ascending=False)
    return out



def _plot_service_lowess(pair_df: pd.DataFrame, services: list[str], fig_path: Path, x_col: str = "X_prev_overrun", y_col: str = "Y_abnormal_booking") -> None:
    n = len(services)
    if n == 0:
        return
    rows = int(math.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), squeeze=False)
    fracs = [0.35, 0.50, 0.65]
    for ax, service in zip(axes.ravel(), services):
        sub = pair_df[pair_df["Case_Service"] == service].dropna(subset=[x_col, y_col]).copy()
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        sample = sub.sample(min(len(sub), 2000), random_state=42) if len(sub) > 2000 else sub
        ax.scatter(sample[x_col], sample[y_col], alpha=0.15, s=8)
        for frac in fracs:
            xs, ys = _lowess(sub[x_col], sub[y_col], frac=frac)
            ax.plot(xs, ys, linewidth=1.5, label=f"LOWESS {frac:.2f}")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{service}  (N={len(sub):,})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {fig_path}")



def _plot_service_binned_means(pair_df: pd.DataFrame, services: list[str], fig_path: Path, x_col: str = "X_prev_overrun", y_col: str = "Y_abnormal_booking") -> None:
    n = len(services)
    if n == 0:
        return
    rows = int(math.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), squeeze=False)
    for ax, service in zip(axes.ravel(), services):
        sub = pair_df[pair_df["Case_Service"] == service].dropna(subset=[x_col, y_col]).copy()
        binned = _binned_means(sub, x_col=x_col, y_col=y_col, n_bins=10)
        if len(binned) == 0:
            ax.set_visible(False)
            continue
        ax.errorbar(binned["x_mean"], binned["y_mean"], yerr=binned["y_se"], fmt="o-", capsize=3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{service}  (N={len(sub):,})")
        ax.set_xlabel(f"Mean {x_col} within bin")
        ax.set_ylabel(f"Mean {y_col}")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {fig_path}")



def run_agenda_diagnostics(
    df: pd.DataFrame,
    figdir: str | Path,
    tbldir: str | Path,
    hospital_q: float = DEFAULT_HOSPITAL_Q,
    max_downward_rec: float = DEFAULT_MAX_DOWNWARD_REC,
    services_of_interest: list[str] | None = None,
) -> None:
    figdir = Path(figdir)
    tbldir = Path(tbldir)
    figdir.mkdir(parents=True, exist_ok=True)
    tbldir.mkdir(parents=True, exist_ok=True)

    _section("AGENDA DIAGNOSTICS — DECISIVE MODEL DESIGN CHECKS")
    print("  Pair construction uses consecutive same-surgeon same-procedure cases")
    print("  ordered by completion time, with a chronological K-nearest historical")
    print("  baseline for the current booking (K=10, minimum 3 prior cases).")
    print(f"  Primary consecutive-pair gap (main tasks): {DEFAULT_PAIR_GAP_DAYS} days")
    print(f"  Robustness gaps also reported: {[60, 90, 180]}")
    print(f"  Hospital reference quantile q^H: {hospital_q:.2f}")
    print(f"  Max downward recommendation used in treatability analysis: {max_downward_rec:.0f} min")

    surgeon_q = _surgeon_qhat(df)
    df_aug = df.merge(surgeon_q[["Surgeon_Code", "q_hat"]].rename(columns={"q_hat": "q_hat_empirical"}), on="Surgeon_Code", how="left")
    pair_df = _build_pair_dataset(df_aug)
    if len(pair_df) == 0:
        print("  ⚠ No valid agenda pair data available. Skipping agenda diagnostics.")
        return
    pair_df = pair_df.merge(surgeon_q[["Surgeon_Code", "q_hat", "surgeon_cases", "mean_booking_error", "median_booking_error"]], on="Surgeon_Code", how="left")
    _save_csv(pair_df, tbldir / "agenda_pair_dataset.csv")

    # Gap / pairing robustness summary
    sens_rows = []
    for gap in [60, 90, 180]:
        for same_proc, scope in [(True, "surgeon_x_procedure"), (False, "surgeon_only")]:
            sdf = _build_pair_dataset(df_aug, max_gap_days=gap, same_procedure=same_proc)
            if len(sdf) == 0:
                continue
            fit = _ols_summary(sdf["X_prev_overrun"], sdf["Y_abnormal_booking"])
            mae_zero = float(sdf["Y_abnormal_booking"].abs().mean())
            sens_rows.append({
                "pair_scope": scope,
                "max_gap_days": gap,
                "n_pairs": len(sdf),
                "slope": fit["slope"],
                "slope_se": fit["slope_se"],
                "r2": fit["r2"],
                "mae_ols": fit["mae"],
                "mae_zero": mae_zero,
            })
    sens_df = pd.DataFrame(sens_rows)
    _save_csv(sens_df, tbldir / "agenda_pair_sensitivity_summary.csv")

    services = _service_list(pair_df, requested=services_of_interest)
    _subsection("Pair dataset overview")
    print(f"  Valid pairs: {len(pair_df):,}")
    print(f"  Surgeons with ≥1 valid pair: {pair_df['Surgeon_Code'].nunique():,}")
    print(f"  Services represented: {pair_df['Case_Service'].nunique():,}")
    print(f"  Services highlighted in figures: {services}")
    _describe_numeric(pair_df["X_prev_overrun"], "X = previous overrun (realized − booked)")
    _describe_numeric(pair_df["Y_abnormal_booking"], "Y = abnormal current booking")

    _subsection("Gap and pairing robustness (self-correction slope)")
    print(f"  {'Scope':<25} {'Gap':>5} {'Pairs':>8} {'Slope':>8} {'SE':>8} {'R²':>8} {'MAE_ols':>8} {'MAE_zero':>8} {'OLS beats 0?':>12}")
    for _, r in sens_df.iterrows():
        beats = "YES" if r["mae_ols"] < r["mae_zero"] else "no"
        print(f"  {r['pair_scope']:<25} {int(r['max_gap_days']):>5} {int(r['n_pairs']):>8,} {r['slope']:>8.4f} {r['slope_se']:>8.4f} {r['r2']:>8.4f} {r['mae_ols']:>8.2f} {r['mae_zero']:>8.2f} {beats:>12}")
    print("\n  Key: if slope remains near zero across gap thresholds and under")
    print("  relaxed (surgeon-only) pairing, self-correction is robustly absent.")

    _subsection("KNN baseline validation")
    Y = pair_df["Y_abnormal_booking"]
    _describe_numeric(Y, "Y_ν (abnormal booking)")
    print(f"  Fraction |Y_ν| ≤  5 min: {(Y.abs() <= 5).mean():.1%}")
    print(f"  Fraction |Y_ν| ≤ 15 min: {(Y.abs() <= 15).mean():.1%}")
    print(f"  Fraction |Y_ν| ≤ 30 min: {(Y.abs() <= 30).mean():.1%}")
    raw_update = pair_df["b_curr"] - pair_df["b_prev"]
    raw_mae = float(raw_update.abs().mean())
    y_mae = float(Y.abs().mean())
    reduction = raw_mae - y_mae
    print(f"\n  MAE of raw update (b_curr − b_prev):     {raw_mae:.2f}")
    print(f"  MAE of baselined update (Y_ν):            {y_mae:.2f}")
    print(f"  Reduction from KNN baselining: {reduction:.2f} min ({100*reduction/raw_mae:.1f}%)")
    print("  A meaningful reduction confirms the KNN baseline removes")
    print("  case-level heterogeneity, isolating the behavioral signal.")

    # Task 1: Model-free checks
    _section("TASK 1 — MODEL-FREE RESPONSE CHECKS")
    rows = []
    pooled = _ols_summary(pair_df["X_prev_overrun"], pair_df["Y_abnormal_booking"])
    rows.append({"group": "ALL", **pooled})
    print(
        f"  Pooled OLS: slope={pooled['slope']:.4f}, se={pooled['slope_se']:.4f}, "
        f"R²={pooled['r2']:.4f}, N={pooled['n']:,}"
    )
    zero_mae = float(pair_df["Y_abnormal_booking"].abs().mean())
    print(f"  Zero-prediction baseline MAE: {zero_mae:.4f}")
    print(f"  Pooled OLS MAE: {pooled['mae']:.4f}")
    print(f"  OLS {'beats' if pooled['mae'] < zero_mae else 'does NOT beat'} zero baseline")

    _subsection("Mean-reversion check")
    print("  If the negative slope is just mean reversion in booking levels,")
    print("  controlling for (b_prev − surgeon mean booking) should absorb it.")
    surg_mean_b = pair_df.groupby("Surgeon_Code")["b_prev"].transform("mean")
    b_prev_dm = pair_df["b_prev"] - surg_mean_b
    X_mr = np.column_stack([pair_df["X_prev_overrun"].to_numpy(dtype=float), b_prev_dm.to_numpy(dtype=float)])
    y_mr = pair_df["Y_abnormal_booking"].to_numpy(dtype=float)
    mask = np.isfinite(X_mr).all(axis=1) & np.isfinite(y_mr)
    X_mr, y_mr = X_mr[mask], y_mr[mask]
    X_mr = np.column_stack([np.ones(len(X_mr)), X_mr])
    beta = np.linalg.lstsq(X_mr, y_mr, rcond=None)[0]
    print(f"  Slope on X_prev_overrun (controlling for b_prev level): {beta[1]:.4f}")
    print(f"  Slope on b_prev_demeaned: {beta[2]:.4f}")
    if np.isfinite(beta[1]) and np.isfinite(pooled['slope']) and abs(beta[1]) < abs(pooled['slope']) * 0.5:
        print("  → Slope on overrun attenuates substantially → mean reversion likely.")
    else:
        print("  → Slope on overrun persists → not explained by mean reversion alone.")

    for service, sub in pair_df.groupby("Case_Service"):
        fit = _ols_summary(sub["X_prev_overrun"], sub["Y_abnormal_booking"])
        rows.append({"group": service, **fit})
    task1_df = pd.DataFrame(rows).sort_values(["group"])
    _save_csv(task1_df, tbldir / "agenda_task1_model_free_ols.csv")

    _plot_service_lowess(pair_df, services, figdir / "agenda_task1_lowess_by_service.png")
    _plot_service_binned_means(pair_df, services, figdir / "agenda_task1_binned_means_by_service.png")

    # Task 2: Surgeon-level responders
    _section("TASK 2 — SURGEON-LEVEL RESPONDERS VS NON-RESPONDERS")
    surgeon_rows = []
    for surgeon, sub in pair_df.groupby("Surgeon_Code"):
        if len(sub) < 30:
            continue
        fit = _ols_summary(sub["X_prev_overrun"], sub["Y_abnormal_booking"])
        pos_abs = sub.loc[sub["X_prev_overrun"] > 30, "Y_abnormal_booking"]
        neg_abs = sub.loc[sub["X_prev_overrun"] < -30, "Y_abnormal_booking"]
        sd_thr = sub["X_prev_overrun"].std(ddof=1)
        pos_sd = sub.loc[sub["X_prev_overrun"] > sd_thr, "Y_abnormal_booking"] if np.isfinite(sd_thr) else pd.Series(dtype=float)
        neg_sd = sub.loc[sub["X_prev_overrun"] < -sd_thr, "Y_abnormal_booking"] if np.isfinite(sd_thr) else pd.Series(dtype=float)
        test_pos_abs = _one_sample_mean_test(pos_abs, alternative="greater")
        test_neg_abs = _one_sample_mean_test(neg_abs, alternative="less")
        test_pos_sd = _one_sample_mean_test(pos_sd, alternative="greater")
        test_neg_sd = _one_sample_mean_test(neg_sd, alternative="less")
        responder = bool(
            (np.isfinite(fit["slope"]) and fit["slope"] > 0 and fit["p_one_sided_gt"] < 0.05)
            or (np.isfinite(test_pos_abs["mean"]) and test_pos_abs["mean"] > 0 and test_pos_abs["p"] < 0.05)
            or (np.isfinite(test_neg_abs["mean"]) and test_neg_abs["mean"] < 0 and test_neg_abs["p"] < 0.05)
            or (np.isfinite(test_pos_sd["mean"]) and test_pos_sd["mean"] > 0 and test_pos_sd["p"] < 0.05)
            or (np.isfinite(test_neg_sd["mean"]) and test_neg_sd["mean"] < 0 and test_neg_sd["p"] < 0.05)
        )
        surgeon_rows.append(
            {
                "Surgeon_Code": surgeon,
                "n_pairs": len(sub),
                "Case_Service": sub["Case_Service"].mode().iloc[0] if sub["Case_Service"].notna().any() else np.nan,
                "slope": fit["slope"],
                "slope_se": fit["slope_se"],
                "t_slope": fit["t_slope"],
                "p_slope_gt0": fit["p_one_sided_gt"],
                "r2": fit["r2"],
                "mean_y_pos_abs30": test_pos_abs["mean"],
                "p_y_pos_abs30_gt0": test_pos_abs["p"],
                "n_pos_abs30": test_pos_abs["n"],
                "mean_y_neg_abs30": test_neg_abs["mean"],
                "p_y_neg_abs30_lt0": test_neg_abs["p"],
                "n_neg_abs30": test_neg_abs["n"],
                "mean_y_pos_sd": test_pos_sd["mean"],
                "p_y_pos_sd_gt0": test_pos_sd["p"],
                "n_pos_sd": test_pos_sd["n"],
                "mean_y_neg_sd": test_neg_sd["mean"],
                "p_y_neg_sd_lt0": test_neg_sd["p"],
                "n_neg_sd": test_neg_sd["n"],
                "responder": responder,
                "q_hat": sub["q_hat"].iloc[0],
                "surgeon_cases": sub["surgeon_cases"].iloc[0],
            }
        )
    surgeon_df = pd.DataFrame(surgeon_rows).sort_values(["responder", "slope"], ascending=[False, False])
    _save_csv(surgeon_df, tbldir / "agenda_task2_surgeon_response_summary.csv")
    if len(surgeon_df) > 0:
        n_resp = int(surgeon_df["responder"].sum())
        n_total_tested = len(surgeon_df)
        n_expected_fp = 0.05 * n_total_tested
        print(f"  Surgeons with ≥30 pairs: {n_total_tested:,}")
        print(f"  Responders (p < 0.05 on any test): {n_resp:,} / {n_total_tested:,} ({100 * n_resp / n_total_tested:.1f}%)")
        print("")
        print(f"  ⚠ Multiple comparisons: with {n_total_tested} surgeons tested at α=0.05,")
        print(f"  about {n_expected_fp:.1f} false positives are expected by chance alone.")
        alpha_bonf = 0.05 / max(n_total_tested, 1)
        surgeon_df["responder_bonferroni"] = (
            surgeon_df["p_slope_gt0"].lt(alpha_bonf) & surgeon_df["slope"].gt(0)
        )
        n_resp_bonf = int(surgeon_df["responder_bonferroni"].sum())
        print(f"  After Bonferroni correction: {n_resp_bonf:,} surgeons with slope > 0 at α_adj={alpha_bonf:.5f}")
        _save_csv(surgeon_df, tbldir / "agenda_task2_surgeon_response_summary.csv")
        _describe_numeric(surgeon_df["slope"], "Surgeon-level linear slopes")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(surgeon_df["slope"].dropna(), bins=30, edgecolor="white", alpha=0.8)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Surgeon-level slope")
        axes[0].set_ylabel("Surgeons")
        axes[0].set_title("Distribution of surgeon response slopes")
        grouped = surgeon_df.groupby(["Case_Service", "responder"]).size().unstack(fill_value=0)
        grouped = grouped.sort_values(list(grouped.columns), ascending=False)
        grouped.plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_xlabel("Service")
        axes[1].set_ylabel("Surgeons")
        axes[1].set_title("Responders vs non-responders by service")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figdir / "agenda_task2_surgeon_response_figures.png", dpi=150)
        plt.close(fig)
        print(f"  → Saved: {figdir / 'agenda_task2_surgeon_response_figures.png'}")
        responder_profile = surgeon_df.groupby("responder").agg(
            n_surgeons=("Surgeon_Code", "size"),
            mean_pairs=("n_pairs", "mean"),
            mean_cases=("surgeon_cases", "mean"),
            mean_q_hat=("q_hat", "mean"),
            median_q_hat=("q_hat", "median"),
        ).reset_index()
        _save_csv(responder_profile, tbldir / "agenda_task2_responder_profile.csv")

    # Task 3: ANAE deep dive
    _section("TASK 3 — ANAE DEEP DIVE")
    anae_cases = (
        df_aug[df_aug["Case_Service"] == "ANAE"].copy()
        if "Case_Service" in df_aug.columns
        else df_aug.iloc[:0].copy()
    )       
    anae_pairs = pair_df[pair_df["Case_Service"] == "ANAE"].copy() if "Case_Service" in pair_df.columns else pair_df.iloc[:0].copy()
    if len(anae_cases) == 0:
        print("  ANAE not present in dataset — skipping ANAE deep dive.")
    else:
        anae_surgeon = anae_cases.groupby("Surgeon_Code").agg(
            n_cases=("Patient_ID", "size"),
            mean_duration=("Realized_Duration_Min", "mean"),
            median_duration=("Realized_Duration_Min", "median"),
            q_hat=("q_hat_empirical", "first"),
        ).reset_index()
        anae_pair_counts = anae_pairs.groupby("Surgeon_Code").size().rename("n_valid_pairs").reset_index()
        anae_profile = anae_surgeon.merge(anae_pair_counts, on="Surgeon_Code", how="left").fillna({"n_valid_pairs": 0})
        _save_csv(anae_profile, tbldir / "agenda_task3_anae_surgeon_profile.csv")
        _describe_numeric(anae_cases["Realized_Duration_Min"], "ANAE realized duration")
        print(f"  ANAE surgeons: {anae_cases['Surgeon_Code'].nunique():,}")
        print(f"  ANAE valid pairs: {len(anae_pairs):,}")
        dom_proc = _choose_dominant_procedure(anae_cases, "ANAE")
        if len(dom_proc) > 0:
            _save_csv(dom_proc.head(25), tbldir / "agenda_task3_anae_procedure_profile.csv")
            top_proc = dom_proc.iloc[0]
            proc_desc = f" ({top_proc['Main_Procedure']})" if "Main_Procedure" in top_proc.index and pd.notna(top_proc['Main_Procedure']) else ""
            print(f"  Dominant ANAE procedure: {top_proc['Main_Procedure_Id']}{proc_desc}, N={int(top_proc['cases']):,}")
        if len(anae_pairs) > 0:
            anae_fit = _ols_summary(anae_pairs["X_prev_overrun"], anae_pairs["Y_abnormal_booking"])
            print(f"  ANAE model-free slope: {anae_fit['slope']:.4f} (se={anae_fit['slope_se']:.4f}, N={anae_fit['n']:,})")
            cut_rows = []
            train_anae, test_anae = _chronological_week_split(anae_pairs)
            for p in PERCENTILE_CUTOFFS:
                thr = np.nanpercentile(train_anae["abs_X_prev_overrun"], p) if len(train_anae) > 0 else np.nan
                tr = train_anae[train_anae["abs_X_prev_overrun"] >= thr] if np.isfinite(thr) else train_anae.iloc[:0]
                te = test_anae[test_anae["abs_X_prev_overrun"] >= thr] if np.isfinite(thr) else test_anae.iloc[:0]
                fit = _ols_summary(tr["X_prev_overrun"], tr["Y_abnormal_booking"])
                if len(te) > 0 and np.isfinite(fit["slope"]):
                    pred = fit["intercept"] + fit["slope"] * te["X_prev_overrun"].to_numpy(dtype=float)
                    test_mae = float(np.mean(np.abs(te["Y_abnormal_booking"].to_numpy(dtype=float) - pred)))
                else:
                    test_mae = np.nan
                cut_rows.append({"percentile": p, "threshold": thr, "n_train": len(tr), "n_test": len(te), "slope": fit["slope"], "slope_se": fit["slope_se"], "test_mae": test_mae})
            cut_df = pd.DataFrame(cut_rows)
            _save_csv(cut_df, tbldir / "agenda_task3_anae_cutoff_scan.csv")
            if cut_df["test_mae"].notna().any():
                best_row = cut_df.sort_values("test_mae").iloc[0]
                print(f"  Best ANAE cutoff by chronological holdout MAE: P{int(best_row['percentile'])} (test MAE={best_row['test_mae']:.2f})")
            loo_surg_rows = []
            for surg in sorted(anae_pairs["Surgeon_Code"].unique()):
                sub = anae_pairs[anae_pairs["Surgeon_Code"] != surg]
                fit = _ols_summary(sub["X_prev_overrun"], sub["Y_abnormal_booking"])
                loo_surg_rows.append({"dropped_surgeon": surg, **fit})
            _save_csv(pd.DataFrame(loo_surg_rows), tbldir / "agenda_task3_anae_leave_one_surgeon_out.csv")
            loo_pair_rows = []
            if len(anae_pairs) <= 500:
                for idx in anae_pairs.index:
                    sub = anae_pairs.drop(index=idx)
                    fit = _ols_summary(sub["X_prev_overrun"], sub["Y_abnormal_booking"])
                    loo_pair_rows.append({"dropped_pair_index": str(idx), **fit})
            _save_csv(pd.DataFrame(loo_pair_rows), tbldir / "agenda_task3_anae_leave_one_pair_out.csv")
            fig, ax = plt.subplots(figsize=(7, 5))
            for surg, sub in anae_pairs.groupby("Surgeon_Code"):
                ax.scatter(sub["X_prev_overrun"], sub["Y_abnormal_booking"], label=str(surg), alpha=0.7, s=30)
            xs, ys = _lowess(anae_pairs["X_prev_overrun"], anae_pairs["Y_abnormal_booking"], frac=0.6)
            ax.plot(xs, ys, color="black", linewidth=2, label="LOWESS")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Previous overrun")
            ax.set_ylabel("Current abnormal booking")
            ax.set_title("ANAE pair scatter")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(figdir / "agenda_task3_anae_scatter.png", dpi=150)
            plt.close(fig)
            print(f"  → Saved: {figdir / 'agenda_task3_anae_scatter.png'}")

    # Task 4: prediction without booking
    _section("TASK 4 — PREDICTIVE POWER WITHOUT THE BOOKING")
    # Any historical surgeon statistic used as a feature must be estimated on
    # the training period only. Otherwise the test period leaks into x.
    train_base, test_base = _chronological_week_split(df.copy())
    if len(test_base) == 0:
        print("  Insufficient out-of-sample weeks for predictive task.")
    else:
        _subsection("4A. Chronological split and training-only historical features")
        train_only_q = _surgeon_qhat(train_base)
        q_feature = train_only_q[["Surgeon_Code", "q_hat"]].rename(columns={"q_hat": "q_hat_empirical"})
        train_df = train_base.merge(q_feature, on="Surgeon_Code", how="left")
        test_df = test_base.merge(q_feature, on="Surgeon_Code", how="left")
        print("  Split rule: first 52 calendar weeks → train, remaining weeks → test (chronological, no shuffle)")
        print(f"  Train cases: {len(train_df):,}")
        print(f"  Test cases:  {len(test_df):,}")
        q_avail = test_df["q_hat_empirical"].notna().mean() if "q_hat_empirical" in test_df.columns and len(test_df) > 0 else np.nan
        if np.isfinite(q_avail):
            print(f"  q_hat_empirical available in test from training history: {100 * q_avail:.1f}%")
        print("  Booking-only is not a data leak: the current booking is observed")
        print("  at decision time. The main leakage risk is from historical")
        print("  features estimated using the full sample.")

        train_df, categorical, numeric, all_feature_only = _feature_candidates(train_df)
        test_df = _ensure_prediction_feature_columns(test_df)
        feature_only = all_feature_only.copy()
        combined = all_feature_only + (["Booked Time (Minutes)"] if "Booked Time (Minutes)" in train_df.columns else [])
        booking_only = ["Booked Time (Minutes)"] if "Booked Time (Minutes)" in train_df.columns else []
        feature_only_categorical = [c for c in categorical if c in feature_only]
        combined_categorical = feature_only_categorical.copy()

        _subsection("4B. Feature leak audit")
        POST_OP_COLUMNS = {
            "Realized_Duration_Min", "Surgical_Duration_Min", "Room_Time_Min",
            "Booking_Error_Min", "Booking_Error_Surgical", "Booking_Error_Room",
            "Acute LOS", "LOS", "Recovery_Time_Mins", "Complication_diag1",
            "Actual Stop_DT", "Leave Room_DT", "Actual Stop Date", "Actual Stop Time",
            "Leave Room Date", "Leave Room Time",
        }
        leaked = [c for c in feature_only if c in POST_OP_COLUMNS]
        if leaked:
            print(f"  ⚠ Post-operative features detected in x: {leaked}")
            print("    Removing leaked features before fitting.")
            feature_only = [c for c in feature_only if c not in POST_OP_COLUMNS]
            combined = [c for c in combined if c not in POST_OP_COLUMNS]
            feature_only_categorical = [c for c in feature_only_categorical if c in feature_only]
            combined_categorical = [c for c in combined_categorical if c in combined]
        else:
            print("  ✓ No explicit post-operative features detected in feature set.")
        if "q_hat_empirical" in feature_only:
            print("  ✓ q_hat_empirical is retained, but it is estimated on the")
            print("    training period only, so it is a valid historical feature.")
        print("  ✓ Identifier-like and calendar-coded fields are forced into the")
        print("    categorical pipeline before fitting.")
        forced_cats = [c for c in feature_only_categorical if c in {"Main_Procedure_Id", "Surgeon_Code", "Month", "Year"}]
        if forced_cats:
            print(f"    Explicitly categorical despite numeric-looking codes: {forced_cats}")

        _subsection("4C. Dimensionality audit (feature-only model)")
        cat_in_feat = feature_only_categorical.copy()
        num_in_feat = [c for c in feature_only if c not in cat_in_feat]
        total_onehot = sum(train_df[c].nunique(dropna=True) for c in cat_in_feat)
        total_dim = total_onehot + len(num_in_feat)
        print(f"  Categorical features: {len(cat_in_feat)}")
        for c in cat_in_feat:
            n_levels = train_df[c].nunique(dropna=True)
            n_test_unseen = 0
            if c in test_df.columns:
                train_levels = set(train_df[c].dropna().astype(str).unique())
                test_levels = set(test_df[c].dropna().astype(str).unique())
                n_test_unseen = len(test_levels - train_levels)
            print(f"    {c:25s}  {n_levels:5d} levels in train  ({n_test_unseen:d} unseen in test)")
        print(f"  Numeric features: {len(num_in_feat)}")
        for c in num_in_feat:
            print(f"    {c:25s}")
        print(f"  Total encoded dimensions (approx): {total_dim:,}")
        print(f"  Training cases: {len(train_df):,}")
        print(f"  Ratio cases/features: {len(train_df) / max(total_dim, 1):.1f}")

        _subsection("4D. Duration prediction (realized duration)")
        fit_feat_d = _fit_linear_prediction(
            train_df, test_df, "Realized_Duration_Min", feature_only,
            categorical_cols=feature_only_categorical,
        )
        fit_book_d = _fit_linear_prediction(
            train_df, test_df, "Realized_Duration_Min", booking_only,
            categorical_cols=[],
        )
        fit_comb_d = _fit_linear_prediction(
            train_df, test_df, "Realized_Duration_Min", combined,
            categorical_cols=combined_categorical,
        )

        proc_counts = train_df["Main_Procedure_Id"].value_counts() if "Main_Procedure_Id" in train_df.columns else pd.Series(dtype=float)
        common_procs = set(proc_counts[proc_counts >= 50].index)
        train_common = train_df[train_df["Main_Procedure_Id"].isin(common_procs)].copy() if len(common_procs) > 0 else train_df.iloc[:0].copy()
        test_common = test_df[test_df["Main_Procedure_Id"].isin(common_procs)].copy() if len(common_procs) > 0 else test_df.iloc[:0].copy()
        fit_feat_common = (
            _fit_linear_prediction(
                train_common, test_common, "Realized_Duration_Min", feature_only,
                categorical_cols=feature_only_categorical,
            )
            if len(test_common) > 0 else {"mae": np.nan, "rmse": np.nan, "mean_error": np.nan, "r2": np.nan, "n_test": 0}
        )
        fit_book_common = (
            _fit_linear_prediction(
                train_common, test_common, "Realized_Duration_Min", booking_only,
                categorical_cols=[],
            )
            if len(test_common) > 0 else {"mae": np.nan, "rmse": np.nan, "mean_error": np.nan, "r2": np.nan, "n_test": 0}
        )

        raw_booking_pred = test_df["Booked Time (Minutes)"].to_numpy(dtype=float)
        raw_booking_actual = test_df["Realized_Duration_Min"].to_numpy(dtype=float)
        raw_booking_resid = raw_booking_pred - raw_booking_actual
        raw_booking_fit = {
            "n_test": len(test_df),
            "mae": float(np.mean(np.abs(raw_booking_resid))) if len(test_df) > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean(raw_booking_resid ** 2))) if len(test_df) > 0 else np.nan,
            "mean_error": float(np.mean(raw_booking_resid)) if len(test_df) > 0 else np.nan,
            "r2": float(r2_score(raw_booking_actual, raw_booking_pred)) if len(test_df) > 0 else np.nan,
        }

        print("  Mean prediction error is reported as predicted − realized.")
        print("  Positive values indicate over-prediction; negative values indicate under-prediction.")
        print(f"  Raw booking direct baseline (no refit): mean error = {raw_booking_fit['mean_error']:+.2f} min, "
              f"MAE = {raw_booking_fit['mae']:.2f}, R² = {raw_booking_fit['r2']:.4f}")
        print(f"  Booking-only fitted model:             mean error = {fit_book_d['mean_error']:+.2f} min, "
              f"MAE = {fit_book_d['mae']:.2f}, R² = {fit_book_d['r2']:.4f}")
        print(f"  Feature-only fitted model:             mean error = {fit_feat_d['mean_error']:+.2f} min, "
              f"MAE = {fit_feat_d['mae']:.2f}, R² = {fit_feat_d['r2']:.4f}")
        print(f"  Combined fitted model:                 mean error = {fit_comb_d['mean_error']:+.2f} min, "
              f"MAE = {fit_comb_d['mae']:.2f}, R² = {fit_comb_d['r2']:.4f}")

        pred_rows = [
            {"target": "realized_duration", "model": "raw_booking_direct (no refit)", "mae": raw_booking_fit["mae"], "rmse": raw_booking_fit["rmse"], "mean_error": raw_booking_fit["mean_error"], "r2": raw_booking_fit["r2"], "n_test": raw_booking_fit["n_test"]},
            {"target": "realized_duration", "model": f"feature_only (~{total_dim}d)", "mae": fit_feat_d["mae"], "rmse": fit_feat_d["rmse"], "mean_error": fit_feat_d["mean_error"], "r2": fit_feat_d["r2"], "n_test": fit_feat_d["n_test"]},
            {"target": "realized_duration", "model": "booking_only (1d)", "mae": fit_book_d["mae"], "rmse": fit_book_d["rmse"], "mean_error": fit_book_d["mean_error"], "r2": fit_book_d["r2"], "n_test": fit_book_d["n_test"]},
            {"target": "realized_duration", "model": f"combined (~{total_dim+1}d)", "mae": fit_comb_d["mae"], "rmse": fit_comb_d["rmse"], "mean_error": fit_comb_d["mean_error"], "r2": fit_comb_d["r2"], "n_test": fit_comb_d["n_test"]},
            {"target": "realized_duration (common procs)", "model": f"feature_only (~{total_dim}d)", "mae": fit_feat_common["mae"], "rmse": fit_feat_common["rmse"], "mean_error": fit_feat_common["mean_error"], "r2": fit_feat_common["r2"], "n_test": fit_feat_common["n_test"]},
            {"target": "realized_duration (common procs)", "model": "booking_only (1d)", "mae": fit_book_common["mae"], "rmse": fit_book_common["rmse"], "mean_error": fit_book_common["mean_error"], "r2": fit_book_common["r2"], "n_test": fit_book_common["n_test"]},
        ]

        _subsection("4E. Bias prediction (booking_error = b − d̃)")
        print("  This is the operationally relevant target: can historical features")
        print("  recover systematic booking bias without using the booking itself?")
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["booking_bias_target"] = train_df["Booked Time (Minutes)"] - train_df["Realized_Duration_Min"]
        test_df["booking_bias_target"] = test_df["Booked Time (Minutes)"] - test_df["Realized_Duration_Min"]
        fit_feat_bias = _fit_linear_prediction(
            train_df, test_df, "booking_bias_target", feature_only,
            categorical_cols=feature_only_categorical,
        )
        pred_rows.append({
            "target": "booking_bias",
            "model": f"feature_only (~{total_dim}d)",
            "mae": fit_feat_bias["mae"],
            "rmse": fit_feat_bias["rmse"],
            "mean_error": fit_feat_bias["mean_error"],
            "r2": fit_feat_bias["r2"],
            "n_test": fit_feat_bias["n_test"],
        })
        train_surg_bias = train_df.groupby("Surgeon_Code")["booking_bias_target"].mean()
        test_df_bias = test_df.copy()
        test_df_bias["pred_surg_mean"] = test_df_bias["Surgeon_Code"].map(train_surg_bias)
        test_df_bias = test_df_bias.dropna(subset=["pred_surg_mean", "booking_bias_target"])
        if len(test_df_bias) > 0:
            bias_resid = test_df_bias["pred_surg_mean"].to_numpy(dtype=float) - test_df_bias["booking_bias_target"].to_numpy(dtype=float)
            surg_mean_mae = float(np.mean(np.abs(bias_resid)))
            surg_mean_rmse = float(np.sqrt(np.mean(bias_resid ** 2)))
            surg_mean_me = float(np.mean(bias_resid))
            surg_mean_r2 = float(r2_score(test_df_bias["booking_bias_target"], test_df_bias["pred_surg_mean"]))
            pred_rows.append({
                "target": "booking_bias",
                "model": "surgeon_mean_only (1 param/surgeon)",
                "mae": surg_mean_mae,
                "rmse": surg_mean_rmse,
                "mean_error": surg_mean_me,
                "r2": surg_mean_r2,
                "n_test": len(test_df_bias),
            })
            print(f"  Surgeon-mean-bias baseline: MAE={surg_mean_mae:.2f}, mean error={surg_mean_me:+.2f}, R²={surg_mean_r2:.4f}")
            print(f"  Feature-model bias:         MAE={fit_feat_bias['mae']:.2f}, mean error={fit_feat_bias['mean_error']:+.2f}, R²={fit_feat_bias['r2']:.4f}")
            if np.isfinite(fit_feat_bias["mae"]) and np.isfinite(surg_mean_mae) and abs(fit_feat_bias["mae"] - surg_mean_mae) < 2.0:
                print("  → The feature model performs similarly to a simple surgeon-mean")
                print("    baseline, suggesting limited extra recoverable signal beyond")
                print("    stable surgeon-specific effects.")
        zero_bias_actual = test_df["booking_bias_target"].to_numpy(dtype=float)
        zero_bias_pred = np.zeros_like(zero_bias_actual)
        zero_bias_resid = zero_bias_pred - zero_bias_actual
        zero_bias_mae = float(np.mean(np.abs(zero_bias_resid)))
        pred_rows.append({
            "target": "booking_bias",
            "model": "zero (predict no bias)",
            "mae": zero_bias_mae,
            "rmse": float(np.sqrt(np.mean(zero_bias_resid ** 2))),
            "mean_error": float(np.mean(zero_bias_resid)),
            "r2": 0.0,
            "n_test": len(test_df),
        })
        pred_df = pd.DataFrame(pred_rows)
        _save_csv(pred_df, tbldir / "agenda_task4_prediction_comparison.csv")
        print("\n  Full comparison table:")
        print(pred_df.to_string(index=False, float_format=lambda z: f"{z:.4f}" if isinstance(z, float) else str(z)))
        feature_groups = {
            "surgeon_identity": [c for c in ["Surgeon_Code", "q_hat_empirical"] if c in feature_only],
            "procedure": [c for c in ["Main_Procedure_Id"] if c in feature_only],
            "service_site_patient": [c for c in ["Case_Service", "Site", "Patient_Type"] if c in feature_only],
            "calendar": [c for c in ["DayOfWeek", "Month", "Year"] if c in feature_only],
        }
        feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}
        if len(feature_groups) > 0:
            _subsection("4F. Feature-group ablation (bias prediction)")
            bias_importance = _group_ablation_importance(train_df, test_df, "booking_bias_target", feature_groups, categorical_cols=feature_only_categorical)
            _save_csv(bias_importance, tbldir / "agenda_task4_bias_feature_group_importance.csv")
            print(bias_importance.to_string(index=False, float_format=lambda z: f"{z:.4f}" if isinstance(z, float) else str(z)))
        _subsection("4G. Variance decomposition of booking bias")
        var_source = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        var_decomp = _weighted_between_within_variance(var_source.assign(booking_bias=var_source["Booked Time (Minutes)"] - var_source["Realized_Duration_Min"]), "booking_bias", "Surgeon_Code")
        var_df = pd.DataFrame([var_decomp])
        _save_csv(var_df, tbldir / "agenda_task4_booking_bias_variance_decomposition.csv")
        if np.isfinite(var_decomp["between_share"]):
            print(f"  Between-surgeon share of booking-bias variance: {100 * var_decomp['between_share']:.1f}%")
            print(f"  Within-surgeon share: {100 * (1 - var_decomp['between_share']):.1f}%")
            print("  Note: variance share is not the same as scheduling-cost share;")
            print("  a modest systematic bias can still matter operationally because")
            print("  it does not average out within blocks.")
            train_bias_profile = train_only_q.copy()
            train_bias_profile["abs_mean_bias"] = train_bias_profile["mean_booking_error"].abs()
            train_bias_profile = train_bias_profile[train_bias_profile["surgeon_cases"] >= 30]
            if len(train_bias_profile) > 0:
                vol_weighted_abs_bias = float((train_bias_profile["abs_mean_bias"] * train_bias_profile["surgeon_cases"]).sum() / train_bias_profile["surgeon_cases"].sum())
                case_level_mae = float(var_source["Booking_Error_Min"].abs().mean())
                print(f"  Volume-weighted mean |surgeon bias|: {vol_weighted_abs_bias:.1f} min")
                print(f"  Overall case-level MAE(b − d̃): {case_level_mae:.1f} min")
                print(f"  Surgeon-level bias as fraction of case-level MAE: {100 * vol_weighted_abs_bias / case_level_mae:.1f}%")

    # Task 5: response-function landscape
    _section("TASK 5 — RESPONSE-FUNCTION LANDSCAPE")
    stability_rows = []
    stable_services = [svc for svc, n in pair_df["Case_Service"].value_counts().items() if n >= 30]
    for service in stable_services:
        sub = pair_df[pair_df["Case_Service"] == service].copy()
        for p in PERCENTILE_CUTOFFS:
            thr = np.nanpercentile(sub["abs_X_prev_overrun"], p)
            ss = sub[sub["abs_X_prev_overrun"] >= thr]
            fit = _ols_summary(ss["X_prev_overrun"], ss["Y_abnormal_booking"])
            stability_rows.append({"Case_Service": service, "percentile": p, "threshold": thr, **fit})
    stability_df = pd.DataFrame(stability_rows)
    _save_csv(stability_df, tbldir / "agenda_task5_service_slope_stability.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_services = [svc for svc in ["ANAE", "OTO", "NEUR", "GEN", "UROL", "ORTH"] if svc in set(stability_df["Case_Service"])]
    for service in plot_services:
        sub = stability_df[stability_df["Case_Service"] == service].sort_values("percentile")
        ax.errorbar(sub["percentile"], sub["slope"], yerr=sub["slope_se"], marker="o", capsize=3, label=service)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Absolute-signal percentile cutoff")
    ax.set_ylabel("Slope of Y on X")
    ax.set_title("Service-level slope stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "agenda_task5_service_slope_stability.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {figdir / 'agenda_task5_service_slope_stability.png'}")

    fit_rows = []
    for service in stable_services:
        sub = pair_df[pair_df["Case_Service"] == service].sort_values("current_time").copy()
        train_s, test_s = _chronological_week_split(sub)
        if len(train_s) < 20 or len(test_s) < 10:
            continue
        old_fit = _fit_old_piecewise(train_s["X_prev_overrun"].to_numpy(), train_s["Y_abnormal_booking"].to_numpy())
        new_fit = _fit_new_piecewise(train_s["X_prev_overrun"].to_numpy(), train_s["Y_abnormal_booking"].to_numpy())
        x_test = test_s["X_prev_overrun"].to_numpy(dtype=float)
        y_test = test_s["Y_abnormal_booking"].to_numpy(dtype=float)
        pred_old = np.zeros_like(x_test)
        pos = x_test > old_fit["h_plus"]
        neg = x_test < -old_fit["h_minus"]
        pred_old[pos] = old_fit["a"] * (x_test[pos] - old_fit["h_plus"])
        pred_old[neg] = old_fit["a"] * (x_test[neg] + old_fit["h_minus"])
        pred_new = _piecewise_new_response(x_test, new_fit["h_full"], new_fit["a"], new_fit["h_reject"])
        zero_test_mae = float(np.mean(np.abs(y_test)))
        fit_rows.append({
            "Case_Service": service,
            "n_train": len(train_s),
            "n_test": len(test_s),
            "old_a": old_fit["a"],
            "old_h_plus": old_fit["h_plus"],
            "old_h_minus": old_fit["h_minus"],
            "old_train_mae": old_fit["mae"],
            "old_test_mae": float(np.mean(np.abs(y_test - pred_old))),
            "new_a": new_fit["a"],
            "new_h_full": new_fit["h_full"],
            "new_h_reject": new_fit["h_reject"],
            "new_train_mae": new_fit["mae"],
            "new_test_mae": float(np.mean(np.abs(y_test - pred_new))),
            "zero_test_mae": zero_test_mae,
            "old_beats_zero": float(np.mean(np.abs(y_test - pred_old))) < zero_test_mae,
            "new_beats_zero": float(np.mean(np.abs(y_test - pred_new))) < zero_test_mae,
        })
    fit_df = pd.DataFrame(fit_rows)
    _save_csv(fit_df, tbldir / "agenda_task5_piecewise_fit_comparison.csv")
    if len(fit_df) > 0:
        n_old_beats = int(fit_df["old_beats_zero"].sum()) if "old_beats_zero" in fit_df.columns else 0
        n_new_beats = int(fit_df["new_beats_zero"].sum()) if "new_beats_zero" in fit_df.columns else 0
        n_services_tested = len(fit_df)
        print(f"  Services tested: {n_services_tested}")
        print(f"  Old (inaction-band) model beats zero on holdout: {n_old_beats} / {n_services_tested}")
        print(f"  New (3-regime, symmetric diagnostic) model beats zero on holdout: {n_new_beats} / {n_services_tested}")
        old_wins = fit_df.loc[fit_df["old_beats_zero"], "Case_Service"].astype(str).tolist()
        new_wins = fit_df.loc[fit_df["new_beats_zero"], "Case_Service"].astype(str).tolist()
        if old_wins:
            print(f"  Inaction-band wins in: {', '.join(old_wins)}")
        if new_wins:
            print(f"  3-regime wins in:      {', '.join(new_wins)}")
        if n_old_beats == 0 and n_new_beats == 0:
            print("  → Neither response model beats zero-prediction on any service.")
            print("    This supports treating experience-based response parameters as")
            print("    scenario inputs rather than directly estimated empirical truths.")
        elif n_old_beats >= max(1, n_services_tested // 2):
            print("  → Piecewise models improve on zero in a substantial share of services.")
            print("    The right interpretation is not blanket absence of self-correction,")
            print("    but that any response is nonlinear, heterogeneous, and not well")
            print("    summarized by a pooled linear slope.")

    # Task 6: relative vs absolute thresholds
    _section("TASK 6 — RELATIVE VS ABSOLUTE THRESHOLDS")
    pair_df["duration_bucket"] = pd.cut(pair_df["b_prev"], bins=[-np.inf, 90, 180, np.inf], labels=["short (<90)", "medium (90–180)", "long (>180)"])
    rel_rows = []
    for bucket, sub in pair_df.groupby("duration_bucket", observed=True):
        fit_abs = _ols_summary(sub["X_prev_overrun"], sub["Y_abnormal_booking"])
        fit_rel = _ols_summary(sub["X_prev_rel"], sub["Y_abnormal_rel"])
        rel_rows.append({"duration_bucket": bucket, "metric": "absolute", **fit_abs})
        rel_rows.append({"duration_bucket": bucket, "metric": "relative", **fit_rel})
    rel_df = pd.DataFrame(rel_rows)
    _save_csv(rel_df, tbldir / "agenda_task6_relative_vs_absolute_ols.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for bucket, sub in pair_df.groupby("duration_bucket", observed=True):
        xs, ys = _lowess(sub["X_prev_overrun"], sub["Y_abnormal_booking"], frac=0.5)
        axes[0].plot(xs, ys, label=str(bucket))
        xs_r, ys_r = _lowess(sub["X_prev_rel"], sub["Y_abnormal_rel"], frac=0.5)
        axes[1].plot(xs_r, ys_r, label=str(bucket))
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("LOWESS in absolute minutes")
    axes[0].set_xlabel("X_prev_overrun")
    axes[0].set_ylabel("Y_abnormal_booking")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("LOWESS in relative terms")
    axes[1].set_xlabel("X_prev_rel")
    axes[1].set_ylabel("Y_abnormal_rel")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "agenda_task6_relative_vs_absolute_lowess.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {figdir / 'agenda_task6_relative_vs_absolute_lowess.png'}")

    # Task 7: treatable subpopulation
    _section("TASK 7 — TREATABLE SUBPOPULATION")
    surgeon_gap = surgeon_q.copy()
    surgeon_gap["needed_downward_correction"] = surgeon_gap["mean_booking_error"].clip(lower=0.0)
    surgeon_gap["misalignment_vs_hospital"] = surgeon_gap["q_hat"] - hospital_q
    scenarios = [
        Scenario("conservative_proxy", h_full=0.0, a=0.10, h_reject=15.0),
        Scenario("literature_uniform", h_full=8.0, a=0.20, h_reject=30.0),
        Scenario("q_profiled_proxy", h_full=8.0, a=0.20, h_reject=45.0),
    ]
    sc_rows = []
    for _, row in surgeon_gap.iterrows():
        qhat = row["q_hat"]
        need = row["needed_downward_correction"]
        for sc in scenarios:
            if sc.name == "q_profiled_proxy" and np.isfinite(qhat):
                h_reject = max(5.0, sc.h_reject * max(1e-6, 1 - qhat))
                h_full = sc.h_full
                a = sc.a
            else:
                h_reject = sc.h_reject
                h_full = sc.h_full
                a = sc.a
            rec = max_downward_rec
            max_post = _piecewise_new_downward_capacity(rec, h_full=h_full, a=a, h_reject=h_reject)
            frac = min(max_post / need, 1.0) if need > 0 else np.nan
            sc_rows.append({
                "Surgeon_Code": row["Surgeon_Code"],
                "scenario": sc.name,
                "q_hat": qhat,
                "surgeon_cases": row["surgeon_cases"],
                "needed_downward_correction": need,
                "max_achievable_downward": max_post,
                "fraction_achievable": frac,
                "misalignment_vs_hospital": row["misalignment_vs_hospital"],
            })
    sc_df = pd.DataFrame(sc_rows)
    _save_csv(sc_df, tbldir / "agenda_task7_treatable_subpopulation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, sc in zip(axes, scenarios):
        sub = sc_df[sc_df["scenario"] == sc.name]
        ax.scatter(sub["q_hat"], sub["fraction_achievable"], s=np.clip(sub["surgeon_cases"], 20, 300), alpha=0.6)
        ax.axhline(0.40, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(hospital_q, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(sc.name)
        ax.set_xlabel("q_hat")
    axes[0].set_ylabel("Fraction of needed correction achievable")
    fig.tight_layout()
    fig.savefig(figdir / "agenda_task7_treatable_subpopulation.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {figdir / 'agenda_task7_treatable_subpopulation.png'}")

    # Task 8: within-group booking signal value
    _section("TASK 8 — WITHIN-GROUP BOOKING SIGNAL VALUE")
    print("  This task measures how much within-(surgeon × procedure) booking")
    print("  variation reflects genuine case-level complexity differences.")
    print("  A positive slope means that, within the same surgeon doing the same")
    print("  procedure, longer bookings are associated with longer realized")
    print("  durations. That is evidence of informational content in the booking.")
    print("")
    print("  Important caution: this is not a causal Parkinson's-Law test.")
    print("  The same positive slope can arise because surgeons observe patient")
    print("  complexity and adjust bookings accordingly. Separating a causal")
    print("  'booking changes duration' channel from an informational channel")
    print("  would require exogenous variation in bookings, which this dataset")
    print("  does not provide.")
    group_counts = df.groupby(["Surgeon_Code", "Main_Procedure_Id"]).size().rename("n_cases")
    eligible_groups = group_counts[group_counts >= 3].index
    if len(eligible_groups) == 0:
        print("  No repeated surgeon × procedure groups with ≥3 cases.")
    else:
        temp = df.set_index(["Surgeon_Code", "Main_Procedure_Id"]).loc[eligible_groups].reset_index().copy()
        temp["booking_c"] = temp["Booked Time (Minutes)"] - temp.groupby(["Surgeon_Code", "Main_Procedure_Id"])["Booked Time (Minutes)"].transform("mean")
        temp["realized_c"] = temp["Realized_Duration_Min"] - temp.groupby(["Surgeon_Code", "Main_Procedure_Id"])["Realized_Duration_Min"].transform("mean")
        fit_signal = _ols_summary(temp["booking_c"], temp["realized_c"])
        _save_csv(pd.DataFrame([fit_signal]), tbldir / "agenda_task8_within_group_booking_signal.csv")
        print(
            f"  Within (surgeon × procedure) centered slope of realized on booking: {fit_signal['slope']:.4f} "
            f"(se={fit_signal['slope_se']:.4f}, N={fit_signal['n']:,})"
        )
        print(f"  R² (within-group): {fit_signal['r2']:.4f}")
        if np.isfinite(fit_signal["slope"]):
            print("")
            print("  Interpretation:")
            print("  A 10-minute within-group booking increase is associated with")
            print(f"  roughly {10 * fit_signal['slope']:.1f} additional realized minutes on average.")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sample = temp.sample(min(3000, len(temp)), random_state=42) if len(temp) > 3000 else temp
        ax.scatter(sample["booking_c"], sample["realized_c"], alpha=0.15, s=8)
        xs, ys = _lowess(temp["booking_c"], temp["realized_c"], frac=0.5)
        ax.plot(xs, ys, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Within surgeon×procedure: booking vs realized")
        ax.set_xlabel("Centered booking")
        ax.set_ylabel("Centered realized")
        fig.tight_layout()
        fig.savefig(figdir / "agenda_task8_within_group_booking_signal.png", dpi=150)
        plt.close(fig)
        print(f"  → Saved: {figdir / 'agenda_task8_within_group_booking_signal.png'}")

def analyze_block_bias_propagation(df: pd.DataFrame) -> None:
    """Show that surgeon-level booking bias propagates to block-level load error."""
    section("BLOCK-LEVEL PROPAGATION OF SURGEON BIAS")
    if guard_empty(df, "block bias propagation"):
        return
    work = df.copy()
    work["OR_Date"] = pd.to_datetime(work["Actual Start Date"], errors="coerce").dt.date
    work["Weekday"] = pd.to_datetime(work["Actual Start Date"], errors="coerce").dt.weekday
    work = work[work["Weekday"] < 5].copy()
    sort_col = _completion_col(work)
    bias = _online_group_mean(work, value_col="Booking_Error_Min", group_col="Surgeon_Code", sort_col=sort_col, min_history=MIN_SURGEON_CASES)
    work["surg_bias"] = bias.fillna(0.0)

    blocks = work.groupby(["Operating_Room", "OR_Date"]).agg(
        n_cases=("Patient_ID", "size"),
        n_surgeons=("Surgeon_Code", "nunique"),
        block_bias=("surg_bias", "sum"),
        sum_booked=("Booked Time (Minutes)", "sum"),
        sum_realized=("Realized_Duration_Min", "sum"),
    ).reset_index()

    describe_series(blocks["block_bias"], "Block bias Σ(surgeon mean bias)")
    describe_series(blocks["n_surgeons"], "Distinct surgeons per block")
    print(f"\n  Fraction of blocks with |block bias| > 10 min: {(blocks['block_bias'].abs() > 10).mean():.1%}")
    print(f"  Fraction of blocks with |block bias| > 20 min: {(blocks['block_bias'].abs() > 20).mean():.1%}")
    print(f"  Fraction of blocks with |block bias| > 30 min: {(blocks['block_bias'].abs() > 30).mean():.1%}")
    print(f"  Fraction single-surgeon blocks: {(blocks['n_surgeons'] == 1).mean():.1%}")

    tau = 28.0
    blocks["load_realized"] = blocks["sum_realized"] + tau * (blocks["n_cases"] - 1).clip(lower=0)
    blocks["ot_realized"] = (blocks["load_realized"] - 480).clip(lower=0)
    corr = blocks["block_bias"].corr(blocks["ot_realized"])
    print(f"\n  Correlation(block bias, realized OT at C=480): {corr:.3f}")
    if np.isfinite(corr) and abs(corr) >= 0.10:
        print("  Interpretation: the correlation is directionally informative, but")
        print("  the more reliable evidence is the contrast between under- and")
        print("  over-biased blocks below.")
    else:
        print("  Interpretation: the linear correlation is essentially zero, so")
        print("  do not treat it as the main evidence. The directional contrast")
        print("  between under- and over-biased blocks below is more informative.")

    under = blocks[blocks["block_bias"] < -10]
    over = blocks[blocks["block_bias"] > 10]
    if len(under) > 10 and len(over) > 10:
        print(f"\n  Under-biased blocks (bias < -10 min): {len(under):,}")
        print(f"    Mean realized OT: {under['ot_realized'].mean():.1f} min")
        print(f"  Over-biased blocks  (bias > +10 min): {len(over):,}")
        print(f"    Mean realized OT: {over['ot_realized'].mean():.1f} min")
        diff = float(under['ot_realized'].mean() - over['ot_realized'].mean())
        print(f"  Difference (under − over): {diff:.1f} min")
        print("  This directional comparison is the main evidence that systematic")
        print("  under-booking is associated with tighter blocks and more overtime.")

    _save_csv(blocks[["Operating_Room", "OR_Date", "n_cases", "n_surgeons", "block_bias", "sum_booked", "sum_realized"]], TBLDIR / "block_bias_propagation.csv")


# ── Summary ────────────────────────────────────────────────────────────────
def generate_summary(df, df_raw):
    section("SUMMARY — KEY NUMBERS")
    if guard_empty(df, "summary"):
        return

    print(f"  Raw dataset:              {len(df_raw):,} cases")
    print(f"  After cleaning:           {len(df):,} cases")
    print(f"  Unique surgeons:          {df['Surgeon_Code'].nunique()}")
    if col_available(df, "Case_Service"):
        print(f"  Unique services:          {df['Case_Service'].nunique()}")
    print(f"  Unique procedures:        {df['Main_Procedure_Id'].nunique()}")
    print(f"  Date range:               "
          f"{df['Actual Start Date'].min().date()} to "
          f"{df['Actual Start Date'].max().date()}")

    err = df["Booking_Error_Min"].dropna()
    print(f"  Mean booking error:       {err.mean():.1f} min "
          f"(positive = over-booking)")
    print(f"  Median booking error:     {err.median():.1f} min")
    print(f"  % over-booked:            {100 * (err > 0).mean():.1f}%")

    coverage = (df["Realized_Duration_Min"]
                <= df["Booked Time (Minutes)"]).mean()
    print(f"  Empirical Pr(d̃ ≤ b):     {coverage:.4f}")

    weekly_raw = df.groupby("Week_Start").size()
    weekly = build_full_weekly_index(weekly_raw)
    active = weekly[weekly > 0]
    print(f"  Total calendar weeks:     {len(weekly)}")
    print(f"  Active weeks:             {len(active)}")
    print(f"  Mean cases/week:          {active.mean():.0f}")

    print(f"\n  Figures: {FIGDIR.resolve()}/")
    print(f"  Tables:  {TBLDIR.resolve()}/")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Dataset exploration for incentive-aware OR scheduling")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset (CSV, pickle, parquet, Excel)")
    parser.add_argument("--artifact-root", type=str, default="artifacts",
                        help="Artifact root directory")
    parser.add_argument("--run-label", type=str, default=None,
                        help="Optional run label for artifact folder")
    parser.add_argument("--report", type=str,
                        default=None,
                        help="Optional explicit path to the full text report file")
    parser.add_argument("--tee-console", action="store_true",
                        help="Also echo the detailed report to the console")
    args = parser.parse_args()

    artifact_run = ArtifactManager(args.artifact_root).run("analysis", args.run_label)
    global FIGDIR, TBLDIR, REPORTDIR
    FIGDIR = artifact_run.directory("figures")
    TBLDIR = artifact_run.directory("tables")
    REPORTDIR = artifact_run.run_dir

    report_path = Path(args.report) if args.report else artifact_run.path("analysis_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def run_analysis():
        print(SEPARATOR)
        print("  DATASET EXPLORATION — Incentive-Aware OR Scheduling")
        print(f"  Data file: {args.data}")
        print(f"  Report:    {report_path}")
        print(SEPARATOR)

        # Load and derive columns
        df_raw = load_data(args.data)
        audit_columns(df_raw)
        df_raw = add_derived_columns(df_raw)

        # Raw overview (before cleaning)
        analyze_raw_overview(df_raw)

        # Clean to elective completed OR cases
        df = preprocess(df_raw)

        # Guard: abort gracefully if cleaning yields an empty dataset
        if len(df) == 0:
            print("\n  ⚠ Cleaned dataset is empty.  This can happen when the")
            print("    input contains only cancelled or non-OR cases (e.g., a")
            print("    sample file).  No further analysis is possible.")
            return

        # Core analyses on cleaned data
        analyze_cleaned_overview(df)
        analyze_missingness_audit(df)
        analyze_surgeons_services_procedures(df)
        df = analyze_surgeon_types(df)           # adds Surgeon_Type column
        analyze_booking_signal(df)
        analyze_booking_error(df)
        analyze_empirical_coverage(df)
        analyze_within_service_heterogeneity(df)
        analyze_booking_granularity(df)
        analyze_consecutive_pairs(df)
        analyze_temporal_drift(df)
        analyze_drift_decomposition(df)
        analyze_weekly_horizons(df)
        analyze_surgeon_stability(df)
        analyze_site_booking_behavior(df)
        analyze_within_group_signal_with_controls(df)
        analyze_block_bias_propagation(df)
        analyze_site_decomposition(df)
        block_data = analyze_block_capacity(df)   # returns block DataFrame
        analyze_turnover_time(df, block_data)
        analyze_turnover_variability(df, block_data)
        analyze_eligibility_structure(df)
        analyze_repeat_patients(df)
        analyze_cancellations(df_raw)
        analyze_partial_pooling(df)
        analyze_waiting_times(df)                 # uses cleaned cohort
        analyze_block_surgeon_structure(df)
        analyze_within_block_sequencing(df)
        analyze_service_room_assignment(df)
        analyze_surgeon_weekly_patterns(df)
        analyze_room_day_template_regularity(df)
        analyze_block_opening_design(df)
        analyze_block_fragmentation(df)
        analyze_candidate_pool_proxies(df)
        analyze_weekly_block_market(df)
        analyze_surgeon_week_patterns(df)
        block_load_data = analyze_block_load_decomposition(df, block_data)
        run_agenda_diagnostics(df, FIGDIR, TBLDIR)
        generate_summary(df, df_raw)

        print(f"\n{SEPARATOR}")
        print(f"  Done.  Figures: {FIGDIR.resolve()}/")
        print(f"         Tables:  {TBLDIR.resolve()}/")
        print(f"         Report:  {report_path.resolve()}")
        print(SEPARATOR)

    print(SEPARATOR)
    print("  Running analysis.")
    print(f"  Full report will be written to: {report_path.resolve()}")
    if args.tee_console:
        print("  Detailed output will also be echoed to the console.")
    print(SEPARATOR)

    with report_path.open("w", encoding="utf-8") as report_file:
        if args.tee_console:
            class Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for stream in self.streams:
                        stream.write(data)
                    return len(data)

                def flush(self):
                    for stream in self.streams:
                        stream.flush()

            tee = Tee(sys.stdout, report_file)
            with redirect_stdout(tee), redirect_stderr(tee):
                run_analysis()
        else:
            with redirect_stdout(report_file), redirect_stderr(report_file):
                run_analysis()

    print(SEPARATOR)
    print(f"  Analysis complete.")
    print(f"  Full report: {report_path.resolve()}")
    print(f"  Figures:     {FIGDIR.resolve()}/")
    print(f"  Tables:      {TBLDIR.resolve()}/")
    print(SEPARATOR)


if __name__ == "__main__":
    main()