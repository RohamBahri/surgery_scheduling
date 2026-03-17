"""
Dataset exploration for incentive-aware operating-room scheduling.

Reads the UHN surgical dataset, applies a cleaning pipeline for elective
completed OR cases, and produces descriptive statistics, tables, and figures
that inform every stage of the modeling framework:
  - Inverse optimization (booking-error asymmetry, empirical quantile coverage)
  - Response estimation  (consecutive same-surgeon-procedure pair counts)
  - Bilevel scheduling    (block structure, turnover, eligibility)

Usage:
    python data_analysis.py --data path/to/dataset.xlsx
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
FIGDIR = Path("figures")
TBLDIR = Path("tables")
FIGDIR.mkdir(exist_ok=True)
TBLDIR.mkdir(exist_ok=True)

# Minimum counts for reliable estimation
MIN_SURGEON_CASES  = 30
MIN_PROCEDURE_CASES = 20
MIN_SERVICE_CASES  = 50
MIN_TYPE_CASES     = 50       # paper's n_min for surgeon-type groups

# Consecutive-pair gap thresholds (days)
MAX_GAP_DAYS_LIST = [30, 60, 90]

# Warm-up for train/test split
WARMUP_WEEKS = 52

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

# ── 0. Raw overview ─────────────────────────────────────────────────────────
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


# ── 2. Missingness and temporal-provenance audit ────────────────────────────
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


# ── 4. Surgeon type construction ───────────────────────────────────────────
def analyze_surgeon_types(df):
    """Prototype the surgeon-type grouping used in the behavioral model.

    The paper constructs types as service × experience band, with rare groups
    merged until every type has at least n_min cases.  Since the dataset does
    not contain years-since-credentialing, we proxy experience with caseload
    volume tertile (low / medium / high) within each service.
    """
    section("SURGEON TYPE CONSTRUCTION (service × volume band)")
    if guard_empty(df, "surgeon types"):
        return
    if not col_available(df, "Case_Service"):
        print("  Case_Service not available — cannot build types.")
        return

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


# ── 7. Empirical quantile coverage ────────────────────────────────────────
def analyze_empirical_coverage(df):
    """Compute Pr(d̃ ≤ b) as a descriptive proxy for the critical ratio q.

    If a surgeon targets quantile q, then on average b should exceed d̃ with
    probability q.  This is a model-free bridge to the inverse step.
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
    ax.set_title("Empirical Coverage by Surgeon (proxy for critical ratio)")
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
        print("  the variation the behavioral model (inverse step) captures.")


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


# ── 13. Surgeon stability ─────────────────────────────────────────────────
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


# ── 15. Block capacity analysis ───────────────────────────────────────────
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

    subsection("Total realized duration per block (surgical time only)")
    describe_series(block["total_realized"], "Realized (min)")

    # Fixed-capacity: surgical time only (underestimates true load)
    subsection("Fixed-capacity analysis — surgical time only")
    print("  Note: True block load includes turnover between cases.")
    print("  See the turnover section for combined estimates.\n")
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


# ── 16. Turnover time ─────────────────────────────────────────────────────
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


# ── 17. Eligibility set structure ──────────────────────────────────────────
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


# ── 18. Repeat patient analysis ───────────────────────────────────────────
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


# ── 19. Cancellation patterns ─────────────────────────────────────────────
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


# ── 20. Partial pooling assessment ─────────────────────────────────────────
def analyze_partial_pooling(df):
    """Determine if partial pooling (paper eq. 15) is needed for
    surgeon-level critical ratio estimation."""
    section("PARTIAL POOLING ASSESSMENT (paper eq. 15)")
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

    subsection("Implication")
    n50 = (surg_vol < 50).sum()
    c50 = surg_vol[surg_vol < 50].sum()
    print(f"  With n_min=50: {n50} surgeons ({c50:,} cases) would rely "
          f"on type-level q̂_k.")
    print(f"  Partial pooling (paper eq. 15) is "
          f"{'recommended' if n50 > 5 else 'optional'}.")


# ── 21. Waiting time data ─────────────────────────────────────────────────
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
    args = parser.parse_args()

    print(SEPARATOR)
    print("  DATASET EXPLORATION — Incentive-Aware OR Scheduling")
    print(f"  Data file: {args.data}")
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
    analyze_weekly_horizons(df)
    analyze_surgeon_stability(df)
    analyze_site_booking_behavior(df)
    block_data = analyze_block_capacity(df)   # returns block DataFrame
    analyze_turnover_time(df, block_data)
    analyze_eligibility_structure(df)
    analyze_repeat_patients(df)
    analyze_cancellations(df_raw)
    analyze_partial_pooling(df)
    analyze_waiting_times(df)                 # uses cleaned cohort
    generate_summary(df, df_raw)

    print(f"\n{SEPARATOR}")
    print(f"  Done.  Figures: {FIGDIR.resolve()}/")
    print(f"         Tables:  {TBLDIR.resolve()}/")
    print(SEPARATOR)


if __name__ == "__main__":
    main()