#!/usr/bin/env python3
"""Standalone two-plan gap runner for surgery_scheduling.

Runs the weekly scheduling experiment on two plans only:
    1) StatusQuo   (planned with booked durations)
    2) Oracle      (planned with realized durations; diagnostic lower bound)

Design goals
------------
- Reuse the current repo's weekly-instance builder, candidate-pool logic,
  eligibility maps, deterministic solver, and evaluator.
- Fix the cohort mismatch locally for this experiment by optionally excluding
  cancelled cases during loading, without changing the repository loader.
- Solve weekly problems in parallel with one Gurobi thread per worker
  (default: 14 workers x 1 thread each).
- Save rich artifacts robustly: config snapshots, load/scope summaries,
  per-week/per-plan metrics, assignment-level results, block-level load
  diagnostics, paired weekly deltas, aggregate summaries, and text report.

Expected usage
--------------
python scripts/three_scenario_gap_runner.py \
    --excel data/UHNOperating_RoomScheduling2011-2013.xlsx \
    --planning-sites TGH TWH \
    --workers 14 \
    --threads-per-worker 1 \
    --run-label gap_two_plan_full_dataset

Assumptions
-----------
- Run from the repository root, or with the project installed in the current
  environment so that `src.*` imports resolve.
- Gurobi is available.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.paths import ArtifactManager
from src.core.types import BlockId, Col, Domain, KPIResult, ScheduleResult, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
from src.data.scope import apply_experiment_scope
from src.planning.instance import build_weekly_instance
from src.solvers.deterministic import solve_deterministic

LOGGER = logging.getLogger("two_plan_gap_runner")

# -----------------------------------------------------------------------------
# Gurobi log capture (adapted from the user's parallel VFCG helper script)
# -----------------------------------------------------------------------------

_ORIG_MODEL_CTOR = gp.Model
_MODEL_COUNTER = itertools.count(1)
_LOG_DIR: Path | None = None
_LOG_LOCK = threading.Lock()


def _safe_log_stem(name: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "model")).strip("._")
    return stem[:100] or "model"


def install_gurobi_log_capture(log_dir: Path) -> None:
    """Monkeypatch gp.Model so each model writes its own LogFile."""
    global _LOG_DIR
    _LOG_DIR = log_dir
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _logging_model(*args, **kwargs):
        model = _ORIG_MODEL_CTOR(*args, **kwargs)
        model_name = kwargs.get("name", args[0] if args else "model")
        with _LOG_LOCK:
            log_path = _LOG_DIR / f"{next(_MODEL_COUNTER):05d}_{_safe_log_stem(model_name)}.log"
        model.Params.LogFile = str(log_path)
        return model

    gp.Model = _logging_model


# -----------------------------------------------------------------------------
# Loading for this standalone run
# -----------------------------------------------------------------------------

_EXCEL_COLUMNS_BASE = [
    "Patient_Type",
    "Case_Service",
    "Main_Procedure",
    "Main_Procedure_Id",
    "Operating_Room",
    "Site",
    "Booked Time (Minutes)",
    "Enter Room Date",
    "Enter Room Time",
    "Actual Start Date",
    "Actual Start Time",
    "Actual Stop Date",
    "Actual Stop Time",
    "Leave Room Date",
    "Leave Room Time",
    "Patient_ID",
    "Surgeon",
    "Surgeon_Code",
]

_EXCEL_COLUMNS_CANCELLATION = [
    "Case_Cancelled_Reason",
    "Case Cancel Date",
    "Case Cancel Time",
]


@dataclass
class LoadSummary:
    raw_rows: int
    after_or_room_filter: int
    cancelled_rows_excluded: int
    emergency_rows_excluded: int
    missing_actual_rows_excluded: int
    invalid_duration_rows_excluded: int
    nonpositive_booked_rows_excluded: int
    severe_timestamp_rows_excluded: int
    final_rows: int
    used_room_time_rows: int
    fell_back_to_surgical_rows: int
    mild_timestamp_violation_rows: int
    overhead_capped_rows: int
    missing_site_rows_after_imputation: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )
    return out


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
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
        result.loc[valid] = pd.to_datetime(
            dates.loc[valid].dt.strftime("%Y-%m-%d") + " " + times.loc[valid],
            errors="coerce",
        )
    return result


def _recode_rare(series: pd.Series, threshold: int) -> pd.Series:
    s = series.astype(object).copy()
    counts = s.value_counts(dropna=False)
    rare = counts[counts < threshold].index
    s[s.isin(rare)] = Domain.OTHER
    return s


def _canonicalize_id_value(x: object, default: str) -> str:
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


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[Col.ACTUAL_START], errors="coerce")
    iso = dt.dt.isocalendar()
    out[Col.WEEK_OF_YEAR] = iso.week.astype(int)
    out[Col.MONTH] = dt.dt.month.astype(int)
    out[Col.YEAR] = dt.dt.year.astype(int)
    return out


def load_data_for_gap_experiment(
    config: Config,
    *,
    exclude_cancelled: bool = True,
    positive_booked_only: bool = True,
) -> tuple[pd.DataFrame, LoadSummary]:
    """Load a cleaned cohort for the two-plan gap experiment.

    This intentionally mirrors the repo loader where practical, but adds
    optional exclusion of cancelled cases to align the optimization cohort
    with the analysis cohort.
    """
    path = config.data.excel_file_path
    LOGGER.info("Loading data for gap experiment from %s", path)

    usecols = list(dict.fromkeys(_EXCEL_COLUMNS_BASE + _EXCEL_COLUMNS_CANCELLATION))
    try:
        df_raw = pd.read_excel(path, usecols=usecols)
    except ValueError:
        # Fall back to full read if usecols fails because column spellings differ.
        df_raw = pd.read_excel(path)
    df = _normalize_columns(df_raw)
    raw_rows = len(df)

    # Common cohort rule: exact OR-room names after whitespace removal.
    room_compact = (
        df.get(Col.OPERATING_ROOM, pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    is_or = room_compact.str.fullmatch(r"OR\d+")
    df = df[is_or].copy()
    df[Col.OPERATING_ROOM] = room_compact.loc[df.index]
    after_or_room_filter = len(df)

    cancelled_rows_excluded = 0
    if exclude_cancelled:
        reason_flag = pd.Series(False, index=df.index)
        if "case_cancelled_reason" in df.columns:
            reason_text = df["case_cancelled_reason"].fillna("").astype(str).str.strip()
            reason_flag = (reason_text != "") & (reason_text.str.lower() != "nan")
        date_flag = pd.Series(False, index=df.index)
        if "case_cancel_date" in df.columns:
            date_flag = pd.to_datetime(df["case_cancel_date"], errors="coerce").notna()
        is_cancelled = reason_flag | date_flag
        cancelled_rows_excluded = int(is_cancelled.sum())
        df = df[~is_cancelled].copy()

    emergency_rows_excluded = 0
    if Col.PATIENT_TYPE in df.columns:
        emerg = (
            df[Col.PATIENT_TYPE]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .isin([Domain.EMERGENCY_PATIENT, "EMERGENCY"])
        )
        emergency_rows_excluded = int(emerg.sum())
        df = df[~emerg].copy()

    # Build timestamps.
    df[Col.ACTUAL_START] = _combine_date_time(df, Col.ACTUAL_START_DATE, Col.ACTUAL_START_TIME)
    df[Col.ACTUAL_STOP] = _combine_date_time(df, Col.ACTUAL_STOP_DATE, Col.ACTUAL_STOP_TIME)
    df[Col.ENTER_ROOM] = _combine_date_time(df, Col.ENTER_ROOM_DATE, Col.ENTER_ROOM_TIME)
    df[Col.LEAVE_ROOM] = _combine_date_time(df, Col.LEAVE_ROOM_DATE, Col.LEAVE_ROOM_TIME)

    # Durations.
    df[Col.SURGICAL_DURATION] = (df[Col.ACTUAL_STOP] - df[Col.ACTUAL_START]).dt.total_seconds() / 60.0
    df[Col.ROOM_DURATION] = (df[Col.LEAVE_ROOM] - df[Col.ENTER_ROOM]).dt.total_seconds() / 60.0

    has_all_four = (
        df[Col.ENTER_ROOM].notna()
        & df[Col.ACTUAL_START].notna()
        & df[Col.ACTUAL_STOP].notna()
        & df[Col.LEAVE_ROOM].notna()
    )
    missing_actual_rows_excluded = int((~has_all_four).sum())
    df = df[has_all_four].copy()
    order_ok = (
        (df[Col.ENTER_ROOM] <= df[Col.ACTUAL_START])
        & (df[Col.ACTUAL_START] <= df[Col.ACTUAL_STOP])
        & (df[Col.ACTUAL_STOP] <= df[Col.LEAVE_ROOM])
    )

    df[Col.TIMESTAMP_VIOLATION] = False
    df[Col.OVERHEAD_CAPPED] = False
    df[Col.USED_ROOM_TIME] = True
    df[Col.FELL_BACK_SURGICAL] = False
    df[Col.PROCEDURE_DURATION] = df[Col.ROOM_DURATION]
    df[Col.PREPARATION_DURATION] = (
        (df[Col.ACTUAL_START] - df[Col.ENTER_ROOM]).dt.total_seconds().clip(lower=0) / 60.0
    )

    # Keep finite positive booked, room, and surgical durations under the
    # common 480-minute planning limit.
    valid_duration = (
        (pd.to_numeric(df[Col.BOOKED_MINUTES], errors="coerce") > 0)
        & (df[Col.ROOM_DURATION] > 0)
        & (df[Col.SURGICAL_DURATION] > 0)
        & (pd.to_numeric(df[Col.BOOKED_MINUTES], errors="coerce") <= Domain.MAX_PLANNING_CASE_MINUTES)
        & (df[Col.ROOM_DURATION] <= Domain.MAX_PLANNING_CASE_MINUTES)
        & (df[Col.SURGICAL_DURATION] <= Domain.MAX_PLANNING_CASE_MINUTES)
    )
    invalid_duration_rows_excluded = int((~valid_duration).sum())
    df = df[valid_duration].copy()

    order_ok = order_ok.loc[df.index]
    severe_timestamp_rows_excluded = int((~order_ok).sum())
    df = df[order_ok].copy()

    nonpositive_booked_rows_excluded = 0

    # Site imputation exactly in the spirit of the repo loader.
    df[Col.SITE] = df.get(Col.SITE, "").fillna("").astype(str).str.strip().str.upper()
    room_site_nuniq = df[df[Col.SITE] != ""].groupby(Col.OPERATING_ROOM)[Col.SITE].nunique()
    unambiguous_rooms = set(room_site_nuniq[room_site_nuniq == 1].index)
    room_site_lookup = (
        df[(df[Col.OPERATING_ROOM].isin(unambiguous_rooms)) & (df[Col.SITE] != "")]
        .groupby(Col.OPERATING_ROOM)[Col.SITE]
        .first()
        .to_dict()
    )
    missing_site = df[Col.SITE] == ""
    df.loc[missing_site, Col.SITE] = (
        df.loc[missing_site, Col.OPERATING_ROOM].map(room_site_lookup).fillna("")
    )

    # Recode rare identifiers using the repo thresholds/config.
    df[Col.PROCEDURE_ID] = _recode_rare(df[Col.PROCEDURE_ID], config.data.min_samples_procedure)
    df[Col.SURGEON_CODE] = _recode_rare(df[Col.SURGEON_CODE], config.data.min_samples_surgeon)
    df[Col.CASE_SERVICE] = _recode_rare(df[Col.CASE_SERVICE], config.data.min_samples_service)
    df = _canonicalize_identifier_columns(df)

    df = _add_time_features(df)
    df = df.sort_values(Col.ACTUAL_START).reset_index(drop=True)
    df[Col.CASE_UID] = np.arange(len(df), dtype=int)

    summary = LoadSummary(
        raw_rows=raw_rows,
        after_or_room_filter=after_or_room_filter,
        cancelled_rows_excluded=cancelled_rows_excluded,
        emergency_rows_excluded=emergency_rows_excluded,
        missing_actual_rows_excluded=missing_actual_rows_excluded,
        invalid_duration_rows_excluded=invalid_duration_rows_excluded,
        nonpositive_booked_rows_excluded=nonpositive_booked_rows_excluded,
        severe_timestamp_rows_excluded=severe_timestamp_rows_excluded,
        final_rows=len(df),
        used_room_time_rows=int(df[Col.USED_ROOM_TIME].sum()),
        fell_back_to_surgical_rows=int(df[Col.FELL_BACK_SURGICAL].sum()),
        mild_timestamp_violation_rows=int(df[Col.TIMESTAMP_VIOLATION].sum()),
        overhead_capped_rows=int(df[Col.OVERHEAD_CAPPED].sum()),
        missing_site_rows_after_imputation=int((df[Col.SITE] == "").sum()),
    )
    LOGGER.info("Loaded %d cases for gap experiment.", len(df))
    return df, summary


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_with_durations(
    instance: WeeklyInstance,
    schedule: ScheduleResult,
    durations: np.ndarray,
    turnover: float,
    costs,
) -> tuple[KPIResult, list[dict[str, Any]]]:
    """Evaluate a fixed schedule under arbitrary durations.

    Mirrors the repo evaluator but replaces actual durations by the supplied
    vector so we can save both planned and realized metrics per scenario.
    """
    case_index = {case.case_id: i for i, case in enumerate(instance.cases)}
    block_load: dict[BlockId, float] = {bid: 0.0 for bid in schedule.opened_blocks}
    block_case_count: dict[BlockId, int] = {bid: 0 for bid in schedule.opened_blocks}

    scheduled_count = 0
    deferred_count = 0
    deferral_cost = 0.0

    for assignment in schedule.assignments:
        i = case_index.get(assignment.case_id)
        if i is None:
            continue
        if assignment.is_deferred:
            deferral_cost += costs.deferral_per_case
            deferred_count += 1
            continue
        bid = assignment.block_id
        if bid not in block_load:
            deferral_cost += costs.deferral_per_case
            deferred_count += 1
            continue
        block_load[bid] += float(durations[i])
        block_case_count[bid] += 1
        scheduled_count += 1

    total_turnover = 0.0
    block_rows = []
    total_overtime = 0.0
    total_idle = 0.0

    for bid in schedule.opened_blocks:
        k = int(block_case_count.get(bid, 0))
        turn = float(turnover) * max(k - 1, 0)
        load_no_turn = float(block_load.get(bid, 0.0))
        load = load_no_turn + turn
        total_turnover += turn

        cap = float(instance.calendar.capacity(bid))
        overtime = max(load - cap, 0.0)
        idle = max(cap - load, 0.0)
        total_overtime += overtime
        total_idle += idle

        block_rows.append(
            {
                "day_index": bid.day_index,
                "site": bid.site,
                "room": bid.room,
                "capacity_minutes": cap,
                "n_cases": k,
                "case_load_minutes": load_no_turn,
                "turnover_minutes": turn,
                "total_load_minutes": load,
                "overtime_minutes": overtime,
                "idle_minutes": idle,
                "activation_cost": float(instance.calendar.activation_cost(bid)),
            }
        )

    activation_cost = sum(instance.calendar.activation_cost(bid) for bid in schedule.opened_blocks)
    overtime_cost = costs.overtime_per_minute * total_overtime
    idle_cost = costs.idle_per_minute * total_idle
    total_cost = activation_cost + deferral_cost + overtime_cost + idle_cost

    return (
        KPIResult(
            total_cost=float(total_cost),
            activation_cost=float(activation_cost),
            overtime_cost=float(overtime_cost),
            idle_cost=float(idle_cost),
            deferral_cost=float(deferral_cost),
            overtime_minutes=float(total_overtime),
            idle_minutes=float(total_idle),
            turnover_minutes=float(total_turnover),
            scheduled_count=int(scheduled_count),
            deferred_count=int(deferred_count),
            blocks_opened=int(len(schedule.opened_blocks)),
        ),
        block_rows,
    )


def flatten_assignments(
    instance: WeeklyInstance,
    schedule: ScheduleResult,
    scenario: str,
) -> list[dict[str, Any]]:
    case_map = {c.case_id: c for c in instance.cases}
    rows = []
    for a in schedule.assignments:
        case = case_map.get(a.case_id)
        rows.append(
            {
                "week": int(instance.week_index),
                "scenario": scenario,
                "case_id": int(a.case_id),
                "surgeon_code": str(case.surgeon_code) if case else None,
                "service": str(case.service) if case else None,
                "procedure_id": str(case.procedure_id) if case else None,
                "booked_duration_min": float(case.booked_duration_min) if case else None,
                "actual_duration_min": float(case.actual_duration_min) if case else None,
                "is_deferred": bool(a.is_deferred),
                "day_index": None if a.is_deferred else int(a.day_index),
                "site": None if a.is_deferred else str(a.site),
                "room": None if a.is_deferred else str(a.room),
            }
        )
    return rows


def kpi_to_prefixed_dict(kpi: KPIResult, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_total_cost": float(kpi.total_cost),
        f"{prefix}_activation_cost": float(kpi.activation_cost),
        f"{prefix}_overtime_cost": float(kpi.overtime_cost),
        f"{prefix}_idle_cost": float(kpi.idle_cost),
        f"{prefix}_deferral_cost": float(kpi.deferral_cost),
        f"{prefix}_overtime_minutes": float(kpi.overtime_minutes),
        f"{prefix}_idle_minutes": float(kpi.idle_minutes),
        f"{prefix}_turnover_minutes": float(kpi.turnover_minutes),
        f"{prefix}_scheduled_count": int(kpi.scheduled_count),
        f"{prefix}_deferred_count": int(kpi.deferred_count),
        f"{prefix}_blocks_opened": int(kpi.blocks_opened),
    }


# -----------------------------------------------------------------------------
# Weekly task
# -----------------------------------------------------------------------------

def _full_monday_to_sunday_window(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    actual = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce").dropna()
    if actual.empty:
        return None, None

    earliest = actual.min().normalize()
    latest = actual.max().normalize()

    first_monday = earliest + pd.Timedelta(days=(7 - earliest.weekday()) % 7)
    last_sunday = latest - pd.Timedelta(days=(latest.weekday() + 1) % 7)
    if last_sunday < first_monday:
        return None, None
    return first_monday, last_sunday


def build_all_instances(config: Config, df: pd.DataFrame) -> tuple[list[WeeklyInstance], dict[str, Any]]:
    first_monday, last_sunday = _full_monday_to_sunday_window(df)
    if first_monday is None or last_sunday is None:
        return [], {
            "first_monday": None,
            "last_sunday": None,
            "dropped_before_first_monday": int(len(df)),
            "dropped_after_last_sunday": 0,
            "n_aligned_rows": 0,
            "n_scoped_rows": 0,
            "n_instances": 0,
        }

    actual_days = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce").dt.normalize()
    aligned_mask = (actual_days >= first_monday) & (actual_days <= last_sunday)
    dropped_before = int((actual_days < first_monday).sum())
    dropped_after = int((actual_days > last_sunday).sum())
    df_aligned = df[aligned_mask].copy()

    if config.scope.use_all_sites_for_warmup:
        df_for_elig = df_aligned
    else:
        df_for_elig, _ = apply_experiment_scope(df_aligned, config)
    elig_maps = build_eligibility_maps(df_for_elig, config)

    df_scoped, scope_summary = apply_experiment_scope(df_aligned, config)
    candidate_pools = build_candidate_pools(df_scoped, config)

    n_weeks = int(((last_sunday - first_monday).days + 1) // 7)
    instances: list[WeeklyInstance] = []
    current_start = first_monday
    for h in range(n_weeks):
        instance = build_weekly_instance(
            df_pool=df_scoped,
            horizon_start=current_start,
            week_index=h,
            config=config,
            candidate_pools=candidate_pools,
            eligibility_maps=elig_maps,
        )
        instances.append(instance)
        current_start += pd.Timedelta(days=config.scope.stride_days)

    meta = {
        "first_monday": str(first_monday.date()),
        "last_sunday": str(last_sunday.date()),
        "dropped_before_first_monday": int(dropped_before),
        "dropped_after_last_sunday": int(dropped_after),
        "n_aligned_rows": int(len(df_aligned)),
        "n_scoped_rows": int(len(df_scoped)),
        "n_calendar_weeks": int(n_weeks),
        "n_instances": int(len(instances)),
        "scope_summary": asdict(scope_summary),
    }
    return instances, meta


def solve_one_week(
    instance: WeeklyInstance,
    *,
    config: Config,
) -> dict[str, Any]:
    turnover = float(config.capacity.turnover_minutes)
    costs = config.costs
    n_forced_defer = int(sum(1 for i in range(instance.num_cases) if len(instance.case_eligible_blocks.get(i, [])) == 0))

    scenarios = {
        "StatusQuo": np.asarray(instance.booked_durations(), dtype=float),
        "Oracle": np.asarray(instance.actual_durations(), dtype=float),
    }

    week_meta = {
        "week": int(instance.week_index),
        "start_date": str(instance.start_date),
        "end_date": str(instance.end_date),
        "n_cases": int(instance.num_cases),
        "candidate_blocks": int(instance.calendar.total_candidates),
        "forced_defer_count": n_forced_defer,
        "n_sites_in_calendar": int(len({b.site for b in instance.calendar.candidates})),
        "n_rooms_in_calendar": int(len({(b.site, b.room) for b in instance.calendar.candidates})),
    }

    scenario_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for scenario, planning_durations in scenarios.items():
        model_name = f"{scenario}_week_{instance.week_index}"
        t0 = time.perf_counter()
        try:
            schedule = solve_deterministic(
                cases=instance.cases,
                durations=planning_durations,
                calendar=instance.calendar,
                costs=costs,
                solver_cfg=config.solver,
                case_eligible_blocks=instance.case_eligible_blocks,
                turnover=turnover,
                model_name=model_name,
            )
            solve_time = time.perf_counter() - t0

            planned_kpi, planned_blocks = evaluate_with_durations(
                instance=instance,
                schedule=schedule,
                durations=planning_durations,
                turnover=turnover,
                costs=costs,
            )
            realized_kpi, realized_blocks = evaluate_with_durations(
                instance=instance,
                schedule=schedule,
                durations=np.asarray(instance.actual_durations(), dtype=float),
                turnover=turnover,
                costs=costs,
            )

            row = {
                "week": int(instance.week_index),
                "start_date": str(instance.start_date),
                "end_date": str(instance.end_date),
                "scenario": scenario,
                "solver_status": str(schedule.solver_status),
                "solve_time_seconds": float(solve_time),
                "objective_value": None if schedule.objective_value is None else float(schedule.objective_value),
                "n_cases": int(instance.num_cases),
                "candidate_blocks": int(instance.calendar.total_candidates),
                "forced_defer_count": int(n_forced_defer),
                "opened_blocks_count": int(len(schedule.opened_blocks)),
                "booked_mean": float(np.mean(instance.booked_durations())) if instance.num_cases else math.nan,
                "actual_mean": float(np.mean(instance.actual_durations())) if instance.num_cases else math.nan,
                "planning_mean": float(np.mean(planning_durations)) if instance.num_cases else math.nan,
                "planning_minus_booked_mean": float(np.mean(planning_durations - np.asarray(instance.booked_durations(), dtype=float))) if instance.num_cases else math.nan,
                "planning_minus_actual_mean": float(np.mean(planning_durations - np.asarray(instance.actual_durations(), dtype=float))) if instance.num_cases else math.nan,
            }
            row.update(kpi_to_prefixed_dict(planned_kpi, "planned"))
            row.update(kpi_to_prefixed_dict(realized_kpi, "realized"))

            scenario_rows.append(row)

            assignment_rows.extend(flatten_assignments(instance, schedule, scenario=scenario))

            realized_lookup = {
                (r["day_index"], r["site"], r["room"]): r for r in realized_blocks
            }
            planned_lookup = {
                (r["day_index"], r["site"], r["room"]): r for r in planned_blocks
            }
            all_keys = sorted(set(planned_lookup) | set(realized_lookup))
            for key in all_keys:
                p = planned_lookup.get(key, {})
                r = realized_lookup.get(key, {})
                block_rows.append(
                    {
                        "week": int(instance.week_index),
                        "scenario": scenario,
                        "day_index": key[0],
                        "site": key[1],
                        "room": key[2],
                        "n_cases": p.get("n_cases", r.get("n_cases", 0)),
                        "capacity_minutes": p.get("capacity_minutes", r.get("capacity_minutes")),
                        "planned_case_load_minutes": p.get("case_load_minutes"),
                        "planned_turnover_minutes": p.get("turnover_minutes"),
                        "planned_total_load_minutes": p.get("total_load_minutes"),
                        "planned_overtime_minutes": p.get("overtime_minutes"),
                        "planned_idle_minutes": p.get("idle_minutes"),
                        "realized_case_load_minutes": r.get("case_load_minutes"),
                        "realized_turnover_minutes": r.get("turnover_minutes"),
                        "realized_total_load_minutes": r.get("total_load_minutes"),
                        "realized_overtime_minutes": r.get("overtime_minutes"),
                        "realized_idle_minutes": r.get("idle_minutes"),
                        "activation_cost": p.get("activation_cost", r.get("activation_cost")),
                    }
                )

        except Exception as exc:  # pragma: no cover - defensive on user machine
            solve_time = time.perf_counter() - t0
            failures.append(
                {
                    "week": int(instance.week_index),
                    "scenario": scenario,
                    "solve_time_seconds": float(solve_time),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            scenario_rows.append(
                {
                    "week": int(instance.week_index),
                    "start_date": str(instance.start_date),
                    "end_date": str(instance.end_date),
                    "scenario": scenario,
                    "solver_status": "FAILED",
                    "solve_time_seconds": float(solve_time),
                    "objective_value": math.nan,
                    "n_cases": int(instance.num_cases),
                    "candidate_blocks": int(instance.calendar.total_candidates),
                    "forced_defer_count": int(n_forced_defer),
                    "opened_blocks_count": math.nan,
                }
            )

    return {
        "week_meta": week_meta,
        "scenario_rows": scenario_rows,
        "assignment_rows": assignment_rows,
        "block_rows": block_rows,
        "failures": failures,
    }


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def summarize_scenarios(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    numeric_cols = [
        c for c in results_df.columns
        if c not in {"week", "scenario", "start_date", "end_date", "solver_status"}
        and pd.api.types.is_numeric_dtype(results_df[c])
    ]

    rows = []
    for scenario, sub in results_df.groupby("scenario", sort=True):
        row = {"scenario": scenario, "n_weeks": int(sub["week"].nunique())}
        for col in numeric_cols:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}__mean"] = float(s.mean()) if not s.empty else math.nan
            row[f"{col}__std"] = float(s.std(ddof=1)) if len(s) >= 2 else math.nan
            row[f"{col}__min"] = float(s.min()) if not s.empty else math.nan
            row[f"{col}__max"] = float(s.max()) if not s.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def build_paired_weekly(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = results_df.pivot(index="week", columns="scenario")
    expected = {"StatusQuo", "Oracle"}
    if not expected.issubset(set(results_df["scenario"].unique())):
        return pd.DataFrame(), pd.DataFrame()

    base_metrics = [
        "realized_total_cost",
        "realized_activation_cost",
        "realized_overtime_cost",
        "realized_idle_cost",
        "realized_deferral_cost",
        "realized_overtime_minutes",
        "realized_idle_minutes",
        "realized_turnover_minutes",
        "realized_scheduled_count",
        "realized_deferred_count",
        "realized_blocks_opened",
        "planned_total_cost",
        "planned_overtime_minutes",
        "planned_idle_minutes",
        "objective_value",
        "solve_time_seconds",
        "forced_defer_count",
        "n_cases",
        "candidate_blocks",
    ]

    weekly_rows = []
    for week in sorted(results_df["week"].unique()):
        row = {"week": int(week)}

        for metric in base_metrics:
            try:
                sq = float(pivot.loc[week, (metric, "StatusQuo")])
            except Exception:
                sq = math.nan
            try:
                o = float(pivot.loc[week, (metric, "Oracle")])
            except Exception:
                o = math.nan

            row[f"{metric}__StatusQuo"] = sq
            row[f"{metric}__Oracle"] = o
            row[f"{metric}__StatusQuo_minus_Oracle"] = sq - o if math.isfinite(sq) and math.isfinite(o) else math.nan

        row["status_quo_weekly_cost"] = row.get("realized_total_cost__StatusQuo", math.nan)
        row["oracle_weekly_cost"] = row.get("realized_total_cost__Oracle", math.nan)
        row["premium_cost"] = row.get("realized_total_cost__StatusQuo_minus_Oracle", math.nan)
        row["extra_overtime_minutes"] = row.get("realized_overtime_minutes__StatusQuo_minus_Oracle", math.nan)
        row["extra_idle_minutes"] = row.get("realized_idle_minutes__StatusQuo_minus_Oracle", math.nan)

        weekly_rows.append(row)

    weekly_df = pd.DataFrame(weekly_rows).sort_values("week").reset_index(drop=True)

    summary_rows = []
    for col in weekly_df.columns:
        if col == "week" or not pd.api.types.is_numeric_dtype(weekly_df[col]):
            continue
        s = pd.to_numeric(weekly_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        summary_rows.append(
            {
                "metric": col,
                "n_weeks": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) >= 2 else math.nan,
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "positive_weeks": int((s > 0).sum()),
                "positive_share": float((s > 0).mean()),
                "negative_weeks": int((s < 0).sum()),
                "nonzero_weeks": int((s != 0).sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)
    return weekly_df, summary_df


def render_text_report(
    *,
    args: argparse.Namespace,
    load_summary: LoadSummary,
    build_meta: dict[str, Any],
    scenario_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    lines = []
    lines.append("Two-plan full-dataset gap experiment")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Run configuration")
    lines.append("-" * 80)
    lines.append(f"Excel file: {args.excel}")
    lines.append(f"Planning sites: {', '.join(args.planning_sites)}")
    lines.append(f"Parallel workers: {args.workers}")
    lines.append(f"Threads per worker: {args.threads_per_worker}")
    lines.append("")

    lines.append("Load summary")
    lines.append("-" * 80)
    for k, v in load_summary.to_dict().items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Alignment / scope summary")
    lines.append("-" * 80)
    for k, v in build_meta.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    if not scenario_summary.empty:
        lines.append("Plan summary (means)")
        lines.append("-" * 80)
        view_cols = [
            "scenario",
            "n_weeks",
            "realized_total_cost__mean",
            "realized_overtime_minutes__mean",
            "realized_idle_minutes__mean",
            "realized_deferred_count__mean",
            "planned_total_cost__mean",
            "solve_time_seconds__mean",
        ]
        view_cols = [c for c in view_cols if c in scenario_summary.columns]
        lines.append(scenario_summary[view_cols].to_string(index=False))
        lines.append("")

    if not paired_summary.empty:
        lines.append("Paired premium summary")
        lines.append("-" * 80)
        key_metrics = {
            "status_quo_weekly_cost",
            "oracle_weekly_cost",
            "premium_cost",
            "extra_overtime_minutes",
            "extra_idle_minutes",
            "realized_total_cost__StatusQuo_minus_Oracle",
            "realized_overtime_minutes__StatusQuo_minus_Oracle",
            "realized_idle_minutes__StatusQuo_minus_Oracle",
        }
        view = paired_summary[paired_summary["metric"].isin(key_metrics)]
        if not view.empty:
            lines.append(view.to_string(index=False))
            lines.append("")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Incremental writers
# -----------------------------------------------------------------------------

class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write_many(self, records: list[dict[str, Any]]) -> None:
        for rec in records:
            self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _records_to_dataframe(records: list[dict[str, Any]], sort_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    available_sort_cols = [c for c in sort_cols if c in df.columns]
    if available_sort_cols:
        df = df.sort_values(available_sort_cols)
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone two-plan full-dataset gap runner.")
    parser.add_argument("--excel", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--run-label", default="gap_two_plan_full_dataset")
    parser.add_argument("--planning-sites", nargs="+", default=["TGH", "TWH"])
    parser.add_argument("--warmup-weeks", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-horizons", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--time-limit-seconds", type=int, default=600)
    parser.add_argument("--mip-gap", type=float, default=0.05)
    parser.add_argument("--block-capacity-minutes", type=float, default=480.0)
    parser.add_argument("--activation-cost-per-block", type=float, default=2000.0)
    parser.add_argument("--min-activation-rate", type=float, default=0.25)
    parser.add_argument("--turnover-minutes", type=float, default=30.0)
    parser.add_argument("--overtime-per-minute", type=float, default=15.0)
    parser.add_argument("--idle-per-minute", type=float, default=10.0)
    parser.add_argument("--deferral-per-case", type=float, default=2000.0)
    parser.add_argument("--weekday-scope", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--exclude-cancelled", action="store_true", default=True)
    parser.add_argument("--include-cancelled", action="store_true", default=False)
    parser.add_argument("--positive-booked-only", action="store_true", default=True)
    parser.add_argument("--disable-booked-positive-filter", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.data.excel_file_path = args.excel
    if args.warmup_weeks is not None:
        cfg.data.warmup_weeks = args.warmup_weeks
    if args.num_horizons is not None:
        cfg.data.num_horizons = args.num_horizons

    cfg.scope.planning_sites = tuple(args.planning_sites)
    cfg.scope.planning_weekdays = tuple(int(x) for x in args.weekday_scope)

    cfg.capacity.block_capacity_minutes = float(args.block_capacity_minutes)
    cfg.capacity.activation_cost_per_block = float(args.activation_cost_per_block)
    cfg.capacity.min_activation_rate = float(args.min_activation_rate)
    cfg.capacity.turnover_minutes = float(args.turnover_minutes)

    cfg.costs.overtime_per_minute = float(args.overtime_per_minute)
    cfg.costs.idle_per_minute = float(args.idle_per_minute)
    cfg.costs.deferral_per_case = float(args.deferral_per_case)

    cfg.solver.time_limit_seconds = int(args.time_limit_seconds)
    cfg.solver.mip_gap = float(args.mip_gap)
    cfg.solver.threads = int(args.threads_per_worker)
    cfg.solver.verbose = bool(args.verbose)
    return cfg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    if args.include_cancelled:
        exclude_cancelled = False
    else:
        exclude_cancelled = bool(args.exclude_cancelled)

    positive_booked_only = not bool(args.disable_booked_positive_filter)

    cfg = build_config(args)

    artifact_run = ArtifactManager(args.artifact_root).run("experiments", args.run_label)
    run_dir = artifact_run.ensure_run_dir()
    install_gurobi_log_capture(artifact_run.directory("gurobi_logs"))

    # Save args/config early.
    artifact_run.path("args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    artifact_run.path("config_snapshot.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    # Load and save load summary.
    df, load_summary = load_data_for_gap_experiment(
        cfg,
        exclude_cancelled=exclude_cancelled,
        positive_booked_only=positive_booked_only,
    )
    artifact_run.path("load_summary.json").write_text(
        json.dumps(load_summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    df.head(200).to_csv(artifact_run.path("cohort_head.csv"), index=False)

    # Build instances from the cleaned/scoped cohort.
    instances, build_meta = build_all_instances(cfg, df)
    artifact_run.path("build_meta.json").write_text(json.dumps(build_meta, indent=2, default=str), encoding="utf-8")
    week_df = pd.DataFrame(
        [
            {
                "week": int(inst.week_index),
                "start_date": str(inst.start_date),
                "end_date": str(inst.end_date),
                "n_cases": int(inst.num_cases),
                "candidate_blocks": int(inst.calendar.total_candidates),
                "forced_defer_count": int(sum(1 for i in range(inst.num_cases) if len(inst.case_eligible_blocks.get(i, [])) == 0)),
            }
            for inst in instances
        ]
    )
    week_df.to_csv(artifact_run.path("weeks.csv"), index=False)

    # Incremental writers for robust saving.
    scenario_writer = JsonlWriter(artifact_run.path("jsonl", "scenario_results.jsonl"))
    assignment_writer = JsonlWriter(artifact_run.path("jsonl", "assignment_results.jsonl"))
    block_writer = JsonlWriter(artifact_run.path("jsonl", "block_results.jsonl"))
    failure_writer = JsonlWriter(artifact_run.path("jsonl", "failures.jsonl"))
    week_meta_writer = JsonlWriter(artifact_run.path("jsonl", "week_meta.jsonl"))

    all_scenario_rows: list[dict[str, Any]] = []
    all_assignment_rows: list[dict[str, Any]] = []
    all_block_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_week_meta: list[dict[str, Any]] = []

    workers = max(1, min(int(args.workers), len(instances) if instances else 1))
    LOGGER.info("Running %d weekly instances with %d workers and %d Gurobi thread(s) per worker.", len(instances), workers, cfg.solver.threads)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(solve_one_week, inst, config=cfg): inst.week_index
            for inst in instances
        }
        for future in as_completed(futures):
            week_idx = futures[future]
            try:
                payload = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                err = {
                    "week": int(week_idx),
                    "scenario": "__WEEK_TASK__",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failure_writer.write_many([err])
                all_failures.append(err)
                continue

            scenario_writer.write_many(payload["scenario_rows"])
            assignment_writer.write_many(payload["assignment_rows"])
            block_writer.write_many(payload["block_rows"])
            failure_writer.write_many(payload["failures"])
            week_meta_writer.write_many([payload["week_meta"]])

            all_scenario_rows.extend(payload["scenario_rows"])
            all_assignment_rows.extend(payload["assignment_rows"])
            all_block_rows.extend(payload["block_rows"])
            all_failures.extend(payload["failures"])
            all_week_meta.append(payload["week_meta"])

    scenario_writer.close()
    assignment_writer.close()
    block_writer.close()
    failure_writer.close()
    week_meta_writer.close()

    # Final tabular artifacts.
    results_df = _records_to_dataframe(all_scenario_rows, ["week", "scenario"])
    assignments_df = _records_to_dataframe(all_assignment_rows, ["week", "scenario", "case_id"])
    blocks_df = _records_to_dataframe(all_block_rows, ["week", "scenario", "day_index", "site", "room"])
    failures_df = pd.DataFrame(all_failures)
    week_meta_df = _records_to_dataframe(all_week_meta, ["week"])

    results_df.to_csv(artifact_run.path("scenario_results.csv"), index=False)
    assignments_df.to_csv(artifact_run.path("assignment_results.csv"), index=False)
    blocks_df.to_csv(artifact_run.path("block_results.csv"), index=False)
    failures_df.to_csv(artifact_run.path("failures.csv"), index=False)
    week_meta_df.to_csv(artifact_run.path("week_meta.csv"), index=False)

    scenario_summary = summarize_scenarios(results_df)
    scenario_summary.to_csv(artifact_run.path("scenario_summary.csv"), index=False)

    paired_weekly, paired_summary = build_paired_weekly(results_df)
    paired_weekly.to_csv(artifact_run.path("paired_weekly_deltas.csv"), index=False)
    paired_summary.to_csv(artifact_run.path("paired_summary.csv"), index=False)

    report_text = render_text_report(
        args=args,
        load_summary=load_summary,
        build_meta=build_meta,
        scenario_summary=scenario_summary,
        paired_summary=paired_summary,
    )
    artifact_run.path("report.txt").write_text(report_text, encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(f"Scenario results: {artifact_run.path('scenario_results.csv')}")
    print(f"Scenario summary: {artifact_run.path('scenario_summary.csv')}")
    print(f"Paired weekly deltas: {artifact_run.path('paired_weekly_deltas.csv')}")
    print(f"Gurobi logs: {artifact_run.path('gurobi_logs')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
