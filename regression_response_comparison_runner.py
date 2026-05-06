
#!/usr/bin/env python3
"""Unified offline comparison runner for surgery-duration planning baselines.

Scenarios
---------
1) Oracle                  : plan on realized durations (diagnostic lower bound)
2) Booked                  : plan on surgeon-booked durations (status quo)
3) RegressionThenResponse  : fit an exact LP for raw correction, then pass it
                             through the response function before planning
4) ResponseAwareRegression : fit an exact response-aware MAE model with Gurobi
                             PWL constraints, then pass predictions through the
                             same response function before planning

Key guarantees
--------------
- Weekly scheduling/evaluation uses the repository's weekly-instance builder,
  candidate pools, eligibility maps, and deterministic solver.
- Weekly solves run in parallel across weeks.
- Site is NEVER used as a regression feature, per user instruction.
- Rare-category recoding is fit on warmup only and then applied to pool, to
  avoid train/test leakage.
- The response-aware fit is exact only when Gurobi returns OPTIMAL; by default
  this runner requires that and raises otherwise.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy.optimize import minimize
from scipy import sparse

from src.core.config import Config
from src.core.paths import ArtifactManager
from src.core.types import BlockId, CaseRecord, Col, Domain, KPIResult, ScheduleResult, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
from src.data.scope import apply_experiment_scope
from src.data.splits import split_warmup_pool
from src.planning.instance import build_weekly_instance
from src.solvers.deterministic import solve_deterministic


LOGGER = logging.getLogger("regression_response_comparison_runner")


# -----------------------------------------------------------------------------
# Gurobi log capture
# -----------------------------------------------------------------------------

_ORIG_MODEL_CTOR = gp.Model
_MODEL_COUNTER = itertools.count(1)
_LOG_LOCK = threading.Lock()


def _safe_log_stem(name: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "model")).strip("._")
    return stem[:100] or "model"


def install_gurobi_log_capture(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    def _logging_model(*args, **kwargs):
        model = _ORIG_MODEL_CTOR(*args, **kwargs)
        model_name = kwargs.get("name", args[0] if args else "model")
        with _LOG_LOCK:
            log_path = log_dir / f"{next(_MODEL_COUNTER):05d}_{_safe_log_stem(model_name)}.log"
        model.Params.LogFile = str(log_path)
        return model

    gp.Model = _logging_model

def _apply_response_fit_params(model: gp.Model, time_limit_seconds: int, verbose: bool) -> None:
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = int(time_limit_seconds)

    # Best fix for the failure you saw:
    # force simplex, do not let Gurobi pick barrier.
    model.Params.Method = 1          # dual simplex
    model.Params.NumericFocus = 3
    model.Params.ScaleFlag = 2
    model.Params.Aggregate = 0
    model.Params.Presolve = 1

    # These were not helping the failing stage.
    model.Params.Symmetry = 0

# -----------------------------------------------------------------------------
# Data loading and leakage-safe recoding
# -----------------------------------------------------------------------------

EXCEL_COLUMNS = [
    "Patient_Type",
    "Case_Service",
    "Main_Procedure",
    "Main_Procedure_Id",
    "Operating_Room",
    "Site",
    "Consult_Date",
    "Decision_Date",
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


@dataclass
class ScopeBundle:
    df_all: pd.DataFrame
    df_warmup: pd.DataFrame
    df_pool: pd.DataFrame
    df_warmup_scoped: pd.DataFrame
    df_pool_scoped: pd.DataFrame
    pool_start: pd.Timestamp
    warmup_scope_summary: Any
    pool_scope_summary: Any


@dataclass
class BuildMeta:
    pool_start: str
    n_total_rows: int
    n_warmup_rows: int
    n_pool_rows: int
    n_warmup_scoped_rows: int
    n_pool_scoped_rows: int
    n_instances: int
    n_candidate_pool_days: int
    scenarios_requested: List[str]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )
    return out


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    if date_col not in df.columns or time_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    times = (
        df[time_col].astype(str).str.strip()
        .replace({"nan": "", "NaT": "", "None": "", "<NA>": ""})
        .str.split(".").str[0]
    )
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    valid = dates.notna() & (times != "")
    if valid.any():
        result.loc[valid] = pd.to_datetime(
            dates.loc[valid].dt.strftime("%Y-%m-%d") + " " + times.loc[valid],
            errors="coerce",
        )
    return result


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


def _apply_rare_mapping(series: pd.Series, rare_values: set[str]) -> pd.Series:
    s = series.astype(object).copy()
    s[s.astype(str).isin(rare_values)] = Domain.OTHER
    return s


def load_clean_data_for_experiment(
    config: Config,
    *,
    exclude_cancelled: bool = True,
    positive_booked_only: bool = True,
) -> tuple[pd.DataFrame, LoadSummary]:
    path = config.data.excel_file_path
    LOGGER.info("Loading experiment data from %s", path)
    try:
        df_raw = pd.read_excel(path, usecols=EXCEL_COLUMNS)
    except ValueError:
        df_raw = pd.read_excel(path)
    df = _normalize_columns(df_raw)
    raw_rows = len(df)

    room_series = df.get(Col.OPERATING_ROOM, pd.Series("", index=df.index)).fillna("").astype(str)
    room_compact = room_series.str.replace(r"\s+", "", regex=True).str.upper()
    is_or = room_compact.str.match(r"^OR\d+$")
    is_emergency_room = room_compact.isin([r.upper() for r in Domain.EMERGENCY_ROOMS])
    df = df[is_or & ~is_emergency_room].copy()
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
            df[Col.PATIENT_TYPE].fillna("").astype(str).str.strip().str.upper()
            .isin([Domain.EMERGENCY_PATIENT, "EMERGENCY"])
        )
        emergency_rows_excluded = int(emerg.sum())
        df = df[~emerg].copy()

    df[Col.ACTUAL_START] = _combine_date_time(df, Col.ACTUAL_START_DATE, Col.ACTUAL_START_TIME)
    df[Col.ACTUAL_STOP] = _combine_date_time(df, Col.ACTUAL_STOP_DATE, Col.ACTUAL_STOP_TIME)
    df[Col.ENTER_ROOM] = _combine_date_time(df, Col.ENTER_ROOM_DATE, Col.ENTER_ROOM_TIME)
    df[Col.LEAVE_ROOM] = _combine_date_time(df, Col.LEAVE_ROOM_DATE, Col.LEAVE_ROOM_TIME)

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
    df.loc[missing_site, Col.SITE] = df.loc[missing_site, Col.OPERATING_ROOM].map(room_site_lookup).fillna("")

    # Canonicalize but do NOT recode rare categories here.
    if Col.SURGEON_CODE in df.columns:
        df[Col.SURGEON_CODE] = df[Col.SURGEON_CODE].map(lambda x: _canonicalize_id_value(x, Domain.OTHER))
    if Col.PROCEDURE_ID in df.columns:
        df[Col.PROCEDURE_ID] = df[Col.PROCEDURE_ID].map(lambda x: _canonicalize_id_value(x, Domain.OTHER))
    if Col.CASE_SERVICE in df.columns:
        df[Col.CASE_SERVICE] = df[Col.CASE_SERVICE].map(lambda x: _canonicalize_id_value(x, Domain.UNKNOWN))
    if Col.PATIENT_TYPE in df.columns:
        df[Col.PATIENT_TYPE] = df[Col.PATIENT_TYPE].fillna(Domain.UNKNOWN).astype(str)
    if Col.OPERATING_ROOM in df.columns:
        df[Col.OPERATING_ROOM] = df[Col.OPERATING_ROOM].fillna(Domain.UNKNOWN).astype(str)

    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
    iso = dt.dt.isocalendar()
    df[Col.WEEK_OF_YEAR] = iso.week.astype(int)
    df[Col.MONTH] = dt.dt.month.astype(int)
    df[Col.YEAR] = dt.dt.year.astype(int)
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
    return df, summary


def apply_leakage_safe_recoding(
    df_warmup: pd.DataFrame,
    df_pool: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    warm = df_warmup.copy()
    pool = df_pool.copy()

    proc_counts = warm[Col.PROCEDURE_ID].astype(str).value_counts(dropna=False)
    proc_rare = set(proc_counts[proc_counts < config.data.min_samples_procedure].index.astype(str))
    surg_counts = warm[Col.SURGEON_CODE].astype(str).value_counts(dropna=False)
    surg_rare = set(surg_counts[surg_counts < config.data.min_samples_surgeon].index.astype(str))
    svc_counts = warm[Col.CASE_SERVICE].astype(str).value_counts(dropna=False)
    svc_rare = set(svc_counts[svc_counts < config.data.min_samples_service].index.astype(str))

    for df in (warm, pool):
        df[Col.PROCEDURE_ID] = _apply_rare_mapping(df[Col.PROCEDURE_ID].astype(str), proc_rare)
        df[Col.SURGEON_CODE] = _apply_rare_mapping(df[Col.SURGEON_CODE].astype(str), surg_rare)
        df[Col.CASE_SERVICE] = _apply_rare_mapping(df[Col.CASE_SERVICE].astype(str), svc_rare)

    recode_info = {
        "procedure_rare_values": sorted(proc_rare),
        "surgeon_rare_values": sorted(surg_rare),
        "service_rare_values": sorted(svc_rare),
    }
    return warm, pool, recode_info


def build_scope_bundle(df_all: pd.DataFrame, config: Config) -> tuple[ScopeBundle, dict[str, list[str]]]:
    df_warmup_raw, df_pool_raw, pool_start = split_warmup_pool(df_all, config)
    df_warmup, df_pool, recode_info = apply_leakage_safe_recoding(df_warmup_raw, df_pool_raw, config)

    df_warmup_scoped, warmup_scope_summary = apply_experiment_scope(df_warmup, config)
    df_pool_scoped, pool_scope_summary = apply_experiment_scope(df_pool, config)

    bundle = ScopeBundle(
        df_all=df_all,
        df_warmup=df_warmup,
        df_pool=df_pool,
        df_warmup_scoped=df_warmup_scoped,
        df_pool_scoped=df_pool_scoped,
        pool_start=pool_start,
        warmup_scope_summary=warmup_scope_summary,
        pool_scope_summary=pool_scope_summary,
    )
    return bundle, recode_info


# -----------------------------------------------------------------------------
# Response function
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseParams:
    a_s: float
    h_c: float
    rho_s: float

    @property
    def T_rej(self) -> float:
        if self.rho_s <= self.a_s:
            return float("inf")
        return self.rho_s * self.h_c / (self.rho_s - self.a_s)

    @property
    def peak_edit(self) -> float:
        return self.a_s * self.h_c


class ThreeRegimeResponse:
    def __init__(self, default_params: ResponseParams) -> None:
        if default_params.rho_s <= default_params.a_s:
            raise ValueError("response_rho must be strictly greater than response_a.")
        self.default_params = default_params
        self._surgeon_params: dict[str, ResponseParams] = {}

    def get_params(self, surgeon_code: str) -> ResponseParams:
        return self._surgeon_params.get(str(surgeon_code), self.default_params)

    def apply_scalar(self, u: float, surgeon_code: str | None = None) -> float:
        p = self.get_params(surgeon_code or "")
        abs_u = abs(float(u))
        sign_u = 1.0 if u >= 0 else -1.0
        if abs_u <= p.h_c:
            return p.a_s * float(u)
        if abs_u <= p.T_rej:
            return (p.a_s - p.rho_s) * float(u) + p.rho_s * p.h_c * sign_u
        return 0.0

    def apply_vector(self, u: np.ndarray, surgeon_codes: Sequence[str]) -> np.ndarray:
        out = np.zeros_like(u, dtype=float)
        for i, sc in enumerate(surgeon_codes):
            out[i] = self.apply_scalar(float(u[i]), sc)
        return out

    def slope_vector(self, u: np.ndarray, surgeon_codes: Sequence[str]) -> np.ndarray:
        out = np.zeros_like(u, dtype=float)
        for i, sc in enumerate(surgeon_codes):
            p = self.get_params(sc)
            abs_u = abs(float(u[i]))
            if abs_u < p.h_c:
                out[i] = p.a_s
            elif abs_u < p.T_rej:
                out[i] = p.a_s - p.rho_s
            else:
                out[i] = 0.0
        return out

    def classify_regimes(self, u: np.ndarray, surgeon_codes: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(u), dtype=int)
        for i, sc in enumerate(surgeon_codes):
            p = self.get_params(sc)
            abs_u = abs(float(u[i]))
            if abs_u <= p.h_c:
                out[i] = 0
            elif abs_u <= p.T_rej:
                out[i] = 1
            else:
                out[i] = 2
        return out

    @staticmethod
    def regime_name(code: int) -> str:
        return {0: "accept", 1: "decay", 2: "discard"}.get(int(code), "unknown")


def _response_graph_points(U: float, p: ResponseParams) -> tuple[list[float], list[float]]:
    if U <= 1e-8:
        return [-1e-6, 0.0, 1e-6], [0.0, 0.0, 0.0]

    if not np.isfinite(U) or U <= 0:
        raise ValueError(f"Invalid U bound in _response_graph_points: {U}")

    def eval_response(u: float) -> float:
        abs_u = abs(float(u))
        sign_u = 1.0 if u >= 0 else -1.0
        if abs_u <= p.h_c:
            return p.a_s * float(u)
        if abs_u <= p.T_rej:
            return (p.a_s - p.rho_s) * float(u) + p.rho_s * p.h_c * sign_u
        return 0.0

    h = min(float(p.h_c), float(U))
    T = float(p.T_rej)
    if not np.isfinite(T):
        T = float(U)
    T = min(T, float(U))

    x_candidates = [-float(U), -T, -h, 0.0, h, T, float(U)]
    y_candidates = [eval_response(x) for x in x_candidates]

    xpts: list[float] = []
    ypts: list[float] = []
    for x, y in zip(x_candidates, y_candidates):
        if xpts and abs(x - xpts[-1]) <= 1e-10:
            ypts[-1] = float(y)
        else:
            xpts.append(float(x))
            ypts.append(float(y))

    if len(xpts) < 2:
        xpts = [-float(U), float(U)]
        ypts = [eval_response(-float(U)), eval_response(float(U))]

    return xpts, ypts


# -----------------------------------------------------------------------------
# Feature builder
# -----------------------------------------------------------------------------

@dataclass
class FeatureManifest:
    numeric_features: list[str]
    numeric_missing_indicators: list[str]
    categorical_features: dict[str, list[str]]
    total_features: int
    notes: list[str]


@dataclass
class GroupStatsDefaults:
    mean_error: float
    median_error: float
    qhat: float
    mean_booked: float
    mean_realized: float
    log_cases: float


def _month_sin(month: float) -> float:
    return math.sin(2.0 * math.pi * float(month) / 12.0)


def _month_cos(month: float) -> float:
    return math.cos(2.0 * math.pi * float(month) / 12.0)


def _week_sin(week: float) -> float:
    return math.sin(2.0 * math.pi * float(week) / 53.0)


def _week_cos(week: float) -> float:
    return math.cos(2.0 * math.pi * float(week) / 53.0)


class FeatureBuilder:
    """Build safe booking-time features. Site is excluded by design."""

    def __init__(self, min_group_cases: int = 5) -> None:
        self.min_group_cases = int(min_group_cases)
        self.numeric_features: list[str] = []
        self.numeric_missing_indicators: list[str] = []
        self.categorical_vocab: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.feature_manifest: FeatureManifest | None = None
        self._fitted = False

        self._surgeon_stats: dict[str, dict[str, float]] = {}
        self._service_stats: dict[str, dict[str, float]] = {}
        self._procedure_stats: dict[str, dict[str, float]] = {}
        self._room_stats: dict[str, dict[str, float]] = {}
        self._defaults: GroupStatsDefaults | None = None

    def fit(self, df_warmup_scoped: pd.DataFrame) -> "FeatureBuilder":
        work = df_warmup_scoped.copy()
        booked = pd.to_numeric(work[Col.BOOKED_MINUTES], errors="coerce").fillna(0.0)
        realized = pd.to_numeric(work[Col.PROCEDURE_DURATION], errors="coerce").fillna(0.0)
        err = booked - realized
        covered = (realized <= booked).astype(float)

        work["_booked"] = booked
        work["_realized"] = realized
        work["_error"] = err
        work["_covered"] = covered

        def _build_stats(group_col: str) -> dict[str, dict[str, float]]:
            grouped = (
                work.groupby(group_col)
                .agg(
                    n_cases=("_error", "size"),
                    mean_error=("_error", "mean"),
                    median_error=("_error", "median"),
                    qhat=("_covered", "mean"),
                    mean_booked=("_booked", "mean"),
                    mean_realized=("_realized", "mean"),
                )
                .reset_index()
            )
            grouped = grouped[grouped["n_cases"] >= self.min_group_cases]
            out: dict[str, dict[str, float]] = {}
            for _, row in grouped.iterrows():
                out[str(row[group_col])] = {
                    "mean_error": float(row["mean_error"]),
                    "median_error": float(row["median_error"]),
                    "qhat": float(row["qhat"]),
                    "mean_booked": float(row["mean_booked"]),
                    "mean_realized": float(row["mean_realized"]),
                    "log_cases": float(math.log1p(float(row["n_cases"]))),
                }
            return out

        self._surgeon_stats = _build_stats(Col.SURGEON_CODE)
        self._service_stats = _build_stats(Col.CASE_SERVICE)
        self._procedure_stats = _build_stats(Col.PROCEDURE_ID)
        self._room_stats = _build_stats(Col.OPERATING_ROOM)
        self._defaults = GroupStatsDefaults(
            mean_error=float(err.mean()),
            median_error=float(err.median()),
            qhat=float(covered.mean()),
            mean_booked=float(booked.mean()),
            mean_realized=float(realized.mean()),
            log_cases=float(math.log1p(len(work))),
        )

        self.numeric_features = [
            "intercept",
            "booked_minutes",
            "log_booked_minutes",
            "month_sin",
            "month_cos",
            "week_sin",
            "week_cos",
            "wait_consult_to_surgery_days",
            "wait_decision_to_surgery_days",
            "wait_consult_to_decision_days",
            "surgeon_mean_error",
            "surgeon_median_error",
            "surgeon_qhat",
            "surgeon_mean_booked",
            "surgeon_mean_realized",
            "surgeon_log_cases",
            "service_mean_error",
            "service_median_error",
            "service_qhat",
            "service_mean_booked",
            "service_mean_realized",
            "service_log_cases",
            "procedure_mean_error",
            "procedure_median_error",
            "procedure_qhat",
            "procedure_mean_booked",
            "procedure_mean_realized",
            "procedure_log_cases",
            "room_mean_error",
            "room_median_error",
            "room_qhat",
            "room_mean_booked",
            "room_mean_realized",
            "room_log_cases",
        ]
        self.numeric_missing_indicators = [
            "miss_wait_consult_to_surgery_days",
            "miss_wait_decision_to_surgery_days",
            "miss_wait_consult_to_decision_days",
        ]
        self.categorical_vocab = {
            "patient_type": sorted(work[Col.PATIENT_TYPE].fillna(Domain.UNKNOWN).astype(str).unique().tolist()),
            "case_service": sorted(work[Col.CASE_SERVICE].fillna(Domain.UNKNOWN).astype(str).unique().tolist()),
            "procedure_id": sorted(work[Col.PROCEDURE_ID].fillna(Domain.OTHER).astype(str).unique().tolist()),
            "surgeon_code": sorted(work[Col.SURGEON_CODE].fillna(Domain.OTHER).astype(str).unique().tolist()),
            "operating_room": sorted(work[Col.OPERATING_ROOM].fillna(Domain.UNKNOWN).astype(str).unique().tolist()),
        }

        design_numeric = self._build_numeric_frame_from_df(work)
        for col in self.numeric_features + self.numeric_missing_indicators:
            vals = pd.to_numeric(design_numeric[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if col == "intercept":
                self.numeric_means[col] = 0.0
                self.numeric_stds[col] = 1.0
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if not np.isfinite(std) or std <= 1e-10:
                std = 1.0
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std

        self.feature_names = []
        self.feature_names.extend(self.numeric_features)
        self.feature_names.extend(self.numeric_missing_indicators)
        for feat, levels in self.categorical_vocab.items():
            for level in levels:
                self.feature_names.append(f"{feat}::{level}")

        self.feature_manifest = FeatureManifest(
            numeric_features=list(self.numeric_features),
            numeric_missing_indicators=list(self.numeric_missing_indicators),
            categorical_features={k: list(v) for k, v in self.categorical_vocab.items()},
            total_features=len(self.feature_names),
            notes=[
                "Site is intentionally excluded as a feature, per instruction.",
                "Only booking-time or pre-operative features are used.",
                "Rare-category recoding is fit on warmup only and then applied to pool.",
            ],
        )
        self._fitted = True
        return self

    def _lookup_stats(self, store: dict[str, dict[str, float]], key: str) -> dict[str, float]:
        assert self._defaults is not None
        return store.get(
            str(key),
            {
                "mean_error": self._defaults.mean_error,
                "median_error": self._defaults.median_error,
                "qhat": self._defaults.qhat,
                "mean_booked": self._defaults.mean_booked,
                "mean_realized": self._defaults.mean_realized,
                "log_cases": self._defaults.log_cases,
            },
        )

    @staticmethod
    def _days_between(a: pd.Series, b: pd.Series) -> pd.Series:
        ad = pd.to_datetime(a, errors="coerce")
        bd = pd.to_datetime(b, errors="coerce")
        return (ad - bd).dt.days.astype("float")

    def _build_numeric_frame_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: dict[str, Any] = {}
        booked = pd.to_numeric(df[Col.BOOKED_MINUTES], errors="coerce").fillna(0.0)
        month = pd.to_numeric(df[Col.MONTH], errors="coerce").fillna(0.0)
        week = pd.to_numeric(df[Col.WEEK_OF_YEAR], errors="coerce").fillna(0.0)

        wait_consult = self._days_between(df[Col.ACTUAL_START_DATE], df.get("consult_date"))
        wait_decision = self._days_between(df[Col.ACTUAL_START_DATE], df.get("decision_date"))
        consult_to_decision = self._days_between(df.get("decision_date"), df.get("consult_date"))

        def _fill_and_miss(s: pd.Series, name: str) -> tuple[pd.Series, pd.Series]:
            miss = s.isna().astype(float)
            filled = s.fillna(s.median() if s.notna().any() else 0.0).astype(float)
            return filled, miss

        wait_consult_fill, wait_consult_miss = _fill_and_miss(wait_consult, "wait_consult_to_surgery_days")
        wait_decision_fill, wait_decision_miss = _fill_and_miss(wait_decision, "wait_decision_to_surgery_days")
        consult_to_decision_fill, consult_to_decision_miss = _fill_and_miss(consult_to_decision, "wait_consult_to_decision_days")

        rows["intercept"] = pd.Series(np.ones(len(df), dtype=float), index=df.index)
        rows["booked_minutes"] = booked.astype(float)
        rows["log_booked_minutes"] = np.log1p(np.maximum(booked.astype(float), 0.0))
        rows["month_sin"] = month.map(_month_sin)
        rows["month_cos"] = month.map(_month_cos)
        rows["week_sin"] = week.map(_week_sin)
        rows["week_cos"] = week.map(_week_cos)
        rows["wait_consult_to_surgery_days"] = wait_consult_fill
        rows["wait_decision_to_surgery_days"] = wait_decision_fill
        rows["wait_consult_to_decision_days"] = consult_to_decision_fill
        rows["miss_wait_consult_to_surgery_days"] = wait_consult_miss
        rows["miss_wait_decision_to_surgery_days"] = wait_decision_miss
        rows["miss_wait_consult_to_decision_days"] = consult_to_decision_miss

        surgeon_codes = df[Col.SURGEON_CODE].fillna(Domain.OTHER).astype(str)
        services = df[Col.CASE_SERVICE].fillna(Domain.UNKNOWN).astype(str)
        procedures = df[Col.PROCEDURE_ID].fillna(Domain.OTHER).astype(str)
        rooms = df[Col.OPERATING_ROOM].fillna(Domain.UNKNOWN).astype(str)

        def pull(keys: pd.Series, store: dict[str, dict[str, float]], field: str) -> list[float]:
            return [float(self._lookup_stats(store, k)[field]) for k in keys]

        rows["surgeon_mean_error"] = pull(surgeon_codes, self._surgeon_stats, "mean_error")
        rows["surgeon_median_error"] = pull(surgeon_codes, self._surgeon_stats, "median_error")
        rows["surgeon_qhat"] = pull(surgeon_codes, self._surgeon_stats, "qhat")
        rows["surgeon_mean_booked"] = pull(surgeon_codes, self._surgeon_stats, "mean_booked")
        rows["surgeon_mean_realized"] = pull(surgeon_codes, self._surgeon_stats, "mean_realized")
        rows["surgeon_log_cases"] = pull(surgeon_codes, self._surgeon_stats, "log_cases")

        rows["service_mean_error"] = pull(services, self._service_stats, "mean_error")
        rows["service_median_error"] = pull(services, self._service_stats, "median_error")
        rows["service_qhat"] = pull(services, self._service_stats, "qhat")
        rows["service_mean_booked"] = pull(services, self._service_stats, "mean_booked")
        rows["service_mean_realized"] = pull(services, self._service_stats, "mean_realized")
        rows["service_log_cases"] = pull(services, self._service_stats, "log_cases")

        rows["procedure_mean_error"] = pull(procedures, self._procedure_stats, "mean_error")
        rows["procedure_median_error"] = pull(procedures, self._procedure_stats, "median_error")
        rows["procedure_qhat"] = pull(procedures, self._procedure_stats, "qhat")
        rows["procedure_mean_booked"] = pull(procedures, self._procedure_stats, "mean_booked")
        rows["procedure_mean_realized"] = pull(procedures, self._procedure_stats, "mean_realized")
        rows["procedure_log_cases"] = pull(procedures, self._procedure_stats, "log_cases")

        rows["room_mean_error"] = pull(rooms, self._room_stats, "mean_error")
        rows["room_median_error"] = pull(rooms, self._room_stats, "median_error")
        rows["room_qhat"] = pull(rooms, self._room_stats, "qhat")
        rows["room_mean_booked"] = pull(rooms, self._room_stats, "mean_booked")
        rows["room_mean_realized"] = pull(rooms, self._room_stats, "mean_realized")
        rows["room_log_cases"] = pull(rooms, self._room_stats, "log_cases")

        return pd.DataFrame(rows, index=df.index)

    def _build_numeric_frame_from_cases(self, cases: Sequence[CaseRecord], case_df: pd.DataFrame) -> pd.DataFrame:
        # case_df must align with cases and contain raw columns needed for wait-time features.
        return self._build_numeric_frame_from_df(case_df)

    def transform_df(self, df: pd.DataFrame) -> sparse.csr_matrix:
        assert self._fitted
        numeric_df = self._build_numeric_frame_from_df(df)
        return self._assemble_matrix(
            numeric_df=numeric_df,
            patient_type=df[Col.PATIENT_TYPE].fillna(Domain.UNKNOWN).astype(str).tolist(),
            case_service=df[Col.CASE_SERVICE].fillna(Domain.UNKNOWN).astype(str).tolist(),
            procedure_id=df[Col.PROCEDURE_ID].fillna(Domain.OTHER).astype(str).tolist(),
            surgeon_code=df[Col.SURGEON_CODE].fillna(Domain.OTHER).astype(str).tolist(),
            operating_room=df[Col.OPERATING_ROOM].fillna(Domain.UNKNOWN).astype(str).tolist(),
        )

    def transform_cases(self, cases: Sequence[CaseRecord], case_df: pd.DataFrame) -> sparse.csr_matrix:
        assert self._fitted
        numeric_df = self._build_numeric_frame_from_cases(cases, case_df)
        return self._assemble_matrix(
            numeric_df=numeric_df,
            patient_type=[str(c.patient_type) for c in cases],
            case_service=[str(c.service) for c in cases],
            procedure_id=[str(c.procedure_id) for c in cases],
            surgeon_code=[str(c.surgeon_code) for c in cases],
            operating_room=[str(c.operating_room) for c in cases],
        )

    def _assemble_matrix(
        self,
        *,
        numeric_df: pd.DataFrame,
        patient_type: Sequence[str],
        case_service: Sequence[str],
        procedure_id: Sequence[str],
        surgeon_code: Sequence[str],
        operating_room: Sequence[str],
    ) -> sparse.csr_matrix:
        n = len(numeric_df)
        blocks: list[sparse.csr_matrix] = []

        numeric_cols = []
        for col in self.numeric_features + self.numeric_missing_indicators:
            vals = pd.to_numeric(numeric_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            if col != "intercept":
                vals = (vals - self.numeric_means[col]) / self.numeric_stds[col]
            numeric_cols.append(vals.reshape(-1, 1))
        numeric_block = np.hstack(numeric_cols) if numeric_cols else np.zeros((n, 0), dtype=float)
        blocks.append(sparse.csr_matrix(numeric_block))

        cat_values = {
            "patient_type": list(patient_type),
            "case_service": list(case_service),
            "procedure_id": list(procedure_id),
            "surgeon_code": list(surgeon_code),
            "operating_room": list(operating_room),
        }
        for feat, levels in self.categorical_vocab.items():
            level_to_col = {lev: j for j, lev in enumerate(levels)}
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            for i, value in enumerate(cat_values[feat]):
                j = level_to_col.get(str(value))
                if j is None:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(1.0)
            block = sparse.csr_matrix((data, (rows, cols)), shape=(n, len(levels)))
            blocks.append(block)

        return sparse.hstack(blocks, format="csr")


# -----------------------------------------------------------------------------
# Model diagnostics
# -----------------------------------------------------------------------------

@dataclass
class FitDiagnostics:
    model_name: str
    fit_family: str
    optimizer: str
    exact_required: bool
    fit_status: str
    fit_obj_value: float
    fit_bound: float | None
    fit_gap: float | None
    fit_runtime_seconds: float
    n_training_cases: int
    n_features: int
    w_max: float
    ridge_lambda: float | None
    warm_start_enabled: bool
    warm_start_status: str | None = None
    warm_start_objective: float | None = None
    exact_training_mae_post: float | None = None
    exact_training_mae_booking: float | None = None
    exact_training_mae_vs_oracle_gap_closed: float | None = None
    mean_abs_delta_rec: float | None = None
    mean_abs_delta_post: float | None = None
    regime_acceptance_frac: float | None = None
    regime_decay_frac: float | None = None
    regime_discard_frac: float | None = None
    feature_manifest_path: str | None = None


@dataclass
class WeeklyScenarioPayload:
    planning_durations: np.ndarray
    case_diag_rows: list[dict[str, Any]]
    scenario_diag: dict[str, Any]


@dataclass
class SolveWeekPayload:
    week_meta: dict[str, Any]
    scenario_rows: list[dict[str, Any]]
    assignment_rows: list[dict[str, Any]]
    block_rows: list[dict[str, Any]]
    case_rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Base predictive model
# -----------------------------------------------------------------------------

class BasePredictiveModel:
    def __init__(
        self,
        *,
        model_name: str,
        response: ThreeRegimeResponse,
        feature_builder: FeatureBuilder,
        w_max: float,
        min_duration: float = 30.0,
        round_to_grid: bool = True,
        grid_step: float = 5.0,
    ) -> None:
        self.model_name = model_name
        self.response = response
        self.feature_builder = feature_builder
        self.w_max = float(w_max)
        self.min_duration = float(min_duration)
        self.round_to_grid = bool(round_to_grid)
        self.grid_step = float(grid_step)
        self.grid_residue = 0.0
        self.weights_: np.ndarray | None = None
        self.fit_diagnostics_: FitDiagnostics | None = None
        self.training_case_predictions_: pd.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return self.weights_ is not None

    def _compute_grid_residue(self, booked_vals: np.ndarray) -> None:
        if booked_vals.size == 0:
            self.grid_residue = 0.0
            return
        residues = np.mod(booked_vals, self.grid_step)
        residue = float(np.median(residues))
        if abs(residue - self.grid_step) <= 1e-8:
            residue = 0.0
        self.grid_residue = residue

    def _round_to_grid(self, vals: np.ndarray) -> np.ndarray:
        z = (vals - self.grid_residue) / self.grid_step
        return self.grid_residue + self.grid_step * np.round(z)

    def _predict_arrays(self, X: sparse.csr_matrix, booked: np.ndarray, surgeon_codes: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.weights_ is not None
        delta_rec = np.asarray(X @ self.weights_).reshape(-1)
        delta_post = self.response.apply_vector(delta_rec, surgeon_codes)
        d_post = booked + delta_post
        d_post = np.maximum(d_post, self.min_duration)
        if self.round_to_grid:
            d_post = self._round_to_grid(d_post)
            d_post = np.maximum(d_post, self.min_duration)
        regime = self.response.classify_regimes(delta_rec, surgeon_codes)
        return delta_rec, delta_post, d_post, regime

    def predict_week(self, instance: WeeklyInstance, case_df: pd.DataFrame) -> WeeklyScenarioPayload:
        assert self.is_fitted
        X = self.feature_builder.transform_cases(instance.cases, case_df)
        booked = np.asarray(instance.booked_durations(), dtype=float)
        realized = np.asarray(instance.actual_durations(), dtype=float)
        surgeon_codes = [c.surgeon_code for c in instance.cases]
        delta_rec, delta_post, d_post, regime = self._predict_arrays(X, booked, surgeon_codes)

        case_rows: list[dict[str, Any]] = []
        for i, c in enumerate(instance.cases):
            case_rows.append(
                {
                    "week": int(instance.week_index),
                    "case_id": int(c.case_id),
                    "surgeon_code": str(c.surgeon_code),
                    "service": str(c.service),
                    "procedure_id": str(c.procedure_id),
                    "patient_type": str(c.patient_type),
                    "operating_room": str(c.operating_room),
                    "booked_duration_min": float(booked[i]),
                    "actual_duration_min": float(realized[i]),
                    "planning_duration_min": float(d_post[i]),
                    "delta_rec": float(delta_rec[i]),
                    "delta_post": float(delta_post[i]),
                    "regime_code": int(regime[i]),
                    "regime": self.response.regime_name(int(regime[i])),
                    "abs_error_booked": float(abs(realized[i] - booked[i])),
                    "abs_error_planning": float(abs(realized[i] - d_post[i])),
                    "improvement_abs_error": float(abs(realized[i] - booked[i]) - abs(realized[i] - d_post[i])),
                }
            )

        diag = {
            "mean_delta_rec": float(np.mean(delta_rec)) if len(delta_rec) else math.nan,
            "mean_abs_delta_rec": float(np.mean(np.abs(delta_rec))) if len(delta_rec) else math.nan,
            "mean_delta_post": float(np.mean(delta_post)) if len(delta_post) else math.nan,
            "mean_abs_delta_post": float(np.mean(np.abs(delta_post))) if len(delta_post) else math.nan,
            "mae_vs_realized": float(np.mean(np.abs(realized - d_post))) if len(d_post) else math.nan,
            "mae_booked_vs_realized": float(np.mean(np.abs(realized - booked))) if len(booked) else math.nan,
            "regime_acceptance_frac": float(np.mean(regime == 0)) if len(regime) else math.nan,
            "regime_decay_frac": float(np.mean(regime == 1)) if len(regime) else math.nan,
            "regime_discard_frac": float(np.mean(regime == 2)) if len(regime) else math.nan,
        }
        return WeeklyScenarioPayload(planning_durations=d_post, case_diag_rows=case_rows, scenario_diag=diag)

    def _finalize_training_artifacts(
        self,
        *,
        X: sparse.csr_matrix,
        booked: np.ndarray,
        realized: np.ndarray,
        surgeon_codes: Sequence[str],
        df_train_scoped: pd.DataFrame,
        feature_manifest_path: str,
        fit_diag_base: dict[str, Any],
    ) -> None:
        delta_rec, delta_post, d_post, regime = self._predict_arrays(X, booked, surgeon_codes)
        exact_training_mae_post = float(np.mean(np.abs(realized - d_post)))
        exact_training_mae_booking = float(np.mean(np.abs(realized - booked)))
        gap_closed = ((exact_training_mae_booking - exact_training_mae_post) / exact_training_mae_booking) if exact_training_mae_booking > 1e-9 else float("nan")

        self.fit_diagnostics_ = FitDiagnostics(
            exact_training_mae_post=exact_training_mae_post,
            exact_training_mae_booking=exact_training_mae_booking,
            exact_training_mae_vs_oracle_gap_closed=float(gap_closed),
            mean_abs_delta_rec=float(np.mean(np.abs(delta_rec))),
            mean_abs_delta_post=float(np.mean(np.abs(delta_post))),
            regime_acceptance_frac=float(np.mean(regime == 0)),
            regime_decay_frac=float(np.mean(regime == 1)),
            regime_discard_frac=float(np.mean(regime == 2)),
            feature_manifest_path=feature_manifest_path,
            **fit_diag_base,
        )

        warmup_df = df_train_scoped.copy().reset_index(drop=True)
        warmup_df["service"] = warmup_df[Col.CASE_SERVICE]
        warmup_df["procedure_id"] = warmup_df[Col.PROCEDURE_ID]
        warmup_df["patient_type"] = warmup_df[Col.PATIENT_TYPE]
        warmup_df["operating_room"] = warmup_df[Col.OPERATING_ROOM]
        warmup_df["booked_duration_min"] = booked
        warmup_df["actual_duration_min"] = realized
        warmup_df["planning_duration_min"] = d_post
        warmup_df["delta_rec"] = delta_rec
        warmup_df["delta_post"] = delta_post
        warmup_df["d_post"] = d_post
        warmup_df["regime_code"] = regime
        warmup_df["regime"] = [self.response.regime_name(r) for r in regime]
        warmup_df["abs_error_booked"] = np.abs(realized - booked)
        warmup_df["abs_error_post"] = np.abs(realized - d_post)
        warmup_df["abs_error_planning"] = np.abs(realized - d_post)
        warmup_df["improvement_abs_error"] = warmup_df["abs_error_booked"] - warmup_df["abs_error_post"]
        self.training_case_predictions_ = warmup_df


# -----------------------------------------------------------------------------
# Plain exact LP: RegressionThenResponse
# -----------------------------------------------------------------------------

class RegressionThenResponseModel(BasePredictiveModel):
    def __init__(
        self,
        *,
        response: ThreeRegimeResponse,
        feature_builder: FeatureBuilder,
        w_max: float,
        fit_threads: int = 0,
        fit_time_limit_seconds: int = 0,
        fit_verbose: bool = False,
    ) -> None:
        super().__init__(
            model_name="RegressionThenResponse",
            response=response,
            feature_builder=feature_builder,
            w_max=w_max,
        )
        self.fit_threads = int(fit_threads)
        self.fit_time_limit_seconds = int(fit_time_limit_seconds)
        self.fit_verbose = bool(fit_verbose)

    def fit(self, df_train_scoped: pd.DataFrame, feature_manifest_path: str = "") -> "RegressionThenResponseModel":
        self.feature_builder.fit(df_train_scoped)
        booked = pd.to_numeric(df_train_scoped[Col.BOOKED_MINUTES], errors="coerce").to_numpy(dtype=float)
        realized = pd.to_numeric(df_train_scoped[Col.PROCEDURE_DURATION], errors="coerce").to_numpy(dtype=float)
        target_delta = realized - booked
        surgeon_codes = df_train_scoped[Col.SURGEON_CODE].fillna(Domain.OTHER).astype(str).tolist()
        self._compute_grid_residue(booked)

        X = self.feature_builder.transform_df(df_train_scoped)
        valid = np.isfinite(booked) & np.isfinite(realized)
        X = X[valid]
        booked = booked[valid]
        realized = realized[valid]
        target_delta = target_delta[valid]
        surgeon_codes = [surgeon_codes[i] for i in np.where(valid)[0]]

        weights, fit_info = self._solve_exact_linear_model(X=X, target_delta=target_delta)
        self.weights_ = weights
        self._finalize_training_artifacts(
            X=X,
            booked=booked,
            realized=realized,
            surgeon_codes=surgeon_codes,
            df_train_scoped=df_train_scoped.loc[valid].copy(),
            feature_manifest_path=feature_manifest_path,
            fit_diag_base={
                "model_name": self.model_name,
                "fit_family": "exact_linear_lp_then_response",
                "optimizer": "Gurobi exact LP",
                "exact_required": True,
                "fit_status": fit_info["status"],
                "fit_obj_value": float(fit_info["obj"]),
                "fit_bound": float(fit_info["bound"]),
                "fit_gap": 0.0,
                "fit_runtime_seconds": float(fit_info["runtime_seconds"]),
                "n_training_cases": int(len(target_delta)),
                "n_features": int(X.shape[1]),
                "w_max": self.w_max,
                "ridge_lambda": None,
                "warm_start_enabled": False,
                "warm_start_status": None,
                "warm_start_objective": None,
            },
        )
        return self

    def _solve_exact_linear_model(self, *, X: sparse.csr_matrix, target_delta: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        n, p = X.shape
        t0 = time.perf_counter()
        model = gp.Model("RegressionThenResponse_fit")
        model.Params.OutputFlag = 1 if self.fit_verbose else 0
        if self.fit_threads > 0:
            model.Params.Threads = self.fit_threads
        if self.fit_time_limit_seconds > 0:
            model.Params.TimeLimit = self.fit_time_limit_seconds

        w_vars = model.addVars(p, lb=-self.w_max, ub=self.w_max, vtype=GRB.CONTINUOUS, name="w")
        u_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
        e_pos = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="e_pos")
        e_neg = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="e_neg")

        for i in range(n):
            row = X.getrow(i)
            expr = gp.quicksum(float(val) * w_vars[int(j)] for val, j in zip(row.data.tolist(), row.indices.tolist()))
            model.addConstr(expr == u_vars[i], name=f"u_link[{i}]")
            model.addConstr(float(target_delta[i]) - u_vars[i] == e_pos[i] - e_neg[i], name=f"resid[{i}]")

        model.setObjective((1.0 / n) * gp.quicksum(e_pos[i] + e_neg[i] for i in range(n)), GRB.MINIMIZE)
        model.optimize()

        acceptable = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        if model.Status not in acceptable or model.SolCount == 0:
            raise RuntimeError(
                f"Exact LP fit failed. Status={model.Status}, SolCount={model.SolCount}"
            )

        if model.Status != GRB.OPTIMAL:
            LOGGER.warning(
                "RegressionThenResponse LP terminated with status %s; using incumbent solution.",
                model.Status,
            )

        weights = np.array([w_vars[j].X for j in range(p)], dtype=float)
        obj_val = float(model.ObjVal)
        bound_val = float(model.ObjBound) if hasattr(model, "ObjBound") else obj_val
        return weights, {
            "status": "OPTIMAL" if model.Status == GRB.OPTIMAL else str(model.Status),
            "obj": obj_val,
            "bound": bound_val,
            "runtime_seconds": time.perf_counter() - t0,
        }


# -----------------------------------------------------------------------------
# Exact response-aware PWL model
# -----------------------------------------------------------------------------

def pseudo_huber(x: np.ndarray, delta: float) -> np.ndarray:
    z = x / float(delta)
    return (float(delta) ** 2) * (np.sqrt(1.0 + z * z) - 1.0)


def pseudo_huber_grad(x: np.ndarray, delta: float) -> np.ndarray:
    z = x / float(delta)
    return x / np.sqrt(1.0 + z * z)


class ResponseAwareRegressionModel(BasePredictiveModel):
    def __init__(
        self,
        *,
        response: ThreeRegimeResponse,
        feature_builder: FeatureBuilder,
        w_max: float,
        fit_threads: int = 0,
        fit_time_limit_seconds: int = 0,
        fit_mip_gap: float = 0.0,
        fit_verbose: bool = False,
        require_exact_fit: bool = True,
        enable_smooth_warm_start: bool = True,
        warm_start_n_starts: int = 5,
        warm_start_maxiter: int = 400,
        warm_start_seed: int = 42,
        warm_start_ridge_lambda: float = 1e-4,
        warm_start_huber_delta: float = 2.5,
    ) -> None:
        super().__init__(
            model_name="ResponseAwareRegression",
            response=response,
            feature_builder=feature_builder,
            w_max=w_max,
        )
        self.fit_threads = int(fit_threads)
        self.fit_time_limit_seconds = int(fit_time_limit_seconds)
        self.fit_mip_gap = float(fit_mip_gap)
        self.fit_verbose = bool(fit_verbose)
        self.require_exact_fit = bool(require_exact_fit)
        self.enable_smooth_warm_start = bool(enable_smooth_warm_start)
        self.warm_start_n_starts = int(warm_start_n_starts)
        self.warm_start_maxiter = int(warm_start_maxiter)
        self.warm_start_seed = int(warm_start_seed)
        self.warm_start_ridge_lambda = float(warm_start_ridge_lambda)
        self.warm_start_huber_delta = float(warm_start_huber_delta)

    def fit(self, df_train_scoped: pd.DataFrame, feature_manifest_path: str = "") -> "ResponseAwareRegressionModel":
        if self.require_exact_fit and self.fit_mip_gap > 0:
            raise ValueError("require_exact_fit=True is incompatible with fit_mip_gap > 0. Set fit_mip_gap=0.")
        self.feature_builder.fit(df_train_scoped)
        booked = pd.to_numeric(df_train_scoped[Col.BOOKED_MINUTES], errors="coerce").to_numpy(dtype=float)
        realized = pd.to_numeric(df_train_scoped[Col.PROCEDURE_DURATION], errors="coerce").to_numpy(dtype=float)
        target_delta = realized - booked
        surgeon_codes = df_train_scoped[Col.SURGEON_CODE].fillna(Domain.OTHER).astype(str).tolist()
        self._compute_grid_residue(booked)

        X = self.feature_builder.transform_df(df_train_scoped)
        valid = np.isfinite(booked) & np.isfinite(realized)
        X = X[valid]
        booked = booked[valid]
        realized = realized[valid]
        target_delta = target_delta[valid]
        surgeon_codes = [surgeon_codes[i] for i in np.where(valid)[0]]

        warm_w = None
        warm_info: dict[str, Any] | None = None
        if self.enable_smooth_warm_start:
            warm_w, warm_info = self._solve_smooth_warm_start(X=X, target_delta=target_delta, surgeon_codes=surgeon_codes)

        weights, fit_info = self._solve_exact_model(
            X=X,
            target_delta=target_delta,
            surgeon_codes=surgeon_codes,
            warm_start_w=warm_w,
        )
        self.weights_ = weights
        self._finalize_training_artifacts(
            X=X,
            booked=booked,
            realized=realized,
            surgeon_codes=surgeon_codes,
            df_train_scoped=df_train_scoped.loc[valid].copy(),
            feature_manifest_path=feature_manifest_path,
            fit_diag_base={
                "model_name": self.model_name,
                "fit_family": "exact_response_aware_pwl_mip",
                "optimizer": "Gurobi exact PWL/MIP",
                "exact_required": self.require_exact_fit,
                "fit_status": fit_info["status"],
                "fit_obj_value": float(fit_info["obj"]),
                "fit_bound": None if fit_info["bound"] is None else float(fit_info["bound"]),
                "fit_gap": None if fit_info["gap"] is None else float(fit_info["gap"]),
                "fit_runtime_seconds": float(fit_info["runtime_seconds"]),
                "n_training_cases": int(len(target_delta)),
                "n_features": int(X.shape[1]),
                "w_max": self.w_max,
                "ridge_lambda": self.warm_start_ridge_lambda if self.enable_smooth_warm_start else None,
                "warm_start_enabled": self.enable_smooth_warm_start,
                "warm_start_status": None if warm_info is None else str(warm_info["status"]),
                "warm_start_objective": None if warm_info is None else float(warm_info["objective"]),
            },
        )
        return self

    def _solve_smooth_warm_start(
        self,
        *,
        X: sparse.csr_matrix,
        target_delta: np.ndarray,
        surgeon_codes: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n, p = X.shape
        rng = np.random.default_rng(self.warm_start_seed)
        starts: list[tuple[str, np.ndarray]] = [("zero", np.zeros(p, dtype=float))]

        # Linearized least-squares start under acceptance slope.
        a0 = self.response.default_params.a_s
        if abs(a0) > 1e-8:
            try:
                y_lin = target_delta / a0
                lsqr = sparse.linalg.lsqr(X, y_lin, atol=1e-6, btol=1e-6, iter_lim=200)[0]
                lsqr = np.asarray(lsqr, dtype=float)
                lsqr = np.clip(lsqr, -self.w_max, self.w_max)
                starts.append(("linearized_lsqr", lsqr))
            except Exception:
                LOGGER.exception("LSQR warm start failed; continuing.")

        bases = [starts[0][1]]
        if len(starts) > 1:
            bases.append(starts[1][1])
        while len(starts) < self.warm_start_n_starts:
            base = bases[(len(starts) - 1) % len(bases)]
            jitter = rng.normal(loc=0.0, scale=0.2, size=p)
            starts.append((f"jitter_{len(starts)}", np.clip(base + jitter, -self.w_max, self.w_max)))

        def objective_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
            u = np.asarray(X @ w).reshape(-1)
            delta_post = self.response.apply_vector(u, surgeon_codes)
            err = target_delta - delta_post
            loss = float(pseudo_huber(err, self.warm_start_huber_delta).mean()) + 0.5 * self.warm_start_ridge_lambda * float(np.dot(w, w))
            slopes = self.response.slope_vector(u, surgeon_codes)
            v = -pseudo_huber_grad(err, self.warm_start_huber_delta) * slopes / n
            grad = np.asarray(X.T @ v).reshape(-1) + self.warm_start_ridge_lambda * w
            return loss, grad

        best_w = np.zeros(p, dtype=float)
        best_obj = float("inf")
        best_status = "FAILED"
        bounds = [(-self.w_max, self.w_max)] * p

        for start_name, w0 in starts:
            try:
                res = minimize(
                    fun=objective_and_grad,
                    jac=True,
                    x0=np.asarray(w0, dtype=float),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.warm_start_maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
                )
                if float(res.fun) < best_obj:
                    best_obj = float(res.fun)
                    best_w = np.asarray(res.x, dtype=float).copy()
                    best_status = f"{start_name}:{res.message}"
            except Exception as exc:
                LOGGER.warning("Warm start %s failed: %s", start_name, exc)

        return best_w, {"status": best_status, "objective": best_obj}

    def _solve_exact_model(
        self,
        *,
        X: sparse.csr_matrix,
        target_delta: np.ndarray,
        surgeon_codes: Sequence[str],
        warm_start_w: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n, p = X.shape
        t0 = time.perf_counter()
        model = gp.Model("ResponseAwareRegression_fit")
        _apply_response_fit_params(
            model,
            time_limit_seconds=self.fit_time_limit_seconds,
            verbose=self.fit_verbose,
        )
        model.Params.OutputFlag = 1 if self.fit_verbose else 0
        # model.Params.MIPFocus = 1
        model.Params.Symmetry = 0
        if self.fit_threads > 0:
            model.Params.Threads = self.fit_threads
        if self.fit_time_limit_seconds > 0:
            model.Params.TimeLimit = self.fit_time_limit_seconds
        model.Params.MIPGap = self.fit_mip_gap

        w_vars = model.addVars(p, lb=-self.w_max, ub=self.w_max, vtype=GRB.CONTINUOUS, name="w")
        u_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
        delta_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta")
        e_pos = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="e_pos")
        e_neg = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="e_neg")

        for i in range(n):
            row = X.getrow(i)
            expr = gp.quicksum(float(val) * w_vars[int(j)] for val, j in zip(row.data.tolist(), row.indices.tolist()))
            model.addConstr(expr == u_vars[i], name=f"u_link[{i}]")
            Ui = self.w_max * float(np.sum(np.abs(row.data))) + 1e-6
            model.addConstr(u_vars[i] >= -Ui, name=f"u_lb[{i}]")
            model.addConstr(u_vars[i] <= Ui, name=f"u_ub[{i}]")
            p_i = self.response.get_params(surgeon_codes[i])
            xpts, ypts = _response_graph_points(U=Ui, p=p_i)
            model.addGenConstrPWL(u_vars[i], delta_vars[i], xpts, ypts, name=f"resp[{i}]")
            model.addConstr(float(target_delta[i]) - delta_vars[i] == e_pos[i] - e_neg[i], name=f"resid[{i}]")

        model.setObjective((1.0 / n) * gp.quicksum(e_pos[i] + e_neg[i] for i in range(n)), GRB.MINIMIZE)

        if warm_start_w is not None:
            for j in range(p):
                w_vars[j].Start = float(warm_start_w[j])
            warm_u = np.asarray(X @ warm_start_w).reshape(-1)
            warm_delta = self.response.apply_vector(warm_u, surgeon_codes)
            for i in range(n):
                u_vars[i].Start = float(warm_u[i])
                delta_vars[i].Start = float(warm_delta[i])
                resid = float(target_delta[i] - warm_delta[i])
                e_pos[i].Start = max(resid, 0.0)
                e_neg[i].Start = max(-resid, 0.0)

        model.optimize()

        acceptable = {GRB.OPTIMAL}
        if not self.require_exact_fit:
            acceptable.update({GRB.TIME_LIMIT, GRB.SUBOPTIMAL})
        if model.Status not in acceptable or model.SolCount == 0:
            raise RuntimeError(
                f"Response-aware fit did not produce an acceptable solution. "
                f"Status={model.Status}, SolCount={model.SolCount}, ObjBound={getattr(model,'ObjBound',None)}"
            )
        if self.require_exact_fit and model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Response-aware fit is not exact. Status={model.Status}")

        weights = np.array([w_vars[j].X for j in range(p)], dtype=float)
        status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL"}
        return weights, {
            "status": status_map.get(model.Status, str(model.Status)),
            "obj": float(model.ObjVal),
            "bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
            "gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
            "runtime_seconds": time.perf_counter() - t0,
        }


# -----------------------------------------------------------------------------
# Weekly scheduling helpers
# -----------------------------------------------------------------------------

def evaluate_with_durations(
    instance: WeeklyInstance,
    schedule: ScheduleResult,
    durations: np.ndarray,
    turnover: float,
    costs,
) -> tuple[KPIResult, list[dict[str, Any]]]:
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
    total_overtime = 0.0
    total_idle = 0.0
    block_rows: list[dict[str, Any]] = []
    for bid in schedule.opened_blocks:
        k = int(block_case_count.get(bid, 0))
        turn = float(turnover) * max(k - 1, 0)
        case_load = float(block_load.get(bid, 0.0))
        total_load = case_load + turn
        total_turnover += turn
        cap = float(instance.calendar.capacity(bid))
        overtime = max(total_load - cap, 0.0)
        idle = max(cap - total_load, 0.0)
        total_overtime += overtime
        total_idle += idle
        block_rows.append(
            {
                "day_index": int(bid.day_index),
                "site": str(bid.site),
                "room": str(bid.room),
                "capacity_minutes": cap,
                "n_cases": k,
                "case_load_minutes": case_load,
                "turnover_minutes": turn,
                "total_load_minutes": total_load,
                "overtime_minutes": overtime,
                "idle_minutes": idle,
                "activation_cost": float(instance.calendar.activation_cost(bid)),
            }
        )

    activation_cost = float(sum(instance.calendar.activation_cost(bid) for bid in schedule.opened_blocks))
    overtime_cost = float(costs.overtime_per_minute * total_overtime)
    idle_cost = float(costs.idle_per_minute * total_idle)
    total_cost = float(activation_cost + deferral_cost + overtime_cost + idle_cost)
    kpi = KPIResult(
        total_cost=total_cost,
        activation_cost=activation_cost,
        overtime_cost=overtime_cost,
        idle_cost=idle_cost,
        deferral_cost=float(deferral_cost),
        overtime_minutes=float(total_overtime),
        idle_minutes=float(total_idle),
        turnover_minutes=float(total_turnover),
        scheduled_count=int(scheduled_count),
        deferred_count=int(deferred_count),
        blocks_opened=int(len(schedule.opened_blocks)),
    )
    return kpi, block_rows


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


def flatten_assignments(
    instance: WeeklyInstance,
    schedule: ScheduleResult,
    scenario: str,
    planning_durations: np.ndarray,
    case_diag_lookup: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    case_map = {c.case_id: c for c in instance.cases}
    case_index = {c.case_id: i for i, c in enumerate(instance.cases)}
    rows: list[dict[str, Any]] = []
    for a in schedule.assignments:
        case = case_map.get(a.case_id)
        i = case_index.get(a.case_id)
        diag = case_diag_lookup.get(a.case_id, {}) if case_diag_lookup is not None else {}
        planning = float(planning_durations[i]) if i is not None else math.nan
        booked = float(case.booked_duration_min) if case is not None else math.nan
        actual = float(case.actual_duration_min) if case is not None else math.nan
        rows.append(
            {
                "week": int(instance.week_index),
                "scenario": scenario,
                "case_id": int(a.case_id),
                "surgeon_code": str(case.surgeon_code) if case else None,
                "service": str(case.service) if case else None,
                "procedure_id": str(case.procedure_id) if case else None,
                "patient_type": str(case.patient_type) if case else None,
                "operating_room": str(case.operating_room) if case else None,
                "booked_duration_min": booked,
                "actual_duration_min": actual,
                "planning_duration_min": planning,
                "planning_minus_booked": float(planning - booked) if np.isfinite(planning) and np.isfinite(booked) else math.nan,
                "planning_minus_actual": float(planning - actual) if np.isfinite(planning) and np.isfinite(actual) else math.nan,
                "abs_error_actual": float(abs(planning - actual)) if np.isfinite(planning) and np.isfinite(actual) else math.nan,
                "is_deferred": bool(a.is_deferred),
                "day_index": None if a.is_deferred else int(a.day_index),
                "site": None if a.is_deferred else str(a.site),
                "room": None if a.is_deferred else str(a.room),
                "delta_rec": diag.get("delta_rec"),
                "delta_post": diag.get("delta_post"),
                "regime": diag.get("regime"),
                "regime_code": diag.get("regime_code"),
            }
        )
    return rows


def build_all_instances(scope_bundle: ScopeBundle, config: Config, scenarios_requested: Sequence[str]) -> tuple[list[WeeklyInstance], BuildMeta]:
    if config.scope.use_all_sites_for_warmup:
        warmup_for_elig = scope_bundle.df_warmup
    else:
        warmup_for_elig = scope_bundle.df_warmup_scoped

    elig_maps = build_eligibility_maps(warmup_for_elig, config)
    candidate_pools = build_candidate_pools(scope_bundle.df_warmup_scoped, config)

    instances: list[WeeklyInstance] = []
    current_start = scope_bundle.pool_start
    for h in range(config.data.num_horizons):
        instance = build_weekly_instance(
            df_pool=scope_bundle.df_pool_scoped,
            horizon_start=current_start,
            week_index=h,
            config=config,
            candidate_pools=candidate_pools,
            eligibility_maps=elig_maps,
        )
        if instance.num_cases == 0:
            break
        instances.append(instance)
        current_start += pd.Timedelta(days=config.scope.stride_days)

    meta = BuildMeta(
        pool_start=str(scope_bundle.pool_start),
        n_total_rows=int(len(scope_bundle.df_all)),
        n_warmup_rows=int(len(scope_bundle.df_warmup)),
        n_pool_rows=int(len(scope_bundle.df_pool)),
        n_warmup_scoped_rows=int(len(scope_bundle.df_warmup_scoped)),
        n_pool_scoped_rows=int(len(scope_bundle.df_pool_scoped)),
        n_instances=int(len(instances)),
        n_candidate_pool_days=int(len(candidate_pools)),
        scenarios_requested=list(scenarios_requested),
    )
    return instances, meta


def build_case_df_for_instance(instance: WeeklyInstance, pool_scoped_df: pd.DataFrame) -> pd.DataFrame:
    # Build a dataframe aligned with instance.cases for feature transformation.
    df_week = pool_scoped_df[pool_scoped_df[Col.CASE_UID].isin([c.case_id for c in instance.cases])].copy()
    df_week = df_week.set_index(Col.CASE_UID).reindex([c.case_id for c in instance.cases]).reset_index()
    return df_week


def build_weekly_scenario_payloads(
    instance: WeeklyInstance,
    *,
    pool_scoped_df: pd.DataFrame,
    active_scenarios: Sequence[str],
    plain_model: RegressionThenResponseModel | None,
    response_aware_model: ResponseAwareRegressionModel | None,
) -> dict[str, WeeklyScenarioPayload]:
    payloads: dict[str, WeeklyScenarioPayload] = {}
    booked = np.asarray(instance.booked_durations(), dtype=float)
    actual = np.asarray(instance.actual_durations(), dtype=float)

    if "Booked" in active_scenarios:
        rows = []
        for c, p in zip(instance.cases, booked):
            rows.append(
                {
                    "week": int(instance.week_index),
                    "case_id": int(c.case_id),
                    "surgeon_code": str(c.surgeon_code),
                    "service": str(c.service),
                    "procedure_id": str(c.procedure_id),
                    "patient_type": str(c.patient_type),
                    "operating_room": str(c.operating_room),
                    "booked_duration_min": float(c.booked_duration_min),
                    "actual_duration_min": float(c.actual_duration_min),
                    "planning_duration_min": float(p),
                    "delta_rec": None,
                    "delta_post": None,
                    "regime_code": None,
                    "regime": None,
                    "abs_error_booked": float(abs(c.actual_duration_min - c.booked_duration_min)),
                    "abs_error_planning": float(abs(c.actual_duration_min - p)),
                    "improvement_abs_error": 0.0,
                }
            )
        payloads["Booked"] = WeeklyScenarioPayload(
            planning_durations=booked,
            case_diag_rows=rows,
            scenario_diag={"mae_vs_realized": float(np.mean(np.abs(actual - booked))) if len(booked) else math.nan},
        )

    if "Oracle" in active_scenarios:
        rows = []
        for c, p in zip(instance.cases, actual):
            rows.append(
                {
                    "week": int(instance.week_index),
                    "case_id": int(c.case_id),
                    "surgeon_code": str(c.surgeon_code),
                    "service": str(c.service),
                    "procedure_id": str(c.procedure_id),
                    "patient_type": str(c.patient_type),
                    "operating_room": str(c.operating_room),
                    "booked_duration_min": float(c.booked_duration_min),
                    "actual_duration_min": float(c.actual_duration_min),
                    "planning_duration_min": float(p),
                    "delta_rec": None,
                    "delta_post": None,
                    "regime_code": None,
                    "regime": None,
                    "abs_error_booked": float(abs(c.actual_duration_min - c.booked_duration_min)),
                    "abs_error_planning": 0.0,
                    "improvement_abs_error": float(abs(c.actual_duration_min - c.booked_duration_min)),
                }
            )
        payloads["Oracle"] = WeeklyScenarioPayload(
            planning_durations=actual,
            case_diag_rows=rows,
            scenario_diag={"mae_vs_realized": 0.0},
        )

    if "RegressionThenResponse" in active_scenarios:
        if plain_model is None:
            raise RuntimeError("RegressionThenResponse requested but plain model was not fitted.")
        case_df = build_case_df_for_instance(instance, pool_scoped_df)
        payloads["RegressionThenResponse"] = plain_model.predict_week(instance, case_df)

    if "ResponseAwareRegression" in active_scenarios:
        if response_aware_model is None:
            raise RuntimeError("ResponseAwareRegression requested but response-aware model was not fitted.")
        case_df = build_case_df_for_instance(instance, pool_scoped_df)
        payloads["ResponseAwareRegression"] = response_aware_model.predict_week(instance, case_df)

    return payloads


def solve_one_week(
    instance: WeeklyInstance,
    *,
    scope_bundle: ScopeBundle,
    config: Config,
    active_scenarios: Sequence[str],
    plain_model: RegressionThenResponseModel | None,
    response_aware_model: ResponseAwareRegressionModel | None,
) -> SolveWeekPayload:
    turnover = float(config.capacity.turnover_minutes)
    costs = config.costs
    actual_durations = np.asarray(instance.actual_durations(), dtype=float)
    booked_durations = np.asarray(instance.booked_durations(), dtype=float)
    n_forced_defer = int(sum(1 for i in range(instance.num_cases) if len(instance.case_eligible_blocks.get(i, [])) == 0))

    scenario_payloads = build_weekly_scenario_payloads(
        instance,
        pool_scoped_df=scope_bundle.df_pool_scoped,
        active_scenarios=active_scenarios,
        plain_model=plain_model,
        response_aware_model=response_aware_model,
    )

    week_meta = {
        "week": int(instance.week_index),
        "start_date": str(instance.start_date),
        "end_date": str(instance.end_date),
        "n_cases": int(instance.num_cases),
        "candidate_blocks": int(instance.calendar.total_candidates),
        "forced_defer_count": n_forced_defer,
    }

    scenario_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for scenario in active_scenarios:
        payload = scenario_payloads[scenario]
        planning_durations = payload.planning_durations
        case_rows.extend([{**row, "scenario": scenario} for row in payload.case_diag_rows])
        case_diag_lookup = {int(row["case_id"]): row for row in payload.case_diag_rows}

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
                model_name=f"{scenario}_week_{instance.week_index}",
            )
            solve_time = time.perf_counter() - t0

            planned_kpi, planned_block_rows = evaluate_with_durations(instance, schedule, planning_durations, turnover, costs)
            realized_kpi, realized_block_rows = evaluate_with_durations(instance, schedule, actual_durations, turnover, costs)

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
                "booked_mean": float(np.mean(booked_durations)) if len(booked_durations) else math.nan,
                "actual_mean": float(np.mean(actual_durations)) if len(actual_durations) else math.nan,
                "planning_mean": float(np.mean(planning_durations)) if len(planning_durations) else math.nan,
                "planning_minus_booked_mean": float(np.mean(planning_durations - booked_durations)) if len(planning_durations) else math.nan,
                "planning_minus_actual_mean": float(np.mean(planning_durations - actual_durations)) if len(planning_durations) else math.nan,
                "mae_planning_vs_realized_casewise": float(np.mean(np.abs(planning_durations - actual_durations))) if len(planning_durations) else math.nan,
                "mae_booked_vs_realized_casewise": float(np.mean(np.abs(booked_durations - actual_durations))) if len(planning_durations) else math.nan,
            }
            row.update(kpi_to_prefixed_dict(planned_kpi, "planned"))
            row.update(kpi_to_prefixed_dict(realized_kpi, "realized"))
            for k, v in payload.scenario_diag.items():
                row[f"scenario_diag__{k}"] = v
            scenario_rows.append(row)

            assignment_rows.extend(flatten_assignments(instance, schedule, scenario, planning_durations, case_diag_lookup))

            planned_lookup = {(r["day_index"], r["site"], r["room"]): r for r in planned_block_rows}
            realized_lookup = {(r["day_index"], r["site"], r["room"]): r for r in realized_block_rows}
            for key in sorted(set(planned_lookup) | set(realized_lookup)):
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
                        "planned_total_load_minutes": p.get("total_load_minutes"),
                        "planned_overtime_minutes": p.get("overtime_minutes"),
                        "planned_idle_minutes": p.get("idle_minutes"),
                        "realized_total_load_minutes": r.get("total_load_minutes"),
                        "realized_overtime_minutes": r.get("overtime_minutes"),
                        "realized_idle_minutes": r.get("idle_minutes"),
                    }
                )
        except Exception as exc:
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
                    "n_cases": int(instance.num_cases),
                    "candidate_blocks": int(instance.calendar.total_candidates),
                }
            )

    return SolveWeekPayload(
        week_meta=week_meta,
        scenario_rows=scenario_rows,
        assignment_rows=assignment_rows,
        block_rows=block_rows,
        case_rows=case_rows,
        failures=failures,
    )


# -----------------------------------------------------------------------------
# Summaries and reporting
# -----------------------------------------------------------------------------

def summarize_scenarios(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        c
        for c in results_df.columns
        if c not in {"week", "scenario", "start_date", "end_date", "solver_status"}
        and pd.api.types.is_numeric_dtype(results_df[c])
    ]
    rows: list[dict[str, Any]] = []
    for scenario, sub in results_df.groupby("scenario", sort=True):
        row: dict[str, Any] = {"scenario": scenario, "n_weeks": int(sub["week"].nunique())}
        for col in numeric_cols:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}__mean"] = float(s.mean()) if not s.empty else math.nan
            row[f"{col}__std"] = float(s.std(ddof=1)) if len(s) >= 2 else math.nan
            row[f"{col}__median"] = float(s.median()) if not s.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


def build_pairwise_weekly_deltas(results_df: pd.DataFrame, active_scenarios: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results_df.empty or len(active_scenarios) < 2:
        return pd.DataFrame(), pd.DataFrame()

    pivot = results_df.pivot(index="week", columns="scenario")
    base_metrics = [
        "realized_total_cost",
        "realized_activation_cost",
        "realized_overtime_cost",
        "realized_idle_cost",
        "realized_deferral_cost",
        "realized_overtime_minutes",
        "realized_idle_minutes",
        "realized_deferred_count",
        "realized_blocks_opened",
    ]

    weekly_rows: list[dict[str, Any]] = []
    for week in sorted(results_df["week"].unique()):
        row: dict[str, Any] = {"week": int(week)}
        for i, a in enumerate(active_scenarios):
            for b in active_scenarios[i + 1 :]:
                pair = f"{a}_minus_{b}"
                for metric in base_metrics:
                    a_val = float(pivot.loc[week, (metric, a)]) if (metric, a) in pivot.columns else math.nan
                    b_val = float(pivot.loc[week, (metric, b)]) if (metric, b) in pivot.columns else math.nan
                    row[f"{pair}__{metric}"] = a_val - b_val if math.isfinite(a_val) and math.isfinite(b_val) else math.nan
        weekly_rows.append(row)

    weekly_df = pd.DataFrame(weekly_rows).sort_values("week").reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    for col in weekly_df.columns:
        if col == "week":
            continue
        s = pd.to_numeric(weekly_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        pair_name, metric = col.split("__", 1)
        summary_rows.append(
            {
                "pair_name": pair_name,
                "metric": metric,
                "n_weeks": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) >= 2 else math.nan,
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "positive_weeks": int((s > 0).sum()),
                "negative_weeks": int((s < 0).sum()),
            }
        )
    return weekly_df, pd.DataFrame(summary_rows)


def render_text_report(
    *,
    args: argparse.Namespace,
    load_summary: LoadSummary,
    build_meta: BuildMeta,
    recode_info: dict[str, list[str]],
    scope_bundle: ScopeBundle,
    fit_diagnostics: list[FitDiagnostics],
    scenario_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("Unified offline comparison experiment")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 88)
    lines.append(f"Excel file: {args.excel}")
    lines.append(f"Active scenarios: {', '.join(args.scenarios)}")
    lines.append(f"Planning sites: {', '.join(args.planning_sites)}")
    lines.append(f"Warmup weeks: {args.warmup_weeks}")
    lines.append(f"Num horizons: {args.num_horizons}")
    lines.append(f"Workers: {args.workers} | threads per weekly solve: {args.threads_per_worker}")
    lines.append(f"Response params: a={args.response_a}, h={args.response_h}, rho={args.response_rho}")
    lines.append(f"Response-aware exact fit required: {args.require_exact_response_aware_fit}")
    lines.append("")

    lines.append("Cohort summary")
    lines.append("-" * 88)
    for k, v in load_summary.to_dict().items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"pool_start: {build_meta.pool_start}")
    lines.append(f"n_total_rows: {build_meta.n_total_rows}")
    lines.append(f"n_warmup_rows: {build_meta.n_warmup_rows}")
    lines.append(f"n_pool_rows: {build_meta.n_pool_rows}")
    lines.append(f"n_warmup_scoped_rows: {build_meta.n_warmup_scoped_rows}")
    lines.append(f"n_pool_scoped_rows: {build_meta.n_pool_scoped_rows}")
    lines.append(f"n_instances: {build_meta.n_instances}")
    lines.append(f"n_candidate_pool_days: {build_meta.n_candidate_pool_days}")
    lines.append(f"scenarios_requested: {build_meta.scenarios_requested}")
    lines.append("")
    lines.append("Scope summary")
    lines.append("-" * 88)
    lines.append(json.dumps({
        "warmup_scope_summary": asdict(scope_bundle.warmup_scope_summary),
        "pool_scope_summary": asdict(scope_bundle.pool_scope_summary),
    }, indent=2))
    lines.append("")
    lines.append("Leakage-safe recoding summary")
    lines.append("-" * 88)
    lines.append(f"n_rare_procedure_values: {len(recode_info['procedure_rare_values'])}")
    lines.append(f"n_rare_surgeon_values: {len(recode_info['surgeon_rare_values'])}")
    lines.append(f"n_rare_service_values: {len(recode_info['service_rare_values'])}")
    lines.append("")

    for fit_diag in fit_diagnostics:
        lines.append(f"{fit_diag.model_name} training diagnostics")
        lines.append("-" * 88)
        for k, v in asdict(fit_diag).items():
            lines.append(f"{k}: {v}")
        lines.append("")

    if not scenario_summary.empty:
        lines.append("Scenario summary (means over evaluated weeks)")
        lines.append("-" * 88)
        view_cols = [
            "scenario",
            "n_weeks",
            "realized_total_cost__mean",
            "realized_overtime_minutes__mean",
            "realized_idle_minutes__mean",
            "realized_deferred_count__mean",
            "realized_blocks_opened__mean",
            "solve_time_seconds__mean",
        ]
        view_cols = [c for c in view_cols if c in scenario_summary.columns]
        lines.append(scenario_summary[view_cols].to_string(index=False))
        lines.append("")

    if not pairwise_summary.empty:
        lines.append("Pairwise weekly-delta summary")
        lines.append("-" * 88)
        lines.append(pairwise_summary.to_string(index=False))
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

    def write_many(self, records: list[dict]) -> None:
        for rec in records:
            self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

ALL_SCENARIOS = ["Oracle", "Booked", "RegressionThenResponse", "ResponseAwareRegression"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified regression/response comparison runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--excel", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--run-label", default="regression_response_compare")
    parser.add_argument("--planning-sites", nargs="+", default=["TGH", "TWH"])
    parser.add_argument("--warmup-weeks", type=int, default=52)
    parser.add_argument("--num-horizons", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", default=["ResponseAwareRegression", "Booked", "Oracle"], choices=ALL_SCENARIOS)

    # Parallel weekly solves
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=1)

    # Weekly deterministic solver settings
    parser.add_argument("--time-limit-seconds", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.05)

    # OR planning costs
    parser.add_argument("--block-capacity-minutes", type=float, default=480.0)
    parser.add_argument("--activation-cost-per-block", type=float, default=2000.0)
    parser.add_argument("--min-activation-rate", type=float, default=0.25)
    parser.add_argument("--turnover-minutes", type=float, default=30.0)
    parser.add_argument("--overtime-per-minute", type=float, default=15.0)
    parser.add_argument("--idle-per-minute", type=float, default=10.0)
    parser.add_argument("--deferral-per-case", type=float, default=2000.0)

    # Response function
    parser.add_argument("--response-a", type=float, default=1)
    parser.add_argument("--response-h", type=float, default=60)
    parser.add_argument("--response-rho", type=float, default=1.5)

    # Feature / model settings
    parser.add_argument("--w-max", type=float, default=25.0)
    parser.add_argument("--feature-min-group-cases", type=int, default=5)

    # Response-aware exact fit
    parser.add_argument("--fit-threads", type=int, default=0)
    parser.add_argument("--fit-time-limit-seconds", type=int, default=30)
    parser.add_argument("--fit-mip-gap", type=float, default=0.01)
    parser.add_argument("--require-exact-response-aware-fit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-smooth-warm-start", action="store_true", default=False)
    parser.add_argument("--warm-start-n-starts", type=int, default=5)
    parser.add_argument("--warm-start-maxiter", type=int, default=400)
    parser.add_argument("--warm-start-seed", type=int, default=42)
    parser.add_argument("--warm-start-ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--warm-start-huber-delta", type=float, default=2.5)

    # Cohort flags
    parser.add_argument("--include-cancelled", action="store_true", default=False)
    parser.add_argument("--disable-booked-positive-filter", action="store_true", default=False)
    parser.add_argument("--weekday-scope", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.data.excel_file_path = args.excel
    cfg.data.warmup_weeks = args.warmup_weeks
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
        format="%(asctime)s  %(name)-32s  %(levelname)-7s  %(message)s",
    )
    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    cfg = build_config(args)
    exclude_cancelled = not bool(args.include_cancelled)
    positive_booked_only = not bool(args.disable_booked_positive_filter)

    artifact_run = ArtifactManager(args.artifact_root).run("experiments", args.run_label)
    run_dir = artifact_run.ensure_run_dir()
    install_gurobi_log_capture(artifact_run.directory("gurobi_logs"))

    artifact_run.path("args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    artifact_run.path("config_snapshot.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    df_all, load_summary = load_clean_data_for_experiment(
        cfg,
        exclude_cancelled=exclude_cancelled,
        positive_booked_only=positive_booked_only,
    )
    artifact_run.path("load_summary.json").write_text(json.dumps(load_summary.to_dict(), indent=2), encoding="utf-8")

    scope_bundle, recode_info = build_scope_bundle(df_all, cfg)
    artifact_run.path("recode_info.json").write_text(json.dumps(recode_info, indent=2), encoding="utf-8")
    artifact_run.path("scope_summary.json").write_text(
        json.dumps(
            {
                "warmup_scope_summary": asdict(scope_bundle.warmup_scope_summary),
                "pool_scope_summary": asdict(scope_bundle.pool_scope_summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    instances, build_meta = build_all_instances(scope_bundle, cfg, args.scenarios)
    artifact_run.path("build_meta.json").write_text(json.dumps(asdict(build_meta), indent=2), encoding="utf-8")

    response = ThreeRegimeResponse(
        default_params=ResponseParams(
            a_s=float(args.response_a),
            h_c=float(args.response_h),
            rho_s=float(args.response_rho),
        )
    )

    feature_builder = FeatureBuilder(min_group_cases=int(args.feature_min_group_cases))
    plain_model: RegressionThenResponseModel | None = None
    response_aware_model: ResponseAwareRegressionModel | None = None
    fit_diags: list[FitDiagnostics] = []

    if "RegressionThenResponse" in args.scenarios:
        plain_manifest_path = "plain_feature_manifest.json"
        plain_model = RegressionThenResponseModel(
            response=response,
            feature_builder=feature_builder,
            w_max=float(args.w_max),
            fit_threads=int(args.fit_threads),
            fit_time_limit_seconds=int(args.fit_time_limit_seconds),
            fit_verbose=bool(args.verbose),
        )
        plain_model.fit(scope_bundle.df_warmup_scoped, feature_manifest_path=plain_manifest_path)
        artifact_run.path(plain_manifest_path).write_text(json.dumps(asdict(feature_builder.feature_manifest), indent=2), encoding="utf-8")
        pd.DataFrame({"feature": feature_builder.feature_names, "weight": plain_model.weights_.tolist()}).to_csv(
            artifact_run.path("plain_feature_weights.csv"), index=False
        )
        if plain_model.training_case_predictions_ is not None:
            plain_model.training_case_predictions_.to_csv(artifact_run.path("plain_training_case_predictions.csv"), index=False)
        if plain_model.fit_diagnostics_ is not None:
            fit_diags.append(plain_model.fit_diagnostics_)

    if "ResponseAwareRegression" in args.scenarios:
        # Use a fresh feature builder to avoid accidental state coupling.
        feature_builder_resp = FeatureBuilder(min_group_cases=int(args.feature_min_group_cases))
        response_aware_model = ResponseAwareRegressionModel(
            response=response,
            feature_builder=feature_builder_resp,
            w_max=float(args.w_max),
            fit_threads=int(args.fit_threads),
            fit_time_limit_seconds=int(args.fit_time_limit_seconds),
            fit_mip_gap=float(args.fit_mip_gap),
            fit_verbose=bool(args.verbose),
            require_exact_fit=bool(args.require_exact_response_aware_fit),
            enable_smooth_warm_start=not bool(args.disable_smooth_warm_start),
            warm_start_n_starts=int(args.warm_start_n_starts),
            warm_start_maxiter=int(args.warm_start_maxiter),
            warm_start_seed=int(args.warm_start_seed),
            warm_start_ridge_lambda=float(args.warm_start_ridge_lambda),
            warm_start_huber_delta=float(args.warm_start_huber_delta),
        )
        resp_manifest_path = "response_aware_feature_manifest.json"
        response_aware_model.fit(scope_bundle.df_warmup_scoped, feature_manifest_path=resp_manifest_path)
        artifact_run.path(resp_manifest_path).write_text(json.dumps(asdict(feature_builder_resp.feature_manifest), indent=2), encoding="utf-8")
        pd.DataFrame({"feature": feature_builder_resp.feature_names, "weight": response_aware_model.weights_.tolist()}).to_csv(
            artifact_run.path("response_aware_feature_weights.csv"), index=False
        )
        if response_aware_model.training_case_predictions_ is not None:
            response_aware_model.training_case_predictions_.to_csv(artifact_run.path("response_aware_training_case_predictions.csv"), index=False)
        if response_aware_model.fit_diagnostics_ is not None:
            fit_diags.append(response_aware_model.fit_diagnostics_)

    artifact_run.path("fit_diagnostics.json").write_text(
        json.dumps([asdict(d) for d in fit_diags], indent=2, default=str),
        encoding="utf-8",
    )

    scenario_writer = JsonlWriter(artifact_run.path("jsonl", "scenario_results.jsonl"))
    assignment_writer = JsonlWriter(artifact_run.path("jsonl", "assignment_results.jsonl"))
    block_writer = JsonlWriter(artifact_run.path("jsonl", "block_results.jsonl"))
    case_writer = JsonlWriter(artifact_run.path("jsonl", "case_results.jsonl"))
    failure_writer = JsonlWriter(artifact_run.path("jsonl", "failures.jsonl"))

    all_scenario_rows: list[dict[str, Any]] = []
    all_assignment_rows: list[dict[str, Any]] = []
    all_block_rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    workers = max(1, min(int(args.workers), len(instances) if instances else 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                solve_one_week,
                inst,
                scope_bundle=scope_bundle,
                config=cfg,
                active_scenarios=args.scenarios,
                plain_model=plain_model,
                response_aware_model=response_aware_model,
            ): inst.week_index
            for inst in instances
        }
        for future in as_completed(futures):
            week_idx = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
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

            scenario_writer.write_many(payload.scenario_rows)
            assignment_writer.write_many(payload.assignment_rows)
            block_writer.write_many(payload.block_rows)
            case_writer.write_many(payload.case_rows)
            failure_writer.write_many(payload.failures)

            all_scenario_rows.extend(payload.scenario_rows)
            all_assignment_rows.extend(payload.assignment_rows)
            all_block_rows.extend(payload.block_rows)
            all_case_rows.extend(payload.case_rows)
            all_failures.extend(payload.failures)

    scenario_writer.close()
    assignment_writer.close()
    block_writer.close()
    case_writer.close()
    failure_writer.close()

    results_df = pd.DataFrame(all_scenario_rows).sort_values(["week", "scenario"]).reset_index(drop=True)
    assignments_df = pd.DataFrame(all_assignment_rows).sort_values(["week", "scenario", "case_id"]).reset_index(drop=True)
    blocks_df = pd.DataFrame(all_block_rows).sort_values(["week", "scenario", "day_index", "site", "room"]).reset_index(drop=True)
    cases_df = pd.DataFrame(all_case_rows).sort_values(["week", "scenario", "case_id"]).reset_index(drop=True)
    failures_df = pd.DataFrame(all_failures)

    results_df.to_csv(artifact_run.path("scenario_results.csv"), index=False)
    assignments_df.to_csv(artifact_run.path("assignment_results.csv"), index=False)
    blocks_df.to_csv(artifact_run.path("block_results.csv"), index=False)
    cases_df.to_csv(artifact_run.path("case_results.csv"), index=False)
    failures_df.to_csv(artifact_run.path("failures.csv"), index=False)

    scenario_summary = summarize_scenarios(results_df)
    scenario_summary.to_csv(artifact_run.path("scenario_summary.csv"), index=False)

    pairwise_weekly, pairwise_summary = build_pairwise_weekly_deltas(results_df, args.scenarios)
    pairwise_weekly.to_csv(artifact_run.path("pairwise_weekly_deltas.csv"), index=False)
    pairwise_summary.to_csv(artifact_run.path("pairwise_summary.csv"), index=False)

    report_text = render_text_report(
        args=args,
        load_summary=load_summary,
        build_meta=build_meta,
        recode_info=recode_info,
        scope_bundle=scope_bundle,
        fit_diagnostics=fit_diags,
        scenario_summary=scenario_summary,
        pairwise_summary=pairwise_summary,
    )
    artifact_run.path("report.txt").write_text(report_text, encoding="utf-8")

    print(f"\nRun directory: {run_dir}")
    print("Artifacts saved:")
    print("  scenario_results.csv")
    print("  scenario_summary.csv")
    print("  pairwise_weekly_deltas.csv")
    print("  pairwise_summary.csv")
    print("  case_results.csv")
    print("  assignment_results.csv")
    print("  block_results.csv")
    print("  report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
