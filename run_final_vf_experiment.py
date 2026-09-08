#!/usr/bin/env python3
"""
FINAL full-scale experiment for the surgeon-in-the-loop OR planning paper.

This script is intentionally self-contained at the experiment level. It imports
only the stable project modules under src/ for canonical data cleaning,
capacity/eligibility construction, weekly instance types, and schedule cost
calculation. It does NOT import any previous experiment driver, gate, patch, or
artifact.

What it does
------------
1. Freezes the canonical 72-train / 22-holdout TGH data split and writes data
   fingerprints. By default it aborts if the current stable pipeline does not
   reproduce 9,289 training cases and 2,747 holdout cases.
2. Solves realized-duration oracle weeks with long, symmetry-broken MILPs and
   stores both incumbent and ObjBound. The lower bound is used anywhere a
   regret certificate needs a safe oracle constant.
3. Fits clean baselines from scratch: Naive prediction, response-aware (RA),
   and oracle-schedule (OS).
4. Trains the final schedule-library VF method from scratch with exact
   finite-library rescoring and pDCA policy updates. The policy-facing weekly
   MILPs use a 1% target by default so the D-versus-epsilon diagnostic is
   meaningful early in the run.
5. Runs a reachable-box saturation stress test. With alpha=.8 and h=30, the
   implemented correction is always in [-24,+24]. The test samples that full
   Cartesian box and checks whether new optimal/near-optimal schedule surfaces
   appear outside the learned library. THIS IS EMPIRICAL SATURATION EVIDENCE,
   NOT A MATHEMATICAL PROOF OF GLOBAL LIBRARY COMPLETENESS.
6. Only after training is frozen, materializes the 22-week holdout and performs
   a strong a-posteriori realized-cost/regret evaluation for Booked, Naive, RA,
   OS, and VF. No theta/envelope term is used in the reported realized regret.
7. Writes the decomposition that matters for the paper: case envelope term,
   schedule/value-function term, L1 term, and total trainable certificate.

Default frozen design
---------------------
- TGH only; weekdays Mon-Fri.
- 72 training weeks, final 22 eligible weeks held out.
- Minimum 50 cases/week.
- p=111 frozen case-local feature schema.
- alpha=.8, h=30 => reachable implemented correction [-24,+24] min.
- overtime=15/min, idle=10/min, opening=2000, capacity=480, turnover=0.
- coefficient box +/-25, display cap +/-180.
- one common L1 scale: lambda = .01 * A(0) / p, intercept unpenalized.

Recommended overnight run
-------------------------
caffeinate -i python run_final_vf_experiment.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_vf_experiment \
  --cores 15 \
  --max-wall-minutes 720

Main outputs
------------
DATA_FREEZE.json
WEEK_SPLIT.csv
ORACLE_TRAIN.csv
TRAIN_DECOMPOSITION.csv
VF_TRAJECTORY.csv
LIBRARY_SUMMARY.csv
SATURATION_SUMMARY.csv
FINAL_HOLDOUT_WEEKLY.csv
FINAL_HOLDOUT_SUMMARY.csv
FINAL_DECISION.json
REPORT.md
POLICIES.npz
run.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB, quicksum
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.core.column import ScheduleColumn
from src.core.config import Config, SolverConfig
from src.core.types import BlockId, Col, Domain, WeeklyInstance
from src.data.capacity import build_candidate_pools
from src.data.eligibility import build_eligibility_maps
from src.data.loader import load_data
from src.data.scope import apply_experiment_scope
from src.planning.instance import build_weekly_instance

try:
    from src.solvers.deterministic import solve_pricing as stable_solve_pricing
except Exception:
    stable_solve_pricing = None

SCRIPT_VERSION = "final_vf_experiment_2026_09_08_v1"
LOG = logging.getLogger("final_vf_experiment")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Settings:
    data: str
    artifact_root: str
    cores: int = 15
    max_wall_minutes: float = 720.0
    final_reserve_minutes: float = 240.0
    verbose: bool = False
    random_seed: int = 42

    site: str = "TGH"
    min_cases_per_week: int = 50
    train_weeks: int = 72
    holdout_weeks: int = 22
    expected_train_cases: int = 9289
    expected_holdout_cases: int = 2747
    strict_data_freeze: bool = True

    alpha: float = 0.8
    h: float = 30.0
    overtime: float = 15.0
    idle: float = 10.0
    opening: float = 2000.0
    capacity: float = 480.0
    turnover: float = 0.0
    coefficient_bound: float = 25.0
    display_cap: float = 180.0
    l1_eta: float = 0.01

    # Long realized-duration oracle solves.
    oracle_seconds: int = 1800
    oracle_gap: float = 1e-6
    oracle_numeric_tol: float = 1e-5

    # Planner used in training VF outer loops.
    train_planner_seconds: int = 300
    train_planner_gap: float = 0.01
    seed_planner_seconds: int = 60
    seed_planner_gap: float = 0.05

    # Fixed-schedule pDCA.
    pdca_convex_seconds: int = 180
    pdca_fixed_max_iterations: int = 30
    pdca_vf_inner_iterations: int = 3
    pdca_initial_gamma: float = 1.0
    pdca_min_gamma: float = 1e-4
    pdca_gamma_factor: float = 0.5
    pdca_stabilization_tol: float = 1e-3
    pdca_step_tol: float = 1e-5
    pdca_obj_tol: float = 1e-6
    pdca_backtracks: int = 2

    # VF outer search.
    vf_outer: int = 6
    vf_stop_relative_certificate: float = 1e-4
    vf_stop_schedule_changes: int = 2
    vf_stop_patience: int = 2

    # Saturation test on the full implementable duration box.
    saturation_draws_per_week: int = 64
    saturation_batch: int = 16
    saturation_patience_batches: int = 3
    saturation_min_draws_per_week: int = 32
    saturation_seconds: int = 30
    saturation_gap: float = 0.005

    # Strong final evaluation.
    final_planner_seconds: int = 1200
    final_planner_gap: float = 5e-4
    final_tiebreak_seconds: int = 120
    paper_ready_planner_gap: float = 0.001
    paper_ready_oracle_gap: float = 0.001

    stable_solver_audit_weeks: int = 2

    def validate(self) -> None:
        if self.site.upper() != "TGH":
            raise ValueError("Final experiment is frozen to TGH.")
        if self.turnover != 0.0:
            raise ValueError("Final experiment requires turnover=0.")
        if self.train_weeks != 72 or self.holdout_weeks != 22:
            raise ValueError("Final split is frozen to 72 train / 22 holdout weeks.")
        if abs(self.alpha - 0.8) > 1e-12 or abs(self.h - 30.0) > 1e-12:
            raise ValueError("Final response is frozen to alpha=.8, h=30.")
        if self.cores <= 0:
            raise ValueError("cores must be positive")
        if self.max_wall_minutes <= self.final_reserve_minutes:
            raise ValueError("max wall must exceed final reserve")


# =============================================================================
# Utilities / artifacts
# =============================================================================

def setup_logging(root: Path, verbose: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(sh)
    fh = logging.FileHandler(root / "run.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


def json_default(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Timestamp):
        return str(x)
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def rel_gap(ub: float, lb: float) -> float:
    if not np.isfinite(ub) or not np.isfinite(lb):
        return math.inf
    return max(0.0, float(ub - lb) / max(1.0, abs(float(ub))))


def remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def theta(x: np.ndarray | float, co: float, cu: float) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return co * np.maximum(a, 0.0) + cu * np.maximum(-a, 0.0)


# =============================================================================
# Frozen feature schema
# =============================================================================

PILOT_FEATURE_NAMES: tuple[str, ...] = (
    "bias", "booked_z", "sin_week", "cos_week", "sin_month", "cos_month",
    "service_GEN", "service_GYN ONC", "service_INT CARDIO", "service_OTO",
    "service_Other", "service_PACE DEFIB", "service_PLAS", "service_THOR",
    "service_TX", "service_UROL", "service_VASC", "service___OTHER__",
    "surgeon_101", "surgeon_103", "surgeon_104", "surgeon_109", "surgeon_11",
    "surgeon_111", "surgeon_112", "surgeon_117", "surgeon_120", "surgeon_123",
    "surgeon_13", "surgeon_134", "surgeon_135", "surgeon_14", "surgeon_143",
    "surgeon_145", "surgeon_147", "surgeon_150", "surgeon_151", "surgeon_156",
    "surgeon_159", "surgeon_167", "surgeon_17", "surgeon_170", "surgeon_173",
    "surgeon_185", "surgeon_20", "surgeon_201", "surgeon_21", "surgeon_23",
    "surgeon_234", "surgeon_24", "surgeon_242", "surgeon_269", "surgeon_27",
    "surgeon_28", "surgeon_30", "surgeon_32", "surgeon_33", "surgeon_34",
    "surgeon_36", "surgeon_37", "surgeon_38", "surgeon_45", "surgeon_46",
    "surgeon_47", "surgeon_48", "surgeon_50", "surgeon_55", "surgeon_56",
    "surgeon_57", "surgeon_58", "surgeon_59", "surgeon_61", "surgeon_83",
    "surgeon_84", "surgeon_88", "surgeon_89", "surgeon_95", "surgeon_96",
    "surgeon_Other", "surgeon___OTHER__",
    "procedure_G014988", "procedure_G015085", "procedure_G09120",
    "procedure_G09410", "procedure_G10390", "procedure_G10554", "procedure_G11020",
    "procedure_G11195", "procedure_G11255", "procedure_G11270", "procedure_G11290",
    "procedure_G11295", "procedure_G11300", "procedure_G11345", "procedure_G11390",
    "procedure_G11400", "procedure_G11980", "procedure_G12005", "procedure_G12160",
    "procedure_G12210", "procedure_G12580", "procedure_G130720", "procedure_G130930",
    "procedure_G131240", "procedure_G160535", "procedure_G19615", "procedure_G19970",
    "procedure_G20415", "procedure_Other", "procedure___OTHER__", "site___OTHER__",
)
assert len(PILOT_FEATURE_NAMES) == 111


class FrozenFeatureEncoder:
    PREFIX_TO_COLUMN = {
        "service": Col.CASE_SERVICE,
        "surgeon": Col.SURGEON_CODE,
        "procedure": Col.PROCEDURE_ID,
        "site": Col.SITE,
    }

    def __init__(self) -> None:
        self.feature_names = list(PILOT_FEATURE_NAMES)
        self.booked_mean = 0.0
        self.booked_std = 1.0
        self.references: dict[str, str] = {}
        self.represented: dict[str, list[str]] = {}
        self.fitted = False

    @staticmethod
    def canon(x: object) -> str:
        if pd.isna(x):
            return Domain.OTHER
        s = str(x).strip()
        return s if s and s.lower() not in {"nan", "none", "<na>"} else Domain.OTHER

    def fit(self, frame: pd.DataFrame) -> "FrozenFeatureEncoder":
        b = pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce").to_numpy(float)
        self.booked_mean = float(np.nanmean(b))
        self.booked_std = float(np.nanstd(b))
        if not np.isfinite(self.booked_std) or self.booked_std <= 1e-12:
            self.booked_std = 1.0
        for prefix, column in self.PREFIX_TO_COLUMN.items():
            marker = prefix + "_"
            explicit = [n[len(marker):] for n in self.feature_names if n.startswith(marker)]
            explicit = [v for v in explicit if v != "__OTHER__"]
            self.represented[prefix] = explicit
            values = frame[column].map(self.canon)
            omitted = values[~values.isin(explicit)]
            self.references[prefix] = str(omitted.value_counts().index[0]) if len(omitted) else "__REFERENCE__"
        self.fitted = True
        return self

    def transform_frame(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.fitted:
            raise RuntimeError("feature encoder not fitted")
        n = len(frame)
        booked = pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce").fillna(self.booked_mean).to_numpy(float)
        week = pd.to_numeric(frame[Col.WEEK_OF_YEAR], errors="coerce").fillna(1).to_numpy(float)
        month = pd.to_numeric(frame[Col.MONTH], errors="coerce").fillna(1).to_numpy(float)
        cols: list[np.ndarray] = [
            np.ones(n),
            (booked - self.booked_mean) / self.booked_std,
            np.sin(2 * np.pi * week / 52.0),
            np.cos(2 * np.pi * week / 52.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ]
        names = list(self.feature_names[:6])
        for prefix, column in self.PREFIX_TO_COLUMN.items():
            vals = frame[column].map(self.canon).to_numpy(object)
            explicit = self.represented[prefix]
            ref = self.references[prefix]
            for level in explicit:
                cols.append((vals == level).astype(float))
                names.append(f"{prefix}_{level}")
            other = (~np.isin(vals, np.asarray(explicit, dtype=object))) & (vals != ref)
            cols.append(other.astype(float))
            names.append(f"{prefix}___OTHER__")
        if names != self.feature_names:
            raise AssertionError("frozen feature schema reconstruction failed")
        return sparse.csr_matrix(np.column_stack(cols), dtype=float)

    def transform_cases(self, cases: Sequence[Any]) -> sparse.csr_matrix:
        f = pd.DataFrame({
            Col.BOOKED_MINUTES: [float(c.booked_duration_min) for c in cases],
            Col.CASE_SERVICE: [str(c.service) for c in cases],
            Col.SURGEON_CODE: [str(c.surgeon_code) for c in cases],
            Col.PROCEDURE_ID: [str(c.procedure_id) for c in cases],
            Col.SITE: [str(c.site) for c in cases],
            Col.WEEK_OF_YEAR: [int(c.week_of_year) for c in cases],
            Col.MONTH: [int(c.month) for c in cases],
        })
        return self.transform_frame(f)

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "booked_mean": self.booked_mean,
            "booked_std": self.booked_std,
            "references": self.references,
            "represented": self.represented,
            "p": len(self.feature_names),
        }


# =============================================================================
# Data and week objects
# =============================================================================

@dataclass
class WeekBundle:
    position: int
    start: pd.Timestamp
    instance: WeeklyInstance


@dataclass
class Arrays:
    X: sparse.csr_matrix
    booked: np.ndarray
    actual: np.ndarray
    error: np.ndarray
    week_ids: np.ndarray
    case_ids: np.ndarray
    week_slices: dict[int, np.ndarray]

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_weeks(self) -> int:
        return len(self.week_slices)


def build_config(s: Settings) -> Config:
    c = Config()
    c.data.excel_file_path = s.data
    c.scope.planning_sites = (s.site.upper(),)
    c.scope.planning_weekdays = (0, 1, 2, 3, 4)
    c.capacity.block_capacity_minutes = s.capacity
    c.capacity.activation_cost_per_block = s.opening
    c.capacity.turnover_minutes = s.turnover
    c.costs.overtime_per_minute = s.overtime
    c.costs.idle_per_minute = s.idle
    c.costs.deferral_per_case = 1e12
    c.solver.verbose = False
    return c


def week_start_series(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce").dt.normalize()
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def choose_split(df_scoped: pd.DataFrame, s: Settings) -> tuple[list[pd.Timestamp], list[pd.Timestamp], pd.DataFrame]:
    w = df_scoped.copy()
    w["_week_start"] = week_start_series(w)
    counts = w.groupby("_week_start").size().sort_index()
    eligible = counts[counts >= s.min_cases_per_week]
    needed = s.train_weeks + s.holdout_weeks
    if len(eligible) < needed:
        raise RuntimeError(f"Only {len(eligible)} eligible weeks; need {needed}")
    selected = eligible.iloc[-needed:]
    train = [pd.Timestamp(x) for x in selected.index[:s.train_weeks]]
    hold = [pd.Timestamp(x) for x in selected.index[s.train_weeks:]]
    if hold[0] != pd.Timestamp("2013-01-28"):
        raise RuntimeError(f"Holdout start changed: {hold[0].date()} != 2013-01-28")
    audit = counts.rename("n_cases").reset_index().rename(columns={"_week_start": "week_start"})
    audit["eligible"] = audit["n_cases"] >= s.min_cases_per_week
    audit["split"] = np.where(
        audit["week_start"].isin(train), "train",
        np.where(audit["week_start"].isin(hold), "holdout", "unused"),
    )
    return train, hold, audit


def build_bundles(
    df_scoped: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    *,
    cfg: Config,
    candidate_pools: Any,
    eligibility_maps: Any,
    offset: int,
) -> list[WeekBundle]:
    out: list[WeekBundle] = []
    for j, start in enumerate(starts):
        pos = offset + j
        inst = build_weekly_instance(
            df_pool=df_scoped,
            horizon_start=pd.Timestamp(start),
            week_index=pos,
            config=cfg,
            candidate_pools=candidate_pools,
            eligibility_maps=eligibility_maps,
        )
        if inst.num_cases == 0:
            raise RuntimeError(f"Empty selected week {start}")
        missing = [i for i in range(inst.num_cases) if not inst.case_eligible_blocks.get(i, [])]
        if missing:
            raise RuntimeError(f"Week {start.date()} has {len(missing)} cases with no eligible block")
        out.append(WeekBundle(pos, pd.Timestamp(start), inst))
    return out


def build_arrays(weeks: Sequence[WeekBundle], enc: FrozenFeatureEncoder) -> Arrays:
    xb: list[sparse.csr_matrix] = []
    booked, actual, week_ids, case_ids = [], [], [], []
    slices: dict[int, np.ndarray] = {}
    cursor = 0
    for w in weeks:
        xb.append(enc.transform_cases(w.instance.cases))
        b = np.asarray(w.instance.booked_durations(), float)
        a = np.asarray(w.instance.actual_durations(), float)
        booked.extend(b.tolist())
        actual.extend(a.tolist())
        week_ids.extend([w.position] * len(b))
        case_ids.extend([int(c.case_id) for c in w.instance.cases])
        slices[w.position] = np.arange(cursor, cursor + len(b), dtype=int)
        cursor += len(b)
    X = sparse.vstack(xb, format="csr")
    b = np.asarray(booked, float)
    a = np.asarray(actual, float)
    return Arrays(X, b, a, a-b, np.asarray(week_ids, int), np.asarray(case_ids, int), slices)


# =============================================================================
# Response map and policy functions
# =============================================================================

def response_value(delta: np.ndarray, s: Settings) -> np.ndarray:
    d = np.asarray(delta, float)
    m = np.abs(d)
    eps = s.h / (1.0 - s.alpha)
    impl = np.where(
        m <= s.h,
        s.alpha * m,
        np.where(m <= eps, np.maximum(s.h - (1.0-s.alpha)*m, 0.0), 0.0),
    )
    return np.sign(d) * impl


def response_dc_parts(delta: np.ndarray, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    q = 1.0 - s.alpha
    eps = s.h / q
    d = np.asarray(delta, float)
    P = np.maximum(d + s.h, 0.0) + q * np.maximum(d - eps, 0.0)
    N = q * np.maximum(d + eps, 0.0) + np.maximum(d - s.h, 0.0)
    return P, N


def hinge_slope(x: np.ndarray, kink: float, tol: float = 1e-10) -> np.ndarray:
    z = np.asarray(x, float) - float(kink)
    return np.where(z > tol, 1.0, np.where(z < -tol, 0.0, 0.5))


def response_dc_slopes(delta: np.ndarray, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    q = 1.0 - s.alpha
    eps = s.h / q
    d = np.asarray(delta, float)
    Pp = hinge_slope(d, -s.h) + q * hinge_slope(d, eps)
    Np = q * hinge_slope(d, -eps) + hinge_slope(d, s.h)
    return Pp, Np


def correction_and_planning(w: np.ndarray, a: Arrays, s: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(a.X @ np.asarray(w, float), float).reshape(-1)
    corr = response_value(delta, s)
    planning = a.booked + corr
    return delta, corr, planning


def response_regions(delta: np.ndarray, s: Settings) -> np.ndarray:
    m = np.abs(np.asarray(delta, float))
    eps = s.h / (1-s.alpha)
    return np.where(m <= s.h + 1e-9, "acceptance", np.where(m <= eps + 1e-9, "decay", "discard"))


def case_envelope(w: np.ndarray, a: Arrays, s: Settings) -> float:
    _, corr, _ = correction_and_planning(w, a, s)
    r = a.error - corr
    return float(np.sum(theta(r, s.overtime, s.idle)) / a.n_weeks)


# =============================================================================
# Clean weekly MILP with symmetry breaking and deterministic tie-break
# =============================================================================

@dataclass
class PlanResult:
    week: int
    column: ScheduleColumn
    objective: float
    bound: float
    gap: float
    status: str
    solve_seconds: float
    exact: bool
    tiebreak_used: bool = False


def status_name(st: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(st, str(st))


def block_equivalence_groups(inst: WeeklyInstance) -> list[list[BlockId]]:
    blocks = list(inst.calendar.block_ids)
    eligible_by_block: dict[BlockId, tuple[int, ...]] = {}
    for b in blocks:
        eligible_by_block[b] = tuple(sorted(i for i in range(inst.num_cases) if b in inst.case_eligible_blocks.get(i, [])))
    meta = {b.id: b for b in inst.calendar.candidates}
    groups: dict[tuple[Any, ...], list[BlockId]] = {}
    for bid in blocks:
        bb = meta[bid]
        key = (
            int(bid.day_index), str(bid.site), float(bb.capacity_minutes),
            float(bb.activation_cost), eligible_by_block[bid],
        )
        groups.setdefault(key, []).append(bid)
    return [sorted(g, key=lambda b: str(b.room)) for g in groups.values()]


def canonical_schedule_signature(col: ScheduleColumn, inst: WeeklyInstance) -> str:
    # Quotient out permutations among blocks having identical day/site/capacity/
    # activation and identical eligible-case sets. For each equivalence group we
    # retain only the sorted partition of cases across used blocks.
    groups = block_equivalence_groups(inst)
    payload_groups = []
    for group in groups:
        partitions = []
        opened = 0
        for bid in group:
            if bid in col.v_open:
                opened += 1
                partitions.append(tuple(col.cases_in_block(bid)))
        partitions = sorted(partitions)
        if opened:
            sample = group[0]
            eligible = tuple(sorted(i for i in range(inst.num_cases) if sample in inst.case_eligible_blocks.get(i, [])))
            payload_groups.append((int(sample.day_index), str(sample.site), eligible, opened, partitions))
    payload = {
        "groups": payload_groups,
        "defer": sorted(int(i) for i in col.z_defer),
        "n": int(col.n_cases),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def _build_week_model(
    week: WeekBundle,
    durations: np.ndarray,
    s: Settings,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int,
    warm: ScheduleColumn | None,
    name: str,
) -> tuple[gp.Model, dict[tuple[int, BlockId], gp.Var], dict[BlockId, gp.Var], gp.LinExpr]:
    inst = week.instance
    n = inst.num_cases
    d = np.asarray(durations, float)
    if len(d) != n:
        raise ValueError("duration length mismatch")
    m = gp.Model(name)
    m.Params.OutputFlag = 1 if s.verbose else 0
    m.Params.TimeLimit = max(1, int(time_limit))
    m.Params.MIPGap = max(0.0, float(mip_gap))
    m.Params.Threads = max(1, int(threads))
    m.Params.MIPFocus = 1
    m.Params.Symmetry = 2
    m.Params.Seed = int(s.random_seed)

    blocks = list(inst.calendar.block_ids)
    meta = {b.id: b for b in inst.calendar.candidates}
    x: dict[tuple[int, BlockId], gp.Var] = {}
    y = {bid: m.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j, bid in enumerate(blocks)}
    for i in range(n):
        elig = list(inst.case_eligible_blocks.get(i, []))
        if not elig:
            raise RuntimeError(f"Week {week.position}: case {i} has no eligible block")
        for bid in elig:
            x[i, bid] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{bid.day_index}_{bid.site}_{bid.room}")
        m.addConstr(quicksum(x[i,b] for b in elig) == 1, name=f"assign_{i}")
        for bid in elig:
            m.addConstr(x[i,bid] <= y[bid], name=f"link_{i}_{bid.day_index}_{bid.room}")

    block_cases = {bid: [i for i in range(n) if (i,bid) in x] for bid in blocks}
    ot = {bid: m.addVar(lb=0.0, name=f"ot_{j}") for j,bid in enumerate(blocks)}
    idle = {bid: m.addVar(lb=0.0, name=f"idle_{j}") for j,bid in enumerate(blocks)}
    for bid in blocks:
        inds = block_cases[bid]
        if not inds:
            m.addConstr(y[bid] == 0)
            continue
        load = quicksum(float(d[i]) * x[i,bid] for i in inds)
        cap = float(meta[bid].capacity_minutes) * y[bid]
        m.addConstr(ot[bid] >= load - cap)
        m.addConstr(idle[bid] >= cap - load)

    # Safe symmetry breaking among truly interchangeable blocks.
    for group in block_equivalence_groups(inst):
        if len(group) <= 1:
            continue
        for a, b in zip(group[:-1], group[1:]):
            m.addConstr(y[a] >= y[b])
            ia = [i for i in range(n) if (i,a) in x]
            ib = [i for i in range(n) if (i,b) in x]
            common = sorted(set(ia) & set(ib))
            if common:
                wa = quicksum((i+1) * x[i,a] for i in common)
                wb = quicksum((i+1) * x[i,b] for i in common)
                m.addConstr(wa >= wb)

    objective = (
        quicksum(float(meta[b].activation_cost) * y[b] for b in blocks)
        + s.overtime * quicksum(ot.values())
        + s.idle * quicksum(idle.values())
    )
    if warm is not None:
        for (i,bid), var in x.items():
            var.Start = 1.0 if warm.z_assign.get((i,bid), 0.0) > 0.5 else 0.0
        for bid, var in y.items():
            var.Start = 1.0 if bid in warm.v_open else 0.0
    return m, x, y, objective


def extract_column(week: WeekBundle, x: Mapping[tuple[int,BlockId], gp.Var], y: Mapping[BlockId,gp.Var]) -> ScheduleColumn:
    inst = week.instance
    z_assign = {(i,b):1.0 for (i,b),v in x.items() if v.X > 0.5}
    v_open = frozenset(b for b,v in y.items() if v.X > 0.5)
    y_used = frozenset(b for b in v_open if any(j==i and bb==b and val>0.5 for (j,bb), val in z_assign.items() for i in [j]))
    # simpler / deterministic y_used construction
    y_used = frozenset(b for b in v_open if any(bb == b and val > 0.5 for (_,bb), val in z_assign.items()))
    cap = {b.id: float(b.capacity_minutes) for b in inst.calendar.candidates}
    act = {b.id: float(b.activation_cost) for b in inst.calendar.candidates}
    return ScheduleColumn(
        z_assign=z_assign,
        z_defer=frozenset(),
        v_open=v_open,
        y_used=y_used,
        n_cases=inst.num_cases,
        block_capacities=cap,
        block_activation_costs=act,
    )


def solve_week(
    week: WeekBundle,
    durations: np.ndarray,
    s: Settings,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int = 1,
    warm: ScheduleColumn | None = None,
    label: str = "plan",
    deterministic_tiebreak: bool = False,
    tiebreak_seconds: int | None = None,
) -> PlanResult:
    t0 = time.perf_counter()
    m, x, y, obj = _build_week_model(
        week, durations, s, time_limit=time_limit, mip_gap=mip_gap,
        threads=threads, warm=warm, name=f"{label}_w{week.position}"
    )
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount <= 0:
        st = status_name(m.Status)
        m.dispose()
        raise RuntimeError(f"Week {week.position} {label}: no feasible solution ({st})")
    ub = float(m.ObjVal)
    lb = float(m.ObjBound)
    gap = rel_gap(ub,lb)
    st = status_name(m.Status)
    exact = bool(m.Status == GRB.OPTIMAL and abs(ub-lb) <= s.oracle_numeric_tol * max(1.0, abs(ub)))
    col = extract_column(week,x,y)
    pstar = ub
    m.dispose()
    used_tb = False

    if deterministic_tiebreak and exact:
        tb_sec = int(tiebreak_seconds or min(120, max(10, time_limit//5)))
        m2, x2, y2, obj2 = _build_week_model(
            week, durations, s, time_limit=tb_sec, mip_gap=0.0,
            threads=threads, warm=col, name=f"{label}_tb_w{week.position}"
        )
        tol = max(1e-6, 1e-8 * max(1.0, abs(pstar)))
        m2.addConstr(obj2 <= pstar + tol, name="primary_optimality_band")
        blocks = list(week.instance.calendar.block_ids)
        block_pos = {b:j for j,b in enumerate(blocks)}
        # Data-independent deterministic secondary score.
        secondary = quicksum((i+1)*(block_pos[b]+1)*var for (i,b),var in x2.items()) \
                    + (week.instance.num_cases+1) * quicksum((block_pos[b]+1)*var for b,var in y2.items())
        m2.setObjective(secondary, GRB.MINIMIZE)
        m2.optimize()
        if m2.SolCount > 0:
            c2 = extract_column(week,x2,y2)
            c2_cost = float(c2.compute_cost(np.asarray(durations,float), _cost_cfg(s), s.turnover))
            if c2_cost <= pstar + 10*tol:
                col = c2
                used_tb = True
        m2.dispose()

    elapsed = time.perf_counter() - t0
    return PlanResult(week.position,col,ub,lb,gap,st,elapsed,exact,used_tb)


def _cost_cfg(s: Settings):
    c = Config().costs
    c.overtime_per_minute = s.overtime
    c.idle_per_minute = s.idle
    c.deferral_per_case = 1e12
    return c


def solve_batch(
    weeks: Sequence[WeekBundle],
    duration_by_week: Mapping[int,np.ndarray],
    s: Settings,
    *,
    seconds: int,
    gap: float,
    label: str,
    warm_by_week: Mapping[int,ScheduleColumn] | None = None,
    deterministic_tiebreak: bool = False,
    tiebreak_seconds: int | None = None,
) -> dict[int,PlanResult]:
    out: dict[int,PlanResult] = {}
    workers = min(s.cores, len(weeks))
    LOG.info("[PLANNER] %s | weeks=%d workers=%d limit=%ss gap=%.4g", label,len(weeks),workers,seconds,gap)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {
            ex.submit(
                solve_week,w,np.asarray(duration_by_week[w.position],float),s,
                time_limit=seconds,mip_gap=gap,threads=1,
                warm=None if warm_by_week is None else warm_by_week.get(w.position),
                label=label,deterministic_tiebreak=deterministic_tiebreak,
                tiebreak_seconds=tiebreak_seconds,
            ): w.position for w in weeks
        }
        done = 0
        for f in as_completed(fut):
            pos = fut[f]
            out[pos] = f.result()
            done += 1
            if done == 1 or done % 10 == 0 or done == len(fut):
                gs = [r.gap for r in out.values() if np.isfinite(r.gap)]
                LOG.info("[PLANNER] %s %d/%d | elapsed=%.1f min | maxgap=%s",
                         label,done,len(fut),(time.perf_counter()-t0)/60,
                         "NA" if not gs else f"{max(gs):.3%}")
    return out


def solve_oracle_batch(
    weeks: Sequence[WeekBundle],
    duration_by_week: Mapping[int,np.ndarray],
    s: Settings,
    *,
    label: str,
) -> dict[int,PlanResult]:
    """Two-stage oracle solve: a shorter first pass, then long retries only for unresolved weeks."""
    first=min(600,int(s.oracle_seconds))
    out=solve_batch(weeks,duration_by_week,s,seconds=first,gap=s.oracle_gap,label=f"{label}_pass1")
    unresolved=[w for w in weeks if not out[w.position].exact and out[w.position].gap > s.oracle_numeric_tol]
    if unresolved and s.oracle_seconds>first:
        LOG.info("[ORACLE] %s retrying %d unresolved weeks up to %ss",label,len(unresolved),s.oracle_seconds)
        warm={w.position:out[w.position].column for w in unresolved}
        subset_map={w.position:duration_by_week[w.position] for w in unresolved}
        retry=solve_batch(unresolved,subset_map,s,seconds=s.oracle_seconds,gap=s.oracle_gap,label=f"{label}_retry",warm_by_week=warm)
        for w in unresolved:
            old=out[w.position]; new=retry[w.position]
            # Keep the better incumbent and the stronger valid lower bound. Usually
            # both come from the retry; this merge is safe even if a retry is interrupted.
            best_col=new.column if new.objective <= old.objective+1e-9 else old.column
            ub=min(old.objective,new.objective); lb=max(old.bound,new.bound)
            g=rel_gap(ub,lb); exact=bool(g <= s.oracle_numeric_tol)
            out[w.position]=PlanResult(w.position,best_col,ub,lb,g,new.status,new.solve_seconds+old.solve_seconds,exact,new.tiebreak_used)
    return out


# =============================================================================
# Library
# =============================================================================

@dataclass
class LibraryEntry:
    week: int
    signature: str
    source: str
    column: ScheduleColumn


class ScheduleLibrary:
    def __init__(self, week_lookup: Mapping[int,WeekBundle]):
        self.week_lookup = dict(week_lookup)
        self.data: dict[int,dict[str,LibraryEntry]] = {}
        self.pinned: dict[int,set[str]] = {}

    def add(self, week: int, col: ScheduleColumn, source: str, *, pin: bool=False) -> bool:
        sig = canonical_schedule_signature(col,self.week_lookup[int(week)].instance)
        bucket = self.data.setdefault(int(week),{})
        if sig in bucket:
            if pin:
                self.pinned.setdefault(int(week),set()).add(sig)
            return False
        bucket[sig] = LibraryEntry(int(week),sig,str(source),col)
        if pin:
            self.pinned.setdefault(int(week),set()).add(sig)
        return True

    def add_plans(self, plans: Mapping[int,PlanResult], source: str, *, pin: bool=False) -> int:
        return sum(int(self.add(w,r.column,source,pin=pin)) for w,r in plans.items())

    def entries(self, week: int) -> list[LibraryEntry]:
        return list(self.data.get(int(week),{}).values())

    def size(self, week: int | None=None) -> int:
        if week is not None:
            return len(self.data.get(int(week),{}))
        return sum(len(v) for v in self.data.values())

    def best(self, week: int, durations: np.ndarray, s: Settings) -> tuple[ScheduleColumn,float,str]:
        best_col=None; best=math.inf; src=""
        for e in self.entries(week):
            val=float(e.column.compute_cost(np.asarray(durations,float),_cost_cfg(s),s.turnover))
            if val < best - 1e-9:
                best=val; best_col=e.column; src=e.source
        if best_col is None:
            raise RuntimeError(f"empty library week {week}")
        return best_col,float(best),src

    def check_pins(self) -> bool:
        for w,sigs in self.pinned.items():
            if not sigs.issubset(set(self.data.get(w,{}))):
                return False
        return True

    def summary_rows(self) -> list[dict[str,Any]]:
        out=[]
        for w in sorted(self.data):
            srcs={}
            for e in self.entries(w): srcs[e.source]=srcs.get(e.source,0)+1
            out.append({"week":w,"n_columns":len(self.data[w]),"pinned":len(self.pinned.get(w,set())),"sources":json.dumps(srcs,sort_keys=True)})
        return out


# =============================================================================
# RA exposure probabilities
# =============================================================================

def exposure_labels(booked_plans: Mapping[int,PlanResult], a: Arrays, weeks: Mapping[int,WeekBundle], s: Settings) -> np.ndarray:
    y=np.zeros(len(a.booked),dtype=int)
    for w,idx in a.week_slices.items():
        col=booked_plans[w].column
        b=a.booked[idx]
        loads=col.compute_block_load(b,s.turnover)
        tight={bid: (load >= col.block_capacities[bid]-1e-9) for bid,load in loads.items()}
        for local_i,global_i in enumerate(idx):
            bid=next((bb for (j,bb),v in col.z_assign.items() if j==local_i and v>0.5),None)
            y[global_i]=1 if bid is not None and tight.get(bid,False) else 0
    return y


def crossfit_pi(a: Arrays, labels: np.ndarray, s: Settings) -> tuple[np.ndarray,dict[str,Any]]:
    X=sparse.csr_matrix(a.X,float); y=np.asarray(labels,int); groups=a.week_ids
    pred=np.zeros(len(y),float)
    k=min(5,len(np.unique(groups)))
    splitter=GroupKFold(n_splits=k)
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups),start=1):
        if np.unique(y[tr]).size<2:
            pred[te]=float(np.mean(y[tr]))
        else:
            mdl=LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=2000,fit_intercept=False,random_state=s.random_seed+fold)
            mdl.fit(X[tr],y[tr]); pred[te]=mdl.predict_proba(X[te])[:,1]
    pred=np.clip(pred,1e-6,1-1e-6)
    prev=float(np.mean(y))
    metrics={
        "prevalence":prev,"pi_mean":float(np.mean(pred)),
        "brier":float(brier_score_loss(y,pred)),
        "constant_brier":float(np.mean((y-prev)**2)),
        "auc":float(roc_auc_score(y,pred)) if np.unique(y).size==2 else math.nan,
    }
    return pred,metrics


# =============================================================================
# Convex prediction baseline
# =============================================================================

def train_naive(a: Arrays, s: Settings, lam: float) -> np.ndarray:
    n,p=a.X.shape; X=a.X.tocsr(); m=gp.Model("naive_lad")
    m.Params.OutputFlag=1 if s.verbose else 0; m.Params.Threads=s.cores; m.Params.TimeLimit=900
    w=m.addVars(p,lb=-s.coefficient_bound,ub=s.coefficient_bound,name="w")
    absr=m.addVars(n,lb=0.0,name="absr")
    absw=m.addVars(range(1,p),lb=0.0,name="absw")
    for i in range(n):
        row=X.getrow(i); expr=gp.LinExpr(row.data.tolist(),[w[int(j)] for j in row.indices])
        m.addConstr(absr[i] >= float(a.error[i]) - expr)
        m.addConstr(absr[i] >= -float(a.error[i]) + expr)
        m.addConstr(expr <= s.display_cap); m.addConstr(expr >= -s.display_cap)
    for j in range(1,p):
        m.addConstr(absw[j]>=w[j]); m.addConstr(absw[j]>=-w[j])
    obj=quicksum(absr[i] for i in range(n))/a.n_weeks + lam*quicksum(absw[j] for j in range(1,p))
    m.setObjective(obj,GRB.MINIMIZE); m.optimize()
    if m.SolCount<=0: raise RuntimeError("naive LAD failed")
    out=np.array([w[j].X for j in range(p)],float); m.dispose(); return out


# =============================================================================
# Fixed-schedule DC surrogate + pDCA
# =============================================================================

@dataclass
class FixedSpec:
    name: str
    arrays: Arrays
    p_plus: np.ndarray
    p_minus: np.ndarray
    fixed: dict[int,ScheduleColumn] | None
    lam: float
    settings: Settings

    def value(self,w:np.ndarray) -> float:
        d,c,planning=correction_and_planning(w,self.arrays,self.settings)
        r=self.arrays.error-c
        case=float(np.sum(self.p_plus*np.maximum(r,0)+self.p_minus*np.maximum(-r,0))/self.arrays.n_weeks)
        sched=0.0
        if self.fixed is not None:
            for wk,idx in self.arrays.week_slices.items():
                sched += self.fixed[wk].compute_cost(planning[idx],_cost_cfg(self.settings),self.settings.turnover)
            sched /= self.arrays.n_weeks
        l1=self.lam*float(np.sum(np.abs(np.asarray(w)[1:])))
        return case+sched+l1

    def h_subgradient_w(self,w:np.ndarray) -> np.ndarray:
        delta=np.asarray(self.arrays.X@w,float).reshape(-1)
        Pp,Np=response_dc_slopes(delta,self.settings)
        wp=np.asarray(self.p_plus,float).copy(); wn=np.asarray(self.p_minus,float).copy()
        if self.fixed is not None:
            # Schedule H contributes idle*P + overtime*N for each scheduled case.
            scheduled=np.zeros(len(delta),bool)
            for wk,idx in self.arrays.week_slices.items():
                col=self.fixed[wk]
                for local_i,global_i in enumerate(idx):
                    if any(j==local_i and v>0.5 for (j,_),v in col.z_assign.items()):
                        scheduled[global_i]=True
            wp += self.settings.idle*scheduled.astype(float)
            wn += self.settings.overtime*scheduled.astype(float)
        q_delta=(wp*Pp+wn*Np)/self.arrays.n_weeks
        return np.asarray(self.arrays.X.T@q_delta,float).reshape(-1)


class ConvexPDCASubproblem:
    def __init__(self,spec:FixedSpec,s:Settings,name:str):
        self.spec=spec; self.s=s; a=spec.arrays; n,p=a.X.shape; X=a.X.tocsr()
        m=gp.Model(name); m.Params.OutputFlag=1 if s.verbose else 0; m.Params.Threads=s.cores; m.Params.TimeLimit=s.pdca_convex_seconds; m.Params.Method=2
        self.m=m; self.w=m.addVars(p,lb=-s.coefficient_bound,ub=s.coefficient_bound,name="w")
        ph=m.addVars(n,lb=0,name="ph"); pe=m.addVars(n,lb=0,name="pe"); ne=m.addVars(n,lb=0,name="ne"); nh=m.addVars(n,lb=0,name="nh")
        R=m.addVars(n,lb=0,name="R"); q=1-s.alpha; eps=s.h/q
        self.P={}; self.N={}; self.delta={}
        for i in range(n):
            row=X.getrow(i); de=gp.LinExpr(row.data.tolist(),[self.w[int(j)] for j in row.indices]); self.delta[i]=de
            m.addConstr(de<=s.display_cap); m.addConstr(de>=-s.display_cap)
            m.addConstr(ph[i]>=de+s.h); m.addConstr(pe[i]>=de-eps); m.addConstr(ne[i]>=de+eps); m.addConstr(nh[i]>=de-s.h)
            P=ph[i]+q*pe[i]; N=q*ne[i]+nh[i]; self.P[i]=P; self.N[i]=N
            m.addConstr(R[i]>=float(a.error[i])+N); m.addConstr(R[i]>=P)
        G=quicksum(float(spec.p_plus[i]+spec.p_minus[i])*R[i] for i in range(n))/a.n_weeks
        if spec.fixed is not None:
            schedG=gp.LinExpr()
            for wk,idx in a.week_slices.items():
                col=spec.fixed[wk]
                for bid in col.v_open:
                    locals_=col.cases_in_block(bid); globals_=[int(idx[i]) for i in locals_]
                    pb=quicksum(self.P[g] for g in globals_) if globals_ else gp.LinExpr(0.0)
                    nb=quicksum(self.N[g] for g in globals_) if globals_ else gp.LinExpr(0.0)
                    kappa=sum(float(a.booked[g]) for g in globals_)-float(col.block_capacities[bid])
                    go=m.addVar(lb=-GRB.INFINITY,name=f"go_{wk}_{len(locals_)}_{len(m.getVars())}")
                    gi=m.addVar(lb=-GRB.INFINITY,name=f"gi_{wk}_{len(locals_)}_{len(m.getVars())}")
                    m.addConstr(go>=kappa+pb); m.addConstr(go>=nb)
                    m.addConstr(gi>=-kappa+nb); m.addConstr(gi>=pb)
                    schedG += s.overtime*go+s.idle*gi+float(col.block_activation_costs.get(bid,0.0))
            G += schedG/a.n_weeks
        self.absw=m.addVars(range(1,p),lb=0,name="absw")
        for j in range(1,p): m.addConstr(self.absw[j]>=self.w[j]); m.addConstr(self.absw[j]>=-self.w[j])
        G += spec.lam*quicksum(self.absw[j] for j in range(1,p))
        self.G=G; self.p=p

    def solve(self,current:np.ndarray,q_w:np.ndarray,gamma:float) -> np.ndarray:
        linear=quicksum(-float(q_w[j]+gamma*current[j])*self.w[j] for j in range(self.p))
        quad=0.5*float(gamma)*quicksum(self.w[j]*self.w[j] for j in range(self.p))
        self.m.setObjective(self.G+linear+quad,GRB.MINIMIZE)
        for j in range(self.p): self.w[j].Start=float(current[j])
        self.m.optimize()
        if self.m.SolCount<=0: raise RuntimeError(f"pDCA convex subproblem {self.m.ModelName} produced no solution")
        return np.array([self.w[j].X for j in range(self.p)],float)

    def dispose(self): self.m.dispose()


def project_policy(w:np.ndarray,a:Arrays,s:Settings) -> np.ndarray:
    w=np.clip(np.asarray(w,float),-s.coefficient_bound,s.coefficient_bound)
    d=np.asarray(a.X@w,float).reshape(-1); mx=float(np.max(np.abs(d))) if len(d) else 0
    if mx>s.display_cap: w=w*(s.display_cap/mx)
    return w


def run_pdca(spec:FixedSpec,initial:np.ndarray,s:Settings,*,max_iterations:int) -> tuple[np.ndarray,list[dict[str,Any]]]:
    cur=project_policy(initial,spec.arrays,s); curv=spec.value(cur); gamma=s.pdca_initial_gamma; hist=[]
    solver=ConvexPDCASubproblem(spec,s,f"pdca_{spec.name}")
    try:
        for it in range(1,max_iterations+1):
            q=spec.h_subgradient_w(cur); accepted=False; cand=cur.copy(); candv=curv; used=gamma
            for bt in range(s.pdca_backtracks+1):
                cand=solver.solve(cur,q,used); cand=project_policy(cand,spec.arrays,s); candv=spec.value(cand)
                if candv<=curv+1e-7*max(1.0,abs(curv)):
                    accepted=True; break
                used*=2.0
            if not accepted: break
            step=float(np.max(np.abs(cand-cur))/max(1.0,np.max(np.abs(cur))))
            rel=(curv-candv)/max(1.0,abs(curv))
            hist.append({"iteration":it,"before":curv,"after":candv,"relative_improvement":rel,"relative_step":step,"gamma":used})
            cur,curv=cand,candv; gamma=used
            if step<s.pdca_stabilization_tol and gamma>s.pdca_min_gamma*(1+1e-12): gamma=max(s.pdca_min_gamma,gamma*s.pdca_gamma_factor)
            elif gamma<=s.pdca_min_gamma*(1+1e-12) and step<s.pdca_step_tol and abs(rel)<s.pdca_obj_tol: break
    finally:
        solver.dispose()
    return cur,hist


# =============================================================================
# Library certificate and VF training
# =============================================================================

def library_metrics(w:np.ndarray,a:Arrays,lib:ScheduleLibrary,oracle_lb:Mapping[int,float],s:Settings,lam:float) -> dict[str,float]:
    _,corr,planning=correction_and_planning(w,a,s)
    case=float(np.sum(theta(a.error-corr,s.overtime,s.idle))/a.n_weeks)
    pred=0.0
    for wk,idx in a.week_slices.items(): pred += lib.best(wk,planning[idx],s)[1]
    pred/=a.n_weeks
    oracle=float(np.mean([oracle_lb[w] for w in a.week_slices]))
    sched=pred-oracle; l1=lam*float(np.sum(np.abs(w[1:])))
    return {"case_term":case,"library_planning":pred,"oracle_lb_mean":oracle,"scheduling_term":sched,"l1":l1,"certificate":case+sched+l1,"pure_certificate":case+sched}


def selected_library_columns(w:np.ndarray,a:Arrays,lib:ScheduleLibrary,s:Settings) -> dict[int,ScheduleColumn]:
    _,_,planning=correction_and_planning(w,a,s); out={}
    for wk,idx in a.week_slices.items(): out[wk]=lib.best(wk,planning[idx],s)[0]
    return out


def plan_policy_training(w:np.ndarray,a:Arrays,weeks:Sequence[WeekBundle],lib:ScheduleLibrary,s:Settings,*,seconds:int,gap:float,label:str) -> dict[int,PlanResult]:
    _,_,planning=correction_and_planning(w,a,s)
    dm={wk:planning[idx] for wk,idx in a.week_slices.items()}
    warm={wk:lib.best(wk,dm[wk],s)[0] for wk in a.week_slices if lib.entries(wk)}
    return solve_batch(weeks,dm,s,seconds=seconds,gap=gap,label=label,warm_by_week=warm)


def train_vf(
    start_w:np.ndarray,a:Arrays,weeks:Sequence[WeekBundle],lib:ScheduleLibrary,oracle_lb:Mapping[int,float],s:Settings,lam:float,root:Path,
    *, search_deadline: float | None = None,
) -> tuple[np.ndarray,list[dict[str,Any]]]:
    """Run the clean outer schedule-library MM search.

    There is only ONE expensive policy-facing planner batch per outer iteration.
    The previous outer iteration's candidate solve is the current-point oracle
    information for the next iteration. This keeps the D-vs-epsilon ledger
    meaningful without wasting a second 72-week planner batch.
    """
    cur=project_policy(start_w,a,s); traj=[]; stagnant=0
    # Establish a 1%-target current-point planner bracket once.
    current_plans=plan_policy_training(cur,a,weeks,lib,s,seconds=s.train_planner_seconds,gap=s.train_planner_gap,label="vf_anchor_0")
    lib.add_plans(current_plans,"VF_ANCHOR_0")
    for outer in range(1,s.vf_outer+1):
        if search_deadline is not None and remaining(search_deadline) < 120:
            LOG.warning("[VF] stopping before outer %d to preserve final-evaluation reserve",outer)
            break
        before_pool=lib.size()
        anchor=library_metrics(cur,a,lib,oracle_lb,s,lam)
        # Anchor inexactness: library value minus a valid weekly lower bound.
        _,_,cur_planning=correction_and_planning(cur,a,s)
        eps_parts=[]
        for wk,idx in a.week_slices.items():
            lib_obj=lib.best(wk,cur_planning[idx],s)[1]
            eps_parts.append(max(0.0,lib_obj-current_plans[wk].bound))
        eps=float(np.mean(eps_parts))

        fixed=selected_library_columns(cur,a,lib,s)
        spec=FixedSpec(f"VF_OUTER_{outer}",a,np.full(len(a.error),s.overtime),np.full(len(a.error),s.idle),fixed,lam,s)
        cand,hist=run_pdca(spec,cur,s,max_iterations=s.pdca_vf_inner_iterations)
        candm=library_metrics(cand,a,lib,oracle_lb,s,lam)
        accepted=candm["certificate"]<=anchor["certificate"]+1e-7*max(1.0,abs(anchor["certificate"]))
        D=max(0.0,anchor["certificate"]-candm["certificate"])
        maxdw=float(np.max(np.abs(cand-cur)))

        if accepted:
            cplans=plan_policy_training(cand,a,weeks,lib,s,seconds=s.train_planner_seconds,gap=s.train_planner_gap,label=f"vf_candidate_{outer}")
            cadded=lib.add_plans(cplans,f"VF_CANDIDATE_{outer}")
            changes=0
            for wk in a.week_slices:
                sig0=canonical_schedule_signature(current_plans[wk].column,lib.week_lookup[wk].instance)
                sig1=canonical_schedule_signature(cplans[wk].column,lib.week_lookup[wk].instance)
                changes+=int(sig0!=sig1)
        else:
            cand=cur.copy(); candm=anchor.copy(); cplans=current_plans; cadded=0; changes=0; maxdw=0.0

        finalm=library_metrics(cand,a,lib,oracle_lb,s,lam)
        if finalm["certificate"] > anchor["certificate"] + 1e-6*max(1.0,abs(anchor["certificate"])):
            raise AssertionError("Anytime certificate increased after an accepted outer step")
        rel=(anchor["certificate"]-finalm["certificate"])/max(1.0,abs(anchor["certificate"]))
        row={
            "outer":outer,"pool_before":before_pool,"candidate_added":cadded,"pool_after":lib.size(),
            "certificate_anchor":anchor["certificate"],"certificate_candidate_before_refresh":candm["certificate"],
            "certificate_final":finalm["certificate"],"relative_total_tightening":rel,"policy_decrease_D":D,
            "planner_uncertainty_epsilon":eps,"true_vf_descent_certified":bool(D>eps+1e-8),
            "max_anchor_planner_gap":max(r.gap for r in current_plans.values()),
            "max_candidate_planner_gap":max(r.gap for r in cplans.values()),
            "schedule_changes":changes,"accepted":accepted,"pdca_iterations":len(hist),"max_abs_dw":maxdw,
        }
        traj.append(row)
        LOG.info("[VF] outer=%d C %.3f -> %.3f | add=%d | D=%.2f eps=%.2f | changes=%d | trueVFcert=%s",
                 outer,anchor["certificate"],finalm["certificate"],cadded,D,eps,changes,row["true_vf_descent_certified"])
        cur=cand; current_plans=cplans
        if rel<s.vf_stop_relative_certificate and changes<=s.vf_stop_schedule_changes: stagnant+=1
        else: stagnant=0
        if stagnant>=s.vf_stop_patience: break
    write_csv(root/"VF_TRAJECTORY.csv",traj)
    return cur,traj


# =============================================================================
# Reachable-box saturation stress test
# =============================================================================

def saturation_correction(draw:int,n:int,rng:np.random.Generator,reach:float) -> np.ndarray:
    if draw==0: return np.zeros(n)
    if draw==1: return np.full(n,reach)
    if draw==2: return np.full(n,-reach)
    if draw==3: return reach*np.where(np.arange(n)%2==0,1.0,-1.0)
    if draw==4: return -reach*np.where(np.arange(n)%2==0,1.0,-1.0)
    mode=draw%3
    if mode==0: return reach*rng.choice([-1.0,1.0],size=n)
    if mode==1: return rng.uniform(-reach,reach,size=n)
    mask=rng.random(n)<0.25; z=np.zeros(n); z[mask]=reach*rng.choice([-1.0,1.0],size=int(mask.sum())); return z


def saturation_test(weeks:Sequence[WeekBundle],a:Arrays,lib:ScheduleLibrary,s:Settings,root:Path,deadline:float) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    reach=s.alpha*s.h
    week_lookup={w.position:w for w in weeks}
    rngs={w.position:np.random.default_rng(s.random_seed+100003*w.position) for w in weeks}
    active={w.position:True for w in weeks}; no_new={w.position:0 for w in weeks}; draws={w.position:0 for w in weeks}
    rows=[]; batch_id=0
    while any(active.values()):
        if remaining(deadline)<60: break
        batch_id+=1; tasks=[]
        for w in weeks:
            if not active[w.position]: continue
            for _ in range(s.saturation_batch):
                if draws[w.position]>=s.saturation_draws_per_week: active[w.position]=False; break
                draw=draws[w.position]; draws[w.position]+=1
                idx=a.week_slices[w.position]; c=saturation_correction(draw,len(idx),rngs[w.position],reach); d=a.booked[idx]+c
                warm=lib.best(w.position,d,s)[0]
                tasks.append((w,draw,d,warm))
        if not tasks: break
        LOG.info("[SAT] batch %d | solves=%d",batch_id,len(tasks))
        results=[]
        with ThreadPoolExecutor(max_workers=min(s.cores,len(tasks))) as ex:
            fut={ex.submit(solve_week,w,d,s,time_limit=s.saturation_seconds,mip_gap=s.saturation_gap,threads=1,warm=warm,label=f"sat_b{batch_id}_d{draw}"): (w,draw) for w,draw,d,warm in tasks}
            for f in as_completed(fut):
                w,draw=fut[f]; r=f.result(); before=lib.size(w.position); new=lib.add(w.position,r.column,f"SAT_{draw}"); results.append((w.position,draw,r,new,before))
        byweek={}
        for wk,draw,r,new,before in results:
            rows.append({"batch":batch_id,"week":wk,"draw":draw,"new_surface":new,"gap":r.gap,"status":r.status,"pool_before":before,"pool_after":lib.size(wk)})
            byweek.setdefault(wk,0); byweek[wk]+=int(new)
        for wk in list(active):
            if not active[wk]: continue
            if byweek.get(wk,0)==0: no_new[wk]+=1
            else: no_new[wk]=0
            if draws[wk]>=s.saturation_min_draws_per_week and no_new[wk]>=s.saturation_patience_batches: active[wk]=False
            if draws[wk]>=s.saturation_draws_per_week: active[wk]=False
    frame=pd.DataFrame(rows); frame.to_csv(root/"SATURATION_DRAWS.csv",index=False)
    late_threshold=max(0,s.saturation_draws_per_week-s.saturation_batch)
    summary={
        "reachable_correction_minutes":reach,
        "draws_total":int(sum(draws.values())),
        "draws_min_week":int(min(draws.values())),"draws_max_week":int(max(draws.values())),
        "new_surfaces":int(frame["new_surface"].sum()) if len(frame) else 0,
        "new_surfaces_last_batch":int(frame[frame["batch"]==frame["batch"].max()]["new_surface"].sum()) if len(frame) else 0,
        "weeks_stopped_by_patience":int(sum(no_new[w]>=s.saturation_patience_batches for w in no_new)),
        "max_gap":float(frame["gap"].max()) if len(frame) else math.nan,
        "empirical_saturation_strong":bool(len(frame)>0 and sum(no_new[w]>=s.saturation_patience_batches for w in no_new)==len(no_new)),
        "interpretation":"Empirical stress-test evidence only; random/structured sampling of the box does not prove mathematical completeness over the continuum.",
    }
    write_json(root/"SATURATION_SUMMARY.json",summary)
    return rows,summary


# =============================================================================
# A-posteriori policy evaluation (no theta envelope in reported regret)
# =============================================================================

def policy_duration_map(name:str,w:np.ndarray|None,a:Arrays,s:Settings) -> tuple[dict[int,np.ndarray],dict[str,float]]:
    if name=="BOOKED":
        planning=a.booked.copy(); delta=np.zeros(len(a.booked)); corr=np.zeros(len(a.booked))
    elif name=="ORACLE":
        planning=a.actual.copy(); delta=np.full(len(a.booked),np.nan); corr=a.error.copy()
    else:
        assert w is not None; delta,corr,planning=correction_and_planning(w,a,s)
    dm={wk:planning[idx] for wk,idx in a.week_slices.items()}
    if name in {"BOOKED","ORACLE"}:
        meta={"raw_mae":math.nan,"implemented_mae":float(np.mean(np.abs(a.error-corr))) if name=="BOOKED" else 0.0,"acceptance":math.nan,"decay":math.nan,"discard":math.nan}
    else:
        reg=response_regions(delta,s)
        meta={"raw_mae":float(np.mean(np.abs(delta-a.error))),"implemented_mae":float(np.mean(np.abs(corr-a.error))),"acceptance":float(np.mean(reg=="acceptance")),"decay":float(np.mean(reg=="decay")),"discard":float(np.mean(reg=="discard"))}
    return dm,meta


def evaluate_policies(
    label:str,weeks:Sequence[WeekBundle],a:Arrays,policies:Mapping[str,np.ndarray|None],oracle_plans:Mapping[int,PlanResult],s:Settings,root:Path
) -> tuple[pd.DataFrame,pd.DataFrame]:
    weekly_rows=[]; summary=[]
    # Start oracle brackets from the dedicated actual-duration solve.
    oracle_lb={w:r.bound for w,r in oracle_plans.items()}; oracle_ub={w:r.objective for w,r in oracle_plans.items()}
    all_results={}
    metas={}
    for name,w in policies.items():
        if name=="ORACLE":
            all_results[name]=dict(oracle_plans); metas[name]={"raw_mae":math.nan,"implemented_mae":0.0,"acceptance":math.nan,"decay":math.nan,"discard":math.nan}; continue
        dm,meta=policy_duration_map(name,w,a,s); metas[name]=meta
        plans=solve_batch(weeks,dm,s,seconds=s.final_planner_seconds,gap=s.final_planner_gap,label=f"final_{label}_{name}",deterministic_tiebreak=True,tiebreak_seconds=s.final_tiebreak_seconds)
        all_results[name]=plans
    # Every returned policy schedule is feasible under actual durations, so it can tighten the oracle UB.
    realized_costs_by_week={wk:[] for wk in a.week_slices}
    for name,plans in all_results.items():
        if name=="ORACLE": continue
        for wk,idx in a.week_slices.items():
            rc=float(plans[wk].column.compute_cost(a.actual[idx],_cost_cfg(s),s.turnover)); realized_costs_by_week[wk].append(rc)
    eff_oracle_ub={wk:min([oracle_ub[wk]]+realized_costs_by_week[wk]) for wk in a.week_slices}
    for name,plans in all_results.items():
        total_real=total_lo=total_hi=0.0; maxgap=0.0; exact_count=0
        for wk,idx in a.week_slices.items():
            if name=="ORACLE": rc=float(plans[wk].objective)
            else: rc=float(plans[wk].column.compute_cost(a.actual[idx],_cost_cfg(s),s.turnover))
            lo=max(0.0,rc-eff_oracle_ub[wk]); hi=max(0.0,rc-oracle_lb[wk]); total_real+=rc; total_lo+=lo; total_hi+=hi
            maxgap=max(maxgap,float(plans[wk].gap)); exact_count+=int(plans[wk].exact)
            weekly_rows.append({"split":label,"method":name,"week":wk,"week_start":str(next(w.start.date() for w in weeks if w.position==wk)),"planning_obj":plans[wk].objective,"planning_bound":plans[wk].bound,"planning_gap":plans[wk].gap,"planning_exact":plans[wk].exact,"realized_cost":rc,"oracle_lb":oracle_lb[wk],"oracle_ub_effective":eff_oracle_ub[wk],"regret_lower":lo,"regret_upper":hi})
        n=a.n_weeks; mm=metas[name]
        summary.append({"split":label,"method":name,"avg_realized_cost":total_real/n,"avg_regret_lower":total_lo/n,"avg_regret_upper":total_hi/n,"avg_regret_mid":0.5*(total_lo+total_hi)/n,"max_planning_gap":maxgap,"planning_exact_weeks":exact_count,"raw_mae":mm["raw_mae"],"implemented_mae":mm["implemented_mae"],"acceptance":mm["acceptance"],"decay":mm["decay"],"discard":mm["discard"]})
    sf=pd.DataFrame(summary); wf=pd.DataFrame(weekly_rows)
    # Gap closed intervals versus Booked.
    b=sf[sf.method=="BOOKED"].iloc[0]; blo=max(1e-9,float(b.avg_regret_lower)); bhi=max(1e-9,float(b.avg_regret_upper))
    gc_lo=[]; gc_hi=[]
    for _,r in sf.iterrows():
        # Conservative interval from regret brackets.
        lo=1.0-float(r.avg_regret_upper)/blo if blo>1e-8 else math.nan
        hi=1.0-float(r.avg_regret_lower)/bhi if bhi>1e-8 else math.nan
        gc_lo.append(100*lo if np.isfinite(lo) else math.nan); gc_hi.append(100*hi if np.isfinite(hi) else math.nan)
    sf["gap_closed_lower_pct"]=gc_lo; sf["gap_closed_upper_pct"]=gc_hi
    wf.to_csv(root/f"FINAL_{label.upper()}_WEEKLY.csv",index=False); sf.to_csv(root/f"FINAL_{label.upper()}_SUMMARY.csv",index=False)
    return wf,sf


# =============================================================================
# Stable solver audit
# =============================================================================

def stable_solver_audit(weeks:Sequence[WeekBundle],s:Settings) -> list[dict[str,Any]]:
    if stable_solve_pricing is None or s.stable_solver_audit_weeks<=0: return []
    rows=[]; cfg=SolverConfig(); cfg.time_limit_seconds=600; cfg.mip_gap=0.0001; cfg.threads=1; cfg.verbose=False
    for w in weeks[:s.stable_solver_audit_weeks]:
        d=np.asarray(w.instance.booked_durations(),float)
        ours=solve_week(w,d,s,time_limit=600,mip_gap=0.0001,label="stable_audit_ours")
        col,obj=stable_solve_pricing(w.instance.num_cases,d,w.instance.calendar,_cost_cfg(s),cfg,w.instance.case_eligible_blocks,s.turnover,model_name=f"stable_audit_repo_{w.position}")
        diff=math.nan if col is None else float(ours.objective-obj)
        rows.append({"week":w.position,"ours":ours.objective,"stable":obj,"difference":diff,"passed":bool(col is not None and abs(diff)<=2e-4*max(1.0,abs(obj)))})
    return rows


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument("--artifact-root",default="artifacts/final_vf_experiment")
    p.add_argument("--cores",type=int,default=15)
    p.add_argument("--max-wall-minutes",type=float,default=720)
    p.add_argument("--final-reserve-minutes",type=float,default=240)
    p.add_argument("--outer",type=int,default=6)
    p.add_argument("--oracle-seconds",type=int,default=1800)
    p.add_argument("--train-planner-seconds",type=int,default=300)
    p.add_argument("--train-planner-gap",type=float,default=0.01)
    p.add_argument("--saturation-draws",type=int,default=64)
    p.add_argument("--saturation-seconds",type=int,default=30)
    p.add_argument("--final-planner-seconds",type=int,default=1200)
    p.add_argument("--final-planner-gap",type=float,default=5e-4)
    p.add_argument("--allow-data-mismatch",action="store_true")
    p.add_argument("--skip-saturation",action="store_true")
    p.add_argument("--verbose",action="store_true")
    return p.parse_args()


def main() -> None:
    args=parse_args(); root=Path(args.artifact_root).resolve(); setup_logging(root,args.verbose)
    s=Settings(data=args.data,artifact_root=str(root),cores=args.cores,max_wall_minutes=args.max_wall_minutes,final_reserve_minutes=args.final_reserve_minutes,verbose=args.verbose,
               vf_outer=args.outer,oracle_seconds=args.oracle_seconds,train_planner_seconds=args.train_planner_seconds,train_planner_gap=args.train_planner_gap,
               saturation_draws_per_week=args.saturation_draws,saturation_seconds=args.saturation_seconds,final_planner_seconds=args.final_planner_seconds,final_planner_gap=args.final_planner_gap,
               strict_data_freeze=not args.allow_data_mismatch)
    s.validate(); start=time.monotonic(); deadline=start+60*s.max_wall_minutes; final_reserve=60*s.final_reserve_minutes
    write_json(root/"RUN_STATUS.json",{"status":"STARTED","version":SCRIPT_VERSION,"settings":asdict(s)})
    try:
        cfg=build_config(s); data_path=Path(s.data).resolve(); LOG.info("Loading canonical data from %s",data_path)
        df=load_data(cfg); scoped,scope_summary=apply_experiment_scope(df,cfg); train_starts,hold_starts,split_audit=choose_split(scoped,s)
        split_audit.to_csv(root/"WEEK_SPLIT.csv",index=False)
        ws=week_start_series(scoped); train_df=scoped[ws.isin(train_starts)].copy()
        hold_count=int(split_audit.loc[split_audit["split"]=="holdout","n_cases"].sum())
        if s.strict_data_freeze and (len(train_df)!=s.expected_train_cases or hold_count!=s.expected_holdout_cases):
            raise RuntimeError(f"DATA FREEZE MISMATCH: train={len(train_df)} expected={s.expected_train_cases}, holdout={hold_count} expected={s.expected_holdout_cases}")
        # Learn planning infrastructure only from pre-holdout history.
        hold_start=min(hold_starts); hist=scoped[ws<hold_start].copy(); pools=build_candidate_pools(hist,cfg); elig=build_eligibility_maps(hist,cfg)
        train_weeks=build_bundles(scoped,train_starts,cfg=cfg,candidate_pools=pools,eligibility_maps=elig,offset=0)
        enc=FrozenFeatureEncoder().fit(train_df); train_a=build_arrays(train_weeks,enc)
        if train_a.p!=111: raise AssertionError("p != 111")
        if float(np.min(train_a.booked)) <= s.alpha*s.h:
            LOG.warning("Some booked durations are <= 24 min; post-review positivity should be inspected explicitly. min=%.3f",float(np.min(train_a.booked)))
        freeze={"script_version":SCRIPT_VERSION,"git_head":git_head(),"input_sha256":sha256_file(data_path),"cleaned_cases":len(df),"scoped_cases":len(scoped),"train_cases":len(train_df),"holdout_cases":hold_count,"train_first":str(train_starts[0].date()),"train_last":str(train_starts[-1].date()),"holdout_first":str(hold_starts[0].date()),"holdout_last":str(hold_starts[-1].date()),"train_case_ids_sha256":sha256_text(",".join(map(str,sorted(train_a.case_ids.tolist())))),"feature_manifest":enc.manifest(),"scope_summary":asdict(scope_summary)}
        write_json(root/"DATA_FREEZE.json",freeze)
        LOG.info("[DATA] train=%d holdout=%d weeks=%d/%d p=%d",len(train_df),hold_count,len(train_starts),len(hold_starts),train_a.p)

        audit=stable_solver_audit(train_weeks,s); write_csv(root/"STABLE_SOLVER_AUDIT.csv",audit)
        if audit and not all(r["passed"] for r in audit): raise AssertionError("Clean weekly solver does not match stable repository solver on audit weeks")

        week_lookup={w.position:w for w in train_weeks}; lib=ScheduleLibrary(week_lookup)
        # Long training oracle: realized durations only.
        actual_map={wk:train_a.actual[idx] for wk,idx in train_a.week_slices.items()}
        train_oracle=solve_oracle_batch(train_weeks,actual_map,s,label="oracle_train")
        oracle_rows=[{"week":w,"objective":r.objective,"bound":r.bound,"gap":r.gap,"status":r.status,"exact":r.exact,"seconds":r.solve_seconds} for w,r in sorted(train_oracle.items())]
        write_csv(root/"ORACLE_TRAIN.csv",oracle_rows); oracle_lb={w:r.bound for w,r in train_oracle.items()}; lib.add_plans(train_oracle,"REALIZED_ORACLE",pin=True)

        # Status quo booked plans and RA exposure.
        booked_map={wk:train_a.booked[idx] for wk,idx in train_a.week_slices.items()}
        booked_plans=solve_batch(train_weeks,booked_map,s,seconds=s.train_planner_seconds,gap=s.train_planner_gap,label="booked_train")
        lib.add_plans(booked_plans,"BOOKED")
        labels=exposure_labels(booked_plans,train_a,week_lookup,s); pi,pi_metrics=crossfit_pi(train_a,labels,s); write_json(root/"RA_PI_METRICS.json",pi_metrics)

        zero=np.zeros(train_a.p); A0=case_envelope(zero,train_a,s); lam=s.l1_eta*A0/train_a.p; LOG.info("[REG] A(0)=%.3f lambda=%.6f",A0,lam)
        naive=train_naive(train_a,s,lam)
        ra_spec=FixedSpec("RA",train_a,s.overtime*pi,s.idle*(1-pi),None,lam,s); ra,_=run_pdca(ra_spec,naive,s,max_iterations=s.pdca_fixed_max_iterations)
        oracle_cols={w:r.column for w,r in train_oracle.items()}
        os_spec=FixedSpec("OS",train_a,np.full(len(train_a.error),s.overtime),np.full(len(train_a.error),s.idle),oracle_cols,lam,s)
        os_w,_=run_pdca(os_spec,ra,s,max_iterations=s.pdca_fixed_max_iterations)
        policies0={"ZERO":zero,"NAIVE":naive,"RA":ra,"OS":os_w}

        # Seed library from each learned policy with cheap but valid feasible schedules.
        for name,w in policies0.items():
            if name=="ZERO": continue
            plans=plan_policy_training(w,train_a,train_weeks,lib,s,seconds=s.seed_planner_seconds,gap=s.seed_planner_gap,label=f"seed_{name}")
            lib.add_plans(plans,f"SEED_{name}")
        start_table=[]
        for name,w in policies0.items():
            m=library_metrics(w,train_a,lib,oracle_lb,s,lam); start_table.append({"method":name,**m})
        pd.DataFrame(start_table).to_csv(root/"START_DECOMPOSITION.csv",index=False)
        best_start=min(policies0,key=lambda n:library_metrics(policies0[n],train_a,lib,oracle_lb,s,lam)["certificate"])
        LOG.info("[VF] best clean start=%s",best_start)
        vf_w,vf_traj=train_vf(policies0[best_start],train_a,train_weeks,lib,oracle_lb,s,lam,root,search_deadline=deadline-final_reserve)
        policies={"NAIVE":naive,"RA":ra,"OS":os_w,"VF":vf_w}
        np.savez(root/"POLICIES.npz",**policies)
        write_json(root/"FEATURE_MANIFEST.json",enc.manifest())

        # Final training decomposition table; this is the paper table revealing
        # prediction/error versus scheduling discrimination.
        decomp=[]
        for name,w in {"ZERO":zero,**policies}.items():
            m=library_metrics(w,train_a,lib,oracle_lb,s,lam); d,c,_=correction_and_planning(w,train_a,s)
            decomp.append({"method":name,**m,"raw_mae":float(np.mean(np.abs(d-train_a.error))),"implemented_mae":float(np.mean(np.abs(c-train_a.error)))})
        pd.DataFrame(decomp).to_csv(root/"TRAIN_DECOMPOSITION.csv",index=False)
        write_csv(root/"LIBRARY_SUMMARY.csv",lib.summary_rows())
        if not lib.check_pins(): raise AssertionError("Pinned oracle schedule disappeared from library")

        # Saturation uses only training data and a full superset box of all
        # implementable corrections. Skip if final evaluation reserve is needed.
        sat_summary={"skipped":True,"reason":"user flag or wall reserve"}
        if not args.skip_saturation and remaining(deadline)>final_reserve+600:
            sat_deadline=deadline-final_reserve
            _,sat_summary=saturation_test(train_weeks,train_a,lib,s,root,sat_deadline)
            write_csv(root/"LIBRARY_SUMMARY_AFTER_SATURATION.csv",lib.summary_rows())

        # HOLDOUT IS MATERIALIZED ONLY NOW.
        LOG.info("[HOLDOUT] materializing final 22 weeks now; training is frozen")
        hold_df=scoped[ws.isin(hold_starts)].copy()
        hold_weeks=build_bundles(scoped,hold_starts,cfg=cfg,candidate_pools=pools,eligibility_maps=elig,offset=s.train_weeks)
        hold_a=build_arrays(hold_weeks,enc)
        # Strong realized-duration oracle on holdout.
        hold_actual={wk:hold_a.actual[idx] for wk,idx in hold_a.week_slices.items()}
        hold_oracle=solve_oracle_batch(hold_weeks,hold_actual,s,label="oracle_holdout")
        write_csv(root/"ORACLE_HOLDOUT.csv",[{"week":w,"objective":r.objective,"bound":r.bound,"gap":r.gap,"status":r.status,"exact":r.exact,"seconds":r.solve_seconds} for w,r in sorted(hold_oracle.items())])
        eval_policies={"BOOKED":None,"NAIVE":naive,"RA":ra,"OS":os_w,"VF":vf_w,"ORACLE":None}
        _,hold_summary=evaluate_policies("holdout",hold_weeks,hold_a,eval_policies,hold_oracle,s,root)

        oracle_train_max=max(r.gap for r in train_oracle.values()); oracle_hold_max=max(r.gap for r in hold_oracle.values())
        final_nonoracle=hold_summary[~hold_summary.method.isin(["ORACLE"])]
        max_final_gap=float(final_nonoracle.max_planning_gap.max())
        data_ok=(len(train_df)==s.expected_train_cases and len(hold_df)==s.expected_holdout_cases)
        stable_ok=(not audit) or all(r["passed"] for r in audit)
        sat_strong=bool(sat_summary.get("empirical_saturation_strong",False)) if not sat_summary.get("skipped",False) else False
        paper_ready=bool(data_ok and stable_ok and max(oracle_train_max,oracle_hold_max)<=s.paper_ready_oracle_gap and max_final_gap<=s.paper_ready_planner_gap)
        decision="FINAL_EXPERIMENT_PAPER_READY" if paper_ready else "FINAL_EXPERIMENT_VALID_BUT_NUMERICAL_BRACKETS_REMAIN"
        payload={
            "decision":decision,"script_version":SCRIPT_VERSION,"elapsed_hours":(time.monotonic()-start)/3600,
            "data_freeze_ok":data_ok,"stable_solver_audit_ok":stable_ok,"train_cases":len(train_df),"holdout_cases":hold_count,
            "oracle_train_max_gap":oracle_train_max,"oracle_holdout_max_gap":oracle_hold_max,"final_policy_max_planner_gap":max_final_gap,
            "train_oracle_exact_weeks":sum(r.exact for r in train_oracle.values()),"holdout_oracle_exact_weeks":sum(r.exact for r in hold_oracle.values()),
            "library_size_final":lib.size(),"saturation":sat_summary,"holdout_consumed":True,
        }
        write_json(root/"FINAL_DECISION.json",payload)
        report=[
            "# Final VF experiment", "", f"**Decision:** {decision}", "",
            f"- canonical training sample: {len(train_df):,} cases / {s.train_weeks} weeks",
            f"- final holdout: {len(hold_df):,} cases / {s.holdout_weeks} weeks",
            f"- p: {train_a.p}", f"- final schedule library: {lib.size()} canonical schedule surfaces",
            f"- training oracle max gap: {100*oracle_train_max:.4f}%",
            f"- holdout oracle max gap: {100*oracle_hold_max:.4f}%",
            f"- final policy planner max gap: {100*max_final_gap:.4f}%",
            f"- empirical saturation strong: {sat_strong}", "",
            "## Interpretation", "",
            "TRAIN_DECOMPOSITION.csv separates the large case-envelope level from the schedule/value-function term that discriminates policies.",
            "FINAL_HOLDOUT_SUMMARY.csv reports realized-cost/regret brackets from explicit weekly schedules; it does not use the theta envelope as the reported regret.",
            "SATURATION_SUMMARY.json is a stress test over the full b +/- 24 minute implementable box. It is evidence of library saturation, not a proof over the continuous box.",
        ]
        (root/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
        write_json(root/"RUN_STATUS.json",{"status":"COMPLETED","decision":decision,"elapsed_seconds":time.monotonic()-start})
        LOG.info("[DONE] %s | elapsed %.2f h",decision,(time.monotonic()-start)/3600)
    except Exception as exc:
        write_json(root/"RUN_STATUS.json",{"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-start})
        LOG.exception("Fatal: %s",exc); raise


if __name__=="__main__":
    main()
