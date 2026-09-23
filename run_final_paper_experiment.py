#!/usr/bin/env python3
"""Final two-site paper experiment using the audited fixed-capacity planner.

The primary experiment pools the two largest cleaned UHN sites, TGH and TWH,
for learning while preserving physically separate site capacity.  The learned
recommendation policy is shared across sites, but every downstream weekly
planning call is solved as two independent site MILPs and recombined exactly:

    Q_t(w) = Q_{t,TGH}(w) + Q_{t,TWH}(w).

Because there are no cross-site assignment edges or coupling constraints, this
is mathematically identical to the monolithic two-site MIP and much easier to
solve.

Frozen scientific specification
--------------------------------
* TGH + TWH, Monday-Friday, 72 training weeks / 22 final holdout weeks;
* one shared case-level policy learned from both sites;
* training-only pooled feature vocabulary with the original 111-feature budget;
* regular_template fixed capacity fitted on the 72 training weeks only;
* 480 minute blocks, zero activation cost, no deferral;
* raw-service/same-site eligibility observed in >=3 training weeks;
* 30 minute occupation-based turnover;
* overtime = 15/minute, idle = 10/minute;
* reduced Psi site MILPs, with Phi recovered exactly by Phi = K + Psi;
* symmetric primary response alpha=.8, h=30;
* displayed recommendations are clipped to the configured display cap and to
  a minimum recommended duration of 1 minute; all training optimizers enforce
  the same restriction explicitly, so clipping is inactive on fitted training
  policies and only acts as a deployment safety guard out of sample;
* Booked, Naive, RA, OS, VF, and Oracle evaluation.

The two sites are selected from cleaned pre-holdout weekday data only.  The
code verifies that TWH and TGH are the two largest sites before continuing.
The feature vocabulary is also selected using the 72 training weeks only.

Recommended macOS run
---------------------
First run ``run_final_paper_training.py`` and review the training artifacts.
Only then run this file once to consume the final holdout.

caffeinate -i python run_final_paper_experiment.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_experiment \
  --cores 15 \
  --max-wall-minutes 720
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.config import Config, CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, Col, Domain, WeeklyInstance
from src.data.loader import load_data as canonical_load_data
from src.planning.eligibility import fit_service_room_history
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.roster import build_fixed_roster
from src.solvers.fixed_capacity import schedule_metrics, solve_fixed_capacity_assignment

SCRIPT_VERSION = "final_paper_experiment_2026_09_23_two_site_v4"
HOLDOUT_BOUNDARY = pd.Timestamp("2013-01-28")
PRIMARY_SITES = ("TGH", "TWH")
EXPECTED_SITE_RANKING = ("TWH", "TGH")
PRIMARY_ROSTER = "regular_template"
PRIMARY_ELIGIBILITY_WEEKS = 3
PRIMARY_TURNOVER = 30.0
PRIMARY_MIN_ACTIVATION_RATE = 0.25
PRIMARY_OBJECTIVE = "psi"
PRIMARY_ORACLE_GAP = 1e-3
PRIMARY_ORACLE_RETRY_TOL = 1e-3
MIN_RECOMMENDED_DURATION = 1.0
EXPECTED_TRAIN_CASES = 20519
EXPECTED_HOLDOUT_CASES = 6561
EXPECTED_TRAIN_SITE_COUNTS = {"TWH": 11295, "TGH": 9224}

FEATURE_FAMILY_BUDGETS = {
    "service": 12,
    "surgeon": 62,
    "procedure": 30,
    "site": 1,
}
BASE_FEATURE_NAMES = (
    "bias",
    "booked_z",
    "sin_week",
    "cos_week",
    "sin_month",
    "cos_month",
)


@dataclass
class FinalSettings(base.Settings):
    """Paper-frozen settings; computational budgets remain CLI configurable."""

    site: str = "TGH+TWH"
    expected_train_cases: int = EXPECTED_TRAIN_CASES
    expected_holdout_cases: int = EXPECTED_HOLDOUT_CASES
    opening: float = 0.0
    turnover: float = PRIMARY_TURNOVER
    oracle_gap: float = PRIMARY_ORACLE_GAP
    oracle_numeric_tol: float = PRIMARY_ORACLE_RETRY_TOL

    def validate(self) -> None:
        if self.site.upper() != "TGH+TWH":
            raise ValueError("Final experiment is frozen to pooled TGH+TWH.")
        if self.train_weeks != 72 or self.holdout_weeks != 22:
            raise ValueError("Final split is frozen to 72 train / 22 holdout weeks.")
        if abs(self.alpha - 0.8) > 1e-12 or abs(self.h - 30.0) > 1e-12:
            raise ValueError("Primary paper response is frozen to alpha=.8, h=30.")
        if abs(self.turnover - PRIMARY_TURNOVER) > 1e-12:
            raise ValueError("Primary paper turnover is frozen to 30 minutes.")
        if abs(self.opening) > 1e-12:
            raise ValueError("Fixed capacity has zero block activation cost.")
        if abs(self.oracle_gap - PRIMARY_ORACLE_GAP) > 1e-12:
            raise ValueError("Primary oracle target is frozen to a 0.1% native Psi gap.")
        if abs(self.oracle_numeric_tol - PRIMARY_ORACLE_RETRY_TOL) > 1e-12:
            raise ValueError("Primary oracle retry tolerance is frozen to 0.1%.")
        if self.cores <= 0:
            raise ValueError("cores must be positive")
        if self.max_wall_minutes <= self.final_reserve_minutes:
            raise ValueError("max wall must exceed final reserve")


@dataclass(frozen=True)
class PlanningContext:
    train: pd.DataFrame
    history: Any
    week_starts: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class SiteView:
    site: str
    global_indices: np.ndarray
    instance: WeeklyInstance


class FinalFeatureEncoder:
    """Leakage-safe pooled TGH/TWH case-local feature encoder."""

    PREFIX_TO_COLUMN = {
        "service": Col.CASE_SERVICE,
        "surgeon": Col.SURGEON_CODE,
        "procedure": Col.PROCEDURE_ID,
        "site": Col.SITE,
    }

    def __init__(self) -> None:
        self.feature_names: list[str] = list(BASE_FEATURE_NAMES)
        self.booked_mean = 0.0
        self.booked_std = 1.0
        self.references: dict[str, str] = {}
        self.selected_levels: dict[str, list[str]] = {}
        self.explicit_levels: dict[str, list[str]] = {}
        self.fitted = False

    @staticmethod
    def canon(x: object) -> str:
        if pd.isna(x):
            return Domain.OTHER
        value = str(x).strip()
        return value if value and value.lower() not in {"nan", "none", "<na>"} else Domain.OTHER

    @staticmethod
    def _rank_levels(values: pd.Series) -> list[str]:
        counts = values.value_counts(dropna=False)
        return sorted((str(v) for v in counts.index), key=lambda v: (-int(counts.loc[v]), v))

    def fit(self, frame: pd.DataFrame) -> "FinalFeatureEncoder":
        b = pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce").to_numpy(float)
        self.booked_mean = float(np.nanmean(b))
        self.booked_std = float(np.nanstd(b))
        if not np.isfinite(self.booked_std) or self.booked_std <= 1e-12:
            self.booked_std = 1.0

        names = list(BASE_FEATURE_NAMES)
        for prefix, column in self.PREFIX_TO_COLUMN.items():
            values = frame[column].map(self.canon)
            ranked = self._rank_levels(values)
            if prefix == "site":
                if set(ranked) != set(PRIMARY_SITES):
                    raise RuntimeError(f"Training feature scope has sites {ranked}, expected {PRIMARY_SITES}")
                reference = ranked[0]
                explicit = [v for v in ranked if v != reference]
                if len(explicit) != FEATURE_FAMILY_BUDGETS[prefix]:
                    raise AssertionError("Two-site feature budget changed")
                selected = list(ranked)
                names.extend(f"site_{v}" for v in explicit)
            else:
                budget = FEATURE_FAMILY_BUDGETS[prefix]
                if len(ranked) < budget:
                    raise RuntimeError(f"Only {len(ranked)} training levels for {prefix}; need at least {budget}")
                selected = ranked[:budget]
                reference = selected[0]
                explicit = [v for v in selected if v != reference]
                names.extend(f"{prefix}_{v}" for v in explicit)
                names.append(f"{prefix}___OTHER__")
            self.references[prefix] = reference
            self.selected_levels[prefix] = selected
            self.explicit_levels[prefix] = explicit

        if len(names) != 111:
            raise AssertionError(f"Pooled feature schema has p={len(names)}, expected 111")
        self.feature_names = names
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
        names = list(BASE_FEATURE_NAMES)
        for prefix, column in self.PREFIX_TO_COLUMN.items():
            vals = frame[column].map(self.canon).to_numpy(object)
            for level in self.explicit_levels[prefix]:
                cols.append((vals == level).astype(float))
                names.append(f"{prefix}_{level}")
            if prefix != "site":
                selected = np.asarray(self.selected_levels[prefix], dtype=object)
                cols.append((~np.isin(vals, selected)).astype(float))
                names.append(f"{prefix}___OTHER__")
        if names != self.feature_names:
            raise AssertionError("Pooled feature schema reconstruction failed")
        return sparse.csr_matrix(np.column_stack(cols), dtype=float)

    def transform_cases(self, cases: Sequence[Any]) -> sparse.csr_matrix:
        frame = pd.DataFrame({
            Col.BOOKED_MINUTES: [float(c.booked_duration_min) for c in cases],
            Col.CASE_SERVICE: [str(c.service) for c in cases],
            Col.SURGEON_CODE: [str(c.surgeon_code) for c in cases],
            Col.PROCEDURE_ID: [str(c.procedure_id) for c in cases],
            Col.SITE: [str(c.site) for c in cases],
            Col.WEEK_OF_YEAR: [int(c.week_of_year) for c in cases],
            Col.MONTH: [int(c.month) for c in cases],
        })
        return self.transform_frame(frame)

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "booked_mean": self.booked_mean,
            "booked_std": self.booked_std,
            "references": self.references,
            "selected_levels": self.selected_levels,
            "feature_family_budgets": FEATURE_FAMILY_BUDGETS,
            "selection_rule": "training-only frequency ranking; count descending, lexical tie break",
            "sites": list(PRIMARY_SITES),
            "p": len(self.feature_names),
            "minimum_recommended_duration": MIN_RECOMMENDED_DURATION,
            "out_of_sample_display_rule": "clip Xw to [-display_cap, display_cap] and enforce booked + displayed_correction >= minimum_recommended_duration",
        }


def final_load_data(config: Config, *args, **kwargs):
    kwargs["site_history_end"] = HOLDOUT_BOUNDARY
    frame = canonical_load_data(config, *args, **kwargs)
    dt = pd.to_datetime(frame[Col.ACTUAL_START], errors="coerce")
    pre = frame[(dt < HOLDOUT_BOUNDARY) & dt.dt.weekday.isin(range(5))].copy()
    counts = pre[Col.SITE].value_counts()
    ranking = tuple(str(x) for x in counts.index[:2])
    if ranking != EXPECTED_SITE_RANKING:
        raise RuntimeError(f"Two largest pre-holdout weekday sites changed: {ranking}; expected {EXPECTED_SITE_RANKING}")
    base.LOG.info("[SITES] cleaned pre-holdout weekday counts=%s; primary=%s", counts.to_dict(), PRIMARY_SITES)
    frame[Col.CASE_SERVICE] = frame[Col.CASE_SERVICE_RAW]
    frame[Col.SURGEON_CODE] = frame[Col.SURGEON_CODE_RAW]
    frame[Col.PROCEDURE_ID] = frame[Col.PROCEDURE_ID_RAW]
    return frame


def final_build_config(s: FinalSettings) -> Config:
    cfg = Config()
    cfg.data.excel_file_path = s.data
    cfg.data.horizon_days = 7
    cfg.scope.planning_sites = PRIMARY_SITES
    cfg.scope.planning_weekdays = (0, 1, 2, 3, 4)
    cfg.capacity.block_capacity_minutes = s.capacity
    cfg.capacity.activation_cost_per_block = 0.0
    cfg.capacity.min_activation_rate = PRIMARY_MIN_ACTIVATION_RATE
    cfg.capacity.turnover_minutes = PRIMARY_TURNOVER
    cfg.capacity.eligibility_min_weeks = PRIMARY_ELIGIBILITY_WEEKS
    cfg.costs.overtime_per_minute = s.overtime
    cfg.costs.idle_per_minute = s.idle
    cfg.costs.deferral_per_case = 1e12
    cfg.solver.verbose = False
    return cfg


def final_build_candidate_pools(df_preholdout: pd.DataFrame, config: Config) -> PlanningContext:
    work = df_preholdout.copy()
    starts = base.week_start_series(work)
    counts = work.assign(_week_start=starts).groupby("_week_start").size().sort_index()
    eligible = counts[counts >= 50]
    if len(eligible) < 72:
        raise RuntimeError(f"Only {len(eligible)} pre-holdout eligible weeks; expected at least 72")
    selected = tuple(pd.Timestamp(x) for x in eligible.index[-72:])
    train = work[starts.isin(selected)].copy()
    if len(train) != EXPECTED_TRAIN_CASES:
        raise RuntimeError(f"Fixed two-site planning history changed: {len(train)} cases != {EXPECTED_TRAIN_CASES}")
    site_counts = train[Col.SITE].value_counts().to_dict()
    if site_counts != EXPECTED_TRAIN_SITE_COUNTS:
        raise RuntimeError(f"Two-site training composition changed: {site_counts} != {EXPECTED_TRAIN_SITE_COUNTS}")
    return PlanningContext(train=train, history=fit_service_room_history(train), week_starts=selected)


def final_build_eligibility_maps(df_preholdout: pd.DataFrame, config: Config):
    return None


def final_build_bundles(
    df_scoped: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    *,
    cfg: Config,
    candidate_pools: PlanningContext,
    eligibility_maps: Any,
    offset: int,
) -> list[base.WeekBundle]:
    if not isinstance(candidate_pools, PlanningContext):
        raise TypeError("Final experiment requires PlanningContext from the frozen 72-week cohort")
    out: list[base.WeekBundle] = []
    for j, start in enumerate(starts):
        start = pd.Timestamp(start).normalize()
        roster = build_fixed_roster(candidate_pools.train, start, cfg, PRIMARY_ROSTER)
        roster_sites = {b.site for b in roster.calendar.candidates}
        if roster_sites != set(PRIMARY_SITES):
            raise RuntimeError(f"Roster sites changed at {start.date()}: {roster_sites}")
        inst = build_weekly_instance_with_calendar(
            df_scoped, start, offset + j, cfg, roster.calendar,
            candidate_pools.history, PRIMARY_ELIGIBILITY_WEEKS,
        )
        if inst.num_cases == 0:
            raise RuntimeError(f"Empty selected week {start.date()}")
        missing = [i for i in range(inst.num_cases) if not inst.case_eligible_blocks.get(i)]
        if missing:
            raise RuntimeError(f"Week {start.date()} has {len(missing)} cases without fixed-roster eligibility")
        for i, case in enumerate(inst.cases):
            if any(bid.site != case.site for bid in inst.case_eligible_blocks[i]):
                raise AssertionError(f"Cross-site eligibility detected for case {case.case_id}")
        out.append(base.WeekBundle(offset + j, start, inst))
    return out


def final_cost_cfg(s: FinalSettings) -> CostConfig:
    return CostConfig(overtime_per_minute=s.overtime, idle_per_minute=s.idle, deferral_per_case=1e12)


def _display_lower_bound(booked: np.ndarray, s: FinalSettings) -> np.ndarray:
    b = np.asarray(booked, float)
    return np.maximum(-float(s.display_cap), MIN_RECOMMENDED_DURATION - b)


def final_correction_and_planning(
    w: np.ndarray, a: base.Arrays, s: FinalSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(a.X @ np.asarray(w, float), float).reshape(-1)
    delta = np.minimum(raw, float(s.display_cap))
    delta = np.maximum(delta, _display_lower_bound(a.booked, s))
    correction = base.response_value(delta, s)
    planning = np.asarray(a.booked, float) + correction
    if np.any(planning <= 0):
        raise AssertionError("Recommendation safety guard produced a nonpositive planning duration")
    return delta, correction, planning


def final_train_naive(a: base.Arrays, s: FinalSettings, lam: float) -> np.ndarray:
    n, p = a.X.shape
    X = a.X.tocsr()
    m = base.gp.Model("naive_lad_safe")
    m.Params.OutputFlag = 1 if s.verbose else 0
    m.Params.Threads = s.cores
    m.Params.TimeLimit = 900
    w = m.addVars(p, lb=-s.coefficient_bound, ub=s.coefficient_bound, name="w")
    absr = m.addVars(n, lb=0.0, name="absr")
    absw = m.addVars(range(1, p), lb=0.0, name="absw")
    lower = _display_lower_bound(a.booked, s)
    for i in range(n):
        row = X.getrow(i)
        expr = base.gp.LinExpr(row.data.tolist(), [w[int(j)] for j in row.indices])
        m.addConstr(absr[i] >= float(a.error[i]) - expr)
        m.addConstr(absr[i] >= -float(a.error[i]) + expr)
        m.addConstr(expr <= s.display_cap)
        m.addConstr(expr >= float(lower[i]))
    for j in range(1, p):
        m.addConstr(absw[j] >= w[j])
        m.addConstr(absw[j] >= -w[j])
    obj = base.quicksum(absr[i] for i in range(n)) / a.n_weeks + lam * base.quicksum(absw[j] for j in range(1, p))
    m.setObjective(obj, base.GRB.MINIMIZE)
    m.optimize()
    if m.SolCount <= 0:
        raise RuntimeError("safe naive LAD failed")
    out = np.array([w[j].X for j in range(p)], float)
    m.dispose()
    raw = np.asarray(a.X @ out, float).reshape(-1)
    if np.any(raw < lower - 1e-7) or np.any(raw > s.display_cap + 1e-7):
        raise AssertionError("Naive solution violates recommendation safety constraints")
    return out


def _site_view(week: base.WeekBundle, site: str) -> SiteView:
    inst = week.instance
    global_indices = np.asarray([i for i, case in enumerate(inst.cases) if str(case.site) == site], dtype=int)
    site_blocks = [b for b in inst.calendar.candidates if str(b.site) == site]
    if not site_blocks:
        raise RuntimeError(f"Week {week.position} has no fixed roster blocks for {site}")
    block_ids = {b.id for b in site_blocks}
    local_eligibility: dict[int, list[BlockId]] = {}
    local_diagnostics: dict[int, dict[str, Any]] = {}
    for local_i, global_i in enumerate(global_indices.tolist()):
        bids = [bid for bid in inst.case_eligible_blocks.get(global_i, []) if bid in block_ids and bid.site == site]
        if not bids:
            raise RuntimeError(f"Week {week.position} site {site}: pooled case {global_i} has no local eligible block")
        local_eligibility[local_i] = bids
        if global_i in inst.eligibility_diagnostics:
            local_diagnostics[local_i] = dict(inst.eligibility_diagnostics[global_i])
    sub = WeeklyInstance(
        week_index=inst.week_index,
        start_date=inst.start_date,
        end_date=inst.end_date,
        cases=[inst.cases[int(i)] for i in global_indices],
        calendar=BlockCalendar(site_blocks),
        case_eligible_blocks=local_eligibility,
        eligibility_diagnostics=local_diagnostics,
    )
    return SiteView(site=site, global_indices=global_indices, instance=sub)


def _localize_warm(warm: ScheduleColumn | None, view: SiteView) -> ScheduleColumn | None:
    if warm is None:
        return None

    # A site solve returns a site-local column indexed 0..n_site-1. The regular
    # production warm start is instead a pooled column indexed in the original
    # two-site week. Accept both representations explicitly; confusing them was
    # the source of the audit failure when the Psi site solution was passed as
    # the warm start for the direct-Phi site audit.
    local_case_ids = {
        int(i)
        for (i, bid), value in warm.z_assign.items()
        if bid.site == view.site and value > 0.5
    }
    if (
        warm.n_cases == view.instance.num_cases
        and local_case_ids == set(range(view.instance.num_cases))
        and all(bid.site == view.site for (_, bid), value in warm.z_assign.items() if value > 0.5)
    ):
        z_assign = {
            (int(i), bid): float(value)
            for (i, bid), value in warm.z_assign.items()
            if bid.site == view.site and value > 0.5
        }
    else:
        global_to_local = {int(g): i for i, g in enumerate(view.global_indices.tolist())}
        z_assign = {
            (global_to_local[int(i)], bid): float(value)
            for (i, bid), value in warm.z_assign.items()
            if int(i) in global_to_local and bid.site == view.site and value > 0.5
        }

    assigned = {i for i, _ in z_assign}
    if assigned != set(range(view.instance.num_cases)):
        raise ValueError(f"Warm start is incomplete for site {view.site}")
    blocks = frozenset(view.instance.calendar.block_ids)
    y_used = frozenset(bid for _, bid in z_assign)
    return ScheduleColumn(
        z_assign=z_assign,
        z_defer=frozenset(),
        v_open=blocks,
        y_used=y_used,
        n_cases=view.instance.num_cases,
        block_capacities={b.id: float(b.capacity_minutes) for b in view.instance.calendar.candidates},
        block_activation_costs={b.id: 0.0 for b in view.instance.calendar.candidates},
    )


def _solve_site_assignment(
    week: base.WeekBundle,
    durations: np.ndarray,
    s: FinalSettings,
    site: str,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int,
    warm: ScheduleColumn | None,
    objective_mode: str = PRIMARY_OBJECTIVE,
):
    view = _site_view(week, site)
    local_durations = np.asarray(durations, float)[view.global_indices]
    solver_cfg = SolverConfig(
        time_limit_seconds=max(1, int(time_limit)), mip_gap=max(0.0, float(mip_gap)),
        threads=max(1, int(threads)), verbose=bool(s.verbose), mip_gap_abs=1e-10,
        seed=int(s.random_seed),
    )
    result = solve_fixed_capacity_assignment(
        view.instance, local_durations, final_cost_cfg(s), PRIMARY_TURNOVER,
        solver_cfg, objective_mode=objective_mode, warm_start=_localize_warm(warm, view),
        symmetry_breaking=True,
    )
    return view, local_durations, result


def _merge_site_columns(
    pooled_instance: WeeklyInstance,
    parts: Sequence[tuple[SiteView, ScheduleColumn]],
) -> ScheduleColumn:
    z_assign: dict[tuple[int, BlockId], float] = {}
    v_open: set[BlockId] = set()
    y_used: set[BlockId] = set()
    capacities: dict[BlockId, float] = {}
    activation: dict[BlockId, float] = {}
    covered_cases: set[int] = set()
    for view, column in parts:
        if column.n_cases != len(view.global_indices):
            raise AssertionError("Site column/local case mapping mismatch")
        for (local_i, bid), value in column.z_assign.items():
            global_i = int(view.global_indices[int(local_i)])
            z_assign[global_i, bid] = float(value)
            if value > 0.5:
                covered_cases.add(global_i)
        v_open.update(column.v_open)
        y_used.update(column.y_used)
        capacities.update(column.block_capacities)
        activation.update(column.block_activation_costs)
    if covered_cases != set(range(pooled_instance.num_cases)):
        missing = sorted(set(range(pooled_instance.num_cases)) - covered_cases)
        raise AssertionError(f"Merged site schedules do not cover all pooled cases: {missing[:10]}")
    if v_open != set(pooled_instance.calendar.block_ids):
        raise AssertionError("Merged site schedules do not preserve all pooled fixed capacity")
    return ScheduleColumn(
        z_assign=z_assign, z_defer=frozenset(), v_open=frozenset(v_open), y_used=frozenset(y_used),
        n_cases=pooled_instance.num_cases, block_capacities=capacities, block_activation_costs=activation,
    )


def fixed_signature(col: ScheduleColumn, inst: WeeklyInstance) -> str:
    eligible_by_block = {
        b.id: tuple(sorted(i for i in range(inst.num_cases) if b.id in inst.case_eligible_blocks.get(i, [])))
        for b in inst.calendar.candidates
    }
    groups: dict[tuple[Any, ...], list[BlockId]] = {}
    for b in inst.calendar.candidates:
        key = (str(b.site), float(b.capacity_minutes), eligible_by_block[b.id])
        groups.setdefault(key, []).append(b.id)
    payload = []
    for key, bids in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        partitions = sorted(tuple(col.cases_in_block(bid)) for bid in bids)
        payload.append((key, partitions))
    return base.sha256_text(base.json.dumps(payload, sort_keys=True, default=str))


def final_solve_week(
    week: base.WeekBundle,
    durations: np.ndarray,
    s: FinalSettings,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int = 1,
    warm: ScheduleColumn | None = None,
    label: str = "plan",
    deterministic_tiebreak: bool = False,
    tiebreak_seconds: int | None = None,
) -> base.PlanResult:
    del deterministic_tiebreak, tiebreak_seconds
    d = np.asarray(durations, float)
    if d.shape != (week.instance.num_cases,):
        raise ValueError("duration length mismatch")
    t0 = base.time.perf_counter()
    site_parts: list[tuple[SiteView, ScheduleColumn]] = []
    phi_ub = phi_lb = psi_ub = psi_lb = 0.0
    statuses: list[str] = []
    all_proven_optimal = True
    for site in PRIMARY_SITES:
        view, _, result = _solve_site_assignment(
            week, d, s, site, time_limit=time_limit, mip_gap=mip_gap,
            threads=threads, warm=warm, objective_mode=PRIMARY_OBJECTIVE,
        )
        if result.column is None or result.phi_ub is None or result.phi_lb is None or result.psi_ub is None or result.psi_lb is None:
            raise RuntimeError(f"Week {week.position} {label} site {site}: fixed planner returned no incumbent/bound")
        site_parts.append((view, result.column))
        phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_proven_optimal = all_proven_optimal and bool(result.diagnostics.proven_optimal)

    column = _merge_site_columns(week.instance, site_parts)
    combined_metrics = schedule_metrics(column, d, final_cost_cfg(s), PRIMARY_TURNOVER)
    if abs(float(combined_metrics["phi"]) - phi_ub) > 1e-5:
        raise AssertionError(f"Week {week.position}: decomposed Phi mismatch {combined_metrics['phi']} != {phi_ub}")
    native_gap = base.rel_gap(psi_ub, psi_lb)
    exact = bool(all_proven_optimal and abs(psi_ub - psi_lb) <= 1e-6)
    status = "OPTIMAL" if all_proven_optimal else "|".join(statuses)
    return base.PlanResult(
        week=week.position, column=column, objective=float(phi_ub), bound=float(phi_lb), gap=float(native_gap),
        status=status, solve_seconds=base.time.perf_counter() - t0, exact=exact, tiebreak_used=False,
    )


class FinalConvexPDCASubproblem:
    def __init__(self, spec: base.FixedSpec, s: FinalSettings, name: str):
        self.spec = spec
        self.s = s
        a = spec.arrays
        n, p = a.X.shape
        X = a.X.tocsr()
        m = base.gp.Model(name)
        m.Params.OutputFlag = 1 if s.verbose else 0
        m.Params.Threads = s.cores
        m.Params.TimeLimit = s.pdca_convex_seconds
        m.Params.Method = 2
        self.m = m
        self.w = m.addVars(p, lb=-s.coefficient_bound, ub=s.coefficient_bound, name="w")
        ph = m.addVars(n, lb=0, name="ph")
        pe = m.addVars(n, lb=0, name="pe")
        ne = m.addVars(n, lb=0, name="ne")
        nh = m.addVars(n, lb=0, name="nh")
        R = m.addVars(n, lb=0, name="R")
        q = 1 - s.alpha
        eps = s.h / q
        lower = _display_lower_bound(a.booked, s)
        self.P, self.N, self.delta = {}, {}, {}
        for i in range(n):
            row = X.getrow(i)
            de = base.gp.LinExpr(row.data.tolist(), [self.w[int(j)] for j in row.indices])
            self.delta[i] = de
            m.addConstr(de <= s.display_cap)
            m.addConstr(de >= float(lower[i]))
            m.addConstr(ph[i] >= de + s.h)
            m.addConstr(pe[i] >= de - eps)
            m.addConstr(ne[i] >= de + eps)
            m.addConstr(nh[i] >= de - s.h)
            P = ph[i] + q * pe[i]
            N = q * ne[i] + nh[i]
            self.P[i], self.N[i] = P, N
            m.addConstr(R[i] >= float(a.error[i]) + N)
            m.addConstr(R[i] >= P)
        G = base.quicksum(float(spec.p_plus[i] + spec.p_minus[i]) * R[i] for i in range(n)) / a.n_weeks
        if spec.fixed is not None:
            schedG = base.gp.LinExpr()
            for wk, idx in a.week_slices.items():
                col = spec.fixed[wk]
                for bid in col.v_open:
                    locals_ = col.cases_in_block(bid)
                    globals_ = [int(idx[i]) for i in locals_]
                    pb = base.quicksum(self.P[g] for g in globals_) if globals_ else base.gp.LinExpr(0.0)
                    nb = base.quicksum(self.N[g] for g in globals_) if globals_ else base.gp.LinExpr(0.0)
                    turnover_const = PRIMARY_TURNOVER * max(0, len(locals_) - 1)
                    kappa = sum(float(a.booked[g]) for g in globals_) + turnover_const - float(col.block_capacities[bid])
                    go = m.addVar(lb=-base.GRB.INFINITY, name=f"go_{wk}_{len(locals_)}_{len(m.getVars())}")
                    gi = m.addVar(lb=-base.GRB.INFINITY, name=f"gi_{wk}_{len(locals_)}_{len(m.getVars())}")
                    m.addConstr(go >= kappa + pb)
                    m.addConstr(go >= nb)
                    m.addConstr(gi >= -kappa + nb)
                    m.addConstr(gi >= pb)
                    schedG += s.overtime * go + s.idle * gi
            G += schedG / a.n_weeks
        self.absw = m.addVars(range(1, p), lb=0, name="absw")
        for j in range(1, p):
            m.addConstr(self.absw[j] >= self.w[j])
            m.addConstr(self.absw[j] >= -self.w[j])
        G += spec.lam * base.quicksum(self.absw[j] for j in range(1, p))
        self.G = G
        self.p = p

    def solve(self, current: np.ndarray, q_w: np.ndarray, gamma: float) -> np.ndarray:
        linear = base.quicksum(-float(q_w[j] + gamma * current[j]) * self.w[j] for j in range(self.p))
        quad = 0.5 * float(gamma) * base.quicksum(self.w[j] * self.w[j] for j in range(self.p))
        self.m.setObjective(self.G + linear + quad, base.GRB.MINIMIZE)
        for j in range(self.p):
            self.w[j].Start = float(current[j])
        self.m.optimize()
        if self.m.SolCount <= 0:
            raise RuntimeError(f"pDCA convex subproblem {self.m.ModelName} produced no solution")
        return np.array([self.w[j].X for j in range(self.p)], float)

    def dispose(self):
        self.m.dispose()


def final_solver_audit(weeks: Sequence[base.WeekBundle], s: FinalSettings) -> list[dict[str, Any]]:
    rows = []
    for week in weeks[: min(2, len(weeks))]:
        d = np.asarray(week.instance.booked_durations(), float)
        for site in PRIMARY_SITES:
            view, local_d, psi = _solve_site_assignment(
                week, d, s, site, time_limit=60, mip_gap=0.01,
                threads=1, warm=None, objective_mode="psi",
            )
            if psi.column is None:
                rows.append({"week": week.position, "site": site, "passed": False, "reason": "missing psi incumbent"})
                continue
            _, _, phi = _solve_site_assignment(
                week, d, s, site, time_limit=60, mip_gap=0.01,
                threads=1, warm=psi.column, objective_mode="phi",
            )
            if phi.column is None:
                rows.append({"week": week.position, "site": site, "passed": False, "reason": "missing phi incumbent"})
                continue
            psi_cost = schedule_metrics(psi.column, local_d, final_cost_cfg(s), PRIMARY_TURNOVER)["phi"]
            phi_cost = schedule_metrics(phi.column, local_d, final_cost_cfg(s), PRIMARY_TURNOVER)["phi"]
            intervals_overlap = not (
                float(psi.phi_ub) < float(phi.phi_lb) - 1e-6
                or float(phi.phi_ub) < float(psi.phi_lb) - 1e-6
            )
            rows.append({
                "week": week.position,
                "site": site,
                "n_cases": view.instance.num_cases,
                "psi_schedule_phi": psi_cost,
                "phi_schedule_phi": phi_cost,
                "psi_phi_lb": psi.phi_lb,
                "psi_phi_ub": psi.phi_ub,
                "phi_lb": phi.phi_lb,
                "phi_ub": phi.phi_ub,
                "identity_error": psi.metrics["identity_error"],
                "passed": bool(intervals_overlap and psi.metrics["identity_error"] <= 1e-6),
            })
    return rows


def install_final_adapter() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.Settings = FinalSettings
    base.FrozenFeatureEncoder = FinalFeatureEncoder
    base.load_data = final_load_data
    base.build_config = final_build_config
    base.build_candidate_pools = final_build_candidate_pools
    base.build_eligibility_maps = final_build_eligibility_maps
    base.build_bundles = final_build_bundles
    base.solve_week = final_solve_week
    base._cost_cfg = final_cost_cfg
    base.canonical_schedule_signature = fixed_signature
    base.correction_and_planning = final_correction_and_planning
    base.train_naive = final_train_naive
    base.ConvexPDCASubproblem = FinalConvexPDCASubproblem
    base.stable_solver_audit = final_solver_audit


if __name__ == "__main__":
    install_final_adapter()
    base.main()
