#!/usr/bin/env python3
"""Final two-site paper experiment using the audited fixed-capacity planner.

The primary experiment pools the two largest cleaned UHN sites, TGH and TWH,
for learning while preserving physically separate site capacity.  Every weekly
instance contains both sites; same-site eligibility forbids cross-site
assignment, so the weekly objective is exactly the sum of the TGH and TWH
planning costs under one shared recommendation policy.

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
* reduced Psi weekly MILP, with Phi recovered exactly by Phi = K + Psi;
* symmetric primary response alpha=.8, h=30;
* Booked, Naive, RA, OS, VF, and Oracle evaluation.

The two sites were selected from cleaned pre-holdout weekday data only.  The
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
from src.core.types import BlockId, Col, Domain, WeeklyInstance
from src.data.loader import load_data as canonical_load_data
from src.planning.eligibility import fit_service_room_history
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.roster import build_fixed_roster
from src.solvers.fixed_capacity import schedule_metrics, solve_fixed_capacity_assignment

SCRIPT_VERSION = "final_paper_experiment_2026_09_23_two_site_v2"
HOLDOUT_BOUNDARY = pd.Timestamp("2013-01-28")
PRIMARY_SITES = ("TGH", "TWH")
EXPECTED_SITE_RANKING = ("TWH", "TGH")
PRIMARY_ROSTER = "regular_template"
PRIMARY_ELIGIBILITY_WEEKS = 3
PRIMARY_TURNOVER = 30.0
PRIMARY_MIN_ACTIVATION_RATE = 0.25
PRIMARY_OBJECTIVE = "psi"
EXPECTED_TRAIN_CASES = 20519
EXPECTED_HOLDOUT_CASES = 6561
EXPECTED_TRAIN_SITE_COUNTS = {"TWH": 11295, "TGH": 9224}

# Keep the feature-family sizes from the approved 111-dimensional TGH model,
# but reselect represented levels by pooled two-site TRAINING frequency only.
# For service/surgeon/procedure, a family budget B means B-1 explicit dummies
# plus one __OTHER__ bucket. Site uses one TGH-vs-TWH dummy and no OTHER bucket
# because the experiment scope is frozen to exactly these two sites.
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

    site: str = "TGH+TWH"  # legacy scalar field; planning_sites is set below.
    expected_train_cases: int = EXPECTED_TRAIN_CASES
    expected_holdout_cases: int = EXPECTED_HOLDOUT_CASES
    opening: float = 0.0
    turnover: float = PRIMARY_TURNOVER

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
        if self.cores <= 0:
            raise ValueError("cores must be positive")
        if self.max_wall_minutes <= self.final_reserve_minutes:
            raise ValueError("max wall must exceed final reserve")


@dataclass(frozen=True)
class PlanningContext:
    train: pd.DataFrame
    history: Any
    week_starts: tuple[pd.Timestamp, ...]


class FinalFeatureEncoder:
    """Leakage-safe pooled TGH/TWH case-local feature encoder.

    The total dimension remains 111 so the statistical capacity and L1 scaling
    are comparable with the previously approved experiment.  Which categorical
    levels receive their own coefficient is re-estimated from the pooled 72-week
    training sample only.  All omitted or future unseen levels map to OTHER.
    """

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
                    raise RuntimeError(
                        f"Only {len(ranked)} training levels for {prefix}; need at least {budget}"
                    )
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
        frame = pd.DataFrame(
            {
                Col.BOOKED_MINUTES: [float(c.booked_duration_min) for c in cases],
                Col.CASE_SERVICE: [str(c.service) for c in cases],
                Col.SURGEON_CODE: [str(c.surgeon_code) for c in cases],
                Col.PROCEDURE_ID: [str(c.procedure_id) for c in cases],
                Col.SITE: [str(c.site) for c in cases],
                Col.WEEK_OF_YEAR: [int(c.week_of_year) for c in cases],
                Col.MONTH: [int(c.month) for c in cases],
            }
        )
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
        }


def final_load_data(config: Config, *args, **kwargs):
    """Leakage-safe cleaning, site selection audit, and raw modeling labels."""

    kwargs["site_history_end"] = HOLDOUT_BOUNDARY
    frame = canonical_load_data(config, *args, **kwargs)

    dt = pd.to_datetime(frame[Col.ACTUAL_START], errors="coerce")
    pre = frame[(dt < HOLDOUT_BOUNDARY) & dt.dt.weekday.isin(range(5))].copy()
    counts = pre[Col.SITE].value_counts()
    ranking = tuple(str(x) for x in counts.index[:2])
    if ranking != EXPECTED_SITE_RANKING:
        raise RuntimeError(
            f"Two largest pre-holdout weekday sites changed: {ranking}; expected {EXPECTED_SITE_RANKING}"
        )
    base.LOG.info("[SITES] cleaned pre-holdout weekday counts=%s; primary=%s", counts.to_dict(), PRIMARY_SITES)

    # The canonical loader keeps these raw structural labels before its legacy
    # full-sample rare-category pooling.  Restore them here so the final feature
    # vocabulary is learned exclusively from the 72 training weeks.
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
    """Return the exact pooled 72-week training cohort."""

    work = df_preholdout.copy()
    starts = base.week_start_series(work)
    counts = work.assign(_week_start=starts).groupby("_week_start").size().sort_index()
    eligible = counts[counts >= 50]
    if len(eligible) < 72:
        raise RuntimeError(f"Only {len(eligible)} pre-holdout eligible weeks; expected at least 72")
    selected = tuple(pd.Timestamp(x) for x in eligible.index[-72:])
    train = work[starts.isin(selected)].copy()
    if len(train) != EXPECTED_TRAIN_CASES:
        raise RuntimeError(
            f"Fixed two-site planning history changed: {len(train)} cases != {EXPECTED_TRAIN_CASES}"
        )
    site_counts = train[Col.SITE].value_counts().to_dict()
    if site_counts != EXPECTED_TRAIN_SITE_COUNTS:
        raise RuntimeError(
            f"Two-site training composition changed: {site_counts} != {EXPECTED_TRAIN_SITE_COUNTS}"
        )
    return PlanningContext(
        train=train,
        history=fit_service_room_history(train),
        week_starts=selected,
    )


def final_build_eligibility_maps(df_preholdout: pd.DataFrame, config: Config):
    # Compatibility is embedded in PlanningContext and resolved against each
    # fixed roster by build_weekly_instance_with_calendar.
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
            df_scoped,
            start,
            offset + j,
            cfg,
            roster.calendar,
            candidate_pools.history,
            PRIMARY_ELIGIBILITY_WEEKS,
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
    return CostConfig(
        overtime_per_minute=s.overtime,
        idle_per_minute=s.idle,
        deferral_per_case=1e12,
    )


def fixed_signature(col: ScheduleColumn, inst: WeeklyInstance) -> str:
    """Canonicalize schedules under exactly the fixed planner's block symmetry."""

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
    cfg = SolverConfig(
        time_limit_seconds=max(1, int(time_limit)),
        mip_gap=max(0.0, float(mip_gap)),
        threads=max(1, int(threads)),
        verbose=bool(s.verbose),
        mip_gap_abs=1e-10,
        seed=int(s.random_seed),
    )
    t0 = base.time.perf_counter()
    result = solve_fixed_capacity_assignment(
        week.instance,
        np.asarray(durations, float),
        final_cost_cfg(s),
        PRIMARY_TURNOVER,
        cfg,
        objective_mode=PRIMARY_OBJECTIVE,
        warm_start=warm,
        symmetry_breaking=True,
    )
    if result.column is None or result.phi_ub is None or result.phi_lb is None:
        raise RuntimeError(f"Week {week.position} {label}: fixed planner returned no incumbent/bound")
    psi_ub = float(result.psi_ub)
    psi_lb = float(result.psi_lb)
    native_gap = base.rel_gap(psi_ub, psi_lb)
    exact = bool(result.diagnostics.proven_optimal and abs(psi_ub - psi_lb) <= 1e-6)
    return base.PlanResult(
        week=week.position,
        column=result.column,
        objective=float(result.phi_ub),
        bound=float(result.phi_lb),
        gap=float(native_gap),
        status=result.diagnostics.status,
        solve_seconds=base.time.perf_counter() - t0,
        exact=exact,
        tiebreak_used=False,
    )


class FinalConvexPDCASubproblem:
    """The pDCA majorizer with fixed-schedule turnover included in kappa."""

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
        self.P, self.N, self.delta = {}, {}, {}
        for i in range(n):
            row = X.getrow(i)
            de = base.gp.LinExpr(row.data.tolist(), [self.w[int(j)] for j in row.indices])
            self.delta[i] = de
            m.addConstr(de <= s.display_cap)
            m.addConstr(de >= -s.display_cap)
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
                    kappa = (
                        sum(float(a.booked[g]) for g in globals_)
                        + turnover_const
                        - float(col.block_capacities[bid])
                    )
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
    """Audit direct Phi against reduced Psi on two pooled training weeks."""

    rows = []
    for w in weeks[: min(2, len(weeks))]:
        d = np.asarray(w.instance.booked_durations(), float)
        settings = SolverConfig(
            time_limit_seconds=60,
            mip_gap=0.01,
            threads=1,
            verbose=False,
            mip_gap_abs=1e-10,
            seed=s.random_seed,
        )
        psi = solve_fixed_capacity_assignment(
            w.instance, d, final_cost_cfg(s), PRIMARY_TURNOVER, settings, "psi"
        )
        phi = solve_fixed_capacity_assignment(
            w.instance, d, final_cost_cfg(s), PRIMARY_TURNOVER, settings, "phi"
        )
        if psi.column is None or phi.column is None:
            rows.append({"week": w.position, "passed": False, "reason": "missing incumbent"})
            continue
        psi_cost = schedule_metrics(psi.column, d, final_cost_cfg(s), PRIMARY_TURNOVER)["phi"]
        phi_cost = schedule_metrics(phi.column, d, final_cost_cfg(s), PRIMARY_TURNOVER)["phi"]
        intervals_overlap = not (
            float(psi.phi_ub) < float(phi.phi_lb) - 1e-6
            or float(phi.phi_ub) < float(psi.phi_lb) - 1e-6
        )
        rows.append(
            {
                "week": w.position,
                "sites": "+".join(PRIMARY_SITES),
                "psi_schedule_phi": psi_cost,
                "phi_schedule_phi": phi_cost,
                "psi_phi_lb": psi.phi_lb,
                "psi_phi_ub": psi.phi_ub,
                "phi_lb": phi.phi_lb,
                "phi_ub": phi.phi_ub,
                "identity_error": psi.metrics["identity_error"],
                "passed": bool(intervals_overlap and psi.metrics["identity_error"] <= 1e-6),
            }
        )
    return rows


def install_final_adapter() -> None:
    """Replace only the pre-audit layer used by the existing experiment engine."""

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
    base.ConvexPDCASubproblem = FinalConvexPDCASubproblem
    base.stable_solver_audit = final_solver_audit


if __name__ == "__main__":
    install_final_adapter()
    base.main()
