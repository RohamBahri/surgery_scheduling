"""Scientific specification layer for the final two-site paper experiment.

These changes are deliberate research-design choices, not runtime bug fixes:

* define the planning cohort using booked duration only (<=480 minutes); long
  realized durations remain in the outcome and can generate overtime;
* use the training-only median-count fixed roster as the primary capacity proxy;
* use only predecision case features (no actual-day/month calendar features);
* enlarge the coefficient box to +/-100 while retaining per-case display/safety
  constraints;
* scale L1 penalties to each method's own zero-policy loss scale;
* exploit exact site separability in the VF schedule library by storing TGH and
  TWH surfaces separately and taking their Cartesian-product minimum implicitly;
* order weekly cases by predecision fields before model construction.

The deployment tie rule is intentionally cheap: one single-threaded solve with
fixed seed and fixed predecision model ordering.  There is no second tie-break
MIP and no realized-outcome information in the rule.  Final evaluation records
and saves the resulting assignments so this operational selection is auditable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.config import Config
from src.core.types import Col, Domain
from src.planning.eligibility import fit_service_room_history
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.roster import build_fixed_roster


SCIENTIFIC_SPEC_VERSION = "final_paper_scientific_spec_2026_09_23_v1"
PRIMARY_ROSTER = "median_count_template"
PLANNING_CASE_LIMIT_MINUTES = 480.0
EXPECTED_CLEANED_CASES = 32397
EXPECTED_SCOPED_CASES = 30363
EXPECTED_TRAIN_CASES = 21033
EXPECTED_HOLDOUT_CASES = 6713
EXPECTED_TRAIN_SITE_COUNTS = {"TWH": 11394, "TGH": 9639}
EXPECTED_HOLDOUT_SITE_COUNTS = {"TWH": 3849, "TGH": 2864}
EXPECTED_PREHOLDOUT_SITE_COUNTS = {"TWH": 12797, "TGH": 10853, "PMH": 1}
EXPECTED_FEATURES = 107
COEFFICIENT_BOUND = 100.0
DEPLOYMENT_TIE_RULE = (
    "single-pass deterministic solver selection: fixed predecision case/block ordering, "
    "Gurobi Threads=1, fixed Seed=42; no realized-outcome secondary objective and no second MIP"
)

# Method-specific regularization values are populated during training and written
# into the frozen training bundle.
REGULARIZATION_AUDIT: dict[str, dict[str, float]] = {}

# Capture the original base implementations before monkeypatching.
_LegacyFixedSpec = base.FixedSpec
_legacy_library_metrics = base.library_metrics


@dataclass
class ScientificFinalSettings(final.FinalSettings):
    expected_train_cases: int = EXPECTED_TRAIN_CASES
    expected_holdout_cases: int = EXPECTED_HOLDOUT_CASES
    coefficient_bound: float = COEFFICIENT_BOUND

    def validate(self) -> None:
        super().validate()
        if abs(float(self.coefficient_bound) - COEFFICIENT_BOUND) > 1e-12:
            raise ValueError("Final coefficient box is frozen to +/-100.")


class ScientificFeatureEncoder:
    """Training-only pooled encoder using only predecision case information."""

    PREFIX_TO_COLUMN = {
        "service": Col.CASE_SERVICE,
        "surgeon": Col.SURGEON_CODE,
        "procedure": Col.PROCEDURE_ID,
        "site": Col.SITE,
    }
    BASE_FEATURE_NAMES = ("bias", "booked_z")

    def __init__(self) -> None:
        self.feature_names: list[str] = list(self.BASE_FEATURE_NAMES)
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

    def fit(self, frame: pd.DataFrame) -> "ScientificFeatureEncoder":
        b = pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce").to_numpy(float)
        self.booked_mean = float(np.nanmean(b))
        self.booked_std = float(np.nanstd(b))
        if not np.isfinite(self.booked_std) or self.booked_std <= 1e-12:
            self.booked_std = 1.0

        names = list(self.BASE_FEATURE_NAMES)
        for prefix, column in self.PREFIX_TO_COLUMN.items():
            values = frame[column].map(self.canon)
            ranked = self._rank_levels(values)
            if prefix == "site":
                if set(ranked) != set(final.PRIMARY_SITES):
                    raise RuntimeError(
                        f"Training feature scope has sites {ranked}, expected {final.PRIMARY_SITES}"
                    )
                reference = ranked[0]
                explicit = [v for v in ranked if v != reference]
                selected = list(ranked)
                if len(explicit) != final.FEATURE_FAMILY_BUDGETS[prefix]:
                    raise AssertionError("Two-site feature budget changed")
                names.extend(f"site_{v}" for v in explicit)
            else:
                budget = final.FEATURE_FAMILY_BUDGETS[prefix]
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

        if len(names) != EXPECTED_FEATURES:
            raise AssertionError(
                f"Final predecision feature schema has p={len(names)}, expected {EXPECTED_FEATURES}"
            )
        self.feature_names = names
        self.fitted = True
        return self

    def transform_frame(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.fitted:
            raise RuntimeError("feature encoder not fitted")
        n = len(frame)
        booked = (
            pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce")
            .fillna(self.booked_mean)
            .to_numpy(float)
        )
        cols: list[np.ndarray] = [
            np.ones(n),
            (booked - self.booked_mean) / self.booked_std,
        ]
        names = list(self.BASE_FEATURE_NAMES)
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
            raise AssertionError("Final feature schema reconstruction failed")
        return sparse.csr_matrix(np.column_stack(cols), dtype=float)

    def transform_cases(self, cases: Sequence[Any]) -> sparse.csr_matrix:
        frame = pd.DataFrame(
            {
                Col.BOOKED_MINUTES: [float(c.booked_duration_min) for c in cases],
                Col.CASE_SERVICE: [str(c.service) for c in cases],
                Col.SURGEON_CODE: [str(c.surgeon_code) for c in cases],
                Col.PROCEDURE_ID: [str(c.procedure_id) for c in cases],
                Col.SITE: [str(c.site) for c in cases],
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
            "explicit_levels": self.explicit_levels,
            "feature_family_budgets": final.FEATURE_FAMILY_BUDGETS,
            "selection_rule": "training-only frequency ranking; count descending, lexical tie break",
            "calendar_features": "none",
            "sites": list(final.PRIMARY_SITES),
            "p": len(self.feature_names),
            "minimum_recommended_duration": final.MIN_RECOMMENDED_DURATION,
            "out_of_sample_display_rule": (
                "clip Xw to [-display_cap, display_cap] and enforce the configured "
                "minimum recommended duration"
            ),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ScientificFeatureEncoder":
        obj = cls()
        obj.feature_names = [str(x) for x in payload["feature_names"]]
        if len(obj.feature_names) != EXPECTED_FEATURES:
            raise RuntimeError("Frozen feature manifest dimension does not match final specification")
        obj.booked_mean = float(payload["booked_mean"])
        obj.booked_std = float(payload["booked_std"])
        obj.references = {str(k): str(v) for k, v in dict(payload["references"]).items()}
        obj.selected_levels = {
            str(k): [str(x) for x in v]
            for k, v in dict(payload["selected_levels"]).items()
        }
        if "explicit_levels" in payload:
            obj.explicit_levels = {
                str(k): [str(x) for x in v]
                for k, v in dict(payload["explicit_levels"]).items()
            }
        else:
            obj.explicit_levels = {
                k: [x for x in vals if x != obj.references[k]]
                for k, vals in obj.selected_levels.items()
            }
        obj.fitted = True
        return obj


def scientific_load_data(config: Config, *args, **kwargs) -> pd.DataFrame:
    """Canonical cleaning, but the 480-minute eligibility cap uses booking only.

    The legacy loader applies the same 480-minute cap to booked and realized
    room/surgical durations.  For the final paper we temporarily disable that
    cap during canonical cleaning, then apply it only to the predecision booked
    duration.  Positive/timestamp/cancellation/emergency rules are unchanged.
    """

    kwargs["site_history_end"] = final.HOLDOUT_BOUNDARY
    old_limit = Domain.MAX_PLANNING_CASE_MINUTES
    try:
        Domain.MAX_PLANNING_CASE_MINUTES = float("inf")
        frame = final.canonical_load_data(config, *args, **kwargs)
    finally:
        Domain.MAX_PLANNING_CASE_MINUTES = old_limit

    booked = pd.to_numeric(frame[Col.BOOKED_MINUTES], errors="coerce")
    frame = frame[(booked > 0) & (booked <= PLANNING_CASE_LIMIT_MINUTES)].copy()
    if len(frame) != EXPECTED_CLEANED_CASES:
        raise RuntimeError(
            f"Booked-only cleaned cohort changed: {len(frame)} != {EXPECTED_CLEANED_CASES}"
        )

    dt = pd.to_datetime(frame[Col.ACTUAL_START], errors="coerce")
    pre = frame[(dt < final.HOLDOUT_BOUNDARY) & dt.dt.weekday.isin(range(5))].copy()
    counts = pre[Col.SITE].value_counts().to_dict()
    ranking = tuple(str(x) for x in pre[Col.SITE].value_counts().index[:2])
    if ranking != final.EXPECTED_SITE_RANKING:
        raise RuntimeError(
            f"Two largest pre-holdout weekday sites changed: {ranking}; "
            f"expected {final.EXPECTED_SITE_RANKING}"
        )
    for site, expected in EXPECTED_PREHOLDOUT_SITE_COUNTS.items():
        if int(counts.get(site, 0)) != expected:
            raise RuntimeError(
                f"Pre-holdout site count changed for {site}: {counts.get(site, 0)} != {expected}"
            )

    # Restore raw categories for leakage-safe training-only vocabulary fitting.
    frame[Col.CASE_SERVICE] = frame[Col.CASE_SERVICE_RAW]
    frame[Col.SURGEON_CODE] = frame[Col.SURGEON_CODE_RAW]
    frame[Col.PROCEDURE_ID] = frame[Col.PROCEDURE_ID_RAW]
    base.LOG.info(
        "[DATA SPEC] booked-only <=480 filter retained %d cases; long realized outcomes are retained",
        len(frame),
    )
    return frame


def scientific_build_config(s: ScientificFinalSettings) -> Config:
    return final.final_build_config(s)


def scientific_build_candidate_pools(df_preholdout: pd.DataFrame, config: Config) -> final.PlanningContext:
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
            f"Fixed two-site planning history changed: {len(train)} != {EXPECTED_TRAIN_CASES}"
        )
    site_counts = {str(k): int(v) for k, v in train[Col.SITE].value_counts().to_dict().items()}
    if site_counts != EXPECTED_TRAIN_SITE_COUNTS:
        raise RuntimeError(
            f"Two-site training composition changed: {site_counts} != {EXPECTED_TRAIN_SITE_COUNTS}"
        )
    return final.PlanningContext(
        train=train,
        history=fit_service_room_history(train),
        week_starts=selected,
    )


def _predecision_ordered_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable order using only variables available before surgery occurs."""

    work = frame.copy()
    fields = [
        Col.PATIENT_ID,
        Col.SITE,
        Col.CASE_SERVICE_RAW,
        Col.SURGEON_CODE_RAW,
        Col.PROCEDURE_ID_RAW,
        Col.BOOKED_MINUTES,
    ]
    temp = []
    for j, col in enumerate(fields):
        key = f"__preorder_{j}"
        temp.append(key)
        if col == Col.BOOKED_MINUTES:
            work[key] = pd.to_numeric(work[col], errors="coerce").fillna(-1.0)
        else:
            work[key] = work[col].fillna("").astype(str)
    work = work.sort_values(temp, kind="mergesort").drop(columns=temp)
    return work


def scientific_build_bundles(
    df_scoped: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    *,
    cfg: Config,
    candidate_pools: final.PlanningContext,
    eligibility_maps: Any,
    offset: int,
) -> list[base.WeekBundle]:
    if not isinstance(candidate_pools, final.PlanningContext):
        raise TypeError("Final experiment requires the frozen PlanningContext")
    ordered_pool = _predecision_ordered_pool(df_scoped)
    out: list[base.WeekBundle] = []
    for j, start in enumerate(starts):
        start = pd.Timestamp(start).normalize()
        roster = build_fixed_roster(candidate_pools.train, start, cfg, PRIMARY_ROSTER)
        roster_sites = {b.site for b in roster.calendar.candidates}
        if roster_sites != set(final.PRIMARY_SITES):
            raise RuntimeError(f"Roster sites changed at {start.date()}: {roster_sites}")
        inst = build_weekly_instance_with_calendar(
            ordered_pool,
            start,
            offset + j,
            cfg,
            roster.calendar,
            candidate_pools.history,
            final.PRIMARY_ELIGIBILITY_WEEKS,
        )
        if inst.num_cases == 0:
            raise RuntimeError(f"Empty selected week {start.date()}")
        missing = [i for i in range(inst.num_cases) if not inst.case_eligible_blocks.get(i)]
        if missing:
            raise RuntimeError(
                f"Week {start.date()} has {len(missing)} cases without fixed-roster eligibility"
            )
        for i, case in enumerate(inst.cases):
            if any(bid.site != case.site for bid in inst.case_eligible_blocks[i]):
                raise AssertionError(f"Cross-site eligibility detected for case {case.case_id}")
        out.append(base.WeekBundle(offset + j, start, inst))
    return out


class DecomposedScheduleLibrary:
    """Per-site schedule surfaces with an implicit Cartesian-product library.

    Planning is exactly additive across TGH and TWH.  Storing site surfaces
    separately lets ``best`` choose each site's best known surface independently,
    which is weakly tighter than restricting the value function to pooled columns
    that happened to be generated together.  No additional MILP solves are used.
    """

    def __init__(self, week_lookup: Mapping[int, base.WeekBundle]):
        self.week_lookup = dict(week_lookup)
        self.data: dict[int, dict[str, dict[str, base.LibraryEntry]]] = {}
        self.pinned: dict[int, dict[str, set[str]]] = {}

    def _site_entries(self, week: int, site: str) -> list[base.LibraryEntry]:
        return list(self.data.get(int(week), {}).get(site, {}).values())

    def add(self, week: int, col: ScheduleColumn, source: str, *, pin: bool = False) -> bool:
        wk = int(week)
        wb = self.week_lookup[wk]
        any_new = False
        for site in final.PRIMARY_SITES:
            view = final._site_view(wb, site)
            local = final._localize_warm(col, view)
            if local is None:
                raise RuntimeError("Cannot add an empty schedule to the decomposed library")
            sig = final.fixed_signature(local, view.instance)
            bucket = self.data.setdefault(wk, {}).setdefault(site, {})
            if sig not in bucket:
                bucket[sig] = base.LibraryEntry(wk, sig, str(source), local)
                any_new = True
            if pin:
                self.pinned.setdefault(wk, {}).setdefault(site, set()).add(sig)
        return any_new

    def add_plans(
        self,
        plans: Mapping[int, base.PlanResult],
        source: str,
        *,
        pin: bool = False,
    ) -> int:
        return sum(int(self.add(w, r.column, source, pin=pin)) for w, r in plans.items())

    def entries(self, week: int) -> list[base.LibraryEntry]:
        out: list[base.LibraryEntry] = []
        for site in final.PRIMARY_SITES:
            out.extend(self._site_entries(int(week), site))
        return out

    def size(self, week: int | None = None) -> int:
        if week is not None:
            return sum(len(self.data.get(int(week), {}).get(site, {})) for site in final.PRIMARY_SITES)
        return sum(
            len(bucket)
            for by_site in self.data.values()
            for bucket in by_site.values()
        )

    def best(
        self,
        week: int,
        durations: np.ndarray,
        s: ScientificFinalSettings,
    ) -> tuple[ScheduleColumn, float, str]:
        wk = int(week)
        wb = self.week_lookup[wk]
        d = np.asarray(durations, dtype=float)
        if d.shape != (wb.instance.num_cases,):
            raise ValueError("Pooled library duration vector has the wrong shape")
        parts = []
        total = 0.0
        sources = []
        for site in final.PRIMARY_SITES:
            view = final._site_view(wb, site)
            local_d = d[view.global_indices]
            best_entry = None
            best_value = math.inf
            for entry in self._site_entries(wk, site):
                value = float(
                    entry.column.compute_cost(local_d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
                )
                if value < best_value - 1e-9:
                    best_value = value
                    best_entry = entry
            if best_entry is None:
                raise RuntimeError(f"Empty site library: week={wk} site={site}")
            parts.append((view, best_entry.column))
            total += best_value
            sources.append(f"{site}:{best_entry.source}")
        merged = final._merge_site_columns(wb.instance, parts)
        return merged, float(total), "|".join(sources)

    def check_pins(self) -> bool:
        for wk, by_site in self.pinned.items():
            for site, sigs in by_site.items():
                if not sigs.issubset(set(self.data.get(wk, {}).get(site, {}))):
                    return False
        return True

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for wk in sorted(self.data):
            counts = {
                site: len(self.data.get(wk, {}).get(site, {}))
                for site in final.PRIMARY_SITES
            }
            product = int(np.prod([max(1, counts[s]) for s in final.PRIMARY_SITES]))
            for site in final.PRIMARY_SITES:
                srcs: dict[str, int] = {}
                for entry in self._site_entries(wk, site):
                    srcs[entry.source] = srcs.get(entry.source, 0) + 1
                rows.append(
                    {
                        "week": wk,
                        "site": site,
                        "n_site_surfaces": counts[site],
                        "implicit_joint_combinations": product,
                        "pinned": len(self.pinned.get(wk, {}).get(site, set())),
                        "sources": json.dumps(srcs, sort_keys=True),
                    }
                )
        return rows


class ScientificFixedSpec(_LegacyFixedSpec):
    """Use the same relative L1 strength eta on each method's own loss scale."""

    def __init__(
        self,
        name: str,
        arrays: base.Arrays,
        p_plus: np.ndarray,
        p_minus: np.ndarray,
        fixed: dict[int, ScheduleColumn] | None,
        lam: float,
        settings: ScientificFinalSettings,
    ) -> None:
        if name in {"RA", "OS"}:
            super().__init__(name, arrays, p_plus, p_minus, fixed, 0.0, settings)
            zero = np.zeros(arrays.p, dtype=float)
            loss0 = float(super().value(zero))
            effective = float(settings.l1_eta) * loss0 / arrays.p
            self.lam = effective
            REGULARIZATION_AUDIT[name] = {
                "zero_policy_unregularized_loss": loss0,
                "lambda": effective,
                "eta": float(settings.l1_eta),
                "p": float(arrays.p),
            }
        elif name.startswith("VF_OUTER") and hasattr(settings, "_vf_lambda"):
            super().__init__(
                name,
                arrays,
                p_plus,
                p_minus,
                fixed,
                float(getattr(settings, "_vf_lambda")),
                settings,
            )
        else:
            super().__init__(name, arrays, p_plus, p_minus, fixed, lam, settings)


def scientific_train_naive(
    a: base.Arrays, s: ScientificFinalSettings, ignored_lam: float
) -> np.ndarray:
    loss0 = float(np.sum(np.abs(a.error)) / a.n_weeks)
    lam = float(s.l1_eta) * loss0 / a.p
    REGULARIZATION_AUDIT["NAIVE"] = {
        "zero_policy_unregularized_loss": loss0,
        "lambda": lam,
        "eta": float(s.l1_eta),
        "p": float(a.p),
    }
    base.LOG.info("[REG] NAIVE L0=%.3f lambda=%.6f", loss0, lam)
    return final.final_train_naive(a, s, lam)


def _vf_zero_pure_certificate(
    a: base.Arrays,
    lib: DecomposedScheduleLibrary,
    oracle_lb: Mapping[int, float],
    s: ScientificFinalSettings,
) -> float:
    zero = np.zeros(a.p, dtype=float)
    _, corr, planning = base.correction_and_planning(zero, a, s)
    case = float(np.sum(base.theta(a.error - corr, s.overtime, s.idle)) / a.n_weeks)
    pred = 0.0
    for wk, idx in a.week_slices.items():
        pred += lib.best(wk, planning[idx], s)[1]
    pred /= a.n_weeks
    oracle = float(np.mean([oracle_lb[w] for w in a.week_slices]))
    return case + pred - oracle


def scientific_library_metrics(
    w: np.ndarray,
    a: base.Arrays,
    lib: DecomposedScheduleLibrary,
    oracle_lb: Mapping[int, float],
    s: ScientificFinalSettings,
    ignored_lam: float,
) -> dict[str, float]:
    if not hasattr(s, "_vf_lambda"):
        loss0 = _vf_zero_pure_certificate(a, lib, oracle_lb, s)
        lam = float(s.l1_eta) * loss0 / a.p
        setattr(s, "_vf_lambda", lam)
        REGULARIZATION_AUDIT["VF"] = {
            "zero_policy_unregularized_loss": float(loss0),
            "lambda": float(lam),
            "eta": float(s.l1_eta),
            "p": float(a.p),
        }
        base.LOG.info("[REG] VF L0=%.3f lambda=%.6f", loss0, lam)
    return _legacy_library_metrics(
        w, a, lib, oracle_lb, s, float(getattr(s, "_vf_lambda"))
    )


def apply_scientific_fixes() -> None:
    """Install the agreed scientific specification after the final adapter."""

    final.PRIMARY_ROSTER = PRIMARY_ROSTER
    final.EXPECTED_TRAIN_CASES = EXPECTED_TRAIN_CASES
    final.EXPECTED_HOLDOUT_CASES = EXPECTED_HOLDOUT_CASES
    final.EXPECTED_TRAIN_SITE_COUNTS = dict(EXPECTED_TRAIN_SITE_COUNTS)
    final.SCRIPT_VERSION = SCIENTIFIC_SPEC_VERSION
    final.FinalSettings = ScientificFinalSettings
    final.FinalFeatureEncoder = ScientificFeatureEncoder

    base.SCRIPT_VERSION = SCIENTIFIC_SPEC_VERSION
    base.Settings = ScientificFinalSettings
    base.FrozenFeatureEncoder = ScientificFeatureEncoder
    base.load_data = scientific_load_data
    base.build_config = scientific_build_config
    base.build_candidate_pools = scientific_build_candidate_pools
    base.build_eligibility_maps = final.final_build_eligibility_maps
    base.build_bundles = scientific_build_bundles
    base.ScheduleLibrary = DecomposedScheduleLibrary
    base.FixedSpec = ScientificFixedSpec
    base.train_naive = scientific_train_naive
    base.library_metrics = scientific_library_metrics
    base.FINAL_SCIENTIFIC_SPEC_VERSION = SCIENTIFIC_SPEC_VERSION
    base.DEPLOYMENT_TIE_RULE = DEPLOYMENT_TIE_RULE
