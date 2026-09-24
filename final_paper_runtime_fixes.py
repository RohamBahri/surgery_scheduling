"""Hardening layer for the frozen two-site final-paper pipeline.

This module contains only fixes that are unambiguous from the current
mathematics/specification:

1. correct SciPy CSR dtype construction in RA cross-fitting;
2. make the saturation stress box respect the short-case duration floor;
3. isolate concurrent Gurobi solves in separate OS processes (one default
   Gurobi environment per process, rather than sharing the default environment
   across Python threads);
4. keep oracle retry gaps on the native reduced-Psi scale while incumbents and
   bounds remain reported on the shifted Phi scale.

It deliberately does *not* change scientific choices such as cohort inclusion,
regularization strength, feature families, capacity templates, or tie semantics.
Those require an explicit research decision rather than a silent code patch.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn


RUNTIME_FIXES_VERSION = "final_paper_runtime_fixes_2026_09_23_v1"
MIN_RECOMMENDED_DURATION = final.MIN_RECOMMENDED_DURATION


def safe_crossfit_pi(
    a: base.Arrays, labels: np.ndarray, s: final.FinalSettings
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-fit exposure probabilities with an explicit CSR dtype.

    ``csr_matrix(a.X, float)`` passes ``float`` as the positional ``shape``
    argument in SciPy.  The intended constructor is ``dtype=float``.
    """

    X = sparse.csr_matrix(a.X, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(a.week_ids)
    if X.shape[0] != len(y) or len(groups) != len(y):
        raise ValueError("crossfit_pi input lengths do not agree")

    unique_groups = np.unique(groups)
    pred = np.zeros(len(y), dtype=float)
    if len(unique_groups) < 2:
        pred[:] = float(np.mean(y)) if len(y) else 0.5
    else:
        k = min(5, len(unique_groups))
        splitter = GroupKFold(n_splits=k)
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
            if np.unique(y[tr]).size < 2:
                pred[te] = float(np.mean(y[tr]))
            else:
                mdl = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    fit_intercept=False,
                    random_state=s.random_seed + fold,
                )
                mdl.fit(X[tr], y[tr])
                pred[te] = mdl.predict_proba(X[te])[:, 1]

    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    prevalence = float(np.mean(y)) if len(y) else math.nan
    metrics = {
        "prevalence": prevalence,
        "pi_mean": float(np.mean(pred)) if len(pred) else math.nan,
        "brier": float(brier_score_loss(y, pred)) if len(y) else math.nan,
        "constant_brier": float(np.mean((y - prevalence) ** 2)) if len(y) else math.nan,
        "auc": float(roc_auc_score(y, pred)) if np.unique(y).size == 2 else math.nan,
    }
    return pred, metrics


def safe_saturation_correction(
    draw: int,
    booked: np.ndarray,
    rng: np.random.Generator,
    s: final.FinalSettings,
) -> np.ndarray:
    """Sample a safe *implemented-correction* outer box case by case.

    The final deployment rule requires the displayed recommended duration to be
    at least ``MIN_RECOMMENDED_DURATION``.  Under the primary symmetric response,
    the most negative implementable correction for case i is therefore

        -alpha * min(h, booked_i - MIN_RECOMMENDED_DURATION),

    while the positive extreme remains ``+alpha*h``.  This is an outer-box
    stress test; it is not a claim that one shared linear policy can attain every
    sampled Cartesian combination.
    """

    b = np.asarray(booked, dtype=float)
    if b.ndim != 1 or np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("booked durations must be positive and finite")
    upper = np.full(len(b), float(s.alpha * s.h), dtype=float)
    available_down = np.maximum(b - MIN_RECOMMENDED_DURATION, 0.0)
    lower = -float(s.alpha) * np.minimum(float(s.h), available_down)

    if draw == 0:
        return np.zeros(len(b), dtype=float)
    if draw == 1:
        return upper.copy()
    if draw == 2:
        return lower.copy()
    if draw == 3:
        return np.where(np.arange(len(b)) % 2 == 0, upper, lower)
    if draw == 4:
        return np.where(np.arange(len(b)) % 2 == 0, lower, upper)

    mode = draw % 3
    if mode == 0:
        choose_upper = rng.random(len(b)) < 0.5
        return np.where(choose_upper, upper, lower)
    if mode == 1:
        return rng.uniform(lower, upper)

    mask = rng.random(len(b)) < 0.25
    out = np.zeros(len(b), dtype=float)
    choose_upper = rng.random(len(b)) < 0.5
    endpoints = np.where(choose_upper, upper, lower)
    out[mask] = endpoints[mask]
    return out


def _solve_process_task(
    week: base.WeekBundle,
    durations: np.ndarray,
    s: final.FinalSettings,
    *,
    seconds: int,
    gap: float,
    warm: ScheduleColumn | None,
    label: str,
    deterministic_tiebreak: bool,
    tiebreak_seconds: int | None,
) -> base.PlanResult:
    """Top-level process worker; each OS process owns its Gurobi default env."""

    return final.final_solve_week(
        week,
        np.asarray(durations, dtype=float),
        s,
        time_limit=seconds,
        mip_gap=gap,
        threads=1,
        warm=warm,
        label=label,
        deterministic_tiebreak=deterministic_tiebreak,
        tiebreak_seconds=tiebreak_seconds,
    )


def safe_solve_batch(
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s: final.FinalSettings,
    *,
    seconds: int,
    gap: float,
    label: str,
    warm_by_week: Mapping[int, ScheduleColumn] | None = None,
    deterministic_tiebreak: bool = False,
    tiebreak_seconds: int | None = None,
) -> dict[int, base.PlanResult]:
    """Run weekly Gurobi jobs in separate processes, not shared-env threads."""

    out: dict[int, base.PlanResult] = {}
    if not weeks:
        return out
    workers = min(int(s.cores), len(weeks))
    base.LOG.info(
        "[PLANNER] %s | weeks=%d process_workers=%d limit=%ss gap=%.4g",
        label,
        len(weeks),
        workers,
        seconds,
        gap,
    )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {
            ex.submit(
                _solve_process_task,
                w,
                np.asarray(duration_by_week[w.position], dtype=float),
                s,
                seconds=seconds,
                gap=gap,
                warm=None if warm_by_week is None else warm_by_week.get(w.position),
                label=label,
                deterministic_tiebreak=deterministic_tiebreak,
                tiebreak_seconds=tiebreak_seconds,
            ): w.position
            for w in weeks
        }
        done = 0
        for f in as_completed(fut):
            pos = fut[f]
            out[pos] = f.result()
            done += 1
            if done == 1 or done % 10 == 0 or done == len(fut):
                gaps = [r.gap for r in out.values() if np.isfinite(r.gap)]
                base.LOG.info(
                    "[PLANNER] %s %d/%d | elapsed=%.1f min | max_native_psi_gap=%s",
                    label,
                    done,
                    len(fut),
                    (time.perf_counter() - t0) / 60.0,
                    "NA" if not gaps else f"{max(gaps):.3%}",
                )
    return out


def _psi_constant(
    week: base.WeekBundle, durations: np.ndarray, s: final.FinalSettings
) -> float:
    """Return K(d) in Phi = K(d) + Psi for the fixed-capacity pooled week."""

    d = np.asarray(durations, dtype=float)
    total_capacity = float(
        sum(float(b.capacity_minutes) for b in week.instance.calendar.candidates)
    )
    return float(s.idle) * (
        total_capacity - float(d.sum()) - float(s.turnover) * week.instance.num_cases
    )


def safe_solve_oracle_batch(
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s: final.FinalSettings,
    *,
    label: str,
) -> dict[int, base.PlanResult]:
    """Two-stage oracle solve with retry merging on one consistent gap scale."""

    first = min(600, int(s.oracle_seconds))
    out = safe_solve_batch(
        weeks,
        duration_by_week,
        s,
        seconds=first,
        gap=s.oracle_gap,
        label=f"{label}_pass1",
    )
    unresolved = [
        w
        for w in weeks
        if (not out[w.position].exact) and out[w.position].gap > float(s.oracle_gap)
    ]
    if unresolved and s.oracle_seconds > first:
        base.LOG.info(
            "[ORACLE] %s retrying %d unresolved weeks up to %ss",
            label,
            len(unresolved),
            s.oracle_seconds,
        )
        warm = {w.position: out[w.position].column for w in unresolved}
        subset_map = {w.position: duration_by_week[w.position] for w in unresolved}
        retry = safe_solve_batch(
            unresolved,
            subset_map,
            s,
            seconds=s.oracle_seconds,
            gap=s.oracle_gap,
            label=f"{label}_retry",
            warm_by_week=warm,
        )
        for w in unresolved:
            old = out[w.position]
            new = retry[w.position]
            chosen = new if new.objective <= old.objective + 1e-9 else old
            phi_ub = min(float(old.objective), float(new.objective))
            phi_lb = max(float(old.bound), float(new.bound))
            k = _psi_constant(w, duration_by_week[w.position], s)
            psi_ub = phi_ub - k
            psi_lb = phi_lb - k
            native_gap = base.rel_gap(psi_ub, psi_lb)
            # Do not relabel a time-limited 0.1%-gap solve as mathematically
            # exact. Exactness remains the conservative status from a constituent
            # solve; the 0.1% quantity is a paper-readiness gap target.
            exact = bool(old.exact or new.exact)
            out[w.position] = base.PlanResult(
                week=w.position,
                column=chosen.column,
                objective=phi_ub,
                bound=phi_lb,
                gap=float(native_gap),
                status=new.status,
                solve_seconds=float(old.solve_seconds + new.solve_seconds),
                exact=exact,
                tiebreak_used=bool(chosen.tiebreak_used),
            )
    return out


def safe_saturation_test(
    weeks: Sequence[base.WeekBundle],
    a: base.Arrays,
    lib: base.ScheduleLibrary,
    s: final.FinalSettings,
    root: Path,
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Saturation stress test over the case-specific safe implemented box."""

    rngs = {
        w.position: np.random.default_rng(s.random_seed + 100003 * w.position)
        for w in weeks
    }
    active = {w.position: True for w in weeks}
    no_new = {w.position: 0 for w in weeks}
    draws = {w.position: 0 for w in weeks}
    rows: list[dict[str, Any]] = []
    batch_id = 0

    workers = min(int(s.cores), max(1, len(weeks) * int(s.saturation_batch)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        while any(active.values()):
            if base.remaining(deadline) < 60:
                break
            batch_id += 1
            tasks = []
            for w in weeks:
                if not active[w.position]:
                    continue
                for _ in range(s.saturation_batch):
                    if draws[w.position] >= s.saturation_draws_per_week:
                        active[w.position] = False
                        break
                    draw = draws[w.position]
                    draws[w.position] += 1
                    idx = a.week_slices[w.position]
                    booked = np.asarray(a.booked[idx], dtype=float)
                    corr = safe_saturation_correction(draw, booked, rngs[w.position], s)
                    d = booked + corr
                    if np.any(~np.isfinite(d)) or np.any(d <= 0):
                        raise AssertionError("safe saturation generated a nonpositive duration")
                    warm = lib.best(w.position, d, s)[0]
                    tasks.append((w, draw, d, warm))
            if not tasks:
                break

            base.LOG.info("[SAT] batch %d | solves=%d", batch_id, len(tasks))
            fut = {
                ex.submit(
                    _solve_process_task,
                    w,
                    d,
                    s,
                    seconds=s.saturation_seconds,
                    gap=s.saturation_gap,
                    warm=warm,
                    label=f"sat_b{batch_id}_d{draw}",
                    deterministic_tiebreak=False,
                    tiebreak_seconds=None,
                ): (w, draw)
                for w, draw, d, warm in tasks
            }
            results = []
            for f in as_completed(fut):
                w, draw = fut[f]
                r = f.result()
                before = lib.size(w.position)
                new = lib.add(w.position, r.column, f"SAT_{draw}")
                results.append((w.position, draw, r, new, before))

            byweek: dict[int, int] = {}
            for wk, draw, r, new, before in results:
                rows.append(
                    {
                        "batch": batch_id,
                        "week": wk,
                        "draw": draw,
                        "new_surface": new,
                        "gap": r.gap,
                        "gap_scale": "native_psi",
                        "status": r.status,
                        "pool_before": before,
                        "pool_after": lib.size(wk),
                    }
                )
                byweek[wk] = byweek.get(wk, 0) + int(new)

            for wk in list(active):
                if not active[wk]:
                    continue
                no_new[wk] = no_new[wk] + 1 if byweek.get(wk, 0) == 0 else 0
                if (
                    draws[wk] >= s.saturation_min_draws_per_week
                    and no_new[wk] >= s.saturation_patience_batches
                ):
                    active[wk] = False
                if draws[wk] >= s.saturation_draws_per_week:
                    active[wk] = False

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "SATURATION_DRAWS.csv", index=False)
    all_booked = np.asarray(a.booked, dtype=float)
    lower_all = -float(s.alpha) * np.minimum(
        float(s.h), np.maximum(all_booked - MIN_RECOMMENDED_DURATION, 0.0)
    )
    summary = {
        "outer_box": "case-specific implemented-correction safety box",
        "upper_correction_minutes": float(s.alpha * s.h),
        "lowest_case_lower_correction_minutes": float(np.min(lower_all)),
        "highest_case_lower_correction_minutes": float(np.max(lower_all)),
        "minimum_recommended_duration": float(MIN_RECOMMENDED_DURATION),
        "draws_total": int(sum(draws.values())),
        "draws_min_week": int(min(draws.values())) if draws else 0,
        "draws_max_week": int(max(draws.values())) if draws else 0,
        "new_surfaces": int(frame["new_surface"].sum()) if len(frame) else 0,
        "new_surfaces_last_batch": int(
            frame[frame["batch"] == frame["batch"].max()]["new_surface"].sum()
        )
        if len(frame)
        else 0,
        "weeks_stopped_by_patience": int(
            sum(no_new[w] >= s.saturation_patience_batches for w in no_new)
        ),
        "max_gap": float(frame["gap"].max()) if len(frame) else math.nan,
        "gap_scale": "native_psi",
        "empirical_saturation_strong": bool(
            len(frame) > 0
            and sum(no_new[w] >= s.saturation_patience_batches for w in no_new)
            == len(no_new)
        ),
        "interpretation": (
            "Empirical stress-test evidence only. The case-specific box respects "
            "the recommendation-duration safety floor, but sampling the Cartesian "
            "outer box does not prove reachability by one shared linear policy or "
            "mathematical library completeness over the continuum."
        ),
    }
    base.write_json(root / "SATURATION_SUMMARY.json", summary)
    return rows, summary


def _write_csv_with_scale_metadata(path: Path, rows: Any) -> None:
    """Preserve legacy columns while making mixed oracle scales explicit."""

    if path.name.startswith("ORACLE_") and isinstance(rows, list):
        rows = [
            {
                **dict(row),
                "objective_bound_scale": "phi",
                "gap_scale": "native_psi",
            }
            for row in rows
        ]
    _ORIGINAL_WRITE_CSV(path, rows)


_ORIGINAL_WRITE_CSV = base.write_csv


def apply_runtime_fixes() -> None:
    """Install the hardening layer after ``final.install_final_adapter()``."""

    base.crossfit_pi = safe_crossfit_pi
    base.solve_batch = safe_solve_batch
    base.solve_oracle_batch = safe_solve_oracle_batch
    base.saturation_test = safe_saturation_test
    base.write_csv = _write_csv_with_scale_metadata
    base.FINAL_RUNTIME_FIXES_VERSION = RUNTIME_FIXES_VERSION
