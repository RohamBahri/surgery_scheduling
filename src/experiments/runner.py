"""Rolling-horizon out-of-sample experiment runner.

Loads data, splits into warm-up and pool, builds candidate block pools
from training data, fits every registered method on the warm-up set,
then evaluates each method on successive weekly horizons drawn from the
pool.  Results are collected into a single table and optionally saved
to disk.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.core.config import Config
from src.core.types import Col, KPIResult
from src.data.loader import load_data
from src.data.splits import split_warmup_pool
from src.data.capacity import build_candidate_pools
from src.methods.base import Method
from src.methods.registry import MethodRegistry
from src.planning.evaluation import evaluate
from src.planning.instance import build_weekly_instance

logger = logging.getLogger(__name__)


# ─── Result collection ───────────────────────────────────────────────────────

def _empty_kpi_row(method_name: str, week: int) -> Dict[str, Any]:
    return {
        "method": method_name, "week": week,
        "total_cost": None, "activation_cost": None,
        "overtime_cost": None, "idle_cost": None,
        "deferral_cost": None, "overtime_min": None, "idle_min": None,
        "scheduled": None, "deferred": None, "blocks_opened": None,
        "planned_obj": None, "solve_time": None,
    }


def _kpi_to_row(
    method_name: str, week: int,
    kpi: KPIResult, planned_obj: float | None, solve_time: float,
) -> Dict[str, Any]:
    return {
        "method": method_name,
        "week": week,
        "total_cost": kpi.total_cost,
        "activation_cost": kpi.activation_cost,
        "overtime_cost": kpi.overtime_cost,
        "idle_cost": kpi.idle_cost,
        "deferral_cost": kpi.deferral_cost,
        "overtime_min": kpi.overtime_minutes,
        "idle_min": kpi.idle_minutes,
        "scheduled": kpi.scheduled_count,
        "deferred": kpi.deferred_count,
        "blocks_opened": kpi.blocks_opened,
        "planned_obj": planned_obj,
        "solve_time": solve_time,
    }


# ─── Main runner ─────────────────────────────────────────────────────────────

def run_experiment(
    registry: MethodRegistry,
    config: Config,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the full rolling-horizon experiment.

    Parameters
    ----------
    registry : MethodRegistry
        The set of methods to compare.
    config : Config
        Experiment configuration.
    output_dir : path-like, optional
        If given, write per-horizon CSV and aggregate summary JSON here.

    Returns
    -------
    DataFrame
        One row per (method, week) with all KPI columns.
    """
    # ── Load and split ───────────────────────────────────────────────────
    df = load_data(config)
    df_warmup, df_pool, pool_start = split_warmup_pool(df, config)

    if df_pool.empty:
        logger.error("Scheduling pool is empty — nothing to evaluate.")
        return pd.DataFrame()

    # ── Build candidate block pools from training data (once) ────────────
    candidate_pools = build_candidate_pools(df_warmup, config)

    # ── Fit every method on warm-up data ─────────────────────────────────
    logger.info("Fitting %d methods on warm-up data (%d cases).",
                len(registry), len(df_warmup))
    for method in registry:
        t0 = time.perf_counter()
        method.fit(df_warmup)
        elapsed = time.perf_counter() - t0
        logger.info("  %-20s fitted in %.1fs", method.name, elapsed)

    # ── Horizon loop ─────────────────────────────────────────────────────
    num_horizons = config.data.num_horizons
    horizon_days = config.data.horizon_days
    current_start = pool_start
    rows: List[Dict[str, Any]] = []

    logger.info("Running %d horizons × %d methods.", num_horizons, len(registry))
    print()
    _print_header(registry)

    for h in range(num_horizons):
        instance = build_weekly_instance(
            df_pool, current_start, h, config, candidate_pools)

        if instance.num_cases == 0:
            logger.info("Week %d: no cases — stopping early.", h)
            break

        week_results: Dict[str, str] = {}

        for method in registry:
            try:
                t0 = time.perf_counter()
                schedule = method.plan(instance)
                solve_time = time.perf_counter() - t0

                kpi = evaluate(instance, schedule, config.costs)

                rows.append(_kpi_to_row(
                    method.name, h, kpi,
                    schedule.objective_value, solve_time,
                ))
                week_results[method.name] = kpi.summary_line()

            except Exception:
                logger.exception("Method %s failed on week %d.", method.name, h)
                rows.append(_empty_kpi_row(method.name, h))
                week_results[method.name] = "FAILED"

        _print_week(h, instance, week_results, registry)
        current_start += timedelta(days=horizon_days)

    # ── Assemble results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        logger.warning("No results produced.")
        return results_df

    print()
    _print_summary(results_df, registry)

    # ── Save outputs ─────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        csv_path = out / "horizon_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info("Per-horizon results saved to %s", csv_path)

        summary = _aggregate(results_df, registry)
        json_path = out / "aggregate_summary.json"
        json_path.write_text(json.dumps(summary, indent=2))
        logger.info("Aggregate summary saved to %s", json_path)

    return results_df


# ─── Aggregation ─────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, registry: MethodRegistry) -> Dict[str, Any]:
    """Compute mean and median KPIs per method."""
    agg: Dict[str, Any] = {}
    for method in registry:
        sub = df[df["method"] == method.name]
        if sub.empty:
            continue
        entry: Dict[str, float | None] = {}
        for col in ["total_cost", "activation_cost", "overtime_min",
                     "idle_min", "scheduled", "deferred",
                     "blocks_opened", "solve_time"]:
            vals = sub[col].dropna()
            entry[f"mean_{col}"] = float(vals.mean()) if len(vals) > 0 else None
            entry[f"median_{col}"] = float(vals.median()) if len(vals) > 0 else None
        agg[method.name] = entry
    return agg


# ─── Console output ──────────────────────────────────────────────────────────

def _print_header(registry: MethodRegistry) -> None:
    names = registry.names
    header = f"{'Wk':>4s}  {'N':>4s}  {'Cand':>4s}"
    for name in names:
        header += f"  │ {name:>10s}: {'cost':>8s} {'blks':>4s} {'OT':>5s} {'sch':>3s}"
    print(header)
    print("─" * len(header))


def _print_week(
    h: int,
    instance,
    results: Dict[str, str],
    registry: MethodRegistry,
) -> None:
    line = (f"{h:4d}  {instance.num_cases:4d}  "
            f"{instance.calendar.total_candidates:4d}")
    for method in registry:
        txt = results.get(method.name, "N/A")
        line += f"  │ {txt}"
    print(line)


def _print_summary(df: pd.DataFrame, registry: MethodRegistry) -> None:
    print()
    print("=" * 80)
    print("  AGGREGATE RESULTS (mean ± std over horizons)")
    print("=" * 80)
    fmt = "  {:<12s}  {:>12s}  {:>10s}  {:>10s}  {:>10s}  {:>5s}  {:>5s}  {:>5s}"
    print(fmt.format("Method", "Total Cost", "Activation", "OT (min)",
                      "Idle (min)", "Sched", "Def", "Blks"))
    print("  " + "─" * 76)
    for method in registry:
        sub = df[df["method"] == method.name]
        if sub.empty:
            print(f"  {method.name:<12s}  {'no data':>12s}")
            continue

        def _fmt(col: str) -> str:
            vals = sub[col].dropna()
            if vals.empty:
                return "N/A"
            return f"{vals.mean():,.0f}±{vals.std():,.0f}"

        print(fmt.format(
            method.name,
            _fmt("total_cost"),
            _fmt("activation_cost"),
            _fmt("overtime_min"),
            _fmt("idle_min"),
            _fmt("scheduled"),
            _fmt("deferred"),
            _fmt("blocks_opened"),
        ))
    print("=" * 80)
