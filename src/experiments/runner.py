from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.core.config import Config
from src.core.types import KPIResult
from src.data.capacity import build_candidate_pools, classify_fixed_flex
from src.data.eligibility import build_eligibility_maps
from src.data.loader import load_data
from src.data.scope import apply_experiment_scope
from src.data.splits import split_warmup_pool
from src.methods.registry import MethodRegistry
from src.planning.audit import audit_surgeon_feasibility
from src.planning.evaluation import evaluate
from src.planning.instance import build_weekly_instance
from src.validation import validate_week

logger = logging.getLogger(__name__)


def _empty_kpi_row(method_name: str, week: int) -> Dict[str, Any]:
    return {
        "method": method_name,
        "week": week,
        "total_cost": None,
        "activation_cost": None,
        "overtime_cost": None,
        "idle_cost": None,
        "deferral_cost": None,
        "overtime_min": None,
        "idle_min": None,
        "scheduled": None,
        "deferred": None,
        "blocks_opened": None,
        "turnover_minutes": None,
        "n_fixed_blocks": None,
        "n_flex_blocks": None,
        "n_forced_defer": None,
        "n_adaptive_k2": None,
        "planned_obj": None,
        "solve_time": None,
    }


def _kpi_to_row(method_name: str, week: int, kpi: KPIResult, planned_obj: float | None, solve_time: float, n_fixed: int, n_flex: int, n_forced_defer: int, n_adaptive_k2: int) -> Dict[str, Any]:
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
        "turnover_minutes": kpi.turnover_minutes,
        "n_fixed_blocks": n_fixed,
        "n_flex_blocks": n_flex,
        "n_forced_defer": n_forced_defer,
        "n_adaptive_k2": n_adaptive_k2,
        "planned_obj": planned_obj,
        "solve_time": solve_time,
    }


def run_experiment(registry: MethodRegistry, config: Config, output_dir: str | Path | None = None) -> pd.DataFrame:
    df = load_data(config)
    df_warmup, df_pool, pool_start = split_warmup_pool(df, config)

    if config.scope.use_all_sites_for_warmup:
        df_warmup_for_elig = df_warmup
    else:
        df_warmup_for_elig, _ = apply_experiment_scope(df_warmup, config)

    elig_maps = build_eligibility_maps(df_warmup_for_elig, config)
    eligibility = elig_maps.service_rooms

    df_warmup_scoped, warmup_scope_summary = apply_experiment_scope(df_warmup, config)
    candidate_pools = build_candidate_pools(df_warmup_scoped, config)
    fixed_templates = classify_fixed_flex(df_warmup_scoped, candidate_pools, config.capacity.fixed_block_threshold)

    df_pool_scoped, scope_summary = apply_experiment_scope(df_pool, config)

    for method in registry:
        t0 = time.perf_counter()
        method.fit(df_warmup)
        logger.info("  %-20s fitted in %.1fs", method.name, time.perf_counter() - t0)

    num_horizons = config.data.num_horizons
    current_start = pool_start
    rows: List[Dict[str, Any]] = []
    method_rows: Dict[str, List[Dict[str, Any]]] = {m.name: [] for m in registry}

    for h in range(num_horizons):
        instance = build_weekly_instance(df_pool_scoped, current_start, h, config, candidate_pools, eligibility, fixed_templates)
        if instance.num_cases == 0:
            break

        for method in registry:
            try:
                t0 = time.perf_counter()
                schedule = method.plan(instance)
                solve_time = time.perf_counter() - t0
                kpi = evaluate(instance, schedule, config.costs, turnover=config.capacity.turnover_minutes)
                audit = audit_surgeon_feasibility(instance, schedule)
                n_forced = sum(1 for i in range(instance.num_cases) if len(instance.case_eligible_blocks.get(i, [])) == 0)
                n_adaptive_k2 = int(schedule.diagnostics.get("adaptive_k2_count", audit.adaptive_k2_count))
                row = _kpi_to_row(
                    method.name,
                    h,
                    kpi,
                    schedule.objective_value,
                    solve_time,
                    len(instance.calendar.fixed_blocks),
                    len(instance.calendar.flex_blocks),
                    n_forced,
                    n_adaptive_k2,
                )
                rows.append(row)
                method_rows[method.name].append(row)
                validate_week(instance, schedule, config, method.name)
            except Exception:
                logger.exception("Method %s failed on week %d.", method.name, h)
                rows.append(_empty_kpi_row(method.name, h))

        current_start += timedelta(days=config.scope.stride_days)

    results_df = pd.DataFrame(rows)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config_snapshot.json").write_text(json.dumps(asdict(config), indent=2, default=str))
        (out / "scope_summary.json").write_text(json.dumps({"warmup": asdict(warmup_scope_summary), "pool": asdict(scope_summary)}, indent=2))
        (out / "eligibility_summary.json").write_text(json.dumps({
            "n_services": len(eligibility),
            "mean_site_room_pairs_per_service": (sum(len(v) for v in eligibility.values()) / max(len(eligibility), 1)),
        }, indent=2))
        (out / "fixed_flex_summary.json").write_text(json.dumps({
            "fixed_templates": len(fixed_templates),
            "candidate_pool_sizes": {str(k): len(v) for k, v in candidate_pools.items()},
        }, indent=2))
        results_df.to_csv(out / "horizon_results.csv", index=False)

    _validate_oracle_leq_booked(method_rows)
    return results_df


def _validate_oracle_leq_booked(method_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    if "Oracle" not in method_rows or "Booked" not in method_rows:
        return
    for b, o in zip(method_rows["Booked"], method_rows["Oracle"]):
        if b.get("total_cost") is None or o.get("total_cost") is None:
            continue
        if o["total_cost"] > b["total_cost"]:
            logger.warning("Oracle > Booked on week %s (%.2f > %.2f)", b.get("week"), o["total_cost"], b["total_cost"])
