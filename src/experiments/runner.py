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
from src.data.capacity import build_candidate_pools
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
        "n_forced_defer": None,
        "solver_status": None,
        "mip_gap": None,
        "obj_bound": None,
        "planned_obj": None,
        "solve_time": None,
        "training_objective": None,
        "training_bound": None,
        "training_gap": None,
        "certification_status": None,
        "vfcg_iterations": None,
        "vfcg_total_cuts": None,
        "vfcg_max_violation": None,
        "vfcg_tie_break_flags": None,
    }


def _kpi_to_row(method_name: str, week: int, kpi: KPIResult, planned_obj: float | None, solve_time: float, n_forced_defer: int, schedule_diag: Dict[str, Any], method_diag: Dict[str, Any] | None = None) -> Dict[str, Any]:
    method_diag = method_diag or {}
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
        "n_forced_defer": n_forced_defer,
        "solver_status": schedule_diag.get("status_name"),
        "mip_gap": schedule_diag.get("mip_gap"),
        "obj_bound": schedule_diag.get("obj_bound"),
        "planned_obj": planned_obj,
        "solve_time": solve_time,
        "training_objective": method_diag.get("training_objective"),
        "training_bound": method_diag.get("training_bound"),
        "training_gap": method_diag.get("training_gap"),
        "certification_status": method_diag.get("certification_status"),
        "vfcg_iterations": method_diag.get("vfcg_iterations"),
        "vfcg_total_cuts": method_diag.get("vfcg_total_cuts"),
        "vfcg_max_violation": method_diag.get("vfcg_max_violation"),
        "vfcg_tie_break_flags": method_diag.get("vfcg_tie_break_flags"),
    }


def run_experiment(registry: MethodRegistry, config: Config, artifact_run=None) -> pd.DataFrame:
    df = load_data(config)
    df_warmup, df_pool, pool_start = split_warmup_pool(df, config)

    if config.scope.use_all_sites_for_warmup:
        df_warmup_for_elig = df_warmup
    else:
        df_warmup_for_elig, _ = apply_experiment_scope(df_warmup, config)

    elig_maps = build_eligibility_maps(df_warmup_for_elig, config)

    df_warmup_scoped, warmup_scope_summary = apply_experiment_scope(df_warmup, config)
    candidate_pools = build_candidate_pools(df_warmup_scoped, config)

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
        instance = build_weekly_instance(
            df_pool_scoped,
            current_start,
            h,
            config,
            candidate_pools,
            eligibility_maps=elig_maps,
        )
        if instance.num_cases == 0:
            break

        week_rows: List[Dict[str, Any]] = []
        for method in registry:
            try:
                t0 = time.perf_counter()
                schedule = method.plan(instance)
                solve_time = time.perf_counter() - t0
                kpi = evaluate(instance, schedule, config.costs, turnover=config.capacity.turnover_minutes)
                n_forced = sum(1 for i in range(instance.num_cases) if len(instance.case_eligible_blocks.get(i, [])) == 0)
                method_diag = getattr(method, "training_summary", None)
                row = _kpi_to_row(
                    method.name,
                    h,
                    kpi,
                    schedule.objective_value,
                    solve_time,
                    n_forced,
                    schedule.diagnostics,
                    method_diag=method_diag if isinstance(method_diag, dict) else None,
                )
                rows.append(row)
                week_rows.append(row)
                method_rows[method.name].append(row)
                validate_week(instance, schedule, config, method.name)
            except Exception:
                logger.exception("Method %s failed on week %d.", method.name, h)
                empty = _empty_kpi_row(method.name, h)
                rows.append(empty)
                week_rows.append(empty)

        _print_week_summary(h, instance, week_rows)

        current_start += timedelta(days=config.scope.stride_days)

    results_df = pd.DataFrame(rows)

    if artifact_run is not None:
        artifact_run.ensure_run_dir()
        artifact_run.path("config_snapshot.json").write_text(json.dumps(asdict(config), indent=2, default=str))
        artifact_run.path("scope_summary.json").write_text(json.dumps({"warmup": asdict(warmup_scope_summary), "pool": asdict(scope_summary)}, indent=2))
        artifact_run.path("eligibility_summary.json").write_text(json.dumps({
            "n_services": len(elig_maps.service_rooms),
            "mean_site_room_pairs_per_service": (sum(len(v) for v in elig_maps.service_rooms.values()) / max(len(elig_maps.service_rooms), 1)),
        }, indent=2))
        results_df.to_csv(artifact_run.path("horizon_results.csv"), index=False)

        aggregate = {}
        if not results_df.empty:
            for method, sub in results_df.groupby("method"):
                aggregate[method] = {
                    "weeks": int(len(sub)),
                    "mean_total_cost": float(sub["total_cost"].dropna().mean()) if sub["total_cost"].notna().any() else None,
                    "mean_planned_obj": float(sub["planned_obj"].dropna().mean()) if sub["planned_obj"].notna().any() else None,
                }
        artifact_run.path("aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))
        vfcg_training = {}
        for method in registry:
            if method.name != "VFCG":
                continue
            summary = getattr(method, "training_summary", None)
            if isinstance(summary, dict):
                vfcg_training = summary
        artifact_run.path("vfcg_training_summary.json").write_text(json.dumps(vfcg_training, indent=2))

    _validate_oracle_leq_booked(method_rows)
    _print_final_summary(results_df)
    return results_df


def _validate_oracle_leq_booked(method_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    if "Oracle" not in method_rows or "Booked" not in method_rows:
        return
    for b, o in zip(method_rows["Booked"], method_rows["Oracle"]):
        if b.get("total_cost") is None or o.get("total_cost") is None:
            continue
        if o["total_cost"] > b["total_cost"]:
            logger.warning("Oracle > Booked on week %s (%.2f > %.2f)", b.get("week"), o["total_cost"], b["total_cost"])


def _fmt_num(val: Any, digits: int = 1) -> str:
    if val is None or pd.isna(val):
        return "NA"
    return f"{float(val):.{digits}f}"


def _print_week_summary(week: int, instance, rows_for_week: List[Dict[str, Any]]) -> None:
    print(f"Week {week} | cases={instance.num_cases} | blocks={instance.calendar.total_candidates}")
    for row in rows_for_week:
        print(
            f"  {row['method']:<8} | status={row.get('solver_status') or 'NA':<10} "
            f"| obj={_fmt_num(row.get('planned_obj'), 1)} | cost={_fmt_num(row.get('total_cost'), 1)} "
            f"| act={_fmt_num(row.get('activation_cost'), 1)} | OT={_fmt_num(row.get('overtime_min'), 1)} "
            f"| idle={_fmt_num(row.get('idle_min'), 1)} | def={_fmt_num(row.get('deferred'), 0)} "
            f"| open={_fmt_num(row.get('blocks_opened'), 0)} | forced={_fmt_num(row.get('n_forced_defer'), 0)} "
            f"| gap={_fmt_num(row.get('mip_gap'), 3)} | t={_fmt_num(row.get('solve_time'), 2)}s"
        )

    by_method = {r["method"]: r for r in rows_for_week}
    if "Booked" in by_method and "Oracle" in by_method:
        b = by_method["Booked"]
        o = by_method["Oracle"]
        print(
            "  Δ Oracle-Booked "
            f"| obj={_fmt_num((o.get('planned_obj') or 0) - (b.get('planned_obj') or 0), 1)} "
            f"| cost={_fmt_num((o.get('total_cost') or 0) - (b.get('total_cost') or 0), 1)} "
            f"| OT={_fmt_num((o.get('overtime_min') or 0) - (b.get('overtime_min') or 0), 1)} "
            f"| idle={_fmt_num((o.get('idle_min') or 0) - (b.get('idle_min') or 0), 1)} "
            f"| def={_fmt_num((o.get('deferred') or 0) - (b.get('deferred') or 0), 0)} "
            f"| open={_fmt_num((o.get('blocks_opened') or 0) - (b.get('blocks_opened') or 0), 0)} "
            f"| t={_fmt_num((o.get('solve_time') or 0) - (b.get('solve_time') or 0), 2)}s"
        )


def _print_final_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    print("\nFinal aggregate summary")
    for method, sub in results_df.groupby("method"):
        solved = len(sub)
        optimal_frac = (sub["solver_status"] == "OPTIMAL").mean() if solved else 0.0
        tl_frac = (sub["solver_status"] == "TIME_LIMIT").mean() if solved else 0.0
        print(
            f"  {method:<8} | weeks={solved}"
            f" | mean_cost={_fmt_num(sub['total_cost'].mean(), 1)}"
            f" | mean_obj={_fmt_num(sub['planned_obj'].mean(), 1)}"
            f" | mean_act={_fmt_num(sub['activation_cost'].mean(), 1)}"
            f" | mean_OT={_fmt_num(sub['overtime_min'].mean(), 1)}"
            f" | mean_idle={_fmt_num(sub['idle_min'].mean(), 1)}"
            f" | mean_def={_fmt_num(sub['deferred'].mean(), 2)}"
            f" | mean_open={_fmt_num(sub['blocks_opened'].mean(), 2)}"
            f" | mean_forced={_fmt_num(sub['n_forced_defer'].mean(), 2)}"
            f" | mean_t={_fmt_num(sub['solve_time'].mean(), 2)}s"
            f" | OPT={optimal_frac:.2f}"
            f" | TL={tl_frac:.2f}"
            f" | mean_gap={_fmt_num(sub['mip_gap'].mean(), 4)}"
        )
    if {"Booked", "Oracle"}.issubset(set(results_df["method"].dropna().unique())):
        b = results_df[results_df["method"] == "Booked"].set_index("week")
        o = results_df[results_df["method"] == "Oracle"].set_index("week")
        join = o.join(b, lsuffix="_o", rsuffix="_b", how="inner")
        if not join.empty:
            print(
                "  Δ Oracle-Booked (mean) "
                f"| cost={_fmt_num((join['total_cost_o'] - join['total_cost_b']).mean(), 1)}"
                f" | obj={_fmt_num((join['planned_obj_o'] - join['planned_obj_b']).mean(), 1)}"
                f" | OT={_fmt_num((join['overtime_min_o'] - join['overtime_min_b']).mean(), 1)}"
                f" | idle={_fmt_num((join['idle_min_o'] - join['idle_min_b']).mean(), 1)}"
                f" | def={_fmt_num((join['deferred_o'] - join['deferred_b']).mean(), 2)}"
                f" | open={_fmt_num((join['blocks_opened_o'] - join['blocks_opened_b']).mean(), 2)}"
                f" | t={_fmt_num((join['solve_time_o'] - join['solve_time_b']).mean(), 2)}s"
            )
