from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, LinExpr, quicksum

from src.core.column import ScheduleColumn, extract_column_from_model
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CaseRecord, ScheduleAssignment, ScheduleResult

logger = logging.getLogger(__name__)


@dataclass
class WeeklyModelArtifacts:
    model: gp.Model
    x: Dict[Tuple[int, BlockId], gp.Var]
    r: Dict[int, gp.Var]
    v: Dict[BlockId, gp.Var]
    y: Dict[BlockId, gp.Var]
    predicted_total_cost: LinExpr
    realized_total_cost: LinExpr | None
    n_cases: int


def _status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status, str(status))


def _build_weekly_schedule_model(
    cases: List[CaseRecord],
    planning_durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    turnover: float,
    realized_durations: np.ndarray | None = None,
    model_name: str = "weekly",
) -> WeeklyModelArtifacts:
    n_cases = len(cases)
    assert len(planning_durations) == n_cases
    if realized_durations is not None:
        assert len(realized_durations) == n_cases

    blocks_by_id = {b.id: b for b in calendar.candidates}
    all_block_ids = list(blocks_by_id.keys())
    cap = {b.id: b.capacity_minutes for b in calendar.candidates}
    f_cost = {b.id: b.activation_cost for b in calendar.candidates}

    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    x: Dict[Tuple[int, BlockId], gp.Var] = {}
    r = model.addVars(n_cases, vtype=GRB.BINARY, name="r")
    v = model.addVars(all_block_ids, vtype=GRB.BINARY, name="v")

    # Create only eligible assignment variables.
    for i in range(n_cases):
        elig = [bid for bid in case_eligible_blocks.get(i, []) if bid in blocks_by_id]
        if not elig:
            model.addConstr(r[i] == 1, name=f"force_defer_{i}")
            continue
        for bid in elig:
            x[i, bid] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{bid.day_index},{bid.site},{bid.room}]")
        model.addConstr(quicksum(x[i, bid] for bid in elig) + r[i] == 1, name=f"assign_{i}")
        for bid in elig:
            model.addConstr(x[i, bid] <= v[bid], name=f"link_{i}_{bid.day_index}_{bid.site}_{bid.room}")

    block_cases: Dict[BlockId, List[int]] = {bid: [] for bid in all_block_ids}
    for (i, bid) in x:
        block_cases[bid].append(i)

    y: Dict[BlockId, gp.Var] = {}
    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        m_bid = len(eligible_i)
        if m_bid == 0:
            model.addConstr(v[bid] == 0, name=f"close_if_no_eligible_{bid.day_index}_{bid.site}_{bid.room}")
            continue
        y[bid] = model.addVar(vtype=GRB.BINARY, name=f"y[{bid.day_index},{bid.site},{bid.room}]")
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        model.addConstr(n_bid <= m_bid * y[bid], name=f"use_upper_{bid.day_index}_{bid.site}_{bid.room}")
        model.addConstr(n_bid >= y[bid], name=f"use_lower_{bid.day_index}_{bid.site}_{bid.room}")
        model.addConstr(y[bid] <= v[bid], name=f"open_use_link_{bid.day_index}_{bid.site}_{bid.room}")

    pred_ot = model.addVars(all_block_ids, lb=0.0, name="ot_pred")
    pred_idle = model.addVars(all_block_ids, lb=0.0, name="idle_pred")

    realized_ot = None
    realized_idle = None
    if realized_durations is not None:
        realized_ot = model.addVars(all_block_ids, lb=0.0, name="ot_real")
        realized_idle = model.addVars(all_block_ids, lb=0.0, name="idle_real")

    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        if not eligible_i:
            continue

        n_bid = quicksum(x[i, bid] for i in eligible_i)
        pred_case_load = quicksum(float(planning_durations[i]) * x[i, bid] for i in eligible_i)
        pred_load = pred_case_load + turnover * (n_bid - y[bid])
        cap_val = cap[bid] * v[bid]
        model.addConstr(pred_load - cap_val <= pred_ot[bid], name=f"pred_ot_{bid.day_index}_{bid.site}_{bid.room}")
        model.addConstr(cap_val - pred_load <= pred_idle[bid], name=f"pred_idle_{bid.day_index}_{bid.site}_{bid.room}")

        if realized_durations is not None and realized_ot is not None and realized_idle is not None:
            real_case_load = quicksum(float(realized_durations[i]) * x[i, bid] for i in eligible_i)
            real_load = real_case_load + turnover * (n_bid - y[bid])
            model.addConstr(real_load - cap_val <= realized_ot[bid], name=f"real_ot_{bid.day_index}_{bid.site}_{bid.room}")
            model.addConstr(cap_val - real_load <= realized_idle[bid], name=f"real_idle_{bid.day_index}_{bid.site}_{bid.room}")

    # Symmetry breaking: order loads on identical blocks and deferrals within eligibility classes.
    block_eligible_set: Dict[BlockId, frozenset[int]] = {}
    for bid in all_block_ids:
        block_eligible_set[bid] = frozenset(block_cases[bid])

    symmetry_groups: Dict[tuple, list[BlockId]] = defaultdict(list)
    for bid in all_block_ids:
        key = (bid.day_index, bid.site, cap[bid], f_cost[bid], block_eligible_set[bid])
        symmetry_groups[key].append(bid)

    for group_bids in symmetry_groups.values():
        if len(group_bids) < 2:
            continue
        sorted_bids = sorted(group_bids, key=lambda b: b.room)
        for idx in range(len(sorted_bids) - 1):
            bid_hi = sorted_bids[idx]
            bid_lo = sorted_bids[idx + 1]
            eligible_hi = block_cases[bid_hi]
            eligible_lo = block_cases[bid_lo]
            if not eligible_hi or not eligible_lo:
                continue
            load_hi = quicksum(float(planning_durations[i]) * x[i, bid_hi] for i in eligible_hi if (i, bid_hi) in x)
            load_lo = quicksum(float(planning_durations[i]) * x[i, bid_lo] for i in eligible_lo if (i, bid_lo) in x)
            model.addConstr(load_hi >= load_lo, name=f"sym_load_{bid_hi.day_index}_{bid_hi.room}_{bid_lo.room}")

    elig_class_cases: Dict[frozenset[BlockId], list[int]] = defaultdict(list)
    for i in range(n_cases):
        elig = frozenset(case_eligible_blocks.get(i, []))
        if elig:
            elig_class_cases[elig].append(i)

    for case_list in elig_class_cases.values():
        sorted_cases = sorted(case_list)
        for idx in range(len(sorted_cases) - 1):
            i1 = sorted_cases[idx]
            i2 = sorted_cases[idx + 1]
            model.addConstr(r[i1] <= r[i2], name=f"sym_defer_{i1}_{i2}")

    activation = quicksum(f_cost[bid] * v[bid] for bid in all_block_ids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(n_cases))
    predicted_total_cost = (
        activation
        + deferral
        + costs.overtime_per_minute * quicksum(pred_ot[bid] for bid in all_block_ids)
        + costs.idle_per_minute * quicksum(pred_idle[bid] for bid in all_block_ids)
    )

    realized_total_cost: LinExpr | None = None
    if realized_durations is not None and realized_ot is not None and realized_idle is not None:
        realized_total_cost = (
            activation
            + deferral
            + costs.overtime_per_minute * quicksum(realized_ot[bid] for bid in all_block_ids)
            + costs.idle_per_minute * quicksum(realized_idle[bid] for bid in all_block_ids)
        )

    return WeeklyModelArtifacts(
        model=model,
        x=x,
        r=r,
        v=v,
        y=y,
        predicted_total_cost=predicted_total_cost,
        realized_total_cost=realized_total_cost,
        n_cases=n_cases,
    )


def solve_pricing(
    n_cases: int,
    durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    turnover: float,
    model_name: str = "Pricing",
) -> Tuple[Optional[ScheduleColumn], float]:
    if n_cases == 0 or not calendar.block_ids:
        return None, float("inf")

    dummy_cases = [
        CaseRecord(
            case_id=i,
            procedure_id="",
            surgeon_code="",
            service="",
            patient_type="",
            operating_room="",
            booked_duration_min=float(durations[i]),
            actual_duration_min=float(durations[i]),
            actual_start=datetime(1970, 1, 1),
            week_of_year=1,
            month=1,
            year=1970,
        )
        for i in range(n_cases)
    ]

    artifacts = _build_weekly_schedule_model(
        cases=dummy_cases,
        planning_durations=durations,
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=case_eligible_blocks,
        turnover=turnover,
        model_name=model_name,
    )
    artifacts.model.setObjective(artifacts.predicted_total_cost, GRB.MINIMIZE)
    artifacts.model.optimize()

    if artifacts.model.SolCount == 0:
        return None, float("inf")

    col = extract_column_from_model(artifacts.model, n_cases, calendar, artifacts.x, artifacts.v, artifacts.y, artifacts.r)
    return col, float(artifacts.model.ObjVal)


def solve_weekly_optimistic(
    cases: List[CaseRecord],
    planning_durations: np.ndarray,
    realized_durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    turnover: float,
    model_name: str = "weekly_optimistic",
    tol: float = 1e-6,
) -> Tuple[Optional[ScheduleColumn], float, float, str, float]:
    t0 = time.perf_counter()

    # Pass A: minimize predicted cost.
    pass_a = _build_weekly_schedule_model(
        cases=cases,
        planning_durations=planning_durations,
        realized_durations=None,
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=case_eligible_blocks,
        turnover=turnover,
        model_name=f"{model_name}_pass_a",
    )
    pass_a.model.setObjective(pass_a.predicted_total_cost, GRB.MINIMIZE)
    pass_a.model.optimize()

    if pass_a.model.SolCount == 0:
        return None, float("inf"), float("inf"), _status_name(pass_a.model.Status), time.perf_counter() - t0

    p_star = float(pass_a.model.ObjVal)

    # Pass B: among predicted-optimal schedules, minimize realized cost.
    pass_b = _build_weekly_schedule_model(
        cases=cases,
        planning_durations=planning_durations,
        realized_durations=realized_durations,
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=case_eligible_blocks,
        turnover=turnover,
        model_name=f"{model_name}_pass_b",
    )
    pass_b.model.addConstr(pass_b.predicted_total_cost <= p_star + tol, name="predicted_optimality_band")
    assert pass_b.realized_total_cost is not None
    pass_b.model.setObjective(pass_b.realized_total_cost, GRB.MINIMIZE)
    pass_b.model.optimize()

    if pass_b.model.SolCount == 0:
        return None, p_star, float("inf"), _status_name(pass_b.model.Status), time.perf_counter() - t0

    col = extract_column_from_model(pass_b.model, len(cases), calendar, pass_b.x, pass_b.v, pass_b.y, pass_b.r)
    predicted_cost = float(col.compute_cost(np.asarray(planning_durations, dtype=float), costs, turnover))
    realized_cost = float(col.compute_cost(np.asarray(realized_durations, dtype=float), costs, turnover))

    return col, predicted_cost, realized_cost, _status_name(pass_b.model.Status), time.perf_counter() - t0


def solve_deterministic(
    cases: List[CaseRecord],
    durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    turnover: float,
    model_name: str = "Deterministic",
) -> ScheduleResult:
    n_cases = len(cases)
    all_block_ids = list(calendar.block_ids)
    forced_defer = sum(1 for i in range(n_cases) if len(case_eligible_blocks.get(i, [])) == 0)

    if n_cases == 0 or not all_block_ids:
        return ScheduleResult(assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases], solver_status="Empty")

    column, obj = solve_pricing(
        n_cases=n_cases,
        durations=durations,
        calendar=calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=case_eligible_blocks,
        turnover=turnover,
        model_name=model_name,
    )

    if column is None:
        return ScheduleResult(
            assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases],
            solver_status="INFEASIBLE",
            diagnostics={
                "forced_defer_count": forced_defer,
                "turnover_used": turnover,
                "opened_blocks_count": 0,
                "candidate_blocks_count": len(all_block_ids),
            },
        )

    result = column.to_schedule_result(cases)
    result.objective_value = obj
    result.solver_status = "OPTIMAL"
    result.diagnostics = {
        "forced_defer_count": forced_defer,
        "turnover_used": turnover,
        "opened_blocks_count": len(result.opened_blocks),
        "candidate_blocks_count": len(all_block_ids),
    }
    logger.info("%s model solved with opened_blocks=%d", model_name, len(result.opened_blocks))
    return result


def _apply_solver_params(model: gp.Model, cfg: SolverConfig) -> None:
    model.Params.TimeLimit = cfg.time_limit_seconds
    model.Params.MIPGap = cfg.mip_gap
    model.Params.Threads = cfg.threads
    model.Params.OutputFlag = 1 if cfg.verbose else 0
    model.Params.MIPFocus = 1
    model.Params.Symmetry = 2
    model.Params.Presolve = 2
    model.Params.Cuts = 2
    model.Params.Heuristics = 0.1
