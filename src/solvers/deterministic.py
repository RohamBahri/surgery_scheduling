from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.core.column import ScheduleColumn, extract_column_from_model
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CaseRecord, ScheduleAssignment, ScheduleResult

logger = logging.getLogger(__name__)


def _status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status, str(status))


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
    assert len(durations) == n_cases

    blocks_by_id = {b.id: b for b in calendar.candidates}
    all_block_ids = list(blocks_by_id.keys())
    cap = {b.id: b.capacity_minutes for b in calendar.candidates}
    f_cost = {b.id: b.activation_cost for b in calendar.candidates}

    if n_cases == 0 or not all_block_ids:
        return None, float("inf")

    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    x = {}
    r = model.addVars(n_cases, vtype=GRB.BINARY, name="r")
    v = model.addVars(all_block_ids, vtype=GRB.BINARY, name="v")
    ot = model.addVars(all_block_ids, lb=0.0, name="ot")
    idle = model.addVars(all_block_ids, lb=0.0, name="idle")

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

    y_used = {}
    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        m_bid = len(eligible_i)
        if m_bid == 0:
            model.addConstr(v[bid] == 0)
            continue
        y_used[bid] = model.addVar(vtype=GRB.BINARY, name=f"y[{bid.day_index},{bid.site},{bid.room}]")
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        model.addConstr(n_bid <= m_bid * y_used[bid])
        model.addConstr(n_bid >= y_used[bid])
        model.addConstr(y_used[bid] <= v[bid])

    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        if not eligible_i:
            continue
        case_load = quicksum(float(durations[i]) * x[i, bid] for i in eligible_i)
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        turnover_load = turnover * (n_bid - y_used[bid])
        load = case_load + turnover_load
        cap_val = cap[bid] * v[bid]
        model.addConstr(load - cap_val <= ot[bid])
        model.addConstr(cap_val - load <= idle[bid])

    activation = quicksum(f_cost[bid] * v[bid] for bid in all_block_ids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(n_cases))
    overtime = costs.overtime_per_minute * quicksum(ot[bid] for bid in all_block_ids)
    idletime = costs.idle_per_minute * quicksum(idle[bid] for bid in all_block_ids)
    model.setObjective(activation + deferral + overtime + idletime, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount == 0:
        return None, float("inf")

    col = extract_column_from_model(model, n_cases, calendar, x, v, y_used, r)
    return col, float(model.ObjVal)


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
