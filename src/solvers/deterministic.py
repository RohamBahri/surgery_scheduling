from __future__ import annotations

import logging
from typing import Dict, List, Set

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockCalendar, BlockId, CaseRecord, EligibilityMap, ScheduleAssignment, ScheduleResult

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


def solve_deterministic(
    cases: List[CaseRecord],
    durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    eligibility: EligibilityMap,
    turnover: float,
    model_name: str = "Deterministic",
) -> ScheduleResult:
    N = len(cases)
    assert len(durations) == N

    blocks_by_id = {b.id: b for b in calendar.candidates}
    all_block_ids = list(blocks_by_id.keys())
    cap = {b.id: b.capacity_minutes for b in calendar.candidates}
    f_cost = {b.id: b.activation_cost for b in calendar.candidates}

    if N == 0 or not all_block_ids:
        return ScheduleResult(assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases], solver_status="Empty")

    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    x = {}
    r = model.addVars(N, vtype=GRB.BINARY, name="r")
    v = model.addVars(all_block_ids, vtype=GRB.BINARY, name="v")
    ot = model.addVars(all_block_ids, lb=0.0, name="ot")
    idle = model.addVars(all_block_ids, lb=0.0, name="idle")

    forced_defer = 0
    for i in range(N):
        elig = [bid for bid in case_eligible_blocks.get(i, []) if bid in blocks_by_id]
        if not elig:
            model.addConstr(r[i] == 1, name=f"force_defer_{i}")
            forced_defer += 1
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
        M_bid = len(eligible_i)
        if M_bid == 0:
            model.addConstr(v[bid] == 0)
            continue
        y_used[bid] = model.addVar(vtype=GRB.BINARY, name=f"y[{bid.day_index},{bid.site},{bid.room}]")
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        model.addConstr(n_bid <= M_bid * y_used[bid])
        model.addConstr(n_bid >= y_used[bid])
        model.addConstr(y_used[bid] <= v[bid])

    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        if not eligible_i:
            continue
        case_load = quicksum(durations[i] * x[i, bid] for i in eligible_i)
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        turnover_load = turnover * (n_bid - y_used[bid])
        load = case_load + turnover_load
        cap_val = cap[bid] * v[bid]
        model.addConstr(load - cap_val <= ot[bid])
        model.addConstr(cap_val - load <= idle[bid])

    activation = quicksum(f_cost[bid] * v[bid] for bid in all_block_ids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(N))
    overtime = costs.overtime_per_minute * quicksum(ot[bid] for bid in all_block_ids)
    idletime = costs.idle_per_minute * quicksum(idle[bid] for bid in all_block_ids)
    model.setObjective(activation + deferral + overtime + idletime, GRB.MINIMIZE)

    model.optimize()

    status_code = int(model.Status)
    status_name = _status_name(status_code)
    obj_bound = float(model.ObjBound) if hasattr(model, "ObjBound") else None

    if model.SolCount == 0:
        return ScheduleResult(
            assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases],
            solver_status=str(model.Status),
            solve_time_seconds=getattr(model, "Runtime", 0.0),
            diagnostics={
                "forced_defer_count": forced_defer,
                "turnover_used": turnover,
                "eligibility_services": len(eligibility),
                "status_code": status_code,
                "status_name": status_name,
                "mip_gap": None,
                "obj_bound": obj_bound,
                "opened_blocks_count": 0,
                "candidate_blocks_count": len(all_block_ids),
            },
        )

    opened: Set[BlockId] = {bid for bid in all_block_ids if v[bid].X > 0.5}

    assignments: list[ScheduleAssignment] = []
    for i, case in enumerate(cases):
        assigned_bid = None
        for bid in case_eligible_blocks.get(i, []):
            if (i, bid) in x and x[i, bid].X > 0.5:
                assigned_bid = bid
                break
        if assigned_bid is None:
            assignments.append(ScheduleAssignment(case_id=case.case_id))
        else:
            assignments.append(ScheduleAssignment(case_id=case.case_id, day_index=assigned_bid.day_index, site=assigned_bid.site, room=assigned_bid.room))

    logger.info(
        "%s model size: x=%d v=%d y=%d forced_defer=%d",
        model_name, len(x), len(all_block_ids), len(y_used), forced_defer,
    )

    return ScheduleResult(
        assignments=assignments,
        opened_blocks=opened,
        solver_status=str(model.Status),
        objective_value=model.ObjVal,
        solve_time_seconds=model.Runtime,
        diagnostics={
            "forced_defer_count": forced_defer,
            "turnover_used": turnover,
            "eligibility_services": len(eligibility),
            "status_code": status_code,
            "status_name": status_name,
            "mip_gap": float(model.MIPGap),
            "obj_bound": obj_bound,
            "opened_blocks_count": len(opened),
            "candidate_blocks_count": len(all_block_ids),
        },
    )


def _apply_solver_params(model: gp.Model, cfg: SolverConfig) -> None:
    model.Params.TimeLimit = cfg.time_limit_seconds
    model.Params.MIPGap = cfg.mip_gap
    model.Params.Threads = cfg.threads
    model.Params.OutputFlag = 1 if cfg.verbose else 0
    model.Params.MIPFocus = 1
    # Let Gurobi detect and exploit symmetry internally.
    model.Params.Symmetry = 2
