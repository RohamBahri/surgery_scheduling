from __future__ import annotations

import logging
from typing import Dict, List, Set

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.core.config import CostConfig, SolverConfig, SurgeonGroupingConfig
from src.core.types import (
    BlockCalendar,
    BlockId,
    CaseRecord,
    EligibilityMap,
    ScheduleAssignment,
    ScheduleResult,
    SurgeonDaySiteCases,
)

logger = logging.getLogger(__name__)


def solve_deterministic(
    cases: List[CaseRecord],
    durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    case_eligible_blocks: Dict[int, List[BlockId]],
    surgeon_day_site_cases: SurgeonDaySiteCases,
    eligibility: EligibilityMap,
    turnover: float,
    surgeon_grouping: SurgeonGroupingConfig,
    model_name: str = "Deterministic",
) -> ScheduleResult:
    N = len(cases)
    assert len(durations) == N

    blocks_by_id = {b.id: b for b in calendar.candidates}
    all_block_ids = list(blocks_by_id.keys())
    fixed_bids = {b.id for b in calendar.fixed_blocks}
    flex_bids = {b.id for b in calendar.flex_blocks}
    cap = {b.id: b.capacity_minutes for b in calendar.candidates}
    f_cost = {b.id: b.activation_cost for b in calendar.candidates}

    if N == 0 or not all_block_ids:
        return ScheduleResult(assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases], solver_status="Empty")

    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    x = {}
    r = model.addVars(N, vtype=GRB.BINARY, name="r")
    v = model.addVars(list(flex_bids), vtype=GRB.BINARY, name="v")
    ot = model.addVars(all_block_ids, lb=0.0, name="ot")
    idle = model.addVars(all_block_ids, lb=0.0, name="idle")

    def v_expr(bid: BlockId):
        return 1 if bid in fixed_bids else v[bid]

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
            model.addConstr(x[i, bid] <= v_expr(bid), name=f"link_{i}_{bid.day_index}_{bid.site}_{bid.room}")

    block_cases: Dict[BlockId, List[int]] = {bid: [] for bid in all_block_ids}
    for (i, bid) in x:
        block_cases[bid].append(i)

    y_used = {}
    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        M_bid = len(eligible_i)
        if M_bid == 0:
            if bid in flex_bids:
                model.addConstr(v[bid] == 0)
            continue
        y_used[bid] = model.addVar(vtype=GRB.BINARY, name=f"y[{bid.day_index},{bid.site},{bid.room}]")
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        model.addConstr(n_bid <= M_bid * y_used[bid])
        model.addConstr(n_bid >= y_used[bid])
        model.addConstr(y_used[bid] <= v_expr(bid))

    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        if not eligible_i:
            continue
        case_load = quicksum(durations[i] * x[i, bid] for i in eligible_i)
        n_bid = quicksum(x[i, bid] for i in eligible_i)
        turnover_load = turnover * (n_bid - y_used[bid])
        load = case_load + turnover_load
        cap_val = cap[bid] * v_expr(bid)
        model.addConstr(load - cap_val <= ot[bid])
        model.addConstr(cap_val - load <= idle[bid])

    u = {}
    adaptive_k2 = 0
    for (surg, day, site_key), case_idxs in surgeon_day_site_cases.items():
        services_today = {cases[i].service for i in case_idxs}
        eligible_site_rooms = set()
        for svc in services_today:
            eligible_site_rooms |= eligibility.get(svc, set())
        for bid in all_block_ids:
            block = blocks_by_id[bid]
            if bid.day_index == day and block.site == site_key and (block.site, block.room) in eligible_site_rooms:
                u[surg, day, site_key, bid] = model.addVar(vtype=GRB.BINARY, name=f"u[{surg},{day},{site_key},{bid.room}]")

    for i, case in enumerate(cases):
        for bid in case_eligible_blocks.get(i, []):
            key = (case.surgeon_code, bid.day_index, case.site, bid)
            if (i, bid) in x and key in u:
                model.addConstr(x[i, bid] <= u[key])

    for (surg, day, site_key), case_idxs in surgeon_day_site_cases.items():
        predicted_load = sum(durations[i] for i in case_idxs) + turnover * max(len(case_idxs) - 1, 0)
        max_cap = max((cap[bid] for bid in all_block_ids if (surg, day, site_key, bid) in u), default=0)
        K_sd = 1 if predicted_load <= max_cap else 2
        if K_sd == 2:
            adaptive_k2 += 1
        if not surgeon_grouping.adaptive_relaxation:
            K_sd = surgeon_grouping.default_max_blocks_per_day
        day_u = [u[surg, day, site_key, bid] for bid in all_block_ids if (surg, day, site_key, bid) in u]
        if day_u:
            model.addConstr(quicksum(day_u) <= K_sd)

    for (surg, day, site_key, bid), uvar in u.items():
        relevant = [i for i in surgeon_day_site_cases.get((surg, day, site_key), []) if (i, bid) in x]
        if relevant:
            model.addConstr(uvar <= quicksum(x[i, bid] for i in relevant))
        else:
            uvar.ub = 0

    activation = quicksum(f_cost[bid] * v[bid] for bid in flex_bids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(N))
    overtime = costs.overtime_per_minute * quicksum(ot[bid] for bid in all_block_ids)
    idletime = costs.idle_per_minute * quicksum(idle[bid] for bid in all_block_ids)
    model.setObjective(activation + deferral + overtime + idletime, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount == 0:
        return ScheduleResult(assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases], solver_status=str(model.Status), solve_time_seconds=getattr(model, "Runtime", 0.0))

    opened: Set[BlockId] = set(fixed_bids)
    for bid in flex_bids:
        if v[bid].X > 0.5:
            opened.add(bid)

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
        "%s model size: x=%d u=%d v_flex=%d y=%d forced_defer=%d k2=%d",
        model_name, len(x), len(u), len(flex_bids), len(y_used), forced_defer, adaptive_k2,
    )

    return ScheduleResult(
        assignments=assignments,
        opened_blocks=opened,
        solver_status=str(model.Status),
        objective_value=model.ObjVal,
        solve_time_seconds=model.Runtime,
    )


def _apply_solver_params(model: gp.Model, cfg: SolverConfig) -> None:
    model.Params.TimeLimit = cfg.time_limit_seconds
    model.Params.MIPGap = cfg.mip_gap
    model.Params.Threads = cfg.threads
    model.Params.OutputFlag = 1 if cfg.verbose else 0
    model.Params.MIPFocus = 1
