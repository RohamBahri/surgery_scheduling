"""Schedule extraction from solved Gurobi models."""

from __future__ import annotations

from typing import List, Optional

import gurobipy as gp

from src.core.types import BlockId, CaseRecord, ScheduleAssignment, ScheduleResult


def extract_schedule_from_model(model: Optional[gp.Model], cases: List[CaseRecord]) -> ScheduleResult:
    if model is None or model.SolCount == 0:
        status = "NoSolution" if model is None else str(model.Status)
        return ScheduleResult(assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases], solver_status=status)

    assigned: dict[int, BlockId] = {}
    for var in model.getVars():
        if abs(var.X) < 0.5 or not var.VarName.startswith("x["):
            continue
        # supports x[i,day,site,room]
        inside = var.VarName[var.VarName.find("[") + 1: var.VarName.rfind("]")]
        parts = [p.strip() for p in inside.split(",")]
        if len(parts) >= 4:
            i = int(parts[0])
            assigned[i] = BlockId(int(parts[1]), parts[2], parts[3])

    out = []
    for i, case in enumerate(cases):
        bid = assigned.get(i)
        if bid is None:
            out.append(ScheduleAssignment(case_id=case.case_id))
        else:
            out.append(ScheduleAssignment(case_id=case.case_id, day_index=bid.day_index, site=bid.site, room=bid.room))

    return ScheduleResult(
        assignments=out,
        solver_status=str(model.Status),
        objective_value=model.ObjVal if model.SolCount > 0 else None,
        solve_time_seconds=getattr(model, "Runtime", 0.0),
    )
