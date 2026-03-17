"""Schedule extraction from solved Gurobi models.

Converts Gurobi decision-variable values into typed
:class:`ScheduleAssignment` objects.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB

from src.core.types import CaseRecord, ScheduleAssignment, ScheduleResult

logger = logging.getLogger(__name__)


def extract_schedule_from_model(
    model: Optional[gp.Model],
    cases: List[CaseRecord],
) -> ScheduleResult:
    """Read assignment / rejection variables from a solved Gurobi model.

    Expects variable names of the form ``x[i,d,b]`` for assignments and
    ``r[i]`` for rejections, where *i* is a 0-based case index matching
    the order of *cases*.

    Parameters
    ----------
    model : gp.Model or None
        The solved model.  If ``None`` or infeasible, every case is treated
        as rejected.
    cases : list[CaseRecord]
        Cases in the same order they were indexed in the model.

    Returns
    -------
    ScheduleResult
    """
    n = len(cases)

    if model is None or model.SolCount == 0:
        status = "NoSolution" if model is None else str(model.Status)
        return ScheduleResult(
            assignments=[
                ScheduleAssignment(case_id=c.case_id) for c in cases
            ],
            solver_status=status,
        )

    # Parse Gurobi variable names to extract assignments
    idx_re = re.compile(r"\d+")
    assigned: dict[int, tuple[int, int]] = {}    # case_idx → (day, block)
    rejected: set[int] = set()

    for var in model.getVars():
        if abs(var.X) < 0.5:
            continue
        nums = [int(s) for s in idx_re.findall(var.VarName)]
        if not nums:
            continue
        if var.VarName.startswith("x") and len(nums) >= 3:
            case_idx, day, block = nums[:3]
            assigned[case_idx] = (day, block)
        elif var.VarName.startswith("r"):
            rejected.add(nums[0])

    assignments: list[ScheduleAssignment] = []
    for i, case in enumerate(cases):
        if i in assigned:
            d, b = assigned[i]
            assignments.append(
                ScheduleAssignment(case_id=case.case_id,
                                   day_index=d, block_index=b)
            )
        else:
            assignments.append(ScheduleAssignment(case_id=case.case_id))

    obj = model.ObjVal if model.SolCount > 0 else None
    runtime = getattr(model, "Runtime", 0.0)

    return ScheduleResult(
        assignments=assignments,
        solver_status=str(model.Status),
        objective_value=obj,
        solve_time_seconds=runtime,
    )
