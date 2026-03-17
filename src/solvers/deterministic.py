"""Deterministic single-stage OR assignment solver (paper Section 4).

Given a vector of planning durations, solves a mixed-integer program
that jointly decides which OR blocks to open, assigns each case to an
opened block or defers it, and minimises total activation + overtime +
idle + deferral cost.

Formulation (paper equations (1)–(5))
--------------------------------------
Sets:
    i ∈ {0, …, N-1}          cases
    ℓ ∈ B_t                  candidate blocks  (day, block index)

Variables:
    x[i, ℓ] ∈ {0, 1}        case i assigned to block ℓ
    r[i] ∈ {0, 1}            case i deferred
    v[ℓ] ∈ {0, 1}            block ℓ opened (staffed)
    ot[ℓ] ≥ 0                overtime on block ℓ
    idle[ℓ] ≥ 0              idle time on block ℓ

Constraints:
    Σ_ℓ x[i,ℓ] + r[i] = 1                     ∀ i    (assign or defer)
    x[i,ℓ] ≤ v[ℓ]                              ∀ i,ℓ  (linking)
    Σ_i dur[i]·x[i,ℓ] − C_ℓ·v[ℓ] ≤ ot[ℓ]     ∀ ℓ    (overtime)
    C_ℓ·v[ℓ] − Σ_i dur[i]·x[i,ℓ] ≤ idle[ℓ]   ∀ ℓ    (idle)

Objective:
    min  Σ_ℓ F_ℓ·v[ℓ]
       + C^d · Σ_i r[i]
       + C^o · Σ_ℓ ot[ℓ]
       + C^u · Σ_ℓ idle[ℓ]
"""

from __future__ import annotations

import logging
from typing import List, Set

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.core.config import CostConfig, SolverConfig
from src.core.types import (
    BlockCalendar,
    BlockId,
    CandidateBlock,
    CaseRecord,
    ScheduleAssignment,
    ScheduleResult,
)

logger = logging.getLogger(__name__)


def solve_deterministic(
    cases: List[CaseRecord],
    durations: np.ndarray,
    calendar: BlockCalendar,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    model_name: str = "Deterministic",
) -> ScheduleResult:
    """Solve the deterministic assignment MIP with endogenous block opening.

    Parameters
    ----------
    cases : list[CaseRecord]
        Candidate cases.  Length must match *durations*.
    durations : ndarray, shape (N,)
        Planning durations for block-load calculations.
    calendar : BlockCalendar
        Candidate blocks (each may be opened at an activation cost).
    costs : CostConfig
        Cost coefficients (activation, deferral, overtime, idle).
    solver_cfg : SolverConfig
        Gurobi parameter overrides.
    model_name : str
        Label for the Gurobi model.

    Returns
    -------
    ScheduleResult
        Includes ``opened_blocks`` indicating which candidates were staffed.
    """
    N = len(cases)
    assert len(durations) == N, "duration vector length must match case count"
    blocks = calendar.candidates

    if N == 0 or not blocks:
        return ScheduleResult(
            assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases],
            solver_status="Empty",
        )

    # Block data indexed by BlockId
    block_ids = [b.id for b in blocks]
    cap = {b.id: b.capacity_minutes for b in blocks}
    f_cost = {b.id: b.activation_cost for b in blocks}

    # ── Build model ──────────────────────────────────────────────────────
    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    # Assignment variables: x[i, (d, b)]
    x = model.addVars(
        ((i, bid) for i in range(N) for bid in block_ids),
        vtype=GRB.BINARY, name="x",
    )

    # Deferral variables
    r = model.addVars(N, vtype=GRB.BINARY, name="r")

    # Block-opening variables
    v = model.addVars(block_ids, vtype=GRB.BINARY, name="v")

    # Overtime and idle auxiliaries
    ot = model.addVars(
        block_ids, lb=0.0, ub=costs.max_overtime_minutes, name="ot")
    idle = model.addVars(
        block_ids, lb=0.0, name="idle")

    # ── Constraints ──────────────────────────────────────────────────────

    # (1) Assign or defer
    for i in range(N):
        model.addConstr(
            quicksum(x[i, bid] for bid in block_ids) + r[i] == 1,
            name=f"assign_{i}",
        )

    # (2) Linking: no case assigned to a closed block
    for i in range(N):
        for bid in block_ids:
            model.addConstr(
                x[i, bid] <= v[bid],
                name=f"link_{i}_{bid[0]}_{bid[1]}",
            )

    # Block load and overtime/idle accounting
    for bid in block_ids:
        load = quicksum(durations[i] * x[i, bid] for i in range(N))

        # ot[ℓ] ≥ load − C_ℓ · v[ℓ]
        model.addConstr(
            load - cap[bid] * v[bid] <= ot[bid],
            name=f"ot_{bid[0]}_{bid[1]}",
        )
        # idle[ℓ] ≥ C_ℓ · v[ℓ] − load
        model.addConstr(
            cap[bid] * v[bid] - load <= idle[bid],
            name=f"idle_{bid[0]}_{bid[1]}",
        )

    # ── Objective ────────────────────────────────────────────────────────
    activation = quicksum(f_cost[bid] * v[bid] for bid in block_ids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(N))
    overtime = costs.overtime_per_minute * quicksum(ot[bid] for bid in block_ids)
    idletime = costs.idle_per_minute * quicksum(idle[bid] for bid in block_ids)

    model.setObjective(activation + deferral + overtime + idletime, GRB.MINIMIZE)

    # ── Solve ────────────────────────────────────────────────────────────
    logger.info(
        "Solving %s: %d cases, %d candidate blocks",
        model_name, N, len(blocks),
    )
    model.optimize()

    if model.SolCount == 0:
        logger.warning(
            "%s: no feasible solution found (status %d).",
            model_name, model.Status,
        )
        return ScheduleResult(
            assignments=[ScheduleAssignment(case_id=c.case_id) for c in cases],
            solver_status=str(model.Status),
            solve_time_seconds=getattr(model, "Runtime", 0.0),
        )

    # ── Extract solution ─────────────────────────────────────────────────
    # Opened blocks
    opened: Set[BlockId] = set()
    for bid in block_ids:
        if v[bid].X > 0.5:
            opened.add(bid)

    # Case assignments
    assignments: list[ScheduleAssignment] = []
    for i, case in enumerate(cases):
        assigned = False
        for bid in block_ids:
            if x[i, bid].X > 0.5:
                assignments.append(
                    ScheduleAssignment(
                        case_id=case.case_id,
                        day_index=bid[0],
                        block_index=bid[1],
                    )
                )
                assigned = True
                break
        if not assigned:
            # Deferred
            assignments.append(ScheduleAssignment(case_id=case.case_id))

    logger.info(
        "%s solved: obj=%.1f  status=%d  time=%.1fs  "
        "blocks_opened=%d/%d  deferred=%d/%d",
        model_name, model.ObjVal, model.Status, model.Runtime,
        len(opened), len(blocks),
        sum(1 for a in assignments if a.is_deferred), N,
    )

    return ScheduleResult(
        assignments=assignments,
        opened_blocks=opened,
        solver_status=str(model.Status),
        objective_value=model.ObjVal,
        solve_time_seconds=model.Runtime,
    )


def _apply_solver_params(model: gp.Model, cfg: SolverConfig) -> None:
    """Set Gurobi parameters from the solver config."""
    model.Params.TimeLimit = cfg.time_limit_seconds
    model.Params.MIPGap = cfg.mip_gap
    model.Params.Threads = cfg.threads
    model.Params.OutputFlag = 1 if cfg.verbose else 0
    model.Params.MIPFocus = 1       # bias toward feasibility
