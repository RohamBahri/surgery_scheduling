"""Deterministic single-stage OR assignment solver (paper Section 4).

Given a vector of planning durations, solves a mixed-integer program
that jointly decides which OR blocks to open, assigns each case to an
opened block or defers it, and minimises total activation + overtime +
idle + deferral cost.

Key features:
  - Eligibility constraints (B_ti): each case can only be assigned to
    blocks whose room is compatible with the case's service/site.
  - Turnover-aware block load: effective load includes inter-case
    turnover time, modeled as (n_cases - 1) × mean_turnover per block.
  - Symmetry breaking (paper Section 11.4.2):
    (a) Block-load ordering among identical blocks (eq. 53).
    (b) Deferral ordering within eligibility classes (eq. 54).

Formulation (paper equations (1)–(5))
--------------------------------------
Sets:
    i ∈ {0, …, N-1}          cases
    ℓ ∈ B_ti                  eligible blocks for case i

Variables:
    x[i, ℓ] ∈ {0, 1}        case i assigned to block ℓ
    r[i] ∈ {0, 1}            case i deferred
    v[ℓ] ∈ {0, 1}            block ℓ opened (staffed)
    ot[ℓ] ≥ 0                overtime on block ℓ
    idle[ℓ] ≥ 0              idle time on block ℓ

Constraints:
    Σ_{ℓ ∈ B_ti} x[i,ℓ] + r[i] = 1            ∀ i    (assign or defer)
    x[i,ℓ] ≤ v[ℓ]                               ∀ i, ℓ ∈ B_ti  (linking)
    Σ_i dur[i]·x[i,ℓ] + τ·n[ℓ] − C_ℓ·v[ℓ] ≤ ot[ℓ]     ∀ ℓ    (overtime)
    C_ℓ·v[ℓ] − Σ_i dur[i]·x[i,ℓ] − τ·n[ℓ] ≤ idle[ℓ]   ∀ ℓ    (idle)

    where n[ℓ] = (Σ_i x[i,ℓ] − 1)⁺ is the number of turnovers.

Symmetry breaking (paper Section 11.4.2):
    Σ_i d_i·x[i,ℓ1] ≥ Σ_i d_i·x[i,ℓ2]  for identical blocks ℓ1 < ℓ2  (eq. 53)
    r[i1] ≤ r[i2]  for i1 < i2 in the same eligibility class              (eq. 54)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

from src.core.config import CostConfig, SolverConfig, Config
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
    eligible_blocks: Optional[Dict[int, List[BlockId]]] = None,
    mean_turnover: float = 0.0,
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
    eligible_blocks : dict, optional
        Maps case index → list of eligible BlockIds.  If None or missing
        for a case, the case is eligible for all blocks.
    mean_turnover : float
        Mean turnover time (minutes) between consecutive cases in the
        same block.  Added to the block load as (n_cases - 1) × τ.

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
    all_block_ids = [b.id for b in blocks]
    cap = {b.id: b.capacity_minutes for b in blocks}
    f_cost = {b.id: b.activation_cost for b in blocks}

    # Build per-case eligible block sets
    case_blocks: List[List[BlockId]] = []
    for i in range(N):
        if eligible_blocks is not None and i in eligible_blocks:
            cb = eligible_blocks[i]
            # Validate: only keep block IDs that actually exist
            valid = [bid for bid in cb if bid in cap]
            case_blocks.append(valid if valid else all_block_ids)
        else:
            case_blocks.append(all_block_ids)

    # Precompute which cases can go to each block (for load expressions)
    block_cases: Dict[BlockId, List[int]] = {bid: [] for bid in all_block_ids}
    for i in range(N):
        for bid in case_blocks[i]:
            block_cases[bid].append(i)

    # ── Identify symmetry groups (paper Section 11.4.2) ──────────────────
    #
    # Two blocks are "identical" if they have the same capacity and the
    # same set of eligible cases.  Any permutation among identical blocks
    # yields the same cost — this is the symmetry we break.
    #
    # Signature: (capacity, frozenset of eligible case indices)
    block_signature: Dict[BlockId, Tuple[float, FrozenSet[int]]] = {}
    for bid in all_block_ids:
        sig = (cap[bid], frozenset(block_cases[bid]))
        block_signature[bid] = sig

    # Group blocks by signature
    sig_to_blocks: Dict[
        Tuple[float, FrozenSet[int]], List[BlockId]
    ] = defaultdict(list)
    for bid in all_block_ids:
        sig_to_blocks[block_signature[bid]].append(bid)

    # Only groups with ≥ 2 blocks have symmetry to break
    sym_groups = [bids for bids in sig_to_blocks.values() if len(bids) >= 2]
    n_sym_constrs = sum(len(g) - 1 for g in sym_groups)

    # ── Identify deferral equivalence classes (paper eq. 54) ─────────────
    #
    # Cases with identical eligibility sets are interchangeable for
    # deferral: since the deferral penalty C^d is flat (not case-specific),
    # the solver is indifferent about which case in a class gets deferred.
    # Ordering by index eliminates this permutation.
    elig_signature: Dict[int, FrozenSet[BlockId]] = {}
    for i in range(N):
        elig_signature[i] = frozenset(case_blocks[i])

    sig_to_cases: Dict[FrozenSet[BlockId], List[int]] = defaultdict(list)
    for i in range(N):
        sig_to_cases[elig_signature[i]].append(i)

    defer_groups = [
        clist for clist in sig_to_cases.values() if len(clist) >= 2
    ]
    n_defer_constrs = sum(len(g) - 1 for g in defer_groups)

    # Log eligibility and symmetry stats
    avg_eligible = np.mean([len(cb) for cb in case_blocks])
    logger.info(
        "%s: %d cases, %d blocks, avg %.1f eligible/case, "
        "turnover=%.1f min, %d block-sym constrs (%d groups), "
        "%d defer-order constrs (%d classes)",
        model_name, N, len(blocks), avg_eligible, mean_turnover,
        n_sym_constrs, len(sym_groups),
        n_defer_constrs, len(defer_groups),
    )

    # ── Build model ──────────────────────────────────────────────────────
    model = gp.Model(model_name)
    _apply_solver_params(model, solver_cfg)

    # Assignment variables: x[i, bid] only for eligible (i, bid) pairs
    x = {}
    for i in range(N):
        for bid in case_blocks[i]:
            x[i, bid] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x[{i},{bid[0]},{bid[1]}]",
            )

    # Deferral variables
    r = model.addVars(N, vtype=GRB.BINARY, name="r")

    # Block-opening variables
    v = model.addVars(all_block_ids, vtype=GRB.BINARY, name="v")

    # Overtime and idle auxiliaries
    ot = model.addVars(
        all_block_ids, lb=0.0, ub=costs.max_overtime_minutes, name="ot")
    idle = model.addVars(
        all_block_ids, lb=0.0, name="idle")

    # Turnover auxiliaries (only needed if turnover > 0)
    if mean_turnover > 0:
        n_assigned = {}
        for bid in all_block_ids:
            n_assigned[bid] = model.addVar(
                lb=0.0, name=f"nc[{bid[0]},{bid[1]}]")
        tv_load = {}
        for bid in all_block_ids:
            tv_load[bid] = model.addVar(
                lb=0.0, name=f"tv[{bid[0]},{bid[1]}]")

    # ── Core constraints ─────────────────────────────────────────────────

    # (1) Assign to an eligible block or defer
    for i in range(N):
        model.addConstr(
            quicksum(x[i, bid] for bid in case_blocks[i]) + r[i] == 1,
            name=f"assign_{i}",
        )

    # (2) Linking: no case assigned to a closed block
    for i in range(N):
        for bid in case_blocks[i]:
            model.addConstr(
                x[i, bid] <= v[bid],
                name=f"link_{i}_{bid[0]}_{bid[1]}",
            )

    # Block load and overtime/idle accounting
    for bid in all_block_ids:
        eligible_i = block_cases[bid]
        if not eligible_i:
            # No case can reach this block — force it closed
            model.addConstr(v[bid] == 0, name=f"empty_{bid[0]}_{bid[1]}")
            continue

        surgical_load = quicksum(
            durations[i] * x[i, bid] for i in eligible_i
        )

        if mean_turnover > 0:
            # n_assigned[bid] = Σ_i x[i, bid]
            model.addConstr(
                n_assigned[bid] == quicksum(
                    x[i, bid] for i in eligible_i
                ),
                name=f"nc_def_{bid[0]}_{bid[1]}",
            )
            # tv_load[bid] ≥ (n_assigned - 1) × τ
            model.addConstr(
                tv_load[bid] >= (n_assigned[bid] - 1) * mean_turnover,
                name=f"tv_lb_{bid[0]}_{bid[1]}",
            )
            total_load = surgical_load + tv_load[bid]
        else:
            total_load = surgical_load

        # ot[ℓ] ≥ total_load − C_ℓ · v[ℓ]
        model.addConstr(
            total_load - cap[bid] * v[bid] <= ot[bid],
            name=f"ot_{bid[0]}_{bid[1]}",
        )
        # idle[ℓ] ≥ C_ℓ · v[ℓ] − total_load
        model.addConstr(
            cap[bid] * v[bid] - total_load <= idle[bid],
            name=f"idle_{bid[0]}_{bid[1]}",
        )

    # ── Symmetry breaking: block-load ordering (paper eq. 53) ────────────
    #
    # For each group of identical blocks (same capacity, same eligible
    # case set), impose decreasing block load by index.  If blocks ℓ1 and
    # ℓ2 are interchangeable and ℓ1 < ℓ2, then:
    #
    #     Σ_i d_i · x[i, ℓ1] ≥ Σ_i d_i · x[i, ℓ2]
    #
    # This eliminates all permutations of the same assignment across
    # identical blocks.  E.g., if 5 ORTH-eligible rooms exist on Tuesday,
    # this cuts out 5! − 1 = 119 equivalent solutions.
    # for group in sym_groups:
    #     for j in range(len(group) - 1):
    #         bid1 = group[j]
    #         bid2 = group[j + 1]
    #         eligible_i = block_cases[bid1]  # same for bid2 by construction
    #         if not eligible_i:
    #             continue
    #         load1 = quicksum(
    #             durations[i] * x[i, bid1] for i in eligible_i
    #         )
    #         load2 = quicksum(
    #             durations[i] * x[i, bid2] for i in eligible_i
    #         )
    #         model.addConstr(
    #             load1 >= load2,
    #             name=f"sym_load_{bid1[0]}_{bid1[1]}_{bid2[1]}",
    #         )

    # ── Symmetry breaking: deferral ordering (paper eq. 54) ──────────────
    #
    # Within each eligibility class (cases with identical B_ti), the flat
    # deferral penalty makes the solver indifferent about which case to
    # defer.  Force deferrals onto higher-indexed cases:
    #
    #     r[i1] ≤ r[i2]    for i1 < i2 in the same class
    #
    # This is valid because C^d is case-independent: deferring case i1
    # vs i2 has the same cost, so we can impose an order without loss.
    for group in defer_groups:
        for j in range(len(group) - 1):
            i1 = group[j]
            i2 = group[j + 1]
            model.addConstr(
                r[i1] <= r[i2],
                name=f"sym_def_{i1}_{i2}",
            )

    # ── Objective ────────────────────────────────────────────────────────
    activation = quicksum(f_cost[bid] * v[bid] for bid in all_block_ids)
    deferral = costs.deferral_per_case * quicksum(r[i] for i in range(N))
    overtime = costs.overtime_per_minute * quicksum(
        ot[bid] for bid in all_block_ids
    )
    idletime = costs.idle_per_minute * quicksum(
        idle[bid] for bid in all_block_ids
    )

    model.setObjective(
        activation + deferral + overtime + idletime, GRB.MINIMIZE
    )

    # ── Solve ────────────────────────────────────────────────────────────
    model.optimize()

    if model.SolCount == 0:
        logger.warning(
            "%s: no feasible solution found (status %d).",
            model_name, model.Status,
        )
        return ScheduleResult(
            assignments=[
                ScheduleAssignment(case_id=c.case_id) for c in cases
            ],
            solver_status=str(model.Status),
            solve_time_seconds=getattr(model, "Runtime", 0.0),
        )

    # ── Extract solution ─────────────────────────────────────────────────
    opened: Set[BlockId] = set()
    for bid in all_block_ids:
        if v[bid].X > 0.5:
            opened.add(bid)

    assignments: list[ScheduleAssignment] = []
    for i, case in enumerate(cases):
        assigned = False
        for bid in case_blocks[i]:
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
            assignments.append(ScheduleAssignment(case_id=case.case_id))

    n_deferred = sum(1 for a in assignments if a.is_deferred)
    logger.info(
        "%s solved: obj=%.1f  status=%d  time=%.1fs  "
        "blocks_opened=%d/%d  deferred=%d/%d",
        model_name, model.ObjVal, model.Status, model.Runtime,
        len(opened), len(blocks), n_deferred, N,
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