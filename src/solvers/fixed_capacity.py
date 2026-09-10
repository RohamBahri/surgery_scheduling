"""Compact fixed-capacity, day-flexible assignment with Phi/Psi accounting.

There is no opening, deferral, hard overtime limit, or surgeon-room-count rule.
Occupation u affects turnover only. Every roster block contributes capacity.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.core.types import BlockId, WeeklyInstance
from src.planning.eligibility import CapacityModelError
from src.solvers.result import SolveDiagnostics, diagnostics_from_model


def column_from_assignment(
    instance: WeeklyInstance,
    assignment: Mapping[int, BlockId],
    *,
    enforce_eligibility: bool = True,
) -> ScheduleColumn:
    blocks = set(instance.calendar.block_ids)
    if set(assignment) != set(range(instance.num_cases)):
        raise ValueError(
            "Every case must be assigned exactly once; deferral is forbidden"
        )
    for i, bid in assignment.items():
        if bid not in blocks:
            raise CapacityModelError(f"Assigned block {bid} is absent from the roster")
        if enforce_eligibility and bid not in instance.case_eligible_blocks.get(i, []):
            raise ValueError(f"Case {i} assigned to an ineligible block")
    return ScheduleColumn(
        {(i, bid): 1.0 for i, bid in assignment.items()},
        frozenset(),
        frozenset(blocks),
        frozenset(assignment.values()),
        instance.num_cases,
        {b.id: b.capacity_minutes for b in instance.calendar.candidates},
        {bid: 0.0 for bid in blocks},
    )


def schedule_metrics(
    column: ScheduleColumn, durations, costs: CostConfig, turnover: float
) -> dict:
    """Evaluate a complete fixed schedule, including all unused capacity."""
    d = np.asarray(durations, dtype=float)
    if d.shape != (column.n_cases,) or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("Durations must be one finite, nonnegative value per case")
    if not np.isfinite(turnover) or turnover < 0:
        raise ValueError("Turnover must be finite and nonnegative")
    counts = [0] * column.n_cases
    occupied = set()
    for (i, bid), value in column.z_assign.items():
        if value == 0:
            continue
        if value != 1 or not 0 <= i < column.n_cases or bid not in column.v_open:
            raise ValueError("Invalid fixed assignment")
        counts[i] += 1
        occupied.add(bid)
    if (
        column.z_defer
        or any(n != 1 for n in counts)
        or occupied != set(column.y_used)
        or set(column.block_capacities) != set(column.v_open)
        or any(column.block_activation_costs.values())
    ):
        raise ValueError(
            "Fixed columns must assign all cases and preserve all allocated capacity"
        )
    loads = column.compute_block_load(d, turnover)
    ot = sum(max(load - column.block_capacities[b], 0.0) for b, load in loads.items())
    idle = sum(max(column.block_capacities[b] - load, 0.0) for b, load in loads.items())
    k = costs.idle_per_minute * (
        sum(column.block_capacities.values()) - float(d.sum()) - turnover * len(d)
    )
    psi = (
        costs.overtime_per_minute + costs.idle_per_minute
    ) * ot + costs.idle_per_minute * turnover * len(occupied)
    phi = costs.overtime_per_minute * ot + costs.idle_per_minute * idle
    error = abs(phi - (k + psi))
    if not np.isclose(phi, k + psi, rtol=1e-10, atol=1e-6):
        raise AssertionError(f"Phi = K + Psi failed: error={error}")
    return {
        "phi": phi,
        "psi": psi,
        "K": k,
        "K_plus_psi": k + psi,
        "identity_error": error,
        "overtime_minutes": ot,
        "idle_minutes": idle,
        "turnover_minutes": turnover * (len(d) - len(occupied)),
        "occupied_blocks": len(occupied),
        "cases_per_occupied_block": len(d) / len(occupied) if occupied else 0.0,
    }


def interchangeable_block_groups(instance: WeeklyInstance) -> list[list[BlockId]]:
    """Exchange blocks with identical roles in the fixed assignment model.

    Weekday and room labels do not enter its costs or constraints. Any future
    day-dependent constraints/costs must also be respected by this key. Fixed-day
    eligibility alone is already reflected in the complete eligible-case sets.
    """
    cases = {bid: [] for bid in instance.calendar.block_ids}
    for i in range(instance.num_cases):
        for bid in set(instance.case_eligible_blocks.get(i, [])):
            cases[bid].append(i)
    groups = defaultdict(list)
    for b in instance.calendar.candidates:
        groups[(b.site, b.capacity_minutes, tuple(cases[b.id]))].append(b.id)
    return [sorted(bids) for bids in groups.values() if len(bids) > 1]


@dataclass(frozen=True)
class FixedCapacityResult:
    column: ScheduleColumn | None
    diagnostics: SolveDiagnostics
    objective_mode: str
    K: float
    metrics: dict | None
    symmetry_groups: int

    def _shift(self, value: float | None, target: str) -> float | None:
        if value is None:
            return None
        native_scale = "psi" if self.objective_mode == "psi" else "phi"
        if target == native_scale:
            return value
        return value + self.K if target == "phi" else value - self.K

    @property
    def phi_ub(self):
        return self._shift(self.diagnostics.obj_val, "phi")

    @property
    def phi_lb(self):
        return self._shift(self.diagnostics.obj_bound, "phi")

    @property
    def psi_ub(self):
        return self._shift(self.diagnostics.obj_val, "psi")

    @property
    def psi_lb(self):
        return self._shift(self.diagnostics.obj_bound, "psi")


def solve_fixed_capacity_assignment(
    instance: WeeklyInstance,
    durations,
    costs: CostConfig,
    turnover: float,
    solver_config: SolverConfig,
    objective_mode: str = "psi",
    warm_start=None,
    *,
    symmetry_breaking: bool = True,
    backend: str = "gurobi",
) -> FixedCapacityResult:
    """Return a schedule plus native and shifted bounds without backend objects.

    warm_start accepts a ScheduleColumn or a mapping from case index to BlockId.
    Relative stopping gaps refer to the selected native objective; absolute gaps
    are invariant to the constant shift. OPTIMAL is Gurobi's tolerance-qualified
    status, never an assertion of zero uncertainty. Inspect absolute_gap too.
    """
    if backend != "gurobi":
        raise ValueError(f"Unsupported backend: {backend}")
    if objective_mode not in {"phi", "psi_shifted", "psi"}:
        raise ValueError("objective_mode must be 'phi', 'psi_shifted', or 'psi'")
    d = np.asarray(durations, dtype=float)
    if d.shape != (instance.num_cases,) or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError(
            "Durations must be finite and nonnegative with one value per case"
        )
    if not np.isfinite(turnover) or turnover < 0:
        raise ValueError("Turnover must be finite and nonnegative")
    if any(
        not np.isfinite(c) or c < 0
        for c in (costs.overtime_per_minute, costs.idle_per_minute)
    ):
        raise ValueError("Operating costs must be finite and nonnegative")
    blocks = {b.id: b for b in instance.calendar.candidates}
    if len(blocks) != len(instance.calendar.candidates):
        raise ValueError("Duplicate roster block IDs")
    for b in blocks.values():
        if (
            not b.is_fixed
            or b.activation_cost != 0
            or not np.isfinite(b.capacity_minutes)
            or b.capacity_minutes <= 0
        ):
            raise ValueError(
                "Every roster block must have fixed positive capacity and zero activation cost"
            )
    eligible = {}
    for i, case in enumerate(instance.cases):
        bids = sorted(set(instance.case_eligible_blocks.get(i, [])))
        if not bids:
            raise CapacityModelError(
                f"Case {case.case_id} has no eligible allocated block at site {case.site!r}"
            )
        if any(bid not in blocks or bid.site != case.site for bid in bids):
            raise CapacityModelError(
                f"Case {case.case_id} has a missing or cross-site eligible block"
            )
        eligible[i] = bids
    k = costs.idle_per_minute * (
        sum(b.capacity_minutes for b in blocks.values())
        - float(d.sum())
        - turnover * len(d)
    )
    groups = interchangeable_block_groups(instance) if symmetry_breaking else []

    import gurobipy as gp
    from gurobipy import GRB

    from src.solvers.deterministic import _apply_solver_params

    with gp.Model(f"fixed_capacity_{objective_mode}") as model:
        _apply_solver_params(model, solver_config)
        x = {
            (i, bid): model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")
            for i, bids in eligible.items()
            for j, bid in enumerate(bids)
        }
        u = {
            bid: model.addVar(vtype=GRB.BINARY, name=f"u[{j}]")
            for j, bid in enumerate(blocks)
        }
        block_cases = {bid: [] for bid in blocks}
        for i, bid in x:
            block_cases[bid].append(i)
        for i, bids in eligible.items():
            model.addConstr(
                gp.quicksum(x[i, bid] for bid in bids) == 1, name=f"assign[{i}]"
            )
        ots, idles = [], []
        for j, (bid, b) in enumerate(blocks.items()):
            ids = block_cases[bid]
            count = gp.quicksum(x[i, bid] for i in ids)
            model.addConstr(count <= len(ids) * u[bid], name=f"occupation_upper[{j}]")
            model.addConstr(count >= u[bid], name=f"occupation_lower[{j}]")
            load = gp.quicksum(float(d[i]) * x[i, bid] for i in ids) + turnover * (
                count - u[bid]
            )
            ot = model.addVar(lb=0, name=f"ot[{j}]")
            model.addConstr(ot >= load - b.capacity_minutes, name=f"overtime[{j}]")
            ots.append(ot)
            if objective_mode == "phi":
                idle = model.addVar(lb=0, name=f"idle[{j}]")
                model.addConstr(
                    idle >= b.capacity_minutes - load, name=f"idle_capacity[{j}]"
                )
                idles.append(idle)
        for group in groups:
            for left, right in zip(group, group[1:]):
                model.addConstr(u[left] >= u[right])
                # Value precedence: the minimum case index in each occupied
                # block increases across the group. Any schedule admits this
                # relabeling because capacities and eligibility sets coincide.
                prefix = gp.LinExpr()
                for i in block_cases[left]:
                    model.addConstr(x[i, right] <= prefix)
                    prefix = prefix + x[i, left]
        if objective_mode == "phi":
            objective = costs.overtime_per_minute * gp.quicksum(
                ots
            ) + costs.idle_per_minute * gp.quicksum(idles)
        else:
            objective = (
                costs.overtime_per_minute + costs.idle_per_minute
            ) * gp.quicksum(ots) + costs.idle_per_minute * turnover * gp.quicksum(
                u.values()
            )
            # The reduced models differ ONLY by an objective constant. In
            # particular psi_shifted must not create the direct Phi idle terms.
            if objective_mode == "psi_shifted":
                objective += k
        model.setObjective(objective, GRB.MINIMIZE)
        if warm_start is not None:
            if isinstance(warm_start, ScheduleColumn):
                # Validate complete fixed-column occupation before conversion.
                schedule_metrics(warm_start, d, costs, turnover)
                assignment = {
                    i: bid for (i, bid), v in warm_start.z_assign.items() if v > 0.5
                }
            else:
                assignment = dict(warm_start)
            column_from_assignment(instance, assignment)
            for group in groups:
                ordered = sorted(
                    group,
                    key=lambda bid: min(
                        (i for i, b in assignment.items() if b == bid),
                        default=instance.num_cases,
                    ),
                )
                rename = dict(zip(ordered, group))
                assignment = {i: rename.get(bid, bid) for i, bid in assignment.items()}
            for (i, bid), var in x.items():
                var.Start = float(assignment[i] == bid)
            occupied = set(assignment.values())
            for bid, var in u.items():
                var.Start = float(bid in occupied)
        model.optimize()
        diagnostics = diagnostics_from_model(model)
        column, metrics = None, None
        if diagnostics.sol_count:
            assignment = {i: bid for (i, bid), var in x.items() if var.X > 0.5}
            column = column_from_assignment(instance, assignment)
            if column.y_used != frozenset(bid for bid, var in u.items() if var.X > 0.5):
                raise AssertionError("Occupation variables disagree with assignment")
            metrics = schedule_metrics(column, d, costs, turnover)
        return FixedCapacityResult(
            column, diagnostics, objective_mode, k, metrics, len(groups)
        )
