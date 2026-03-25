from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from src.core.config import CostConfig
from src.core.types import BlockCalendar, BlockId, CaseRecord, ScheduleAssignment, ScheduleResult


@dataclass(frozen=True)
class ScheduleColumn:
    """A complete weekly schedule encoded as sparse binary vectors."""

    z_assign: Dict[Tuple[int, BlockId], float]
    z_defer: FrozenSet[int]
    v_open: FrozenSet[BlockId]
    y_used: FrozenSet[BlockId]
    n_cases: int
    block_capacities: Dict[BlockId, float]
    block_activation_costs: Dict[BlockId, float]

    def __hash__(self) -> int:
        return hash(
            (
                frozenset((k, float(v)) for k, v in self.z_assign.items()),
                self.z_defer,
                self.v_open,
                self.y_used,
                self.n_cases,
                frozenset(self.block_capacities.items()),
                frozenset(self.block_activation_costs.items()),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScheduleColumn):
            return False
        return (
            self.z_assign == other.z_assign
            and self.z_defer == other.z_defer
            and self.v_open == other.v_open
            and self.y_used == other.y_used
            and self.n_cases == other.n_cases
            and self.block_capacities == other.block_capacities
            and self.block_activation_costs == other.block_activation_costs
        )

    def cases_in_block(self, block_id: BlockId) -> List[int]:
        return sorted(i for (i, bid), val in self.z_assign.items() if bid == block_id and val > 0.5)

    def compute_block_load(self, durations: np.ndarray, turnover: float) -> Dict[BlockId, float]:
        loads: Dict[BlockId, float] = {bid: 0.0 for bid in self.v_open}
        for bid in self.v_open:
            case_ids = self.cases_in_block(bid)
            case_load = sum(float(durations[i]) for i in case_ids)
            y_val = 1.0 if bid in self.y_used else 0.0
            turnover_load = turnover * (len(case_ids) - y_val)
            loads[bid] = case_load + turnover_load
        return loads

    def compute_cost(self, durations: np.ndarray, costs: CostConfig, turnover: float) -> float:
        activation = sum(self.block_activation_costs.get(bid, 0.0) for bid in self.v_open)
        deferral = costs.deferral_per_case * len(self.z_defer)
        loads = self.compute_block_load(durations=durations, turnover=turnover)

        overtime = 0.0
        idle = 0.0
        for bid, load in loads.items():
            cap = self.block_capacities[bid]
            overtime += max(load - cap, 0.0)
            idle += max(cap - load, 0.0)

        return activation + deferral + costs.overtime_per_minute * overtime + costs.idle_per_minute * idle

    def to_schedule_result(self, cases: List[CaseRecord]) -> ScheduleResult:
        assignments: List[ScheduleAssignment] = []
        for i, case in enumerate(cases):
            if i in self.z_defer:
                assignments.append(ScheduleAssignment(case_id=case.case_id))
                continue
            assigned_bid = next((bid for (j, bid), val in self.z_assign.items() if j == i and val > 0.5), None)
            if assigned_bid is None:
                assignments.append(ScheduleAssignment(case_id=case.case_id))
            else:
                assignments.append(
                    ScheduleAssignment(
                        case_id=case.case_id,
                        day_index=assigned_bid.day_index,
                        site=assigned_bid.site,
                        room=assigned_bid.room,
                    )
                )

        return ScheduleResult(assignments=assignments, opened_blocks=set(self.v_open), solver_status="OPTIMAL")


def extract_column_from_model(
    model,
    n_cases: int,
    calendar: BlockCalendar,
    x_vars: dict,
    v_vars: dict,
    y_vars: dict,
    r_vars: dict,
) -> ScheduleColumn:
    z_assign = {(i, bid): 1.0 for (i, bid), var in x_vars.items() if var.X > 0.5}
    z_defer = frozenset(i for i in range(n_cases) if r_vars[i].X > 0.5)
    v_open = frozenset(bid for bid, var in v_vars.items() if var.X > 0.5)
    y_used = frozenset(bid for bid, var in y_vars.items() if var.X > 0.5)
    block_capacities = {b.id: b.capacity_minutes for b in calendar.candidates}
    block_activation_costs = {b.id: b.activation_cost for b in calendar.candidates}

    _ = model
    return ScheduleColumn(
        z_assign=z_assign,
        z_defer=z_defer,
        v_open=v_open,
        y_used=y_used,
        n_cases=n_cases,
        block_capacities=block_capacities,
        block_activation_costs=block_activation_costs,
    )
