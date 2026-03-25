from __future__ import annotations

import logging
from typing import Dict

from src.core.config import CostConfig
from src.core.types import BlockId, CaseRecord, KPIResult, ScheduleResult, WeeklyInstance

logger = logging.getLogger(__name__)


def evaluate(instance: WeeklyInstance, schedule: ScheduleResult, costs: CostConfig, turnover: float = 0.0) -> KPIResult:
    case_map: Dict[int, CaseRecord] = {c.case_id: c for c in instance.cases}
    block_load: Dict[BlockId, float] = {bid: 0.0 for bid in schedule.opened_blocks}
    block_case_count: Dict[BlockId, int] = {bid: 0 for bid in schedule.opened_blocks}

    scheduled_count = 0
    deferred_count = 0
    total_deferral_cost = 0.0

    for assignment in schedule.assignments:
        case = case_map.get(assignment.case_id)
        if case is None:
            continue
        if assignment.is_deferred:
            total_deferral_cost += costs.deferral_per_case
            deferred_count += 1
            continue
        bid = assignment.block_id
        if bid not in block_load:
            total_deferral_cost += costs.deferral_per_case
            deferred_count += 1
            continue
        block_load[bid] += case.actual_duration_min
        block_case_count[bid] += 1
        scheduled_count += 1

    total_turnover = 0.0
    for bid in block_load:
        k = block_case_count[bid]
        t = turnover * max(k - 1, 0)
        block_load[bid] += t
        total_turnover += t

    total_activation_cost = sum(
        instance.calendar.activation_cost(bid)
        for bid in schedule.opened_blocks
    )

    total_overtime = 0.0
    total_idle = 0.0
    for bid, load in block_load.items():
        cap = instance.calendar.capacity(bid)
        if load > cap:
            total_overtime += load - cap
        else:
            total_idle += cap - load

    overtime_cost = costs.overtime_per_minute * total_overtime
    idle_cost = costs.idle_per_minute * total_idle
    total_cost = total_activation_cost + total_deferral_cost + overtime_cost + idle_cost

    return KPIResult(
        total_cost=total_cost,
        activation_cost=total_activation_cost,
        overtime_cost=overtime_cost,
        idle_cost=idle_cost,
        deferral_cost=total_deferral_cost,
        overtime_minutes=total_overtime,
        idle_minutes=total_idle,
        turnover_minutes=total_turnover,
        scheduled_count=scheduled_count,
        deferred_count=deferred_count,
        blocks_opened=len(schedule.opened_blocks),
    )
