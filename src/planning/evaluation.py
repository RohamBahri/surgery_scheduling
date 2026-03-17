"""Schedule evaluation against realized durations.

This is the single authoritative cost evaluator shared by every method.
It takes a :class:`ScheduleResult`, the :class:`WeeklyInstance` that was
planned, and the cost configuration, then computes realised operational
KPIs using actual surgery durations.

Cost structure (paper Section 4.2):
    activation:  Σ_{ℓ opened} F_ℓ
    deferral:    C^d × (number of deferred cases)
    overtime:    C^o × Σ_{ℓ opened} [load_ℓ − C_ℓ]⁺
    idle:        C^u × Σ_{ℓ opened} [C_ℓ − load_ℓ]⁺

Closed blocks contribute zero cost.  Overtime is NOT capped here —
this evaluates what actually happened.
"""

from __future__ import annotations

import logging
from typing import Dict

from src.core.config import CostConfig
from src.core.types import (
    BlockId,
    CaseRecord,
    KPIResult,
    ScheduleResult,
    WeeklyInstance,
)

logger = logging.getLogger(__name__)


def evaluate(
    instance: WeeklyInstance,
    schedule: ScheduleResult,
    costs: CostConfig,
) -> KPIResult:
    """Evaluate a schedule against realized durations.

    Parameters
    ----------
    instance : WeeklyInstance
        The planning instance (provides cases and block calendar).
    schedule : ScheduleResult
        The scheduling decisions to evaluate, including which blocks
        were opened.
    costs : CostConfig
        Cost coefficients.

    Returns
    -------
    KPIResult
    """
    calendar = instance.calendar

    # Build lookup from case_id → CaseRecord
    case_map: Dict[int, CaseRecord] = {c.case_id: c for c in instance.cases}

    # Accumulate actual load per opened block
    block_load: Dict[BlockId, float] = {
        bid: 0.0 for bid in schedule.opened_blocks
    }

    scheduled_count = 0
    deferred_count = 0
    total_deferral_cost = 0.0

    for assignment in schedule.assignments:
        case = case_map.get(assignment.case_id)
        if case is None:
            logger.warning("Case %d not found in instance — skipping.",
                           assignment.case_id)
            continue

        if assignment.is_deferred:
            total_deferral_cost += costs.deferral_per_case
            deferred_count += 1
        else:
            bid = assignment.block_id
            if bid not in block_load:
                # Assigned to a block that wasn't opened — treat as deferred
                logger.warning(
                    "Case %d assigned to block %s which is not opened — "
                    "treating as deferred.",
                    assignment.case_id, bid,
                )
                total_deferral_cost += costs.deferral_per_case
                deferred_count += 1
                continue
            block_load[bid] += case.actual_duration_min
            scheduled_count += 1

    # Compute activation cost for opened blocks
    total_activation_cost = 0.0
    for bid in schedule.opened_blocks:
        total_activation_cost += calendar.activation_cost(bid)

    # Compute overtime and idle per opened block
    total_overtime = 0.0
    total_idle = 0.0
    for bid, load in block_load.items():
        cap = calendar.capacity(bid)
        if load > cap:
            total_overtime += load - cap
        else:
            total_idle += cap - load

    overtime_cost = costs.overtime_per_minute * total_overtime
    idle_cost = costs.idle_per_minute * total_idle
    total_cost = (total_activation_cost + total_deferral_cost
                  + overtime_cost + idle_cost)

    return KPIResult(
        total_cost=total_cost,
        activation_cost=total_activation_cost,
        overtime_cost=overtime_cost,
        idle_cost=idle_cost,
        deferral_cost=total_deferral_cost,
        overtime_minutes=total_overtime,
        idle_minutes=total_idle,
        scheduled_count=scheduled_count,
        deferred_count=deferred_count,
        blocks_opened=len(schedule.opened_blocks),
    )
