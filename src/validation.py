from __future__ import annotations

import logging

from src.core.config import Config
from src.core.types import WeeklyInstance, ScheduleResult

logger = logging.getLogger(__name__)


def validate_week(instance: WeeklyInstance, schedule: ScheduleResult, config: Config, method_name: str) -> None:
    if any(c.actual_duration_min <= 0 for c in instance.cases):
        logger.warning("%s week %d: non-positive actual duration found", method_name, instance.week_index)

    if len(set(instance.calendar.block_ids)) != len(instance.calendar.block_ids):
        logger.warning("%s week %d: duplicate block IDs", method_name, instance.week_index)

    eligible_map = {i: set(bids) for i, bids in instance.case_eligible_blocks.items()}
    case_index = {c.case_id: i for i, c in enumerate(instance.cases)}
    for a in schedule.assignments:
        if a.is_deferred:
            continue
        i = case_index.get(a.case_id)
        if i is None:
            continue
        if a.block_id not in eligible_map.get(i, set()):
            logger.warning("%s week %d: ineligible assignment case_id=%s bid=%s", method_name, instance.week_index, a.case_id, a.block_id)

    fixed_ids = {b.id for b in instance.calendar.fixed_blocks}
    missing_fixed = fixed_ids - set(schedule.opened_blocks)
    if missing_fixed:
        logger.warning("%s week %d: missing fixed opened blocks count=%d", method_name, instance.week_index, len(missing_fixed))

    if config.capacity.turnover_minutes < 0:
        logger.warning("Negative turnover parameter configured")
