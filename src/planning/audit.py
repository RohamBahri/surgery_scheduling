from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from src.core.types import BlockId, WeeklyInstance, ScheduleResult


@dataclass
class SurgeonAuditResult:
    multi_block_surgeon_day_sites: int
    total_surgeon_day_sites: int
    mean_daily_room_load: float
    mean_daily_surgical_load: float
    adaptive_k2_count: int


def audit_surgeon_feasibility(instance: WeeklyInstance, schedule: ScheduleResult) -> SurgeonAuditResult:
    case_by_id = {c.case_id: c for c in instance.cases}
    groups: Dict[Tuple[str, int, str], set[BlockId]] = {}
    room_load: Dict[Tuple[str, int, str], float] = {}
    surg_load: Dict[Tuple[str, int, str], float] = {}

    for a in schedule.assignments:
        if a.is_deferred:
            continue
        c = case_by_id.get(a.case_id)
        if c is None:
            continue
        key = (c.surgeon_code, a.day_index or 0, c.site)
        groups.setdefault(key, set()).add(a.block_id)
        room_load[key] = room_load.get(key, 0.0) + c.actual_duration_min
        surg_load[key] = surg_load.get(key, 0.0) + c.surgical_duration_min

    historical_groups = _historical_surgeon_day_site_cases(instance)
    total = len(historical_groups)
    multi = sum(1 for s in groups.values() if len(s) > 1)
    mean_room = (sum(room_load.values()) / max(len(room_load), 1)) if room_load else 0.0
    mean_surg = (sum(surg_load.values()) / max(len(surg_load), 1)) if surg_load else 0.0

    adaptive_k2 = 0
    for key, idxs in historical_groups.items():
        pred = sum(instance.cases[i].booked_duration_min for i in idxs)
        _, day, site = key
        max_cap = max((b.capacity_minutes for b in instance.calendar.candidates if b.day_index == day and b.site == site), default=0)
        if pred > max_cap:
            adaptive_k2 += 1

    return SurgeonAuditResult(
        multi_block_surgeon_day_sites=multi,
        total_surgeon_day_sites=total,
        mean_daily_room_load=mean_room,
        mean_daily_surgical_load=mean_surg,
        adaptive_k2_count=adaptive_k2,
    )


def _historical_surgeon_day_site_cases(instance: WeeklyInstance) -> Dict[Tuple[str, int, str], list[int]]:
    groups: Dict[Tuple[str, int, str], list[int]] = {}
    for i, c in enumerate(instance.cases):
        day_index = (c.actual_start.date() - instance.start_date).days
        if day_index < 0:
            continue
        key = (c.surgeon_code, day_index, c.site)
        groups.setdefault(key, []).append(i)
    return groups
