"""Training-cohort diagnostics. Observed assignments never enter optimization."""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.core.types import BlockId, Col, WeeklyInstance
from src.solvers.fixed_capacity import column_from_assignment, schedule_metrics


def historical_assignment(instance: WeeklyInstance) -> dict[int, BlockId]:
    return {
        i: BlockId(
            (case.actual_start.date() - instance.start_date).days,
            case.site,
            case.operating_room,
        )
        for i, case in enumerate(instance.cases)
    }


def historical_baseline(instance: WeeklyInstance, costs, turnover: float) -> dict:
    assignment = historical_assignment(instance)
    present = set(instance.calendar.block_ids)
    required = set(assignment.values())
    missing = sorted(required - present)
    n = instance.num_cases
    coverage = sum(b in present for b in assignment.values()) / n if n else 1.0
    output = {
        "historical_assignment_coverage": coverage,
        "historical_room_day_coverage": (
            len(required & present) / len(required) if required else 1.0
        ),
        "missing_historical_blocks": [list(b) for b in missing],
        "historical_assignment_feasible_under_eligibility": all(
            b in instance.case_eligible_blocks.get(i, []) for i, b in assignment.items()
        ),
        "realized_cost": None,
        "overtime_minutes": None,
        "idle_minutes": None,
        "turnover_minutes": None,
        "occupied_blocks": None,
    }
    if not missing:
        # Historical placement is evaluated even if it violates learned service
        # compatibility; its eligibility feasibility is explicitly reported above.
        column = column_from_assignment(instance, assignment, enforce_eligibility=False)
        output.update(
            schedule_metrics(column, instance.actual_durations(), costs, turnover)
        )
        output["realized_cost"] = output["phi"]
    return output


def eligibility_components(instance: WeeklyInstance) -> list[dict]:
    """Connected case/block graph components; never change eligibility to split it."""
    neighbours = defaultdict(set)
    for bid in instance.calendar.block_ids:
        neighbours[("block", bid)]
    for i in range(instance.num_cases):
        node = ("case", i)
        neighbours[node]
        for bid in instance.case_eligible_blocks.get(i, []):
            other = ("block", bid)
            neighbours[node].add(other)
            neighbours[other].add(node)
    seen, output = set(), []
    for node in neighbours:
        if node in seen:
            continue
        pending, component = [node], set()
        while pending:
            cur = pending.pop()
            if cur in seen:
                continue
            seen.add(cur)
            component.add(cur)
            pending.extend(neighbours[cur] - seen)
        case_nodes = [n for n in component if n[0] == "case"]
        output.append(
            {
                "component": len(output),
                "n_cases": len(case_nodes),
                "n_blocks": len(component) - len(case_nodes),
                "n_edges": sum(len(neighbours[n]) for n in case_nodes),
            }
        )
    return output


def turnover_audit(
    df_train: pd.DataFrame, max_plausible_gap: float = 240
) -> tuple[pd.DataFrame, dict]:
    """Consecutive retained cohort cases in the same site/OR/calendar day.

    Filtering may omit intervening noncohort activity. These are calibration
    gaps, not identified turnover observations; the primary tau remains 30.
    """
    work = df_train.assign(
        _date=pd.to_datetime(df_train[Col.ACTUAL_START]).dt.normalize()
    )
    rows = []
    for (site, room, day), frame in work.groupby(
        [Col.SITE, Col.OPERATING_ROOM, "_date"]
    ):
        frame = frame.sort_values([Col.ENTER_ROOM, Col.CASE_UID])
        previous = None
        for _, case in frame.iterrows():
            if previous is not None:
                gap = (
                    case[Col.ENTER_ROOM] - previous[Col.LEAVE_ROOM]
                ).total_seconds() / 60
                plausible = bool(np.isfinite(gap) and 0 <= gap <= max_plausible_gap)
                rows.append(
                    {
                        "site": site,
                        "room": room,
                        "date": str(day.date()),
                        "previous_case_id": int(previous[Col.CASE_UID]),
                        "next_case_id": int(case[Col.CASE_UID]),
                        "gap_minutes": gap,
                        "plausible_nonnegative": plausible,
                    }
                )
            previous = case
    frame = pd.DataFrame(
        rows,
        columns=[
            "site",
            "room",
            "date",
            "previous_case_id",
            "next_case_id",
            "gap_minutes",
            "plausible_nonnegative",
        ],
    )
    gaps = frame.loc[frame.plausible_nonnegative.astype(bool), "gap_minutes"]
    summary = {
        "candidate_pairs": len(frame),
        "sample_size": len(gaps),
        "excluded_pairs": len(frame) - len(gaps),
        "median": float(gaps.median()) if len(gaps) else None,
        "q25": float(gaps.quantile(0.25)) if len(gaps) else None,
        "q75": float(gaps.quantile(0.75)) if len(gaps) else None,
        "plausible_range_minutes": [0, max_plausible_gap],
        "caveat": "Consecutive retained cohort cases; gaps may include omitted noncohort activity",
        "primary_turnover_unchanged": 30,
    }
    return frame, summary
