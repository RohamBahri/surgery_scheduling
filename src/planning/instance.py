from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import CaseRecord, Col, Domain, WeeklyInstance
from src.data.capacity import build_block_calendar
from src.data.eligibility import EligibilityMaps

logger = logging.getLogger(__name__)


def build_weekly_instance(
    df_pool: pd.DataFrame,
    horizon_start: pd.Timestamp,
    week_index: int,
    config: Config,
    candidate_pools: Dict[int, List[Tuple[str, str]]],
    eligibility_maps: EligibilityMaps,
) -> WeeklyInstance:
    horizon_days = config.data.horizon_days
    start = horizon_start.normalize()
    end = start + pd.Timedelta(days=horizon_days - 1)

    actual = pd.to_datetime(df_pool[Col.ACTUAL_START])
    mask = (actual.dt.normalize() >= start) & (actual.dt.normalize() <= end)
    df_week = df_pool[mask].copy()

    calendar = build_block_calendar(candidate_pools, start, config)
    cases = _dataframe_to_cases(df_week)

    case_eligible_blocks: Dict[int, List] = {}
    for i, case in enumerate(cases):
        allowed = eligibility_maps.eligible_rooms_for_case(
            service=case.service,
            surgeon_code=case.surgeon_code,
            operating_room=case.operating_room,
            config=config,
            case_site=case.site,
        )
        matched = [b.id for b in calendar.candidates if allowed is None or (b.site, b.room) in allowed]
        case_eligible_blocks[i] = matched

    logger.info(
        "Week %d (%s-%s): %d cases, %d candidate blocks",
        week_index,
        start.date(),
        end.date(),
        len(cases),
        calendar.total_candidates,
    )

    return WeeklyInstance(
        week_index=week_index,
        start_date=start.date(),
        end_date=end.date(),
        cases=cases,
        calendar=calendar,
        case_eligible_blocks=case_eligible_blocks,
    )


def _dataframe_to_cases(df: pd.DataFrame) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for _, row in df.iterrows():
        ts = pd.to_datetime(row[Col.ACTUAL_START])
        records.append(
            CaseRecord(
                case_id=int(row[Col.CASE_UID]),
                procedure_id=str(row.get(Col.PROCEDURE_ID, Domain.UNKNOWN)),
                surgeon_code=str(row.get(Col.SURGEON_CODE, Domain.UNKNOWN)),
                service=str(row.get(Col.CASE_SERVICE, Domain.UNKNOWN)),
                patient_type=str(row.get(Col.PATIENT_TYPE, Domain.UNKNOWN)),
                operating_room=str(row.get(Col.OPERATING_ROOM, "")),
                booked_duration_min=float(row[Col.BOOKED_MINUTES]),
                actual_duration_min=float(row[Col.PROCEDURE_DURATION]),
                actual_start=ts.to_pydatetime(),
                week_of_year=int(ts.isocalendar().week),
                month=ts.month,
                year=ts.year,
                site=str(row.get(Col.SITE, "")),
                surgical_duration_min=float(row.get(Col.SURGICAL_DURATION, 0.0)),
            )
        )
    return records
