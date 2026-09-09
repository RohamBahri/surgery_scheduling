"""Explicit fixed-capacity proxies fitted exclusively to supplied training history.

The caller must pass the frozen training cohort, never the complete workbook.
All listed room-days are allocated, including those left unused by a schedule.
"""

from collections import Counter
from dataclasses import dataclass, replace
from math import floor

import pandas as pd

from src.core.config import Config
from src.core.types import BlockCalendar, CandidateBlock, Col
from src.data.capacity import (
    _aligned_complete_week_starts,
    build_block_calendar,
    build_candidate_pools,
)

ROSTER_SOURCES = (
    "regular_template",
    "observed_activity_proxy",
    "median_count_template",
)


def week_starts(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df[Col.ACTUAL_START]).dt.normalize()
    return dt - pd.to_timedelta(dt.dt.weekday, unit="D")


def training_period(df: pd.DataFrame) -> dict:
    starts = week_starts(df)
    return {
        "first_week": str(starts.min().date()) if len(df) else None,
        "last_week": str(starts.max().date()) if len(df) else None,
        "n_weeks": int(starts.nunique()),
        "n_cases": len(df),
    }


@dataclass(frozen=True)
class RosterBuild:
    calendar: BlockCalendar
    provenance: dict


def build_fixed_roster(
    df_train: pd.DataFrame,
    horizon_start: pd.Timestamp,
    config: Config,
    source: str = "regular_template",
) -> RosterBuild:
    start = pd.Timestamp(horizon_start).normalize()
    if source not in ROSTER_SOURCES:
        raise ValueError(f"Unknown roster source: {source}")
    if config.data.horizon_days != 7 or start.weekday() != 0:
        raise ValueError(
            "Fixed weekly rosters require a Monday start and seven-day horizon"
        )
    provenance = {
        "source": source,
        "training_period": training_period(df_train),
        "horizon_start": str(start.date()),
        "fixed_capacity": True,
    }
    if source == "regular_template":
        pools = build_candidate_pools(df_train, config)
        legacy = build_block_calendar(pools, start, config)
        blocks = [
            replace(b, is_fixed=True, activation_cost=0.0) for b in legacy.candidates
        ]
        used = sorted(pd.Timestamp(w) for w in _aligned_complete_week_starts(df_train))
        provenance.update(
            min_activation_rate=config.capacity.min_activation_rate,
            week_rule="legacy candidate pool drops first/last observed weeks when more than two",
            estimation_weeks=[str(w.date()) for w in used],
        )
    else:
        dt = pd.to_datetime(df_train[Col.ACTUAL_START])
        work = df_train.assign(_week=week_starts(df_train), _weekday=dt.dt.weekday)
        if source == "observed_activity_proxy":
            if start not in set(work._week):
                raise ValueError(
                    "observed_activity_proxy is available only for a supplied training week"
                )
            active = work[work._week == start]
            templates = sorted(
                set(zip(active._weekday, active[Col.SITE], active[Col.OPERATING_ROOM]))
            )
            provenance["retrospective_only"] = True
            provenance["interpretation"] = (
                "Activity proxy, not the true historical master roster"
            )
        else:
            templates = []
            targets = {}
            weeks = sorted(work._week.unique())
            for site in sorted(work[Col.SITE].unique()):
                for wd in range(5):
                    sub = work[(work[Col.SITE] == site) & (work._weekday == wd)]
                    counts = (
                        sub.groupby("_week")[Col.OPERATING_ROOM]
                        .nunique()
                        .reindex(weeks, fill_value=0)
                    )
                    target = int(floor(float(counts.median()) + 0.5))
                    frequency = (
                        sub.groupby(Col.OPERATING_ROOM)._week.nunique().to_dict()
                    )
                    ranked = sorted(
                        frequency, key=lambda room: (-frequency[room], room)
                    )
                    templates.extend((wd, site, room) for room in ranked[:target])
                    targets[f"{site}/{wd}"] = target
            provenance.update(
                rounding_rule="floor(median + 0.5), half rounded up",
                inactive_week_counts="zero for every training week with no activity at site/weekday",
                room_tie_break="lexicographic room identifier",
                target_counts=targets,
            )
        blocks = [
            CandidateBlock(
                int(wd),
                str(site),
                str(room),
                config.capacity.block_capacity_minutes,
                0.0,
                True,
            )
            for wd, site, room in templates
        ]
    calendar = BlockCalendar(blocks)
    provenance["number_available_blocks"] = len(blocks)
    provenance["blocks_by_weekday"] = {
        str(wd): sum(b.day_index == wd for b in blocks) for wd in range(7)
    }
    provenance["blocks_by_site_weekday"] = dict(
        sorted(Counter(f"{b.site}/{b.day_index}" for b in blocks).items())
    )
    return RosterBuild(calendar, provenance)
