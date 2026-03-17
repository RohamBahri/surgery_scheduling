"""Candidate block pool construction.

Blocks are endogenous: the planner decides which ORs to open each day.
The *candidate pool* — the set of blocks that *could* be opened — is
determined from training-period weekday patterns, computed once during
setup and reused for every test horizon.

For each weekday (Monday through Friday), we collect every OR room that
was active on that weekday at least once during the training period.
Each (weekday, room) pair becomes a candidate block with capacity
``C_ℓ`` and activation cost ``F_ℓ`` drawn from the configuration.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from src.core.config import Config
from src.core.types import BlockCalendar, CandidateBlock, Col

logger = logging.getLogger(__name__)


# Weekday index → name (for logging)
_WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def build_candidate_pools(
    df_train: pd.DataFrame, config: Config
) -> Dict[int, List[str]]:
    """Build weekday-level candidate room pools from training data.

    Parameters
    ----------
    df_train : DataFrame
        Warm-up (training) data.  Must contain ``actual_start`` and
        ``operating_room``.
    config : Config
        Not currently used but reserved for future pool-construction
        options (e.g., minimum activity thresholds).

    Returns
    -------
    dict
        Maps weekday index (0 = Monday, …, 4 = Friday) to a sorted list
        of OR room names that were active on that weekday during training.
        Weekends (5, 6) are excluded.
    """
    dt = pd.to_datetime(df_train[Col.ACTUAL_START], errors="coerce")
    df = df_train.assign(_weekday=dt.dt.weekday, _room=df_train[Col.OPERATING_ROOM])
    df = df.dropna(subset=["_weekday"])

    pools: Dict[int, List[str]] = {}
    for wd in range(5):  # Monday–Friday only
        rooms = sorted(
            df.loc[df["_weekday"] == wd, "_room"].dropna().unique().tolist()
        )
        pools[wd] = rooms

    for wd, rooms in pools.items():
        logger.info(
            "Candidate pool %s: %d rooms  %s",
            _WEEKDAY_NAMES.get(wd, str(wd)),
            len(rooms),
            rooms[:6],
        )
    total = sum(len(r) for r in pools.values())
    logger.info("Total candidate blocks per full week: %d", total)
    return pools


def build_block_calendar(
    pools: Dict[int, List[str]],
    horizon_start: pd.Timestamp,
    config: Config,
) -> BlockCalendar:
    """Build a :class:`BlockCalendar` for one planning horizon.

    Maps each day of the horizon to its weekday, then looks up the
    candidate rooms for that weekday from the pre-computed pools.

    Parameters
    ----------
    pools : dict
        Weekday → room list, from :func:`build_candidate_pools`.
    horizon_start : Timestamp
        First day of the horizon.
    config : Config
        Provides ``capacity.block_capacity_minutes`` and
        ``capacity.activation_cost_per_block``.

    Returns
    -------
    BlockCalendar
        Contains one :class:`CandidateBlock` per (day, room) pair.
    """
    horizon_days = config.data.horizon_days
    cap = config.capacity.block_capacity_minutes
    act_cost = config.capacity.activation_cost_per_block
    start = horizon_start.normalize()

    candidates: list[CandidateBlock] = []
    for d in range(horizon_days):
        cal_date = start + pd.Timedelta(days=d)
        weekday = cal_date.weekday()

        rooms = pools.get(weekday, [])
        for blk_idx, room in enumerate(rooms):
            candidates.append(CandidateBlock(
                day_index=d,
                block_index=blk_idx,
                room=room,
                capacity_minutes=cap,
                activation_cost=act_cost,
            ))

    logger.debug(
        "Block calendar for %s: %d candidate blocks across %d days",
        start.date(), len(candidates), horizon_days,
    )
    return BlockCalendar(candidates=candidates)
