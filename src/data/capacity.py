"""Candidate block pool construction.

Blocks are endogenous: the planner decides which ORs to open each day.
The *candidate pool* — the set of blocks that *could* be opened — is
determined from training-period weekday patterns, filtered by activation
regularity, and given day-specific capacities that account for turnover.

Design changes motivated by the data report:
  - Template filtering: only include room-day pairs active in ≥ X% of
    training weeks (default 25%).  The report shows 153/168 templates
    activate ≥ 50%; filtering removes noise rooms.
  - Day-specific capacity: Fridays get shorter capacity (report shows
    Friday mean span 445 min vs ~490 for Mon–Thu).
  - Turnover-aware effective capacity: C_eff = C_nominal - (mean_cases-1)*τ
    is computed per template but applied uniformly via a capacity reduction.
    The report shows mean turnover of 28.3 min.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import BlockCalendar, CandidateBlock, Col

logger = logging.getLogger(__name__)

# Weekday index → name (for logging)
_WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

# Friday weekday index
_FRIDAY = 4


def build_candidate_pools(
    df_train: pd.DataFrame, config: Config
) -> Dict[int, List[str]]:
    """Build weekday-level candidate room pools from training data,
    filtered by historical activation regularity.

    Parameters
    ----------
    df_train : DataFrame
        Warm-up (training) data.  Must contain ``actual_start`` and
        ``operating_room``.
    config : Config
        Provides ``capacity.min_activation_rate`` for template filtering.

    Returns
    -------
    dict
        Maps weekday index (0 = Monday, …, 4 = Friday) to a sorted list
        of OR room names that were regularly active on that weekday.
        Weekends (5, 6) are excluded.
    """
    dt = pd.to_datetime(df_train[Col.ACTUAL_START], errors="coerce")
    df = df_train.assign(
        _weekday=dt.dt.weekday,
        _room=df_train[Col.OPERATING_ROOM],
        _week_start=(dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize(),
    )
    df = df.dropna(subset=["_weekday", "_room", "_week_start"])

    # Count total active training weeks (for activation rate denominator)
    total_weeks = df["_week_start"].nunique()
    if total_weeks == 0:
        logger.warning("No training weeks found — returning empty pools.")
        return {wd: [] for wd in range(5)}

    min_rate = config.capacity.min_activation_rate
    logger.info(
        "Building candidate pools: %d training weeks, "
        "min activation rate %.0f%%",
        total_weeks, 100 * min_rate,
    )

    pools: Dict[int, List[str]] = {}
    total_kept = 0
    total_dropped = 0

    for wd in range(5):  # Monday–Friday only
        wd_data = df[df["_weekday"] == wd]

        # For each room, count how many distinct weeks it was active
        room_week_counts = (
            wd_data.groupby("_room")["_week_start"]
            .nunique()
            .rename("active_weeks")
        )
        room_activation = room_week_counts / total_weeks

        # Filter to rooms meeting the activation threshold
        regular_rooms = sorted(
            room_activation[room_activation >= min_rate].index.tolist()
        )
        dropped = len(room_activation) - len(regular_rooms)

        pools[wd] = regular_rooms
        total_kept += len(regular_rooms)
        total_dropped += dropped

    for wd, rooms in pools.items():
        logger.info(
            "Candidate pool %s: %d rooms (kept)  %s",
            _WEEKDAY_NAMES.get(wd, str(wd)),
            len(rooms),
            rooms[:6],
        )

    logger.info(
        "Total candidate blocks per full week: %d "
        "(dropped %d low-activity templates)",
        total_kept, total_dropped,
    )
    return pools


def build_block_calendar(
    pools: Dict[int, List[str]],
    horizon_start: pd.Timestamp,
    config: Config,
) -> BlockCalendar:
    """Build a :class:`BlockCalendar` for one planning horizon.

    Maps each day of the horizon to its weekday, then looks up the
    candidate rooms for that weekday from the pre-computed pools.

    Day-specific capacity:
      - Friday blocks use ``config.capacity.friday_capacity_minutes``
      - All other weekdays use ``config.capacity.block_capacity_minutes``

    Parameters
    ----------
    pools : dict
        Weekday → room list, from :func:`build_candidate_pools`.
    horizon_start : Timestamp
        First day of the horizon.
    config : Config
        Provides capacity and activation cost settings.

    Returns
    -------
    BlockCalendar
        Contains one :class:`CandidateBlock` per (day, room) pair.
    """
    horizon_days = config.data.horizon_days
    cap_default = config.capacity.block_capacity_minutes
    cap_friday = config.capacity.friday_capacity_minutes
    act_cost = config.capacity.activation_cost_per_block
    start = horizon_start.normalize()

    candidates: list[CandidateBlock] = []
    for d in range(horizon_days):
        cal_date = start + pd.Timedelta(days=d)
        weekday = cal_date.weekday()

        # Day-specific capacity
        cap = cap_friday if weekday == _FRIDAY else cap_default

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