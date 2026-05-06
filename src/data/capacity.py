from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import BlockCalendar, CandidateBlock, Col


def _aligned_complete_week_starts(df: pd.DataFrame) -> Set[pd.Timestamp]:
    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
    week_start = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()
    unique = sorted(week_start.dropna().unique())
    if len(unique) <= 2:
        return set(unique)
    return set(unique[1:-1])


def build_candidate_pools(df_train: pd.DataFrame, config: Config) -> Dict[int, List[Tuple[str, str]]]:
    dt = pd.to_datetime(df_train[Col.ACTUAL_START], errors="coerce")
    df = df_train.assign(
        _weekday=dt.dt.weekday,
        _site=df_train[Col.SITE].fillna("").astype(str).str.upper().str.strip(),
        _room=df_train[Col.OPERATING_ROOM],
        _week_start=(dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize(),
    )
    df = df.dropna(subset=["_weekday", "_room", "_week_start"])
    df = df[df["_site"] != ""]

    complete_weeks = _aligned_complete_week_starts(df)
    df = df[df["_week_start"].isin(complete_weeks)]
    total_weeks = df["_week_start"].nunique()
    if total_weeks == 0:
        return {wd: [] for wd in range(5)}

    pools: Dict[int, List[Tuple[str, str]]] = {}
    for wd in range(5):
        wd_data = df[df["_weekday"] == wd]
        pair_counts = wd_data.groupby(["_site", "_room"])["_week_start"].nunique()
        activation = pair_counts / total_weeks
        regular = sorted([pair for pair, rate in activation.items() if rate >= config.capacity.min_activation_rate])
        pools[wd] = regular
    return pools


def build_block_calendar(
    pools: Dict[int, List[Tuple[str, str]]],
    horizon_start: pd.Timestamp,
    config: Config,
    fixed_templates: Set[Tuple[int, str, str]] | None = None,
) -> BlockCalendar:
    horizon_days = config.data.horizon_days
    start = horizon_start.normalize()
    candidates: list[CandidateBlock] = []

    for d in range(horizon_days):
        cal_date = start + pd.Timedelta(days=d)
        weekday = cal_date.weekday()
        for site, room in pools.get(weekday, []):
            candidates.append(CandidateBlock(
                day_index=d,
                site=site,
                room=room,
                capacity_minutes=config.capacity.block_capacity_minutes,
                activation_cost=config.capacity.activation_cost_per_block,
            ))

    return BlockCalendar(candidates=candidates)
