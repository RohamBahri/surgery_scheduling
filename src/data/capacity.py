from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import BlockCalendar, CandidateBlock, Col

logger = logging.getLogger(__name__)
_FRIDAY = 4


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


def classify_fixed_flex(
    df_train_scoped: pd.DataFrame,
    pools: Dict[int, List[Tuple[str, str]]],
    threshold: float = 0.75,
) -> Set[Tuple[int, str, str]]:
    dt = pd.to_datetime(df_train_scoped[Col.ACTUAL_START], errors="coerce")
    df = df_train_scoped.assign(
        _weekday=dt.dt.weekday,
        _site=df_train_scoped[Col.SITE].fillna("").astype(str).str.upper().str.strip(),
        _room=df_train_scoped[Col.OPERATING_ROOM],
        _week_start=(dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize(),
    )
    complete_weeks = _aligned_complete_week_starts(df)
    df = df[df["_week_start"].isin(complete_weeks)]

    fixed: Set[Tuple[int, str, str]] = set()
    for wd, site_rooms in pools.items():
        denom = len(complete_weeks)
        if denom == 0:
            continue
        wd_data = df[df["_weekday"] == wd]
        wk_counts = wd_data.groupby(["_site", "_room"])["_week_start"].nunique().to_dict()
        for site, room in site_rooms:
            rate = wk_counts.get((site, room), 0) / denom
            if rate >= threshold:
                fixed.add((wd, site, room))
    return fixed


def build_block_calendar(
    pools: Dict[int, List[Tuple[str, str]]],
    horizon_start: pd.Timestamp,
    config: Config,
    fixed_templates: Set[Tuple[int, str, str]] | None = None,
) -> BlockCalendar:
    fixed_templates = fixed_templates or set()
    horizon_days = config.data.horizon_days
    start = horizon_start.normalize()
    candidates: list[CandidateBlock] = []

    for d in range(horizon_days):
        cal_date = start + pd.Timedelta(days=d)
        weekday = cal_date.weekday()
        cap = config.capacity.friday_capacity_minutes if weekday == _FRIDAY else config.capacity.block_capacity_minutes
        for site, room in pools.get(weekday, []):
            candidates.append(CandidateBlock(
                day_index=d,
                site=site,
                room=room,
                capacity_minutes=cap,
                activation_cost=config.capacity.activation_cost_per_block,
                is_fixed=(weekday, site, room) in fixed_templates,
            ))

    return BlockCalendar(candidates=candidates)
