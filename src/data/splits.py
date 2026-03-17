"""Temporal data split for rolling-horizon evaluation.

Splits the cleaned DataFrame into a warm-up (training) set and a scheduling
pool (test set), aligned to Monday-based week boundaries.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import Col

logger = logging.getLogger(__name__)


def split_warmup_pool(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split into warm-up and scheduling pool at a Monday boundary.

    Parameters
    ----------
    df : DataFrame
        Cleaned surgery data sorted by ``actual_start``.
    config : Config
        Must provide ``data.warmup_weeks``.

    Returns
    -------
    df_warmup : DataFrame
        Cases whose ``actual_start`` falls before the pool start.
    df_pool : DataFrame
        Cases from the pool start onward.
    pool_start : Timestamp
        The Monday that marks the first day of the pool.
    """
    warmup_weeks = config.data.warmup_weeks

    earliest = pd.to_datetime(df[Col.ACTUAL_START].min()).normalize()
    first_monday = earliest - pd.Timedelta(days=earliest.weekday())

    # Discard anything before the first aligned Monday
    df = df[df[Col.ACTUAL_START] >= first_monday].copy()

    pool_start = first_monday + pd.Timedelta(weeks=warmup_weeks)

    df_warmup = df[df[Col.ACTUAL_START] < pool_start].copy()
    df_pool = df[df[Col.ACTUAL_START] >= pool_start].copy()

    logger.info(
        "Split: warmup %s → %s (%d cases), pool from %s (%d cases)",
        first_monday.date(),
        (pool_start - pd.Timedelta(days=1)).date(),
        len(df_warmup),
        pool_start.date(),
        len(df_pool),
    )
    return df_warmup, df_pool, pool_start
