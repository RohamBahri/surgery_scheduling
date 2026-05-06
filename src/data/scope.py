from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import Col

logger = logging.getLogger(__name__)


@dataclass
class ScopeSummary:
    total_before: int
    excluded_weekend: int
    excluded_site: int
    excluded_missing_site: int
    total_after: int


def apply_experiment_scope(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, ScopeSummary]:
    dt = pd.to_datetime(df[Col.ACTUAL_START], errors="coerce")
    weekdays = set(config.scope.planning_weekdays)
    sites = set(config.scope.planning_sites)

    is_weekday = dt.dt.weekday.isin(weekdays)
    site_vals = df[Col.SITE].fillna("").astype(str).str.upper().str.strip()
    missing_site = site_vals == ""
    in_site = site_vals.isin(sites)

    keep = is_weekday & in_site & ~missing_site
    scoped = df[keep].copy()

    summary = ScopeSummary(
        total_before=len(df),
        excluded_weekend=int((~is_weekday).sum()),
        excluded_site=int((is_weekday & ~in_site & ~missing_site).sum()),
        excluded_missing_site=int((is_weekday & missing_site).sum()),
        total_after=len(scoped),
    )
    logger.info("Scope summary: %s", summary)
    return scoped, summary
