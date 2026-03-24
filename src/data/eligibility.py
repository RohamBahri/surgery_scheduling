"""Eligibility set construction for case-to-block assignment.

Builds the data structures needed to enforce B_ti ⊆ B_t (paper Section 3.5)
from training-period observations.  Three layers of filtering:

  1. Service → room compatibility (hard):
     A case of service s can only be assigned to block ℓ if room(ℓ)
     historically served service s.

  2. Site compatibility (hard):
     A case can only be assigned to a block at the same site.

  3. Surgeon → template preference (soft):
     A surgeon's cases are preferentially assigned to their top-K
     historical room-day templates.  Falls back to service-level
     eligibility if the surgeon has insufficient history.

These are computed once on training data and reused for every test week.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import CandidateBlock, Col

logger = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

class EligibilityMaps:
    """Pre-computed eligibility data from the training period.

    Attributes
    ----------
    service_rooms : dict[str, set[str]]
        Maps surgical service → set of room names that historically
        served that service.
    room_site : dict[str, str]
        Maps room name → site (e.g., "TWH", "TGH", "PMH").
        Rooms with unknown site are mapped to "UNKNOWN".
    surgeon_templates : dict[str, list[tuple[str, int]]]
        Maps surgeon_code → list of (room, weekday) pairs, ordered
        by descending frequency.  Only populated for surgeons with
        sufficient case history.
    """

    def __init__(self) -> None:
        self.service_rooms: Dict[str, Set[str]] = defaultdict(set)
        self.room_site: Dict[str, str] = {}
        self.surgeon_templates: Dict[str, List[Tuple[str, int]]] = {}

    def eligible_rooms_for_case(
        self,
        service: str,
        surgeon_code: str,
        operating_room: str,
        config: Config,
    ) -> Optional[Set[str]]:
        """Return the set of room names a case is eligible for.

        Returns None if no filtering should be applied (i.e., the case
        is eligible for all candidate blocks).  This lets the solver
        skip constraint generation for unrestricted cases.
        """
        elig_cfg = config.eligibility
        allowed: Optional[Set[str]] = None

        # Layer 1: service → room
        if elig_cfg.use_service_room_filter and service in self.service_rooms:
            allowed = set(self.service_rooms[service])

        # Layer 2: site filter
        if elig_cfg.use_site_filter and operating_room in self.room_site:
            case_site = self.room_site[operating_room]
            site_rooms = {
                r for r, s in self.room_site.items() if s == case_site
            }
            if allowed is not None:
                allowed &= site_rooms
            else:
                allowed = site_rooms

        # Layer 3: surgeon preference (narrows further, with fallback)
        if (
            elig_cfg.use_surgeon_preference
            and surgeon_code in self.surgeon_templates
        ):
            top_k = elig_cfg.surgeon_top_k_templates
            pref_rooms = {
                room
                for room, _ in self.surgeon_templates[surgeon_code][:top_k]
            }
            if allowed is not None:
                narrowed = allowed & pref_rooms
                # Fallback: if intersection is empty, keep service-level
                if narrowed:
                    allowed = narrowed
            else:
                allowed = pref_rooms

        return allowed

    def is_block_eligible(
        self,
        block: CandidateBlock,
        service: str,
        surgeon_code: str,
        operating_room: str,
        config: Config,
    ) -> bool:
        """Check whether a specific block is eligible for a case."""
        allowed = self.eligible_rooms_for_case(
            service, surgeon_code, operating_room, config
        )
        if allowed is None:
            return True
        return block.room in allowed


def build_eligibility_maps(
    df_train: pd.DataFrame,
    config: Config,
) -> EligibilityMaps:
    """Build eligibility mappings from training data.

    Parameters
    ----------
    df_train : DataFrame
        Warm-up (training) data with canonical column names.
    config : Config
        Provides eligibility settings (min cases for surgeon prefs, etc.).

    Returns
    -------
    EligibilityMaps
    """
    maps = EligibilityMaps()

    # ── Service → room mapping ───────────────────────────────────────────
    if Col.CASE_SERVICE in df_train.columns:
        svc_room = (
            df_train.dropna(subset=[Col.CASE_SERVICE, Col.OPERATING_ROOM])
            .groupby(Col.CASE_SERVICE)[Col.OPERATING_ROOM]
            .apply(lambda x: set(x.unique()))
        )
        for svc, rooms in svc_room.items():
            maps.service_rooms[svc] = rooms
        logger.info(
            "Service→room map: %d services, median %d rooms/service",
            len(maps.service_rooms),
            int(pd.Series([len(v) for v in maps.service_rooms.values()]).median()),
        )

    # ── Room → site mapping ──────────────────────────────────────────────
    # The UHN data has a "Site" column on the raw data.  After the loader
    # normalises column names, it may appear as "site".  We try both.
    site_col = None
    for candidate in ["site", "Site"]:
        if candidate in df_train.columns:
            site_col = candidate
            break

    if site_col is not None:
        room_site = (
            df_train.dropna(subset=[Col.OPERATING_ROOM, site_col])
            .groupby(Col.OPERATING_ROOM)[site_col]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "UNKNOWN")
        )
        maps.room_site = room_site.to_dict()
        n_sites = len(set(maps.room_site.values()) - {"UNKNOWN"})
        logger.info("Room→site map: %d rooms across %d sites",
                     len(maps.room_site), n_sites)
    else:
        logger.info("No site column found — site filtering disabled.")

    # ── Surgeon → template preferences ───────────────────────────────────
    elig_cfg = config.eligibility
    if elig_cfg.use_surgeon_preference:
        dt = pd.to_datetime(df_train[Col.ACTUAL_START], errors="coerce")
        df_work = df_train.assign(
            _weekday=dt.dt.weekday,
            _room=df_train[Col.OPERATING_ROOM],
        ).dropna(subset=["_weekday", "_room", Col.SURGEON_CODE])

        # Only build preferences for surgeons with enough history
        surg_counts = df_work[Col.SURGEON_CODE].value_counts()
        eligible_surgeons = surg_counts[
            surg_counts >= elig_cfg.surgeon_min_cases_for_preference
        ].index

        for surg in eligible_surgeons:
            sub = df_work[df_work[Col.SURGEON_CODE] == surg]
            template_counts = (
                sub.groupby(["_room", "_weekday"]).size()
                .sort_values(ascending=False)
            )
            templates = [
                (room, int(wd))
                for (room, wd), _ in template_counts.items()
            ]
            maps.surgeon_templates[surg] = templates

        logger.info(
            "Surgeon templates: %d surgeons with ≥ %d cases",
            len(maps.surgeon_templates),
            elig_cfg.surgeon_min_cases_for_preference,
        )

    return maps