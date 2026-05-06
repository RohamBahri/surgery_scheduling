from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.core.config import Config
from src.core.types import CandidateBlock, Col

logger = logging.getLogger(__name__)


class EligibilityMaps:
    def __init__(self) -> None:
        self.service_rooms: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.room_site: Dict[str, str] = {}
        self.surgeon_templates: Dict[str, List[Tuple[str, int]]] = {}

    def eligible_rooms_for_case(
        self,
        service: str,
        surgeon_code: str,
        operating_room: str,
        config: Config,
        case_site: str = "",
    ) -> Optional[Set[Tuple[str, str]]]:
        elig_cfg = config.eligibility
        allowed: Optional[Set[Tuple[str, str]]] = None

        if elig_cfg.use_service_room_filter and service in self.service_rooms:
            allowed = set(self.service_rooms[service])

        if elig_cfg.use_site_filter:
            site = case_site or self.room_site.get(operating_room, "")
            if site:
                site_pairs = {(s, r) for r, s in self.room_site.items() if s == site}
                if allowed is None:
                    allowed = site_pairs
                else:
                    allowed &= site_pairs

        if elig_cfg.use_surgeon_preference and surgeon_code in self.surgeon_templates:
            top_k = elig_cfg.surgeon_top_k_templates
            pref = {(self.room_site.get(room, ""), room) for room, _ in self.surgeon_templates[surgeon_code][:top_k]}
            if allowed is None:
                allowed = pref
            else:
                narrowed = allowed & pref
                if narrowed:
                    allowed = narrowed

        return allowed

    def is_block_eligible(
        self,
        block: CandidateBlock,
        service: str,
        surgeon_code: str,
        operating_room: str,
        config: Config,
        case_site: str = "",
    ) -> bool:
        allowed = self.eligible_rooms_for_case(service, surgeon_code, operating_room, config, case_site)
        if allowed is None:
            return True
        return (block.site, block.room) in allowed


def build_eligibility_maps(df_train: pd.DataFrame, config: Config) -> EligibilityMaps:
    maps = EligibilityMaps()

    work = df_train.copy()
    dt = pd.to_datetime(work[Col.ACTUAL_START], errors="coerce")
    iso = dt.dt.isocalendar()
    work["_iso_year"] = iso.year.astype(int)
    work["_iso_week"] = iso.week.astype(int)

    # Room->site map first
    room_site = (
        work.dropna(subset=[Col.OPERATING_ROOM, Col.SITE])
        .assign(_site=work[Col.SITE].fillna("").astype(str).str.upper().str.strip())
    )
    room_site = room_site[room_site["_site"] != ""]
    maps.room_site = room_site.groupby(Col.OPERATING_ROOM)["_site"].agg(lambda x: x.mode().iloc[0]).to_dict()

    min_weeks = config.capacity.eligibility_min_weeks
    svc_data = work.dropna(subset=[Col.CASE_SERVICE, Col.OPERATING_ROOM])
    svc_data = svc_data.assign(_site=svc_data[Col.SITE].fillna("").astype(str).str.upper().str.strip())
    svc_data = svc_data[svc_data["_site"] != ""]

    grouped = svc_data.groupby([Col.CASE_SERVICE, "_site", Col.OPERATING_ROOM]).apply(
        lambda g: len(set(zip(g["_iso_year"], g["_iso_week"])))
    )
    if len(grouped) > 0:
        for (svc, site, room), n_weeks in grouped.items():
            if n_weeks >= min_weeks:
                maps.service_rooms[svc].add((site, room))

    elig_cfg = config.eligibility
    if elig_cfg.use_surgeon_preference:
        df_work = work.assign(_weekday=dt.dt.weekday, _room=work[Col.OPERATING_ROOM]).dropna(
            subset=["_weekday", "_room", Col.SURGEON_CODE]
        )
        surg_counts = df_work[Col.SURGEON_CODE].value_counts()
        eligible_surgeons = surg_counts[surg_counts >= elig_cfg.surgeon_min_cases_for_preference].index
        for surg in eligible_surgeons:
            sub = df_work[df_work[Col.SURGEON_CODE] == surg]
            template_counts = sub.groupby(["_room", "_weekday"]).size().sort_values(ascending=False)
            maps.surgeon_templates[surg] = [(room, int(wd)) for (room, wd), _ in template_counts.items()]

    logger.info("Eligibility map built: %d services", len(maps.service_rooms))
    return maps
