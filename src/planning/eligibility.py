"""Day-flexible raw-service compatibility with an explicit same-site fallback."""

from dataclasses import asdict, dataclass

import pandas as pd

from src.core.types import BlockCalendar, BlockId, Col
from src.planning.roster import training_period, week_starts


class CapacityModelError(ValueError):
    """The supplied roster cannot serve a case at its site."""


@dataclass(frozen=True)
class EligibilityDecision:
    eligible_blocks: tuple[BlockId, ...]
    eligibility_tier: int
    number_primary_blocks: int
    number_final_blocks: int
    fallback_used: bool

    def diagnostics(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "eligible_blocks"}


@dataclass(frozen=True)
class ServiceRoomHistory:
    weeks_by_pair: dict[tuple[str, str, str], int]
    provenance: dict


def fit_service_room_history(df_train: pd.DataFrame) -> ServiceRoomHistory:
    if Col.CASE_SERVICE_RAW not in df_train:
        raise ValueError(
            "Raw service labels are required; reload the workbook with the current loader"
        )
    counts = (
        df_train.assign(_week=week_starts(df_train))
        .groupby([Col.CASE_SERVICE_RAW, Col.SITE, Col.OPERATING_ROOM])
        ._week.nunique()
    )
    return ServiceRoomHistory(
        {k: int(v) for k, v in counts.items()}, training_period(df_train)
    )


def resolve_eligibility(
    service_raw: str,
    case_site: str,
    calendar: BlockCalendar,
    history: ServiceRoomHistory,
    min_weeks: int = 3,
) -> EligibilityDecision:
    if min_weeks < 1:
        raise ValueError("min_weeks must be positive")
    at_site = tuple(sorted(b.id for b in calendar.candidates if b.site == case_site))
    if not at_site:
        raise CapacityModelError(
            f"No allocated block at case site {case_site!r}; cross-site assignment is forbidden"
        )
    primary = tuple(
        bid
        for bid in at_site
        if history.weeks_by_pair.get((service_raw, case_site, bid.room), 0) >= min_weeks
    )
    final = primary or at_site
    return EligibilityDecision(
        final, 1 if primary else 2, len(primary), len(final), not bool(primary)
    )
