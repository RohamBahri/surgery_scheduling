"""Warm-start reference generation for exact VFCG."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_pricing


def _unique_schedule_columns(columns: Iterable[ScheduleColumn]) -> List[ScheduleColumn]:
    unique: List[ScheduleColumn] = []
    seen: set[ScheduleColumn] = set()
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        unique.append(col)
    return unique


def generate_warmstart_references(
    week_data_list,
    recommendation_model: RecommendationModel,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    n_vectors: int,
) -> Dict[int, List[ScheduleColumn]]:
    _ = capacity_cfg

    references: Dict[int, List[ScheduleColumn]] = {}
    use_count = max(1, min(int(n_vectors), 3))

    for week_data in week_data_list:
        candidates: List[np.ndarray] = [np.asarray(week_data.bookings, dtype=float)]

        # Median-quantile candidate (optional for the first exact implementation).
        try:
            q50 = recommendation_model.predict_at_quantile(week_data, q=0.5)
            if q50 is not None:
                candidates.append(np.asarray(q50, dtype=float))
        except Exception:
            pass

        candidates.append(np.asarray(week_data.realized, dtype=float))
        candidates = candidates[:use_count]

        built: List[ScheduleColumn] = []
        for idx, durations in enumerate(candidates):
            col, _ = solve_pricing(
                n_cases=week_data.n_cases,
                durations=np.asarray(durations, dtype=float),
                calendar=week_data.calendar,
                costs=costs,
                solver_cfg=solver_cfg,
                case_eligible_blocks=week_data.case_eligible_blocks,
                turnover=turnover,
                model_name=f"VFCGWarm_{week_data.week_index}_{idx}",
            )
            if col is not None:
                built.append(col)

        deduped = _unique_schedule_columns(built)

        # Final fallback retry on bookings to guarantee one reference per week.
        if not deduped:
            fallback_col, _ = solve_pricing(
                n_cases=week_data.n_cases,
                durations=np.asarray(week_data.bookings, dtype=float),
                calendar=week_data.calendar,
                costs=costs,
                solver_cfg=solver_cfg,
                case_eligible_blocks=week_data.case_eligible_blocks,
                turnover=turnover,
                model_name=f"VFCGWarm_{week_data.week_index}_fallback",
            )
            if fallback_col is None:
                raise RuntimeError(f"Warm-start generation failed for week {week_data.week_index}.")
            deduped = [fallback_col]

        references[int(week_data.week_index)] = deduped

    return references
