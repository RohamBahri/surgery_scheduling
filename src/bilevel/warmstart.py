from __future__ import annotations

from typing import List

from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_pricing


def generate_warmstart_columns(
    week_data: WeekRecommendationData,
    recommendation_model: RecommendationModel,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    turnover: float,
) -> List[ScheduleColumn]:
    columns: List[ScheduleColumn] = []

    candidates = [
        week_data.bookings,
        recommendation_model.predict_at_quantile(week_data, q=0.5),
        week_data.realized,
    ]

    for i, durations in enumerate(candidates):
        col, _ = solve_pricing(
            n_cases=week_data.n_cases,
            durations=durations,
            calendar=week_data.calendar,
            costs=costs,
            solver_cfg=solver_cfg,
            case_eligible_blocks=week_data.case_eligible_blocks,
            turnover=turnover,
            model_name=f"WarmStart_{week_data.week_index}_{i}",
        )
        if col is not None and col not in columns:
            columns.append(col)

    return columns
