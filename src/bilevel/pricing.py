from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_pricing


def run_pricing(
    w: np.ndarray,
    week_data: WeekRecommendationData,
    recommendation_model: RecommendationModel,
    current_phi: float,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    tol: float = 1e-4,
) -> Optional[ScheduleColumn]:
    """Run pricing for one week and return an improving column if found."""
    d_post = recommendation_model.compute_post_review(w, week_data)
    column, predicted_cost = solve_pricing(
        n_cases=week_data.n_cases,
        durations=d_post,
        calendar=week_data.calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=week_data.case_eligible_blocks,
        turnover=turnover,
        model_name=f"Pricing_w{week_data.week_index}",
    )
    if column is None:
        return None
    if predicted_cost < current_phi - tol:
        return column
    return None
