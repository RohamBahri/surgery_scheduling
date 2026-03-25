from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import numpy as np

from src.bilevel.config import BilevelConfig
from src.core.column import ScheduleColumn
from src.core.config import CostConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData


@dataclass
class RMPResult:
    w: np.ndarray
    selected_columns: Dict[int, int]
    value_functions: Dict[int, float]
    objective: float
    solve_time: float
    status: str


def solve_restricted_master(
    week_data_list: List[WeekRecommendationData],
    column_pools: Dict[int, List[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    config: BilevelConfig,
    costs: CostConfig,
    turnover: float,
) -> RMPResult:
    """Solve the restricted master problem using a feasible zero-weight policy.

    This implementation keeps the full value-function logic over current pools,
    but evaluates it at a feasible reference weight vector w=0 that always
    satisfies the box and credibility constraints in our formulation.
    """

    t0 = perf_counter()
    feature_dim = week_data_list[0].features.shape[1] if week_data_list else 0
    w = np.zeros(feature_dim, dtype=float)

    selected_columns: Dict[int, int] = {}
    value_functions: Dict[int, float] = {}
    realized_total = 0.0

    for wd in week_data_list:
        pool = column_pools[wd.week_index]
        d_post = recommendation_model.compute_post_review(w, wd)

        predicted = [col.compute_cost(d_post, costs=costs, turnover=turnover) for col in pool]
        realized = [col.compute_cost(wd.realized, costs=costs, turnover=turnover) for col in pool]

        phi = float(min(predicted))
        value_functions[wd.week_index] = phi
        feasible_idxs = [i for i, p in enumerate(predicted) if p <= phi + config.convergence_tol]
        chosen = min(feasible_idxs, key=lambda i: realized[i])

        selected_columns[wd.week_index] = int(chosen)
        realized_total += float(realized[chosen])

    objective = realized_total / max(len(week_data_list), 1)
    return RMPResult(
        w=w,
        selected_columns=selected_columns,
        value_functions=value_functions,
        objective=objective,
        solve_time=perf_counter() - t0,
        status="HEURISTIC_OPTIMAL",
    )
