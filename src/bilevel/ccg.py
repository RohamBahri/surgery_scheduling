from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.bilevel.config import LegacyCCGConfig
from src.bilevel.master import RMPResult, solve_restricted_master
from src.bilevel.pricing import run_pricing
from src.bilevel.warmstart import generate_warmstart_columns
from src.core.column import ScheduleColumn
from src.core.config import CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData


@dataclass
class CCGResult:
    w_optimal: np.ndarray
    objective: float
    n_iterations: int
    column_pools: Dict[int, List[ScheduleColumn]]
    rmp_result: RMPResult


def solve_bilevel_ccg(
    week_data_list: List[WeekRecommendationData],
    recommendation_model: RecommendationModel,
    config: LegacyCCGConfig,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    turnover: float,
) -> CCGResult:
    column_pools: Dict[int, List[ScheduleColumn]] = {}
    for wd in week_data_list:
        cols = generate_warmstart_columns(wd, recommendation_model, costs, solver_cfg, turnover)
        if not cols:
            raise RuntimeError(f"No feasible warm-start column for week {wd.week_index}")
        column_pools[wd.week_index] = cols

    rmp_result: RMPResult | None = None
    for iteration in range(config.max_iterations):
        rmp_result = solve_restricted_master(
            week_data_list=week_data_list,
            column_pools=column_pools,
            recommendation_model=recommendation_model,
            config=config,
            costs=costs,
            turnover=turnover,
        )

        any_added = False
        for wd in week_data_list:
            new_col = run_pricing(
                w=rmp_result.w,
                week_data=wd,
                recommendation_model=recommendation_model,
                current_phi=rmp_result.value_functions[wd.week_index],
                costs=costs,
                solver_cfg=solver_cfg,
                turnover=turnover,
                tol=config.convergence_tol,
            )
            if new_col is not None and new_col not in column_pools[wd.week_index]:
                column_pools[wd.week_index].append(new_col)
                any_added = True

        if not any_added:
            break

    assert rmp_result is not None
    final_cost = 0.0
    for wd in week_data_list:
        t = wd.week_index
        selected_idx = rmp_result.selected_columns[t]
        final_cost += column_pools[t][selected_idx].compute_cost(wd.realized, costs, turnover)
    final_cost /= max(len(week_data_list), 1)

    return CCGResult(
        w_optimal=rmp_result.w,
        objective=final_cost,
        n_iterations=iteration + 1,
        column_pools=column_pools,
        rmp_result=rmp_result,
    )
