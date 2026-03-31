"""Independent certification for VFCG outputs."""

from __future__ import annotations

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.vfcg.oracle import ExactFollowerOracle
from src.vfcg.types import CertificationResult


def certify(
    w: np.ndarray,
    week_data_list: list[WeekRecommendationData],
    recommendation_model: RecommendationModel,
    oracle: ExactFollowerOracle,
    config: Config,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    master_objective: float,
    master_bound: float,
    weekly_schedules: dict[int, ScheduleColumn],
) -> CertificationResult:
    tol = config.vfcg.certification_tol

    max_violation = -float("inf")
    realized_costs = []
    tie_break_flags: list[str] = []

    for wd in week_data_list:
        d_post = np.asarray(recommendation_model.compute_post_review(w, wd), dtype=float)
        oracle_res = oracle.solve(
            week_data=wd,
            w=w,
            recommendation_model=recommendation_model,
            costs=costs,
            capacity_cfg=capacity_cfg,
            solver_cfg=solver_cfg,
            turnover=turnover,
            tol=tol,
        )

        master_schedule = weekly_schedules[int(wd.week_index)]
        master_pred = float(master_schedule.compute_cost(d_post, costs, turnover))
        master_real = float(master_schedule.compute_cost(np.asarray(wd.realized, dtype=float), costs, turnover))

        violation = master_pred - oracle_res.predicted_cost
        max_violation = max(max_violation, violation)
        realized_costs.append(master_real)

        if abs(master_pred - oracle_res.predicted_cost) <= tol and oracle_res.realized_cost < master_real - tol:
            tie_break_flags.append(
                f"week={wd.week_index}: equal predicted cost but oracle realized={oracle_res.realized_cost:.6f} < master realized={master_real:.6f}"
            )

    if max_violation == -float("inf"):
        max_violation = 0.0

    reconstructed_objective = float(np.mean(realized_costs)) if realized_costs else 0.0
    obj_match = abs(reconstructed_objective - master_objective) <= tol
    max_ok = max_violation <= tol
    no_tie_issues = len(tie_break_flags) == 0

    if max_ok and obj_match and no_tie_issues:
        status = "OPTIMAL_VERIFIED"
    elif max_violation <= 100 * tol and abs(reconstructed_objective - master_objective) <= 100 * tol:
        status = "TERMINATED_UNVERIFIED"
    else:
        status = "FAILED_CERTIFICATION"

    return CertificationResult(
        status=status,
        max_violation=float(max_violation),
        reconstructed_objective=float(reconstructed_objective),
        master_objective=float(master_objective),
        master_bound=float(master_bound),
        tie_break_flags=tie_break_flags or None,
    )
