"""Independent certification for VFCG outputs."""

from __future__ import annotations

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.vfcg.master import _compute_mae_base
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
    master_realized_objective: float,
    master_bound: float,
    weekly_schedules: dict[int, ScheduleColumn],
) -> CertificationResult:
    tol = config.vfcg.certification_tol

    max_violation = -float("inf")
    realized_costs = []
    diagnostic_flags: list[str] = []
    abs_error_total = 0.0
    total_cases = 0

    for wd in week_data_list:
        d_post = np.asarray(recommendation_model.compute_post_review(w, wd), dtype=float)
        realized = np.asarray(wd.realized, dtype=float)
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
        master_real = float(master_schedule.compute_cost(realized, costs, turnover))

        violation = master_pred - oracle_res.predicted_cost
        max_violation = max(max_violation, violation)
        realized_costs.append(master_real)
        delta_rec = np.asarray(
            recommendation_model.compute_corrections(
                w=w,
                X=wd.features,
                bookings=wd.bookings,
                L=wd.L_bounds,
                U=wd.U_bounds,
            ),
            dtype=float,
        )
        d_rec = np.asarray(wd.bookings, dtype=float) + delta_rec
        abs_error_total += float(np.sum(np.abs(realized - d_rec)))
        total_cases += int(wd.n_cases)

        if abs(master_pred - oracle_res.predicted_cost) <= tol and oracle_res.realized_cost < master_real - tol:
            diagnostic_flags.append(
                f"week={wd.week_index}: equal predicted cost but lower realized-cost oracle witness exists"
            )

    if max_violation == -float("inf"):
        max_violation = 0.0

    reconstructed_realized_objective = float(np.mean(realized_costs)) if realized_costs else 0.0
    reconstructed_credibility_mae = abs_error_total / max(1, total_cases)
    e_pred_max = config.vfcg.credibility_eta * _compute_mae_base(week_data_list)
    reconstructed_credibility_slack = 0.0
    reconstructed_objective = reconstructed_realized_objective + float(config.vfcg.l1_penalty_rho) * float(np.sum(np.abs(np.asarray(w, dtype=float))))
    if str(config.vfcg.credibility_mode).strip().lower() in {"mae_penalty", "elastic_penalty"}:
        reconstructed_objective = reconstructed_objective + float(config.vfcg.credibility_penalty_rho) * reconstructed_credibility_mae

    obj_match = abs(reconstructed_objective - master_objective) <= tol
    real_match = abs(reconstructed_realized_objective - master_realized_objective) <= tol
    max_ok = max_violation <= tol

    if max_ok and obj_match and real_match:
        status = "OPTIMAL_VERIFIED"
    elif max_violation <= 100 * tol and abs(reconstructed_objective - master_objective) <= 100 * tol and abs(reconstructed_realized_objective - master_realized_objective) <= 100 * tol:
        status = "TERMINATED_UNVERIFIED"
    else:
        status = "FAILED_CERTIFICATION"

    return CertificationResult(
        status=status,
        max_violation=float(max_violation),
        reconstructed_objective=float(reconstructed_objective),
        reconstructed_realized_objective=float(reconstructed_realized_objective),
        reconstructed_credibility_mae=float(reconstructed_credibility_mae),
        reconstructed_credibility_slack=float(reconstructed_credibility_slack),
        master_objective=float(master_objective),
        master_realized_objective=float(master_realized_objective),
        master_bound=float(master_bound),
        tie_break_flags=diagnostic_flags or None,
    )
