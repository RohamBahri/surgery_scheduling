"""Main VFCG loop with lazy follower-optimality cuts."""

from __future__ import annotations

import logging

import numpy as np

from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.vfcg.certify import certify
from src.vfcg.diagnostics import log_iteration_summary
from src.vfcg.master import solve_native_master
from src.vfcg.oracle import ExactFollowerOracle
from src.vfcg.types import CertificationResult, VFCGIteration, VFCGResult
from src.vfcg.warmstart import generate_warmstart_references

logger = logging.getLogger(__name__)


def vfcg_solve(
    week_data_list: list[WeekRecommendationData],
    recommendation_model: RecommendationModel,
    config: Config,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
) -> VFCGResult:
    oracle = ExactFollowerOracle()
    reference_sets = generate_warmstart_references(
        week_data_list=week_data_list,
        recommendation_model=recommendation_model,
        costs=costs,
        capacity_cfg=capacity_cfg,
        solver_cfg=solver_cfg,
        turnover=turnover,
        n_vectors=config.vfcg.n_warmstart_vectors,
    )

    iterations: list[VFCGIteration] = []
    total_cuts_added = 0

    final_master = None

    for k in range(config.vfcg.max_iterations):
        master_res = solve_native_master(
            week_data_list=week_data_list,
            reference_sets=reference_sets,
            recommendation_model=recommendation_model,
            config=config,
            costs=costs,
            capacity_cfg=capacity_cfg,
            solver_cfg=solver_cfg,
            turnover=turnover,
        )
        final_master = master_res

        violated_weeks = []
        oracle_time_total = 0.0

        if not master_res.is_fallback:
            for wd in week_data_list:
                oracle_res = oracle.solve(
                    week_data=wd,
                    w=master_res.weights,
                    recommendation_model=recommendation_model,
                    costs=costs,
                    capacity_cfg=capacity_cfg,
                    solver_cfg=solver_cfg,
                    turnover=turnover,
                    tol=config.vfcg.convergence_tol,
                )
                oracle_time_total += oracle_res.solve_time

                d_post = np.asarray(recommendation_model.compute_post_review(master_res.weights, wd), dtype=float)
                master_schedule = master_res.schedules_by_week[int(wd.week_index)]
                current_master_pred = float(master_schedule.compute_cost(d_post, costs, turnover))

                if oracle_res.predicted_cost < current_master_pred - config.vfcg.convergence_tol:
                    violated_weeks.append((wd, oracle_res))

        for wd, oracle_res in violated_weeks:
            week_key = int(wd.week_index)
            ref_list = reference_sets.setdefault(week_key, [])
            if oracle_res.schedule not in ref_list:
                ref_list.append(oracle_res.schedule)
                total_cuts_added += 1
            else:
                logger.warning(
                    "Week %d flagged violated but oracle schedule already in references; potential numerical inconsistency.",
                    week_key,
                )

        iter_record = VFCGIteration(
            iteration_index=k,
            master_status=master_res.status,
            master_objective=master_res.objective,
            master_bound=master_res.bound,
            master_gap=master_res.gap,
            n_reference_cuts=total_cuts_added,
            n_violated_weeks=len(violated_weeks),
            master_solve_time=master_res.solve_time,
            oracle_solve_time_total=oracle_time_total,
        )
        iterations.append(iter_record)
        log_iteration_summary(iter_record)

        if master_res.is_fallback:
            logger.warning("Master returned fallback solution; terminating loop and moving to certification.")
            break

        if len(violated_weeks) == 0:
            break

    if final_master is None:
        raise RuntimeError("VFCG solve loop produced no master result.")

    certification = certify(
        w=final_master.weights,
        week_data_list=week_data_list,
        recommendation_model=recommendation_model,
        oracle=oracle,
        config=config,
        costs=costs,
        capacity_cfg=capacity_cfg,
        solver_cfg=solver_cfg,
        turnover=turnover,
        master_objective=final_master.objective,
        master_bound=final_master.bound,
        weekly_schedules=final_master.schedules_by_week,
    )

    if final_master.is_fallback:
        tie_break_flags = list(certification.tie_break_flags or [])
        if "master_fallback_used" not in tie_break_flags:
            tie_break_flags.append("master_fallback_used")
        certification = CertificationResult(
            status="TERMINATED_UNVERIFIED",
            max_violation=certification.max_violation,
            reconstructed_objective=certification.reconstructed_objective,
            master_objective=certification.master_objective,
            master_bound=certification.master_bound,
            tie_break_flags=tie_break_flags,
        )

    return VFCGResult(
        w_optimal=np.asarray(final_master.weights, dtype=float),
        objective=float(final_master.objective),
        n_iterations=len(iterations),
        certification=certification,
        iterations=iterations,
        total_cuts_added=total_cuts_added,
    )
