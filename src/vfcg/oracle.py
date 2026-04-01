"""Exact follower oracle wrapper for VFCG."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from src.core.config import CapacityConfig, CostConfig, SolverConfig
from src.core.types import CaseRecord
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_pricing, solve_weekly_optimistic
from src.vfcg.types import OracleResult


def build_case_records_from_week_data(week_data: WeekRecommendationData) -> list[CaseRecord]:
    realized = np.asarray(week_data.realized, dtype=float)
    return [
        CaseRecord(
            case_id=i,
            procedure_id="UNK",
            surgeon_code=week_data.surgeon_codes[i] if i < len(week_data.surgeon_codes) else "UNK",
            service="Other",
            patient_type="Elective",
            operating_room="",
            booked_duration_min=float(week_data.bookings[i]),
            actual_duration_min=float(realized[i]),
            actual_start=datetime(2020, 1, 1),
            week_of_year=1,
            month=1,
            year=2020,
        )
        for i in range(week_data.n_cases)
    ]


class ExactFollowerOracle:
    """Single authoritative follower oracle using the compact weekly MIP."""

    def solve(
        self,
        week_data: WeekRecommendationData,
        w: np.ndarray,
        recommendation_model: RecommendationModel,
        costs: CostConfig,
        capacity_cfg: CapacityConfig,
        solver_cfg: SolverConfig,
        turnover: float,
        tol: float = 1e-6,
    ) -> OracleResult:
        _ = capacity_cfg

        d_post = np.asarray(recommendation_model.compute_post_review(w, week_data), dtype=float)
        realized = np.asarray(week_data.realized, dtype=float)

        cases = build_case_records_from_week_data(week_data)

        schedule, predicted_cost, realized_cost, status, solve_time = solve_weekly_optimistic(
            cases=cases,
            planning_durations=d_post,
            realized_durations=realized,
            calendar=week_data.calendar,
            costs=costs,
            solver_cfg=solver_cfg,
            case_eligible_blocks=week_data.case_eligible_blocks,
            turnover=turnover,
            model_name=f"vfcg_oracle_w{week_data.week_index}",
            tol=tol,
        )

        if schedule is None:
            raise RuntimeError(f"Exact follower oracle failed to produce a schedule (status={status}).")

        return OracleResult(
            schedule=schedule,
            predicted_cost=float(predicted_cost),
            realized_cost=float(realized_cost),
            status=status,
            solve_time=float(solve_time),
        )

    def solve_fast(
        self,
        week_data: WeekRecommendationData,
        w: np.ndarray,
        recommendation_model: RecommendationModel,
        costs: CostConfig,
        capacity_cfg: CapacityConfig,
        solver_cfg: SolverConfig,
        turnover: float,
    ) -> OracleResult:
        _ = capacity_cfg
        t0 = time.perf_counter()

        d_post = np.asarray(recommendation_model.compute_post_review(w, week_data), dtype=float)
        col, predicted_cost = solve_pricing(
            n_cases=week_data.n_cases,
            durations=d_post,
            calendar=week_data.calendar,
            costs=costs,
            solver_cfg=solver_cfg,
            case_eligible_blocks=week_data.case_eligible_blocks,
            turnover=turnover,
            model_name=f"vfcg_oracle_fast_w{week_data.week_index}",
        )
        if col is None:
            raise RuntimeError("Exact follower oracle fast solve failed to produce a schedule.")

        realized = np.asarray(week_data.realized, dtype=float)
        realized_cost = float(col.compute_cost(realized, costs, turnover))
        return OracleResult(
            schedule=col,
            predicted_cost=float(predicted_cost),
            realized_cost=realized_cost,
            status="OPTIMAL",
            solve_time=float(time.perf_counter() - t0),
        )
