"""Warm-start reference generation for exact VFCG."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_pricing
from src.vfcg.oracle import ExactFollowerOracle

logger = logging.getLogger(__name__)


def _unique_schedule_columns(columns: Iterable[ScheduleColumn]) -> List[ScheduleColumn]:
    unique: List[ScheduleColumn] = []
    seen: set[ScheduleColumn] = set()
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        unique.append(col)
    return unique


def estimate_linear_regression_warmstart(
    week_data_list: list[WeekRecommendationData],
    w_max: float,
) -> np.ndarray:
    """Estimate an initial weight vector from a minimum-norm linear regression.

    The target is the clipped realized correction, i.e.
        clip(realized, L, U) - bookings,
    so the seed respects the master feasibility box on post-review durations.

    Because the current feature map can be rank-deficient, we intentionally use
    ``np.linalg.lstsq`` and keep the minimum-norm solution returned by the
    pseudoinverse. The seed is then clipped to ``[-w_max, w_max]``.
    """
    if not week_data_list:
        return np.zeros(0, dtype=float)

    feat_dim = int(week_data_list[0].features.shape[1]) if week_data_list[0].features.ndim == 2 else 0
    if feat_dim == 0:
        return np.zeros(0, dtype=float)

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []

    for wd in week_data_list:
        X = np.asarray(wd.features, dtype=float)
        if X.size == 0:
            continue

        clipped_realized = np.clip(
            np.asarray(wd.realized, dtype=float),
            np.asarray(wd.L_bounds, dtype=float),
            np.asarray(wd.U_bounds, dtype=float),
        )
        target_delta = clipped_realized - np.asarray(wd.bookings, dtype=float)

        x_blocks.append(X)
        y_blocks.append(np.asarray(target_delta, dtype=float))

    if not x_blocks:
        return np.zeros(feat_dim, dtype=float)

    X_stack = np.vstack(x_blocks)
    y_stack = np.concatenate(y_blocks)

    try:
        coef, _, rank, _ = np.linalg.lstsq(X_stack, y_stack, rcond=None)
        if int(rank) < int(X_stack.shape[1]):
            logger.info(
                "Initial linear-regression warm start is rank-deficient (rank=%d < %d); using minimum-norm least-squares seed.",
                int(rank),
                int(X_stack.shape[1]),
            )
    except np.linalg.LinAlgError:
        logger.exception("Least-squares warm start failed; reverting to zero vector.")
        return np.zeros(feat_dim, dtype=float)

    coef = np.asarray(coef, dtype=float).reshape(-1)
    if coef.shape[0] != feat_dim:
        logger.warning(
            "Initial linear-regression warm start returned dimension %d, expected %d; reverting to zero vector.",
            int(coef.shape[0]),
            feat_dim,
        )
        return np.zeros(feat_dim, dtype=float)

    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    coef = np.clip(coef, -float(w_max), float(w_max))
    return coef


def generate_seed_weights(
    week_data_list: list[WeekRecommendationData],
    w_max: float,
    scales: tuple[float, ...],
) -> list[np.ndarray]:
    base = estimate_linear_regression_warmstart(
        week_data_list=week_data_list,
        w_max=w_max,
    )
    active_scales = scales or (0.0, 1.0)
    seeds: list[np.ndarray] = []
    for s in active_scales:
        cand = np.clip(float(s) * base, -float(w_max), float(w_max))
        if not any(np.allclose(cand, prev, atol=1e-9, rtol=0.0) for prev in seeds):
            seeds.append(cand)
    return seeds


def _append_duration_reference(
    *,
    built: list[ScheduleColumn],
    week_data: WeekRecommendationData,
    durations: np.ndarray,
    costs: CostConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    tag: str,
) -> None:
    col, _ = solve_pricing(
        n_cases=week_data.n_cases,
        durations=np.asarray(durations, dtype=float),
        calendar=week_data.calendar,
        costs=costs,
        solver_cfg=solver_cfg,
        case_eligible_blocks=week_data.case_eligible_blocks,
        turnover=turnover,
        model_name=f"VFCGWarm_{week_data.week_index}_{tag}",
    )
    if col is not None:
        built.append(col)


def select_best_reference_schedules(
    week_data_list: list[WeekRecommendationData],
    reference_sets: Dict[int, List[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    costs: CostConfig,
    turnover: float,
    weights: np.ndarray,
) -> Dict[int, ScheduleColumn]:
    """Choose the cheapest existing reference schedule under the seeded weights.

    This keeps the initial master start feasible with respect to *all* initial
    follower cuts: for each week we pick the minimum-predicted-cost schedule
    among the currently available references under the seeded ``d_post``.
    """
    schedules: Dict[int, ScheduleColumn] = {}
    for wd in week_data_list:
        week_key = int(wd.week_index)
        ref_list = reference_sets.get(week_key, [])
        if not ref_list:
            raise RuntimeError(f"No reference schedules available for week {week_key}.")

        d_post = np.asarray(recommendation_model.compute_post_review(weights, wd), dtype=float)
        best_ref = min(ref_list, key=lambda col: float(col.compute_cost(d_post, costs, turnover)))
        schedules[week_key] = best_ref

    return schedules


def generate_warmstart_references(
    week_data_list,
    recommendation_model: RecommendationModel,
    config: Config,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    n_vectors: int,
) -> Dict[int, List[ScheduleColumn]]:
    _ = capacity_cfg

    references: Dict[int, List[ScheduleColumn]] = {}
    n_seed = max(1, int(n_vectors))
    active_scales = tuple(config.vfcg.initial_reference_seed_scales[:n_seed]) or (0.0, 1.0)
    seed_weights = generate_seed_weights(
        week_data_list=week_data_list,
        w_max=config.vfcg.w_max,
        scales=active_scales,
    )
    oracle = ExactFollowerOracle()

    for week_data in week_data_list:
        built: List[ScheduleColumn] = []

        if config.vfcg.initial_reference_include_booking:
            _append_duration_reference(
                built=built,
                week_data=week_data,
                durations=np.asarray(week_data.bookings, dtype=float),
                costs=costs,
                solver_cfg=solver_cfg,
                turnover=turnover,
                tag="booking",
            )

        if config.vfcg.initial_reference_include_q50:
            try:
                q50 = recommendation_model.predict_at_quantile(week_data, q=0.5)
                if q50 is not None:
                    _append_duration_reference(
                        built=built,
                        week_data=week_data,
                        durations=np.asarray(q50, dtype=float),
                        costs=costs,
                        solver_cfg=solver_cfg,
                        turnover=turnover,
                        tag="q50",
                    )
            except Exception:
                logger.exception("q50 warm-start reference failed for week %s", week_data.week_index)

        if config.vfcg.initial_reference_include_realized:
            _append_duration_reference(
                built=built,
                week_data=week_data,
                durations=np.asarray(week_data.realized, dtype=float),
                costs=costs,
                solver_cfg=solver_cfg,
                turnover=turnover,
                tag="realized",
            )

        for idx, w_seed in enumerate(seed_weights):
            try:
                oracle_res = oracle.solve(
                    week_data=week_data,
                    w=np.asarray(w_seed, dtype=float),
                    recommendation_model=recommendation_model,
                    costs=costs,
                    capacity_cfg=capacity_cfg,
                    solver_cfg=solver_cfg,
                    turnover=turnover,
                )
                built.append(oracle_res.schedule)
            except Exception:
                logger.exception(
                    "Weight-seeded oracle warm-start failed for week %s seed=%d",
                    week_data.week_index,
                    idx,
                )

        deduped = _unique_schedule_columns(built)
        max_refs = config.vfcg.max_initial_references_per_week
        if max_refs is not None and len(deduped) > int(max_refs):
            deduped = deduped[: int(max_refs)]

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
