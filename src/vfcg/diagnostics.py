"""Diagnostics helpers for VFCG iteration logging."""

from __future__ import annotations

import logging

from src.vfcg.types import VFCGIteration

logger = logging.getLogger(__name__)


def log_iteration_summary(iteration: VFCGIteration) -> None:
    logger.info(
        "VFCG iter=%d status=%s obj=%.4f bound=%.4f gap=%.4f cuts=%d violated=%d t_master=%.2fs t_oracle=%.2fs",
        iteration.iteration_index,
        iteration.master_status,
        iteration.master_objective,
        iteration.master_bound,
        iteration.master_gap,
        iteration.n_reference_cuts,
        iteration.n_violated_weeks,
        iteration.master_solve_time,
        iteration.oracle_solve_time_total,
    )
