"""Typed result containers for exact VFCG components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.column import ScheduleColumn


@dataclass
class OracleResult:
    schedule: ScheduleColumn
    predicted_cost: float
    realized_cost: float
    status: str
    solve_time: float


@dataclass
class VFCGIteration:
    iteration_index: int
    master_status: str
    master_objective: float
    master_bound: float
    master_gap: float
    n_reference_cuts: int
    n_violated_weeks: int
    master_solve_time: float
    oracle_solve_time_total: float


@dataclass
class CertificationResult:
    status: str
    max_violation: float
    reconstructed_objective: float
    master_objective: float
    master_bound: float
    tie_break_flags: list[str] | None = None


@dataclass
class VFCGResult:
    w_optimal: np.ndarray
    objective: float
    n_iterations: int
    certification: CertificationResult
    iterations: list[VFCGIteration]
    total_cuts_added: int
