"""Configuration for the exact VFCG stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VFCGConfig:
    w_max: float = 100.0
    credibility_eta: float = 1.05
    plausibility_tau_L: float = 0.01
    plausibility_tau_U: float = 0.99
    max_iterations: int = 50
    convergence_tol: float = 1e-4
    master_time_limit: int = 1800
    master_mip_gap: float = 0.01
    oracle_time_limit: int = 300
    certification_tol: float = 1e-6
    n_warmstart_vectors: int = 3
    max_training_weeks: int | None = 20
    initial_reference_seed_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    initial_reference_include_booking: bool = True
    initial_reference_include_q50: bool = True
    initial_reference_include_realized: bool = True
    max_initial_references_per_week: int | None = 12
    credibility_mode: str = "mae_penalty"  # "hard" | "mae_penalty"
    credibility_penalty_rho: float = 10.0
    l1_penalty_rho: float = 1.0
