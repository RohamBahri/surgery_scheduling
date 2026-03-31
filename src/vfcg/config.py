"""Configuration for the exact VFCG stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VFCGConfig:
    w_max: float = 10.0
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
