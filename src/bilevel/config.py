from dataclasses import dataclass


@dataclass
class BilevelConfig:
    w_max: float = 10.0
    credibility_eta: float = 1.05
    plausibility_tau_L: float = 0.01
    plausibility_tau_U: float = 0.99
    max_ccg_iterations: int = 100
    pricing_time_limit: int = 300
    master_time_limit: int = 600
    master_mip_gap: float = 0.01
    n_warmstart_columns: int = 3
    convergence_tol: float = 1e-4
