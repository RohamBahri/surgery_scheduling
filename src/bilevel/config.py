from dataclasses import dataclass


@dataclass
class LegacyCCGConfig:
    w_max: float = 10.0
    # Credibility threshold is e_pred_max = credibility_eta * MAE_base.
    credibility_eta: float = 1.05
    plausibility_tau_L: float = 0.01
    plausibility_tau_U: float = 0.99
    max_iterations: int = 100
    pricing_time_limit: int = 300
    master_time_limit: int = 600
    master_mip_gap: float = 0.01
    n_warmstart_columns: int = 3
    convergence_tol: float = 1e-4
