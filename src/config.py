from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class DataConfig:
    """Data loading and temporal split settings."""

    excel_file_path: str = "data/UHNOperating_RoomScheduling2011-2013.xlsx"
    output_file: str = "outputs/results.json"
    aggregated_output_file: str = "outputs/agg_results.json"
    theta_path: str = "outputs/final_theta.json"
    warmup_weeks: int = 52
    planning_horizon_days: int = 7
    num_horizons: int = 10
    min_samples_procedure: int = 50
    min_samples_surgeon: int = 10
    min_samples_service: int = 20


@dataclass
class OperatingRoomConfig:
    """Operating room calendar and capacity settings."""

    day_start_hour: int = 8
    day_end_hour: int = 16
    block_size_minutes: int = 480
    max_overtime_minutes: float = 60.0
    blocks_per_day: int = 1
    capacity_reduction_percentage: float = 0.5
    min_blocks_after_reduction: int = 1
    min_procedure_duration: int = 30
    max_procedure_duration: int = 480


@dataclass
class CostConfig:
    """Cost coefficients for optimization."""

    rejection_per_case: float = 20.0
    overtime_per_min: float = 15.0
    idle_per_min: float = 10.0
    max_overtime_minutes: float = 240.0


@dataclass
class SAAConfig:
    """Sample Average Approximation settings."""

    run_saa: bool = True
    scenarios: int = 100
    random_seed: int = 42


@dataclass
class MLConfig:
    """Machine learning hyperparameters."""

    knn_neighbors: int = 9
    knn_k_options: List[int] = field(default_factory=lambda: [3, 5, 7, 9, 11])
    knn_optimization_k_candidates: List[int] = field(
        default_factory=lambda: [3, 5, 7, 9, 11, 15, 20, 25]
    )
    knn_cv_folds: int = 5
    lasso_alphas: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 5.0])
    lasso_alpha_asym: float = 0.1
    prediction_error_weight: float = 50.0

    # New XGBoost parameters
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_random_state: int = 42


@dataclass
class IntegratedConfig:
    """Integrated model (predict-then-optimize) settings."""

    lambda_reg: float = 0.5  # L1 penalty on regression coefficients
    num_planning_weeks: int = 10

    # Constraint relaxation parameters (matching document notation)
    tau_pred: float = 0.0     # τ_pred: prediction error tolerance (minutes)
    eta_book: float = 10.0    # η_book: booked-time deviation tolerance (minutes)  
    rho_pred: float = 1.0     # ρ_pred: prediction error penalty weight
    rho_book: float = 1.0     # ρ_book: booked-time deviation penalty weight

    # Multi-parameter experiment settings
    param_values_to_test: List[Dict[str, float]] = field(default_factory=lambda: [
        {"rho_pred": 0.5, "rho_book": 0.5},
        {"rho_pred": 1.0, "rho_book": 1.0},
        {"rho_pred": 2.0, "rho_book": 1.0}
    ])
    multi_param_output_dir: str = "outputs/integrated_multi_param"


@dataclass
class GurobiConfig:
    """Gurobi solver parameters."""

    timelimit: int = 600
    mipgap: float = 0.05
    heuristics: float = 0.1
    output_flag: int = 0
    threads: int = 0  # 0 for Gurobi to decide
    presolve: int = -1  # Auto
    mip_focus: int = 2
    cuts: int = 3
    mip_focus: int = 1  # Focus on finding feasible solutions

    # Debug mode overrides
    timelimit_debug: int = 10
    mipgap_debug: float = 0.10
    num_horizons_debug: int = 1

    # Subproblem settings for Benders
    subproblem_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "TimeLimit": 30,
            "MIPGap": 0.05,
            "OutputFlag": 0,
            "Threads": 1,
        }
    )


@dataclass
class BendersConfig:
    """Benders decomposition settings."""

    lambda_l1_theta: float = 10.0
    theta_bound: float = 200.0
    parallel_subproblems: bool = True
    max_workers: int = None  # None means ThreadPoolExecutor default


@dataclass
class AppConfig:
    """Main application configuration."""

    debug_mode: bool = False

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    operating_room: OperatingRoomConfig = field(default_factory=OperatingRoomConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    saa: SAAConfig = field(default_factory=SAAConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    integrated: IntegratedConfig = field(default_factory=IntegratedConfig)
    gurobi: GurobiConfig = field(default_factory=GurobiConfig)
    benders: BendersConfig = field(default_factory=BendersConfig)


# Create default configuration instance
CONFIG = AppConfig()
