"""Application configuration."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    excel_file_path: str = "data/UHNOperating_RoomScheduling2011-2013.xlsx"
    warmup_weeks: int = 52
    horizon_days: int = 7
    num_horizons: int = 53
    min_samples_procedure: int = 50
    min_samples_surgeon: int = 10
    min_samples_service: int = 20


@dataclass
class CapacityConfig:
    block_capacity_minutes: float = 480.0
    activation_cost_per_block: float = 2000.0
    min_activation_rate: float = 0.25
    turnover_minutes: float = 30.0
    eligibility_min_weeks: int = 3


@dataclass
class EligibilityConfig:
    use_service_room_filter: bool = True
    use_site_filter: bool = True
    use_surgeon_preference: bool = False
    surgeon_top_k_templates: int = 5
    surgeon_min_cases_for_preference: int = 30


@dataclass
class CostConfig:
    overtime_per_minute: float = 15.0
    idle_per_minute: float = 10.0
    deferral_per_case: float = 2000.0
    max_overtime_minutes: float = 240.0


@dataclass
class SolverConfig:
    time_limit_seconds: int = 600
    mip_gap: float = 0.05
    threads: int = 0
    verbose: bool = False


@dataclass
class ExperimentScopeConfig:
    planning_sites: Tuple[str, ...] = ("TGH", "TWH")
    planning_weekdays: Tuple[int, ...] = (0, 1, 2, 3, 4)
    use_all_sites_for_warmup: bool = True
    stride_days: int = 7


@dataclass
class QuantileModelConfig:
    q_grid_size: int = 99
    alpha: float = 0.01
    solver: str = "highs"
    max_iter: int = 10000


@dataclass
class InverseConfig:
    n_min: int = 50
    q_grid_size: int = 99
    pooling_lambda: float = 50.0


@dataclass
class ResponseConfig:
    delta_max_days: int = 60
    n_folds: int = 5
    min_pairs: int = 30
    h_grid_max: float = 60.0
    h_grid_step: float = 2.0
    a_min: float = 0.01
    a_max: float = 1.0


@dataclass
class ProfileConfig:
    n_profiles_per_service: int = 3
    min_profiles_total: int = 5
    max_profiles_total: int = 20


@dataclass
class BootstrapConfig:
    n_bootstrap: int = 200
    random_seed: int = 42
    n_jobs: int = 1
    q_grid_size_bootstrap: int = 25
    n_folds_bootstrap: int = 3


@dataclass
class EstimationConfig:
    quantile_model: QuantileModelConfig = field(default_factory=QuantileModelConfig)
    inverse: InverseConfig = field(default_factory=InverseConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    scope: ExperimentScopeConfig = field(default_factory=ExperimentScopeConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)


CONFIG = Config()
