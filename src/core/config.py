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
    friday_capacity_minutes: float = 450.0
    activation_cost_per_block: float = 2000.0
    min_activation_rate: float = 0.25
    turnover_minutes: float = 32.0
    eligibility_min_weeks: int = 3
    fixed_block_threshold: float = 0.75


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
class SurgeonGroupingConfig:
    default_max_blocks_per_day: int = 1
    adaptive_relaxation: bool = True


@dataclass
class ExperimentScopeConfig:
    planning_sites: Tuple[str, ...] = ("TGH", "TWH")
    planning_weekdays: Tuple[int, ...] = (0, 1, 2, 3, 4)
    use_all_sites_for_warmup: bool = True
    stride_days: int = 7


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    surgeon_grouping: SurgeonGroupingConfig = field(default_factory=SurgeonGroupingConfig)
    scope: ExperimentScopeConfig = field(default_factory=ExperimentScopeConfig)


CONFIG = Config()
