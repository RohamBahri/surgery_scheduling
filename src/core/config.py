"""Application configuration.

All tunable parameters live here.  Import the module-level ``CONFIG``
instance everywhere; override individual fields as needed before the
experiment starts.
"""

from dataclasses import dataclass, field


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
    """Block capacity and activation cost (paper Section 4.2).

    Blocks are *candidate* slots that the planner may open at cost
    ``activation_cost_per_block``.  The candidate pool is built from
    training-period weekday room patterns.
    """
    block_capacity_minutes: float = 480.0
    activation_cost_per_block: float = 2000.0


@dataclass
class CostConfig:
    """Operational cost coefficients (paper Section 4.2).

    ``deferral_per_case`` is a flat per-case penalty — every deferred
    patient incurs the same access cost regardless of case complexity.
    Case-level heterogeneity in resource consumption is captured by
    block-load constraints, not by the deferral penalty.
    """
    overtime_per_minute: float = 15.0
    idle_per_minute: float = 10.0
    deferral_per_case: float = 2000.0
    max_overtime_minutes: float = 240.0


@dataclass
class SolverConfig:
    time_limit_seconds: int = 600
    mip_gap: float = 0.05
    threads: int = 0        # 0 = let Gurobi decide
    verbose: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


CONFIG = Config()
