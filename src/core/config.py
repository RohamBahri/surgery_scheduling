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
    training-period weekday room patterns filtered by activation
    regularity.

    Capacity is day-specific: ``friday_capacity_minutes`` allows a
    shorter nominal block on Fridays, matching the empirical finding
    that Friday spans are ~45 min shorter than Mon–Thu.
    """
    block_capacity_minutes: float = 480.0
    friday_capacity_minutes: float = 450.0
    activation_cost_per_block: float = 2000.0

    # Template filtering: only include room-day pairs that were active
    # in at least this fraction of training weeks.  The data report
    # shows 153/168 templates activate ≥ 50% of weeks; setting this
    # to 0.25 is conservative and keeps the pool realistic.
    min_activation_rate: float = 0.25

    # Mean turnover between consecutive cases in the same block (minutes).
    # Used to adjust effective capacity: C_eff = C - (avg_cases - 1) * turnover.
    # Set to 0.0 to disable turnover adjustment.
    # The data report shows mean turnover of 28.3 min (capped at 60 min).
    mean_turnover_minutes: float = 28.0


@dataclass
class EligibilityConfig:
    """Controls how case-level eligibility sets B_ti are built.

    The paper's formulation (Section 3.5) restricts each case to
    B_ti ⊆ B_t based on service compatibility, site, and surgeon
    history.  The data report shows room sets are NOT disjoint across
    services but ARE nearly disjoint across sites.
    """
    # Hard filter: case can only go to blocks whose room has historically
    # served the case's surgical service.
    use_service_room_filter: bool = True

    # Hard filter: case can only go to blocks at the same site as the
    # case's historical operating room.  Nearly free because sites are
    # almost room-disjoint in the UHN data.
    use_site_filter: bool = True

    # Soft narrowing: if enabled, restrict each case to blocks in the
    # surgeon's top-K historical room-day templates, with fallback to
    # service-level eligibility if the surgeon has too few templates.
    use_surgeon_preference: bool = False
    surgeon_top_k_templates: int = 5
    surgeon_min_cases_for_preference: int = 30


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
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


CONFIG = Config()