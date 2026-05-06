from __future__ import annotations

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.core.types import BlockCalendar, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData
from src.vfcg.certify import certify
from src.vfcg.types import OracleResult


class _Rec:
    def compute_post_review(self, w: np.ndarray, week_data: WeekRecommendationData) -> np.ndarray:
        _ = w
        return week_data.bookings


def _week() -> WeekRecommendationData:
    b = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)
    return WeekRecommendationData(
        week_index=0,
        n_cases=1,
        features=np.array([[1.0]], dtype=float),
        bookings=np.array([100.0]),
        realized=np.array([100.0]),
        L_bounds=np.array([80.0]),
        U_bounds=np.array([120.0]),
        surgeon_codes=["S1"],
        sos2_data=[SOS2CaseData(0, 0, np.array([-10.0, 0.0, 10.0]), np.array([-5.0, 0.0, 5.0]), 100.0, 80.0, 120.0)],
        case_eligible_blocks={0: [b.id]},
        calendar=BlockCalendar([b]),
    )


def _assign_col() -> ScheduleColumn:
    bid = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0).id
    return ScheduleColumn(
        z_assign={(0, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=1,
        block_capacities={bid: 100.0},
        block_activation_costs={bid: 10.0},
    )


def _defer_col() -> ScheduleColumn:
    bid = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0).id
    return ScheduleColumn(
        z_assign={},
        z_defer=frozenset({0}),
        v_open=frozenset(),
        y_used=frozenset(),
        n_cases=1,
        block_capacities={bid: 100.0},
        block_activation_costs={bid: 10.0},
    )


def _cfg() -> Config:
    cfg = Config()
    cfg.vfcg.certification_tol = 1e-6
    return cfg


def test_certification_passes_on_consistent_tiny_instance() -> None:
    wd = _week()
    col = _assign_col()

    class _Oracle:
        def solve(self, **kwargs):
            return OracleResult(schedule=col, predicted_cost=10.0, realized_cost=10.0, status="OPTIMAL", solve_time=0.01)

    out = certify(
        w=np.array([0.0]),
        week_data_list=[wd],
        recommendation_model=_Rec(),
        oracle=_Oracle(),
        config=_cfg(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(),
        turnover=0.0,
        master_objective=10.0,
        master_bound=10.0,
        weekly_schedules={0: col},
    )
    assert out.status == "OPTIMAL_VERIFIED"


def test_certification_reports_failure_on_inconsistency() -> None:
    wd = _week()
    col = _assign_col()
    better = _defer_col()

    class _Oracle:
        def solve(self, **kwargs):
            return OracleResult(schedule=better, predicted_cost=0.0, realized_cost=0.0, status="OPTIMAL", solve_time=0.01)

    out = certify(
        w=np.array([0.0]),
        week_data_list=[wd],
        recommendation_model=_Rec(),
        oracle=_Oracle(),
        config=_cfg(),
        costs=CostConfig(deferral_per_case=0.0),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(),
        turnover=0.0,
        master_objective=10.0,
        master_bound=10.0,
        weekly_schedules={0: col},
    )
    assert out.status in {"TERMINATED_UNVERIFIED", "FAILED_CERTIFICATION"}
    assert out.max_violation > 0.0


def test_tie_break_mismatch_produces_warning_flag() -> None:
    wd = _week()
    col = _assign_col()

    class _Oracle:
        def solve(self, **kwargs):
            return OracleResult(schedule=col, predicted_cost=10.0, realized_cost=0.0, status="OPTIMAL", solve_time=0.01)

    out = certify(
        w=np.array([0.0]),
        week_data_list=[wd],
        recommendation_model=_Rec(),
        oracle=_Oracle(),
        config=_cfg(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(),
        turnover=0.0,
        master_objective=10.0,
        master_bound=10.0,
        weekly_schedules={0: col},
    )
    assert out.tie_break_flags is not None
    assert len(out.tie_break_flags) >= 1
