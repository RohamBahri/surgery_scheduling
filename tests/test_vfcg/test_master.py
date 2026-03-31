from __future__ import annotations

import numpy as np

from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.core.types import BlockCalendar, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData
from src.vfcg.master import solve_native_master


class _DummyRecommendation:
    @property
    def feature_dim(self) -> int:
        return 1


def _week(case_eligible_blocks=None) -> WeekRecommendationData:
    b1 = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)
    b2 = CandidateBlock(0, "TGH", "OR2", 100.0, 10.0)
    if case_eligible_blocks is None:
        case_eligible_blocks = {0: [b1.id], 1: [b1.id, b2.id]}
    return WeekRecommendationData(
        week_index=0,
        n_cases=2,
        features=np.array([[1.0], [1.0]], dtype=float),
        bookings=np.array([90.0, 60.0]),
        realized=np.array([100.0, 70.0]),
        L_bounds=np.array([60.0, 40.0]),
        U_bounds=np.array([140.0, 120.0]),
        surgeon_codes=["S1", "S2"],
        sos2_data=[
            SOS2CaseData(0, 0, np.array([-20.0, 0.0, 20.0]), np.array([-10.0, 0.0, 10.0]), 90.0, 60.0, 140.0),
            SOS2CaseData(1, 0, np.array([-20.0, 0.0, 20.0]), np.array([-10.0, 0.0, 10.0]), 60.0, 40.0, 120.0),
        ],
        case_eligible_blocks=case_eligible_blocks,
        calendar=BlockCalendar([b1, b2]),
    )


def _config() -> Config:
    cfg = Config()
    cfg.vfcg.master_time_limit = 30
    cfg.vfcg.master_mip_gap = 0.0
    cfg.vfcg.w_max = 10.0
    return cfg


def test_empty_reference_sets_produce_solveable_relaxation() -> None:
    wd = _week()
    res = solve_native_master(
        week_data_list=[wd],
        reference_sets={},
        recommendation_model=_DummyRecommendation(),
        config=_config(),
        costs=CostConfig(overtime_per_minute=1.0, idle_per_minute=1.0, deferral_per_case=1000.0),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False),
        turnover=0.0,
    )

    assert res.status in {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"}
    assert np.isfinite(res.objective)
    assert 0 in res.schedules_by_week


def test_w_zero_is_feasible_when_bounds_fixed_to_zero() -> None:
    wd = _week()
    cfg = _config()
    cfg.vfcg.w_max = 0.0

    res = solve_native_master(
        week_data_list=[wd],
        reference_sets={},
        recommendation_model=_DummyRecommendation(),
        config=cfg,
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False),
        turnover=0.0,
    )

    assert np.allclose(res.weights, 0.0)


def test_extracted_schedules_are_valid() -> None:
    wd = _week()
    res = solve_native_master(
        week_data_list=[wd],
        reference_sets={},
        recommendation_model=_DummyRecommendation(),
        config=_config(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False),
        turnover=0.0,
    )

    col = res.schedules_by_week[0]
    for i in range(wd.n_cases):
        assigned = sum(1 for (ii, _), val in col.z_assign.items() if ii == i and val > 0.5)
        deferred = 1 if i in col.z_defer else 0
        assert assigned + deferred == 1


def test_eligible_pair_sparsity_respected_in_extraction() -> None:
    b1 = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0)
    b2 = CandidateBlock(0, "TGH", "OR2", 100.0, 10.0)
    wd = _week(case_eligible_blocks={0: [b1.id], 1: [b1.id]})
    res = solve_native_master(
        week_data_list=[wd],
        reference_sets={},
        recommendation_model=_DummyRecommendation(),
        config=_config(),
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        solver_cfg=SolverConfig(time_limit_seconds=30, mip_gap=0.0, verbose=False),
        turnover=0.0,
    )

    col = res.schedules_by_week[0]
    assert all(bid != b2.id for (_, bid) in col.z_assign.keys())
