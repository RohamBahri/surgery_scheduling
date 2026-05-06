from __future__ import annotations

import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.core.types import BlockCalendar, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData
from src.vfcg.master import NativeMasterResult
from src.vfcg.types import CertificationResult, OracleResult
from src.vfcg.solver import vfcg_solve


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


def _col() -> ScheduleColumn:
    b = CandidateBlock(0, "TGH", "OR1", 100.0, 10.0).id
    return ScheduleColumn(
        z_assign={(0, b): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b}),
        y_used=frozenset({b}),
        n_cases=1,
        block_capacities={b: 100.0},
        block_activation_costs={b: 10.0},
    )


def _cfg() -> Config:
    cfg = Config()
    cfg.vfcg.max_iterations = 5
    cfg.vfcg.convergence_tol = 1e-6
    return cfg


def test_trivial_instance_terminates_one_iteration(monkeypatch) -> None:
    wd = _week()
    col = _col()

    monkeypatch.setattr("src.vfcg.solver.generate_warmstart_references", lambda **kwargs: {0: [col]})
    monkeypatch.setattr(
        "src.vfcg.solver.solve_native_master",
        lambda **kwargs: NativeMasterResult(
            weights=np.array([0.0]),
            schedules_by_week={0: col},
            objective=10.0,
            bound=10.0,
            gap=0.0,
            solve_time=0.1,
            status="OPTIMAL",
            predicted_costs_by_week={0: 10.0},
            is_fallback=False,
        ),
    )

    class _Oracle:
        def solve(self, **kwargs):
            return OracleResult(schedule=col, predicted_cost=10.0, realized_cost=10.0, status="OPTIMAL", solve_time=0.01)

    monkeypatch.setattr("src.vfcg.solver.ExactFollowerOracle", _Oracle)
    monkeypatch.setattr(
        "src.vfcg.solver.certify",
        lambda **kwargs: CertificationResult("OPTIMAL_VERIFIED", 0.0, 10.0, 10.0, 10.0, None),
    )

    res = vfcg_solve([wd], _Rec(), _cfg(), CostConfig(), CapacityConfig(), SolverConfig(), 0.0)
    assert res.n_iterations == 1
    assert res.total_cuts_added == 0


def test_detects_violated_week_and_adds_cut(monkeypatch) -> None:
    wd = _week()
    col = _col()
    calls = {"n": 0, "sizes": []}

    monkeypatch.setattr("src.vfcg.solver.generate_warmstart_references", lambda **kwargs: {0: [col]})

    def _master(**kwargs):
        calls["n"] += 1
        calls["sizes"].append(len(kwargs["reference_sets"][0]))
        return NativeMasterResult(
            weights=np.array([0.0]),
            schedules_by_week={0: col},
            objective=10.0,
            bound=10.0,
            gap=0.0,
            solve_time=0.1,
            status="OPTIMAL",
            predicted_costs_by_week={0: 10.0},
            is_fallback=False,
        )

    monkeypatch.setattr("src.vfcg.solver.solve_native_master", _master)

    better_col = ScheduleColumn(
        z_assign={},
        z_defer=frozenset({0}),
        v_open=frozenset(),
        y_used=frozenset(),
        n_cases=1,
        block_capacities=col.block_capacities,
        block_activation_costs=col.block_activation_costs,
    )

    class _Oracle:
        def __init__(self):
            self.k = 0

        def solve(self, **kwargs):
            self.k += 1
            if self.k == 1:
                return OracleResult(schedule=better_col, predicted_cost=0.0, realized_cost=0.0, status="OPTIMAL", solve_time=0.01)
            return OracleResult(schedule=better_col, predicted_cost=10.0, realized_cost=0.0, status="OPTIMAL", solve_time=0.01)

    monkeypatch.setattr("src.vfcg.solver.ExactFollowerOracle", _Oracle)
    monkeypatch.setattr(
        "src.vfcg.solver.certify",
        lambda **kwargs: CertificationResult("TERMINATED_UNVERIFIED", 0.0, 10.0, 10.0, 10.0, None),
    )

    res = vfcg_solve([wd], _Rec(), _cfg(), CostConfig(deferral_per_case=0.0), CapacityConfig(), SolverConfig(), 0.0)
    assert res.total_cuts_added >= 1
    assert res.n_iterations == 2
    assert calls["sizes"][0] == 1
    assert calls["sizes"][1] >= 2


def test_fallback_path_goes_directly_to_certification(monkeypatch) -> None:
    wd = _week()
    col = _col()

    monkeypatch.setattr("src.vfcg.solver.generate_warmstart_references", lambda **kwargs: {0: [col]})
    monkeypatch.setattr(
        "src.vfcg.solver.solve_native_master",
        lambda **kwargs: NativeMasterResult(
            weights=np.array([0.0]),
            schedules_by_week={0: col},
            objective=float("inf"),
            bound=float("inf"),
            gap=float("inf"),
            solve_time=0.1,
            status="TIME_LIMIT_FALLBACK",
            predicted_costs_by_week={0: 10.0},
            is_fallback=True,
        ),
    )
    monkeypatch.setattr("src.vfcg.solver.ExactFollowerOracle", lambda: object())

    marker = {"called": 0}

    def _cert(**kwargs):
        marker["called"] += 1
        return CertificationResult("TERMINATED_UNVERIFIED", 0.0, 0.0, 0.0, 0.0, None)

    monkeypatch.setattr("src.vfcg.solver.certify", _cert)

    res = vfcg_solve([wd], _Rec(), _cfg(), CostConfig(), CapacityConfig(), SolverConfig(), 0.0)
    assert marker["called"] == 1
    assert res.n_iterations == 1


def test_vfcg_solver_downgrades_fallback_certification_status(monkeypatch) -> None:
    wd = _week()
    col = _col()

    monkeypatch.setattr("src.vfcg.solver.generate_warmstart_references", lambda **kwargs: {0: [col]})
    monkeypatch.setattr(
        "src.vfcg.solver.solve_native_master",
        lambda **kwargs: NativeMasterResult(
            weights=np.array([0.0]),
            schedules_by_week={0: col},
            objective=10.0,
            bound=10.0,
            gap=float("nan"),
            solve_time=0.1,
            status="TIME_LIMIT_FALLBACK",
            predicted_costs_by_week={0: 10.0},
            is_fallback=True,
        ),
    )
    monkeypatch.setattr("src.vfcg.solver.ExactFollowerOracle", lambda: object())
    monkeypatch.setattr(
        "src.vfcg.solver.certify",
        lambda **kwargs: CertificationResult("OPTIMAL_VERIFIED", 0.0, 10.0, 10.0, 10.0, None),
    )

    res = vfcg_solve([wd], _Rec(), _cfg(), CostConfig(), CapacityConfig(), SolverConfig(), 0.0)

    assert res.certification.status == "TERMINATED_UNVERIFIED"
    assert "master_fallback_used" in (res.certification.tie_break_flags or [])
