from datetime import date, datetime

import numpy as np
import pandas as pd

import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, Col, WeeklyInstance
from src.solvers.fixed_capacity import solve_fixed_capacity_assignment


def _case(case_id: int, duration: float = 100.0, site: str = "TGH") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        procedure_id="P",
        surgeon_code="S",
        service="Svc",
        patient_type="ELECTIVE",
        operating_room="OR1",
        booked_duration_min=duration,
        actual_duration_min=duration,
        actual_start=datetime(2012, 1, 2, 8, 0),
        week_of_year=1,
        month=1,
        year=2012,
        site=site,
    )


def test_final_config_is_two_site_fixed_capacity_primary_specification() -> None:
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    s.validate()
    cfg = final.final_build_config(s)
    assert cfg.scope.planning_sites == ("TGH", "TWH")
    assert cfg.capacity.activation_cost_per_block == 0.0
    assert cfg.capacity.turnover_minutes == 30.0
    assert cfg.capacity.eligibility_min_weeks == 3
    assert cfg.capacity.min_activation_rate == 0.25
    assert cfg.costs.overtime_per_minute == 15.0
    assert cfg.costs.idle_per_minute == 10.0
    assert s.expected_train_cases == 20519
    assert s.expected_holdout_cases == 6561
    assert abs(s.oracle_gap - 1e-3) <= 1e-12
    assert abs(s.oracle_numeric_tol - 1e-3) <= 1e-12


def test_pooled_feature_encoder_keeps_111_dimensions_and_site_signal() -> None:
    n = 160
    frame = pd.DataFrame(
        {
            Col.BOOKED_MINUTES: np.linspace(60.0, 240.0, n),
            Col.WEEK_OF_YEAR: (np.arange(n) % 52) + 1,
            Col.MONTH: (np.arange(n) % 12) + 1,
            Col.SITE: np.where(np.arange(n) % 2 == 0, "TGH", "TWH"),
            Col.CASE_SERVICE: [f"Svc{i % 20}" for i in range(n)],
            Col.SURGEON_CODE: [f"S{i % 80}" for i in range(n)],
            Col.PROCEDURE_ID: [f"P{i % 40}" for i in range(n)],
        }
    )
    enc = final.FinalFeatureEncoder().fit(frame)
    X = enc.transform_frame(frame)
    assert X.shape == (n, 111)
    assert len(enc.feature_names) == 111
    site_names = [name for name in enc.feature_names if name.startswith("site_")]
    assert len(site_names) == 1
    assert set(enc.selected_levels["site"]) == {"TGH", "TWH"}


def test_final_week_solver_uses_fixed_capacity_and_turnover() -> None:
    block = CandidateBlock(0, "TGH", "OR1", 200.0, 0.0, True)
    cases = [_case(1), _case(2)]
    inst = WeeklyInstance(
        week_index=0,
        start_date=date(2012, 1, 2),
        end_date=date(2012, 1, 8),
        cases=cases,
        calendar=BlockCalendar([block]),
        case_eligible_blocks={0: [block.id], 1: [block.id]},
    )
    week = base.WeekBundle(0, np.datetime64("2012-01-02"), inst)
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test", verbose=False)

    # A single-site unit instance is solved directly because the production
    # decomposer intentionally requires both primary sites to be present.
    cfg = final.SolverConfig(
        time_limit_seconds=30,
        mip_gap=0.0,
        threads=1,
        verbose=False,
        mip_gap_abs=1e-10,
        seed=s.random_seed,
    )
    result = solve_fixed_capacity_assignment(
        week.instance,
        np.array([100.0, 100.0]),
        final.final_cost_cfg(s),
        final.PRIMARY_TURNOVER,
        cfg,
        objective_mode="psi",
        symmetry_breaking=True,
    )
    assert result.column is not None
    assert result.column.z_defer == frozenset()
    assert result.column.v_open == frozenset({block.id})
    # Load = 100 + 100 + one 30-minute transition = 230; overtime = 30.
    assert abs(float(result.phi_ub) - 450.0) <= 1e-6


def test_two_site_decomposition_matches_monolithic_objective() -> None:
    tgh = CandidateBlock(0, "TGH", "OR1", 200.0, 0.0, True)
    twh = CandidateBlock(0, "TWH", "OR2", 200.0, 0.0, True)
    cases = [
        _case(1, 100.0, "TGH"),
        _case(2, 100.0, "TGH"),
        _case(3, 80.0, "TWH"),
        _case(4, 90.0, "TWH"),
    ]
    inst = WeeklyInstance(
        week_index=0,
        start_date=date(2012, 1, 2),
        end_date=date(2012, 1, 8),
        cases=cases,
        calendar=BlockCalendar([tgh, twh]),
        case_eligible_blocks={0: [tgh.id], 1: [tgh.id], 2: [twh.id], 3: [twh.id]},
    )
    week = base.WeekBundle(0, np.datetime64("2012-01-02"), inst)
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test", verbose=False)
    durations = np.array([100.0, 100.0, 80.0, 90.0])

    decomposed = final.final_solve_week(
        week,
        durations,
        s,
        time_limit=30,
        mip_gap=0.0,
        threads=1,
    )
    cfg = final.SolverConfig(
        time_limit_seconds=30,
        mip_gap=0.0,
        threads=1,
        verbose=False,
        mip_gap_abs=1e-10,
        seed=s.random_seed,
    )
    monolithic = solve_fixed_capacity_assignment(
        inst,
        durations,
        final.final_cost_cfg(s),
        final.PRIMARY_TURNOVER,
        cfg,
        objective_mode="psi",
        symmetry_breaking=True,
    )
    assert monolithic.column is not None
    assert abs(decomposed.objective - float(monolithic.phi_ub)) <= 1e-6
    assert abs(decomposed.bound - float(monolithic.phi_lb)) <= 1e-6
    assert all(bid.site == cases[i].site for (i, bid), v in decomposed.column.z_assign.items() if v > 0.5)


def test_recommendation_safety_keeps_short_case_positive() -> None:
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    a = base.Arrays(
        X=final.sparse.csr_matrix([[1.0], [1.0]]),
        booked=np.array([14.0, 120.0]),
        actual=np.array([14.0, 120.0]),
        error=np.array([0.0, 0.0]),
        week_ids=np.array([0, 0]),
        case_ids=np.array([1, 2]),
        week_slices={0: np.array([0, 1])},
    )
    # Raw policy would request -25 minutes for both cases. The 14-minute case
    # must be clipped before behavioral response so its planning duration stays
    # strictly positive.
    delta, _, planning = final.final_correction_and_planning(np.array([-25.0]), a, s)
    assert delta[0] >= final.MIN_RECOMMENDED_DURATION - 14.0 - 1e-12
    assert planning[0] > 0.0
    assert np.all(planning > 0.0)


def test_adapter_installs_two_site_planning_feature_and_safety_layers() -> None:
    final.install_final_adapter()
    assert base.Settings is final.FinalSettings
    assert base.FrozenFeatureEncoder is final.FinalFeatureEncoder
    assert base.solve_week is final.final_solve_week
    assert base.build_bundles is final.final_build_bundles
    assert base._cost_cfg is final.final_cost_cfg
    assert base.correction_and_planning is final.final_correction_and_planning
    assert base.train_naive is final.final_train_naive
    assert base.ConvexPDCASubproblem is final.FinalConvexPDCASubproblem
