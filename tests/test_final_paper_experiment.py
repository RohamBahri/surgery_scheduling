from datetime import date, datetime

import numpy as np

import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, WeeklyInstance


def _case(case_id: int, duration: float = 100.0) -> CaseRecord:
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
        site="TGH",
    )


def test_final_config_is_fixed_capacity_primary_specification() -> None:
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    s.validate()
    cfg = final.final_build_config(s)
    assert cfg.capacity.activation_cost_per_block == 0.0
    assert cfg.capacity.turnover_minutes == 30.0
    assert cfg.capacity.eligibility_min_weeks == 3
    assert cfg.capacity.min_activation_rate == 0.25
    assert cfg.costs.overtime_per_minute == 15.0
    assert cfg.costs.idle_per_minute == 10.0


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
    result = final.final_solve_week(
        week,
        np.array([100.0, 100.0]),
        s,
        time_limit=30,
        mip_gap=0.0,
        threads=1,
    )
    assert result.column.z_defer == frozenset()
    assert result.column.v_open == frozenset({block.id})
    # Load = 100 + 100 + one 30-minute transition = 230; overtime = 30.
    assert abs(result.objective - 450.0) <= 1e-6
    assert result.exact


def test_adapter_installs_final_planning_layer() -> None:
    final.install_final_adapter()
    assert base.Settings is final.FinalSettings
    assert base.solve_week is final.final_solve_week
    assert base.build_bundles is final.final_build_bundles
    assert base._cost_cfg is final.final_cost_cfg
    assert base.ConvexPDCASubproblem is final.FinalConvexPDCASubproblem
