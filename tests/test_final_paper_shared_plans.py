from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import final_paper_release_guard as release
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import final_paper_shared_plans as shared
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import column_from_assignment
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, WeeklyInstance


def _case(case_id: int, site: str, duration: float) -> CaseRecord:
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


def _week() -> base.WeekBundle:
    tgh = CandidateBlock(0, "TGH", "OR1", 480.0, 0.0, True)
    twh = CandidateBlock(0, "TWH", "OR2", 480.0, 0.0, True)
    cases = [_case(101, "TGH", 90.0), _case(102, "TWH", 110.0)]
    inst = WeeklyInstance(
        week_index=0,
        start_date=date(2012, 1, 2),
        end_date=date(2012, 1, 8),
        cases=cases,
        calendar=BlockCalendar([tgh, twh]),
        case_eligible_blocks={0: [tgh.id], 1: [twh.id]},
    )
    return base.WeekBundle(0, pd.Timestamp("2012-01-02"), inst)


def _settings(tmp_path):
    data = tmp_path / "data.xlsx"
    data.write_bytes(b"shared-plan-test-data")
    return science.ScientificFinalSettings(
        data=str(data),
        artifact_root=str(tmp_path / "run"),
        verbose=False,
    )


def test_flexible_behavior_validation_accepts_arbitrary_valid_scenario(tmp_path) -> None:
    shared.install_flexible_behavior_validation()
    s = _settings(tmp_path)
    s.alpha = 0.35
    s.h = 47.5
    s.validate()

    s.alpha = 1.0
    with pytest.raises(ValueError, match="alpha"):
        s.validate()

    s.alpha = 0.35
    s.h = 0.0
    with pytest.raises(ValueError, match="h"):
        s.validate()


def test_shared_plan_roundtrip_preserves_assignment_and_certificate(tmp_path) -> None:
    shared.install_flexible_behavior_validation()
    week = _week()
    weeks = [week]
    s = _settings(tmp_path)
    s.validate()
    assignment = {
        0: week.instance.calendar.candidates[0].id,
        1: week.instance.calendar.candidates[1].id,
    }
    column = column_from_assignment(week.instance, assignment)
    d = np.asarray(week.instance.booked_durations(), dtype=float)
    obj = float(column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER))
    plan = base.PlanResult(
        week=0,
        column=column,
        objective=obj,
        bound=obj,
        gap=0.0,
        status="OPTIMAL",
        solve_seconds=1.25,
        exact=True,
        tiebreak_used=False,
    )
    root = tmp_path / "shared"
    shared.save_plan_set(root, "booked", {0: plan}, weeks, {0: d}, s)
    loaded = shared.load_plan_set(root, "booked", weeks, {0: d}, s)

    assert loaded[0].column == column
    assert loaded[0].objective == pytest.approx(obj)
    assert loaded[0].bound == pytest.approx(obj)
    assert loaded[0].gap == pytest.approx(0.0)
    assert loaded[0].exact


def test_shared_plan_rejects_changed_duration_cost_accounting(tmp_path) -> None:
    shared.install_flexible_behavior_validation()
    week = _week()
    weeks = [week]
    s = _settings(tmp_path)
    assignment = {
        0: week.instance.calendar.candidates[0].id,
        1: week.instance.calendar.candidates[1].id,
    }
    column = column_from_assignment(week.instance, assignment)
    d = np.asarray(week.instance.booked_durations(), dtype=float)
    obj = float(column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER))
    plan = base.PlanResult(0, column, obj, obj, 0.0, "OPTIMAL", 1.0, True, False)
    root = tmp_path / "shared"
    shared.save_plan_set(root, "booked", {0: plan}, weeks, {0: d}, s)

    bad = d.copy()
    bad[0] += 20.0
    with pytest.raises(AssertionError, match="decomposed Phi mismatch"):
        shared.load_plan_set(root, "booked", weeks, {0: bad}, s)


def test_log_file_path_is_label_week_site_and_duration_specific(tmp_path) -> None:
    s = SimpleNamespace(artifact_root=str(tmp_path))
    p1 = shared.weekly_log_file(
        s, week=4, site="TGH", label="oracle_train_retry", durations=np.array([1.0, 2.0])
    )
    p2 = shared.weekly_log_file(
        s, week=4, site="TWH", label="oracle_train_retry", durations=np.array([1.0, 2.0])
    )
    p3 = shared.weekly_log_file(
        s, week=4, site="TGH", label="booked_train", durations=np.array([1.0, 2.0])
    )
    assert p1 != p2 != p3
    assert "oracle_train_retry" in p1
    assert "week_004" in p1
    assert p1.endswith(".log")


def test_release_guard_selects_spawn_safe_logged_workers(monkeypatch) -> None:
    old_train = shared.numeric.training_process_task
    old_det = shared.numeric.deterministic_eval_worker
    old_sens = shared.numeric.sensitivity_worker
    try:
        shared.install_spawn_safe_weekly_logging()
        release.install_reviewed_guards()
        assert runtime._solve_process_task is shared.training_process_task_logged
        assert science.deterministic_eval_worker is shared.deterministic_eval_worker_logged
        assert shared.sensitivity._worker is shared.sensitivity_worker_logged
    finally:
        shared.numeric.training_process_task = old_train
        shared.numeric.deterministic_eval_worker = old_det
        shared.numeric.sensitivity_worker = old_sens
        shared.numeric.install_all_guards()
