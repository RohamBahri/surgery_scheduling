from datetime import date, datetime

import numpy as np

import final_paper_runtime_fixes as fixes
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, WeeklyInstance


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


def _dummy_column(block, n_cases: int = 1) -> ScheduleColumn:
    return ScheduleColumn(
        z_assign={(0, block.id): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({block.id}),
        y_used=frozenset({block.id}),
        n_cases=n_cases,
        block_capacities={block.id: float(block.capacity_minutes)},
        block_activation_costs={block.id: 0.0},
    )


def test_safe_crossfit_pi_uses_dtype_not_positional_shape() -> None:
    n = 20
    a = base.Arrays(
        X=final.sparse.csr_matrix(np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])),
        booked=np.full(n, 100.0),
        actual=np.full(n, 100.0),
        error=np.zeros(n),
        week_ids=np.repeat(np.arange(5), 4),
        case_ids=np.arange(n),
        week_slices={k: np.arange(4 * k, 4 * k + 4) for k in range(5)},
    )
    labels = np.tile(np.array([0, 1, 0, 1]), 5)
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    pred, metrics = fixes.safe_crossfit_pi(a, labels, s)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
    assert np.all((pred > 0) & (pred < 1))
    assert np.isfinite(metrics["brier"])


def test_safe_saturation_box_respects_short_case_duration_floor() -> None:
    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    booked = np.array([14.0, 120.0])
    rng = np.random.default_rng(42)

    lower = fixes.safe_saturation_correction(2, booked, rng, s)
    assert abs(lower[0] - (-0.8 * 13.0)) <= 1e-12
    assert abs(lower[1] - (-24.0)) <= 1e-12
    assert np.all(booked + lower > 0)

    for draw in range(20):
        corr = fixes.safe_saturation_correction(draw, booked, rng, s)
        assert np.all(booked + corr > 0)
        assert np.all(corr <= 24.0 + 1e-12)
        assert corr[0] >= -10.4 - 1e-12


def test_oracle_retry_merge_keeps_gap_on_native_psi_scale(monkeypatch) -> None:
    # Choose capacity so K = idle * (capacity - duration - tau*n) = 1,000,000.
    block = CandidateBlock(0, "TGH", "OR1", 100100.0, 0.0, True)
    inst = WeeklyInstance(
        week_index=0,
        start_date=date(2012, 1, 2),
        end_date=date(2012, 1, 8),
        cases=[_case(1, 100.0)],
        calendar=BlockCalendar([block]),
        case_eligible_blocks={0: [block.id]},
    )
    week = base.WeekBundle(0, np.datetime64("2012-01-02"), inst)
    column = _dummy_column(block)
    k = 1_000_000.0

    first = base.PlanResult(
        week=0,
        column=column,
        objective=k + 2000.0,
        bound=k + 1000.0,
        gap=0.5,
        status="TIME_LIMIT",
        solve_seconds=10.0,
        exact=False,
    )
    retry = base.PlanResult(
        week=0,
        column=column,
        objective=k + 1800.0,
        bound=k + 1000.0,
        gap=800.0 / 1800.0,
        status="TIME_LIMIT",
        solve_seconds=20.0,
        exact=False,
    )
    calls = iter([{0: first}, {0: retry}])
    monkeypatch.setattr(fixes, "safe_solve_batch", lambda *args, **kwargs: next(calls))

    s = final.FinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    out = fixes.safe_solve_oracle_batch(
        [week], {0: np.array([100.0])}, s, label="oracle_test"
    )[0]

    assert abs(out.objective - (k + 1800.0)) <= 1e-9
    assert abs(out.bound - (k + 1000.0)) <= 1e-9
    assert abs(out.gap - (800.0 / 1800.0)) <= 1e-12
    assert not out.exact


def test_runtime_fix_installation_overrides_only_hardening_hooks() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    assert base.crossfit_pi is fixes.safe_crossfit_pi
    assert base.solve_batch is fixes.safe_solve_batch
    assert base.solve_oracle_batch is fixes.safe_solve_oracle_batch
    assert base.saturation_test is fixes.safe_saturation_test
    assert base.FINAL_RUNTIME_FIXES_VERSION == fixes.RUNTIME_FIXES_VERSION
