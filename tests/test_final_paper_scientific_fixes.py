from datetime import date, datetime

import numpy as np
import pandas as pd

import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.types import BlockCalendar, CandidateBlock, CaseRecord, Col, WeeklyInstance


def _case(case_id: int, duration: float, site: str, room: str = "OR1") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        procedure_id="P",
        surgeon_code="S",
        service="Svc",
        patient_type="ELECTIVE",
        operating_room=room,
        booked_duration_min=duration,
        actual_duration_min=duration,
        actual_start=datetime(2012, 1, 2, 8, 0),
        week_of_year=1,
        month=1,
        year=2012,
        site=site,
    )


def _column(inst, assignment):
    blocks = frozenset(inst.calendar.block_ids)
    used = frozenset(assignment.values())
    return ScheduleColumn(
        z_assign={(i, bid): 1.0 for i, bid in assignment.items()},
        z_defer=frozenset(),
        v_open=blocks,
        y_used=used,
        n_cases=inst.num_cases,
        block_capacities={b.id: float(b.capacity_minutes) for b in inst.calendar.candidates},
        block_activation_costs={b.id: 0.0 for b in inst.calendar.candidates},
    )


def test_scientific_settings_freeze_new_counts_capacity_and_box() -> None:
    final.install_final_adapter()
    science.apply_scientific_fixes()
    s = science.ScientificFinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    s.validate()
    assert s.expected_train_cases == 21033
    assert s.expected_holdout_cases == 6713
    assert s.coefficient_bound == 100.0
    assert science.PRIMARY_ROSTER == "median_count_template"
    assert science.EXPECTED_FEATURES == 107


def test_encoder_uses_no_calendar_information() -> None:
    n = 160
    frame = pd.DataFrame(
        {
            Col.BOOKED_MINUTES: np.linspace(60.0, 240.0, n),
            Col.SITE: np.where(np.arange(n) % 2 == 0, "TGH", "TWH"),
            Col.CASE_SERVICE: [f"Svc{i % 20}" for i in range(n)],
            Col.SURGEON_CODE: [f"S{i % 80}" for i in range(n)],
            Col.PROCEDURE_ID: [f"P{i % 40}" for i in range(n)],
        }
    )
    enc = science.ScientificFeatureEncoder().fit(frame)
    X = enc.transform_frame(frame)
    assert X.shape == (n, 107)
    assert not any("week" in name or "month" in name for name in enc.feature_names)

    # Calendar changes alone cannot change the encoded row.
    a = _case(1, 100.0, "TGH")
    b = _case(2, 100.0, "TGH")
    b.actual_start = datetime(2012, 7, 31, 23, 0)
    b.week_of_year = 31
    b.month = 7
    xa = enc.transform_cases([a]).toarray()
    xb = enc.transform_cases([b]).toarray()
    assert np.allclose(xa, xb)


def test_predecision_order_ignores_actual_start() -> None:
    frame = pd.DataFrame(
        {
            Col.PATIENT_ID: [2, 1],
            Col.SITE: ["TGH", "TGH"],
            Col.CASE_SERVICE_RAW: ["Svc", "Svc"],
            Col.SURGEON_CODE_RAW: ["S", "S"],
            Col.PROCEDURE_ID_RAW: ["P", "P"],
            Col.BOOKED_MINUTES: [100.0, 100.0],
            Col.ACTUAL_START: [pd.Timestamp("2012-01-02 08:00"), pd.Timestamp("2012-01-06 20:00")],
        }
    )
    ordered = science._predecision_ordered_pool(frame)
    assert ordered[Col.PATIENT_ID].tolist() == [1, 2]


def test_decomposed_library_uses_implicit_cartesian_product() -> None:
    final.install_final_adapter()
    science.apply_scientific_fixes()

    blocks = [
        CandidateBlock(0, "TGH", "A", 100.0, 0.0, True),
        CandidateBlock(0, "TGH", "B", 200.0, 0.0, True),
        CandidateBlock(0, "TWH", "C", 100.0, 0.0, True),
        CandidateBlock(0, "TWH", "D", 200.0, 0.0, True),
    ]
    cases = [
        _case(1, 90.0, "TGH"),
        _case(2, 190.0, "TGH"),
        _case(3, 90.0, "TWH"),
        _case(4, 190.0, "TWH"),
    ]
    inst = WeeklyInstance(
        week_index=0,
        start_date=date(2012, 1, 2),
        end_date=date(2012, 1, 8),
        cases=cases,
        calendar=BlockCalendar(blocks),
        case_eligible_blocks={
            0: [blocks[0].id, blocks[1].id],
            1: [blocks[0].id, blocks[1].id],
            2: [blocks[2].id, blocks[3].id],
            3: [blocks[2].id, blocks[3].id],
        },
    )
    week = base.WeekBundle(0, pd.Timestamp("2012-01-02"), inst)

    # A = TGH good / TWH bad; B = TGH bad / TWH good.
    col_a = _column(
        inst,
        {0: blocks[0].id, 1: blocks[1].id, 2: blocks[3].id, 3: blocks[2].id},
    )
    col_b = _column(
        inst,
        {0: blocks[1].id, 1: blocks[0].id, 2: blocks[2].id, 3: blocks[3].id},
    )
    lib = science.DecomposedScheduleLibrary({0: week})
    assert lib.add(0, col_a, "A")
    assert lib.add(0, col_b, "B")

    s = science.ScientificFinalSettings(data="dummy.xlsx", artifact_root="artifacts/test")
    durations = np.array([90.0, 190.0, 90.0, 190.0])
    best, value, source = lib.best(0, durations, s)
    direct = best.compute_cost(durations, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
    assert abs(value - direct) <= 1e-9
    assert "TGH:A" in source
    assert "TWH:B" in source
    # The independent combination must be strictly better than either pooled input.
    cost_a = col_a.compute_cost(durations, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
    cost_b = col_b.compute_cost(durations, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
    assert value < min(cost_a, cost_b)


def test_apply_scientific_fixes_installs_final_hooks() -> None:
    final.install_final_adapter()
    science.apply_scientific_fixes()
    assert base.Settings is science.ScientificFinalSettings
    assert base.FrozenFeatureEncoder is science.ScientificFeatureEncoder
    assert base.ScheduleLibrary is science.DecomposedScheduleLibrary
    assert base.FixedSpec is science.ScientificFixedSpec
    assert base.train_naive is science.scientific_train_naive
    assert base.library_metrics is science.scientific_library_metrics
