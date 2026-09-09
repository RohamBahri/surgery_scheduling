"""Small mathematical regressions; no dataset-dependent test framework."""

from dataclasses import replace
from datetime import date, datetime
from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.config import Config, CostConfig, SolverConfig
from src.core.types import (
    BlockCalendar,
    CandidateBlock,
    CaseRecord,
    Col,
    WeeklyInstance,
)
from src.data.loader import load_data
from src.planning.eligibility import (
    CapacityModelError,
    ServiceRoomHistory,
    fit_service_room_history,
    resolve_eligibility,
)
from src.planning.instance import build_weekly_instance_with_calendar
from src.planning.retrospective import historical_baseline, turnover_audit
from src.planning.roster import build_fixed_roster
from src.solvers.deterministic import (
    solve_deterministic,
    solve_pricing,
    solve_pricing_detailed,
)
from src.solvers.fixed_capacity import (
    FixedCapacityResult,
    column_from_assignment,
    interchangeable_block_groups,
    schedule_metrics,
    solve_fixed_capacity_assignment,
)
from src.solvers.result import PricingResult, diagnostics_from_model

COSTS = CostConfig()
EXACT = SolverConfig(
    time_limit_seconds=10, mip_gap=0, mip_gap_abs=0, verbose=False, threads=1
)


def make_instance(durations=(100, 200), capacities=(480, 480)):
    blocks = [
        CandidateBlock(0, "TGH", f"OR{j}", cap, 0, True)
        for j, cap in enumerate(capacities)
    ]
    cases = [
        CaseRecord(
            i,
            "P",
            "S",
            "Svc",
            "Elective",
            "OR0",
            d,
            d,
            datetime(2024, 1, 1, 8),
            1,
            1,
            2024,
            site="TGH",
        )
        for i, d in enumerate(durations)
    ]
    return WeeklyInstance(
        0,
        date(2024, 1, 1),
        date(2024, 1, 7),
        cases,
        BlockCalendar(blocks),
        {i: [b.id for b in blocks] for i in range(len(cases))},
    )


def test_hand_computed_identity_unused_capacity_and_turnover():
    inst = make_instance()
    bid = inst.calendar.block_ids[0]
    col = column_from_assignment(inst, {0: bid, 1: bid})
    result = schedule_metrics(col, [100, 200], COSTS, 30)
    assert result["turnover_minutes"] == 30
    assert result["idle_minutes"] == 630
    assert result["phi"] == 6300
    assert result["K"] == 6000
    assert result["psi"] == 300
    assert result["identity_error"] == 0
    assert col.compute_cost(np.array([100, 200]), COSTS, 30) == 6300
    assert col.v_open == frozenset(inst.calendar.block_ids)
    assert col.y_used == {bid}
    assert col.z_defer == frozenset()
    assert set(col.block_activation_costs.values()) == {0}


def test_one_case_has_zero_turnover():
    inst = make_instance((100,))
    col = column_from_assignment(inst, {0: inst.calendar.block_ids[0]})
    assert schedule_metrics(col, [100], COSTS, 30)["turnover_minutes"] == 0


@pytest.mark.parametrize("tau", [0, 20, 30, 40])
@pytest.mark.parametrize("durations", [(55, 65, 25, 45), (101, 107, 70, 40)])
def test_phi_psi_and_symmetry_match_exhaustive_optimum(tau, durations):
    inst = make_instance(durations, (100, 100, 90))
    # The third block has different capacity/eligibility and cannot be exchanged.
    inst.case_eligible_blocks[0] = inst.calendar.block_ids[:2]
    bids = inst.calendar.block_ids
    scores = []
    for choices in product(
        *(inst.case_eligible_blocks[i] for i in range(inst.num_cases))
    ):
        total = 0
        for j, bid in enumerate(bids):
            ids = [i for i, b in enumerate(choices) if b == bid]
            load = sum(durations[i] for i in ids) + tau * max(len(ids) - 1, 0)
            total += 15 * max(load - inst.calendar.capacity(bid), 0) + 10 * max(
                inst.calendar.capacity(bid) - load, 0
            )
        scores.append(total)
        col = column_from_assignment(inst, dict(enumerate(choices)))
        assert schedule_metrics(col, durations, COSTS, tau)["phi"] == pytest.approx(
            total
        )
    optimum = min(scores)
    for mode, symmetry in product(("phi", "psi"), (False, True)):
        result = solve_fixed_capacity_assignment(
            inst, durations, COSTS, tau, EXACT, mode, symmetry_breaking=symmetry
        )
        assert result.diagnostics.status == "OPTIMAL"
        assert result.diagnostics.proven_optimal
        assert result.phi_ub == pytest.approx(optimum)
        assert result.phi_lb == pytest.approx(optimum)
        assert result.diagnostics.absolute_gap == pytest.approx(0, abs=1e-6)
        assert result.column.y_used == {b for (i, b) in result.column.z_assign}
        assert len(result.column.z_assign) == inst.num_cases


def test_different_eligibility_is_not_symmetric():
    inst = make_instance()
    assert len(interchangeable_block_groups(inst)) == 1
    inst.case_eligible_blocks[0] = inst.calendar.block_ids[:1]
    assert interchangeable_block_groups(inst) == []


def test_warm_start_is_relabelled_safely():
    inst = make_instance((100, 200, 60))
    bids = inst.calendar.block_ids
    warm = column_from_assignment(inst, {0: bids[1], 1: bids[0], 2: bids[1]})
    r = solve_fixed_capacity_assignment(
        inst, inst.booked_durations(), COSTS, 30, EXACT, warm_start=warm
    )
    assert r.diagnostics.proven_optimal
    assert (
        r.metrics["phi"]
        <= schedule_metrics(warm, inst.booked_durations(), COSTS, 30)["phi"]
    )


def test_empty_roster_site_and_no_deferral():
    inst = make_instance()
    with pytest.raises(ValueError, match="deferral"):
        column_from_assignment(inst, {0: inst.calendar.block_ids[0]})
    inst.case_eligible_blocks[1] = []
    with pytest.raises(CapacityModelError, match="no eligible"):
        solve_fixed_capacity_assignment(inst, [100, 200], COSTS, 30, EXACT)


def test_empty_week_keeps_fixed_idle_capacity():
    inst = make_instance(())
    for mode in ("phi", "psi"):
        r = solve_fixed_capacity_assignment(inst, [], COSTS, 30, EXACT, mode)
        assert r.phi_ub == pytest.approx(9600)
        assert r.column.y_used == frozenset()
        assert len(r.column.v_open) == 2


def test_explicit_fallback_never_crosses_site_and_day_stays_flexible():
    cal = BlockCalendar(
        [
            CandidateBlock(0, "TGH", "OR1", 480, 0, True),
            CandidateBlock(4, "TGH", "OR1", 480, 0, True),
            CandidateBlock(1, "TWH", "OR2", 480, 0, True),
        ]
    )
    hist = ServiceRoomHistory({("Svc", "TGH", "OR1"): 3}, {})
    primary = resolve_eligibility("Svc", "TGH", cal, hist, 3)
    assert primary.eligibility_tier == 1 and primary.number_primary_blocks == 2
    assert {bid.day_index for bid in primary.eligible_blocks} == {0, 4}
    fallback = resolve_eligibility("Svc", "TGH", cal, hist, 5)
    assert fallback.eligibility_tier == 2 and fallback.fallback_used
    assert fallback.number_final_blocks == 2 and fallback.number_primary_blocks == 0
    with pytest.raises(CapacityModelError, match="cross-site"):
        resolve_eligibility("Svc", "UNKNOWN_SITE", cal, hist)


def training_frame():
    starts = pd.to_datetime(
        ["2024-01-01 08:00", "2024-01-02 08:00", "2024-01-08 08:00", "2024-01-08 10:00"]
    )
    return pd.DataFrame(
        {
            Col.ACTUAL_START: starts,
            Col.CASE_SERVICE_RAW: ["Svc"] * 4,
            Col.CASE_SERVICE: ["Other"] * 4,
            Col.SITE: ["TGH"] * 4,
            Col.OPERATING_ROOM: ["OR1", "OR2", "OR1", "OR3"],
            Col.CASE_UID: [1, 2, 3, 4],
            Col.BOOKED_MINUTES: [60] * 4,
            Col.PROCEDURE_DURATION: [60] * 4,
        }
    )


def test_observed_roster_and_median_rounding_and_external_calendar():
    df, cfg = training_frame(), Config()
    observed = build_fixed_roster(
        df, pd.Timestamp("2024-01-01"), cfg, "observed_activity_proxy"
    )
    assert {(b.day_index, b.room) for b in observed.calendar.candidates} == {
        (0, "OR1"),
        (1, "OR2"),
    }
    assert all(
        b.is_fixed and b.activation_cost == 0 for b in observed.calendar.candidates
    )
    assert observed.provenance["training_period"]["n_weeks"] == 2
    median = build_fixed_roster(
        df, pd.Timestamp("2024-01-01"), cfg, "median_count_template"
    )
    assert len(median.calendar.blocks_on_day(0)) == 2  # median(1,2)=1.5 -> 2
    assert len(median.calendar.blocks_on_day(1)) == 1  # median(1,0)=0.5 -> 1
    hist = fit_service_room_history(df)
    assert ("Svc", "TGH", "OR1") in hist.weeks_by_pair
    inst = build_weekly_instance_with_calendar(
        df, pd.Timestamp("2024-01-01"), 0, cfg, observed.calendar, hist, 1
    )
    base = historical_baseline(inst, COSTS, 30)
    assert base["historical_assignment_coverage"] == 1
    missing_cal = BlockCalendar(observed.calendar.candidates[:1])
    missing = build_weekly_instance_with_calendar(
        df, pd.Timestamp("2024-01-01"), 0, cfg, missing_cal, hist, 1
    )
    base = historical_baseline(missing, COSTS, 30)
    assert base["historical_assignment_coverage"] == 0.5
    assert base["realized_cost"] is None
    assert len(base["missing_historical_blocks"]) == 1
    with pytest.raises(ValueError, match="training week"):
        build_fixed_roster(
            df, pd.Timestamp("2024-02-05"), cfg, "observed_activity_proxy"
        )


def test_observed_weekday_is_not_an_assignment_constraint():
    df, cfg = training_frame(), Config()
    cal = BlockCalendar([CandidateBlock(4, "TGH", "OR1", 480, 0, True)])
    inst = build_weekly_instance_with_calendar(
        df, pd.Timestamp("2024-01-01"), 0, cfg, cal, fit_service_room_history(df), 1
    )
    assert all(bids == cal.block_ids for bids in inst.case_eligible_blocks.values())
    result = solve_fixed_capacity_assignment(
        inst, inst.booked_durations(), COSTS, 30, EXACT
    )
    assert all(bid.day_index == 4 for i, bid in result.column.z_assign)


@pytest.mark.parametrize("sol_count", [0, 1])
def test_time_limit_snapshot_preserves_bound_and_shift(sol_count):
    snapshot = SimpleNamespace(
        Status=9, ObjVal=120.0, ObjBound=75.0, Runtime=4.5, SolCount=sol_count
    )
    diag = diagnostics_from_model(snapshot)
    assert diag.status == "TIME_LIMIT" and not diag.proven_optimal
    assert diag.obj_bound == 75 and diag.runtime_seconds == 4.5
    assert diag.absolute_gap == (45 if sol_count else float("inf"))
    r = FixedCapacityResult(None, diag, "psi", -50.0, None, 0)
    assert r.phi_lb == 25.0
    assert r.phi_ub == (70.0 if sol_count else None)


def test_legacy_reporting_and_public_tuple_api(monkeypatch):
    inst = make_instance()
    kwargs = dict(
        n_cases=2,
        durations=np.array([100, 200]),
        calendar=inst.calendar,
        costs=COSTS,
        solver_cfg=EXACT,
        case_eligible_blocks=inst.case_eligible_blocks,
        turnover=30,
    )
    detailed = solve_pricing_detailed(**kwargs)
    column, value = solve_pricing(**kwargs)
    assert column is not None and value == detailed.diagnostics.obj_val
    time_limited = replace(
        detailed.diagnostics,
        status="TIME_LIMIT",
        status_code=9,
        proven_optimal=False,
        obj_bound=value - 10,
        absolute_gap=10,
        runtime_seconds=7,
    )
    monkeypatch.setattr(
        "src.solvers.deterministic.solve_pricing_detailed",
        lambda **kw: PricingResult(column, time_limited),
    )
    kwargs.pop("n_cases")
    result = solve_deterministic(cases=inst.cases, **kwargs)
    assert result.solver_status == "TIME_LIMIT"
    assert result.solve_time_seconds == 7
    assert result.objective_value == value
    assert result.diagnostics["obj_bound"] == value - 10
    assert not result.diagnostics["proven_optimal"]
    assert column.to_schedule_result(inst.cases).solver_status == "FEASIBLE"


def test_raw_labels_preserved_before_rare_recoding(monkeypatch):
    raw = pd.DataFrame(
        {
            "Patient_Type": ["ELECTIVE"],
            "Operating_Room": ["OR 1"],
            "Site": ["TGH"],
            "Booked Time (Minutes)": [90],
            "Case_Service": [" Rare service "],
            "Surgeon_Code": [123.0],
            "Main_Procedure_Id": [456.0],
        }
    )
    for prefix, time in [
        ("Enter Room", "08:00:00"),
        ("Actual Start", "08:10:00"),
        ("Actual Stop", "09:00:00"),
        ("Leave Room", "09:10:00"),
    ]:
        raw[f"{prefix} Date"] = pd.to_datetime(["2024-01-01"])
        raw[f"{prefix} Time"] = [time]
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: raw.copy())
    df = load_data(Config())
    assert df[Col.CASE_SERVICE_RAW].iloc[0] == "Rare service"
    assert df[Col.SURGEON_CODE_RAW].iloc[0] == "123"
    assert df[Col.PROCEDURE_ID_RAW].iloc[0] == "456"
    assert df[Col.CASE_SERVICE].iloc[0] == "Other"
    assert df[Col.SURGEON_CODE].iloc[0] == "Other"
    assert df[Col.PROCEDURE_ID].iloc[0] == "Other"


def test_turnover_uses_leave_then_enter_with_site_day_boundaries():
    df = training_frame()
    df[Col.OPERATING_ROOM] = "OR1"
    df[Col.ENTER_ROOM] = df[Col.ACTUAL_START]
    df[Col.LEAVE_ROOM] = df[Col.ENTER_ROOM] + pd.Timedelta(minutes=90)
    table, summary = turnover_audit(df)
    assert len(table) == 1
    assert table.gap_minutes.iloc[0] == 30
    assert summary["sample_size"] == 1 and summary["median"] == 30


def test_real_time_limit_without_incumbent_preserves_termination():
    inst = make_instance()
    cfg = replace(EXACT, time_limit_seconds=0)
    result = solve_fixed_capacity_assignment(inst, [100, 200], COSTS, 30, cfg)
    assert result.diagnostics.status == "TIME_LIMIT"
    assert result.diagnostics.sol_count == 0
    assert result.column is None and result.phi_ub is None
    assert result.diagnostics.obj_bound is not None
    assert not result.diagnostics.proven_optimal
    assert result.diagnostics.absolute_gap == float("inf")


def test_audit_pair_and_calibration_with_real_small_solves():
    from scripts.run_weekly_planner_audit import calibration_row, run_solve, scale_pair

    inst = make_instance()
    rows, errors = [], []
    key = {"week_start": "2024-01-01", "roster": "toy", "durations": "booked"}
    phi, phi_row = run_solve(
        inst, [100, 200], Config(), EXACT, "phi", 30, key, rows, errors
    )
    psi, psi_row = run_solve(
        inst, [100, 200], Config(), EXACT, "psi", 30, key, rows, errors
    )
    scale = scale_pair(key, phi_row, psi_row)
    assert not errors
    assert scale["phi_obj_val"] == pytest.approx(scale["K_plus_psi"])
    assert scale["identity_error"] < 1e-6
    baseline = historical_baseline(inst, COSTS, 30)
    calibration = calibration_row(
        key, inst, baseline, {"booked": [phi, psi], "realized": [phi, psi]}, COSTS
    )
    assert calibration["C_H"] == 6300
    assert calibration["C_O"] <= calibration["C_H"]
    assert calibration["O_planning_absolute_uncertainty"] < 1e-6


def test_no_hidden_hard_overtime_limit():
    inst = make_instance((800,), (100,))
    result = solve_fixed_capacity_assignment(inst, [800], COSTS, 30, EXACT)
    assert result.metrics["overtime_minutes"] == 700
    assert result.phi_ub == pytest.approx(10500)
