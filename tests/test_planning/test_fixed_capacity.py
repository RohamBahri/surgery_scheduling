"""Small mathematical regressions; no dataset-dependent test framework."""

from dataclasses import replace
from datetime import date, datetime
from itertools import permutations, product
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
from src.planning.retrospective import (
    historical_baseline,
    reassignment_metrics,
    turnover_audit,
)
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


def make_instance(durations=(100, 200), capacities=(480, 480), days=None):
    blocks = [
        CandidateBlock(days[j] if days else 0, "TGH", f"OR{j}", cap, 0, True)
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
    inst = make_instance(durations, (100, 100, 90), days=(0, 4, 0))
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
    for mode, symmetry in product(("phi", "psi_shifted", "psi"), (False, True)):
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
    inst = make_instance(days=(0, 4))
    assert len(interchangeable_block_groups(inst)) == 1
    inst.case_eligible_blocks[0] = inst.calendar.block_ids[:1]
    assert interchangeable_block_groups(inst) == []


def test_fixed_day_eligibility_prevents_cross_day_symmetry():
    inst = make_instance(days=(0, 4))
    bids = inst.calendar.block_ids
    inst.case_eligible_blocks = {0: [bids[0]], 1: [bids[1]]}
    assert interchangeable_block_groups(inst) == []
    result = solve_fixed_capacity_assignment(inst, [100, 200], COSTS, 30, EXACT)
    assert set(result.column.z_assign) == {(0, bids[0]), (1, bids[1])}


def test_warm_start_is_relabelled_safely():
    inst = make_instance((100, 200, 60), days=(0, 4))
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
    for mode in ("phi", "psi_shifted", "psi"):
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
        df,
        pd.Timestamp("2024-01-01"),
        cfg,
        "observed_activity_proxy",
        allow_retrospective=True,
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
            df,
            pd.Timestamp("2024-02-05"),
            cfg,
            "observed_activity_proxy",
            allow_retrospective=True,
        )
    with pytest.raises(ValueError, match="allow_retrospective=True"):
        build_fixed_roster(
            df, pd.Timestamp("2024-01-01"), cfg, "observed_activity_proxy"
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
    from scripts.run_weekly_planner_audit import (
        calibration_row,
        run_solve,
        scale_comparison,
    )

    inst = make_instance()
    rows, errors = [], []
    key = {"week_start": "2024-01-01", "roster": "toy", "durations": "booked"}
    phi, phi_row = run_solve(
        inst, [100, 200], Config(), EXACT, "phi", 30, key, rows, errors
    )
    psi, psi_row = run_solve(
        inst, [100, 200], Config(), EXACT, "psi", 30, key, rows, errors
    )
    shifted, shifted_row = run_solve(
        inst, [100, 200], Config(), EXACT, "psi_shifted", 30, key, rows, errors
    )
    scale = scale_comparison(
        key, {"phi": phi_row, "psi_shifted": shifted_row, "psi": psi_row}
    )
    assert not errors
    assert scale["phi_obj_val"] == pytest.approx(scale["K_plus_psi"])
    assert scale["identity_error"] < 1e-6
    assert scale["psi_shifted_psi_ub"] == pytest.approx(scale["psi_obj_val"])
    assert scale["psi_shifted_obj_val"] == pytest.approx(scale["phi_obj_val"])
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


@pytest.mark.parametrize("durations", [(100, 200), (900, 1000)])
def test_reduced_models_differ_only_in_objective_constant(monkeypatch, durations):
    import gurobipy as gp

    snapshots = []
    optimize = gp.Model.optimize

    def inspect_then_optimize(model, *args, **kwargs):
        model.update()
        variables, constraints = model.getVars(), model.getConstrs()
        snapshots.append(
            {
                "A": model.getA().toarray(),
                "variables": [
                    (v.VarName, v.VType, v.LB, v.UB, v.Obj) for v in variables
                ],
                "constraints": [(c.ConstrName, c.Sense, c.RHS) for c in constraints],
                "constant": model.ObjCon,
                "params": [
                    getattr(model.Params, p)
                    for p in ("TimeLimit", "MIPGap", "MIPGapAbs", "Threads", "Seed")
                ],
            }
        )
        return optimize(model, *args, **kwargs)

    monkeypatch.setattr(gp.Model, "optimize", inspect_then_optimize)
    inst = make_instance(durations, days=(0, 4))
    results = [
        solve_fixed_capacity_assignment(
            inst, durations, COSTS, 30, replace(EXACT, seed=17), mode
        )
        for mode in ("psi", "psi_shifted")
    ]
    a, b = snapshots
    np.testing.assert_array_equal(a["A"], b["A"])
    for key in ("variables", "constraints", "params"):
        assert a[key] == b[key]
    assert a["params"] == [10, 0, 0, 1, 17]
    assert b["constant"] - a["constant"] == results[0].K
    assert results[0].phi_ub == pytest.approx(results[1].phi_ub)
    assert results[0].psi_lb == pytest.approx(results[1].psi_lb)


@pytest.mark.parametrize("mode", ["phi", "psi_shifted", "psi"])
@pytest.mark.parametrize("k", [-500, 0, 6000])
@pytest.mark.parametrize("sol_count", [0, 1])
def test_bound_translation_for_all_modes_and_constant_signs(mode, k, sol_count):
    offset = 0 if mode == "psi" else k
    diag = diagnostics_from_model(
        SimpleNamespace(
            Status=9,
            ObjVal=200 + offset,
            ObjBound=150 + offset,
            Runtime=1,
            SolCount=sol_count,
        )
    )
    result = FixedCapacityResult(None, diag, mode, k, None, 0)
    assert result.psi_lb == 150 and result.phi_lb == 150 + k
    assert result.psi_ub == (200 if sol_count else None)
    assert result.phi_ub == (200 + k if sol_count else None)
    assert diag.absolute_gap == (50 if sol_count else float("inf"))


def test_reassignment_invariant_to_block_labels_and_sensitive_to_partition():
    inst = make_instance((100, 100, 100, 100), days=(0, 4))
    a, b = inst.calendar.block_ids
    historical = {0: a, 1: a, 2: b, 3: b}
    swapped = {0: b, 1: b, 2: a, 3: a}
    assert reassignment_metrics(inst, swapped, historical) == {
        "fraction_reassigned_raw": 1,
        "fraction_reassigned_modulo_symmetry": 0,
    }
    split = {0: a, 1: b, 2: a, 3: b}
    assert (
        reassignment_metrics(inst, split, historical)[
            "fraction_reassigned_modulo_symmetry"
        ]
        == 0.5
    )
    # Different eligibility blocks cannot be relabelled, even across weekdays.
    inst.case_eligible_blocks[0] = [b]
    assert (
        reassignment_metrics(inst, swapped, historical)[
            "fraction_reassigned_modulo_symmetry"
        ]
        == 1
    )


def test_reassignment_matches_exhaustive_bijections_with_missing_history():
    inst = make_instance((10, 20, 30), (100, 100, 100), days=(0, 1, 4))
    bids = inst.calendar.block_ids
    missing = bids[0]._replace(room="absent")
    historical = {0: bids[2], 1: bids[2], 2: missing}
    for choices in product(bids, repeat=3):
        assignment = dict(enumerate(choices))
        best = (
            min(
                sum(
                    dict(zip(bids, p))[assignment[i]] != historical[i]
                    for i in historical
                )
                for p in permutations(bids)
            )
            / 3
        )
        assert reassignment_metrics(inst, assignment, historical)[
            "fraction_reassigned_modulo_symmetry"
        ] == pytest.approx(best)


def test_crossfit_excludes_evaluated_week_without_changing_final_rule():
    from scripts.run_weekly_planner_audit import (
        eligibility_crossfit_rows,
        summarize_crossfit,
    )
    from src.planning.roster import week_starts

    df = training_frame()
    extra = df.iloc[[0]].copy()
    extra[Col.ACTUAL_START] += pd.Timedelta(days=14)
    extra[Col.CASE_UID] = 5
    df = pd.concat([df, extra], ignore_index=True)
    starts = sorted(week_starts(df).unique())
    history = fit_service_room_history(df)
    original = dict(history.weeks_by_pair)
    rows = eligibility_crossfit_rows(
        df, [pd.Timestamp(w) for w in starts], Config(), history
    )
    assert len(rows) == 3 * 3 * 4
    assert history.weeks_by_pair == original
    for row in rows:
        assert row["in_sample_training_weeks"] == 3
        assert row["loo_training_weeks"] == 2
        assert not row["roster_refitted"]
        if row["threshold_weeks"] == 3:
            assert row["in_sample_primary_coverage"] == 1
            assert row["loo_primary_coverage"] == 0
            assert row["delta_fallback_rate"] == 1
    for row in summarize_crossfit(rows):
        assert row["n_weeks"] == 3 and row["n_cases"] == 5
        if row["threshold_weeks"] == 3:
            assert row["delta_fallback_rate"] == 1


def test_proxy_comparison_uses_paired_intervals_and_reports_missing_solves():
    from scripts.run_weekly_planner_audit import summarize_proxy_comparison

    rows = [
        {
            "week_start": "w1",
            "roster": "regular_template",
            "durations": "booked",
            "K": 10000,
            "psi_psi_ub": 120,
            "psi_psi_lb": 100,
        },
        {
            "week_start": "w1",
            "roster": "observed_activity_proxy",
            "durations": "booked",
            "K": 5000,
            "psi_psi_ub": 110,
            "psi_psi_lb": 90,
        },
    ]
    row = summarize_proxy_comparison(rows)[0]
    assert row["status"] == "AVAILABLE"
    assert row["mean_K_difference_a_minus_b"] == 5000
    assert row["mean_psi_ub_difference_a_minus_b"] == 10
    assert row["mean_optimal_psi_difference_lb"] == -10
    assert row["mean_optimal_psi_difference_ub"] == 30
    rows.append({**rows[0], "week_start": "w2", "psi_psi_ub": None})
    rows.append({**rows[1], "week_start": "w2"})
    row = summarize_proxy_comparison(rows)[0]
    assert row["n_common_weeks"] == 2 and row["n_paired_finite_intervals"] == 1
    assert row["status"] == "UNAVAILABLE"
    assert row["mean_psi_ub_difference_a_minus_b"] is None
