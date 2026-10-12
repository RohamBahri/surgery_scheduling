"""Small exact tests; no workbook or unrestricted Gurobi license required."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from surgery.weekly_data import Week, build_weeks
from surgery.weekly_planner import evaluate, solve_adversary, solve_week


def toy():
    # Surgeon A: two cases Monday, a supported extra session on Wednesday.
    # Surgeon B: one Wednesday case in the same physical room-day.
    return Week("TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
                np.array([2, 3, 4]), ("A", "B"), np.array([0, 0, 1]),
                np.array([0, 0, 1]), ((0, 1), (1,)), ((0,), (2,)),
                np.array([400., 200., 200.]), np.array([200., 400., 200.]),
                (("A", 2, "OR2", "prior_dates"),)).validate()


def feasible(week):
    for candidate in product(*(week.allowed[s] for s in week.surgeon)):
        try:
            evaluate(week, candidate, week.booked)
        except ValueError:
            continue
        yield list(candidate)


def test_exact_weekly_planner_and_outcome_blind_ties():
    w = toy()
    possible = list(feasible(w))
    p = solve_week(w, w.booked, seconds=20)
    assert p.complete, p
    v = min(evaluate(w, a, w.booked)["cost"] for a in possible)
    assert p.value == pytest.approx(v)
    assert p.value == pytest.approx(130.)
    assert evaluate(w, p.assignment, w.booked)["moved_weekdays"] == 1
    assert evaluate(w, p.assignment, w.actual)["cost"] > p.value

    first = p.assignment[:]
    w.actual[:] = [800., 100., 200.]
    second = solve_week(w, w.booked, seconds=20)
    assert second.complete and first == second.assignment


def test_loss_augmented_max_is_exact_not_epigraph_relaxation():
    w = toy()
    for gamma in (1., 2., 4.):
        out = solve_adversary(w, w.booked, gamma=gamma, seconds=20)
        assert out.complete, out
        expected = max(
            evaluate(w, a, w.actual)["cost"] - gamma * evaluate(w, a, w.booked)["cost"]
            for a in feasible(w)
        )
        assert out.value == pytest.approx(expected, abs=.001)
        assert out.bound >= out.value - .001


def test_decomposition_and_zero_choice_have_exact_zero_regret():
    w = toy()
    w.allowed = ((0,), (1,))
    w.added_days = ()
    components = w.components()
    assert len(components) == 2
    assert sum(c.n for c in components) == 3
    for c in components:
        p = solve_week(c, c.booked)
        oracle = solve_week(c, c.actual)
        adv = solve_adversary(c, c.booked, gamma=2.)
        assert p.complete and oracle.complete and adv.complete
        assert p.assignment == oracle.assignment
        assert evaluate(c, p.assignment, c.actual)["cost"] == pytest.approx(oracle.value)
        assert adv.value == pytest.approx(oracle.value - 2 * p.value)


def test_original_operating_days_cannot_be_emptied():
    w = toy()
    with pytest.raises(ValueError, match="emptied"):
        evaluate(w, [1, 1, 1], w.booked)
    with pytest.raises(ValueError, match="Ineligible"):
        evaluate(w, [0, 0, 0], w.booked)


def cohort_and_events():
    monday = pd.Timestamp("2012-01-02")
    rows = pd.DataFrame({
        "group": ["TGH"] * 3, "surgeon": ["A", "A", "B"],
        "date": pd.to_datetime(["2012-01-02", "2012-01-02", "2012-01-04"]),
        "room": ["OR1", "OR1", "OR2"], "case_id": [2, 3, 4],
        "booked": [400., 200., 200.], "actual": [200., 400., 200.],
        "split": ["train"] * 3,
    })
    prior = pd.DataFrame({
        "case_id": [5], "group": ["TGH"], "surgeon": ["A"],
        "room": ["OR2"], "date": pd.to_datetime(["2011-12-28"]),
    })
    current = rows[["case_id", "group", "surgeon", "room", "date"]]
    events = pd.concat([prior, current], ignore_index=True)
    events["weekday"] = events.date.dt.weekday
    events["monday"] = events.date - pd.to_timedelta(events.weekday, unit="D")
    return rows, events


def test_prior_one_date_works_but_future_activity_cannot_leak():
    rows, events = cohort_and_events()
    one = build_weeks(rows, events, prior_dates=1)[0]
    assert one.added_days == (("A", 2, "OR2", "prior_dates"),)
    assert set(one.allowed[0]) == {0, 1}
    assert len(one.components()) == 1

    two = build_weeks(rows, events, prior_dates=2)[0]
    assert two.added_days == ()
    assert len(two.components()) == 2

    future = events.copy()
    future.loc[future.case_id.eq(5), "date"] = pd.Timestamp("2012-01-11")
    future["weekday"] = future.date.dt.weekday
    future["monday"] = future.date - pd.to_timedelta(future.weekday, unit="D")
    assert build_weeks(rows, future, prior_dates=1)[0].added_days == ()


def test_excluded_performed_day_and_single_case_exception():
    rows, events = cohort_and_events()
    events = events[events.case_id.ne(5)].copy()
    extra = pd.DataFrame({
        "case_id": [99], "group": ["TGH"], "surgeon": ["A"],
        "room": ["OR2"], "date": pd.to_datetime(["2012-01-04"]),
        "weekday": [2], "monday": [pd.Timestamp("2012-01-02")]
    })
    events = pd.concat([events, extra], ignore_index=True)
    w = build_weeks(rows, events, prior_dates=99)[0]
    assert w.added_days == (("A", 2, "OR2", "excluded_performed"),)
    rows = rows[rows.case_id.ne(3)]
    assert build_weeks(rows, events, prior_dates=1)[0].added_days == ()


def test_no_cross_room_group_leakage():
    rows, events = cohort_and_events()
    events.loc[events.case_id.eq(5), "group"] = "TWH-main"
    assert build_weeks(rows, events, prior_dates=1)[0].added_days == ()


def test_all_sessions_are_paid_even_if_empty():
    w = toy()
    # With Wednesday still occupied by B, verify explicit idle minutes there.
    original = [0, 0, 1]
    m = evaluate(w, original, w.booked)
    assert m["cost"] == pytest.approx(542.5)
    # A paid room-day remains in the sum even if its occupancy becomes zero
    # when evaluating a week not subject to the day-coverage restriction.
    w2 = Week("TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
              np.array([2]), ("A",), np.array([0]), np.array([0]),
              ((0,),), ((0,),), np.array([100.]), np.array([100.]), ())
    assert evaluate(w2, [0], w2.booked)["idle"] == pytest.approx(860.)


def test_invalid_durations_and_gamma():
    w = toy()
    with pytest.raises(ValueError):
        solve_week(w, [0, 20, 30])
    with pytest.raises(ValueError):
        solve_adversary(w, w.booked, gamma=.8)


def test_three_surrogate_bounds_and_zero_choice_cancellation():
    w = toy()
    b = solve_week(w, w.booked)
    a = solve_week(w, w.actual, tie=False)
    adv = solve_adversary(w, w.booked, gamma=2.)
    assert b.complete and a.complete and adv.complete
    regret = evaluate(w, b.assignment, w.actual)["cost"] - a.value
    gap = 2 * b.value + adv.value - a.value
    spo = 2 * evaluate(w, a.assignment, w.booked)["cost"] + adv.value - a.value
    error = w.actual - w.booked
    vf = b.value + np.sum(1.75 * np.maximum(error, 0.) + np.maximum(-error, 0.)) - a.value
    assert -1e-5 <= regret <= gap + 1e-4 <= spo + 1e-4
    assert gap <= vf + 1e-4

    w.allowed = ((0,), (1,))
    w.added_days = ()
    for c in w.components():
        booked = solve_week(c, c.booked)
        oracle = solve_week(c, c.actual)
        maxer = solve_adversary(c, c.booked, gamma=2.)
        assert 2 * booked.value + maxer.value - oracle.value == pytest.approx(0.)


def test_raw_operating_evidence_excludes_cancellations_and_emergencies():
    from surgery.data import COLUMNS
    from surgery.weekly_data import performed_events

    def row(surgeon, date, *, cancelled=False, emergency=False):
        item = {name: None for name in COLUMNS}
        item.update(Operating_Room="OR2", Surgeon_Code=surgeon,
                    Main_Procedure_Id="X", Case_Service="S",
                    Patient_Type="EMERGENCY" if emergency else "ELECTIVE",
                    Case_Cancelled_Reason="cancelled" if cancelled else None,
                    **{"Booked Time (Minutes)": 59})
        for col, clock in (
            ("Enter Room", "08:00:00"), ("Actual Start", "08:10:00"),
            ("Actual Stop", "08:45:00"), ("Leave Room", "09:00:00")
        ):
            item[col + " Date"] = date
            item[col + " Time"] = clock
        return item

    raw = pd.DataFrame([
        row("a", "2011-12-28"),
        row("a", "2011-12-28", cancelled=True),
        row("a", "2011-12-28", emergency=True),
    ])
    evidence = performed_events(raw)
    assert len(evidence) == 1
    assert evidence.case_id.tolist() == [2]
    assert evidence.weekday.tolist() == [2]
    assert evidence.room.tolist() == ["OR2"]


def test_true_canonical_tie_uses_case_order_not_weighted_score():
    w = toy()
    w.booked[:] = [300., 300., 200.]
    plan = solve_week(w, w.booked, seconds=20)
    assert plan.complete, plan
    # Moving either A case gives the same minimum cost and number of moves.
    # Lexicographically, the earlier case keeps the lower slot index.
    assert plan.assignment == [0, 1, 1]
    assert plan.primary_seconds >= 0
    assert plan.tie_seconds >= 0
    assert plan.value == pytest.approx(267.5)


def test_balancing_is_a_true_secondary_objective():
    from surgery.weekly_planner import evaluate
    n = 6
    original = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    w = Week(
        "TGH", "2012-01-02", ((0, "OR1"), (0, "OR2")),
        np.arange(n) + 2, tuple(str(i) for i in range(n)),
        np.arange(n), original, tuple((0, 1) for _ in range(n)),
        tuple((0,) for _ in range(n)),
        np.full(n, 400.), np.full(n, 400.), ()
    ).validate()
    baseline = solve_week(w, w.booked)
    balanced = solve_week(w, w.booked, balance=True)
    assert baseline.complete and balanced.complete
    assert balanced.value == pytest.approx(baseline.value)
    assert balanced.assignment != baseline.assignment
    assert evaluate(w, baseline.assignment, w.booked)["max_room_load"] == 1690.
    assert evaluate(w, balanced.assignment, w.booked)["max_room_load"] == 1260.
    assert evaluate(w, baseline.assignment, w.booked)["moved_cases"] == 0
    assert evaluate(w, balanced.assignment, w.booked)["moved_cases"] == 1
