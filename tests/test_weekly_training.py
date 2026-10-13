"""Numerical and orchestration guards for full weekly CASE/VF learning."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate
from surgery.weekly_training import (
    asymmetric_error, response_with_derivative, fixed_exact,
    fixed_smooth, fit_fixed, fit_case, fixed_room_matrix,
    planning_task, training_lambda, fit_vf
)
from run_weekly_train import check_week_cases, summarize, evaluate_week


def example():
    w = Week(
        "TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
        np.array([2, 3, 4]), ("A", "B"),
        np.array([0, 0, 1]), np.array([0, 0, 1]),
        ((0,), (1,)), ((0,), (2,)),
        np.array([220., 230., 175.]),
        np.array([210., 180., 215.]), ()
    )
    return w.validate()


def test_post_response_derivative_matches_finite_differences():
    b = np.full(8, 120.)
    u = np.array([-90., -45., -12., -2., 3., 18., 44., 100.])
    pred, dr = response_with_derivative(u, b, .8, 30.)
    eps = 1e-5
    plus = response_with_derivative(u+eps, b, .8, 30.)[0]
    minus = response_with_derivative(u-eps, b, .8, 30.)[0]
    np.testing.assert_allclose(dr, (plus-minus)/(2*eps), atol=1e-6)
    assert (pred > 0).all()


def test_differentiable_objective_gradients_for_case_and_rooms():
    rng = np.random.default_rng(23)
    n = 18
    X = np.column_stack((np.ones(n), rng.normal(size=(n, 4))))
    booked = np.full(n, 165.)
    actual = booked + rng.normal(-12, 55, n)
    w = np.array([-5., .8, -1., 2., .2])
    S = csr_matrix((np.ones(n), (np.arange(n)//3, np.arange(n))),
                   shape=(6, n))
    offsets = np.full(6, 60.)
    for matrix, cons in ((None, None), (S, offsets)):
        val, grad = fixed_smooth(w, X, booked, actual, .8, 30., .055,
                                 matrix, cons, eps=3.)
        finite = []
        for j in range(len(w)):
            d = np.zeros_like(w)
            d[j] = 1e-5
            plus = fixed_smooth(w+d, X, booked, actual, .8, 30., .055,
                                matrix, cons, eps=3.)[0]
            minus = fixed_smooth(w-d, X, booked, actual, .8, 30., .055,
                                 matrix, cons, eps=3.)[0]
            finite.append((plus-minus)/(2e-5))
        np.testing.assert_allclose(grad, finite, rtol=2e-5, atol=1e-5)
        assert np.isfinite(val)


def test_fixed_schedule_cost_equals_weekly_evaluate():
    week = example()
    index = {int(k): i for i, k in enumerate(week.case_ids)}
    pairs = [[int(i), int(week.slots[j][0]), str(week.slots[j][1])]
             for i, j in zip(week.case_ids, week.original)]
    plans = {week.group + "|" + week.monday:
             {"complete": True, "assignment": pairs}}
    S, offset = fixed_room_matrix([week], plans, index, week.n)
    w = np.zeros(2)
    X = np.column_stack([np.ones(week.n), np.array([-1., 0., 1.])])
    cost = evaluate(week, week.original, week.booked)["cost"]
    theta = asymmetric_error(week.actual - week.booked).sum()
    assert fixed_exact(w, X, week.booked, week.actual, .8, 30., 0., S, offset) == pytest.approx(
        (theta + cost)/week.n)


def test_case_optimizes_post_response_case_loss():
    rng = np.random.default_rng(19)
    n = 120
    x = rng.normal(size=n)
    X = np.column_stack((np.ones(n), x))
    booked = np.full(n, 160.)
    actual = booked - 12 + x*8
    lam = .01
    fit = fit_case(X, booked, actual, .8, 30., lam,
                   lasso=[0., 0.], iterations=70)
    baseline = fixed_exact(np.zeros(2), X, booked, actual, .8, 30., lam)
    assert fit["objective"] < baseline
    assert fit["method"] == "CASE"
    assert fit["w"][0] < -3.
    assert 0 < training_lambda(X, draws=32) < 2.


def test_training_roster_and_frozen_zero_schedule_solver():
    week = example()
    train = {"case_id": week.case_ids}
    index = check_week_cases([week], train)
    assert set(index) == {2, 3, 4}
    out = planning_task((week, week.booked, "abcdef", 30., 1))
    assert out["complete"] and len(out["assignment"]) == 3
    assert out["sha256"] == "abcdef"
    assert out["predicted_cost"] == pytest.approx(
        evaluate(week, week.original, week.booked)["cost"])


def test_summary_resamples_calendar_weeks_not_cases(tmp_path):
    rows = []
    for wk in ("2012-08-27", "2012-09-03"):
        for group in ("TGH", "TWH-main", "TWH-day"):
            for method, cost in [("BOOKED", 100.), ("LASSO", 98.),
                                 ("CASE", 95.), ("VF", 91.)]:
                rows.append(dict(group=group, monday=wk, method=method,
                                 cases=10, abs_error=2, realized_cost=cost,
                                 regret=cost-80, vf_bound=cost,
                                 overtime=0., room_days_over_720=0))
    summarize(tmp_path, pd.DataFrame(rows), bootstrap_draws=100)
    import json
    report = json.loads((tmp_path/"evaluation_summary.json").read_text())
    assert report["paired_calendar_week_uncertainty"]["VF"]["calendar_weeks"] == 2
    assert report["methods"]["VF"]["group_weeks"] == 6
    assert report["paired_calendar_week_uncertainty"]["VF"]["percent_change"] == pytest.approx(-9.)


def test_vf_full_outer_step_certifies_and_never_increases_majorizer(tmp_path):
    week = example()
    n = week.n
    X = np.column_stack((np.ones(n), np.array([-1., 0., 1.])))
    index = {int(cid): i for i, cid in enumerate(week.case_ids)}
    settings = {"test": "identical configuration"}
    state = fit_vf(
        tmp_path, [week], X, week.booked, week.actual, index,
        .8, 30., .005, [0., 0.], settings,
        max_outer=2, iterations=35, seconds=30., threads=1, workers=1)
    assert state["complete"] is True
    assert 1 <= len(state["iterations"]) <= 2
    assert (tmp_path / "vf_progress.json").exists()
    for entry in state["iterations"]:
        if entry["accepted"]:
            assert entry["candidate_fixed_bound_minus_oracle"] < (
                entry["touching_vf_bound_minus_oracle"])
    again = fit_vf(
        tmp_path, [week], X, week.booked, week.actual, index,
        .8, 30., .005, [0., 0.], settings,
        max_outer=2, iterations=35, seconds=30., threads=1, workers=1)
    assert again == state


def test_exact_evaluation_worker_shares_hindsight_for_all_four_methods():
    week = example()
    policies = {
        "BOOKED": week.booked.copy(),
        "LASSO": week.booked.copy(),
        "CASE": week.booked.copy(),
        "VF": week.booked.copy(),
    }
    rows = evaluate_week((week, policies, 30., 1))
    assert len(rows) == 4 and all(row["complete"] for row in rows)
    assert len({row["oracle_assignment_sha256"] for row in rows}) == 1
    for row in rows:
        assert row["vf_bound"] + 1e-6 >= row["regret"]
        assert row["cases"] == week.n
    assert len({row["realized_cost"] for row in rows}) == 1
