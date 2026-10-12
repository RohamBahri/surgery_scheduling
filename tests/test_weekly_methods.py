"""Fast sanity checks for diagnostic and training pilot jobs."""
import numpy as np
import pandas as pd
import pytest

from run_weekly_learning import _grid, _summarize, candidate_job
from surgery.weekly_data import Week
from surgery.weekly_planner import solve_week
from run_weekly_diagnostics import audit_component


def example():
    return Week("TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
                np.array([2, 3, 4]), ("A", "B"), np.array([0, 0, 1]),
                np.array([0, 0, 1]), ((0, 1), (1,)), ((0,), (2,)),
                np.array([400., 200., 200.]), np.array([200., 400., 200.]),
                (("A", 2, "OR2", "prior_dates"),)).validate()


def test_load_audit_reproduces_certified_booked_cost():
    w = example()
    booked = solve_week(w, w.booked)
    from surgery.weekly_planner import evaluate
    key = dict(group="TGH", monday=w.monday, component=0)
    previous = {"booked_cost": evaluate(w, booked.assignment, w.actual)["cost"]}
    summary, rooms = audit_component((key, w, previous, 20., 1))
    assert summary["baseline_predicted_cost"] == pytest.approx(
        summary["balanced_predicted_cost"], abs=.01)
    assert len(rooms) == w.k
    assert summary["balanced_max_predicted_room_load"] <= (
        summary["baseline_max_predicted_room_load"] + .01)


def test_training_candidate_scores_exact_surrogate_bounds():
    w = example()
    actual = solve_week(w, w.actual, tie=False)
    task = (w, "train", 0., 0., .8, 30., 2.,
            [(actual.assignment, actual.value)], 20., 1)
    row = candidate_job(task)
    assert row["complete"]
    assert row["cases"] == 3
    assert row["gap"] >= row["regret"] - .01
    assert row["spo"] >= row["gap"] - .01
    assert row["vf"] >= row["regret"] - .01
    assert row["case_error"] == pytest.approx(sum(abs(w.actual - w.booked)))


def test_policy_selection_uses_train_not_validation():
    candidates = []
    for group in ("TGH", "TWH-main", "TWH-day"):
        for period in ("train", "validation"):
            for offset in (-15., 0., 15.):
                value = 0. if (offset == 15. and period == "train") else 100.
                if period == "validation":
                    value = 0. if offset == -15. else 100.
                candidates.append(dict(group=group, period=period, monday="2012-01-02",
                                       offset=offset, slope=0.,
                                       complete=True, cases=5,
                                       case_error=value, vf=value, gap=value,
                                       spo=value, regret=value, realized_cost=value+5,
                                       oracle_cost=5., booked_moves=0))
    policies, report = _summarize(pd.DataFrame(candidates),
                                  (-15., 0., 15.), (0.,))
    selected = policies[policies.method.eq("GAP")]
    assert (selected.offset == 15.).all()
    assert (report[report.method.eq("GAP") & report.period.eq("validation")].
            regret_per_case == 20.).all()
    assert _grid("-15,0,15") == (-15., 0., 15.)
    with pytest.raises(ValueError):
        _grid("0,0")
