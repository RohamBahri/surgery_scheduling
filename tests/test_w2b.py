"""Small exact checks for W2b feasibility, loss augmentation and regret bounds."""
from itertools import product

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from w2b.data import Week
from w2b.planner import schedule_cost, solve_week, solve_adversary


def toy():
    slots = ((0, "R1"), (0, "R2"), (1, "R1"), (1, "R2"))
    return Week(
        "TGH", "2012-01-02", np.array([1, 2]),
        np.array([350., 350.]), np.array([300., 500.]),
        np.array([[1., -1.], [1., 1.]]),
        np.array([0, 1]), np.array([0, 0]), np.array([1, 1]),
        ((0, 1), (0, 1)), slots,
        tuple((i, j) for i in range(2) for j in range(4)), 0)


def test_fixed_capacity_counts_unused_sessions():
    w = toy()
    assignment = [0, 1]
    expected = (480 - 300) + 1.75 * (500 - 480) + 2 * 480
    assert schedule_cost(w, assignment, w.actual) == expected
    assert schedule_cost(w, [2, 3], w.actual, 5.) == expected + 10


def test_full_planner_and_adversarial_oracle_match_enumeration():
    w = toy()
    options = list(product(range(4), repeat=2))
    for gamma in (1., 2., 10.):
        pred = w.booked + np.array([-15., 15.])
        plan = solve_week(w, pred, seconds=30., move_penalty=5.)
        assert plan.optimal, plan
        minimum = min(schedule_cost(w, z, pred, 5.) for z in options)
        assert abs(plan.cost - minimum) < 1e-5
        assert abs(schedule_cost(w, plan.assignment, pred, 5.) - minimum) < 1e-5

        adv = solve_adversary(w, pred, gamma=gamma, seconds=30., move_penalty=5.)
        assert adv.optimal, adv
        maximum = max(schedule_cost(w, z, w.actual, 5.)
                      - gamma * schedule_cost(w, z, pred, 5.) for z in options)
        assert abs(adv.cost - maximum) < 1e-5

        actual_oracle = min(schedule_cost(w, z, w.actual, 5.) for z in options)
        regret = schedule_cost(w, plan.assignment, w.actual, 5.) - actual_oracle
        bg = gamma * minimum + maximum - actual_oracle
        hindsight = min(options, key=lambda z: schedule_cost(w, z, w.actual, 5.))
        spo = gamma * schedule_cost(w, hindsight, pred, 5.) + maximum - actual_oracle
        old = (minimum + sum(1.75 * max(e, 0) + max(-e, 0)
                             for e in w.actual - pred) - actual_oracle)
        assert regret <= bg + 1e-5
        assert bg <= spo + 1e-5
        assert bg <= old + 1e-5


def test_one_room_per_surgeon_day_and_fixed_day_count():
    w = toy()
    w.cases = np.array([1, 2, 3])
    w.booked = np.array([180., 180., 180.])
    w.actual = np.array([200., 200., 200.])
    w.X = np.ones((3, 1))
    w.surgeon = np.array([0, 0, 0])
    w.original_day = np.array([0, 1, 1])
    w.days_required = np.array([2])
    w.allowed_days = ((0, 1),)
    w.arcs = tuple((i, j) for i in range(3) for j in range(4))
    solution = solve_week(w, w.booked, seconds=30.)
    assert solution.optimal, solution
    rooms = solution.assignment
    weekdays = [w.slots[j][0] for j in rooms]
    assert set(weekdays) == {0, 1}
    for day in (0, 1):
        assert len({rooms[i] for i in range(3) if weekdays[i] == day}) == 1


def test_three_learning_methods_smoke(tmp_path):
    from w2b.learning import train_method
    w = toy()
    for method in ("vf", "gap", "spo"):
        run = train_method(
            method, [w], start=np.zeros(2), alpha=.8, h=30.,
            gamma=2., lam=.1, outer=1, inner_evals=30,
            seconds=15., adversary_seconds=15., threads=1,
            cache=tmp_path, move_penalty=0.)
        assert len(run["iterations"]) == 2
        assert run["status"] == "heuristic_library_MM"
        for iteration in run["iterations"]:
            assert np.isfinite(iteration["library_proxy"])
            assert np.isfinite(iteration["evaluation"]["realized"])
        assert run["iterations"][-1]["evaluation"]["certified"]
