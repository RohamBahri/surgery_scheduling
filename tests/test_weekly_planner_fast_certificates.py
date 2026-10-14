"""Proof-based exact planner shortcuts: never change the chosen schedule."""
from itertools import product
import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import (
    conservation_certificate, original_wins_tie,
    _objective_tolerance, evaluate, solve_week,
)


def forced_room_instance():
    # Rooms 0 and 1 each contain a surgeon whose required operating weekday
    # can be served ONLY in that physical room; third surgeon may change rooms.
    return Week(
        "TWH-day","2011-11-28",((0,"R0"),(0,"R1")),
        np.array([10,11,12]),("S0","S1","S2"),
        np.array([0,1,2]),np.array([0,1,0]),
        ((0,), (1,), (0,1)),((0,), (0,), (0,)),
        np.array([100.,100.,50.]),np.array([105.,95.,55.]),
        (),
    ).validate()


def unrestricted_instance():
    return Week(
        "TWH-day","2011-11-28",((0,"R0"),(0,"R1")),
        np.array([10,11,12]),("S0","S1","S2"),
        np.array([0,1,2]),np.array([0,1,0]),
        ((0,1),(0,1),(0,1)),((0,), (0,), (0,)),
        np.array([100.,100.,50.]),np.array([105.,95.,55.]),
        (),
    ).validate()


def brute_force(week,d):
    costs=[]
    for assignment in product(*[week.allowed[int(i)] for i in week.surgeon]):
        # Only retain the exact surgeon-day constraints.
        try:
            entry=evaluate(week,assignment,d)
        except ValueError:
            continue
        costs.append((entry["cost"],entry["moved_cases"],tuple(assignment)))
    return sorted(costs)


def test_full_room_conservation_bound_matches_enumeration():
    w=forced_room_instance()
    rng=np.random.default_rng(123)
    for _ in range(80):
        d=rng.uniform(20.,800.,w.n)
        cert=conservation_certificate(w,d)
        true_optimum=brute_force(w,d)[0][0]
        if cert is not None:
            assert cert["cost"]==pytest.approx(true_optimum,abs=1e-6)
            assert cert["lower_bound"]<=true_optimum+1e-6
            assert cert["assignment"]==list(map(int,w.original))
            assert not brute_force(w,d)[0][0]<cert["lower_bound"]-1e-6
    # All room-days are forced open and none is overloaded.
    d=np.array([100.,100.,50.])
    cert=conservation_certificate(w,d)
    assert cert is not None
    assert cert["cost"]==pytest.approx(960.-280.)
    # Mixed over- and under-capacity loads require the exact fallback MIP.
    assert conservation_certificate(w,np.array([100.,100.,700.])) is None


def test_conservation_shortcut_never_assumes_a_room_is_mandatory():
    c=unrestricted_instance()
    assert conservation_certificate(c,np.array([100.,100.,50.])) is None
    # If an alternative can close a room, its lower cost must be considered.
    optimum=brute_force(c,np.array([100.,100.,50.]))[0][0]
    assert optimum<evaluate(c,c.original,c.booked)["cost"]


def test_original_zero_move_tie_condition_uses_actual_planner_tolerance():
    w=forced_room_instance()
    d=np.array([100.,100.,50.])
    optimum=brute_force(w,d)[0][0]
    assert original_wins_tie(w,d,optimum)
    assert _objective_tolerance(optimum)<=.1
    assert not original_wins_tie(w,np.array([100.,100.,700.]),0.)


def test_shortcuts_select_exact_full_planner_assignment():
    pytest.importorskip("gurobipy")
    c=forced_room_instance()
    d=np.array([100.,100.,50.])
    r=solve_week(c,d,seconds=30.,threads=1,tie=True)
    assert r.complete
    assert r.status=="ANALYTIC_CONSERVATION"
    assert r.assignment==list(map(int,c.original))
    assert r.value==pytest.approx(brute_force(c,d)[0][0])
    assert r.bound<=r.value
    r0=solve_week(c,d,seconds=30.,threads=1,tie=False)
    assert r0.complete and r0.status=="ANALYTIC_CONSERVATION"
    # A non-structural instance continues to use the standard exact MIP.
    u=unrestricted_instance()
    r1=solve_week(u,u.booked,seconds=30.,threads=1,tie=True)
    assert r1.complete and r1.status!="ANALYTIC_CONSERVATION"
    assert r1.value==pytest.approx(brute_force(u,u.booked)[0][0])
    assert r1.assignment is not None
