"""Exactness and equivalence tests for the unified anchored weekly MILP."""
from itertools import product
from pathlib import Path
import hashlib

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import (
    solve_week as original_solve, solve_adversary as original_adversary,
    evaluate, _objective_tolerance,
)
from surgery import anchored_weekly_solver as improved
from surgery import anchored_oracles as oracles
import run_weekly_anchored as runner


def example():
    # Two weekday-required singleton rooms, one flexible surgeon, three cases.
    return Week('TWH-main','2012-02-27',((0,'R1'),(0,'R2')),
        np.array([1,2,3]), ('A','B','C'),
        np.array([0,1,2]), np.array([0,1,0]),
        ((0,), (1,), (0,1)), ((0,), (0,), (0,)),
        np.array([220.,160.,130.]),np.array([200.,180.,115.]),()
    ).validate()


def variable_open():
    return Week('TGH','2012-01-02',((0,'OR1'),(1,'OR2'),(2,'OR3')),
        np.array([2,3,4,5]),('A','B'),np.array([0,0,0,1]),
        np.array([0,0,0,2]),((0,1,2),(1,2)),((0,),(2,)),
        np.array([80.,135.,320.,220.]),np.array([100.,155.,325.,215.]),()
    ).validate()


def enumerate_best(week,d):
    feasible=[]
    for z in product(*[week.allowed[int(s)] for s in week.surgeon]):
        try: e=evaluate(week,z,d)
        except ValueError: continue
        feasible.append((e['cost'],e['moved_cases'],tuple(z)))
    optimum=min(x[0] for x in feasible)
    tied=[t for t in feasible if t[0] <= optimum+_objective_tolerance(optimum)-1e-7]
    chosen=min(tied,key=lambda v:(v[1],v[2]))
    return optimum,chosen[2]


def test_strengthening_keeps_frozen_feature_fingerprint_file_unchanged():
    content=(Path(__file__).resolve().parents[1]/'surgery/weekly_planner.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert blob=='3bf935ea74a6b5c6b6a332d9227f3022ad7dffb5'
    assert oracles.solve_week is improved.solve_week
    assert oracles._model is improved._model
    assert runner.evaluate_week is improved.evaluate_week
    assert 'surgery/anchored_weekly_solver.py' in runner.SOURCE_FILES


def test_forced_room_fixing_is_structural_not_heuristic():
    c=example()
    assert improved.mandatory_slots(c)==frozenset({0,1})
    v=variable_open()
    assert improved.mandatory_slots(v)==frozenset({0,2})
    assert improved.conservation_certificate(c,np.array([130.,130.,130.])) is not None
    assert improved.conservation_certificate(c,np.array([650.,100.,100.])) is None


@pytest.mark.parametrize("seed",[1,3,7,11,91])
def test_planner_matches_original_and_full_enumeration(seed):
    pytest.importorskip('gurobipy')
    rng=np.random.default_rng(seed)
    for week in (example(),variable_open()):
        for kind in ('random','integer','equal'):
            d=(rng.uniform(70,510,week.n) if kind=='random'
               else rng.integers(75,400,week.n).astype(float) if kind=='integer'
               else np.full(week.n,135.))
            expected,zstar=enumerate_best(week,d)
            new=improved.solve_week(week,d,seconds=45,threads=1,tie=True)
            old=original_solve(week,d,seconds=45,threads=1,tie=True)
            assert new.complete and old.complete,(new,old)
            assert new.value==pytest.approx(expected,abs=.0002)
            assert old.value==pytest.approx(expected,abs=.0002)
            assert new.assignment==old.assignment==list(zstar)


def test_adversary_uses_same_strengthened_integral_feasible_set():
    pytest.importorskip('gurobipy')
    c=variable_open()
    d=c.booked+np.array([7.,-9.,11.,-5.])
    new=improved.solve_adversary(c,d,gamma=2.,seconds=60,threads=1)
    old=original_adversary(c,d,gamma=2.,seconds=60,threads=1)
    assert new.complete and old.complete
    assert new.value==pytest.approx(old.value,abs=.0002)


def test_lp_link_strengthening_is_present_and_mandatory_slot_is_fixed():
    pytest.importorskip('gurobipy')
    c=variable_open()
    m,x,u,byroom=improved._model(c,20,1)
    try:
        m.update()
        links=[con for con in m.getConstrs() if con.ConstrName.startswith('open_link_')]
        assert len(links)==sum(len(c.allowed[int(s)]) for s in c.surgeon)
        assert all(u[j].LB==1. for j in improved.mandatory_slots(c))
        assert all(u[j].LB==0. for j in set(range(c.k))-improved.mandatory_slots(c))
    finally:
        m.dispose()
