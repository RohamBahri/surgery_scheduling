"""Exactness and equivalence tests for the unified anchored weekly MILP."""
from itertools import product
from pathlib import Path
import hashlib

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import (
    solve_adversary as original_adversary,
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


def enumerate_best(week,d,seed=0):
    feasible=[]
    for z in product(*[week.allowed[int(s)] for s in week.surgeon]):
        try: e=evaluate(week,z,d)
        except ValueError: continue
        feasible.append((e['cost'],tuple(z)))
    optimum=min(x[0] for x in feasible)
    tied=[z for cost,z in feasible if cost<=optimum+_objective_tolerance(optimum)+1e-8]
    chosen=min(tied,key=lambda z:(improved.hash_score(week,z,seed),
                                   improved.hash_score(week,z,seed+1),z))
    return optimum,chosen


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
def test_planner_matches_enumerated_primary_and_hash_selection(seed):
    pytest.importorskip('gurobipy')
    rng=np.random.default_rng(seed)
    for week in (example(),variable_open()):
        for kind in ('random','integer','equal'):
            d=(rng.uniform(70,510,week.n) if kind=='random'
               else rng.integers(75,400,week.n).astype(float) if kind=='integer'
               else np.full(week.n,135.))
            expected,zstar=enumerate_best(week,d)
            deployed=improved.solve_week(week,d,seconds=90,threads=1,
                                         hash_seed=0,tie=True)
            assert deployed.complete and deployed.hash_unique,deployed
            assert deployed.value==pytest.approx(expected,abs=.12)
            assert deployed.assignment==list(zstar)
            assert deployed.hash_seed==0
            assert deployed.uniqueness_status in (
                'UNIQUE_CERTIFIED','LEX_FALLBACK_CERTIFIED','UNIQUE_FEASIBLE')
            primary=improved.solve_week(week,d,seconds=90,threads=1,tie=False)
            assert primary.complete and primary.value==pytest.approx(expected,abs=1e-4)


def test_hash_priorities_ignore_original_assignment_and_actual_durations():
    c=example()
    a=list(map(int,c.original))
    first=improved.hash_score(c,a,0)
    c.original[:]=np.array([0,1,1])
    c.actual[:]=np.array([400.,410.,420.])
    assert improved.hash_score(c,a,0)==first
    assert improved.hash_score(c,a,1)!=first
    assert all(1<=improved.hash_weight(0,cid,*slot)<=2**20
               for cid in c.case_ids for slot in c.slots)


def test_conservation_proof_does_not_select_historical_tie():
    pytest.importorskip('gurobipy')
    c=example()
    d=np.array([130.,130.,130.])
    assert improved.conservation_certificate(c,d) is not None
    expected,zstar=enumerate_best(c,d)
    result=improved.solve_week(c,d,seconds=90,threads=1)
    assert result.complete and result.primary_status=='ANALYTIC_CONSERVATION'
    assert result.assignment==list(zstar)
    assert result.value==pytest.approx(expected)


def test_true_double_hash_collision_uses_certified_canonical_fallback(monkeypatch):
    pytest.importorskip('gurobipy')
    c=example()
    # Every feasible assignment has the SAME primary and hash scores.
    monkeypatch.setattr(improved,'hash_weight',lambda *args:1)
    d=np.full(c.n,135.)
    expected,zstar=enumerate_best(c,d)
    result=improved.solve_week(c,d,seconds=90,threads=1)
    assert result.complete and result.hash_unique,result
    assert result.uniqueness_status=='LEX_FALLBACK_CERTIFIED'
    assert len(result.hash_attempts)==2
    assert result.assignment==list(zstar)
    assert result.value==pytest.approx(expected)


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
