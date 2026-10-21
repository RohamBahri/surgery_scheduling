"""Licensed-Gurobi integration checks against independent enumeration.

Run `pytest -q tests/test_theory_solver.py` on the full research workstation.
These tests are skipped in the lightweight artifact-creation environment.
"""
import itertools
import numpy as np
import pytest

pytest.importorskip('gurobipy')
from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate, _objective_tolerance
from surgery.anchored_weekly_solver import solve_week
from surgery.theory_exact import (expected_reduced, vf_per_week_milp,
             saa_mip, expected_gap_mip, vf_objective)
from surgery.theory_bilevel import value_function_exact


def fixture_week():
    # Real weekly feasibility objects, not a fake solver or omitted constraints.
    return Week(group='TGH',monday='2012-01-02',
       slots=((0,'OR1'),(0,'OR2')),
       case_ids=np.array([1001,1002,1003]),
       surgeon_ids=('S1','S2','S3'),
       surgeon=np.array([0,1,2]),
       original=np.array([0,1,0]),
       allowed=((0,1),(0,1),(0,1)),
       required_days=((0,),(0,),(0,)),
       booked=np.array([160.,200.,270.]),
       actual=np.array([150.,260.,230.]),
       added_days=()).validate()


def feasible_schedules(c):
    for z in itertools.product(*[c.allowed[int(s)] for s in c.surgeon]):
        try:evaluate(c,z,c.booked)
        except ValueError:continue
        yield tuple(z)


def bank():
    return np.array([[160,210,280],[150,260,200],[170,190,320],
                     [180,220,300],[130,250,260]],dtype=float)


def test_saa_is_global_over_every_assignment():
    c=fixture_week();draws=bank()
    answer=saa_mip(c,draws,seconds=30)
    assert answer.complete,answer
    brute=min(expected_reduced(c,z,draws) for z in feasible_schedules(c))
    assert answer.value==pytest.approx(brute,abs=1e-5)
    assert answer.lower==pytest.approx(brute,abs=1e-5)


def test_expected_gap_competitor_max_is_exact():
    c=fixture_week();draws=bank();d=c.booked.copy()
    primary=solve_week(c,d,seconds=30,tie=False)
    assert primary.complete
    epsilon=_objective_tolerance(primary.value)
    brute=max(expected_reduced(c,z,draws)
              -max(0.,evaluate(c,z,d)['cost']-primary.value-epsilon)
              for z in feasible_schedules(c))
    oracle=expected_gap_mip(c,d,draws,seconds=30)
    assert oracle.complete,oracle
    assert oracle.value==pytest.approx(brute,abs=1e-5)


def test_unrestricted_vf_bound_matches_global_lp_oracle():
    from scipy.optimize import linprog
    from surgery.planner import reachable
    c=fixture_week();draws=bank()
    actual=vf_per_week_milp(c,draws,seconds=30,d0=None,lam=0.)
    assert actual.complete,actual
    lo,hi=reachable(c.booked,.8,30.)
    # Independent LP per assignment with duration, overage and sample stop-loss
    # variables. Enumeration is complete for this three-case instance.
    alternatives=[]
    for z in feasible_schedules(c):
        z=np.asarray(z);rooms=sorted(set(z))
        n=c.n;k=len(rooms);N=len(draws)
        size=n+k+n*N
        f=np.zeros(size)
        f[n:n+k]=2.75
        f[n+k:]=2.75/N
        A=[];b=[]
        for rj,room in enumerate(rooms):
            ids=np.flatnonzero(z==room)
            row=np.zeros(size);row[ids]=1.;row[n+rj]=-1.
            A.append(row);b.append(480-30*(len(ids)-1))
        for i in range(n):
            for s in range(N):
                row=np.zeros(size);row[i]=-1;row[n+k+i*N+s]=-1
                A.append(row);b.append(-draws[s,i])
        bounds=[(lo[i],hi[i]) for i in range(n)]+[(0,None)]*(k+n*N)
        lp=linprog(f,A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=bounds,method='highs')
        assert lp.success,lp.message
        alternatives.append(30*k+lp.fun)
    assert actual.value==pytest.approx(min(alternatives),abs=1e-5)


def test_bilevel_bounds_versus_full_grid():
    from surgery.planner import reachable
    from surgery.theory_exact import expected_reduced
    c=fixture_week();draws=bank()
    lo,hi=reachable(c.booked,.8,30.)
    # Full grid is a subset of continuous advice, so its optimum is an upper
    # bound (not an exact continuous optimum).
    best=np.inf
    for d in itertools.product(*[np.linspace(lo[i],hi[i],3) for i in range(c.n)]):
        sol=solve_week(c,np.asarray(d),seconds=30,tie=True,hash_seed=0)
        assert sol.complete
        best=min(best,expected_reduced(c,sol.assignment,draws)
                 + .001/2 * np.square(np.asarray(d)-c.booked).sum())
    result=value_function_exact(c,draws,c.booked,seconds=90,separation_seconds=30,cuts=80)
    assert result['lower']<=best+1e-4,result
    if result['upper'] is not None:
        assert result['upper']>=result['lower']-1e-4
        # Upper bound is a feasible continuous policy; it may be WORSE than a
        # three-level grid policy when the global solver is time-limited.
        # Only a certified global optimum must beat this finite grid.
        if result['complete']:
            assert result['upper']<=best+1e-4