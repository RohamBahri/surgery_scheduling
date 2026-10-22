"""Mathematical and protocol release gates for the four-objective run."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def sample():
    from surgery.weekly_data import Week
    c=Week(group='TGH',monday='2012-01-02',
       slots=((0,'OR1'),(0,'OR2')),
       case_ids=np.array([101,102,103]),
       surgeon_ids=('S1','S2','S3'),
       surgeon=np.array([0,1,2]),
       original=np.array([0,1,0]),
       allowed=((0,1),)*3,
       required_days=((0,),)*3,
       booked=np.array([160.,200.,270.]),
       actual=np.array([150.,260.,230.]),
       added_days=()).validate()
    draws=np.array([[160.,210.,280.],[150.,260.,200.],
                    [170.,190.,320.]],float)
    return c,draws


def test_matched_methods_and_global_wall_budget():
    from pilot import run_objective_selection as run
    assert run.METHODS==('SHARED','VF','EG_LIB','ANCHORED_T')
    assert run.ARMS==('BOOKED','CASE','STAT','SHARED','VF','EG_LIB','ANCHORED_T')
    assert sum(limit for _,limit in run.STAGES)<11.5*3600
    names=[x[0] for x in run.STAGES]
    assert names.index('freeze')<names.index('evaluate')<names.index('report')
    assert dict(run.STAGES)['evaluate']>=2.5*3600


def test_objective_library_matches_independent_EG_and_target_formula(monkeypatch):
    from pilot import run_theory_full as runner
    from surgery.weekly_planner import evaluate,_objective_tolerance
    from surgery.theory_exact import reduced,expected_reduced
    from surgery.anchored_weekly_solver import solve_week
    c,draws=sample()
    schedules=[[0,1,0],[1,0,1]]
    scores=[expected_reduced(c,z,draws) for z in schedules]
    library=dict(schedules=schedules,g_A=scores,g_B=scores)
    runner._SCORE_STATE=dict(libraries={runner.key(c):library})
    try:
        d=c.booked.astype(float)
        d0=d.copy()
        phi=[reduced(c,z,d) for z in schedules]
        g=np.array(scores)
        target=int(np.argmin(g))
        expected_anchor=phi[target]+np.max(g-np.array(phi))
        got_anchor=runner.scores(c,d,d0,draws,'ANCHORED_T',
                                  30.,seed=20261020)
        assert got_anchor==pytest.approx(expected_anchor,abs=1e-6)
        sol=solve_week(c,d,seconds=20,threads=1,tie=False)
        assert sol.complete
        Q=sol.value-480*c.k+d.sum()+30*c.n
        eps=_objective_tolerance(sol.value)
        expected_eg=Q+max(gz-max(fz-Q-eps,0.)
                         for gz,fz in zip(g,phi))
        got_eg=runner.scores(c,d,d0,draws,'EG_LIB',30.,seed=20261020)
        assert got_eg==pytest.approx(expected_eg,abs=1e-5)
        assert got_eg<=runner.scores(c,d,d0,draws,'EG_LIB',30.,
             seed=0)+1e-6
    finally:
        runner._SCORE_STATE=None


def test_restricted_eg_is_lower_than_exact_eg_on_same_bank():
    from surgery.theory_exact import expected_gap_mip
    from surgery.theory_exact import reduced,expected_reduced
    from surgery.weekly_planner import _objective_tolerance
    from surgery.anchored_weekly_solver import solve_week
    c,draws=sample()
    d=c.booked.astype(float)
    sol=solve_week(c,d,seconds=20,tie=False)
    assert sol.complete
    Q=sol.value-480*c.k+d.sum()+30*c.n
    eps=_objective_tolerance(sol.value)
    subset=([0,1,0],[1,0,1])
    restricted=Q+max(expected_reduced(c,z,draws)-
                     max(reduced(c,z,d)-Q-eps,0.) for z in subset)
    oracle=expected_gap_mip(c,d,draws,seconds=20)
    assert oracle.complete
    assert restricted<=Q+oracle.value+1e-5


def test_library_sampling_unique_and_full_training_preserved():
    from surgery.objective_selection import _sample
    objects=[]
    for i in range(200):
        c=SimpleNamespace(n=i+1)
        objects.append((str(i),c,np.array([i])))
    selected=_sample(objects,n=150)
    assert len(selected)==150
    assert len({x[0] for x in selected})==150
    assert '199' in {x[0] for x in selected}


def test_predeclared_test_and_training_banks_have_identical_seeds():
    from pilot import run_objective_selection as run
    assert run.full.SEEDS==(0,)
    from surgery import objective_selection as engine
    src=engine.train_objective.__doc__
    assert 'Bank-A AND Bank-B' in src


def test_recorded_room_fallback_cannot_be_called_exact_deployment(tmp_path):
    from pilot import run_objective_selection as run
    output=tmp_path/run.ROOT_NAME
    run.save(output/'run_status.json',dict(report=dict(status='PARTIAL')))
    assert json.loads((output/'run_status.json').read_text())['report']['status']=='PARTIAL'
