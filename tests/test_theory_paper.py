"""Release-gate regressions for one full-scale, independent-arm paper campaign."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_missing_saa_lower_field_is_valid_zero_fallback(tmp_path,monkeypatch):
    from surgery import theory_regions as regions
    (tmp_path/'saa_bounds').mkdir()
    (tmp_path/'saa_bounds'/'one.json').write_text(
       json.dumps(dict(status='UNRESOLVED',error='forced timeout')))
    monkeypatch.setattr(regions,'scenarios',
                        lambda *a,**k:np.ones((5,1)))
    c=SimpleNamespace(n=1,booked=np.array([100.]))
    data=dict(X=np.zeros((1,1)),case_w=np.zeros(1))
    results=regions.build_problems(
       [('one',c,np.array([0]))],data,np.zeros((1,5)),np.zeros(5),
       {},tmp_path,stage_seconds=0.)
    assert len(results)==1
    assert results[0].lower==0.
    report=json.loads((tmp_path/'saa_preprocessing.json').read_text())
    assert report['unresolved_saa']==1


def test_case_feature_timeout_returns_structured_dependency(monkeypatch):
    from surgery import advice_comparison as advice
    c=SimpleNamespace(booked=np.array([100.]),group='TGH',monday='2012-01-02')
    advice._STATE=dict(data=dict(X=np.zeros((1,1)),case_w=np.zeros(1)),
                       names=[],seconds=.1)
    count=[]
    def fail(*a,**kw):
        count.append(True)
        raise RuntimeError('solver timed out')
    monkeypatch.setattr(advice,'certified',fail)
    ck,rows,rec=advice.baseline_job(('one',c,np.array([0])))
    assert ck=='one' and rec['status']=='UNRESOLVED'
    assert len(count)==2


def test_stat_plus_is_never_worse_than_matched_stat(tmp_path):
    from surgery.theory_stat import fit_stat_plus
    from surgery.advice_comparison import reviewed,LAM
    rng=np.random.default_rng(23)
    n=42
    H=rng.normal(size=(n,5))
    data=dict(X=np.zeros((n,1)),case_w=np.zeros(1),
              booked=np.full(n,110.),
              actual=np.full(n,110.)+3.*H[:,0])
    beta=np.zeros(5)
    output=tmp_path/'stat_plus.json'
    coef=fit_stat_plus(data,H,beta,seconds=5.,out=output)
    d0=reviewed(data['booked'],data['X']@data['case_w'])
    def loss(b):
        d=reviewed(data['booked'],d0-data['booked']+H@b)
        e=data['actual']-d
        return (1.75*np.maximum(e,0.).sum()+np.maximum(-e,0.).sum()
                +LAM/2*np.square(d-d0).sum())/n
    assert loss(coef)<=loss(beta)+1e-8
    assert output.exists()
    assert json.loads(output.read_text())['objective']>=0.


def test_evaluation_arms_have_independent_budgets(monkeypatch):
    from pilot import run_theory_full as runner
    c=SimpleNamespace(n=1,k=1,booked=np.array([100.]),actual=np.array([105.]))
    runner._EVAL_STATE=dict(base=np.array([110.]),H=np.zeros((1,5)),
         pools={},models={'STAT_PLUS':np.zeros(5)},seconds=1.,
         pw_seconds=1.,pw_nodes=10)
    monkeypatch.setattr(runner,'init_bank',
                        lambda *a,**kw:{k:np.full((3,1),120.)
                                        for k in ('A','B','C')})
    monkeypatch.setattr(runner,'deploy',
                        lambda *a,**kw:{0:(0,),1:(0,),2:(0,)})
    monkeypatch.setattr(runner,'evaluate',
                        lambda *a,**kw:dict(cost=15.,room_days_over_720=0,
                                           max_room_load=150.))
    monkeypatch.setattr(runner,'loads',
                        lambda *a,**kw:np.array([150.]))
    monkeypatch.setattr(runner,'expected_reduced',lambda *a,**kw:8.)
    a=runner._eval_arm_task(('fixture',c,np.array([0]),'CASE',30.))
    b=runner._eval_arm_task(('fixture',c,np.array([0]),'STAT_PLUS',30.))
    assert a[2]['status']==b[2]['status']=='COMPLETE'
    assert a[2]['method_wall_seconds']>=0
    assert b[2]['budget_seconds']==30.
    assert b[2]['solver_threads']==1
    skipped=runner._eval_arm_task(('fixture',c,np.array([0]),'CASE',0.))
    assert skipped[2]['status']=='UNRESOLVED'
    runner._EVAL_STATE=None


def test_paper_default_keeps_distinct_methods_and_separate_jobs():
    from pathlib import Path
    src=(Path(__file__).resolve().parents[1]/'pilot'/'run_theory_full.py').read_text()
    assert "STAT_PLUS,VF,SHARED,EG,OBS,ROBUST" in src
    assert "PER_WEEK_ROBUST,QFILL,VF_PW,SAA,HINDSIGHT" in src
    assert "pool.submit(_eval_arm_task,job)" in src
    assert "if arm=='SAA'" in src


def test_full_evaluation_schedules_independent_arms_and_fallback(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pilot import run_theory_full as runner

    class FakePool:
        def __init__(self,max_workers,**kw):
            self.pool=ThreadPoolExecutor(max_workers=max_workers)
        def __enter__(self):return self
        def __exit__(self,*a):self.pool.shutdown(wait=True)
        def submit(self,fn,*args):return self.pool.submit(fn,*args)

    monkeypatch.setattr(runner,'ProcessPoolExecutor',FakePool)
    def fake_eval(job):
        ck,c,rows,arm,limit=job
        if arm=='HINDSIGHT':
            return ck,arm,dict(complete=True,cost=5.,lower=5.,
                               status='OPTIMAL',method_wall_seconds=.1)
        if arm=='SAA':
            return ck,arm,dict(certified=True,lower=6.,upper=6.,seconds=.1,
                               assignment=[0],realized=6.,
                               model={'A':6.,'C':6.},method_wall_seconds=.1)
        if arm=='BOOKED' and ck=='late':
            return ck,arm,dict(status='UNRESOLVED',error='TIME_LIMIT',
                               method_wall_seconds=.2)
        return ck,arm,dict(status='COMPLETE',
              realized={'0':10.,'1':10.,'2':10.},
              model={'A':10.,'C':10.,'C_comonotone':10.},
              regime={'0':'LIGHT_UNIFORM'},hash_assignment_disagreement=0,
              hash_cost_disagreement=0,verification_seconds=.1,
              method_wall_seconds=.3)
    monkeypatch.setattr(runner,'_eval_arm_task',fake_eval)
    class C:pass
    components=[]
    for key,day in [('early','2013-01-07'),('late','2013-01-14')]:
        c=C();c.n=1;c.group='TGH';c.monday=day
        components.append((key,c,np.array([0 if key=='early' else 1])))
    data=dict(X=np.zeros((2,1)),case_w=np.zeros(1))
    ctx=dict(root=tmp_path,held=data,H_eval=np.zeros((2,5)),pools={},
             eval=components)
    args=SimpleNamespace(eval_arms=['CASE','BOOKED','SAA','HINDSIGHT'],
              eval_arm_seconds=30.,seconds=2.,pw_seconds=2.,
              pw_nodes=10,cores=2)
    result=runner.evaluate_all(args,ctx,models={})
    assert result['n_cases']==2
    assert result['arms']['BOOKED']['unresolved']==1
    assert result['fallback_CASE']['BOOKED']['fallback_components']==1
    assert result['fallback_CASE']['BOOKED']['all_case_mean_realized_per_case']==10.
    assert result['arms']['CASE']['unresolved']==0
    assert len(list((tmp_path/'full_evaluation'/'per_arm').glob('*.json')))==8
