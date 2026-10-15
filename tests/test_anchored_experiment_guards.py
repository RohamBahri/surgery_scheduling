"""Clean-run exactness guards: optional aspirations versus actual deployment."""
import numpy as np
import pytest

from surgery import anchored_training as train
from surgery.anchored_core import key
from surgery.weekly_planner import evaluate
from test_anchored_core import component


def test_optional_vf_timeout_yields_labeled_feasible_reference(monkeypatch):
    c=component()
    original=list(map(int,c.original))
    def fake(cmp,d,*,seconds,threads,hindsight=False):
        if hindsight:
            return dict(complete=True,assignment=original,
                        realized_cost=evaluate(cmp,original,cmp.actual)['cost'])
        if np.array_equal(d,cmp.booked):
            return dict(complete=True,status='2',assignment=original,
                        realized_cost=evaluate(cmp,original,cmp.actual)['cost'],
                        predicted_cost=evaluate(cmp,original,d)['cost'])
        return dict(complete=False,status='HASH_UNIQUENESS_UNRESOLVED',assignment=original)
    monkeypatch.setattr(train,'deployed',fake)
    methods={'BOOKED':c.booked.copy(),'VF':c.booked+4.}
    row=train._week_task((c,methods,10.,1))
    record=row[key(c)]
    assert record['complete']
    assert record['oracle_cost']==evaluate(c,original,c.actual)['cost']
    assert record['references'][0]['reference_optimality_certified'] is True
    fallback=record['references'][1]
    assert fallback['source']=='VF'
    assert fallback['type']=='uncertified-feasible-aspiration'
    assert fallback['reference_optimality_certified'] is False
    assert fallback['assignment']==original


def test_incomplete_hindsight_cannot_enter_regret_denominator(monkeypatch):
    c=component()
    monkeypatch.setattr(train,'deployed',lambda *args,**kwargs:dict(
        complete=False,status='9',assignment=list(map(int,c.original))))
    row=train._week_task((c,{'VF':c.booked.copy()},10.,1))
    assert not row[key(c)]['complete']
    assert 'hindsight' in row[key(c)]['failure']


def test_trial_with_unverified_deployment_is_rejected_not_accepted(tmp_path,monkeypatch):
    c=component()
    idx={int(i):p for p,i in enumerate(c.case_ids)}
    X=np.zeros((c.n,74));X[:,0]=1.
    refs={key(c):{'complete':True,'oracle_cost':0.,'references':[
        dict(source='BOOKED',type='shared-policy',
             assignment=list(map(int,c.original)),beta=c.booked.tolist(),
             realized_cost=evaluate(c,c.original,c.actual)['cost'],
             predicted_cost=evaluate(c,c.original,c.booked)['cost'])]}}
    data={'X':X,'booked':c.booked,'actual':c.actual,'case_id':c.case_ids}
    calls=[]
    def verified(output,name,weeks,index,X,booked,w,alpha,h,*args,**kw):
        calls.append((name,kw.get('allow_incomplete',False)))
        if name.startswith('trial'):
            return {key(c):{'complete':False,'status':'HASH_UNIQUENESS_UNRESOLVED'}}
        return {key(c):{'complete':True,
              'assignment':list(map(int,c.original)),
              'realized_cost':evaluate(c,c.original,c.actual)['cost'],
              'predicted_cost':evaluate(c,c.original,c.booked)['cost']}}
    monkeypatch.setattr(train,'verify_policy',verified)
    monkeypatch.setattr(train,'pdca',lambda *a,**kw:dict(
        w=[5.]+[0.]*73,inner=[],restricted_objective=0.))
    monkeypatch.setattr(train,'enrich_cuts',lambda *a,**kw:dict(
        all_closed=True,added=0,closed=1,unresolved=0,tested=1,untested=0))
    result=train.learn(tmp_path,[c],idx,data,refs,np.zeros(74),.8,30.,
                       outer=1,iterations=1,closure_rounds=1,workers=1)
    assert result['complete'] is False
    assert result['stopping']=='unresolved_candidate_deployment'
    assert result['outer']==0
    assert not result['accepted']
    assert all(not row['verified'] for row in result['history'][0]['candidates'])
    assert all(flag for name,flag in calls if name.startswith('trial'))


def test_hash_sensitivity_scores_frozen_weights_without_retraining(tmp_path, monkeypatch):
    import json
    import os
    from types import SimpleNamespace
    import run_weekly_anchored as runner
    count=54
    case_ids=np.arange(1,count+1)
    frames={'validation':dict(
        case_id=case_ids, X=np.column_stack((np.ones(count),np.zeros((count,73)))),
        booked=np.full(count,100.))}
    weeks=[SimpleNamespace(group='TGH',monday=f'week_{i:02d}',
                           case_ids=np.array([i+1])) for i in range(count)]
    policies={k:np.zeros(74) for k in ('CASE','LASSO','VF','VF_GAP')}
    model=dict(complete=True,training_cases=18705,w=[0.]*74,accepted=[])
    checkpoint=tmp_path/'anchored_progress.json'
    checkpoint.write_text(json.dumps(model))
    monkeypatch.setattr(runner,'initialize',lambda args:
        (None,frames,None,None,None,policies,dict(hash_seed=0)))
    monkeypatch.setattr(runner,'load_weeks',lambda *a: weeks)
    monkeypatch.setattr(runner,'check_week_cases',lambda *a:
        {int(i):int(i)-1 for i in case_ids})
    attempts=[]
    def fake_week(task):
        week,durations,*_=task
        h=int(os.environ['SURGERY_ANCHORED_HASH_SEED'])
        attempts.append(h)
        return [dict(group=week.group,monday=week.monday,method=name,
                     complete=True,cases=1,regret=float(h),realized_cost=10.+h,
                     room_days_over_960=0,room_days_over_1440=0)
                for name in durations]
    monkeypatch.setattr(runner,'_eval_week',fake_week)
    args=SimpleNamespace(output=tmp_path,site='all',hash_seed=0,
                         sensitivity_split='validation',alpha=.8,h=30.,
                         workbook=tmp_path/'fake.xlsx',seconds=10.,threads=1,
                         workers=1)
    runner.sensitivity(args)
    results=json.loads((tmp_path/'hash_sensitivity_validation_summary.json').read_text())
    assert results['status']=='FROZEN-POLICY HASH SENSITIVITY'
    assert results['seeds']['1']['ANCHORED']['regret_per_case']==pytest.approx(1.)
    assert results['seeds']['4']['ANCHORED']['regret_per_case']==pytest.approx(4.)
    assert len(attempts)==4*54
    runner.sensitivity(args)  # must resume without recomputing any completed week
    assert len(attempts)==4*54
    assert json.loads(checkpoint.read_text())==model
    assert os.environ['SURGERY_ANCHORED_HASH_SEED']=='0'
