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
        return dict(complete=False,status='MOVE_TIE_9',assignment=original)
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
            return {key(c):{'complete':False,'status':'CANONICAL_TIE_9'}}
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
    assert result['complete']
    assert not result['accepted']
    assert all(not row['verified'] for row in result['history'][0]['candidates'])
    assert all(flag for name,flag in calls if name.startswith('trial'))
