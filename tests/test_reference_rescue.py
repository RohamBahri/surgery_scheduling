"""Reference rescue never drops the original completeness or tie rules."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import rescue_anchored_references as rescue
from surgery.anchored_core import key
from surgery.weekly_planner import evaluate
from test_anchored_core import component


def test_rescue_caches_complete_subsolves_and_retries_only_failure():
    c=component()
    methods={'BOOKED':c.booked.copy(),'BOX_CLIP':c.booked+1,
             'CASE':c.booked-1}
    calls=[]
    failing={'once':True}

    def fake_solver(comp,duration,*,seconds,threads,hindsight=False):
        label='hindsight' if hindsight else next(
            k for k,v in methods.items() if np.array_equal(v,duration))
        calls.append(label)
        if label=='BOX_CLIP' and failing['once']:
            failing['once']=False
            return {'complete':False,'status':'MOVE_TIE_9'}
        z=c.original.astype(int).tolist()
        return dict(complete=True,status='OPTIMAL',assignment=z,
                    realized_cost=evaluate(c,z,c.actual)['cost'],
                    predicted_cost=evaluate(c,z,duration)['cost'])
    cache={}
    updates=[]
    row,failure=rescue.cached_reference_for_component(
        c,np.arange(c.n),methods,cache,200.,2,
        solve=fake_solver,on_update=lambda:updates.append(True))
    assert row is None and failure=='BOX_CLIP: MOVE_TIE_9'
    assert calls==['hindsight','BOOKED','BOX_CLIP']
    assert set(cache['references'])=={'BOOKED'}
    row,failure=rescue.cached_reference_for_component(
        c,np.arange(c.n),methods,cache,1800.,4,
        solve=fake_solver,on_update=lambda:updates.append(True))
    assert failure is None
    assert calls==['hindsight','BOOKED','BOX_CLIP','BOX_CLIP','CASE']
    assert row['complete'] and row['cases']==c.n
    assert [r['source'] for r in row['references']]==list(methods)
    assert row['references'][1]['type']=='box-aspiration'
    assert row['references'][0]['type']=='shared-policy'
    assert row['oracle_cost']==evaluate(c,c.original,c.actual)['cost']
    assert len(updates)>=4


def test_rescue_rejects_unverifiable_completed_reference():
    c=component()
    def invalid_solver(comp,duration,*,seconds,threads,hindsight=False):
        return dict(complete=True,status='OPTIMAL',
                    assignment=c.original.astype(int).tolist(),
                    realized_cost=-100.,predicted_cost=-100.)
    with pytest.raises(AssertionError,match='inconsistent'):
        rescue.cached_reference_for_component(
            c,None,{'BOOKED':c.booked},{},300.,1,
            solve=invalid_solver)


def test_rescue_writes_only_completed_component_and_preserves_identity(
        tmp_path,monkeypatch):
    c=component();group='TGH'
    root=tmp_path/'baseline';root.mkdir()
    (root/'learning_settings.json').write_text(json.dumps({'alpha':.8,'h':30.}))
    output=tmp_path/'anchored_ALL_extended';output.mkdir()
    settings=dict(site='all',gamma=2.,lambda_l1=0.,prox=.15,
                  coef_prox=1e-5,pDCA_iterations=20,closure_rounds=3,
                  max_outer=15,exact_sep_batch=384,
                  baseline_models_sha256={'CASE':'a','VF':'b'})
    (output/'anchored_settings.json').write_text(json.dumps(settings))
    ck=key(c)
    (output/'anchored_references.json').write_text(
        json.dumps({ck:{'complete':False,'failure':'CASE: MOVE_TIE_9'}}))
    data={'X':np.ones((c.n,74)),
          'booked':c.booked.copy(), 'actual':c.actual.copy(),
          'case_id':c.case_ids.copy()}
    index={int(i):k for k,i in enumerate(c.case_ids)}
    calls=[]
    def fake_initialize(args):
        assert args.site=='all' and args.exact_sep_batch==384
        return None,None,[c],index,data,{'CASE':np.zeros(74)},settings
    monkeypatch.setattr(rescue,'initialize',fake_initialize)
    def fake_solver(comp,duration,*,seconds,threads,hindsight=False):
        calls.append(('hindsight' if hindsight else 'policy',seconds,threads))
        z=c.original.astype(int).tolist()
        return dict(complete=True,status='OPTIMAL',assignment=z,
                    realized_cost=evaluate(c,z,c.actual)['cost'],
                    predicted_cost=evaluate(c,z,duration)['cost'])
    monkeypatch.setattr(rescue,'deployed',fake_solver)
    args=SimpleNamespace(output=output,artifacts=root,
            workbook=tmp_path/'unused.xlsx',gap_model=tmp_path/'gap.json',
            component=None,seconds=1800.,threads=4)
    assert rescue.rescue(args)
    saved=json.loads((output/'anchored_references.json').read_text())
    assert saved[ck]['complete']
    assert [x['source'] for x in saved[ck]['references']]==[
        'BOOKED','BOX_CLIP','BOX_PAD','CASE']
    assert (output/'anchored_settings.json').read_text()==json.dumps(settings)
    assert calls and all(v[1:]==(1800.,4) for v in calls)
    n=len(calls)
    assert rescue.rescue(args)
    assert len(calls)==n
    assert rescue_uses_stable_fingerprint()


def rescue_uses_stable_fingerprint():
    import run_weekly_anchored as runner
    return ('rescue_anchored_references.py' not in runner.SOURCE_FILES)
