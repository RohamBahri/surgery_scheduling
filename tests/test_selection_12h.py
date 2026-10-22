"""Release gates for the bounded paper-method-selection experiment."""
import json
from types import SimpleNamespace

import numpy as np


def test_stage_budget_preserves_test_and_finishes_before_deadline():
    from pilot.run_selection_12h import STAGES,ARMS
    assert sum(seconds for _,seconds in STAGES)<=11.5*3600
    names=[name for name,_ in STAGES]
    assert names.index('freeze')<names.index('evaluate')<names.index('report')
    assert dict(STAGES)['evaluate']>=3*3600
    assert {'SHARED_BASE','SHARED_PLUS','VF_PLUS','GAP_PLUS','ANCHORED_PLUS',
            'STAT_PLUS','CASE','BOOKED'}==set(ARMS)


def test_anchor_sampler_preserves_large_components_and_no_array_comparisons():
    from surgery.theory_selection import _sample
    class C:pass
    rows=[]
    for i in range(38):
        c=C()
        c.n=i+1
        c.surgeon=np.array([0])
        c.allowed=((0,1),)
        rows.append((str(i),c,np.array([i])))
    chosen=_sample(rows,limit=20)
    assert len(chosen)==20
    assert len({x[0] for x in chosen})==20
    assert '37' in {x[0] for x in chosen}


def test_frozen_reporting_no_case_outcomes_no_inference(tmp_path,monkeypatch):
    from pilot import run_selection_12h as runner
    c=SimpleNamespace(group='TGH',monday='2013-02-04',n=1)
    ctx=dict(eval=[('one',c,np.array([0]))])
    ctxfile=tmp_path/'selection'/'context.pkl'
    ctxfile.parent.mkdir(parents=True)
    import pickle
    with ctxfile.open('wb') as handle:pickle.dump(ctx,handle)
    args=SimpleNamespace(out=tmp_path)
    runner.summarize(args)
    summary=json.loads((tmp_path/'selection'/'selection_summary.json').read_text())
    assert summary['status']=='INSUFFICIENT_CASE_COVERAGE'
    assert summary['provisional_main_method'] is None
    assert summary['case_fallback']['SHARED_PLUS']['mean_cost_per_case'] is None


def test_policy_provenance_and_untrained_status(tmp_path):
    from pilot import run_selection_12h as runner
    args=SimpleNamespace(out=tmp_path)
    runner.record_policy(args,'GAP_PLUS',None,'NO_EG_VERIFIED_PROPOSAL',
                         source='EG-interval-screen')
    assert runner.get_policy(args,'GAP_PLUS') is None
    meta=json.loads((tmp_path/'selection'/'policies'/'GAP_PLUS.json').read_text())
    assert meta['status']=='NO_EG_VERIFIED_PROPOSAL'
    runner.record_policy(args,'SHARED_PLUS',np.arange(5.),'TRAINED_WITH_ACCEPTED_UPDATES')
    assert np.array_equal(runner.get_policy(args,'SHARED_PLUS'),np.arange(5.))


def test_direct_training_uses_one_declared_primary_hash(monkeypatch):
    monkeypatch.setenv('SURGERY_HASH_SEEDS','0')
    # The launcher chooses the environment before importing the planner.
    from pilot import run_selection_12h as runner
    assert runner.full.SEEDS==(0,)
