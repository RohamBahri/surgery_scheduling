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
    c=SimpleNamespace(group='TGH',monday='2013-02-04',n=1,
        case_ids=np.array([42]),original=np.array([0]),
        actual=np.array([100.]))
    from surgery import weekly_planner
    monkeypatch.setattr(weekly_planner,'evaluate',lambda *a,**k:dict(cost=10.))
    ctx=dict(root=tmp_path,eval=[('one',c,np.array([0]))])
    ctxfile=tmp_path/'selection'/'context.pkl'
    ctxfile.parent.mkdir(parents=True)
    import pickle
    with ctxfile.open('wb') as handle:pickle.dump(ctx,handle)
    args=SimpleNamespace(out=tmp_path)
    runner.summarize(args)
    summary=json.loads((tmp_path/'selection'/'selection_summary.json').read_text())
    assert summary['status']=='COMPLETE_COMPARISON_WITH_FALLBACK'
    assert summary['provisional_main_method'] is None
    assert summary['selection_reason']=='SHARED_PLUS_NOT_SUFFICIENTLY_EVALUATED'
    assert summary['case_fallback']['SHARED_PLUS']['mean_cost_per_case']==10.
    assert summary['case_fallback']['SHARED_PLUS']['original_assignment_fallback_cases']==1
    assert summary['arms']['CASE']['exact_coverage']==0


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


def test_eg_time_limited_dual_interval_is_not_discarded(monkeypatch):
    from surgery import theory_selection as selection
    from surgery.theory_exact import Certificate
    from surgery import theory_exact
    monkeypatch.setattr(theory_exact,'expected_gap_mip',lambda *a,**kw:
                        Certificate(8.,8.,11.,False,'TIME_LIMIT',1.))
    lb,ub,complete=selection._gap_one((
         SimpleNamespace(n=1),np.array([120.]),np.array([100.]),
         np.array([[130.]]),10.))
    from surgery.advice_comparison import LAM
    penalty=LAM/2*(20.**2)
    assert lb==8.+penalty
    assert ub==11.+penalty
    assert not complete


def test_training_paired_subset_never_counts_missing_as_zero(tmp_path):
    from pilot import run_theory_full as full
    folder=tmp_path/'scores';folder.mkdir()
    items=[]
    for i in range(50):
        ck=f'component_{i}'
        c=SimpleNamespace(n=2)
        items.append((ck,c,np.array([2*i,2*i+1])))
        # Last component is deliberately unresolved: exactly 98% coverage.
        sample=[dict(complete=True,value=10.) if i<49 else
                dict(complete=False,value=None),
                dict(complete=True,value=8.) if i<49 else
                dict(complete=False,value=None)]
        (folder/(ck+'_shard_0000.json')).write_text(json.dumps(
           dict(indices=[0,1],results=sample)))
    accepted=full._selection_paired_candidate(folder,items,[0,1])
    assert accepted is not None
    assert accepted['paired_cases']==98
    assert accepted['total_cases']==100
    assert accepted['observed_training_difference']==-98.
    # An additional unresolved component must make the gate fail.
    file=folder/'component_48_shard_0000.json'
    file.write_text(json.dumps(dict(indices=[0,1],
       results=[dict(complete=False,value=None)]*2)))
    assert full._selection_paired_candidate(folder,items,[0,1]) is None
