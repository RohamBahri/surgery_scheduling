"""Saturation-only B response grid correctness and pipeline regressions."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import numpy as np
import pytest

from surgery.response_grid import (
    TRAIN_SCENARIOS,EVAL_SCENARIOS,check_grid,mapping,
    advice_for_policy,implemented_from_display,
    naive_display,verify_same_scenario,evaluation_policy_labels)
from surgery.nonmonotone_learning import response_saturating


def toy():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
         np.array([101,102,103]),("S1",),
         np.zeros(3,dtype=int),np.zeros(3,dtype=int),
         ((0,1),),((0,),),
         np.array([190.,180.,160.]),np.array([195.,175.,170.]),()
         ).validate()


def test_all_six_evaluation_scenarios_are_saturation_only():
    assert check_grid()
    assert len(TRAIN_SCENARIOS)==4
    assert len(EVAL_SCENARIOS)==6
    assert {(s.alpha,s.h) for s in TRAIN_SCENARIOS}=={
        (.5,30.),(.5,60.),(.8,30.),(.8,60.)}
    assert {(s.alpha,s.h) for s in EVAL_SCENARIOS}=={
        (.5,20.),(.5,30.),(.5,60.),
        (.8,20.),(.8,30.),(.8,60.)}
    assert all(s.link=="saturating" for s in
               (*TRAIN_SCENARIOS,*EVAL_SCENARIOS))
    assert len(evaluation_policy_labels())==8


def test_wide_h60_has_effect_only_if_frozen_display_exceeds_h30():
    booked=np.array([100.,100.,100.])
    narrow=next(s for s in TRAIN_SCENARIOS
                if s.alpha==.8 and s.h==30.)
    wide=next(s for s in TRAIN_SCENARIOS
              if s.alpha==.8 and s.h==60.)
    raw=np.array([20.,45.,90.])
    shown30=verify_same_scenario(raw,booked,narrow)
    shown60=verify_same_scenario(raw,booked,wide)
    assert np.array_equal(shown30,[20.,30.,30.])
    assert np.array_equal(shown60,[20.,45.,60.])
    eval30=implemented_from_display(booked,shown60,narrow)
    eval60=implemented_from_display(booked,shown60,wide)
    assert np.array_equal(eval30,[116.,124.,124.])
    assert np.array_equal(eval60,[116.,136.,148.])
    assert np.array_equal(
        implemented_from_display(booked,shown30,narrow),
        implemented_from_display(booked,shown30,wide))


def test_cross_scenario_evaluation_reacts_to_frozen_advice_not_raw_score():
    booked=np.array([130.,150.,1.5])
    trained=next(s for s in TRAIN_SCENARIOS
                 if s.alpha==.8 and s.h==60.)
    other=next(s for s in EVAL_SCENARIOS
               if s.alpha==.5 and s.h==20.)
    raw=np.array([80.,-42.,-150.])
    display=advice_for_policy(raw,booked,trained)
    assert np.abs(display).max()<=60.
    got=implemented_from_display(booked,display,other)
    expected=np.maximum(booked+response_saturating(display,.5,20.),1.)
    assert np.allclose(got,expected)
    # Incorrect recanonicalization under evaluation alpha/h would differ.
    wrong=mapping(booked,other).reviewed(raw)
    assert not np.allclose(got,wrong)
    assert got[-1]>=1.


def test_naive_display_is_response_unaware():
    features=np.array([[1.,1.],[1.,-1.],[1.,0.]])
    scores=naive_display(features,np.array([0.,220.]))
    assert np.array_equal(scores,[180.,-180.,0.])


def test_content_addressed_cache_deduplicates_identical_plans(
        tmp_path,monkeypatch):
    from pilot import run_response_scenario_grid as runner
    c=toy()
    items=[("toy",c,np.arange(3))]
    durations=c.booked.copy()
    calls=[]
    def fake(job):
        calls.append(job[2].copy())
        return job[0],dict(assignment=[0,0,0],cost=1.,bound=1.,
                           complete=True,status="OPTIMAL")
    monkeypatch.setattr(runner,"_plan_job",fake)
    cache=runner.ContentCache(tmp_path,1,180.)
    cache.solve(items,durations,"grid_A",tie=True)
    cache.solve(items,durations,"grid_B",tie=True)
    assert len(calls)==1
    cache.solve(items,durations+5,"grid_B",tie=True)
    assert len(calls)==2
    cache.solve(items,durations,"grid_B",tie=False)
    assert len(calls)==3


def test_full_four_training_x_six_evaluation_grid_smoke(
        tmp_path,monkeypatch):
    from pilot import run_response_scenario_grid as grid
    c=toy()
    data=dict(X=np.array([[1.,-1.],[1.,0.],[1.,1.]]),
              booked=c.booked.copy(),actual=c.actual.copy(),
              case_id=np.array([101,102,103]))
    items=[("toy",c,np.arange(c.n))]
    ctx=dict(train=items,eval=items,fit=data,held=data,
             H=np.ones((c.n,5)),H_eval=np.ones((c.n,5)))
    class Cache:
        def solve(self,items,durations,tag,tie=False):
            return {key:dict(assignment=[0,0,0],complete=True,
                       status="OPTIMAL",cost=1.,bound=1.)
                    for key,c,rows in items}
    def fitted_case(ctx,items,model,folder,steps,cuts,backend,radius,**kw):
        assert model.link=="saturating"
        assert kw["lambda_scale"]==1.0
        grid.save(folder/"case_policy.json",dict(
            lambda_case=.4,alpha=model.alpha,h=model.h,link=model.link))
        return np.array([0.,2.])
    def features(ctx,items,raw,model,cache,stage,scaler=None):
        n=len(raw)
        H=np.column_stack([np.ones(n),np.arange(n),
                           np.ones(n)*.3,np.zeros(n),np.ones(n)*.2])
        assert (stage.startswith("train") or stage.startswith("eval"))
        return H,(np.zeros(5),np.ones(5)),Cache().solve(items,None,"draft")
    def vf(items,booked,H,raw,model,actual,*,root,lam,optimizer,**kw):
        assert optimizer=="pdca"
        assert lam==pytest.approx(.24)
        assert model.link=="saturating"
        root.mkdir(parents=True,exist_ok=True)
        grid.save(root/"vf_library_progress.json",dict(theta=[0.,0.,0.,0.,0.]))
        return dict(theta=[1.,0.,0.,0.,0.],status="RELATIVE_IMPROVEMENT",
                    history=[])
    monkeypatch.setattr(grid,"case_fit",fitted_case)
    monkeypatch.setattr(grid,"five_features",features)
    monkeypatch.setattr(grid,"mm_vf",vf)
    monkeypatch.setattr(grid,"train_lambda",lambda H,n,**kw:.24)
    monkeypatch.setattr(grid,"fit_naive",
      lambda ctx,root: (grid.save(root/"naive_policy.json",
                      dict(weights=[0.,1.])),np.array([0.,1.]))[1])
    opts=SimpleNamespace(case_steps=4,cuts=20,vf_outers=2,
                         vf_pdca_steps=3)
    cache=Cache()
    naive=grid.fit_naive(ctx,tmp_path)
    booked=cache.solve(items,c.booked,"booked",tie=True)
    fits={s.name:grid.fit_family(ctx,opts,tmp_path,cache,s,booked)
          for s in TRAIN_SCENARIOS}
    displays=grid.freeze_displays(ctx,tmp_path,cache,naive,fits)
    assert set(displays)==set(grid.METHODS)-{"BOOKED"}
    grid.evaluate_grid(ctx,tmp_path,cache,displays)
    report=json.loads((tmp_path/"summary.json").read_text())
    assert len(report["scenarios"])==6
    assert set(report["methods"])==set(grid.METHODS)
    for name in report["scenarios"]:
        block=report["scenarios"][name]
        assert block["common_certified_cases"]==3
        assert len(block["methods"])==10
        assert "VF__SAT_A80_H60_minus_CASE__SAT_A80_H60" in (
            block["paired_cost_differences"])
    with (tmp_path/"cost_grid.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert "VF__SAT_A50_H30" in rows[0]
    with (tmp_path/"frozen_advice"/"evaluation_displays.csv").open(
         newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==3
    assert "CASE__SAT_A80_H60" in rows[0]


def test_grid_runner_help():
    import subprocess,sys
    entry=Path(__file__).resolve().parents[1]/"pilot"/"run_response_scenario_grid.py"
    result=subprocess.run([sys.executable,str(entry),"--help"],
                          capture_output=True,text=True,timeout=20,
                          cwd=entry.parents[1])
    assert result.returncode==0,result.stderr
    assert "--vf-outers" in result.stdout
    assert "--plan-seconds" in result.stdout



def test_naive_matches_asymmetric_prediction_loss_on_train_only(tmp_path,
                                                                monkeypatch):
    from pilot import run_response_scenario_grid as grid
    c=toy()
    residual=np.array([8.,13.,18.])
    design=np.column_stack((np.ones(3),np.array([-1.,0.,1.])))
    booked=c.booked.copy()
    actual=booked+residual
    ctx=dict(train=[("toy",c,np.arange(3))],
             fit=dict(X=design,booked=booked,actual=actual))
    monkeypatch.setattr(grid,"train_lambda",lambda X,n,**kw:0.)
    coeff=grid.fit_naive(ctx,tmp_path)
    assert np.allclose(design@coeff,residual,atol=1e-6)
    saved=json.loads((tmp_path/"naive_policy.json").read_text())
    assert saved["nominal_response_used"] is False
    assert saved["objective"]==pytest.approx(0.,abs=1e-7)
    assert saved["solver_status"]==0



def test_grid_contract_can_be_reloaded_without_false_setting_change(tmp_path):
    from pilot import run_response_scenario_grid as grid
    source=tmp_path/"context.pkl"
    source.write_bytes(b"frozen demo input")
    folder=tmp_path/"grid"
    folder.mkdir()
    args=SimpleNamespace(context=source,case_steps=200,vf_outers=30,
         vf_pdca_steps=50,cuts=160,plan_seconds=180.,cores=15)
    grid.contract(args,folder)
    before=json.loads((folder/"run_contract.json").read_text())
    grid.contract(args,folder)
    after=json.loads((folder/"run_contract.json").read_text())
    assert before==after
    assert len(before["arms"])==10
    assert len(before["train"])==4
    assert len(before["evaluation"])==6
