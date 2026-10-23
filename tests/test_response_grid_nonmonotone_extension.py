"""The C extension must preserve finished B training and evaluator artifacts."""
from __future__ import annotations
import csv
import json
import pickle
import sys

import numpy as np
import pytest

from pilot import extend_response_grid_evaluation as ext
from pilot import run_response_scenario_grid as original
from surgery.response_grid import (
    TRAIN_SCENARIOS,EVAL_SCENARIOS as B_SCENARIOS,
    implemented_from_display,response_saturating)
from surgery.nonmonotone_learning import response
from surgery.weekly_planner import evaluate


def toy_week():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
         np.array([101,102,103]),("S1",),
         np.zeros(3,dtype=int),np.zeros(3,dtype=int),
         ((0,1),),((0,),),
         np.array([190.,180.,160.]),np.array([195.,175.,170.]),()
         ).validate()


def test_C_scenarios_match_B_alpha_h_but_are_distinct():
    assert len(ext.C_SCENARIOS)==6
    assert len(ext.ALL_SCENARIOS)==12
    assert {(s.alpha,s.h) for s in ext.C_SCENARIOS}=={
        (s.alpha,s.h) for s in B_SCENARIOS}
    assert all(s.link=="nonmonotone" for s in ext.C_SCENARIOS)
    assert all(s.link=="saturating" for s in B_SCENARIOS)
    booked=np.array([100.,100.])
    display=np.array([50.,-50.])
    c=next(s for s in ext.C_SCENARIOS if s.alpha==.8 and s.h==20.)
    b=next(s for s in B_SCENARIOS if s.alpha==.8 and s.h==20.)
    assert np.allclose(implemented_from_display(booked,display,c),
                       [110.,90.])
    assert np.allclose(implemented_from_display(booked,display,b),
                       [116.,84.])
    c50=next(s for s in ext.C_SCENARIOS if s.alpha==.5 and s.h==20.)
    assert np.allclose(implemented_from_display(booked,display,c50),
                       booked)


def test_smoke_preserves_frozen_policies_B_cells_and_uses_cached_plans(
        tmp_path,monkeypatch):
    week=toy_week()
    data=dict(
        X=np.array([[1.,0.],[1.,1.],[1.,-1.]]),
        booked=week.booked.copy(),actual=week.actual.copy(),
        case_id=week.case_ids.copy())
    items=[("toy",week,np.arange(week.n))]
    ctx=dict(train=items,eval=items,fit=data,held=data)
    context=tmp_path/"context.pkl"
    with context.open("wb") as handle:
        pickle.dump(ctx,handle)
    root=tmp_path/"prior_B"
    root.mkdir()
    args=["--context",str(context),"--out",str(root),
          "--cores","1","--plan-seconds","180"]
    from argparse import Namespace
    opt=Namespace(context=context,out=root,cores=1,plan_seconds=180.,
                  case_steps=200,vf_outers=30,vf_pdca_steps=50,cuts=160)
    original.contract(opt,root)
    original.save(root/"naive_policy.json",dict(weights=[0.,1.]))
    policy_bank={}
    for scenario in TRAIN_SCENARIOS:
        policy_bank[scenario.name]=dict(
            scenario=vars(scenario),case_coeff=[0.,1.],
            vf_theta=[0.,0.,0.,0.,0.])
    original.save(root/"selected_policies.json",
         dict(response_link="SATURATING_B",policies=policy_bank))
    folder=root/"frozen_advice"
    folder.mkdir()
    advice=folder/"evaluation_displays.csv"
    displayed={m:np.array([50.,-50.,10.])
               for m in original.METHODS[1:]}
    # Canonical B policies cannot exceed their training h.
    for m in displayed:
        if "__" in m:
            train=next(s for s in TRAIN_SCENARIOS
                       if s.name==m.split("__",1)[1])
            displayed[m]=np.clip(displayed[m],-train.h,train.h)
    with advice.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["case_id","booked","actual",*original.METHODS[1:]])
        for i in range(week.n):
            writer.writerow([int(week.case_ids[i]),week.booked[i],
                 week.actual[i],*(displayed[m][i]
                 for m in original.METHODS[1:])])
    metrics=evaluate(week,[0]*week.n,week.actual)
    for scenario in B_SCENARIOS:
        p=root/"evaluation_cells"/scenario.name
        p.mkdir(parents=True,exist_ok=True)
        for method in original.METHODS:
            line=dict(component="toy",week=week.monday,site=week.group,
               cases=week.n,method=method,
               evaluation_scenario=scenario.name,
               training_scenario=None,cost=float(metrics["cost"]),
               idle_minutes=float(metrics["idle"]),
               overtime_minutes=float(metrics["overtime"]),mae_sum=0.,
               complete=True,planner_status="OPTIMAL")
            original.save(p/(method+".json"),
                 dict(method=method,evaluation=vars(scenario),entries=[line]))
    snapshot=(
       (root/"run_contract.json").read_bytes(),
       (root/"selected_policies.json").read_bytes(),
       advice.read_bytes(),
       (root/"evaluation_cells"/B_SCENARIOS[0].name/
        "NAIVE.json").read_bytes())
    invoked=[]
    def optimizer(job):
        invoked.append(job[2].copy())
        return job[0],dict(assignment=[0]*job[1].n,cost=1.,
                   complete=True,status="OPTIMAL",bound=1.)
    monkeypatch.setattr(original,"_plan_job",optimizer)
    def forbidden(*args,**kwargs):
        raise AssertionError("No old training, display rebuilding or B replay")
    monkeypatch.setattr(original,"fit_naive",forbidden)
    monkeypatch.setattr(original,"fit_family",forbidden)
    monkeypatch.setattr(original,"freeze_displays",forbidden)
    monkeypatch.setattr(original,"evaluate_grid",forbidden)
    monkeypatch.setattr(sys,"argv",["extend_response_grid_evaluation.py",*args])
    ext.main()
    extension=root/"response_extension"
    assert len(list((extension/"evaluation_cells").glob("*/*.json")))==60
    assert (extension/"cost_grid_global_matched.csv").exists()
    outcome=json.loads((extension/"summary.json").read_text())
    assert outcome["primary_evaluation"]=="NONMONOTONE_C"
    assert len(outcome["scenarios"])==12
    assert outcome["all_scenarios_common_certified_cases"]==3
    assert set(outcome["B_vs_C_global_matched"])=={
        s.name for s in B_SCENARIOS}
    assert snapshot==(
       (root/"run_contract.json").read_bytes(),
       (root/"selected_policies.json").read_bytes(),
       advice.read_bytes(),
       (root/"evaluation_cells"/B_SCENARIOS[0].name/
        "NAIVE.json").read_bytes())
    # Some C duration vectors are identical, content-addressed cache dedups.
    first_count=len(invoked)
    ext.main()
    assert len(invoked)==first_count
    C=next(s for s in ext.C_SCENARIOS if s.alpha==.8 and s.h==20.)
    example=json.loads((extension/"evaluation_cells"/C.name/
                       "NAIVE.json").read_text())["entries"][0]
    expected=implemented_from_display(week.booked,
                 displayed["NAIVE"],C)
    assert example["mae_sum"]==pytest.approx(
        float(np.abs(week.actual-expected).sum()))
    assert example["mae_sum"]!=pytest.approx(
        float(np.abs(week.actual-implemented_from_display(
            week.booked,displayed["NAIVE"],
            next(s for s in B_SCENARIOS if
                 s.alpha==.8 and s.h==20.))).sum()))


def test_extension_cli_help():
    import subprocess
    cmd=[sys.executable,
         "pilot/extend_response_grid_evaluation.py","--help"]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
    assert result.returncode==0,result.stderr
    assert "--out" in result.stdout



def test_nonmonotone_agrees_with_saturation_below_threshold():
    from surgery.response_grid import Scenario
    booked=np.array([1.,4.,30.,100.,250.])
    displayed=np.array([-30.,-20.,0.,20.,30.])
    for alpha in (.5,.8):
        sat=Scenario("B","saturating",alpha,30.)
        non=Scenario("C","nonmonotone",alpha,30.)
        assert np.allclose(
            implemented_from_display(booked,displayed,sat),
            implemented_from_display(booked,displayed,non),
            rtol=0,atol=1e-10)
    # Mismatch can occur for the response-unaware NAIVE policy even when
    # the scenario matches a particular CASE/VF training alpha,h.
    far=np.array([150.])
    assert (implemented_from_display(np.array([100.]),far,
            Scenario("B","saturating",.8,30.))[0]==124.)
    assert (implemented_from_display(np.array([100.]),far,
            Scenario("C","nonmonotone",.8,30.))[0]==100.)
