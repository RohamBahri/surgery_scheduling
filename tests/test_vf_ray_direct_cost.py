"""One-dimensional direct-cost ray training and fully matched B/C replay."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pilot import extend_vf_ray_grid as ray
from pilot import run_response_scenario_grid as original
from pilot import extend_response_grid_evaluation as previous
from surgery.response_grid import TRAIN_SCENARIOS, mapping, implemented_from_display
from surgery.weekly_planner import evaluate


def toy_week():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1",),
        np.zeros(3,dtype=int),np.zeros(3,dtype=int),
        ((0,1),),((0,),),
        np.array([190.,180.,160.]),np.array([195.,175.,170.]),()).validate()


def test_ray_grid_stays_in_box_and_always_contains_CASE_and_VF():
    theta=np.array([60.,10.,0.,0.,0.])
    steps=ray.ray_points(theta,2.,11)
    assert 0. in steps and 1. in steps
    assert steps.max()==pytest.approx(64./60.)
    assert np.max(np.abs(steps[-1]*theta))<=64.+1e-8
    assert ray.ray_points(np.zeros(5),2.,11).tolist()==[0.,1.]
    with pytest.raises(ValueError):
        ray.ray_points(np.array([70.,0.]),2.,11)
    with pytest.raises(ValueError):
        ray.ray_points(np.array([1.]),.9,11)


def test_certified_ray_selects_lowest_training_direct_cost_on_grid(
        tmp_path,monkeypatch):
    week=toy_week()
    items=[("toy",week,np.arange(3))]
    B=next(s for s in TRAIN_SCENARIOS if s.alpha==.8 and s.h==30.)
    ctx=dict(train=items,fit=dict(booked=week.booked.copy(),
             actual=week.actual.copy()))
    theta=np.array([5.,0.,0.,0.,0.])
    H=np.column_stack((np.ones(3),np.zeros((3,4))))
    base=np.zeros(3)
    booked=week.booked
    def training_features(ctx,root,scenario,policy,cache):
        assert scenario==B
        return booked,base,H,theta,mapping(booked,B)
    monkeypatch.setattr(ray,"training_features",training_features)
    class FakeCache:
        def __init__(self): self.calls=[]
        def solve(self,items,reviews,tag,tie=False):
            assert tie
            self.calls.append(np.asarray(reviews,float).copy())
            # A direct-cost step-dependent planner. Realized costs
            # change with the chosen schedule, not predicted cost.
            flag=float(np.mean(reviews-booked))
            choice=[0,1,1] if flag>=2.0 else [0,0,0]
            return {"toy":dict(complete=True,assignment=choice,
                               status="OPTIMAL",bound=0.)}
    cache=FakeCache()
    root=tmp_path/"old"
    output=tmp_path/"new"
    arg=SimpleNamespace(s_max=2.,grid_points=5)
    selection=ray.optimize_ray(ctx,root,output,cache,B,dict(),arg)
    vals=selection["training_ledger"]
    assert {0.,1.}.issubset(set(float(row["s"]) for row in vals))
    assert selection["grid_complete"]
    assert selection["endpoints_certified"]
    assert selection["never_worse_than_case_and_vf_on_certified_training"]
    best=min(vals,key=lambda row:(row["total_cost"],row["s"]))
    assert selection["s"]==pytest.approx(best["s"])
    assert selection["direct_train_cost_per_component"]==pytest.approx(
        best["mean_cost_per_component"])
    first=len(cache.calls)
    second=ray.optimize_ray(ctx,root,output,cache,B,dict(),arg)
    assert len(cache.calls)==first
    assert second["s"]==selection["s"]


def test_unresolved_planner_points_cannot_be_marked_grid_optimal(
        tmp_path,monkeypatch):
    week=toy_week()
    B=TRAIN_SCENARIOS[0]
    H=np.column_stack((np.ones(3),np.zeros((3,4))))
    def features(*args):
        return week.booked,np.zeros(3),H,np.array([2.,0.,0.,0.,0.]),mapping(
            week.booked,B)
    monkeypatch.setattr(ray,"training_features",features)
    ctx=dict(train=[("toy",week,np.arange(3))])
    class Partial:
        def solve(self,items,reviews,tag,tie=False):
            # Uncertified at s=1, valid at s=0 and some other points.
            changed=np.isclose(np.mean(reviews-week.booked),
                               B.alpha*2.,atol=.001)
            return {"toy":dict(complete=not changed,assignment=[0,0,0],
                               status="TIME_LIMIT" if changed else "OPTIMAL")}
    picked=ray.optimize_ray(ctx,tmp_path,tmp_path/"ray",Partial(),B,{},SimpleNamespace(
        s_max=2.,grid_points=5))
    assert not picked["grid_complete"]
    assert not picked["endpoints_certified"]
    assert not picked["never_worse_than_case_and_vf_on_certified_training"]
    assert not picked["continuous_ray_optimality_certified"]


def test_training_cost_uses_realized_durations_not_prediction_or_l1():
    week=toy_week()
    sample=[("toy",week,np.arange(3))]
    class Cache:
        def solve(self,items,durations,tag,tie=False):
            assert tie
            return {"toy":dict(complete=True,assignment=[0,1,1],
                               status="OPTIMAL")}
    record=ray.score_direct_training(sample,
                       np.array([500.,500.,500.]),Cache(),"foo")
    real=evaluate(week,[0,1,1],week.actual)["cost"]
    assert record["total_cost"]==pytest.approx(real)
    assert record["mean_cost_per_component"]==pytest.approx(real)


def test_frozen_ray_replay_is_independent_of_evaluation_response(
        tmp_path,monkeypatch):
    week=toy_week()
    data=dict(X=np.array([[1.,-1.],[1.,0.],[1.,1.]]),
              booked=week.booked.copy(),actual=week.actual.copy(),
              case_id=week.case_ids.copy())
    ctx=dict(held=data,eval=[("toy",week,np.arange(3))])
    policies={}
    choices={}
    for train in TRAIN_SCENARIOS:
        key=train.name
        policies[key]=dict(case_coeff=[0.,10.],
                           vf_theta=[6.,0.,0.,0.,0.])
        choices[key]=dict(s=1.5)
        folder=tmp_path/"old"/"training"/key
        folder.mkdir(parents=True)
        original.save(folder/"vf_features.json",
                      dict(mean=[0.]*5,std=[1.]*5))
    def fake_features(ctx,items,raw,m,cache,stage,scaler=None):
        return np.ones((3,5)),None,None
    monkeypatch.setattr(original,"five_features",fake_features)
    displays=ray.frozen_ray_displays(ctx,tmp_path/"old",
                tmp_path/"ray",None,policies,choices)
    assert len(displays)==4
    assert max(abs(v).max() for v in displays.values())<=60.
    C=next(s for s in previous.C_SCENARIOS if s.h==20. and s.alpha==.8)
    B=next(s for s in previous.B_SCENARIOS if s.h==20. and s.alpha==.8)
    for display in displays.values():
        outC=implemented_from_display(data["booked"],display,C)
        outB=implemented_from_display(data["booked"],display,B)
        assert np.all(outC>0)
        assert np.all(outB>0)
    again=ray.frozen_ray_displays(ctx,tmp_path/"old",
                tmp_path/"ray",None,policies,choices)
    assert all(np.array_equal(again[k],displays[k]) for k in displays)


def test_ray_canonical_policy_stays_s_max_one_endpoints():
    a=np.array([5.,4.,-3.,-2.,1.])
    g=ray.ray_points(a,1.,13)
    assert len(g)==13
    assert g[0]==0. and g[-1]==1.



def test_all_twelve_scenarios_full_fourteen_arm_matched_report(
        tmp_path,monkeypatch):
    """No original arm is recomputed when ray-only evaluation is appended."""
    week=toy_week()
    item=[("toy",week,np.arange(3))]
    data=dict(booked=week.booked.copy(),actual=week.actual.copy(),
              case_id=week.case_ids.copy())
    ctx=dict(eval=item,held=data)
    root=tmp_path/"old"
    extroot=root/"response_extension"
    out=root/"vf_ray_extension"
    base=evaluate(week,[0,0,0],week.actual)
    for sc in previous.ALL_SCENARIOS:
        for arm in original.METHODS:
            target=(root/"evaluation_cells" if sc.link=="saturating"
                    else extroot/"evaluation_cells")
            file=target/sc.name/(arm+".json")
            file.parent.mkdir(parents=True,exist_ok=True)
            row=dict(component="toy",site=week.group,week=week.monday,
                     cases=3,method=arm,evaluation_scenario=sc.name,
                     complete=True,planner_status="OPTIMAL",
                     cost=float(base["cost"]),
                     idle_minutes=float(base["idle"]),
                     overtime_minutes=float(base["overtime"]),
                     mae_sum=0.)
            original.save(file,dict(method=arm,evaluation=vars(sc),
                                    entries=[row]))
    calls=[]
    class FakeCache:
        def solve(self,items,durations,tag,tie=False):
            calls.append((tag,tie))
            return {"toy":dict(complete=True,assignment=[0,0,0],
                               status="OPTIMAL",bound=0.)}
    frozen={"VF_RAY__"+s.name:np.array([0.,10.,-10.])
            for s in TRAIN_SCENARIOS}
    ray.evaluate_ray(ctx,root,out,FakeCache(),frozen)
    assert len(calls)==48
    report=ray.summarize(ctx,root,out)
    saved=json.loads((out/"summary.json").read_text())
    assert saved["global_matched_cases"]==3
    assert saved["global_matched_components"]==1
    assert len(saved["scenarios"])==12
    assert len(saved["methods"])==14
    for sc in saved["scenarios"].values():
        assert len(sc["globally_matched_methods"])==14
        assert len(sc["paired_contrasts"])==16
    with (out/"cost_grid_global_matched.csv").open(newline="") as handle:
        table=list(csv.DictReader(handle))
    assert len(table)==12
    assert "VF_RAY__SAT_A80_H30" in table[0]
    original_bytes=(root/"evaluation_cells"/previous.B_SCENARIOS[0].name/
                    "VF__SAT_A80_H30.json").read_bytes()
    ray.evaluate_ray(ctx,root,out,FakeCache(),frozen)
    assert len(calls)==48
    assert original_bytes==(
       root/"evaluation_cells"/previous.B_SCENARIOS[0].name/
       "VF__SAT_A80_H30.json").read_bytes()
