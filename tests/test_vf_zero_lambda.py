"""Zero-L1 VF uses same MM-pDCA and fully frozen original B/C comparison."""
import json
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from pilot import extend_vf_zero_lambda as vf0
from pilot import run_response_scenario_grid as grid
from pilot import extend_response_grid_evaluation as previous
from surgery.response_grid import TRAIN_SCENARIOS
from surgery.weekly_planner import evaluate


def toy():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1",),
        np.zeros(3,dtype=int),np.zeros(3,dtype=int),
        ((0,1),),((0,),),
        np.array([190.,180.,160.]),np.array([195.,175.,170.]),()).validate()


class Cache:
    def __init__(self): self.calls=[]
    def solve(self,items,d,tag,tie=False):
        self.calls.append((tag,bool(tie)))
        return {key:dict(assignment=[0]*week.n,
                         complete=True,status="OPTIMAL",bound=0.)
                for key,week,rows in items}


def test_fit_one_preserves_case_coefficients_and_uses_zero_penalty(
        tmp_path,monkeypatch):
    week=toy()
    items=[("toy",week,np.arange(3))]
    data=dict(booked=week.booked.copy(),actual=week.actual.copy(),
              X=np.column_stack((np.ones(3),np.arange(3))))
    ctx=dict(train=items,fit=data)
    case=np.array([0.,1.])
    scenario=TRAIN_SCENARIOS[0]
    folder=tmp_path/"prior"/"training"/scenario.name
    folder.mkdir(parents=True)
    grid.save(folder/"vf_features.json",
              dict(mean=[0.]*5,std=[1.]*5))
    original=dict(case_coeff=case.tolist(),
                  vf_theta=[1.,0.,0.,0.,0.],lambda_vf=.319)
    def features(ctx,items,raw,model,cache,stage,scaler=None):
        assert stage.startswith("train_")
        H=np.column_stack((np.ones(3),np.zeros((3,4))))
        return H,None,{"toy":dict(assignment=[0,0,0])}
    def mm(items,booked,H,raw,model,actual,*,root,lam,optimizer,
           initial_libraries,iterations,radius,pdca_steps,max_cuts,
           backend,solve_plans):
        assert lam==0. and optimizer=="pdca"
        assert np.array_equal(raw,data["X"]@case)
        assert initial_libraries=={"toy":[[0,0,0],[0,0,0]]}
        assert radius==64 and iterations==30
        return dict(theta=[2.,0.,0.,0.,0.],
                    status="RELATIVE_IMPROVEMENT",history=[],
                    libraries={"toy":[[0,0,0]]})
    def direct(*args):
        return dict(certified_components=1,total_components=1,
                    mean_cost_per_component=10.,total_cost=10.,
                    unresolved=[])
    monkeypatch.setattr(grid,"five_features",features)
    monkeypatch.setattr(vf0,"mm_vf",mm)
    monkeypatch.setattr(vf0.ray,"score_direct_training",direct)
    args=SimpleNamespace(vf_outers=30,vf_pdca_steps=50,cuts=160)
    output=tmp_path/"new"
    cache=Cache()
    booked={"toy":dict(assignment=[0,0,0])}
    fitted=vf0.fit_one(ctx,tmp_path/"prior",output,cache,scenario,
                       original,booked,args)
    assert fitted["lambda_vf"]==0.
    assert fitted["original_lambda_vf"]==pytest.approx(.319)
    assert fitted["case_coeff"]==original["case_coeff"]
    assert fitted["zero_theta"]==[2.,0.,0.,0.,0.]
    assert fitted["direct_training"]["mean_cost_per_component"]==10.
    before=len(cache.calls)
    again=vf0.fit_one(ctx,tmp_path/"prior",output,cache,scenario,
                      original,booked,args)
    assert again==fitted
    assert len(cache.calls)==before


def test_zero_lambda_grid_evaluation_reuses_all_previous_cells(
        tmp_path):
    w=toy()
    items=[("toy",w,np.arange(3))]
    data=dict(booked=w.booked.copy(),actual=w.actual.copy(),
              case_id=w.case_ids.copy())
    ctx=dict(eval=items,held=data)
    root=tmp_path/"old"
    out=root/"vf_zero_extension"
    base_cost=evaluate(w,[0,0,0],w.actual)
    for sc in previous.ALL_SCENARIOS:
        for arm in grid.METHODS:
            target=(root/"evaluation_cells" if sc.link=="saturating"
                   else root/"response_extension"/"evaluation_cells")
            file=target/sc.name/(arm+".json")
            file.parent.mkdir(parents=True,exist_ok=True)
            entry=dict(component="toy",week=w.monday,site=w.group,
                 cases=3,method=arm,evaluation_scenario=sc.name,
                 cost=float(base_cost["cost"]),
                 idle_minutes=float(base_cost["idle"]),
                 overtime_minutes=float(base_cost["overtime"]),
                 mae_sum=3.,complete=True,planner_status="OPTIMAL")
            grid.save(file,dict(method=arm,evaluation=vars(sc),
                               entries=[entry]))
    original_path=root/"evaluation_cells"/previous.B_SCENARIOS[0].name/(
                    grid.METHODS[0]+".json")
    pristine=original_path.read_bytes()
    displays={arm:np.array([0.,25.,-25.])
              for arm in vf0.EXTRA_ARMS}
    cache=Cache()
    vf0.evaluate_new(ctx,out,cache,displays)
    assert len(cache.calls)==48
    vf0.summarize(ctx,root,out)
    result=json.loads((out/"summary.json").read_text())
    assert result["vf_lambda"]==0.
    assert len(result["per_scenario"])==12
    assert len(result["methods"])==14
    assert result["globally_certified_cases"]==3
    for sc in result["per_scenario"].values():
        assert len(sc["cost_per_case"])==14
        assert len(sc["contrasts"])==16
    with (out/"cost_grid_global_matched.csv").open(newline="") as f:
        table=list(csv.DictReader(f))
    assert len(table)==12
    assert "VF_ZERO__SAT_A80_H30" in table[0]
    vf0.evaluate_new(ctx,out,cache,displays)
    assert len(cache.calls)==48
    assert pristine==original_path.read_bytes()


def test_runner_cli_help():
    import subprocess,sys
    result=subprocess.run(
        [sys.executable,"pilot/extend_vf_zero_lambda.py","--help"],
        capture_output=True,text=True,timeout=25)
    assert result.returncode==0,result.stderr
    assert "--plan-seconds" in result.stdout
    assert "--vf-outers" in result.stdout


def test_zero_penalty_coefficient_box_not_ray_tuning():
    assert vf0.LAMBDA==0.
    assert vf0.RADIUS==64.
    assert len(vf0.EXTRA_ARMS)==4
    assert len(vf0.METHODS)==14
    assert all(label.startswith("VF_ZERO__") for label in vf0.EXTRA_ARMS)
