"""Single retained structural-VF pipeline: unit and regression contracts."""
import csv
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from pilot import run_joint_vf as runner
from pilot import _joint_vf_engine as engine
from surgery.nonmonotone_learning import ResponseDC
from surgery.response_grid import Scenario, implemented_from_display
from surgery.weekly_data import Week


def toy():
    week=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2,3]),("S1",),np.array([0,0,0]),np.array([0,0,1]),
        ((0,1),),((0,),),
        np.array([100.,140.,90.]),np.array([120.,160.,80.]),()).validate()
    rows=np.arange(3)
    data=dict(booked=week.booked,actual=week.actual,
              case_id=week.case_ids,X=np.ones((3,77)))
    return week,dict(train=[("toy",week,rows)],eval=[("toy",week,rows)],
                     fit=data,held=data,H=np.ones((3,2)),H_eval=np.ones((3,2)))


def test_one_public_cli_and_configurable_behaviour(tmp_path):
    file=tmp_path/"context.pkl";file.write_bytes(b"stub")
    args=runner.parse_args(["--context",str(file),
        "--stage","train","--alpha",".5","--h","20",
        "--link","nonmonotone","--lambda-vf",".12",
        "--out",str(tmp_path/"new")])
    assert args.alpha==.5 and args.h==20
    assert args.link=="nonmonotone" and args.lambda_vf==.12
    cli=runner.engine_args(args,"all")
    assert cli[cli.index("--vf-features")+1]=="structural"
    assert cli[cli.index("--alpha")+1]=="0.5"
    assert cli[cli.index("--h")+1]=="20.0"


def test_no_other_model_family_or_provisional_feature_schemas():
    from surgery.vf_features import feature_names
    assert len(feature_names("structural"))==6
    with pytest.raises(ValueError,match="Only"):
        feature_names("baseline")
    # Only the structural representation is admitted by the engine parser.
    import subprocess,sys
    process=subprocess.run([sys.executable,"-m","pilot._joint_vf_engine",
         "--vf-features","baseline"],capture_output=True,text=True)
    assert process.returncode!=0
    assert "invalid choice" in process.stderr


def test_structural_features_do_not_solve_provisional_plan():
    c,ctx=toy()
    class Refuse:
        def solve(self,*args,**kwargs):
            raise AssertionError("Provisional schedule would change policy inputs")
    fit,scale,plan=engine.five_features(ctx,ctx["train"],Refuse(),
                                        "train",schema="structural")
    assert plan is None and fit.shape==(3,6)


def test_replay_user_script_reuses_training_plan_tags(tmp_path,monkeypatch):
    week,ctx=toy()
    folder=tmp_path/"run";folder.mkdir()
    model=ResponseDC(week.booked,.8,30.,180.,1.,link="saturating")
    case_raw=np.array([2.,5.,-8.])
    vf_raw=np.array([6.,3.,-2.])
    with (folder/"cases_train.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case_id","booked","actual","case_raw","case_reviewed",
                    "vf_raw","vf_reviewed"])
        for i in range(3):
            w.writerow([week.case_ids[i],week.booked[i],week.actual[i],
                        case_raw[i],model.reviewed(case_raw)[i],
                        vf_raw[i],model.reviewed(vf_raw)[i]])
    called=[]
    class Cache:
        def __init__(self,*args):pass
        def solve(self,items,duration,tag,tie=False):
            called.append((tag,tie,np.asarray(duration).copy()))
            return {"toy":dict(assignment=list(map(int,week.original)),
                               complete=True)}
    monkeypatch.setattr(engine,"PlannerCache",Cache)
    args=SimpleNamespace(out=folder,alpha=.8,h=30.,
                         cores=1,plan_seconds=1.)
    result=runner.replay_training(args,ctx)
    assert [x[0] for x in called]==[
        "train_case_draft","train_final_vf_tie"]
    assert all(x[1] for x in called)
    assert result["common_certified"]["cases"]==3
    assert (folder/"training_replay_comparison.json").exists()


def test_seed_week_bootstrap_uses_one_week_unit_for_all_seeds():
    roster={"a":dict(week="w1",cases=2),"b":dict(week="w2",cases=3)}
    cells={}
    for seed in [0,1]:
        for label,values in {"BOOKED":[20.,30.],
                              "CASE":[19.,28.],
                              "VF":[18.+seed,27.+seed]}.items():
            cells[seed,label]={k:dict(cost=values[i],complete=True)
                               for i,k in enumerate(roster)}
    result=runner.paired_weeks(cells,roster,("BOOKED","CASE","VF"),[0,1])
    assert result["common_cases"]==5
    assert result["comparisons"]["VF_minus_CASE"]["seed_mean"]==-1./5
    assert result["comparisons"]["VF_minus_CASE"]["per_seed"]["0"]==-2./5
    assert result["comparisons"]["VF_minus_CASE"]["per_seed"]["1"]==0.0


def test_frozen_display_responds_to_true_scenario_not_refit():
    b=np.array([120.,200.])
    response=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    display=response.display(np.array([60.,-20.]))
    scenario=Scenario("stress","nonmonotone",.5,20.)
    actual=implemented_from_display(b,display,scenario)
    assert actual.shape==(2,)
    assert not np.allclose(actual,response.reviewed([60.,-20.]))
