"""Final single-method VF runner and hash-only replay regression tests."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from pilot import run_joint_vf as runner
from pilot import _joint_vf_engine as engine
from surgery.nonmonotone_learning import ResponseDC
from surgery.weekly_data import Week


def toy():
    c=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2,3]),("S1",),np.array([0,0,0]),
        np.array([0,0,1]),((0,1),),((0,),),
        np.array([100.,140.,90.]),np.array([120.,160.,80.]),()).validate()
    data=dict(booked=c.booked,actual=c.actual,case_id=c.case_ids,
              X=np.ones((3,77)))
    return c,dict(train=[("toy",c,np.arange(3))],
        eval=[("toy",c,np.arange(3))],fit=data,held=data)


def test_cli_defaults_are_the_final_hash_specification(tmp_path):
    context=tmp_path/"context.pkl"
    context.write_bytes(b"stub")
    a=runner.parse_args(["--context",str(context)])
    assert a.stage=="all" and a.turnover==0.
    assert a.accuracy_mu==2. and a.lambda_vf==.1
    assert a.train_tie=="primary"
    with pytest.raises(SystemExit):
        runner.parse_args(["--context",str(context),"--turnover","30"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--context",str(context),"--stage","seeds"])


def test_feature_schema_only_structural():
    from surgery.vf_features import feature_names
    assert len(feature_names("structural"))==6
    with pytest.raises(ValueError,match="Only"):
        feature_names("baseline")


def test_hash_only_training_replay_uses_booked_and_frozen_vf(
        tmp_path,monkeypatch):
    c,ctx=toy()
    folder=tmp_path/"run"
    folder.mkdir()
    response=ResponseDC(c.booked,.8,30.,180.,1.,link="saturating")
    raw=np.array([6.,3.,-2.])
    with (folder/"cases_train.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case_id","booked","actual","vf_raw","vf_reviewed"])
        for i in range(c.n):
            w.writerow([int(c.case_ids[i]),float(c.booked[i]),
                float(c.actual[i]),float(raw[i]),float(response.reviewed(raw)[i])])
    calls=[]
    class Cache:
        def __init__(self,*args):pass
        def solve(self,items,durations,tag,tie=False):
            calls.append((tag,tie,np.asarray(durations).copy()))
            return {"toy":dict(assignment=list(map(int,c.original)),
                               complete=True)}
    monkeypatch.setattr(engine,"PlannerCache",Cache)
    a=SimpleNamespace(out=folder,alpha=.8,h=30.,link="saturating",
                      cores=1,plan_seconds=1.)
    result=runner.replay_training(a,ctx)
    assert [z[0] for z in calls]==[
        "train_final_hash_BOOKED","train_final_hash_VF"]
    assert all(z[1] for z in calls)
    assert result["common_certified"]["cases"]==3
    assert "VF_minus_BOOKED" in result["common_certified"]
