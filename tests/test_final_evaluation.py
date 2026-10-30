"""Regression: final exploratory evaluation publishes only VF and BOOKED."""
import json

import numpy as np

from pilot import _joint_vf_engine as engine
from surgery.weekly_data import Week


def test_final_evaluation_uses_hash_only_deployment_and_two_methods(
        tmp_path,monkeypatch):
    b=np.array([100.,120.])
    c=Week("TGH","2013-01-07",((0,"OR1"),(0,"OR2")),
        np.array([1,2]),("S1",),np.array([0,0]),np.array([0,1]),
        ((0,1),),((0,),),b,np.array([110.,125.]),()).validate()
    ctx=dict(held=dict(booked=b,X=np.ones((2,77))),
             eval=[("week",c,np.arange(2))])
    (tmp_path/"vf_features.json").write_text(json.dumps(dict(
        schema="structural",mean=[0.]*6,std=[1.]*6)))
    selected=dict(case_coeff=[0.]*77,vf_theta=[0.]*83)
    args=type("Args",(),dict(alpha=.8,h=30.,cap=180.,link="saturating",
                            vf_features="structural"))()
    monkeypatch.setattr(engine,"five_features",
        lambda *a,**k:(np.zeros((2,6)),None,None))
    monkeypatch.setattr(engine,"save_case_table",lambda *a,**k:None)
    calls=[]
    class Cache:
        def solve(self,items,d,label,tie=False):
            calls.append((label,tie))
            return {"week":dict(assignment=[0,1],complete=True,
                                status="OPTIMAL")}
    engine.eval_2013(ctx,tmp_path,args,Cache(),selected)
    assert calls==[("eval_BOOKED",True),("eval_VF",True)]
    rows=json.loads((tmp_path/"evaluation_2013.json").read_text())["entries"]
    assert {r["method"] for r in rows}=={"BOOKED","VF"}
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["common_certified_cases"]==2
    assert "VF_minus_BOOKED" in summary["paired_cost_differences"]
    assert "CASE" not in summary
