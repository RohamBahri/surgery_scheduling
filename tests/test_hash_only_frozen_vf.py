"""Frozen hash-only deployment uses trained VF without invoking training/CASE."""
import json
from types import SimpleNamespace

import numpy as np

from pilot import run_joint_vf as runner
from surgery.weekly_data import Week


def test_hash_only_reuses_frozen_durations_and_does_not_train(
        tmp_path,monkeypatch):
    b=np.array([120.,150.])
    y=np.array([125.,170.])
    wk=Week("TGH","2013-01-07",((0,"OR1"),(0,"OR2")),
        np.array([1,2]),("S1",),np.array([0,0]),np.array([0,1]),
        ((0,1),),((0,),),b,y,()).validate()
    items=[("toy",wk,np.arange(2))]
    context={"eval":items}
    a=SimpleNamespace(out=tmp_path,turnover=0.,accuracy_mu=2.,
                      cores=1,plan_seconds=300.,alpha=.8,h=30.,
                      link="saturating")
    # Frozen artifacts from a preceding run; the diagnostic should
    # never invoke VF training or the squared-load solver.
    (tmp_path/"cases_test.csv").write_text("available\n")
    def frozen(*args):
        return b,{},{"BOOKED":b,"VF":np.array([115.,145.])}
    monkeypatch.setattr(runner,"frozen_test_durations",frozen)
    calls=[]
    def fake_run(items,d,folder,label,seed,cores,seconds,*,balance):
        calls.append((label,balance,seed))
        return {"toy":dict(assignment=[0,1],complete=True,status="OPTIMAL")}
    monkeypatch.setattr(runner,"run_seed",fake_run)
    from surgery import weekly_planner
    with monkeypatch.context() as m:
        m.setattr(weekly_planner,"RELAX_REQUIRED_DAYS",True)
        answer=runner.audit_hash_only(a,context)
    assert calls==[("hash_only_BOOKED",False,0),
                   ("hash_only_VF",False,0)]
    assert answer["vf_retrained"] is False
    assert answer["tie_rule"]=="primary_optimum_then_hash"
    assert answer["common_certified_cases"]==2
    assert answer["balanced_vs_hash_same_certified_sample"] is None
    assert (tmp_path/"hash_only_vf_comparison.json").is_file()


def test_hash_only_does_not_accept_turnover(tmp_path):
    a=SimpleNamespace(turnover=30.,out=tmp_path)
    import pytest
    with pytest.raises(ValueError,match="turnover 0"):
        runner.audit_hash_only(a,{"eval":[]})
