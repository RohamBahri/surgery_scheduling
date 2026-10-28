"""Regression checks for explicit hash seeds and frozen cliff deployment."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pilot.paper_seed_audit import explicit_job,run_seed
from surgery.nonmonotone_learning import response,response_saturating
from surgery.response_grid import (
    TRAIN_SCENARIOS,Scenario,implemented_from_display)
from surgery.weekly_data import Week


def toy():
    c=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102]),("S1",),
        np.array([0,0]),np.array([0,0]),
        ((0,1),),((0,),),
        np.array([90.,130.]),np.array([110.,140.]),()).validate()
    return c


def test_seed_worker_passes_requested_seed_explicitly(monkeypatch):
    import surgery.anchored_weekly_solver as solver
    seen=[]
    class Fake:
        assignment=[0,0];complete=True;status="FAKE"
    def wrapped(*args,**kwargs):
        seen.append(kwargs["hash_seed"])
        return Fake()
    monkeypatch.setattr(solver,"solve_week",wrapped)
    c=toy()
    for seed in (0,1,2):
        key,rec=explicit_job(("toy",c,c.booked,12.,seed))
        assert key=="toy" and rec["hash_seed"]==seed
    assert seen==[0,1,2]


def test_seed_cache_contains_seed_provenance(tmp_path,monkeypatch):
    import surgery.anchored_weekly_solver as solver
    used=[]
    class Fake:
        assignment=[0,0];complete=True;status="FAKE"
    def wrapped(*args,**kwargs):
        used.append(kwargs["hash_seed"]);return Fake()
    monkeypatch.setattr(solver,"solve_week",wrapped)
    c=toy();items=[("toy",c,np.arange(2))]
    for seed in (0,1,2,1):
        result=run_seed(items,c.booked,tmp_path,"VF",seed,1,30.)
        assert result["toy"]["hash_seed"]==seed
    assert used==[0,1,2]


def test_nonmonotone_cliff_can_reduce_adoption_of_frozen_display():
    booked=np.array([100.,100.,100.])
    display=np.array([10.,30.,60.])
    s=Scenario("S","saturating",.8,20.)
    c=Scenario("C","nonmonotone",.8,20.)
    B=implemented_from_display(booked,display,s)
    C=implemented_from_display(booked,display,c)
    assert C[0]==pytest.approx(B[0])
    assert C[1]<B[1]
    assert C[2]<B[2]
    assert np.allclose(C,booked+response(display,.8,20.))


def test_cliff_followup_requires_completed_grid(tmp_path):
    cmd=[sys.executable,"-m","pilot.paper_followup",
         "--context",str(tmp_path/"missing.pkl"),
         "--out",str(tmp_path/"output")]
    p=subprocess.run(cmd,capture_output=True,text=True)
    assert p.returncode!=0
    assert "Missing completed-run input" in p.stderr


def test_cliff_scripts_expose_documented_args():
    for name in ("pilot.paper_seed_audit","pilot.paper_cliff_audit",
                 "pilot.paper_followup"):
        out=subprocess.run([sys.executable,"-m",name,"--help"],
                           capture_output=True,text=True,check=True)
        assert "--plan-seconds" in out.stdout
