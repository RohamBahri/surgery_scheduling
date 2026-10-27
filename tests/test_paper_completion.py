"""Unit-level contracts for final joint VF audit and response analysis.

No private dataset or Gurobi required for these small validation tests.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.vf_features import (
    booked_plan_features, feature_names, standardize_features)
from surgery.response_grid import TRAIN_SCENARIOS,implemented_from_display
from pilot.paper_response_report import (
    POLICIES,NOMINAL,candidate_regret,selected_policy,stable_uniform,
    heterogeneous_duration)


def toy():
    week=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1","S2"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,1),(1,)),((0,),(0,)),
        np.array([150.,200.,460.]),np.array([160.,190.,455.]),()).validate()
    items=[("toy",week,np.arange(3))]
    plans={"toy":{"assignment":[0,0,1]}}
    history=np.array([[.1,.2],[.3,.4],[.5,.6]])
    return week,items,plans,history


def test_remove_exact_duplicate_plan_features_without_changing_interactions():
    w,items,plans,H=toy()
    full=booked_plan_features(items,w.booked,plans,H,"baseline")
    clean=booked_plan_features(items,w.booked,plans,H,"clean")
    np.testing.assert_array_equal(clean,full[:,[2,3,4]])
    assert len(feature_names("clean"))==3
    assert len(feature_names("case_only"))==0
    empty=booked_plan_features(items,w.booked,plans,H,"case_only")
    assert empty.shape==(3,0)
    transformed,scale=standardize_features(empty)
    assert transformed.shape==(3,0)
    assert len(scale[0])==0
    assert len(scale[1])==0
    w.actual[:]=[999.,300.,1.]
    np.testing.assert_array_equal(
        clean,booked_plan_features(items,w.booked,plans,H,"clean"))


def test_regret_minimax_selected_on_train_not_any_reference_cost():
    p=list(POLICIES)
    matrix={
        "s1":{p[0]:1.,p[1]:2.,p[2]:2.,p[3]:4.},
        "s2":{p[0]:6.,p[1]:2.,p[2]:3.,p[3]:8.},
        "s3":{p[0]:1.,p[1]:2.,p[2]:2.,p[3]:4.}
    }
    assert selected_policy(matrix,"minimax")==p[1]
    assert selected_policy(matrix,"average")==p[1]
    r=candidate_regret(matrix)
    assert r["s2"][p[1]]==pytest.approx(0.)
    assert r["s1"][p[1]]==pytest.approx(1.)
    assert NOMINAL in p


def test_surgeon_response_profiles_are_frozen_and_deterministic():
    week,items,_,_=toy()
    b=week.booked.copy()
    display=np.array([20.,-30.,40.])
    for profile in ("equal_four","nominal_majority"):
        x=heterogeneous_duration(items,b,display,profile)
        y=heterogeneous_duration(items,b,display,profile)
        np.testing.assert_array_equal(x,y)
        assert (x>=1.).all()
        for i in range(week.n):
            sid=week.surgeon_ids[int(week.surgeon[i])]
            u=stable_uniform((str(week.group),str(sid)))
            if profile=="equal_four":
                j=min(int(4*u),3)
            else:
                j=2 if u<.75 else (0 if u<.8333333333333333
                                 else (1 if u<.9166666666666666 else 3))
            expected=implemented_from_display(
                b[i:i+1],display[i:i+1],TRAIN_SCENARIOS[j])
            assert x[i]==pytest.approx(expected[0])


def test_completion_cli_available_without_private_context():
    scripts=("pilot.run_paper_completion","pilot.paper_response_report",
             "pilot.paper_seed_audit","pilot.paper_theory_audit")
    for script in scripts:
        p=subprocess.run([sys.executable,"-m",script,"--help"],
                         capture_output=True,text=True,check=True)
        assert "--help" in p.stdout


def test_missing_context_cannot_silently_start_full_run(tmp_path):
    cmd=[sys.executable,"-m","pilot.run_paper_completion",
         "--context",str(tmp_path/"missing.pkl"),
         "--out",str(tmp_path/"output")]
    p=subprocess.run(cmd,capture_output=True,text=True)
    assert p.returncode!=0
    assert "Missing private context" in p.stderr


def test_installed_gurobi_certifies_small_primary_and_hash_tie():
    pytest.importorskip("gurobipy")
    from surgery.anchored_weekly_solver import solve_week
    from surgery.weekly_planner import evaluate
    week,_,_,_=toy()
    a=solve_week(week,week.booked,seconds=30.,threads=1,tie=True,hash_seed=0)
    b=solve_week(week,week.booked,seconds=30.,threads=1,tie=True,hash_seed=1)
    assert a.assignment is not None and b.assignment is not None
    assert a.complete and b.complete
    assert a.hash_seed==0 and b.hash_seed==1
    ca=evaluate(week,a.assignment,week.booked)["cost"]
    cb=evaluate(week,b.assignment,week.booked)["cost"]
    assert abs(ca-cb)<=max(a.primary_tolerance,b.primary_tolerance)+1e-6
