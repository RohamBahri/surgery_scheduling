"""No-data tests of frozen-advice seed comparison and scenario finalization."""
import csv
import json
import numpy as np
import pytest

from pilot.seed_compare_policies import (paired_over_seeds,frozen_case_table)
from pilot.run_structural_finalization import decision
from surgery.nonmonotone_learning import ResponseDC
from surgery.response_grid import Scenario,implemented_from_display


def test_paired_seed_bootstrap_preserves_week_unit_and_exact_mean():
    cases=[2,3,5,4]
    weeks=["a","a","b","b"]
    costs={(0,"ref"):[10.,20.,30.,40.],
           (1,"ref"):[11.,19.,33.,38.],
           (0,"cand"):[9.,18.,29.,40.],
           (1,"cand"):[10.,18.,30.,36.]}
    result=paired_over_seeds(costs,cases,weeks,"cand","ref",
                             [0,1],draws=3000)
    assert result["per_seed"]["0"]==pytest.approx(-4/14)
    assert result["per_seed"]["1"]==pytest.approx(-7/14)
    assert result["mean_over_seeds"]==pytest.approx(-5.5/14)
    assert result["negative_seed_count"]==2
    assert result["better_calendar_weeks"]==2


def test_seed_decision_supports_positive_and_negative_evidence():
    v={"common_cases":990,"all_cases":1000,
       "comparisons":{"structural_minus_clean":{
           "mean_over_seeds":-.6,
           "paired_calendar_week_bootstrap_95":[-.8,-.2],
           "negative_seed_count":5}}}
    yes=decision(v)
    assert yes["selected_policy"]=="structural"
    v["comparisons"]["structural_minus_clean"]["negative_seed_count"]=3
    assert decision(v)["selected_policy"]=="clean"
    v["comparisons"]["structural_minus_clean"]["negative_seed_count"]=5
    v["comparisons"]["structural_minus_clean"]["paired_calendar_week_bootstrap_95"][1]=.2
    assert decision(v)["selected_policy"]=="clean"
    v["comparisons"]["structural_minus_clean"]["paired_calendar_week_bootstrap_95"][1]=-.1
    v["common_cases"]=900
    assert decision(v)["selected_policy"]=="clean"


def test_stored_nominal_displays_not_recanonicalized_by_new_true_response():
    booked=np.array([100.,180.,250.])
    raw=np.array([40.,-42.,10.])
    nominal=ResponseDC(booked,.8,30.,180.,1.,link="saturating")
    shown=nominal.display(raw)
    nominal_reviewed=nominal.reviewed(raw)
    np.testing.assert_allclose(
        implemented_from_display(booked,shown,
                                 Scenario("nominal","saturating",.8,30.)),
        nominal_reviewed)
    # In a different TRUE scenario the same display must be reused,
    # rather than displayed again using the new alpha/h.
    scenario=Scenario("new","nonmonotone",.5,20.)
    changed=implemented_from_display(booked,shown,scenario)
    assert np.any(abs(changed-nominal_reviewed)>1e-6)


def test_frozen_policy_table_checks_case_order_and_replay(tmp_path):
    folder=tmp_path/"clean";folder.mkdir()
    ids=np.array([13,14],int)
    b=np.array([120.,240.])
    x=np.array([[1.,1.],[1.,-1.]])
    actual=np.array([130.,260.])
    coeff=np.array([5.,-2.])
    raw_case=x@coeff
    vfraw=np.array([12.,-40.])
    model=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    policy=dict(method="JOINT_VF",link="saturating",
                alpha=.8,h=30.,vf_features="clean",case_coeff=coeff.tolist())
    (folder/"selected_policy.json").write_text(json.dumps(policy))
    (folder/"vf_features.json").write_text(json.dumps({"schema":"clean"}))
    with (folder/"cases_test.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("case_id","booked","actual","case_raw","case_display",
                    "case_reviewed","vf_raw","vf_display","vf_reviewed"))
        for i in range(2):
            w.writerow([ids[i],b[i],actual[i],raw_case[i],
                        model.display(raw_case)[i],model.reviewed(raw_case)[i],
                        vfraw[i],model.display(vfraw)[i],model.reviewed(vfraw)[i]])
    ctx=dict(held=dict(case_id=ids,booked=b,actual=actual,X=x))
    result=frozen_case_table(ctx,folder,"clean")
    np.testing.assert_allclose(result["vf"],model.reviewed(vfraw))
    with (folder/"cases_test.csv").open() as f:
        lines=f.readlines()
    (folder/"cases_test.csv").write_text(lines[0]+lines[2]+lines[1])
    with pytest.raises(AssertionError,match="order"):
        frozen_case_table(ctx,folder,"clean")


def test_seed0_archive_reproduction_uses_single_policy_cell(tmp_path):
    """Regression for KeyError: 0 after all five-seed solver jobs finished."""
    from pilot.seed_compare_policies import check_seed0_archive
    archive=tmp_path/"clean"
    archive.mkdir()
    (archive/"evaluation_2013.json").write_text(json.dumps({
        "entries":[
            {"component":"a","method":"VF","complete":True,"cost":12.5},
            {"component":"b","method":"VF","complete":True,"cost":7.5},
            {"component":"a","method":"CASE","complete":True,"cost":11.5},
        ]}))
    # Already cells[0,"VF_clean"], not a seed-indexed dict.
    cells=dict(cost={"a":12.5,"b":7.5},
               certified={"a":True,"b":True},
               assignment={"a":[0],"b":[0]})
    result=check_seed0_archive(cells,archive,"VF",["a","b"])
    assert result["verified_components"]==2
    assert result["changed_cost_components"]==0
    assert result["max_abs_cost_difference"]==0
    cells["cost"]["b"]=7.6
    result=check_seed0_archive(cells,archive,"VF",["a","b"])
    assert result["changed_cost_components"]==1
    assert result["max_abs_cost_difference"]==pytest.approx(.1)
    assert result["examples"][0][0]=="b"


def test_contract_accepts_only_documented_post_solve_reporting_hotfix(tmp_path):
    from pilot.run_structural_finalization import reconcile_contract
    log=tmp_path/"explicit_five_seeds.log"
    log.write_text("Traceback in check_seed0_archive\nKeyError: 0\n")
    old=dict(version="structural_finalization_v1",
             context_sha256="same-data",plan_seconds=180.,seed_set=[0,1,2,3,4],
             implementation_sha256={"seed_compare_policies.py":"old",
                                    "anchored_weekly_solver.py":"unchanged"})
    new={**old,"implementation_sha256":
         {"seed_compare_policies.py":"fixed",
          "anchored_weekly_solver.py":"unchanged"}}
    updated=reconcile_contract(old,new,log)
    assert len(updated["implementation_history"])==1
    assert updated["implementation_sha256"]["seed_compare_policies.py"]=="fixed"
    assert reconcile_contract(updated,new,log)==updated
    with pytest.raises(ValueError,match="Unexpected implementation"):
        reconcile_contract(old,new,log,seed_result_exists=True)
    with pytest.raises(ValueError,match="Unexpected implementation"):
        reconcile_contract(old,new,log,behavior_result_exists=True)
    changed_solver={**new,"implementation_sha256":
         {"seed_compare_policies.py":"fixed",
          "anchored_weekly_solver.py":"changed"}}
    with pytest.raises(ValueError,match="Unexpected implementation"):
        reconcile_contract(old,changed_solver,log)
    changed_science={**new,"plan_seconds":240.}
    with pytest.raises(ValueError,match="Frozen finalization"):
        reconcile_contract(old,changed_science,log)
    log.write_text("Different error")
    with pytest.raises(ValueError,match="Unexpected implementation"):
        reconcile_contract(old,new,log)
