"""Exact VF-GAP identities, smooth gradients, and checkpoint safety."""
import json

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate, solve_week
from surgery.planner import display_and_duration
from legacy.weekly.run_weekly_gap import evaluate_week_gap
from surgery.gap_learning import (
    oracle_week, array_hash, week_key, ScheduleLibrary,
    certify, certified_score, room_cost_and_derivative, train_gap
)


def locked_example():
    week=Week(
        "TWH-day","2011-11-28",((0,"DS101"),(2,"DS102")),
        np.array([100,101,102]),("A","B"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,),(1,)),((0,),(2,)),
        np.array([175.,220.,165.]),
        np.array([210.,180.,155.]),()
    )
    return week.validate()


def flexible_example():
    week=Week(
        "TGH","2011-11-28",((0,"OR1"),(0,"OR2")),
        np.array([2,3,4]),("A","B"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,1),(1,)),((0,),(0,)),
        np.array([220.,230.,175.]),
        np.array([210.,180.,215.]),()
    )
    return week.validate()


def physical(week, positions):
    return [[int(c),int(week.slots[j][0]),week.slots[j][1]]
            for c,j in zip(week.case_ids,positions)]


def checkpoint(week, planner, adversary):
    return {"weeks": {week_key(week): {
        "complete":True,"planner_assignment":physical(week,planner),
        "adversary_assignment":physical(week,adversary)}}}


def test_fully_fixed_schedule_has_identically_zero_vf_gap():
    week=locked_example()
    for duration in (week.booked,week.booked-16,week.booked+11):
        result=oracle_week((week,np.asarray(duration,float),
                            array_hash(duration),2.,30.,1))
        assert result["complete"]
        hindsight=evaluate(week,week.original,week.actual)["cost"]
        assert result["gap_without_hindsight"]==pytest.approx(hindsight)
        assert result["realized_plan_cost"]==pytest.approx(hindsight)
    idx={int(c):i for i,c in enumerate(week.case_ids)}
    lib=ScheduleLibrary([week],[checkpoint(week,week.original,week.original)],
                        idx,week.n,2.)
    X=np.column_stack((np.ones(week.n),np.array([-1.,0.,1.])))
    for v in (-8.,-3.,3.):
        w=np.array([v,.8])
        value,grad=lib.objective_gradient(w,X,week.booked,.8,30.,0.)
        # Smooth costs exactly cancel between predicted planner/adversary.
        assert np.linalg.norm(grad)<1e-8
        assert value==pytest.approx(hindsight/week.n,abs=1e-7)


def test_true_bound_is_between_regret_and_old_vf_bound():
    w=flexible_example()
    durations=w.booked-12.
    actual_opt=solve_week(w,w.actual,seconds=45.,threads=1,tie=False)
    assert actual_opt.complete
    row=oracle_week((w,durations,array_hash(durations),2.,45.,1))
    assert row["complete"]
    bound=row["gap_without_hindsight"]-actual_opt.value
    regret=row["realized_plan_cost"]-actual_opt.value
    theta=(1.75*np.maximum(w.actual-durations,0)+
           np.maximum(durations-w.actual,0)).sum()
    vf=row["predicted_plan_cost"]+theta-actual_opt.value
    assert regret-0.01<=bound<=vf+0.01


def test_schedule_library_analytic_gradient():
    week=flexible_example()
    idx={int(c):i for i,c in enumerate(week.case_ids)}
    cp=[checkpoint(week,[0,0,1],[0,0,1]),
        checkpoint(week,[1,0,1],[1,1,1])]
    lib=ScheduleLibrary([week],cp,idx,week.n,2.)
    X=np.column_stack((np.ones(week.n),np.array([-1.,0.,1.])))
    w=np.array([-9.,3.])
    for lam in (0.,.025):
        score,grad=lib.objective_gradient(w,X,week.booked,.8,30.,lam,
                                          temp=1.7,room_eps=2.)
        approx=[]
        for i in range(len(w)):
            d=np.zeros(len(w));d[i]=1.e-5
            plus=lib.objective_gradient(w+d,X,week.booked,.8,30.,lam,
                                        temp=1.7,room_eps=2.)[0]
            minus=lib.objective_gradient(w-d,X,week.booked,.8,30.,lam,
                                         temp=1.7,room_eps=2.)[0]
            approx.append((plus-minus)/2.e-5)
        np.testing.assert_allclose(grad,approx,rtol=1.e-5,atol=1.e-5)
        assert np.isfinite(score)


def test_exact_training_certificates_checkpoint_and_reuse(tmp_path):
    week=locked_example()
    idx={int(c):i for i,c in enumerate(week.case_ids)}
    X=np.column_stack((np.ones(week.n),np.array([-1.,0.,1.])))
    start=np.array([-5.,.5])
    result=certify(tmp_path,"one.json",[week],idx,X,week.booked,start,
                   .8,30.,2.,30.,1,1)
    score=certified_score(result,[week],start,0.,week.n)
    expected=evaluate(week,week.original,week.actual)["cost"]/week.n
    assert score["gap_without_hindsight_per_case"]==pytest.approx(expected)
    again=certify(tmp_path,"one.json",[week],idx,X,week.booked,start,
                  .8,30.,2.,30.,1,1)
    assert again==result
    with pytest.raises(ValueError,match="different policy"):
        certify(tmp_path,"one.json",[week],idx,X,week.booked,
                start+1,.8,30.,2.,30.,1,1)


def test_training_cannot_improve_fixed_week_only_by_reducing_predicted_cost(tmp_path):
    week=locked_example()
    idx={int(c):i for i,c in enumerate(week.case_ids)}
    X=np.column_stack((np.ones(week.n),np.array([-1.,0.,1.])))
    start=[-6.,0.]
    settings={"workbook":"synthetic test","gamma":2.}
    result=train_gap(tmp_path,[week],idx,X,week.booked,0.,start,settings,
                     .8,30.,2.,2,15,4.,1.5,30.,1,1)
    assert result["complete"]
    assert len(result["history"])>=1
    for row in result["history"]:
        if row["accepted"]:
            assert (row["certified_after"]["regularized_gap_without_hindsight_per_case"] <
                    row["certified_before"]["regularized_gap_without_hindsight_per_case"])
    rerun=train_gap(tmp_path,[week],idx,X,week.booked,0.,start,settings,
                    .8,30.,2.,2,15,4.,1.5,30.,1,1)
    assert rerun==result


def test_five_policy_validation_and_tie_safe_gap_certificate():
    week=flexible_example()
    d=week.booked-7.
    policies={
        "BOOKED":week.booked.copy(),"LASSO":d.copy(),
        "CASE":d.copy(),"VF":d.copy(),"VF_GAP":d.copy()
    }
    rows=evaluate_week_gap((week,policies,45.,1,2.))
    assert len(rows)==5 and all(r["complete"] for r in rows)
    assert len({r["oracle_assignment_sha256"] for r in rows})==1
    gap=next(r for r in rows if r["method"]=="VF_GAP")
    assert gap["vf_gap_bound_tie_safe"]+.02>=gap["regret"]
    assert gap["vf_gap_bound"]<=gap["vf_bound"]+.02
    assert gap["gap_primary_predicted_cost"]<=gap["pred_cost"]+.02
