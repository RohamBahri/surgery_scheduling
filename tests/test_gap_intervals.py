"""Interval-safe VF-GAP acceptance and checkpoint reuse regression tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.gap_intervals import (
    interval_week, evaluate_component, gap_interval, worst_components,
    duration_digest,
)
from surgery.gap_learning import certify, certified_score
from surgery.weekly_training import save_json


def sample():
    week=Week(
        "TWH-main","2012-02-27",((0,"OR101"),(0,"OR102")),
        np.array([51,52,53]),("S1","S2"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,1),(1,)),((0,),(0,)),
        np.array([230.,210.,150.]),np.array([260.,230.,110.]),()
    )
    return week.validate()


def test_interval_arithmetic_with_planner_timeout(monkeypatch):
    import surgery.gap_intervals as gi
    w=sample()
    d=w.booked.copy()
    plan_calls=[]
    adv_calls=[]

    def planner(comp,duration,**kw):
        plan_calls.append(1)
        # Same feasible incumbent and provable 20.162 cost uncertainty.
        from surgery.weekly_planner import evaluate
        val=evaluate(comp,comp.original,duration)["cost"]
        return SimpleNamespace(
            assignment=list(map(int,comp.original)),
            value=val,bound=val-20.162013,
            complete=False,status="9",seconds=600.)

    def adversary(comp,duration,**kw):
        adv_calls.append(1)
        from surgery.weekly_planner import evaluate
        value=(evaluate(comp,comp.original,comp.actual)["cost"]
               -2*evaluate(comp,comp.original,duration)["cost"])
        return SimpleNamespace(
            assignment=list(map(int,comp.original)),
            value=value,bound=value,
            complete=True,status="2",seconds=.01)

    monkeypatch.setattr(gi,"solve_week",planner)
    monkeypatch.setattr(gi,"solve_adversary",adversary)
    row=interval_week((w,d,duration_digest(d),2.,600.,1,None,set()))
    assert row["bounded"] is True
    assert row["complete"] is False
    assert row["gap_upper"]-row["gap_lower"]==pytest.approx(40.324026)
    assert len(plan_calls)==len(adv_calls)==1
    # A partial bound can be re-used with zero solver invocations.
    cached=interval_week((w,d,duration_digest(d),2.,600.,1,row,set()))
    assert cached==row
    assert len(plan_calls)==len(adv_calls)==1
    # Targeted refinement re-solves ONLY the unfinished planner oracle.
    interval_week((w,d,duration_digest(d),2.,600.,1,row,{0}))
    assert len(plan_calls)==2 and len(adv_calls)==1


def test_merge_certificates_never_loses_valid_tighter_bounds():
    from surgery.gap_intervals import _merge
    prior={"bounded":True,"complete":False,"lower":1520.288,
           "upper":1540.450,"assignment":[[1,0,"OR101"]]}
    improvement={"bounded":True,"complete":False,"lower":1530.,
                 "upper":1545.,"assignment":[[1,0,"OR102"]]}
    row=_merge(prior,improvement,"planner")
    assert row["lower"]==1530.
    assert row["upper"]==1540.450
    assert row["assignment"]==prior["assignment"]
    assert row["complete"] is False


def test_interval_scores_certify_descent_without_exact_planner():
    w=sample()
    n=18705
    policy=np.array([-5.,2.])
    def record(lower,upper):
        return {"weeks":{w.group+"|"+w.monday:{
            "complete":lower==upper,
            "bounded":True,
            "gap_lower":lower,"gap_upper":upper,
            "realized_plan_cost":240.}}}
    old=certified_score(record(2500,2500),[w],policy,.05,n)
    candidate=certified_score(record(2430,2470),[w],policy,.05,n)
    assert candidate["regularized_gap_without_hindsight_per_case"] is None
    assert candidate["regularized_gap_upper_per_case"]<old[
        "regularized_gap_lower_per_case"]
    assert candidate["regularized_gap_interval_width_per_case"]==pytest.approx(40/n)


def test_old_exact_checkpoint_reused_without_resolving(monkeypatch,tmp_path):
    import surgery.gap_learning as gl
    week=sample()
    index={int(cid):i for i,cid in enumerate(week.case_ids)}
    X=np.ones((week.n,2))
    booked=week.booked
    w=np.array([-4.,0.])
    from surgery.planner import display_and_duration
    _,d=display_and_duration(X@w,booked,.8,30.)
    identity={"weights_sha256":gl.array_hash(w),"gamma":2.}
    key=gl.week_key(week)
    original={"identity":identity,"weeks":{
        key:{"key":key,"input_sha256":gl.array_hash(d),
             "complete":True,"gap_without_hindsight":100.,
             "realized_plan_cost":200.,
             "planner_assignment":[],"adversary_assignment":[]}}}
    save_json(tmp_path/"gap_current_00.json",original)

    def must_not_run(*a,**k):
        raise AssertionError("Recomputed an already certified legacy week")
    monkeypatch.setattr(gl,"interval_week",must_not_run)
    reused=certify(tmp_path,"gap_current_00.json",[week],index,X,booked,
                   w,.8,30.,2.,300.,1,1)
    assert reused==original


def test_missing_interval_is_not_silently_accepted(tmp_path):
    w=sample()
    with pytest.raises(ValueError,match="finite rigorous"):
        certified_score({"weeks":{w.group+"|"+w.monday:{
            "complete":False,"bounded":False,"realized_plan_cost":3.}}},
            [w],np.array([1.,0.]),.1,w.n)
