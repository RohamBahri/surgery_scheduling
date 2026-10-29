"""Regression contracts for the zero-turnover, direct exact-VF experiment."""
import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import loads, evaluate
from surgery.nonmonotone_learning import ResponseDC, mm_vf
from pilot.operational_audit import necessary_peak_bounds, audit_policy


def example():
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103,104]),("S1",),
        np.zeros(4,dtype=int),np.array([0,0,1,1]),
        ((0,1),),((0,),),
        np.array([500.,300.,400.,200.]),
        np.array([520.,320.,420.,180.]),()).validate()


def test_zero_turnover_conservation_and_idle_on_empty_rooms():
    w=example()
    assert list(loads(w,[0,0,1,1],w.booked))==[800.,600.]
    z=[0,0,0,0]
    metric=evaluate(w,z,w.booked)
    assert metric["idle"]==480.
    assert metric["overtime"]==920.
    assert metric["cost"]==pytest.approx(480.+1.75*920.)


def test_necessary_capacity_and_eligibility_lower_bounds():
    w=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2,3]),("S1","S2"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,),(1,)),((0,),(0,)),
        np.array([600.,200.,200.]),np.array([600.,200.,200.]),()).validate()
    v=necessary_peak_bounds(w,w.booked)
    assert v["aggregate_peak_lower_bound"]==pytest.approx(500.)
    assert v["eligibility_peak_lower_bound"]==pytest.approx(800.)
    assert v["necessary_peak_lower_bound"]==pytest.approx(800.)
    assert v["aggregate_overtime_minutes_lower_bound"]==pytest.approx(40.)


def test_operational_audit_reports_confirmed_avoidable_peak(tmp_path,monkeypatch):
    from pilot import operational_audit as audit
    w=example()
    monkeypatch.setattr(audit,"solve_minimum_peak",
        lambda *a,**k:dict(min_peak_lb=700.,min_peak_ub=700.,
            min_peak_proven=True,solver_status=2,solver_seconds=.1,
            original_peak=800.))
    result=audit_policy([("toy",w,np.arange(4))],w.booked,
        {"toy":dict(assignment=[0,0,1,1],complete=True,status="TEST")},
        tmp_path,threshold=720.)
    assert result["flagged_instances"]==1
    assert result["confirmed_avoidable"]==1
    assert result["entries"][0]["avoidable_peak_minutes_at_least"]==pytest.approx(100.)
    assert result["entries"][0]["forced_over_capacity_proven"]


def test_direct_mm_accepts_only_proven_exact_vf_improvement(tmp_path,monkeypatch):
    from surgery import nonmonotone_learning as nl
    w=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2]),("S1",),np.array([0,0]),np.array([0,0]),
        ((0,1),),((0,),),np.array([100.,100.]),
        np.array([120.,120.]),()).validate()
    X=np.ones((2,1))
    response=ResponseDC(w.booked,.8,30.,180.,1.,link="saturating")
    def solve(items,d,k):
        z=[0,0]
        return {"toy":dict(assignment=z,
            bound=evaluate(w,z,d)["cost"],complete=True)}
    monkeypatch.setattr(nl,"pdca",
        lambda objective,start,**kwargs:
            {"weights":np.array([20.]),"status":"FAKE","iterations":[]})
    out=mm_vf([("toy",w,np.arange(2))],w.booked,X,response,w.actual,
        start=np.zeros(1),root=tmp_path,lam=.1,iterations=1,
        solve_plans=solve,backend="scipy")
    assert "libraries" not in out
    assert out["history"][0]["accepted"]
    assert out["history"][0]["certified_planner_gap"]==pytest.approx(0.)
    assert out["history"][0]["guaranteed_exact_vf_decrease_lower_bound"]>0
    assert out["theta"]==[20.]


def test_secondary_squared_loads_when_gurobi_available():
    gp=pytest.importorskip("gurobipy")
    try:
        probe=gp.Model(); probe.Params.OutputFlag=0
        z=probe.addVar(lb=0.,ub=1.)
        probe.setObjective(z*z);probe.optimize();probe.dispose()
    except gp.GurobiError:
        pytest.skip("No working Gurobi license")
    from surgery.anchored_weekly_solver import solve_week
    w=example()
    out=solve_week(w,w.booked,seconds=40.,threads=1,tie=True,hash_seed=0)
    assert out.complete,(out.status,out.balance_complete)
    assert out.balance_complete
    assert sorted(loads(w,out.assignment,w.booked))==pytest.approx([700.,700.])
    assert out.balance_objective==pytest.approx(2*700.**2)
    assert evaluate(w,out.assignment,w.booked)["cost"]==pytest.approx(
        evaluate(w,[0,0,1,1],w.booked)["cost"])
