"""Contracts for restored-turnover and explicit scheduling diagnostics."""
import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery import weekly_planner as wp
from surgery import nonmonotone_learning as nl
from surgery import anchored_weekly_solver as anchor
from surgery.vf_features import structural_features
from pilot.run_joint_vf import parse_args


def sample():
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2,3]),("S1",),
        np.array([0,0,0]),np.array([0,0,1]),
        ((0,1),),((0,),),
        np.array([200.,180.,100.]),np.array([210.,180.,90.]),()).validate()


def test_turnover_switch_changes_all_cost_identities(monkeypatch):
    week=sample()
    z=[0,0,1]
    with monkeypatch.context() as m:
        for module in (wp,nl,anchor):
            m.setattr(module,"TURNOVER",30.)
        assert np.allclose(wp.loads(week,z,week.booked),[410.,100.])
        assert wp.evaluate(week,z,week.booked)["cost"]==pytest.approx(
            nl.planned_cost_fast(week,z,week.booked))
        assert anchor.conservation_certificate(week,week.booked) is None
        history=np.zeros((week.n,2))
        features=structural_features([("x",week,np.arange(3))],
                                     week.booked,history)
        pressure=np.arcsinh((480.+30.-960.)/(120.*2.))
        assert features[0,0]==pytest.approx(pressure)
        mresp=nl.ResponseDC(week.booked,.8,30.,180.,1.,
                            link="saturating")
        X=np.ones((3,1))
        S,_=nl.room_matrix([("x",week,np.arange(3))],{"x":z},3)
        loss=nl.DCObjective(week.booked,X,np.zeros(3),mresp,
                            mode="VF",actual=week.actual,
                            components=1,rooms=S)
        for w in (np.array([0.]),np.array([15.]),np.array([-15.])):
            d=mresp.reviewed(X@w)
            e=week.actual-d
            expected=wp.evaluate(week,z,d)["cost"]+(
                1.75*np.maximum(e,0.)+np.maximum(-e,0.)).sum()
            assert loss.value(w)==pytest.approx(expected,abs=1e-7)


def test_turnover_cli_and_hash_only_stage(tmp_path,monkeypatch):
    context=tmp_path/"context.pkl"
    context.write_bytes(b"stub")
    with monkeypatch.context() as m:
        m.setenv("SURGERY_TURNOVER_MINUTES","0")
        for module in (wp,nl,anchor):
            m.setattr(module,"TURNOVER",module.TURNOVER)
        a=parse_args(["--context",str(context),"--stage","hash-only",
                      "--turnover","0"])
        assert a.stage=="hash-only" and a.turnover==0.
        a=parse_args(["--context",str(context),"--stage","train",
                      "--turnover","30"])
        assert a.turnover==30.
        assert wp.TURNOVER==nl.TURNOVER==anchor.TURNOVER==30.


def test_quadratic_tie_uses_configured_turnover_if_gurobi_licensed(monkeypatch):
    gp=pytest.importorskip("gurobipy")
    try:
        probe=gp.Model();probe.Params.OutputFlag=0
        v=probe.addVar(lb=0,ub=1)
        probe.setObjective(v*v);probe.optimize();probe.dispose()
    except gp.GurobiError:
        pytest.skip("No Gurobi license")
    w=sample()
    with monkeypatch.context() as m:
        for module in (wp,nl,anchor):
            m.setattr(module,"TURNOVER",30.)
        out=anchor.solve_week(w,w.booked,seconds=40.,threads=1,
                              tie=True,hash_seed=0)
        assert out.complete,out.status
        assert out.balance_complete
        assert out.balance_objective==pytest.approx(
            sum(wp.loads(w,out.assignment,w.booked)**2),abs=1.1)
        assert out.value==pytest.approx(
            wp.evaluate(w,out.assignment,w.booked)["cost"],abs=1e-6)
