"""Regression checks for displayed-accuracy Lagrangian and relaxed surgeon days."""
import numpy as np
import pytest
from scipy import sparse

from surgery import weekly_planner as weekly
from surgery import anchored_weekly_solver as anchored
from surgery.nonmonotone_learning import DCObjective, ResponseDC


@pytest.mark.parametrize("link",["saturating","nonmonotone"])
def test_displayed_accuracy_loss_exactly_matches_dc_difference(link):
    b=np.array([12.,150.,200.,300.])
    y=np.array([25.,160.,180.,340.])
    X=np.array([[1.,0.],[1.,1.],[1.,-1.],[1.,.4]])
    response=ResponseDC(b,alpha=.8,h=30.,cap=180.,floor=1.,link=link)
    # The risk is on what is displayed, not the implemented/reviewed booking.
    rooms=sparse.csr_matrix(([1.,1.,1.,1.],
             ([0,0,1,1],[0,1,2,3])),shape=(2,4))
    plain=DCObjective(b,X,np.zeros(len(b)),response,mode="VF",
        actual=y,components=2,rooms=rooms,lam=.1,penalize=[0.,1.])
    fortified=DCObjective(b,X,np.zeros(len(b)),response,mode="VF",
        actual=y,components=2,rooms=rooms,lam=.1,penalize=[0.,1.],
        accuracy_mu=3.)
    for w in [np.array([0.,0.]),np.array([80.,-50.]),
              np.array([-160.,170.]),np.array([30.,20.])]:
        display=b+response.display(X@w)
        err=y-display
        expected=3.*np.abs(err).sum()/2
        assert fortified.value(w)-plain.value(w)==pytest.approx(expected,abs=1e-7)


def test_quadratic_solver_includes_displayed_accuracy_if_licensed():
    gp=pytest.importorskip("gurobipy")
    try:
        model=gp.Model();model.Params.OutputFlag=0
        variable=model.addVar();model.setObjective(variable*variable)
        model.optimize();model.dispose()
    except gp.GurobiError:
        pytest.skip("Gurobi license unavailable")
    from surgery.nonmonotone_learning import pdca
    b=np.array([100.,120.]);y=np.array([140.,90.])
    X=np.array([[1.,1.],[1.,-1.]])
    response=ResponseDC(b,alpha=.8,h=30.,link="saturating")
    S=sparse.csr_matrix(([1.,1.],([0,1],[0,1])),shape=(2,2))
    obj=DCObjective(b,X,np.zeros(2),response,mode="VF",
         actual=y,components=1,rooms=S,lam=.1,penalize=[0.,1.],
         accuracy_mu=3.)
    start=np.array([0.,0.])
    result=pdca(obj,start,outer=4,radius=100.,backend="gurobi")
    assert np.isfinite(result["objective"])
    assert result["objective"]<=obj.value(start)+1e-7
    assert all(row["inner_certified"] for row in result["iterations"])


def test_relaxed_retention_allows_empty_observed_day(monkeypatch):
    from surgery.weekly_data import Week
    wk=Week("TGH","2012-01-02",((0,"OR1"),(2,"OR2")),
        np.array([10,11]),("S1",),np.array([0,0]),
        np.array([0,1]),((0,1),),((0,2),),
        np.array([100.,100.]),np.array([110.,110.]),()).validate()
    with monkeypatch.context() as m:
        m.setattr(weekly,"RELAX_REQUIRED_DAYS",True)
        loads=weekly.loads(wk,[0,0],wk.booked)
        assert loads.tolist()==pytest.approx([200.,0.])
        assert weekly.evaluate(wk,[0,0],wk.booked)[
            "emptied_historical_surgeon_days"]==1
        assert anchored.mandatory_slots(wk)==frozenset()
        assert anchored.conservation_certificate(wk,wk.booked) is None
    with pytest.raises(ValueError,match="emptied"):
        weekly.loads(wk,[0,0],wk.booked)


@pytest.mark.parametrize("link",["saturating","nonmonotone"])
def test_display_only_objective_exactly_matches_pre_response_accuracy(link):
    b=np.array([8.,100.,190.,275.])
    actual=np.array([20.,133.,150.,298.])
    X=np.array([[1.,-2.],[1.,1.],[1.,0.],[1.,.5]])
    c=ResponseDC(b,.8,30.,180.,1.,link=link)
    obj=DCObjective(b,X,np.zeros(4),c,mode="DISPLAY",
        actual=actual,components=2,lam=0.,penalize=[0.,1.])
    for w in [np.array([0.,0.]),np.array([22.,15.]),
              np.array([-50.,85.]),np.array([180.,-160.])]:
        display=b+c.display(X@w)
        error=actual-display
        expected=np.abs(error).sum()/2
        assert obj.value(w)==pytest.approx(expected,abs=1e-7)


def test_display_reference_fit_uses_same_features_and_improves_accuracy(
        tmp_path,monkeypatch):
    from pilot._joint_vf_engine import displayed_reference_fit
    from surgery.weekly_data import Week
    b=np.array([120.,110.]);actual=np.array([150.,95.])
    X=np.array([[1.,1.,.5],[1.,-1.,-.5]])
    c=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    week=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([11,12]),("S1",),np.array([0,0]),
        np.array([0,1]),((0,1),),((0,),),
        b,actual,()).validate()
    start=np.zeros(3)
    def fake_pdca(objective,start,**kwargs):
        assert objective.mode=="DISPLAY"
        assert objective.X.shape[1]==3
        return dict(weights=start.tolist(),status="NO_VERIFIED_DESCENT",
                    iterations=[])
    monkeypatch.setattr("pilot._joint_vf_engine.pdca",fake_pdca)
    w,score=displayed_reference_fit(
        dict(booked=b,actual=actual),[("k",week,np.arange(2))],
        X,c,start,tmp_path,4,"gurobi",100.)
    assert np.array_equal(w,start)
    assert score==pytest.approx((30+15)/1.)
    assert (tmp_path/"display_accuracy_reference.json").exists()


def test_primary_training_and_accuracy_gate_defaults(tmp_path):
    from pilot.run_joint_vf import parse_args
    context=tmp_path/"context.pkl"
    context.write_bytes(b"checkpoint")
    a=parse_args(["--context",str(context)])
    assert a.turnover==0.
    assert a.train_tie=="primary"
    assert a.accuracy_gate=="off"
    assert a.accuracy_budget==pytest.approx(.05)


def test_display_only_pdca_exact_qp_if_licensed():
    gp=pytest.importorskip("gurobipy")
    try:
        m=gp.Model();m.Params.OutputFlag=0
        q=m.addVar();m.setObjective(q*q);m.optimize();m.dispose()
    except gp.GurobiError:
        pytest.skip("No Gurobi license")
    from surgery.nonmonotone_learning import pdca
    b=np.array([100.,120.]);actual=np.array([125.,110.])
    X=np.array([[1.,1.],[1.,-1.]])
    c=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    obj=DCObjective(b,X,np.zeros(2),c,mode="DISPLAY",
                    actual=actual,components=1,lam=0.)
    before=obj.value(np.zeros(2))
    run=pdca(obj,np.zeros(2),outer=5,radius=100.,backend="gurobi")
    assert np.isfinite(run["objective"])
    assert run["objective"]<=before+1e-7
    assert all(z["inner_certified"] for z in run["iterations"])
