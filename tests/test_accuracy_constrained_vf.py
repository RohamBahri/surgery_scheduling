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
        expected=3.*(1.75*np.maximum(err,0.)+np.maximum(-err,0.)).sum()/2
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
