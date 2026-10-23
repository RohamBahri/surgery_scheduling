"""Mathematical regression tests for full-response CASE and VF MM."""
import numpy as np
import pytest
from scipy import sparse

from surgery.nonmonotone_learning import (
    C_OVER, C_UNDER, KAPPA, TURNOVER, CAPACITY,
    ResponseDC, Scenario, DCObjective, pdca, mm_vf, response,
)


@pytest.mark.parametrize("alpha,h,cap",[(0.8,30,45),(.4,30,60),(.25,15,45),(1.,30,60)])
@pytest.mark.parametrize("canonical", [None,(.8,30.)])
def test_exact_response_dc_including_short_bookings_and_rejection(alpha,h,cap,canonical):
    booked=np.array([2.,12.,45.,120.,560.])
    mapping=ResponseDC(booked,alpha,h,cap,canonical)
    rng=np.random.default_rng(134)
    for u in (np.zeros(5),np.full(5,300.),np.full(5,-300.),
              *[rng.uniform(-200,200,size=5) for _ in range(15)]):
        mapping.verify(u,1e-7)
        P,N,dP,dN=mapping.parts(u)
        assert np.all(dP>=-1e-9)
        assert np.all(dN>=-1e-9)
        assert np.all(mapping.reviewed(u)>=1.-1e-9)
        if canonical is not None:
            a0,h0=canonical
            lo=np.maximum(-cap,1.-booked)
            original=np.clip(u,lo,cap)
            display=mapping.display(u)
            assert np.allclose(response(display,a0,h0),
                               response(original,a0,h0),atol=1e-8)
            assert np.all(abs(display)<=abs(original)+1e-9)


def test_nonmonotone_response_is_not_replaced_by_saturation():
    u=np.array([20.,30.,60.,150.,180.])
    got=response(u,.8,30.)
    assert got[0] == pytest.approx(16.)
    assert got[1] == pytest.approx(24.)
    assert got[2] == pytest.approx(18.)
    assert got[3] == pytest.approx(0.)
    assert got[4] == pytest.approx(0.)


def test_case_decomposition_equals_actual_asymmetric_loss():
    b=np.array([90.,120.,40.,15.])
    a=np.array([70.,145.,65.,11.])
    X=np.array([[1,0], [1,1], [1,-1], [1,2.]],float)
    scenarios=[Scenario(.8,30),Scenario(.4,15)]
    obj=DCObjective(b,X,np.zeros(4),scenarios,60.,mode="CASE",
                    actual=a,ridge=0.,l1=0.,canonical_nominal=(.8,30.))
    w=np.array([16.,-3.])
    target=0.
    for s in scenarios:
        d=ResponseDC(b,s.alpha,s.h,60.,(.8,30.)).reviewed(X@w)
        target+=(C_OVER*np.maximum(a-d,0)+C_UNDER*np.maximum(d-a,0)).sum()/len(b)/2
    assert obj.value(w)==pytest.approx(target,rel=1e-9,abs=1e-8)


def test_vf_decomposition_matches_fixed_plan_cost():
    b=np.array([100.,145.,95.])
    X=np.array([[1,0], [0,1], [-1,2]],float)
    raw=np.array([40.,-40.,10.])
    bank=np.array([[120.,153.,89.],[80.,160.,112.]])
    S=sparse.csr_matrix([[1,1,0],[0,0,1]])
    scenarios=[Scenario(.8,30),Scenario(.4,15)]
    obj=DCObjective(b,X,raw,scenarios,45.,mode="VF",
                    actual_bank=bank,room_matrices=[S,S],
                    ridge=0.,canonical_nominal=(.8,30.))
    w=np.array([5.,-2.])
    expected=0.
    for s in scenarios:
        d=ResponseDC(b,s.alpha,s.h,45.,(.8,30.)).reviewed(raw+X@w)
        loads=S@d+TURNOVER*(np.diff(S.indptr)-1)
        cost=(KAPPA*np.maximum(bank-d[None,:],0).mean(axis=0).sum()
              +TURNOVER*S.shape[0]+KAPPA*np.maximum(loads-CAPACITY,0).sum())
        expected+=cost/len(b)/2
    assert obj.value(w)==pytest.approx(expected,rel=1e-9,abs=1e-8)


def test_scipy_smoke_pdca_case_exact_descent():
    b=np.array([120.,100.,120.,100.])
    a=np.array([107.,111.,105.,112.])
    X=np.array([[1,0],[1,1],[1,-1],[1,2]],float)
    obj=DCObjective(b,X,np.zeros(len(b)),[Scenario(.8,30)],45.,
                    mode="CASE",actual=a,ridge=.001,
                    canonical_nominal=(.8,30.))
    initial=np.zeros(2)
    ans=pdca(obj,initial,radius=40,outer=3,max_cuts=30,backend="scipy")
    assert obj.value(ans["weights"])<=obj.value(initial)+1e-7
    assert all(r["after"]<=r["before"]+1e-7 for r in ans["iterations"] if r["accepted"])


def test_mm_acceptance_is_majorizer_descent(tmp_path):
    from surgery.weekly_data import Week
    w=Week(group="TGH",monday="2012-01-02",slots=((0,"OR1"),),
           case_ids=np.array([1,2,3]),surgeon_ids=("S1",),
           surgeon=np.zeros(3,dtype=int),original=np.zeros(3,dtype=int),
           allowed=((0,),),required_days=((0,),),
           booked=np.array([100.,120.,140.]),actual=np.array([110.,140.,130.]),
           added_days=())
    calls=[]
    def planner(items,durations,j,k):
        calls.append((j,k))
        return {"one":[0,0,0]}
    s=[Scenario(.8,30)]
    ans=mm_vf([("one",w,np.arange(3))],w.booked,
              np.array([[1.],[0.5],[-1.]]),np.zeros(3),
              s,np.array([[110.,145.,125.],[120.,100.,160.]]),45.,
              root=tmp_path,iterations=2,pdca_steps=2,max_cuts=20,
              backend="scipy",solve_plans=planner,
              canonical_nominal=(.8,30.))
    assert len(calls)>=1
    for r in ans["history"]:
        if r["accepted"]:
            assert r["majorizer_after"] <= r["before"]+1e-7


def test_gurobi_convex_cut_master_when_license_available():
    """Exercise the production QP interface, not merely the SciPy fallback."""
    gp=pytest.importorskip("gurobipy")
    try:
        m=gp.Model("test_license")
        m.Params.OutputFlag=0
        x=m.addVar(lb=0,ub=1)
        m.setObjective(x*x)
        m.optimize()
        m.dispose()
    except gp.GurobiError as exc:
        pytest.skip("No usable Gurobi license on this runner: "+str(exc))
    b=np.array([120.,100.,120.,100.])
    a=np.array([107.,111.,105.,112.])
    X=np.array([[1,0],[1,1],[1,-1],[1,2]],float)
    obj=DCObjective(b,X,np.zeros(len(b)),[Scenario(.8,30)],45.,
                    mode="CASE",actual=a,ridge=.001,
                    canonical_nominal=(.8,30.))
    start=np.zeros(2)
    answer=pdca(obj,start,radius=40,outer=3,max_cuts=20,backend="gurobi")
    assert obj.value(answer["weights"]) <= obj.value(start)+1e-7
    for step in answer["iterations"]:
        assert step["after"] <= step["before"]+1e-7 or not step["accepted"]
        if step["inner_status"] == "INNER_CONVERGED":
            assert step["inner_certified"]


def test_gurobi_exact_planner_plus_vf_mm_if_licensed(tmp_path):
    """Small but real planner -> scenario response -> outer MM integration."""
    gp=pytest.importorskip("gurobipy")
    try:
        m=gp.Model("vf_integration_license")
        m.Params.OutputFlag=0
        x=m.addVar(lb=0.,ub=1.)
        m.setObjective(x*x)
        m.optimize()
        m.dispose()
    except gp.GurobiError as exc:
        pytest.skip("No licensed small Gurobi model: "+str(exc))
    from surgery.weekly_data import Week
    from surgery.anchored_weekly_solver import solve_week
    week=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1",),np.zeros(3,dtype=int),
        np.zeros(3,dtype=int),((0,1),),((0,),),
        np.array([190.,180.,160.]),np.array([195.,175.,170.]),())
    week.validate()
    tested=[]
    def true_planner(items,durations,j,k):
        a=solve_week(week,np.asarray(durations),seconds=20,threads=1,tie=False)
        assert a.complete and a.assignment is not None
        tested.append((j,k))
        return {"block":a.assignment}
    ans=mm_vf([("block",week,np.arange(3))],week.booked,
              np.array([[1.],[0.5],[-0.25]]),
              np.array([12.,-4.,8.]),[Scenario(.8,30)],
              np.array([[205.,170.,166.],[185.,190.,175.]]),
              45.,root=tmp_path,iterations=2,pdca_steps=2,
              max_cuts=20,backend="gurobi",solve_plans=true_planner,
              canonical_nominal=(.8,30.))
    assert tested
    assert all(not r["accepted"] or r["majorizer_after"] < r["before"]+1e-8
               for r in ans["history"])
