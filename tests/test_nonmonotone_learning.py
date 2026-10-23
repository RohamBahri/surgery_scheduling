"""Contracts for the single-nominal-response CASE and VF library-MM study."""
import numpy as np
import pytest
from scipy import sparse

from surgery.nonmonotone_learning import (
    C_OVER, C_UNDER, ResponseDC, DCObjective, response, pdca,
    room_matrix, library_value, mm_vf, choose_library_schedules)


def toy_week():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1",),
        np.zeros(3,dtype=int),np.zeros(3,dtype=int),
        ((0,1),),((0,),),
        np.array([190.,180.,160.]),np.array([195.,175.,170.]),()).validate()


def test_three_regime_response_not_inverse_clip():
    u=np.array([0.,20.,30.,60.,150.,180.,-30.,-60.,-150.])
    got=response(u)
    assert np.allclose(got,[0,16,24,18,0,0,-24,-18,0])
    assert np.sign(response(-60.))==-1.


@pytest.mark.parametrize("cap",[20.,30.,180.])
def test_exact_casewise_floor_breakpoint_compiler_and_minimum_display(cap):
    b=np.array([.5,1.,2.,3.,8.,12.,20.,40.,100.,560.])
    m=ResponseDC(b,.8,30.,cap,1.)
    rng=np.random.default_rng(22)
    grid=np.linspace(-300,300,1201)
    for scalar in grid:
        m.verify(np.full(len(b),scalar))
    for _ in range(30):
        u=rng.uniform(-250,250,len(b))
        m.verify(u)
    assert np.all(m.reviewed(np.full(len(b),-50.))>=1.)
    if cap==180.:
        # For b=2, a strong negative raw display induces the FLOOR,
        # not the old artificial display clipping to 1-b.
        assert m.reviewed(np.full(len(b),-30.))[2]==pytest.approx(1.)
        # A sufficiently extreme display is REJECTED, not shortened.
        assert m.reviewed(np.full(len(b),-180.))[2]==pytest.approx(2.)
    assert np.max(abs(m.display(np.full(len(b),100.))))<=30.+1e-9


def test_case_exact_dc_identity_against_direct_asymmetric_loss():
    booked=np.array([2.,40.,120.,100.,9.])
    actual=np.array([3.,66.,105.,130.,12.])
    X=np.array([[1.,-1.],[1.,1.],[1.,0.],[1.,2.],[1.,-.5]])
    m=ResponseDC(booked,.8,30.,180.,1.)
    mask=np.array([0.,1.])
    obj=DCObjective(booked,X,np.zeros(len(booked)),m,mode="CASE",
                    actual=actual,components=3,lam=.003,penalize=mask)
    rng=np.random.default_rng(74)
    for _ in range(20):
        w=rng.uniform(-200,200,size=2)
        d=m.reviewed(X@w)
        err=actual-d
        direct=(C_OVER*np.maximum(err,0).sum()+
                C_UNDER*np.maximum(-err,0).sum())/3+.003*abs(w[1])
        assert obj.value(w)==pytest.approx(direct,rel=1e-10,abs=1e-8)


def test_vf_fixed_schedule_exact_dc_identity_includes_unused_roster():
    c=toy_week()
    items=[("toy",c,np.arange(c.n))]
    z={"toy":[0,0,0]}
    S,counts=room_matrix(items,z,c.n)
    assert list(counts)==[3,0]
    X=np.column_stack((np.ones(c.n),np.array([-1.,2.,0.])))
    base=np.array([50.,-70.,170.])
    m=ResponseDC(c.booked,.8,30.,180.,1.)
    obj=DCObjective(c.booked,X,base,m,mode="VF",actual=c.actual,
                    components=1,rooms=S,lam=.01)
    rng=np.random.default_rng(15)
    from surgery.weekly_planner import evaluate
    for _ in range(20):
        w=rng.uniform(-30,30,size=2)
        d=m.reviewed(base+X@w)
        error=c.actual-d
        direct=(evaluate(c,z["toy"],d)["cost"]+
                (C_OVER*np.maximum(error,0)+C_UNDER*np.maximum(-error,0)).sum()
                +.01*np.abs(w).sum())
        assert obj.value(w)==pytest.approx(direct,rel=1e-10,abs=1e-8)


def test_library_majorizer_is_touching_and_global_upper_bound():
    c=toy_week();items=[("toy",c,np.arange(c.n))]
    libraries={"toy":[[0,0,0],[0,1,1]]}
    X=np.column_stack((np.ones(c.n),np.array([1.,-1.,.5])))
    raw=np.array([50.,-70.,80.])
    m=ResponseDC(c.booked,.8,30.,180.,1.)
    w=np.array([3.,-2.])
    d=m.reviewed(raw+X@w)
    selected=choose_library_schedules(items,libraries,d)
    S,_=room_matrix(items,selected,c.n)
    obj=DCObjective(c.booked,X,raw,m,mode="VF",actual=c.actual,
                    components=1,rooms=S,lam=0.)
    touch=library_value(items,libraries,d,c.actual,0.,w)
    assert obj.value(w)==pytest.approx(touch,abs=1e-8)
    rng=np.random.default_rng(12)
    for _ in range(30):
        trial=rng.uniform(-50,50,size=2)
        future=m.reviewed(raw+X@trial)
        assert library_value(items,libraries,future,c.actual,0.,trial) <= (
               obj.value(trial)+1e-7)


@pytest.mark.parametrize("backend",["scipy","gurobi"])
def test_case_pdca_exact_objective_descent(backend):
    if backend=="gurobi":
        gp=pytest.importorskip("gurobipy")
        try:
            model=gp.Model();model.Params.OutputFlag=0
            x=model.addVar(lb=0,ub=1)
            model.setObjective(x*x);model.optimize();model.dispose()
        except gp.GurobiError:
            pytest.skip("No Gurobi license")
    b=np.array([120.,100.,40.,15.])
    a=np.array([107.,111.,45.,12.])
    X=np.array([[1.,0.],[1.,1.],[1.,-1.],[1.,2.]])
    m=ResponseDC(b,.8,30.,180.,1.)
    obj=DCObjective(b,X,np.zeros(len(b)),m,mode="CASE",
                    actual=a,components=1,lam=.001,
                    penalize=np.array([0.,1.]))
    start=np.zeros(2)
    out=pdca(obj,start,radius=50.,outer=4,max_cuts=35,backend=backend)
    assert obj.value(out["weights"]) <= obj.value(start)+1e-7
    assert all(not row["accepted"] or row["after"] < row["before"]
               for row in out["iterations"])


@pytest.mark.parametrize("backend",["scipy","gurobi"])
def test_library_mm_trains_and_keeps_feasible_timeout_incumbent(tmp_path,backend):
    if backend=="gurobi":
        gp=pytest.importorskip("gurobipy")
        try:
            mm=gp.Model();mm.Params.OutputFlag=0
            v=mm.addVar(lb=0,ub=1);mm.setObjective(v*v)
            mm.optimize();mm.dispose()
        except gp.GurobiError:
            pytest.skip("No Gurobi license")
    week=toy_week()
    items=[("toy",week,np.arange(week.n))]
    m=ResponseDC(week.booked,.8,30.,180.,1.)
    calls=[]
    def oracle(items,d,k):
        calls.append(k)
        return {"toy":dict(assignment=[0,0,0],bound=0.,
                           complete=False,status="TIME_LIMIT")}
    results=mm_vf(items,week.booked,
            np.array([[1.],[.5],[-1.]]),np.array([12.,-4.,40.]),
            m,week.actual,root=tmp_path,lam=.001,
            initial_libraries={"toy":[[0,0,0],[0,1,1]]},
            iterations=2,pdca_steps=3,max_cuts=25,
            backend=backend,solve_plans=oracle)
    assert calls
    for r in results["history"]:
        if r["accepted"]:
            assert r["delta"]>0.
    assert results["libraries"]["toy"]


def test_runner_help_imports_without_training():
    import subprocess,sys
    from pathlib import Path
    entry=Path(__file__).resolve().parents[1]/"pilot"/"run_nonmonotone_learning.py"
    out=subprocess.run([sys.executable,str(entry),"--help"],cwd=entry.parents[1],
                       capture_output=True,text=True,timeout=25)
    assert out.returncode==0,out.stderr
    assert "evaluate" in out.stdout and "vf" in out.stdout
    assert "--scenario" not in out.stdout


def test_case_lad_and_penalty_rules_on_small_instance():
    from pilot.run_nonmonotone_learning import lad_start,train_lambda
    X=np.array([[1.,-2.],[1.,-1.],[1.,0.],[1.,1.],[1.,2.]])
    actual_error=4.+2.*X[:,1]
    w,diag=lad_start(X,actual_error,lam=0.,components=1,box=30.)
    assert diag["success"]
    assert np.max(abs(X@w-actual_error))<1e-5
    h=np.column_stack([np.ones(10),np.linspace(-2,2,10),
                       np.zeros(10)])
    # VF has no intercept, and lambda calibration includes column 1.
    a=train_lambda(h,2,intercept=False)
    b=train_lambda(h,2,intercept=True)
    assert a>0 and b>0 and a!=pytest.approx(b)


def test_case_is_exactly_vf_at_zero_increment_for_all_cases():
    b=np.array([1.,2.,15.,50.,120.])
    c=ResponseDC(b,.8,30.,180.,1.)
    base=np.array([-180.,-40.,35.,85.,30.])
    H=np.random.default_rng(19).normal(size=(5,5))
    zero=np.zeros(5)
    assert np.array_equal(c.reviewed(base),c.reviewed(base+H@zero))
    c.verify(base)
    assert np.allclose(
        b+response(c.display(base)),c.reviewed(base),atol=1e-8)



def test_exact_case_qp_matches_convex_majorizer_when_licensed():
    """Reconstruct the actual pDCA QP objective, including all floor hinges."""
    gp=pytest.importorskip("gurobipy")
    try:
        m=gp.Model("license_probe")
        m.Params.OutputFlag=0
        x=m.addVar(lb=0.,ub=1.)
        m.setObjective(x*x)
        m.optimize()
        m.dispose()
    except gp.GurobiError as exc:
        pytest.skip(str(exc))
    from surgery.nonmonotone_learning import CaseConvexQP
    rng=np.random.default_rng(77)
    n,p=24,5
    b=np.linspace(1.,150.,n)
    X=np.column_stack([np.ones(n),rng.normal(size=(n,p-1))])
    base=np.linspace(-160.,160.,n)
    actual=b+rng.uniform(-30.,30.,n)
    actual=np.maximum(1.,actual)
    response_map=ResponseDC(b,.8,30.,180.,1.)
    obj=DCObjective(b,X,base,response_map,mode="CASE",
                    actual=actual,components=4,lam=.02,
                    penalize=np.r_[0.,np.ones(p-1)])
    center=np.array([2.,-3.,4.,1.,-2.])
    gradient=obj.parts(center)[3]
    solver=CaseConvexQP(obj,np.full(p,-40.),np.full(p,40.))
    try:
        trial,status=solver.solve(center,gradient,prox=.001)
        assert status["certified"],status
        G,_,_,_=obj.parts(trial)
        reconstructed=G-gradient@trial+.0005*np.sum((trial-center)**2)
        assert reconstructed==pytest.approx(
            status["objective"],rel=1e-6,abs=1e-5)
        assert np.isfinite(obj.value(trial))
    finally:
        solver.close()


def test_vf_cut_cap_does_not_stop_accepted_progress(monkeypatch):
    """A hit cut budget is not a spurious pDCA stopping condition."""
    from surgery import nonmonotone_learning as nl
    class QuadraticObjective:
        mode="VF"
        def parts(self,w):
            return float(w@w),0.,2.*w,np.zeros_like(w)
        def value(self,w):
            return float(w@w)
    calls=[]
    def capped(obj,center,*args,**kwargs):
        calls.append(len(calls))
        return center*.5,dict(
            status="INNER_CUT_CAP",trace=[dict(gap=1.)],
            certified=False,gap=1.)
    monkeypatch.setattr(nl,"_solve_qp_cuts",capped)
    ans=nl.pdca(QuadraticObjective(),np.array([1.,2.]),
                radius=10.,outer=3,backend="scipy")
    assert len(calls)==3
    assert len(ans["iterations"])==3
    assert all(row["accepted"] for row in ans["iterations"])
    assert ans["objective"]<1.



def test_case_exact_qp_improves_with_70_shared_features():
    """Regression for high-dimensional CASE failing at the first bundle cut cap."""
    gp=pytest.importorskip("gurobipy")
    try:
        probe=gp.Model()
        probe.Params.OutputFlag=0
        z=probe.addVar(lb=0.,ub=1.)
        probe.setObjective(z*z)
        probe.optimize()
        probe.dispose()
    except gp.GurobiError as exc:
        pytest.skip(str(exc))
    rng=np.random.default_rng(2061)
    n,p=120,70
    X=rng.normal(size=(n,p))
    X[:,0]=1.
    booked=np.full(n,150.)
    actual=booked+np.where(X[:,1]+.7*X[:,2]>0.,18.,-15.)
    response_map=ResponseDC(booked,.8,30.,180.,1.)
    obj=DCObjective(booked,X,np.zeros(n),response_map,mode="CASE",
                    actual=actual,components=12,lam=.002,
                    penalize=np.r_[0.,np.ones(p-1)])
    start=np.zeros(p)
    result=pdca(obj,start,radius=40.,outer=4,backend="gurobi")
    assert result["iterations"]
    assert all(r["inner_status"].startswith("CASE_QP_")
               for r in result["iterations"])
    assert result["objective"]<obj.value(start)-1e-3
    assert any(r["accepted"] for r in result["iterations"])
