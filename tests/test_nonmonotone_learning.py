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
    b=np.array([1.,2.,3.,8.,12.,20.,40.,100.,560.])
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
        assert m.reviewed(np.full(len(b),-30.))[1]==pytest.approx(1.)
        # A sufficiently extreme display is REJECTED, not shortened.
        assert m.reviewed(np.full(len(b),-180.))[1]==pytest.approx(2.)
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
