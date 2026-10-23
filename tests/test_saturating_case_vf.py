"""Exact saturated-link B regression tests and one-command runner smoke."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from surgery.nonmonotone_learning import (
    ResponseDC, DCObjective, response, response_saturating,
    library_value, room_matrix)


def toy():
    from surgery.weekly_data import Week
    return Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1",),
        np.zeros(3,dtype=int),np.zeros(3,dtype=int),
        ((0,1),),((0,),),
        np.array([190.,180.,160.]),np.array([195.,175.,170.]),()).validate()


def test_formulation_B_matches_two_case_counterexample():
    scores=np.array([20.,60.,150.,300.,-60.])
    b=response_saturating(scores)
    c=response(scores)
    assert np.allclose(b,[16.,24.,24.,24.,-24.])
    assert np.allclose(c[:3],[16.,18.,0.])
    assert np.allclose(b[:2],[16.,24.])
    assert not np.array_equal(b,c)


@pytest.mark.parametrize("cap",[20.,30.,180.])
def test_saturated_DC_is_exact_with_after_response_floor(cap):
    booked=np.array([1.,2.,12.,30.,50.,120.,500.])
    m=ResponseDC(booked,.8,30.,cap,1.,link="saturating")
    rng=np.random.default_rng(121)
    for raw in (np.full(len(booked),-400.),
                np.full(len(booked),400.),
                np.linspace(-300,300,len(booked))):
        m.verify(raw)
    for _ in range(30):
        raw=rng.uniform(-400.,400.,len(booked))
        m.verify(raw)
        expected=np.maximum(
            response_saturating(np.clip(raw,-cap,cap),.8,30.),
            1.-booked)
        assert np.allclose(m.correction(raw),expected,atol=1e-10)
        assert np.allclose(m.parts(raw)[0]-m.parts(raw)[1],
                           expected,atol=1e-9)
        assert np.allclose(m.reviewed(raw),booked+expected)
        assert np.all(np.abs(m.display(raw))<=min(cap,30.)+1e-9)


def test_saturating_QP_objective_and_casewise_unrestricted_scores():
    booked=np.array([190.,180.,160.])
    actual=np.array([195.,175.,170.])
    X=np.array([[1.,1.],[1.,3.],[1.,10.]])
    m=ResponseDC(booked,.8,30.,180.,1.,link="saturating")
    obj=DCObjective(booked,X,np.zeros(3),m,mode="CASE",actual=actual,
                    components=1,lam=.1,
                    penalize=np.array([0.,1.]))
    w=np.array([0.,20.])  # case scores 20,60,200: unrestricted
    direct=(1.75*np.maximum(actual-m.reviewed(X@w),0.)
            +np.maximum(m.reviewed(X@w)-actual,0.)).sum()+.1*abs(w[1])
    assert obj.value(w)==pytest.approx(direct,rel=1e-10,abs=1e-8)
    assert np.array_equal(m.correction(X@w),[16.,24.,24.])


def test_saturated_VF_touches_fixed_physical_schedules():
    c=toy()
    items=[("toy",c,np.arange(c.n))]
    libraries={"toy":[[0,0,0],[0,1,1]]}
    H=np.array([[1.,-1.],[0.,1.],[2.,3.]])
    base=np.array([20.,60.,-100.])
    m=ResponseDC(c.booked,.8,30.,180.,1.,link="saturating")
    w=np.array([10.,4.])
    d=m.reviewed(base+H@w)
    from surgery.nonmonotone_learning import choose_library_schedules
    plans=choose_library_schedules(items,libraries,d)
    S,_=room_matrix(items,plans,c.n)
    obj=DCObjective(c.booked,H,base,m,mode="VF",actual=c.actual,
                    rooms=S,components=1,lam=.1)
    assert obj.value(w)==pytest.approx(
        library_value(items,libraries,d,c.actual,.1,w),abs=1e-8)


def test_saturated_exact_case_qp_if_gurobi_available():
    gp=pytest.importorskip("gurobipy")
    from surgery.nonmonotone_learning import CaseConvexQP
    b=np.linspace(1.,200.,18)
    a=np.maximum(1.,b+np.sin(b)*12.)
    X=np.column_stack([np.ones(len(b)),np.linspace(-4,4,len(b)),
                       np.linspace(0,1,len(b))])
    link=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    obj=DCObjective(b,X,np.zeros(len(b)),link,mode="CASE",actual=a,
                    components=4,lam=.1,penalize=np.array([0.,1.,1.]))
    x=np.array([0.,5.,1.])
    solver=None
    try:
        solver=CaseConvexQP(obj,np.full(3,-100.),np.full(3,100.))
        grad=obj.parts(x)[3]
        sol,rec=solver.solve(x,grad,.001)
        assert rec["certified"]
        G=obj.parts(sol)[0]
        assert G-grad@sol+.0005*np.sum((sol-x)**2)==pytest.approx(
            rec["objective"],rel=1e-6,abs=1e-5)
    except gp.GurobiError as exc:
        if "size-limited license" in str(exc).lower():
            pytest.skip("Gurobi size-limited license")
        raise
    finally:
        if solver is not None:
            solver.close()


def test_saturating_runner_train_and_evaluate_wiring(tmp_path,monkeypatch):
    from pilot import run_saturating_case_vf as runner
    c=toy()
    items=[("toy",c,np.arange(c.n))]
    data=dict(X=np.array([[1.,-1.],[1.,0.],[1.,1.]]),
         booked=c.booked.copy(),actual=c.actual.copy(),
         case_id=np.array([101,102,103]))
    ctx=dict(train=items,eval=items,fit=data,held=data,
             H=np.zeros((c.n,5)),H_eval=np.zeros((c.n,5)))
    class Cache:
        def solve(self,items,d,tag,tie=False):
            return {key:dict(assignment=[0]*week.n,
                 complete=True,status="OPTIMAL",bound=0.)
                 for key,week,rows in items}
    def fitted_case(ctx,items,link,root,*args,**kwargs):
        assert link.link=="saturating"
        assert kwargs["lambda_scale"]==pytest.approx(.75)
        (root/"case_policy.json").write_text(json.dumps({
             "lambda_case":.12,"objective":12.,"initial_objective":14.}))
        return np.array([0.,1.])
    def features(ctx,items,scores,link,cache,stage,scaler=None):
        H=np.column_stack([np.ones(3),np.arange(3),np.ones(3)*.5,
                           np.zeros(3),np.ones(3)])
        return H,(np.zeros(5),np.ones(5)),Cache().solve(items,None,"draft")
    def vf(items,b,H,raw,link,actual,*,root,optimizer,**kwargs):
        assert optimizer=="pdca"
        assert link.link=="saturating"
        assert kwargs["lam"]==pytest.approx(.5*.4)
        return dict(theta=[1.,0.,0.,0.,0.],
                    status="RELATIVE_IMPROVEMENT",history=[])
    monkeypatch.setattr(runner,"case_fit",fitted_case)
    monkeypatch.setattr(runner,"five_features",features)
    monkeypatch.setattr(runner,"train_lambda",lambda x,n:.4)
    monkeypatch.setattr(runner,"mm_vf",vf)
    args=SimpleNamespace(case_steps=5,cuts=20,vf_outers=2,vf_pdca_steps=3)
    selected=runner.train(ctx,args,tmp_path,Cache())
    assert selected["response_link"]=="saturating"
    runner.evaluate_2013(ctx,tmp_path,Cache(),selected)
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["common_certified_components"]==1
    entries=json.loads((tmp_path/"evaluation_2013.json").read_text())["entries"]
    assert {row["method"] for row in entries}==set(runner.ARMS)
    assert (tmp_path/"cases_test.csv").exists()
    assert "VF_B_minus_CASE_B" in summary["paired_cost_differences"]
