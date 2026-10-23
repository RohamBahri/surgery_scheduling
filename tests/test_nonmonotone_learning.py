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
    try:
        result=pdca(obj,start,radius=40.,outer=4,backend="gurobi")
    except gp.GurobiError as exc:
        if "size-limited license" in str(exc).lower():
            pytest.skip("70-feature regression requires full Gurobi license")
        raise
    assert result["iterations"]
    assert all(r["inner_status"].startswith("CASE_QP_")
               for r in result["iterations"])
    assert result["objective"]<obj.value(start)-1e-3
    assert any(r["accepted"] for r in result["iterations"])



def test_fast_planned_cost_matches_current_weekly_evaluate_exactly():
    from surgery.nonmonotone_learning import planned_cost_fast
    from surgery.weekly_planner import evaluate
    c=toy_week()
    rng=np.random.default_rng(2101)
    for z in ([0,0,0],[0,1,1],[1,0,1],[1,1,1]):
        for durations in (c.booked, c.actual, rng.uniform(1.,650.,size=c.n)):
            expected=evaluate(c,z,durations)["cost"]
            assert planned_cost_fast(c,z,durations)==pytest.approx(
                expected,rel=0,abs=1e-9)


def test_pdca_progress_jsonl_persists_iteration_diagnostics(tmp_path,capsys):
    b=np.array([120.,100.,40.,15.])
    X=np.array([[1.,0.],[1.,1.],[1.,-1.],[1.,2.]])
    m=ResponseDC(b,.8,30.,180.,1.)
    obj=DCObjective(b,X,np.zeros(len(b)),m,mode="CASE",
                    actual=np.array([107.,111.,45.,12.]),
                    components=1,lam=.001,penalize=np.array([0.,1.]))
    file=tmp_path/"case_progress.jsonl"
    ans=pdca(obj,np.zeros(2),radius=50.,outer=3,
             max_cuts=35,backend="scipy",
             progress_label="CASE",progress_path=file)
    import json
    lines=[json.loads(row) for row in file.read_text().splitlines()]
    assert len(lines)==len(ans["iterations"])
    assert all("elapsed_seconds" in row and "weights" in row
               and "inner_status" in row for row in lines)
    assert "CASE pDCA" in capsys.readouterr().out


def test_case_csv_writer_has_all_raw_display_reviewed_and_features(tmp_path):
    from pilot.run_nonmonotone_learning import save_case_table
    import csv
    b=np.array([10.,100.])
    X=np.ones((2,5))
    m=ResponseDC(b,.8,30.,180.,1.)
    data=dict(case_id=np.array([101,102]),booked=b,
              actual=np.array([12.,90.]))
    out=tmp_path/"train.csv"
    save_case_table(out,data,np.array([18.,-160.]),
                    np.array([35.,-90.]),X,m)
    with out.open(newline="") as handle:
        records=list(csv.DictReader(handle))
    assert len(records)==2
    assert "case_display" in records[0]
    assert "vf_reviewed" in records[0]
    assert sum(k.startswith("feature_") for k in records[0])==5
    assert float(records[1]["case_reviewed"]) == pytest.approx(
        m.reviewed(np.array([18.,-160.]))[1])



def test_vf_pdca_and_vf_directions_are_separate_learning_algorithms(tmp_path):
    from surgery.nonmonotone_learning import mm_vf
    week=toy_week()
    items=[("toy",week,np.arange(week.n))]
    base=np.array([12.,-4.,40.])
    H=np.array([[1.,0.],[.5,1.],[-1.,.7]])
    response_map=ResponseDC(week.booked,.8,30.,180.,1.)
    def planner(items,d,k):
        return {"toy":dict(assignment=[0,0,0],bound=0.,
                           complete=False,status="TIME_LIMIT")}
    results={}
    for method in ("pdca","directions"):
        path=tmp_path/method
        results[method]=mm_vf(items,week.booked,H,base,response_map,
             week.actual,root=path,lam=.001,
             initial_libraries={"toy":[[0,0,0],[0,1,1]]},
             iterations=2,radius=16.,pdca_steps=3,max_cuts=25,
             backend="scipy",solve_plans=planner,optimizer=method)
        assert results[method]["optimizer"]==method
        assert (path/"vf_library_progress.json").exists()
        assert results[method]["libraries"]["toy"]
        for row in results[method]["history"]:
            if row["accepted"]:
                assert row["delta"]>0.
        assert "search" in results[method]["history"][0]
    assert "directions_checked" in results["directions"]["history"][0]["search"]
    assert "pdca_status" in results["pdca"]["history"][0]["search"]


def test_shared_realized_cost_is_not_predicted_library_cost():
    from surgery.nominal_shared import score,proxy
    from surgery.nonmonotone_learning import planned_cost_fast
    c=toy_week()
    items=[("toy",c,np.arange(c.n))]
    assignment=[0,0,0]
    theta=np.array([2.,-3.])
    rec={"toy":dict(assignment=assignment,complete=True)}
    direct=score(items,rec,theta,.01)
    expected=planned_cost_fast(c,assignment,c.actual)+.01*np.abs(theta).sum()
    assert direct==pytest.approx(expected)
    assert score(items,{"toy":dict(assignment=assignment,complete=False)},
                 theta,.01) is None
    libraries={"toy":[assignment,[0,1,1]]}
    costs={"toy":[planned_cost_fast(c,z,c.actual)
                  for z in libraries["toy"]]}
    predicted=c.booked+np.array([30.,-20.,5.])
    surrogate=proxy(items,libraries,predicted,costs,theta,.01)
    j=min(range(2),key=lambda k:(planned_cost_fast(c,libraries["toy"][k],
                                 predicted),tuple(libraries["toy"][k])))
    assert surrogate==pytest.approx(costs["toy"][j]+.01*np.abs(theta).sum())


def test_nominal_four_method_report_uses_five_way_common_coverage(tmp_path):
    from pilot.run_nominal_four_methods import ARMS,report
    import json
    rows=[]
    for key,week,cases in (("a","2013-01-07",3),("b","2013-01-14",2)):
        for i,arm in enumerate(ARMS):
            rows.append(dict(component=key,method=arm,week=week,
                             cases=cases,complete=not(key=="b" and
                                      arm=="VF_DIRECTIONS"),
                             cost=float(10+i),mae=2.,idle_minutes=3.,
                             overtime_minutes=4.))
    (tmp_path/"evaluation_2013.json").write_text(json.dumps({"entries":rows}))
    report(tmp_path)
    out=json.loads((tmp_path/"summary.json").read_text())
    assert out["common_certified_components"]==1
    assert out["common_certified_cases"]==3
    assert len(out["paired_cost_differences"])==7
    assert out["VF_DIRECTIONS"]["cost_per_case"]==pytest.approx(13./3)
    assert out["arm_coverage"]["VF_DIRECTIONS"]["certified_components"]==1


def test_single_command_runner_cli_and_method_roster():
    from pilot.run_nominal_four_methods import ARMS,CASE_L1_SCALE
    assert ARMS==("BOOKED","CASE","VF_PDCA","VF_DIRECTIONS","SHARED")
    assert CASE_L1_SCALE==pytest.approx(.75)
    import subprocess,sys
    from pathlib import Path
    entry=Path(__file__).resolve().parents[1]/"pilot"/"run_nominal_four_methods.py"
    run=subprocess.run([sys.executable,str(entry),"--help"],
        cwd=entry.parents[1],capture_output=True,text=True,timeout=25)
    assert run.returncode==0,run.stderr
    assert "--shared-rounds" in run.stdout
    assert "--plan-seconds" in run.stdout



def test_four_method_train_to_evaluation_smoke_with_stubbed_optimizers(
        tmp_path,monkeypatch):
    """Check the single-run wiring without an unrestricted Gurobi license."""
    import json
    from pilot import run_nominal_four_methods as runner
    from surgery import nonmonotone_learning
    c=toy_week()
    items=[("toy",c,np.arange(c.n))]
    data=dict(X=np.array([[1.,-1.],[1.,0.],[1.,1.]]),
              booked=c.booked.copy(),actual=c.actual.copy(),
              case_id=c.case_ids.copy())
    ctx=dict(train=items,eval=items,fit=data,held=data,
             H=np.ones((c.n,5)),H_eval=np.ones((c.n,5)))
    class FakeCache:
        def solve(self,items,d,tag,tie=False):
            return {key:dict(assignment=[0]*week.n,
                             complete=True,status="OPTIMAL",bound=0.)
                    for key,week,rows in items}
    def case_fit(ctx,items,behavior,root,*args,**kwargs):
        (root/"case_policy.json").write_text(json.dumps(
            dict(lambda_case=.2,objective=1.,initial_objective=1.2)))
        return np.array([0.,1.])
    def features(ctx,items,case_raw,behavior,cache,stage,scaler=None):
        H=np.column_stack([np.ones(c.n),
             np.arange(c.n),np.ones(c.n)*.5,np.zeros(c.n),np.ones(c.n)])
        return H,(np.zeros(5),np.ones(5)),FakeCache().solve(items,None,"draft")
    def vf(items,booked,H,case_raw,behavior,actual,*,root,
           initial_libraries,optimizer,**kw):
        assert optimizer in ("pdca","directions")
        (root/"vf_library_progress.json").parent.mkdir(parents=True,exist_ok=True)
        (root/"vf_library_progress.json").write_text("{}")
        return dict(theta=[.5 if optimizer=="pdca" else -.5,0.,0.,0.,0.],
             status="RELATIVE_IMPROVEMENT",
             history=[dict(optimizer=optimizer,accepted=True)],
             libraries=initial_libraries,optimizer=optimizer)
    def shared(*args,root,**kwargs):
        root.mkdir(exist_ok=True,parents=True)
        (root/"shared_progress.json").write_text("{}")
        return dict(theta=[0.,0.,0.,0.,0.],
                    score=22.,status="COMPLETE")
    monkeypatch.setattr(runner,"case_fit",case_fit)
    monkeypatch.setattr(runner,"five_features",features)
    monkeypatch.setattr(runner,"mm_vf",vf)
    monkeypatch.setattr(runner,"train_shared",shared)
    monkeypatch.setattr(runner,"train_lambda",
                        lambda x,n,**kw:.15)
    from types import SimpleNamespace
    args=SimpleNamespace(case_steps=4,cuts=10,vf_outers=2,
                         vf_pdca_steps=3,shared_rounds=2)
    chosen=runner.train(ctx,args,tmp_path,FakeCache())
    assert set(chosen["policies"])=={
        "CASE","VF_PDCA","VF_DIRECTIONS","SHARED"}
    runner.evaluate(ctx,args,tmp_path,FakeCache(),chosen)
    out=json.loads((tmp_path/"summary.json").read_text())
    assert out["common_certified_components"]==1
    assert len(json.loads((tmp_path/"evaluation_2013.json").read_text())["entries"])==5
    import csv
    with (tmp_path/"cases_test.csv").open(newline="") as handle:
        case_rows=list(csv.DictReader(handle))
    assert len(case_rows)==3
    assert "VF_DIRECTIONS_reviewed" in case_rows[0]
    assert "SHARED_display" in case_rows[0]
