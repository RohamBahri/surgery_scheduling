"""Regression tests for statistical safeguards, global bound accounting and pooling."""
from types import SimpleNamespace
import numpy as np
import pytest

from surgery import theory_regions
from surgery.theory_reporting import summarize


def test_tolerance_pruned_regions_retain_their_dual_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(theory_regions, "_evaluate", lambda *a, **kw: 10.0)
    monkeypatch.setattr(theory_regions, "_node_lower",
                        lambda *a, **kw: (9.9995, 0))
    fake=[SimpleNamespace(key='fixture',week=SimpleNamespace(n=1),
                          targets=None,H=np.ones((1,1)))]
    answer=theory_regions.global_shared_search(
        fake,np.zeros(1),out=tmp_path,seconds=5.,solve_seconds=.1,
        max_nodes=5,radius=1.,gap_tolerance=.001)
    assert answer["lower"] == pytest.approx(9.9995)
    assert answer["upper"] == pytest.approx(10.)
    assert answer["gap"] == pytest.approx(.0005)
    assert answer["status"] == "GAP_CERTIFIED"


def test_training_bank_b_is_independent_and_changes_each_round(monkeypatch):
    from pilot import run_theory_full as runner
    seeds=[]
    monkeypatch.setattr(runner,"reviewed",lambda booked,pred: booked+pred)
    monkeypatch.setattr(runner,"reviewed_policy",
                        lambda c,rows,H,d0,coef: d0)
    def fake_scenarios(c,d0,pools,seed,count):
        seeds.append(seed)
        return np.ones((count,1))
    monkeypatch.setattr(runner,"scenarios",fake_scenarios)
    monkeypatch.setattr(runner,"scores",lambda *a,**kw: 1.)
    c=SimpleNamespace(booked=np.array([100.]),n=1)
    task=("id",c,np.array([0]),np.array([0.]),np.ones((1,5)),None,
          [[0.,0.,0.,0.,0.]],"SHARED",{},1.,0)
    runner._train_component(task)
    runner._train_component(task[:-1]+(-1,))
    runner._train_component(task[:-1]+(-2,))
    assert seeds==[20261020,20261021,20262030]


def _row(day,cases,case_cost,shared_cost,complete=True):
    return dict(monday=day,cases=cases,site="TGH",
        methods={
          "CASE":dict(status="COMPLETE",realized={"0":case_cost,"1":case_cost,"2":case_cost}),
          "SHARED":dict(status="COMPLETE" if complete else "UNRESOLVED",
                        realized={"0":shared_cost,"1":shared_cost,"2":shared_cost})},
        hindsight=dict(complete=True,cost=case_cost/2))


def test_pooled_reporting_is_case_weighted_and_all_or_nothing():
    rows=[_row("2012-01-02",3,30.,27.),
          _row("2012-01-09",7,70.,63.)]
    out=summarize(rows,draws=1000)
    assert out["n_cases"]==10 and out["n_mondays"]==2
    assert out["arms"]["CASE"]["mean_realized_cost_per_case"]==pytest.approx(10.)
    assert out["arms"]["SHARED"]["mean_realized_cost_per_case"]==pytest.approx(9.)
    assert out["paired_vs_CASE"]["SHARED"]["difference"]==pytest.approx(-1.)
    rows[1]["methods"]["SHARED"]["status"]="UNRESOLVED"
    unresolved=summarize(rows,draws=100)
    assert unresolved["arms"]["SHARED"]["mean_realized_cost_per_case"] is None
    assert "SHARED" not in unresolved["paired_vs_CASE"]


def test_exact_planner_cache_never_caches_incomplete_solve(tmp_path):
    from surgery.theory_cache import connect,cached_deploy
    c=SimpleNamespace(n=2,case_ids=np.array([10,11]),booked=np.array([100.,120.]))
    d=np.array([101.,123.]);seeds=(0,1,2)
    conn=connect(tmp_path/"cache.sqlite3")
    called=[]
    def certify(component,durations,seed):
        called.append(seed)
        return (0,1) if seed!=2 else (1,0)
    x=cached_deploy(c,d,seeds,certify,conn)
    y=cached_deploy(c,d,seeds,certify,conn)
    assert x==y and called==[0,1,2]
    cached_deploy(c,d+1,seeds,certify,conn)
    assert len(called)==6
    def fail(component,durations,seed):
        if seed==1:raise RuntimeError("uncertified")
        return (0,1)
    with pytest.raises(RuntimeError):
        cached_deploy(c,d+2,seeds,fail,conn)
    # No partial schedule was inserted on failure.
    assert conn.execute("SELECT COUNT(*) FROM deployments").fetchone()[0]==2
    conn.close()


def test_global_search_restart_restores_queue_and_bounds(tmp_path,monkeypatch):
    monkeypatch.setattr(theory_regions,"_evaluate",lambda *a,**kw: 10.)
    monkeypatch.setattr(theory_regions,"_node_lower",lambda *a,**kw: (0.,0))
    p=SimpleNamespace(key="restart",week=SimpleNamespace(n=1),
                      targets=None,H=np.ones((1,1)))
    kw=dict(out=tmp_path,seconds=5.,solve_seconds=.1,max_nodes=1,radius=2.)
    a=theory_regions.global_shared_search([p],np.zeros(1),**kw)
    frozen=theory_regions.global_shared_search([p],np.zeros(1),**kw)
    assert frozen["nodes"]==a["nodes"]  # Finished models are immutable.
    (tmp_path/"result.json").unlink()  # Simulate an interrupted run before final save.
    b=theory_regions.global_shared_search([p],np.zeros(1),**kw)
    assert a["nodes"]==1 and b["nodes"]==2
    assert b["lower"]<=b["upper"]
    assert b["gap"]>=0.


def test_saa_reference_bounds_remain_separate_from_advice_policies():
    from surgery.theory_reporting import summarize
    def mk(day,sa,pa):
        return dict(monday=day,cases=2,site="TGH",
            methods={"CASE":dict(status="COMPLETE",realized={"0":10.},
                                  model={"A":8.}),
                     "PER_WEEK_ROBUST":dict(status="COMPLETE",realized={"0":9.},
                                            model={"A":pa})},
            saa_reference=dict(certified=True,lower=sa,upper=sa,
                    assignment=[0,1],realized=8.,seconds=1.,
                    model={"A":sa,"C":sa+.5}))
    result=summarize([mk("2012-01-02",4.,6.),mk("2012-01-09",5.,7.)],draws=100)
    assert result["saa_reference"]["model_A_lower_per_case"]==pytest.approx(2.25)
    assert result["saa_reference"]["model_C_feasible_per_case"]==pytest.approx(2.5)
    assert result["inducibility_bounds"]["PER_WEEK_ROBUST"]["gap_lower"]==0
    assert result["inducibility_bounds"]["PER_WEEK_ROBUST"]["gap_upper"]==pytest.approx(1.)


def test_parallel_candidate_shards_cover_every_score_without_gurobi(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pilot import run_theory_full as runner
    c=SimpleNamespace(n=1,booked=np.array([100.]),case_ids=np.array([701]),
                      group='TGH',monday='2012-02-06')
    runner._SCORE_STATE=dict(base=np.array([110.]),H=np.zeros((1,5)),
                       method='SHARED',pools={},seconds=1.,planner_stats={})
    monkeypatch.setattr(runner,"scenarios",lambda *a,**kw:np.ones((50,1)))
    monkeypatch.setattr(runner,"scores",lambda *a,**kw:7.)
    coeff=[[float(k)]*5 for k in range(13)]
    data=dict(X=np.ones((1,1)),case_w=np.array([110.]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        score,valid=runner.parallel_score(
             [("TGH|2012-02-06|701",c,np.array([0]))],data,np.zeros((1,5)),
             coeff,'SHARED',{},1.,3,0,tmp_path,pool=pool,task_seconds=30.)
    assert valid.all() and list(score)==[7.]*len(coeff)
    # Exact checkpoint restart needs no new solver calls.
    with ThreadPoolExecutor(max_workers=3) as pool:
        again,valid2=runner.parallel_score(
             [("TGH|2012-02-06|701",c,np.array([0]))],data,np.zeros((1,5)),
             coeff,'SHARED',{},1.,3,0,tmp_path,pool=pool,task_seconds=30.)
    assert np.array_equal(score,again) and valid2.all()
    assert len(list(tmp_path.glob("*_shard_*.json")))>1
    runner._SCORE_STATE=None


def test_expired_candidate_shard_is_unresolved_not_crash(monkeypatch):
    from pilot import run_theory_full as runner
    c=SimpleNamespace(n=1,booked=np.array([100.]))
    runner._SCORE_STATE=dict(base=np.array([110.]),H=np.zeros((1,5)),
                      method='SHARED',pools={},seconds=5.,planner_stats={})
    monkeypatch.setattr(runner,"scenarios",lambda *a,**kw:np.ones((50,1)))
    key,n,results=runner._train_component(
        ('component',c,np.array([0]),[[0.]*5],0,[0],0.))
    assert key=='component' and n==1
    assert not results[0][1]['complete']
    assert results[0][1]['status']=='TASK_WALL_LIMIT'
    runner._SCORE_STATE=None


def test_preprocessing_deadline_preserves_full_coverage_with_zero_dual(tmp_path,monkeypatch):
    c=SimpleNamespace(n=1,booked=np.array([100.]))
    monkeypatch.setattr(theory_regions,"scenarios",
                        lambda *a,**kw:np.ones((50,1)))
    data=dict(X=np.zeros((1,1)),case_w=np.zeros(1))
    ans=theory_regions.build_problems(
        [('one',c,np.array([0]))],data,np.zeros((1,5)),np.zeros(5),
        {},tmp_path,seconds=5.,cores=1,stage_seconds=0.)
    assert len(ans)==1 and ans[0].lower==0.
    rec=__import__('json').loads((tmp_path/'saa_preprocessing.json').read_text())
    assert rec['unresolved_saa']==1 and rec['ready_components']==1


def test_single_split_run_has_explicit_train_test_design():
    from pathlib import Path
    src=(Path(__file__).resolve().parents[1]/'pilot'/'run_theory_full.py').read_text()
    assert "for fold in ('2013',):" in src
    assert 'training_cases=24506,test_cases=8774' in src
    assert "choices=['2013']" in src
