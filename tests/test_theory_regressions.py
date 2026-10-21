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
    fake=[SimpleNamespace(targets=None)]
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
    c=SimpleNamespace(booked=np.array([100.]))
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
