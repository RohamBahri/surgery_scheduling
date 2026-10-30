"""Regression checks for the final four-method behavioral grid."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pilot import run_vf_grid as grid


def test_six_scenarios_and_contract(tmp_path):
    context = tmp_path / "context.pkl"
    context.write_bytes(b"test-context")
    a = grid.parse_args(["--context", str(context), "--out", str(tmp_path / "grid")])
    assert len(a.scenarios) == 6
    assert {link for link, _, _ in a.scenarios} == {"saturating"}
    assert grid.scenario_id("saturating", .5, 30) == "saturating_a0p5_h30"
    grid.ensure_contract(a)
    grid.ensure_contract(a)
    with pytest.raises(SystemExit):
        grid.parse_args(["--context", str(context), "--alphas", "1"])
    a.accuracy_mu = 5
    with pytest.raises(ValueError, match="changed"):
        grid.ensure_contract(a)


def test_grid_uses_four_way_common_certified_components():
    a = SimpleNamespace(scenarios=[("saturating", .5, 20.),
                                   ("saturating", .8, 60.)])
    methods = grid.METHODS
    entries = []
    checks = {}
    for s in a.scenarios:
        sid = grid.scenario_id(*s)
        checks[sid] = dict(link=s[0], alpha=s[1], h=s[2],
                           training_accuracy_ratio=1.01,
                           training_accuracy_met=True)
        for comp in ("a", "b"):
            for method, cost in zip(methods, (10., 9., 8., 5.)):
                entries.append(dict(scenario=sid, component=comp, method=method,
                    complete=not (s[1] == .8 and comp == "b" and method == "NAIVE"),
                    cases=2, week="2013-01-07", cost=cost,
                    mae=1., idle=0., overtime=0., over_720_rooms=0))
    table, confidence, report = grid.summarize(entries, checks, a)
    assert report["grid_common_certified_components"] == 1
    assert report["grid_common_certified_cases"] == 2
    global_rows = [r for r in table if r["sample"] == "grid_common_certified"]
    assert len(global_rows) == 2
    assert all(r["BOOKED_cost_per_case"] == 5. for r in global_rows)
    assert all(r["VF_minus_NAIVE"] == -0.5 for r in global_rows)
    assert all(r["VF_gap_closed_fraction"] == .4 for r in global_rows)
    assert all("VF_minus_BOOKED" in confidence[r["scenario"]] for r in global_rows)


def test_naive_is_fitted_only_once_and_has_no_response_parameters(tmp_path, monkeypatch):
    context = tmp_path / "context.pkl"
    context.write_bytes(b"test-data")
    a = grid.parse_args(["--context", str(context), "--out", str(tmp_path / "grid")])
    ctx = dict(fit=dict(X=np.column_stack([np.ones(3), [1., 2., 3.]]),
                        actual=np.array([5., 7., 9.]),
                        booked=np.array([4., 6., 8.])), train=[None])
    calls = []
    monkeypatch.setattr(grid.engine, "train_lambda", lambda *args, **kw: .1)
    def lad(X, errors, lam, ncomp, box):
        calls.append((lam, ncomp, box))
        return np.array([1., 0.]), dict(success=True)
    monkeypatch.setattr(grid.engine, "lad_start", lad)
    first = grid.fit_naive(ctx, a)
    second = grid.fit_naive(ctx, a)
    assert np.array_equal(first, second)
    assert len(calls) == 1
    record = json.loads((a.out / "naive_lad.json").read_text())
    assert "alpha" not in record and "h" not in record
    assert record["method"] == "NAIVE_LAD"
