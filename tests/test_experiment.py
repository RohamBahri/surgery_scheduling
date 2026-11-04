"""Regression tests for the single-dataset, six-policy retrospective experiment."""
from types import SimpleNamespace

import numpy as np
import pytest

from pilot import experiment


def test_default_scenarios_and_immutable_contract(tmp_path):
    context = tmp_path / "context.pkl"
    context.write_bytes(b"toy-context")
    args = experiment.parse_args(["--context", str(context), "--out", str(tmp_path / "run")])
    assert len(args.scenarios) == 8
    assert ("saturating", .3, 15.) in args.scenarios
    assert ("saturating", .8, 90.) in args.scenarios
    assert len(experiment.METHODS) == 6
    experiment.prepare_output(args)
    experiment.prepare_output(args)
    args.lambda_decision = .3
    with pytest.raises(ValueError, match="settings changed"):
        experiment.prepare_output(args)


def test_calendar_week_is_complete_only_if_every_method_certified():
    scenario = ("saturating", .5, 30.)
    sid = experiment.scenario_id(*scenario)
    rows = []
    for week in ("2013-01-07", "2013-01-14"):
        for component in ("a", "b"):
            for index, method in enumerate(experiment.METHODS):
                rows.append(dict(scenario=sid, week=week, component=component,
                    method=method, complete=not (
                        week == "2013-01-14" and component == "b" and
                        method == "ACCURACY_TRAINED"),
                    cases=2, cost=10.-index, planned_cost=10.-index,
                    asymmetric_error=0., loss_bound=10.-index,
                    idle=0., overtime=0., mae=1., displayed_mae=1.))
    assert set(experiment.complete_week_keys(rows, sid)) == {"2013-01-07"}
    checks = {sid: dict(decision_accuracy_ratio=1., decision_accuracy_met=True,
                        response_accuracy_ratio=1., response_accuracy_met=True,
                        selected_mu_decision=2., selected_mu_response=2.)}
    table, intervals, weekly, complete_weeks = experiment.summarize(
        rows, [scenario], checks)
    assert complete_weeks == ["2013-01-07"]
    assert table[0]["weeks"] == 1
    assert table[0]["BOOKED_cost_per_week"] == 20.
    assert table[0]["DECISION_AWARE_cost_per_week"] == 14.
    assert intervals[sid]["DECISION_AWARE_minus_BOOKED"]["mean_weekly_difference"] == -6.
    assert intervals[sid]["ACCURACY_TRAINED_minus_BOOKED"]["mean_weekly_difference"] == -2.
    assert intervals[sid]["RESPONSE_AWARE_minus_BOOKED"]["mean_weekly_difference"] == -4.
    assert intervals[sid]["PERFECT_ADVICE_minus_BOOKED"]["mean_weekly_difference"] == -8.
    assert intervals[sid]["HINDSIGHT_minus_BOOKED"]["mean_weekly_difference"] == -10.
    assert table[0]["PERFECT_ADVICE_gap_closed_fraction"] == pytest.approx(.8)
    assert table[0]["ACCURACY_TRAINED_gap_closed_fraction"] == pytest.approx(.2)
    assert len(weekly) == len(experiment.METHODS)


def test_initializer_is_training_only(tmp_path, monkeypatch):
    context = tmp_path / "context.pkl"
    context.write_bytes(b"toy-context")
    args = experiment.parse_args(["--context", str(context), "--out", str(tmp_path / "run")])
    X = np.column_stack([np.ones(3), [1., 2., 3.]])
    training = dict(fit=dict(X=X, actual=np.array([5., 7., 9.]),
                             booked=np.array([4., 6., 8.])), train=[None])
    calls = []
    monkeypatch.setattr(experiment.training, "train_lambda", lambda *a, **kw: .1)
    def solve(*a):
        calls.append(a)
        return np.array([1., 0.]), {"success": True}
    monkeypatch.setattr(experiment.training, "lad_start", solve)
    first = experiment.fit_initializer(training, args)
    second = experiment.fit_initializer(training, args)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1
