import numpy as np
import pytest
from surgery.weekly_data import Week
from surgery.advice_comparison import (reviewed, display, scenarios, prescription,
                                       case_features, transition)
from surgery.weekly_planner import evaluate


def toy():
    return Week("TGH", "2012-01-02", ((0, "OR1"), (0, "OR2")),
                np.array([2, 3]), ("S1",), np.array([0, 0]),
                np.array([0, 0]), ((0, 1),), ((0,),),
                np.array([100., 100.]), np.array([100., 100.]), ())


def test_inverse_response_and_positive_floor():
    b = np.array([20., 200., 200.])
    d = reviewed(b, np.array([-100., -100., 100.]))
    assert np.allclose(d, [4.8, 176., 224.])
    u = display(b, d)
    assert np.all(abs(u) <= 30. + 1e-8)
    assert np.allclose(d-b, .8*u)
    with pytest.raises(ValueError):
        display(b, d+50.)


def test_scenarios_exclude_target_week_and_are_reproducible():
    c = toy()
    p = {"S1": np.array([["2012-01-02", np.log(100), np.log(10.)],
                         ["2012-01-09", np.log(100), np.log(2.)]], dtype=object)}
    a = scenarios(c, c.booked, p, 12, 10)
    assert np.allclose(a, 200.)
    assert np.array_equal(a, scenarios(c, c.booked, p, 12, 10))


def test_bank_B_rejects_an_A_optimized_proposal(monkeypatch):
    import surgery.advice_comparison as ac
    c = toy()
    # More than 110 minutes switches from packed to split room assignments.
    monkeypatch.setattr(ac, "certified",
                        lambda c, d, seed, seconds: (0, 1) if d[0] > 110 else (0, 0))
    banks = {"A": np.full((5, 2), 500.), "B": np.full((5, 2), 100.),
             "C": np.full((5, 2), 100.)}
    d, z, accepted, status, _ = prescription(c, c.booked, banks,
                                                  seconds=1., budget=20.)
    assert not accepted and status == "REJECTED_B"
    assert np.array_equal(d, c.booked)
    assert z[0] == (0, 0)


def test_features_are_fixed_from_CASE_schedule():
    c = toy()
    x = np.array([[2., 3.], [4., 5.]])
    h = case_features(c, (0, 0), c.booked, x,
                      ["hist_surgeon_mean_abs_error", "hist_procedure_mean_abs_error"])
    assert h.shape == (2, 5)
    assert np.allclose(h[:, :2], x)
    assert np.isfinite(h).all()


def test_transition_reachable():
    c = toy()
    result = transition(c, c.booked.copy(), (0, 0), (0, 1))
    assert np.isfinite(result).all()
    assert np.all(result >= 76.) and np.all(result <= 124.)
    assert evaluate(c, (0, 0), c.booked)["cost"] > 0.


def test_advice_does_not_read_evaluation_realizations(monkeypatch):
    import surgery.advice_comparison as ac
    c = toy()
    monkeypatch.setattr(ac, "certified",
                        lambda c, d, seed, seconds: (0, 1) if d[0] > 110 else (0, 0))
    data = dict(X=np.ones((2, 1)), case_w=np.zeros(1),
                booked=c.booked.copy(), actual=c.actual.copy())
    pools = {"S1": np.array([["2011-10-03", np.log(100.), 0.]],
                              dtype=object)}
    item = ("toy", c, np.array([0, 1]), data, np.zeros((2, 5)),
            pools, np.zeros(5), np.zeros(5), (0, 1, 2), 1., 1.)
    first = ac.component_job(item)[1]
    c.actual[:] = 500.
    second = ac.component_job(item)[1]
    for name in first["arms"]:
        assert first["arms"][name]["displayed"] == second["arms"][name]["displayed"]
        assert first["arms"][name]["assignments"] == second["arms"][name]["assignments"]


def test_exact_hash_planner_on_toy():
    from surgery.advice_comparison import certified
    c = toy()
    for seed in (0, 1, 2):
        z = certified(c, c.booked, seed, 30.)
        assert evaluate(c, z, c.booked)["cost"] == pytest.approx(730.)


def test_full_runner_loads_and_reports_paired_cohort(tmp_path):
    from pilot.run_advice_comparison import report
    rows = [
        dict(monday="2012-01-02", site="TGH", cases=2, arms={
            "CASE": dict(realized={"0": 100.}, changed=0),
            "SHARED": dict(realized={"0": 80.}, changed=0)}),
        dict(monday="2012-01-09", site="TWH", cases=3, arms={
            "CASE": dict(realized={"0": 100.}, changed=0),
            "SHARED": dict(realized={"0": 95.}, changed=0)}),
    ]
    outcome = report(rows, (0,), tmp_path/"summary.json")
    assert outcome["n_cases"] == 5
    assert outcome["arms"]["SHARED"]["per_seed"][0]["difference"] == -5.
    assert outcome["by_site"]["TGH"]["cases"] == 2
    assert outcome["by_size"]["2-5"]["cases"] == 5



def test_flat_margin_crosses_strict_overtime_threshold():
    from surgery.weekly_planner import loads
    c = toy()
    d = np.array([200., 250.])  # packed load: 200 + 250 + 30 = 480.
    c.booked[:] = d  # The starting point must be inside the response box.
    assert loads(c, (0, 0), d)[0] == 480.
    proposal = transition(c, d, (0, 0), (0, 1))
    assert loads(c, (0, 0), proposal)[0] > 480.
    assert np.all(proposal <= d + 24. + 1e-8)


def test_intermediate_transition_chain_can_be_non_improving(monkeypatch):
    import surgery.advice_comparison as ac
    from surgery.asdbs import saa
    c = Week("TGH", "2012-01-02", ((0, "OR1"), (0, "OR2")),
             np.arange(2, 23), ("S1",), np.zeros(21, dtype=int),
             np.zeros(21, dtype=int), ((0, 1),), ((0,),),
             np.full(21, 100.), np.full(21, 100.), ())
    packed = (0,) * 21
    split = (0,) * 10 + (1,) * 11

    def fake_certified(c, d, seed, seconds):
        return split if 102.5 < d[0] < 103.5 else packed

    def one_minute(c, d, current, target):
        proposal = d.copy()
        proposal[0] += 1.
        return proposal

    monkeypatch.setattr(ac, "certified", fake_certified)
    monkeypatch.setattr(ac, "transition", one_minute)
    monkeypatch.setattr("surgery.asdbs.saa", lambda *a, **kw: split)
    banks = {name: np.full((4, 21), 100.) for name in ("A", "B", "C")}
    d, z, accepted, status, _ = ac.prescription(
        c, c.booked.copy(), banks, seeds=(0,), seconds=1., budget=15.)
    assert accepted and status == "ACCEPTED_B"
    assert d[0] == pytest.approx(103.)
    assert z[0] == split


def test_matched_stat_control_uses_deployed_clipping(monkeypatch):
    import pilot.run_advice_comparison as runner
    from types import SimpleNamespace
    booked = np.array([200., 200.])
    raw = np.array([100., -100.])
    x = raw[:, None]
    H = np.zeros((2, 5))
    H[:, 0] = 1.
    beta = np.array([5., 0., 0., 0., 0.])
    data = dict(X=x, case_w=np.array([1.]), booked=booked,
                actual=np.array([230., 190.]))
    observed = {}

    def fake_minimize(fn, x0, **kwargs):
        observed["value"] = fn(beta)
        return SimpleNamespace(x=beta, fun=observed["value"])

    monkeypatch.setattr(runner, "minimize", fake_minimize)
    assert np.array_equal(runner.shared_control(data, H), beta)
    d0 = reviewed(booked, raw)
    deployed = reviewed(booked, d0-booked+H@beta)
    error = data["actual"]-deployed
    expected = (1.75*np.maximum(error, 0).sum()
                + np.maximum(-error, 0).sum()
                + runner.LAM/2*np.sum((deployed-d0)**2))/2
    assert observed["value"] == pytest.approx(expected)
    assert deployed[1] > d0[1]


def test_average_hash_report_and_holm(tmp_path):
    from pilot.run_advice_comparison import report
    rows = []
    for week, case, shared, perweek in [
        ("2012-01-02", 2, (90., 95., 95.), (80., 100., 90.)),
        ("2012-01-09", 3, (97., 94., 91.), (92., 92., 92.)),
    ]:
        case_cost = {str(j): 100. for j in range(3)}
        rows.append(dict(monday=week, site="TGH", cases=case, fold=0,
                         arms=dict(
                             CASE=dict(realized=case_cost, changed=0),
                             SHARED=dict(realized={str(j): v for j, v in enumerate(shared)}, changed=1),
                             PER_WEEK=dict(realized={str(j): v for j, v in enumerate(perweek)}, changed=1))))
    output = report(rows, (0, 1, 2), tmp_path/"summary.json")
    assert output["bootstrap_resamples"] == 20000
    # Two Mondays each with mean differences -6.666.. and -8.
    assert output["contrasts"]["PER_WEEK_minus_CASE"]["difference"] == pytest.approx(
        (-10. - 8.) / 5.)
    assert output["contrasts"]["SHARED_minus_CASE"]["difference"] == pytest.approx(
        (-20./3. - 6.) / 5.)
    assert output["arms"]["PER_WEEK"]["averaged_seeds"]["difference"] == pytest.approx(-3.6)
    assert output["by_fold"]["0"]["cases"] == 5
    for name in ("PER_WEEK_minus_CASE", "SHARED_minus_CASE"):
        assert output["contrasts"][name]["holm_p"] >= output["contrasts"][name]["p_two_sided"]


def test_cache_fingerprint_covers_planner_and_scenario_costs():
    from pilot.run_advice_comparison import source_hashes
    keys = source_hashes()
    assert "surgery/weekly_planner.py" in keys
    assert "surgery/asdbs.py" in keys
    assert "surgery/planner.py" in keys
    assert all(len(value) == 64 for value in keys.values())


def test_sensitivity_rescores_only_displayed_recommendations(monkeypatch, tmp_path):
    import json
    import surgery.advice_comparison as ac
    c = toy()
    root = tmp_path
    original = {"arms": {
        "CASE": {"displayed": [0., 0.], "assignments": {str(r): [0, 0] for r in range(3)}},
        "PER_WEEK": {"displayed": [30., 0.], "assignments": {str(r): [0, 1] for r in range(3)}},
    }}
    (root/"toy.json").write_text(json.dumps(original))
    monkeypatch.setattr(ac, "certified",
                        lambda c, d, seed, seconds: (0, 1) if d[0] > 115. else (0, 0))
    state = dict(components_dir=root, decision_dates={2: "2011-12-30", 3: "2012-01-02"},
                 response_sweep=((.5, 15.), (.8, 15.)), seconds=1.,
                 seeds=(0, 1, 2))
    monkeypatch.setattr(ac, "_STATE", state)
    ck, outcomes = ac.sensitivity_job(("toy", c, np.array([0, 1])))
    assert ck == "toy"
    assert outcomes["cutoff_week_start"]["eligible_at_cutoff"] == 1
    assert outcomes["cutoff_week_start"]["arms"]["PER_WEEK"]["realized"]
    assert outcomes["alpha_0.5_h_15"]["arms"]["PER_WEEK"]["realized"]



def test_exact_feature_checkpoint_resumes_after_interruption(tmp_path, monkeypatch):
    import pilot.run_advice_comparison as runner
    items = [("a", None, np.array([0])), ("b", None, np.array([1]))]
    data = dict(booked=np.array([100., 100.]))
    checkpoint = tmp_path/"features_progress.npz"
    attempts = []

    def fake_parallel(fn, pending, state, cores):
        attempts.append([x[0] for x in pending])
        for item in pending:
            if item[0] == "b" and len(attempts) == 1:
                raise RuntimeError("simulated interruption")
            yield item[0], item[2], np.full((len(item[2]), 5), 1. if item[0] == "a" else 2.)

    monkeypatch.setattr(runner, "parallel", fake_parallel)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.collect_features(items, data, [], 1., 2, checkpoint)
    assert checkpoint.exists()
    result = runner.collect_features(items, data, [], 1., 2, checkpoint)
    assert attempts == [["a", "b"], ["b"]]
    assert np.array_equal(result[:, 0], [1., 2.])
    assert not checkpoint.exists()


def test_h60_has_no_effect_on_existing_30_minute_displays():
    from surgery.planner import response
    u = np.linspace(-30., 30., 101)
    assert np.allclose(response(u, .8, 30.), response(u, .8, 60.))



def test_missing_decision_dates_are_normalized_to_none():
    import pandas as pd
    from pilot.run_advice_comparison import decision_date_lookup
    cohort = pd.DataFrame({
        "case_id": [2, 3, 4],
        "dtt": pd.to_datetime(["2011-12-30", None, "2012-01-02"]),
    })
    assert decision_date_lookup(cohort) == {
        2: "2011-12-30", 3: None, 4: "2012-01-02",
    }


@pytest.mark.parametrize("missing", [None, np.nan])
def test_cutoff_sensitivity_handles_missing_or_nan_dates(monkeypatch, tmp_path, missing):
    import json
    import surgery.advice_comparison as ac
    c = toy()
    original = {"arms": {
        "CASE": {"displayed": [0., 0.], "assignments": {str(r): [0, 0] for r in range(3)}},
        "PER_WEEK": {"displayed": [30., 0.], "assignments": {str(r): [0, 1] for r in range(3)}},
    }}
    (tmp_path/"toy.json").write_text(json.dumps(original))
    monkeypatch.setattr(ac, "certified",
                        lambda c, d, seed, seconds: (0, 1) if d[0] > 115. else (0, 0))
    monkeypatch.setattr(ac, "_STATE", dict(
        components_dir=tmp_path, decision_dates={2: "2011-12-30", 3: missing},
        response_sweep=((.8, 30.),), seconds=1., seeds=(0, 1, 2)))
    _, output = ac.sensitivity_job(("toy", c, np.array([0, 1])))
    cutoff = output["cutoff_week_start"]
    assert cutoff["eligible_at_cutoff"] == 1
    assert cutoff["unknown_decision_date"] == 1
    assert cutoff["arms"]["PER_WEEK"]["status"] == "CERTIFIED"
    assert len(cutoff["arms"]["PER_WEEK"]["realized"]) == 3
