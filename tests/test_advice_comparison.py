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
            "CASE": dict(realized={"0": 100.}),
            "SHARED": dict(realized={"0": 80.})}),
        dict(monday="2012-01-09", site="TWH", cases=3, arms={
            "CASE": dict(realized={"0": 100.}),
            "SHARED": dict(realized={"0": 95.})}),
    ]
    outcome = report(rows, (0,), tmp_path/"summary.json")
    assert outcome["n_cases"] == 5
    assert outcome["arms"]["SHARED"]["per_seed"][0]["difference"] == -5.
    assert outcome["by_site"]["TGH"]["cases"] == 2
    assert outcome["by_size"]["2-5"]["cases"] == 5
