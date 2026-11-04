"""Focused regression tests for the new research workflow."""
import numpy as np
import pandas as pd
import pytest

from pilot import experiment, policy_runner, policy_training
from pilot.prepare_data import assemble_components
from surgery.learning import DCObjective, ResponseDC


def test_full_training_flags_remain_aligned(tmp_path):
    context = tmp_path / "context.pkl"
    context.write_bytes(b"x")
    runner = policy_runner.parse_args(["--context", str(context),
                                       "--features", "structural"])
    assert runner.features == "structural"
    assert runner.initial_weights is None


def test_fixed_display_is_not_recomputed_under_new_behavior():
    booked = np.array([100., 100.])
    advice = np.array([60., -60.])
    actual = ResponseDC(booked, .8, 30., 180., 1., link="nonmonotone")
    observed = actual.reviewed_from_display(advice)
    np.testing.assert_allclose(observed, [118., 82.])
    np.testing.assert_allclose(actual.display(advice), [22.5, -22.5])
    assert not np.allclose(actual.display(advice), advice)
    with pytest.raises(ValueError):
        actual.reviewed_from_display(np.array([1.]))


def test_perfect_advice_is_best_attainable_casewise_accuracy():
    booked = np.array([100., 100., 100.])
    realized = np.array([160., 90., 112.])
    alpha, threshold = .5, 30.
    shown = np.clip((realized - booked) / alpha, -threshold, threshold)
    response = ResponseDC(booked, alpha, threshold, 180., 1., link="saturating")
    implemented = response.reviewed_from_display(shown)
    np.testing.assert_allclose(implemented, [115., 90., 112.])


def test_response_aware_objective_includes_display_accuracy_penalty():
    booked = np.array([100., 160.])
    realized = np.array([122., 145.])
    features = np.array([[1., 1.], [1., -1.]])
    response = ResponseDC(booked, .8, 30., link="saturating")
    common = dict(booked=booked, X=features, base=np.zeros(2),
                  response_map=response, mode="CASE", actual=realized,
                  components=1, lam=.1)
    standard = DCObjective(**common)
    with_accuracy = DCObjective(**common, accuracy_mu=2.)
    coefficients = np.array([3., 2.])
    displayed_duration = booked + response.display(features @ coefficients)
    expected = 2. * np.abs(realized - displayed_duration).sum()
    assert with_accuracy.value(coefficients) - standard.value(coefficients) == pytest.approx(
        expected, abs=1e-7)


def test_assemble_components_partitions_single_dataset():
    cases = pd.DataFrame(dict(
        case_id=[1, 2], group=["TGH", "TGH"],
        date=pd.to_datetime(["2012-01-02", "2012-01-02"]),
        room=["OR1", "OR1"], surgeon=["S1", "S1"],
        booked=[100., 100.], actual=[110., 90.],
        split=["train", "train"]))
    events = pd.DataFrame(columns=[
        "case_id", "group", "surgeon", "monday", "weekday", "room", "date"])
    selected, components = assemble_components(cases, events, "train")
    assert len(selected) == 2
    assert sum(len(positions) for _, _, positions in components) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([positions for _, _, positions in components])),
        np.arange(2))
