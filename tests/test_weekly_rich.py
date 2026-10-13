"""Regression guards for the supported-week rich feature pilot."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_weekly_rich import (choose_pilot_weeks, source_fingerprint,
                             verify_bounds, main, oracle_assignment_sha256,
                             paired_week_is_certified)
from surgery.data import GROUPS
from surgery.rich_features import RichCorrectionEncoder
from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate, solve_week
from run_weekly_diagnostics import audit_component


def example():
    return Week("TGH", "2012-01-02", ((0, "OR1"), (2, "OR2")),
                np.array([2, 3, 4]), ("A", "B"), np.array([0, 0, 1]),
                np.array([0, 0, 1]), ((0, 1), (1,)), ((0,), (2,)),
                np.array([400., 200., 200.]), np.array([200., 400., 200.]),
                (("A", 2, "OR2", "prior_dates"),)).validate()


def test_diagnostics_still_covers_cost_preserving_tie():
    w = example()
    p = solve_week(w, w.booked)
    baseline = evaluate(w, p.assignment, w.actual)["cost"]
    summary, room_rows = audit_component((
        {"group": "TGH", "monday": w.monday, "component": 0},
        w, {"booked_cost": baseline}, 20., 1
    ))
    assert summary["balanced_predicted_cost"] == pytest.approx(
        summary["baseline_predicted_cost"], abs=.01)
    assert summary["balanced_max_predicted_room_load"] <= (
        summary["baseline_max_predicted_room_load"] + .01)
    assert len(room_rows) == w.k


def test_pilot_bound_chain_includes_nonnegative_regret():
    verify_bounds(0., 1., 2., 3.)
    with pytest.raises(AssertionError, match="nonnegative"):
        verify_bounds(-0.5, 1., 2., 3.)
    with pytest.raises(AssertionError, match="ordering"):
        verify_bounds(1., 0.5, 2., 3.)
    with pytest.raises(AssertionError):
        verify_bounds(1., 2., 0.5, 3.)
    with pytest.raises(AssertionError):
        verify_bounds(1., 2., 3., 2.5)


def test_default_pilot_covers_three_validation_sites():
    class Stub:
        def __init__(self, group, monday):
            self.group, self.monday = group, monday
    pairs = [
        (Stub(g, f"2012-01-{n:02d}"), part)
        for part in ("train", "validation")
        for g in GROUPS for n in (2, 9, 16)
    ]
    first = choose_pilot_weeks(pairs, 3)
    assert len(first) == len(GROUPS)
    assert {w.group for w, _ in first} == set(GROUPS)
    assert all(part == "validation" for _, part in first)
    assert len(choose_pilot_weeks(pairs, 0)) == len(pairs)


def test_policy_overwrite_guard_uses_actual_filename(tmp_path, monkeypatch):
    (tmp_path / "lasso_policy.json").write_text('{"sentinel":1}')
    monkeypatch.setattr(sys, "argv", ["run_weekly_rich.py", "not_used.xlsx",
                                    "--stage", "fit", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert (tmp_path / "lasso_policy.json").read_text() == '{"sentinel":1}'


def test_feature_code_includes_source_digest_and_no_calendar():
    assert len(source_fingerprint()) == 64
    # Quick synthetic sample: rich additional features are content-dependent,
    # not unrestricted procedure/surgeon dummy variables.
    n = 40
    f = pd.DataFrame({
        "case_id": np.arange(n) + 2,
        "date": np.repeat(pd.bdate_range("2012-01-02", periods=5), 8),
        "group": np.tile(["TGH", "TWH-main"], n // 2),
        "service": np.tile(["A", "B"], n // 2),
        "procedure": np.tile(["1", "2", "3", "4"], n // 4),
        "surgeon": np.tile(["S1", "S2", "S3", "S4"], n // 4),
        "booked": np.tile([60., 120., 180., 240.], n // 4),
        "actual": np.tile([75., 140., 195., 210.], n // 4),
        "patient_type": ["DAY"] * n
    })
    enc = RichCorrectionEncoder()
    X = enc.fit_transform(f)
    assert X.shape[0] == n
    assert "booked" not in enc.names
    assert not any(name.startswith("calendar_") for name in enc.names)
    assert all("proportional_booked_x_" + level in enc.names for level in
               ("service", "procedure", "surgeon", "pair"))
    assert np.isfinite(X).all()
    test = f.iloc[:3].copy()
    test["actual"] += 10000.
    np.testing.assert_allclose(enc.transform(test), enc.transform(f.iloc[:3]))


def test_resumed_week_requires_one_shared_hindsight_assignment():
    """BOOKED/LASSO cannot be mixed across oracle ties or partial retries."""
    week = example()
    # Both schedules are feasible but differ in the physical room-day of case 2.
    first = SimpleNamespace(complete=True, assignment=[0, 0, 1])
    alternate = SimpleNamespace(complete=True, assignment=[1, 0, 1])
    digest = oracle_assignment_sha256([week], [first])
    other = oracle_assignment_sha256([week], [alternate])
    assert digest != other and len(digest) == 64
    assert oracle_assignment_sha256([week], [first]) == digest

    key = ("validation", week.group, week.monday)
    bkey, lkey = (*key, "BOOKED"), (*key, "LASSO")
    rows = {
        bkey: {"complete": True, "oracle_assignment_sha256": digest},
        lkey: {"complete": True, "oracle_assignment_sha256": digest},
    }
    assert paired_week_is_certified(rows, *key)

    rows[lkey]["oracle_assignment_sha256"] = other
    assert not paired_week_is_certified(rows, *key)
    rows[lkey]["oracle_assignment_sha256"] = digest
    rows[lkey]["complete"] = False
    assert not paired_week_is_certified(rows, *key)
    rows[lkey]["complete"] = True
    del rows[lkey]["oracle_assignment_sha256"]
    assert not paired_week_is_certified(rows, *key)
    del rows[lkey]
    assert not paired_week_is_certified(rows, *key)
    with pytest.raises(ValueError, match="Only certified"):
        oracle_assignment_sha256([week], [SimpleNamespace(
            complete=False, assignment=[0, 0, 1])])
