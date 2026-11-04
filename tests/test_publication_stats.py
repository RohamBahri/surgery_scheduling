"""Publication summaries: pooled descriptive data and all-method intervals."""
import numpy as np
import pandas as pd
import pytest

from analysis.report import descriptive_statistics
from pilot.experiment import METHODS, scenario_id, summarize


def test_descriptives_pool_training_and_evaluation(tmp_path):
    cases = pd.DataFrame({
        "case_id": [1, 2, 3, 4],
        "group": ["TGH", "TGH", "TWH-main", "TWH-main"],
        "split": ["train", "train", "test", "test"],
        "booked": [60., 90., 120., 55.],
        "actual": [50., 95., 100., 60.],
        "surgeon": ["S1", "S1", "S2", "S2"],
        "procedure": ["P1", "P1", "P2", "P2"],
    })
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir()
    result = descriptive_statistics(cases, tables, figures)
    assert result["cases"] == 4
    assert result["rounded_30_minute_bookings_share"] == pytest.approx(.75)
    assert result["overbooking_share"] == pytest.approx(.5)
    assert result["conditional_mean_overbooking_minutes"] == pytest.approx(15.)
    assert result["conditional_mean_underbooking_minutes"] == pytest.approx(5.)
    summary = pd.read_csv(tables / "cohort_by_site.csv")
    assert summary.cases.sum() == 4
    assert "split" not in summary.columns
    assert (tables / "cohort_by_site_and_period.csv").exists()


def test_all_method_intervals_and_gap_closed():
    scenario = ("saturating", .5, 30.)
    sid = scenario_id(*scenario)
    costs = dict(zip(METHODS, (10., 9., 8., 7., 6., 5.)))
    rows = []
    for week in ("2013-01-07", "2013-01-14", "2013-01-21"):
        for method in METHODS:
            rows.append(dict(scenario=sid, component=week, method=method,
                             complete=True, cases=2, week=week,
                             cost=costs[method], planned_cost=costs[method],
                             asymmetric_error=0., loss_bound=costs[method],
                             idle=0., overtime=0., mae=0., displayed_mae=0.))
    results, intervals, weekly, weeks = summarize(
        rows, [scenario], {sid: dict(
            fraction_booking_errors_unreachable=.2,
            perfect_advice_minimum_mae_per_case=2.)})
    assert len(weeks) == 3
    assert len(intervals[sid]) == 9  # five vs Booked, four extra Decision-aware contrasts
    assert results[0]["PERFECT_ADVICE_gap_closed_fraction"] == pytest.approx(.8)
    assert results[0]["HINDSIGHT_gap_closed_fraction"] == pytest.approx(1.)
    interval = intervals[sid]["ACCURACY_TRAINED_minus_BOOKED"]
    assert interval["mean_weekly_difference"] == pytest.approx(-1.)
    assert interval["confidence_95"] == pytest.approx([-1., -1.])
