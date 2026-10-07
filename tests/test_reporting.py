import numpy as np
import pandas as pd

from surgery.data import GROUPS, SCENARIOS
from surgery.reporting import METHODS, opportunity_interval, summarize
from test_optimization import make_day


def test_ratio_bounds_and_zero_denominator():
    assert opportunity_interval(100, 70, 40, 60)[:2] == (.5, .75)
    assert opportunity_interval(100, 70, 40, 100)[2] == 'denominator_may_be_zero'
    assert opportunity_interval(100, 110, 40, 60)[:2] == (-.25, -1 / 6)
    assert opportunity_interval(100, 100, 100, 100)[0] is None


def fixture_rows():
    days, daily, oracles = [], [], []
    for group in GROUPS:
        for date in ('2013-01-02', '2013-01-09'):
            day = make_day([60]); day.group = group; day.date = date; day.key = f'{group}_{date}'
            days.append(day)
            for alpha, h in SCENARIOS:
                for method in METHODS:
                    daily.append({'day': day.key, 'group': group, 'date': date, 'alpha': alpha, 'h': h,
                                  'method': method, 'complete': True, 'cost': 80 if method == 'VF' else 100,
                                  'overtime': 10, 'idle': 20, 'rooms': 1, 'cases': 1, 'absolute_error': 5})
                oracles.append({'day': day.key, 'group': group, 'date': date, 'alpha': alpha, 'h': h,
                                'lower': 40, 'upper': 60, 'complete': False, 'actual_oracle': 30})
    return days, daily, oracles


def test_shared_week_resamples_comparator_percentages_and_incomplete_guard(tmp_path):
    days, daily, oracles = fixture_rows()
    result = summarize(tmp_path, daily, oracles, days)
    assert result['ready']
    comparison = pd.read_csv(tmp_path / 'comparisons.csv')
    assert np.allclose(comparison.difference_pct_of_comparator, -20)
    assert comparison.weeks_favoring_vf.eq(2).all()
    assert comparison.weeks.eq(26).all()
    assert np.load(tmp_path / 'bootstrap_week_indices.npy').shape == (10000, 26)
    rows = pd.read_csv(tmp_path / 'methods.csv')
    assert not rows.response_oracle_exact.any()
    daily[0]['complete'] = False
    assert not summarize(tmp_path / 'pending', daily, oracles, days)['ready']
    assert not (tmp_path / 'pending' / 'methods.csv').exists()


def test_methods_share_the_best_daily_oracle_upper_bound(tmp_path):
    days, daily, oracles = fixture_rows()
    for row in oracles:
        row['upper'] = 100.
    assert summarize(tmp_path, daily, oracles, days)['ready']
    bounds = pd.read_csv(tmp_path / 'daily_oracles.csv')
    assert bounds.upper.eq(80).all() and bounds.solver_upper.eq(100).all()
    methods = pd.read_csv(tmp_path / 'methods.csv')
    assert methods.groupby(['group', 'alpha', 'h']).response_oracle_upper.nunique().eq(1).all()
    assert methods[methods.method.eq('VF')].opportunity_captured_upper.eq(1).all()
