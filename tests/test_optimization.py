from itertools import product

import numpy as np
import pytest

from surgery.data import Day
from surgery.learning import dc_parts, fit_policy, library_bound, theta, train_vf
from surgery import planner
from surgery.planner import display_and_duration, metrics, reachable, response_oracle, solve_day
from surgery.storage import Store


def make_day(booked, actual=None, surgeon=None, rooms=2, eligible=None):
    booked = np.asarray(booked, float)
    surgeon = np.arange(len(booked)) if surgeon is None else np.asarray(surgeon)
    ns = max(surgeon) + 1
    return Day('TGH_2011-07-04', 'TGH', '2011-07-04', tuple(f'R{i}' for i in range(rooms)),
               np.arange(len(booked)), tuple(str(i) for i in range(ns)), surgeon,
               tuple(tuple(range(rooms)) for _ in range(ns)) if eligible is None else eligible,
               booked, booked.copy() if actual is None else np.asarray(actual, float),
               np.arange(len(booked)) % rooms, np.arange(len(booked)))


def brute(day, tie='squares'):
    def key(a):
        m = metrics(day, a, day.booked)
        balance = (m['squared_load'],) if tie == 'squares' else tuple(sorted(
            m['loads'] + [0.] * (day.n_rooms - m['rooms']), reverse=True))
        return m['cost'], balance, a
    return min(product(*day.eligible), key=key)


@pytest.mark.parametrize('tie', ['squares', 'lexloads'])
@pytest.mark.parametrize('booked,surgeon,eligible', [
    ([400, 400, 400], None, None), ([200, 200, 300, 100], [0, 0, 1, 2], None),
    ([200, 300, 350], None, ((0,), (0, 1), (1,))), ([300, 300], None, None)])
def test_all_planner_tie_stages_against_exhaustive_assignments(tie, booked, surgeon, eligible):
    day = make_day(booked, surgeon=surgeon, eligible=eligible)
    result = solve_day(day, day.booked, seconds=10, tie=tie)
    assert result['complete'], result
    assert tuple(result['assignment']) == brute(day, tie)
    m = result['planned']
    assert np.isclose(m['cost'], 510 * m['rooms'] + 2.75 * m['overtime'] - sum(booked) - 30 * len(booked))


def test_policy_ties_do_not_consult_realized_outcomes(tmp_path):
    day = make_day([400, 400, 400], [500, 200, 100])
    store = Store(tmp_path)
    first = solve_day(day, day.booked, store)
    day.actual = np.array([100, 200, 500])
    second = solve_day(day, day.booked, store)
    assert first['assignment'] == second['assignment']
    assert len(store.solve_rows()) == len(first['stages']) + 1
    store.close()


@pytest.mark.parametrize('booked,expected', [([260, 260], 437.5), ([300, 300], 315.)])
def test_response_oracle_prices_competitors_exactly(booked, expected):
    day = make_day(booked, [500, 200])
    b, a = solve_day(day, day.booked), solve_day(day, day.actual)
    result = response_oracle(day, .5, 60, b, a, seconds=10)
    assert result['complete'], result
    assert np.isclose(result['lower'], expected)
    assert np.isclose(result['upper'], expected)
    lo, hi = reachable(day.booked, .5, 60)
    assert np.all(np.asarray(result['durations']) >= lo - 1e-6)
    assert np.all(np.asarray(result['durations']) <= hi + 1e-6)


def test_master_timeout_never_becomes_exact(monkeypatch):
    day = make_day([300, 300], [500, 200])
    b, a = solve_day(day, day.booked), solve_day(day, day.actual)
    optimize = planner.optimize
    def unfinished(m, store, label, **metadata):
        result = optimize(m, store, label, **metadata)
        if label == 'response_master':
            result['optimal'] = False
        return result
    monkeypatch.setattr(planner, 'optimize', unfinished)
    result = response_oracle(day, .5, 60, b, a, seconds=10)
    assert not result['complete']
    assert result['lower'] <= 315 + 1e-6 <= result['upper'] + 1e-6


def test_incomplete_plan_is_retried_and_eligibility_changes_cache(tmp_path, monkeypatch):
    day = make_day([300, 300], [500, 200])
    store = Store(tmp_path)
    optimize = planner.optimize
    def unfinished(m, store, label, **metadata):
        result = optimize(m, store, label, **metadata)
        result['optimal'] = False
        return result
    monkeypatch.setattr(planner, 'optimize', unfinished)
    assert not solve_day(day, day.booked, store)['complete']
    monkeypatch.setattr(planner, 'optimize', optimize)
    assert solve_day(day, day.booked, store)['complete']
    day.eligible = ((0,), (1,))
    assert solve_day(day, day.booked, store)['assignment'] == [0, 1]
    store.close()


def test_floor_precedes_response_and_dc_matches_response():
    booked = np.array([5., 300.])
    display, duration = display_and_duration(np.array([-60., -60.]), booked, .5, 60)
    np.testing.assert_allclose(display, [-4, -60])
    np.testing.assert_allclose(duration, [3, 270])
    raw = np.linspace(-180, 180, 101)
    for alpha, h in [(.5, 30), (.5, 60), (.8, 30), (.8, 60)]:
        P, N, _, _ = dc_parts(raw, alpha, h)
        np.testing.assert_allclose(P - N, planner.response(raw, alpha, h), atol=1e-12)


def test_convex_updates_and_library_bounds(tmp_path):
    day = make_day([100, 150, 200, 250], [110, 160, 210, 260], surgeon=[0, 0, 1, 1])
    X = np.column_stack([np.ones(4), [-1.5, -.5, .5, 1.5]])
    store = Store(tmp_path)
    b, a = solve_day(day, day.booked, store), solve_day(day, day.actual, store)
    direct = fit_policy(X, day.booked, day.actual, 1, 1, None, .1, np.zeros(2), store, 'direct')
    assert direct['complete']
    assert direct['objective'] < 1e-5
    case = fit_policy(X, day.booked, day.actual, 1, .5, 30, .05, np.zeros(2), store, 'case')
    assert case['complete']
    assert case['objective'] <= theta(day.actual - day.booked).sum()
    vf = train_vf(X, day.booked, day.actual, [day], .5, 30, .05, case['w'],
                  {day.key: {'booked': b, 'actual': a}}, store, 'vf', max_outer=3)
    assert vf['complete']
    assert vf['final_bound']['regularized_bound'] <= vf['trajectory'][0]['before']['regularized_bound'] + 1e-6
    _, duration = display_and_duration(X @ np.asarray(vf['w']), day.booked, .5, 30)
    plan = solve_day(day, duration)
    regret = metrics(day, plan['assignment'], day.actual)['cost'] - a['primary_cost']
    assert regret <= vf['final_bound']['bound_sum'] + 1e-6
    zero, _ = library_bound(np.zeros(2), X, day.booked, day.actual, [day], vf['library'],
                            {day.key: a['bound']}, .5, 30, .05)
    assert zero == vf['zero_bound']
    store.close()
