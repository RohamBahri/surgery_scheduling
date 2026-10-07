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


def brute(day):
    def key(a):
        m = metrics(day, a, day.booked)
        balance = m['max_load']
        return m['cost'], balance, a
    return min(product(*day.eligible), key=key)


@pytest.mark.parametrize('backend', ['compact', 'patterns'])
@pytest.mark.parametrize('booked,surgeon,eligible', [
    ([400, 400, 400], None, None), ([200, 200, 300, 100], [0, 0, 1, 2], None),
    ([200, 300, 350], None, ((0,), (0, 1), (1,))), ([300, 300], None, None)])
def test_all_planner_tie_stages_against_exhaustive_assignments(backend, booked, surgeon, eligible):
    day = make_day(booked, surgeon=surgeon, eligible=eligible)
    result = solve_day(day, day.booked, seconds=10, backend=backend)
    assert result['complete'], result
    assert tuple(result['assignment']) == brute(day)
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
    assert not solve_day(day, day.booked, store, backend='compact')['complete']
    monkeypatch.setattr(planner, 'optimize', optimize)
    assert solve_day(day, day.booked, store, backend='compact')['complete']
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


def test_pattern_enumeration_has_no_surgeon_count_cap():
    day = make_day([50.] * 6, rooms=3)
    plan = solve_day(day, day.booked, backend='patterns')
    assert plan['complete'] and plan['assignment'] == [0] * 6
    assert plan['primary_cost'] == 30.
    interrupted = solve_day(day, day.booked, backend='patterns', seconds=1e-9)
    assert not interrupted['complete']
    assert 'enumeration' in interrupted['reason']


def test_primary_only_omits_ties_and_fallback_matches_full_plan(monkeypatch):
    day = make_day([190., 170., 310., 20.], rooms=3)
    direct = solve_day(day, day.booked, backend='patterns')
    primary = solve_day(day, day.booked, primary_only=True)
    assert primary['complete'] and primary['stages'] == []
    compact = planner.compact_plan
    def pending(*args, **kwargs):
        result = compact(*args, **kwargs)
        result['complete'] = False
        return result
    monkeypatch.setattr(planner, 'compact_plan', pending)
    fallback = solve_day(day, day.booked)
    assert fallback['complete'] and fallback['backend'] == 'patterns'
    assert fallback['assignment'] == direct['assignment']


def test_case_error_cap_is_visible_and_resumes(tmp_path):
    day = make_day([100., 150., 200., 250.], [110., 160., 210., 260.], surgeon=[0, 0, 1, 1])
    X = np.column_stack([np.ones(4), [-1.5, -.5, .5, 1.5]])
    store = Store(tmp_path)
    args = (X, day.booked, day.actual, 1, .5, 30, .05, np.zeros(2), store, 'capped')
    first = fit_policy(*args, iterations=1)
    assert first['complete'] and first['hit_inner_cap'] and not first['converged']
    second = fit_policy(*args, iterations=30)
    assert second['converged'] and not second['hit_inner_cap']
    assert second['ever_hit_inner_cap'] and second['inner_iterations'] > first['inner_iterations']
    assert second['objective'] <= first['objective']
    store.close()


def test_shift_floor_and_exact_duration_cache_across_scenarios(monkeypatch):
    from surgery import learning
    day = make_day([5., 20., 600.])
    captured = []
    def record(day, duration, *args, **kwargs):
        captured.append(duration)
        return {'complete': True, 'assignment': [0, 0, 1]}
    monkeypatch.setattr(learning, 'solve_day', record)
    for alpha, h in [(0.5, 30), (0.5, 60), (0.8, 30), (0.8, 60)]:
        captured.clear()
        learning.shift_day(day, alpha, h, {'tie': 'maxload'})
        for shift, duration in zip(range(-round(alpha*h), round(alpha*h)+1), captured):
            _, expected = display_and_duration(np.full(3, shift/alpha), day.booked, alpha, h)
            np.testing.assert_allclose(duration, expected, atol=1e-12)


def test_pattern_screening_and_room_types_match_exhaustive_plans():
    rng = np.random.default_rng(2701)
    for _ in range(12):
        booked = rng.integers(40, 550, 5).astype(float)
        eligibility = tuple(tuple(sorted(rng.choice(3, size=rng.integers(1, 4), replace=False))) for _ in booked)
        day = make_day(booked, rooms=3, eligible=eligibility)
        plan = solve_day(day, booked, backend='patterns')
        assert plan['complete'], plan
        assert tuple(plan['assignment']) == brute(day)
        assert plan['patterns_after_dual_screen'] <= plan['patterns']


def test_invalid_solver_incumbent_never_becomes_an_exact_plan(monkeypatch):
    from scipy.optimize import OptimizeResult
    def invalid(c, **kwargs):
        return OptimizeResult(status=0, fun=0., x=np.zeros(len(c)), mip_dual_bound=0., message='injected')
    monkeypatch.setattr(planner, 'milp', invalid)
    day = make_day([300., 300.])
    result = solve_day(day, day.booked, backend='patterns')
    assert not result['complete'] and not result['primary']['feasible']
    assert result['primary']['coverage_residual'] == 1.
