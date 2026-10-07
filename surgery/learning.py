"""Noise calibration, case-error fits, and schedule-library majorization."""

import gurobipy as gp
import numpy as np
from scipy import sparse

from .planner import display_and_duration, metrics, model, optimize, solve_day
from .storage import digest

BOUND = 100.


def theta(error):
    return 1.75 * np.maximum(error, 0) + np.maximum(-error, 0)


def penalty(X, days, seed=20261007):
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(10):
        noise = 7 / 11 - (rng.random((100, len(X))) <= 7 / 11)
        maxima.extend(np.abs(noise @ X[:, 1:]).max(axis=1))
    q90 = float(np.quantile(maxima, .9, method='linear'))
    return {'lambda1': 2.75 * 1.1 * q90 / days, 'q90': q90,
            'days': days, 'draws': 1000, 'seed': seed, 'tau': '7/11'}


def project(w, X, booked):
    w = np.clip(np.asarray(w, float), -BOUND, BOUND)
    u = X @ w
    lower = np.maximum(-180, 1 - booked)
    ratios = [1.]
    if np.any(u > 180):
        ratios.append(float(np.min(180 / u[u > 180])))
    if np.any(u < lower):
        ratios.append(float(np.min(lower[u < lower] / u[u < lower])))
    return w * min(ratios)


def dc_parts(u, alpha, h):
    if h is None:
        return u, np.zeros_like(u), np.ones_like(u), np.zeros_like(u)
    q, edge = 1 - alpha, h / (1 - alpha)
    hinge = lambda z: np.maximum(z, 0)
    slope = lambda z: np.where(z > 1e-10, 1., np.where(z < -1e-10, 0., .5))
    return (hinge(u + h) + q * hinge(u - edge), q * hinge(u + edge) + hinge(u - h),
            slope(u + h) + q * slope(u - edge), q * slope(u + edge) + slope(u - h))


def room_matrix(days, assignments, n):
    rr, cc = [], []
    row = 0
    for day in days:
        assignment = np.asarray(assignments[day.key])[day.case_surgeon]
        for room in sorted(set(assignment)):
            ids = day.rows[assignment == room]
            rr.extend([row] * len(ids)); cc.extend(ids)
            row += 1
    return sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(row, n))


def fixed_value(w, X, booked, actual, days, alpha, h, lam, S=None):
    _, duration = display_and_duration(X @ w, booked, alpha, h)
    value = theta(actual - duration).sum()
    if S is not None:
        value += theta(S @ (duration + 30) - 510).sum()
    return float(value / days + lam * np.abs(w[1:]).sum())


def fit_policy(X, booked, actual, days, alpha, h, lam, start, store, tag, *,
               assignments=None, bundles=None, seconds=300, threads=1, iterations=30):
    S = None if assignments is None else room_matrix(bundles, assignments, len(booked))
    key = 'fit:' + digest([tag, X, booked, actual, days, alpha, h, lam, start, assignments, iterations])
    saved = store.get(key) if store else None
    if saved and saved['complete']:
        return saved
    current = project(saved['w'] if saved else start, X, booked)
    history = saved['history'] if saved else []
    value = fixed_value(current, X, booked, actual, days, alpha, h, lam, S)
    result = {'complete': False, 'w': current.tolist(), 'history': history, 'objective': value,
              'alpha': alpha, 'h': h, 'lambda': lam, 'training_days': days,
              'training_cases': len(booked), 'loss_normalization': 'training_days'}
    n, p = X.shape
    with model(tag, seconds, threads) as m:
        active = np.any(np.abs(X) > 1e-12, axis=0).astype(float)
        w = m.addMVar(p, lb=-BOUND * active, ub=BOUND * active, name='coefficient')
        u = sparse.csr_matrix(X) @ w
        m.addConstr(u >= np.maximum(-180, 1 - booked))
        m.addConstr(u <= 180)
        if h is None:
            P, N = u, np.zeros(n)
        else:
            q, edge = 1 - alpha, h / (1 - alpha)
            ph, pe, ne, nh = [m.addMVar(n, lb=0) for _ in range(4)]
            m.addConstr(ph >= u + h); m.addConstr(pe >= u - edge)
            m.addConstr(ne >= u + edge); m.addConstr(nh >= u - h)
            P, N = ph + q * pe, q * ne + nh
        A = m.addMVar(n, lb=-gp.GRB.INFINITY)
        m.addConstr(A >= actual - booked + N); m.addConstr(A >= P)
        G = 2.75 * A.sum() / days
        if S is not None:
            kappa = S @ (booked + 30) - 510
            O, I = [m.addMVar(S.shape[0], lb=-gp.GRB.INFINITY) for _ in range(2)]
            m.addConstr(O >= kappa + S @ P); m.addConstr(O >= S @ N)
            m.addConstr(I >= S @ N - kappa); m.addConstr(I >= S @ P)
            G += (1.75 * O.sum() + I.sum()) / days
        absw = m.addMVar(p - 1, lb=0)
        m.addConstr(absw >= w[1:]); m.addConstr(absw >= -w[1:])
        G += lam * absw.sum()
        gamma = 0. if h is None else 1e-4
        for iteration in range(sum(row['solver_optimal'] for row in history), iterations):
            _, _, Pp, Np = dc_parts(X @ current, alpha, h)
            slope = (1.75 * Pp + Np) if S is None else 2.75 * (Pp + Np)
            gradient = X.T @ slope / days
            m.setObjective(G - (gradient + gamma * current) @ w + .5 * gamma * (w @ w))
            row = optimize(m, store, 'policy_convex', fit=tag, fit_key=key, iteration=iteration + 1)
            if not row['solutions']:
                result['reason'] = row.get('error', f"solver status {row['status']}")
                break
            candidate = project(w.X, X, booked)
            proposed = fixed_value(candidate, X, booked, actual, days, alpha, h, lam, S)
            accepted = proposed <= value
            improvement = (value - proposed) / max(1., abs(value)) if accepted else 0.
            step = float(np.max(np.abs(candidate - current)))
            history.append({'iteration': iteration + 1, 'before': value, 'proposed': proposed,
                            'accepted': accepted, 'relative_improvement': improvement,
                            'step': step, 'solver_optimal': row['optimal']})
            if accepted:
                current, value = candidate, proposed
            result.update(w=current.tolist(), objective=value, history=history)
            if not row['optimal']:
                result['reason'] = 'convex solve unfinished; accepted progress saved'
                if store:
                    store.put(key, result)
                break
            converged = h is None or not accepted or (improvement < 1e-7 and step < 1e-3)
            result.update(complete=converged or iteration + 1 == iterations,
                          stopping='converged' if converged else 'inner_cap')
            if store:
                store.put(key, result)
            if result['complete']:
                break
    result['coefficient_bound_binds'] = bool(np.max(np.abs(current)) >= BOUND - 1e-5)
    result['max_abs_coefficient'] = float(np.max(np.abs(current)))
    if store:
        store.put(key, result)
    return result


def library_bound(w, X, booked, actual, days, library, oracle_bounds, alpha, h, lam):
    _, duration = display_and_duration(X @ w, booked, alpha, h)
    case = float(theta(actual - duration).sum())
    planned, selected = 0., {}
    for day in days:
        choices = [(metrics(day, a, duration[day.rows])['cost'], tuple(a)) for a in library[day.key]]
        cost, assignment = min(choices)
        planned += cost
        selected[day.key] = list(assignment)
    oracle = sum(oracle_bounds.values())
    unregularized = (case + planned - oracle) / len(days)
    regularizer = lam * float(np.abs(w[1:]).sum())
    return {'bound_mean': unregularized, 'bound_sum': case + planned - oracle,
            'regularized_bound': unregularized + regularizer, 'penalty': regularizer,
            'case_error_mean': case / len(days), 'planned_cost_sum': planned,
            'oracle_lower_sum': oracle}, selected


def train_vf(X, booked, actual, days, alpha, h, lam, start, seeds, store, tag, *,
             seconds=300, threads=1, tie='squares', max_outer=15, inner=10, progress=None):
    key = 'vf:' + digest([tag, X, booked, actual, alpha, h, lam, start, tie, max_outer, inner])
    state = store.get(key) if store else None
    if state and state['complete']:
        return state
    if state is None:
        state = {'complete': False, 'w': list(start), 'trajectory': [], 'stagnant': 0,
                 'library': {d.key: [seeds[d.key]['booked']['assignment'],
                                     seeds[d.key]['actual']['assignment']] for d in days}}
    library = state['library']
    oracle_bounds = {d.key: seeds[d.key]['actual']['bound'] for d in days}

    def enrich(weights, phase):
        _, duration = display_and_duration(X @ weights, booked, alpha, h)
        pending = []
        for i, day in enumerate(days):
            warm = min(library[day.key], key=lambda a: metrics(day, a, duration[day.rows])['cost'])
            result = solve_day(day, duration[day.rows], store, seconds=seconds, threads=threads,
                               tie=tie, warm_start=warm)
            if result['complete']:
                a = result['assignment']
                if a not in library[day.key]:
                    library[day.key].append(a)
            else:
                pending.append(day.key)
            if progress and (i % 25 == 0 or i + 1 == len(days)):
                progress(f'{tag} {phase}: {i + 1}/{len(days)} days; {len(pending)} pending')
        return pending

    if not state.get('anchor_ready'):
        pending = enrich(np.asarray(state['w']), 'initial plans')
        state.update(anchor_ready=not pending, pending_days=pending)
        if store:
            store.put(key, state)
        if pending:
            return state
    for outer in range(len(state['trajectory']) + 1, max_outer + 1):
        current = np.asarray(state['w'])
        anchor, fixed = library_bound(current, X, booked, actual, days, library, oracle_bounds, alpha, h, lam)
        fit = state.get('candidate_fit')
        if fit is None:
            fit = fit_policy(X, booked, actual, len(days), alpha, h, lam, current, store,
                             f'{tag}_outer_{outer}', assignments=fixed, bundles=days,
                             seconds=seconds, threads=threads, iterations=inner)
            if not fit['complete']:
                state['reason'] = 'policy update pending'
                if store:
                    store.put(key, state)
                return state
            state['candidate_fit'] = fit
            state['anchor_bound'] = anchor
        anchor = state['anchor_bound']
        candidate = np.asarray(fit['w'])
        proposed, _ = library_bound(candidate, X, booked, actual, days, library, oracle_bounds, alpha, h, lam)
        accepted = proposed['regularized_bound'] <= anchor['regularized_bound']
        if not accepted:
            candidate = current
        pending = enrich(candidate, f'outer {outer}')
        state['pending_days'] = pending
        if store:
            store.put(key, state)
        if pending:
            return state
        final, _ = library_bound(candidate, X, booked, actual, days, library, oracle_bounds, alpha, h, lam)
        improvement = ((anchor['regularized_bound'] - final['regularized_bound'])
                       / max(1., abs(anchor['regularized_bound'])))
        if improvement < -1e-8:
            state['reason'] = 'numerical increase in training bound; candidate retained for inspection'
            if store:
                store.put(key, state)
            return state
        state['trajectory'].append({'outer': outer, 'before': anchor, 'after': final,
                                    'relative_improvement': improvement, 'accepted': accepted,
                                    'library_size': sum(len(v) for v in library.values()),
                                    'inner_stopping': fit['stopping']})
        state['w'] = candidate.tolist()
        state['stagnant'] = state['stagnant'] + 1 if improvement < .001 else 0
        state.pop('candidate_fit', None); state.pop('anchor_bound', None)
        state.update(complete=state['stagnant'] >= 2 or outer == max_outer,
                     hit_outer_cap=outer == max_outer)
        state['final_bound'] = final
        state['zero_bound'], _ = library_bound(np.zeros(X.shape[1]), X, booked, actual,
                                               days, library, oracle_bounds, alpha, h, lam)
        state['max_abs_coefficient'] = float(np.max(np.abs(candidate)))
        state['coefficient_bound_binds'] = bool(state['max_abs_coefficient'] >= BOUND - 1e-5)
        if progress:
            progress(f'{tag}: outer {outer}, bound {final["regularized_bound"]:.3f}, improvement {improvement:.3%}')
        if store:
            store.put(key, state)
        if state['complete']:
            state.pop('reason', None)
            if store:
                store.put(key, state)
            break
    return state


def train_shift(days, alpha, h, store, *, seconds=300, threads=1, tie='squares', progress=None):
    trials, pending, warm = [], [], {}
    scope = digest([(d.signature(), d.booked, d.actual) for d in days])
    for shift in range(-int(round(alpha * h)), int(round(alpha * h)) + 1):
        key = 'shift:' + digest([scope, alpha, shift, tie])
        saved = store.get(key) if store else None
        if saved and saved['complete']:
            trials.append(saved)
            continue
        cost = 0.
        missing = []
        for day in days:
            _, duration = display_and_duration(np.full(len(day.booked), shift / alpha), day.booked, alpha, h)
            plan = solve_day(day, duration, store, seconds=seconds, threads=threads, tie=tie,
                             warm_start=warm.get(day.key))
            if plan.get('assignment') is not None:
                warm[day.key] = plan['assignment']
            if plan['complete']:
                cost += metrics(day, plan['assignment'], day.actual)['cost']
            else:
                missing.append(day.key)
        result = {'shift': shift, 'cost': cost if not missing else None, 'complete': not missing,
                  'pending_days': missing}
        if store:
            store.put(key, result)
        trials.append(result)
        pending.extend(missing)
        if progress:
            progress(f'Shift alpha={alpha} h={h}: shift {shift:+d}; {len(missing)} pending days')
    if pending:
        return {'complete': False, 'trials': trials, 'pending_days': sorted(set(pending))}
    best_cost = min(t['cost'] for t in trials)
    best = min((t for t in trials if abs(t['cost'] - best_cost) <= 1e-6),
               key=lambda t: (abs(t['shift']), -t['shift']))
    return {'complete': True, 'shift': best['shift'], 'training_cost': best['cost'], 'trials': trials}
