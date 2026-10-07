"""Daily room assignment, deterministic ties, and response-limited oracle bounds."""
import math
import time
import warnings
from pathlib import Path

import gurobipy as gp
import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from gurobipy import GRB

from .storage import digest

TOL = 1e-6
MODEL_ID = digest([Path(__file__).read_text(), gp.gurobi.version(), scipy.__version__])


def response(display, alpha, h):
    u = np.asarray(display, float)
    if h is None:
        return u.copy()
    return np.sign(u) * np.minimum(alpha * abs(u), np.maximum(h - (1 - alpha) * abs(u), 0))


def display_and_duration(raw, booked, alpha, h):
    b = np.asarray(booked)
    u = np.clip(raw, np.maximum(-180, 1 - b), 180)
    return u, b + response(u, alpha, h)


def reachable(booked, alpha, h):
    b = np.asarray(booked)
    return b - alpha * np.minimum(np.minimum(h, 180), b - 1), b + alpha * min(h, 180)


def metrics(day, assignment, durations, *, case_assignment=False):
    assignment = np.asarray(assignment, int)
    room = assignment if case_assignment else assignment[day.case_surgeon]
    counts = np.bincount(room, minlength=day.n_rooms)
    opened = counts > 0
    load = np.bincount(room, weights=np.asarray(durations) + 30, minlength=day.n_rooms) - 30 * opened
    excess = load - 480 * opened
    overtime, idle = np.maximum(excess, 0), np.maximum(-excess, 0)
    return {'cost': float((1.75 * overtime + idle).sum()), 'overtime': float(overtime.sum()),
            'idle': float(idle.sum()), 'rooms': int(opened.sum()), 'loads': load[opened].tolist(),
            'squared_load': float(load @ load), 'max_load': float(load.max(initial=0))}


def is_feasible(day, assignment):
    return (len(assignment) == day.n_surgeons
            and all(r == int(r) and int(r) in day.eligible[s] for s, r in enumerate(assignment)))


def model(name, seconds, threads):
    m = gp.Model(name)
    m.Params.OutputFlag = 0
    m.Params.Threads = threads
    m.Params.Seed = 0
    m.Params.MIPGap = 0
    m.Params.MIPGapAbs = 1e-8
    m.Params.FeasibilityTol = 1e-9
    m.Params.IntFeasTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    if seconds is not None:
        m.Params.TimeLimit = max(0.001, seconds)
    return m


def optimize(m, store, label, **metadata):
    start = time.monotonic()
    result = {'label': label, **metadata}
    try:
        m.update()
        result.update(variables=m.NumVars, constraints=m.NumConstrs,
                      solver='.'.join(map(str, gp.gurobi.version())))
        m.optimize()
        bound = float(m.ObjBound) if m.IsMIP else (float(m.ObjVal) if m.Status == GRB.OPTIMAL else None)
        result.update(status=int(m.Status), optimal=m.Status == GRB.OPTIMAL,
                      objective=float(m.ObjVal) if m.SolCount else None,
                      bound=bound if bound is not None and abs(bound) < GRB.INFINITY else None,
                      solutions=int(m.SolCount), variables=m.NumVars, constraints=m.NumConstrs,
                      solver='.'.join(map(str, gp.gurobi.version())))
    except gp.GurobiError as exc:
        result.update(status='error', optimal=False, solutions=0, bound=None, objective=None,
                      error=str(exc), error_code=exc.errno)
    result['seconds'] = time.monotonic() - start
    if store:
        store.log(result)
    return result


def assignment_variables(m, day):
    x = m.addVars([(s, r) for s, rooms in enumerate(day.eligible) for r in rooms],
                  vtype=GRB.BINARY, name='assign')
    y = m.addVars(day.n_rooms, vtype=GRB.BINARY, name='open')
    for s, rooms in enumerate(day.eligible):
        m.addConstr(gp.quicksum(x[s, r] for r in rooms) == 1)
    equivalent = {}
    for r in range(day.n_rooms):
        members = tuple(s for s in range(day.n_surgeons) if (s, r) in x)
        for s in members:
            m.addConstr(x[s, r] <= y[r])
        m.addConstr(y[r] <= gp.quicksum(x[s, r] for s in members))
        equivalent.setdefault(members, []).append(r)
    # Exchangeable room labels can be ordered by their first assigned surgeon.
    for members, rooms in equivalent.items():
        for left, right in zip(rooms, rooms[1:]):
            for s in members:
                m.addConstr(x[s, right] <= gp.quicksum(x[t, left] for t in members if t < s))
    return x, y


def extract(day, x):
    return [max(rooms, key=lambda r: x[s, r].X) for s, rooms in enumerate(day.eligible)]


def canonical_chunks(day):
    base = max(2, day.n_rooms)
    chunk = max(1, int(math.log(1_000_000, base)))
    for first in range(0, day.n_surgeons, chunk):
        ids = list(range(first, min(first + chunk, day.n_surgeons)))
        yield ids, [base ** (len(ids) - j - 1) for j in range(len(ids))]


def compact_plan(day, durations, warm, store, deadline, threads, primary_only, key):
    with model(day.key, max(.001, deadline - time.monotonic()), threads) as m:
        x, y = assignment_variables(m, day)
        for (s, r), var in x.items():
            var.Start = int(warm[s] == r)
        weights = np.bincount(day.case_surgeon, weights=durations + 30, minlength=day.n_surgeons)
        grid = next((g for g in (30., 15., 10., 5., 1., .5, .25)
                     if np.array_equal(weights, np.rint(weights / g) * g)), None)
        kind, unit = (GRB.INTEGER, grid) if grid is not None else (GRB.CONTINUOUS, 1.)
        load_units = m.addVars(day.n_rooms, lb=0, vtype=kind)
        overtime_units = m.addVars(day.n_rooms, lb=0, vtype=kind)
        load = {r: unit * load_units[r] for r in range(day.n_rooms)}
        overtime = {r: unit * overtime_units[r] for r in range(day.n_rooms)}
        for r in range(day.n_rooms):
            m.addConstr(load[r] == gp.quicksum(weights[s] * x[s, r] for s in range(day.n_surgeons)
                                               if (s, r) in x) - 30 * y[r])
            m.addConstr(overtime[r] >= load[r] - 480 * y[r])
        cost = 510 * y.sum() + 2.75 * gp.quicksum(overtime.values()) - float(weights.sum())
        result = {'complete': False, 'assignment': warm, 'bound': 0., 'stages': [], 'backend': 'compact'}

        def stage(label, expr, **metadata):
            m.setObjective(expr)
            m.Params.TimeLimit = max(.001, deadline - time.monotonic())
            row = optimize(m, store, label, day=day.key, problem_key=key, **metadata)
            if row['solutions']:
                result['assignment'] = extract(day, x)
            return row

        row = stage('planner_primary', cost)
        result.update(primary=row, bound=row['bound'] if row['bound'] is not None else 0.)
        if not row['optimal']:
            return result
        optimum = metrics(day, result['assignment'], durations)['cost']
        result['primary_cost'] = optimum
        if primary_only:
            result['complete'] = True
            return result
        m.addConstr(cost <= optimum + 1e-8)
        peak = m.addVar(lb=float(weights.max() - 30))
        for r in range(day.n_rooms):
            m.addConstr(peak >= load[r])
        row = stage('planner_max_load', peak)
        result['stages'].append(row)
        if not row['optimal']:
            return result
        maximum = metrics(day, result['assignment'], durations)['max_load']
        result['balance_value'] = maximum
        m.addConstr(peak <= maximum + 1e-8)
        for ids, powers in canonical_chunks(day):
            expr = gp.quicksum(power * r * x[s, r] for s, power in zip(ids, powers) for r in day.eligible[s])
            row = stage('planner_assignment_tie', expr, first_surgeon=ids[0])
            result['stages'].append(row)
            if not row['optimal']:
                return result
            for s in ids:
                m.addConstr(x[s, result['assignment'][s]] == 1)
        result['complete'] = True
        return result


def pattern_plan(day, durations, warm, store, deadline, threads, primary_only, key):
    """Complete subsets with interchangeable rooms represented by a capacity, not duplicate columns."""
    weights = np.bincount(day.case_surgeon, weights=durations + 30, minlength=day.n_surgeons)
    upper = metrics(day, warm, durations)['cost']
    limit = 510 + (upper + TOL) / 1.75
    types = {}
    for r in range(day.n_rooms):
        members = tuple(s for s, rooms in enumerate(day.eligible) if r in rooms)
        types.setdefault(members, []).append(r)
    rooms_by_type = list(types.values())
    room_type = {r: t for t, rooms in enumerate(rooms_by_type) for r in rooms}
    masks = [sum(1 << t for t, members in enumerate(types) if s in members) for s in range(day.n_surgeons)]
    patterns, loads, costs = [], [], []
    result = {'complete': False, 'assignment': warm, 'bound': 0., 'stages': [], 'backend': 'patterns'}

    def enumerate_subsets(first, members, weight, common):
        if time.monotonic() >= deadline:
            raise TimeoutError
        for s in range(first, day.n_surgeons):
            eligible, total = common & masks[s], weight + weights[s]
            if not eligible or total > limit:
                continue
            group, load = members + (s,), total - 30
            cost = max(480 - load, 0) + 1.75 * max(load - 480, 0)
            if cost <= upper + TOL:
                for t in range(len(types)):
                    if eligible & (1 << t):
                        patterns.append((group, t)); loads.append(load); costs.append(cost)
            enumerate_subsets(s + 1, group, total, eligible)

    try:
        enumerate_subsets(0, (), 0., (1 << len(types)) - 1)
    except TimeoutError:
        result['reason'] = 'pattern enumeration pending; partial enumeration is never certified'
        return result
    costs, loads = np.asarray(costs), np.asarray(loads)
    result.update(patterns=len(patterns), room_types=len(types))
    active = np.ones(len(patterns), dtype=bool)

    def stage(label, prefix=None, maximum=float('inf'), **metadata):
        begin = time.monotonic()
        prefix = prefix or {}
        fixed = {r: {s for s in prefix if prefix[s] == r} for r in set(prefix.values())}
        free = {t: [r for r in rooms if r not in fixed] for t, rooms in enumerate(rooms_by_type)}
        buckets = [('fixed', r) for r in sorted(fixed)] + [('free', t) for t in free if free[t]]
        bucket_id = {value: day.n_surgeons + j for j, value in enumerate(buckets)}
        allowed, rr, cc = [], [], []
        for j, (members, t) in enumerate(patterns):
            if not active[j] or loads[j] > maximum + 1e-8:
                continue
            assigned = {prefix[s] for s in members if s in prefix}
            if len(assigned) > 1:
                continue
            if assigned:
                room = next(iter(assigned))
                if room_type[room] != t or not fixed[room].issubset(members):
                    continue
                bucket = ('fixed', room)
            else:
                bucket = ('free', t)
                if not free[t]:
                    continue
            col = len(allowed)
            allowed.append(j)
            rr.extend((*members, bucket_id[bucket])); cc.extend([col] * (len(members) + 1))
        columns = len(allowed)
        row = {'label': label, 'day': day.key, 'problem_key': key, **metadata,
               'variables': columns, 'constraints': day.n_surgeons + len(buckets),
               'solver': 'HiGHS via SciPy ' + scipy.__version__, 'solutions': 0, 'bound': None,
               'objective': None, 'optimal': False, 'feasible': False}
        if not columns:
            row.update(status=2, message='No compatible patterns')
        elif time.monotonic() >= deadline:
            row.update(status=1, message='Daily time budget exhausted')
        else:
            A = sparse.csc_matrix((np.ones(len(rr)), (rr, cc)), shape=(row['constraints'], columns))
            capacity = np.r_[np.ones(day.n_surgeons), [1 if kind == 'fixed' else len(free[v]) for kind, v in buckets]]
            if 'primary_cost' not in result:
                # A dual-feasible relaxation removes columns that cannot belong to any incumbent-better plan.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
                    relaxed = linprog(costs[allowed], A_ub=A[day.n_surgeons:], b_ub=capacity[day.n_surgeons:],
                        A_eq=A[:day.n_surgeons], b_eq=np.ones(day.n_surgeons), bounds=(0, None), method='highs',
                        options={'threads': threads, 'time_limit': max(.001, deadline - time.monotonic()),
                                 'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9})
                lp_row = dict(row, label='planner_pattern_relaxation', status=int(relaxed.status),
                              optimal=relaxed.status == 0, objective=relaxed.fun, seconds=time.monotonic() - begin)
                if relaxed.status != 0:
                    if store:
                        store.log(lp_row)
                    return dict(lp_row, feasible=False)
                dual = np.r_[relaxed.eqlin.marginals, np.minimum(relaxed.ineqlin.marginals, 0)]
                reduced = costs[allowed] - A.T @ dual
                dual[:day.n_surgeons] -= max(0., -float(reduced.min())) + 1e-8
                reduced = costs[allowed] - A.T @ dual
                lower_bound = float(capacity @ dual)
                keep = lower_bound + reduced <= upper + TOL
                lp_row['bound'] = lower_bound
                if store:
                    store.log(lp_row)
                active[np.asarray(allowed)[~keep]] = False
                allowed = np.asarray(allowed)[keep].tolist()
                A = A[:, keep]
                columns = len(allowed)
                row['variables'] = columns
                result['patterns_after_dual_screen'] = columns
            con = [LinearConstraint(A, np.r_[np.ones(day.n_surgeons), np.zeros(len(buckets))], capacity)]
            objective = costs[allowed]
            if 'primary_cost' in result:
                con.append(LinearConstraint(sparse.csc_matrix(objective[None, :]), -np.inf, result['primary_cost'] + 1e-8))
                objective = np.zeros(columns)
            row['constraints'] = sum(c.A.shape[0] for c in con)
            if store:
                store.put('active_solve:' + key, row)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
                raw = milp(objective, integrality=np.ones(columns), bounds=Bounds(0, 1), constraints=con,
                           options={'time_limit': max(.001, deadline - time.monotonic()), 'mip_rel_gap': 0.,
                                    'mip_abs_gap': 1e-8, 'threads': threads, 'mip_feasibility_tolerance': 1e-9,
                                    'presolve': 'primary_cost' not in result})
            finite = lambda v: float(v) if v is not None and np.isfinite(v) else None
            row.update(status=int(raw.status), optimal=raw.status == 0, objective=finite(raw.fun),
                       bound=finite(raw.get('mip_dual_bound')), solutions=int(raw.x is not None), message=raw.message)
            if raw.x is not None:
                total = A @ np.rint(raw.x)
                valid = (np.max(np.abs(total[:day.n_surgeons] - 1)) < TOL and
                         np.all(total <= capacity + TOL) and np.min(raw.x) >= -TOL and
                         np.max(raw.x) <= 1 + TOL and np.max(np.abs(raw.x - np.rint(raw.x))) < TOL)
                assignment = [-1] * day.n_surgeons
                if valid:
                    chosen = [patterns[allowed[j]] for j in np.flatnonzero(raw.x > .5)]
                    available = {t: list(rooms) for t, rooms in free.items()}
                    for members, t in sorted(chosen):
                        named = {prefix[s] for s in members if s in prefix}
                        room = next(iter(named)) if named else available[t].pop(0)
                        for member in members:
                            assignment[member] = room
                    valid = is_feasible(day, assignment)
                if valid:
                    value = metrics(day, assignment, durations)
                    valid = (value['max_load'] <= maximum + TOL and
                             value['cost'] <= result.get('primary_cost', upper) + TOL)
                if valid:
                    result['assignment'] = assignment
                    row['feasible'] = True
                else:
                    row.update(optimal=False, message='Extracted pattern assignment failed numerical validation',
                               extracted=metrics(day, assignment, durations) if is_feasible(day, assignment) else assignment,
                               integrality_residual=float(np.max(np.abs(raw.x - np.rint(raw.x)))), expected_cost=result.get('primary_cost', upper),
                               expected_maximum=maximum if np.isfinite(maximum) else None,
                               coverage_residual=float(np.max(np.abs(total[:day.n_surgeons] - 1))),
                               capacity_violation=float(np.max(total - capacity)))
        row['seconds'] = time.monotonic() - begin
        if store:
            store.log(row)
            store.put('active_solve:' + key, None)
        return row

    row = stage('planner_pattern_primary')
    result.update(primary=row, bound=row['bound'] if row['bound'] is not None else 0.)
    if not row['optimal'] or not row['feasible']:
        return result
    result['primary_cost'] = metrics(day, result['assignment'], durations)['cost']
    if abs(result['primary_cost'] - row['objective']) > TOL:
        result['reason'] = 'pattern solution failed objective validation'
        return result
    if primary_only:
        result['complete'] = True
        return result
    lower = weights.max() - 30
    upper_load = metrics(day, result['assignment'], durations)['max_load']
    candidates = np.unique(loads[(loads >= lower - 1e-8) & (loads <= upper_load + 1e-8)])
    left, right = 0, len(candidates) - 1
    while left < right:
        mid = (left + right) // 2
        row = stage('planner_pattern_max_load', maximum=candidates[mid], threshold=float(candidates[mid]))
        result['stages'].append(row)
        if row['feasible']:
            right = mid
        elif row['status'] == 2:
            left = mid + 1
        else:
            return result
    maximum = float(candidates[left])
    result['balance_value'] = maximum
    prefix = {}
    for s in range(day.n_surgeons):
        for room in day.eligible[s]:
            if room == result['assignment'][s]:
                prefix[s] = room
                break
            if weights[s] + sum(weights[t] for t, r in prefix.items() if r == room) - 30 > maximum + TOL:
                continue
            row = stage('planner_pattern_assignment_tie', {**prefix, s: room}, maximum,
                        surgeon=s, room=room)
            result['stages'].append(row)
            if row['feasible']:
                prefix[s] = room
                break
            if row['status'] != 2:
                return result
    result['complete'] = True
    return result


def solve_day(day, durations, store=None, *, seconds=300, threads=1, tie='maxload',
              primary_only=False, warm_start=None, backend='auto'):
    durations = np.asarray(durations, float)
    if (len(durations) != len(day.booked) or not np.isfinite(durations).all()
            or np.any(durations <= 0) or tie != 'maxload' or backend not in ('auto', 'compact', 'patterns')):
        raise ValueError('Invalid daily durations, tie rule, or backend')
    key = 'plan:' + digest([MODEL_ID, day.signature(), durations, tie, primary_only, backend])
    saved = store.get(key) if store else None
    if saved and saved['complete']:
        return saved
    start = time.monotonic()
    deadline = start + seconds if seconds is not None else float('inf')
    warm = saved.get('assignment') if saved else warm_start
    if warm is None or not is_feasible(day, warm):
        warm = [rooms[0] for rooms in day.eligible]
        # A feasible incumbent bounds enumeration; moving one surgeon never compromises eligibility.
        for _ in range(2):
            for s in sorted(range(day.n_surgeons), key=lambda s: -sum(durations[day.case_surgeon == s])):
                warm[s] = min(day.eligible[s], key=lambda r: metrics(day, warm[:s] + [r] + warm[s + 1:], durations)['cost'])
    use_patterns = backend == 'patterns'
    if use_patterns:
        result = pattern_plan(day, durations, warm, store, deadline, threads, primary_only, key)
    else:
        first_deadline = min(deadline, start + min(2., seconds / 4)) if backend == 'auto' and seconds is not None else deadline
        result = compact_plan(day, durations, warm, store, first_deadline, threads, primary_only, key)
        if backend == 'auto' and not result['complete'] and time.monotonic() < deadline:
            first = result
            result = pattern_plan(day, durations, first['assignment'], store, deadline, threads, primary_only, key)
            result['compact_attempt'] = first
            result['bound'] = max(result['bound'], first['bound'])
    result.update(tie=tie, primary_only=primary_only, seconds=time.monotonic() - start)
    result['planned'] = metrics(day, result['assignment'], durations)
    if result['complete']:
        if abs(result['planned']['cost'] - result['primary_cost']) > TOL:
            result.update(complete=False, reason='primary objective changed during tie resolution')
        elif not primary_only and result['planned']['max_load'] > result['balance_value'] + TOL:
            result.update(complete=False, reason='maximum load changed during tie resolution')
    if store:
        store.put(key, result)
    return result


def plan_job(day, durations, options, store=None):
    return solve_day(day, durations, store, **options)


def response_oracle(day, alpha, h, booked_plan, actual_plan, store=None, *, seconds=300, threads=1):
    """Optimistic primary-cost benchmark; unfinished master/follower solves retain bounds."""
    started = time.monotonic()
    key = 'oracle:' + digest([MODEL_ID, day.signature(), day.booked, day.actual, alpha, h])
    previous = store.get(key) if store else None
    if previous and previous['complete']:
        return previous
    if not booked_plan.get('complete') or not actual_plan.get('complete'):
        return {'complete': False, 'reason': 'exact seed plans pending', 'lower': None, 'upper': None}
    cuts = previous['cuts'] if previous else [booked_plan['assignment'], actual_plan['assignment']]
    lower = max(0., actual_plan['bound'], previous['lower'] if previous else 0.)
    upper = min(metrics(day, booked_plan['assignment'], day.actual)['cost'],
                previous['upper'] if previous else float('inf'))
    best = previous['assignment'] if previous else booked_plan['assignment']
    witness = previous['durations'] if previous else day.booked.tolist()
    lo, hi = reachable(day.booked, alpha, h)
    wl = np.bincount(day.case_surgeon, weights=lo + 30, minlength=day.n_surgeons)
    wu = np.bincount(day.case_surgeon, weights=hi + 30, minlength=day.n_surgeons)
    actual_w = np.bincount(day.case_surgeon, weights=day.actual + 30, minlength=day.n_surgeons)
    deadline = time.monotonic() + seconds
    result = {'complete': False, 'lower': lower, 'upper': upper, 'assignment': best,
              'durations': witness, 'cuts': cuts, 'iterations': previous['iterations'] if previous else 0,
              'master_optimal': False, 'follower_confirmed': False}
    with model('response_oracle_' + day.key, seconds, threads) as m:
        x, y = assignment_variables(m, day)
        w = m.addVars(day.n_surgeons, lb=wl.tolist(), ub=wu.tolist(), name='surgeon_load')
        witness_w = np.bincount(day.case_surgeon, weights=np.asarray(witness) + 30, minlength=day.n_surgeons)
        for s in range(day.n_surgeons):
            w[s].Start = witness_w[s]
        for (s, r), var in x.items():
            var.Start = int(best[s] == r)
        product = m.addVars(list(x), lb=0)
        for s, r in x:
            z = product[s, r]
            m.addConstr(z >= wl[s] * x[s, r])
            m.addConstr(z <= wu[s] * x[s, r])
            m.addConstr(z >= w[s] - wu[s] * (1 - x[s, r]))
            m.addConstr(z <= w[s] - wl[s] * (1 - x[s, r]))
        op = m.addVars(day.n_rooms, lb=0)
        oa = m.addVars(day.n_rooms, lb=0)
        for r in range(day.n_rooms):
            m.addConstr(op[r] >= gp.quicksum(product[s, r] for s in range(day.n_surgeons)
                                            if (s, r) in x) - 510 * y[r])
            m.addConstr(oa[r] >= gp.quicksum(actual_w[s] * x[s, r] for s in range(day.n_surgeons)
                                            if (s, r) in x) - 510 * y[r])
        planned_cost = 510 * y.sum() + 2.75 * op.sum()
        m.setObjective(510 * y.sum() + 2.75 * oa.sum() - float(actual_w.sum()))
        known = set()

        def add_cut(assignment):
            signature = tuple(assignment)
            if signature in known:
                return False
            known.add(signature)
            terms = []
            for r in sorted(set(assignment)):
                members = [s for s, room in enumerate(assignment) if room == r]
                excess = m.addVar(lb=float(wl[members].sum() - 510), ub=float(wu[members].sum() - 510))
                over = m.addVar(lb=0, ub=max(0., float(wu[members].sum() - 510)))
                m.addConstr(excess == gp.quicksum(w[s] for s in members) - 510)
                m.addGenConstrMax(over, [excess], constant=0)
                terms.append(over)
            m.addConstr(planned_cost <= 510 * len(set(assignment)) + 2.75 * gp.quicksum(terms))
            return True

        for cut in cuts:
            add_cut(cut)
        while time.monotonic() < deadline:
            m.Params.TimeLimit = max(0.001, deadline - time.monotonic())
            row = optimize(m, store, 'response_master', day=day.key, alpha=alpha, h=h,
                           iteration=result['iterations'], problem_key=key)
            result['iterations'] += 1
            if row['bound'] is not None:
                lower = max(lower, row['bound'])
            if not row['solutions']:
                break
            candidate = extract(day, x)
            totals = np.array([w[s].X for s in range(day.n_surgeons)])
            fraction = np.divide(totals - wl, wu - wl, out=np.zeros_like(wl), where=wu > wl)
            duration = lo + (hi - lo) * np.clip(fraction[day.case_surgeon], 0, 1)
            follower = solve_day(day, duration, store, seconds=max(0.001, deadline - time.monotonic()),
                                 threads=threads, primary_only=True)
            selected_cost = metrics(day, candidate, duration)['cost']
            confirmed = (follower['complete'] and
                         abs(selected_cost - follower['primary_cost']) <= TOL)
            if follower['complete']:
                for assignment in [follower['assignment']] + ([candidate] if confirmed else []):
                    value = metrics(day, assignment, day.actual)['cost']
                    if value < upper:
                        upper, best, witness = value, assignment, duration.tolist()
            violation = (follower.get('assignment') is not None and
                         selected_cost > metrics(day, follower['assignment'], duration)['cost'] + TOL)
            added = add_cut(follower['assignment']) if violation else False
            result.update(lower=lower, upper=upper, assignment=best, durations=witness,
                          cuts=[list(c) for c in sorted(known)], master_optimal=row['optimal'],
                          follower_confirmed=confirmed,
                          complete=bool(row['optimal'] and confirmed and abs(upper - lower) <= TOL))
            if store:
                store.put(key, result)
            if result['complete'] or not added:
                break
    result.update(lower=lower, upper=upper, assignment=best, durations=witness,
                  cuts=[list(c) for c in sorted(known)],
                  seconds=(previous.get('seconds', 0.) if previous else 0.) + time.monotonic() - started)
    if lower > upper + TOL:
        result.update(complete=False, reason='inconsistent numerical bounds; needs tighter solve')
    if store:
        store.put(key, result)
    return result
