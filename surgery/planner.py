"""Daily room assignment, deterministic ties, and response-limited oracle bounds."""
import math
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .storage import digest

TOL = 1e-6
MODEL_ID = digest([Path(__file__).read_text(), gp.gurobi.version()])


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


def solve_day(day, durations, store=None, *, seconds=300, threads=1, tie='squares', primary_only=False, warm_start=None):
    durations = np.asarray(durations, float)
    if (len(durations) != len(day.booked) or not np.isfinite(durations).all()
            or np.any(durations <= 0) or tie not in ('squares', 'lexloads')):
        raise ValueError('Invalid daily durations or tie rule')
    key = 'plan:' + digest([MODEL_ID, day.signature(), durations, tie, primary_only])
    saved = store.get(key) if store else None
    if saved and saved['complete']:
        return saved
    start = time.monotonic()
    deadline = start + seconds if seconds is not None else float('inf')
    with model(day.key, seconds, threads) as m:
        x, y = assignment_variables(m, day)
        warm = saved.get('assignment') if saved else warm_start
        if warm is not None and is_feasible(day, warm):
            for (s, r), var in x.items():
                var.Start = int(warm[s] == r)
        weights = np.bincount(day.case_surgeon, weights=durations + 30, minlength=day.n_surgeons)
        grid = next((g for g in (30., 15., 10., 5., 1., .5, .25)
                     if np.array_equal(weights, np.rint(weights / g) * g)), None)
        kind = GRB.INTEGER if grid is not None else GRB.CONTINUOUS
        unit = grid if grid is not None else 1.
        load_units = m.addVars(day.n_rooms, lb=0, vtype=kind, name='load_units')
        overtime_units = m.addVars(day.n_rooms, lb=0, vtype=kind, name='overtime_units')
        load = {r: unit * load_units[r] for r in range(day.n_rooms)}
        overtime = {r: unit * overtime_units[r] for r in range(day.n_rooms)}
        for r in range(day.n_rooms):
            m.addConstr(load[r] == gp.quicksum(weights[s] * x[s, r] for s in range(day.n_surgeons)
                                               if (s, r) in x) - 30 * y[r])
            m.addConstr(overtime[r] >= load[r] - 480 * y[r])
        objective = 510 * y.sum() + 2.75 * gp.quicksum(overtime.values()) - float(weights.sum())
        m.setObjective(objective)
        primary = optimize(m, store, 'planner_primary', day=day.key, problem_key=key)
        result = {'complete': False, 'assignment': extract(day, x) if primary['solutions'] else None,
                  'bound': primary['bound'], 'primary': primary, 'stages': [], 'tie': tie,
                  'primary_only': primary_only, 'load_grid': grid}

        def stage(label, expr, **metadata):
            m.setObjective(expr)
            m.Params.TimeLimit = max(0.001, deadline - time.monotonic()) if seconds is not None else GRB.INFINITY
            row = optimize(m, store, label, day=day.key, problem_key=key, **metadata)
            result['stages'].append(row)
            if row['solutions']:
                result['assignment'] = extract(day, x)
            return row

        if primary['optimal']:
            optimum = metrics(day, result['assignment'], durations)['cost']
            result['primary_cost'] = optimum
            result['bound'] = primary['bound']
            if primary_only:
                result['complete'] = True
            else:
                m.addConstr(objective <= optimum + 1e-8)
                balanced = True
                if tie == 'squares':
                    expr = gp.quicksum(load[r] * load[r] for r in range(day.n_rooms))
                    row = stage('planner_balance', expr)
                    balanced = row['optimal']
                    if balanced:
                        squared = metrics(day, result['assignment'], durations)['squared_load']
                        m.addQConstr(expr <= squared + 1e-7)
                        result['balance_value'] = squared
                else:
                    for k in range(1, day.n_rooms + 1):
                        t = m.addVar(lb=0)
                        e = m.addVars(day.n_rooms, lb=0)
                        for r in range(day.n_rooms):
                            m.addConstr(e[r] >= load[r] - t)
                        expr = k * t + e.sum()
                        row = stage('planner_lexload', expr, rank=k)
                        if not row['optimal']:
                            balanced = False
                            break
                        m.addConstr(expr <= row['objective'] + 1e-8)
                if balanced:
                    result['complete'] = True
                    base = max(2, day.n_rooms)
                    chunk = max(1, int(math.log(1_000_000, base)))
                    for first in range(0, day.n_surgeons, chunk):
                        indices = list(range(first, min(first + chunk, day.n_surgeons)))
                        expr = gp.quicksum(base ** (len(indices) - j - 1) * r * x[s, r]
                                           for j, s in enumerate(indices) for r in day.eligible[s])
                        row = stage('planner_assignment_tie', expr, first_surgeon=first)
                        if not row['optimal']:
                            result['complete'] = False
                            break
                        for s in indices:
                            m.addConstr(x[s, result['assignment'][s]] == 1)
        if result['assignment'] is not None:
            result['planned'] = metrics(day, result['assignment'], durations)
            if not is_feasible(day, result['assignment']):
                result.update(complete=False, reason='invalid extracted assignment')
            elif result['complete'] and abs(result['planned']['cost'] - result['primary_cost']) > TOL:
                result.update(complete=False, reason='primary objective changed during tie resolution')
            elif (result['complete'] and tie == 'squares' and not primary_only
                  and result['planned']['squared_load'] > result['balance_value'] + TOL):
                result.update(complete=False, reason='balance objective changed during tie resolution')
        result['seconds'] = time.monotonic() - start
    if store:
        store.put(key, result)
    return result


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
