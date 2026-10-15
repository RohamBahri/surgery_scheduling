"""Anchored-experiment exact weekly scheduling solver.

Single implementation for every scheduling decision made by the anchored
experiment: reference proposals, actual deployment, training acceptance,
regret certificates, and validation. The original weekly_planner is left
byte-for-byte unchanged to preserve frozen historical feature matrices.

The feasible integer schedules and the three-tier objective are identical to
the original planner. Strengthening uses disaggregated x[i,j] <= used[j] arcs
and fixes provably mandatory room-days open; it never restricts case schedules.
Time-limited solves remain uncertified.
"""
import math
import time

import numpy as np

from .weekly_planner import (
    CAPACITY, TURNOVER, OVERTIME, Result, _cost, _assignment,
    _feasible_duration, _objective_tolerance, _check_reconstruction,
    evaluate, gp_sum_moves,
)


def mandatory_slots(week):
    """Exactly forced-open physical room-days implied by surgeon-day retention."""
    forced = set()
    for s, days in enumerate(week.required_days):
        for day in days:
            eligible = [j for j in week.allowed[s] if week.slots[j][0] == day]
            if len(eligible) == 1:
                forced.add(eligible[0])
    return frozenset(forced)


def _model(week, seconds, threads):
    """Exact original Gurobi MILP with valid extended-formulation strengthening."""
    from .weekly_planner import _model as base_model
    m, x, used, by_room = base_model(week, seconds, threads)
    for i in range(week.n):
        for j in week.allowed[int(week.surgeon[i])]:
            m.addConstr(x[i, j] <= used[j], name=f'open_link_{i}_{j}')
    for j in sorted(mandatory_slots(week)):
        used[j].LB = 1.
    return m, x, used, by_room


def conservation_certificate(week, duration):
    """Sufficient certificate: all slots forced open, original attains convex LB.

    All opened K slots have total workload W=sum_i d_i+TURNOVER*(n-K);
    sum_j max(CAPACITY-L_j,0)+OVERTIME*max(L_j-CAPACITY,0)
    >= max(K*CAPACITY-W,0)+OVERTIME*max(W-K*CAPACITY,0).
    A feasible original assignment matching the bound is globally optimal.
    It is also the unique zero-move assignment.
    """
    if len(mandatory_slots(week)) != week.k:
        return None
    d = _feasible_duration(week, duration)
    total = float(d.sum()) + TURNOVER * (week.n - week.k)
    excess = total - CAPACITY * week.k
    lb = max(-excess, 0.) + OVERTIME * max(excess, 0.)
    original = float(evaluate(week, week.original, d)['cost'])
    # Absolute numerical tolerance only; never claim a demonstrably
    # non-optimal original schedule is an exact certificate.
    numeric = 1e-9 * max(1., abs(lb), abs(original))
    if original + numeric < lb:
        raise AssertionError('Conservation lower bound exceeds feasible cost')
    if original > lb + numeric:
        return None
    return dict(assignment=list(map(int, week.original)),
                value=original, bound=min(original,lb))


def historical_wins_tie(week, durations, optimal_value):
    """Sufficient, strictly interior certificate of zero-move tie optimality."""
    orig = evaluate(week, week.original, durations)['cost']
    eps = 1e-8 * max(1.,abs(float(optimal_value)))
    return orig <= optimal_value + _objective_tolerance(optimal_value) - eps


def solve_week(week, duration, *, seconds=120., threads=1, tie=True, balance=False):
    """Minimize predicted cost; optionally balance peak load; then minimize
    moved cases and choose the canonical case-wise assignment.

    The optional balance stage preserves the primary cost optimum. No realized
    durations enter any stage. All stages must be certified for complete=True.
    """
    from gurobipy import GRB

    if balance and not tie:
        raise ValueError("Load balancing requires tie=True")
    d = _feasible_duration(week, duration)
    week.validate()
    start = time.monotonic()
    if not balance:
        cert = conservation_certificate(week, d)
        if cert is not None:
            return Result(cert['assignment'], cert['value'], cert['bound'],
                          True, 'ANALYTIC_CONSERVATION', time.monotonic()-start,
                          True, True)
    if all(len(week.allowed[int(s)]) == 1 for s in week.surgeon):
        a = [int(j) for j in week.original]
        return Result(a, evaluate(week, a, d)["cost"], evaluate(week, a, d)["cost"],
                      True, "ANALYTIC", 0., True, True)

    m, x, used, by_room = _model(week, seconds, threads)
    try:
        primary = _cost(m, x, used, by_room, d, exact=False, name="plan")
        m.setObjective(primary, GRB.MINIMIZE)
        m.optimize()
        primary_elapsed = time.monotonic() - start
        optimal = m.Status == GRB.OPTIMAL
        bound = float(m.ObjBound) if m.SolCount or m.Status == GRB.TIME_LIMIT else None
        a = _assignment(week, x) if m.SolCount else None
        if a is None:
            return Result(None, None, bound, False, str(m.Status), time.monotonic() - start,
                          primary_seconds=primary_elapsed)
        optimum = _check_reconstruction(m.ObjVal, evaluate(week, a, d)["cost"])
        bound = min(bound, optimum) if bound is not None else None
        if not optimal or not tie:
            return Result(a, optimum, bound, optimal, str(m.Status),
                          time.monotonic() - start, optimal, not tie and optimal,
                          primary_seconds=primary_elapsed)

        if not balance and historical_wins_tie(week, d, m.ObjVal):
            z = list(map(int, week.original))
            historical_cost = evaluate(week, z, d)['cost']
            return Result(z, historical_cost, bound, True,
                          'OPTIMAL_ZERO_MOVES', time.monotonic()-start,
                          True, True, primary_elapsed, 0.)
        m.addConstr(primary <= float(m.ObjVal) + _objective_tolerance(m.ObjVal))
        if balance:
            import gurobipy as gp
            peak = m.addVar(lb=0., name="max_predicted_room_load")
            for j, cases in enumerate(by_room):
                total_load = (
                    gp.quicksum((float(d[i]) + TURNOVER) * x[i, j] for i in cases)
                    - TURNOVER * used[j]
                )
                m.addConstr(peak >= total_load)
            m.setObjective(peak, GRB.MINIMIZE)
            m.Params.TimeLimit = max(.001, seconds - (time.monotonic() - start))
            m.optimize()
            if m.SolCount:
                a = _assignment(week, x)
            if m.Status != GRB.OPTIMAL:
                checked = evaluate(week, a, d)["cost"]
                elapsed = time.monotonic() - start
                return Result(a, checked, bound, False, "BALANCE_TIE_" + str(m.Status),
                              elapsed, True, False,
                              primary_elapsed, elapsed-primary_elapsed)
            # Keep the optimized peak without allowing movement from its
            # certified minimum (feasibility allowance for floating point).
            m.addConstr(peak <= float(m.ObjVal) + .002)
        moved = gp_sum_moves(x, week)
        m.setObjective(moved, GRB.MINIMIZE)
        m.Params.TimeLimit = max(.001, seconds - (time.monotonic() - start))
        m.optimize()
        if m.SolCount:
            a = _assignment(week, x)
        if m.Status != GRB.OPTIMAL:
            checked = evaluate(week, a, d)["cost"]
            elapsed = time.monotonic() - start
            return Result(a, checked, bound, False, "MOVE_TIE_" + str(m.Status),
                          elapsed, True, False, primary_elapsed, elapsed-primary_elapsed)
        best_moves = int(round(m.ObjVal))
        m.addConstr(moved <= best_moves)
        # Exactly lexicographic over case_id-ordered slot assignments.
        # Keep each integer score <1e6 to avoid Gurobi numeric instability.
        base = week.k + 1
        chunk_size = max(1, int(math.log(1e6) / math.log(base)))
        canonical_complete = True
        for first in range(0, week.n, chunk_size):
            ids = list(range(first, min(week.n, first + chunk_size)))
            canonical = sum(
                base ** (len(ids) - pos - 1) * j * x[i, j]
                for pos, i in enumerate(ids)
                for j in week.allowed[int(week.surgeon[i])]
            )
            m.setObjective(canonical, GRB.MINIMIZE)
            m.Params.TimeLimit = max(.001, seconds - (time.monotonic() - start))
            m.optimize()
            if m.SolCount:
                a = _assignment(week, x)
            if m.Status != GRB.OPTIMAL:
                canonical_complete = False
                break
            for i in ids:
                m.addConstr(x[i, int(a[i])] == 1)
        checked = evaluate(week, a, d)["cost"]
        if checked > optimum + 2 * _objective_tolerance(optimum):
            raise AssertionError("Lexicographic tie changed the primary cost materially")
        elapsed = time.monotonic() - start
        return Result(a, checked, bound, canonical_complete,
                      'OPTIMAL' if canonical_complete else 'CANONICAL_TIE_' + str(m.Status),
                      elapsed, True, canonical_complete,
                      primary_elapsed, elapsed-primary_elapsed)
    finally:
        m.dispose()


def solve_adversary(week, predicted, *, gamma=2., seconds=120., threads=1):
    """MAX_z [J(z;actual) - gamma*J(z;predicted)], over the FULL feasible set.

    Returns a feasible maximization value and a valid solver upper bound.
    Exactly the same eligible assignments and surgeon-day constraints as the
    deployment planner. This is a primitive for testing gap-VF/SPO losses.
    """
    from gurobipy import GRB

    if not np.isfinite(gamma) or gamma < 1.:
        raise ValueError("gamma must be at least 1")
    d = _feasible_duration(week, predicted)
    week.validate()
    start = time.monotonic()
    if all(len(week.allowed[int(s)]) == 1 for s in week.surgeon):
        a = [int(j) for j in week.original]
        value = evaluate(week, a, week.actual)["cost"] - gamma * evaluate(week, a, d)["cost"]
        return Result(a, value, value, True, "ANALYTIC", 0., True, True)

    m, x, used, by_room = _model(week, seconds, threads)
    try:
        actual = _cost(m, x, used, by_room, week.actual, exact=True, name="actual")
        plan = _cost(m, x, used, by_room, d, exact=False, name="predicted")
        m.setObjective(actual - gamma * plan, GRB.MAXIMIZE)
        m.optimize()
        bound = float(m.ObjBound) if m.SolCount or m.Status == GRB.TIME_LIMIT else None
        a = _assignment(week, x) if m.SolCount else None
        if a is None:
            return Result(None, None, bound, False, str(m.Status), time.monotonic() - start)
        realized = (evaluate(week, a, week.actual)["cost"]
                    - gamma * evaluate(week, a, d)["cost"])
        value = _check_reconstruction(m.ObjVal, realized)
        bound = max(bound, value) if bound is not None else None
        done = m.Status == GRB.OPTIMAL
        return Result(a, value, bound, done, str(m.Status),
                      time.monotonic() - start, done, done)
    finally:
        m.dispose()


def evaluate_week(task):
    """Development validation with the SAME exact strengthened deployed planner."""
    from run_weekly_rich import oracle_assignment_sha256, verify_bounds
    from .weekly_training import asymmetric_error
    week, model_durations, seconds, threads = task
    comps = week.components()
    oracle = [solve_week(c, c.actual, seconds=seconds,
                         threads=threads, tie=False) for c in comps]
    oracle_ok = all(o.complete and o.assignment is not None for o in oracle)
    fingerprint = oracle_assignment_sha256(comps, oracle) if oracle_ok else None
    rows = []
    for method, durations in model_durations.items():
        accumulated = {key: 0. for key in (
            "cases", "abs_error", "asym_error", "pred_cost",
            "realized_cost", "oracle_cost", "regret", "vf_bound",
            "idle", "overtime", "moved_cases",
            "room_days_over_720", "changed_days")}
        success = oracle_ok
        full = dict(zip(map(int, week.case_ids), map(float, durations)))
        for c, o in zip(comps, oracle):
            if not success:
                break
            d = np.array([full[int(cid)] for cid in c.case_ids], float)
            planned = solve_week(c, d, seconds=seconds, threads=threads, tie=True)
            if not planned.complete or planned.assignment is None:
                success = False
                break
            pred = evaluate(c, planned.assignment, d)
            real = evaluate(c, planned.assignment, c.actual)
            error = c.actual - d
            asymmetric = float(asymmetric_error(error).sum())
            regret = real["cost"] - o.value
            bound = pred["cost"] + asymmetric - o.value
            verify_bounds(regret, bound, bound, bound)
            quantities = {
                "cases": c.n, "abs_error": np.abs(error).sum(),
                "asym_error": asymmetric, "pred_cost": pred["cost"],
                "realized_cost": real["cost"], "oracle_cost": o.value,
                "regret": regret, "vf_bound": bound, "idle": real["idle"],
                "overtime": real["overtime"], "moved_cases": real["moved_cases"],
                "room_days_over_720": real["room_days_over_720"],
                "changed_days": real["moved_weekdays"]
            }
            for field, val in quantities.items():
                accumulated[field] += float(val)
        rows.append({
            "group": week.group, "monday": week.monday, "method": method,
            "complete": bool(success), "oracle_assignment_sha256": fingerprint,
            **(accumulated if success else {key: np.nan for key in accumulated})
        })
    return rows


