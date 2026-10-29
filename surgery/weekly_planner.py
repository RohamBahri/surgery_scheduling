"""Exact weekly planner and loss-augmented oracle on small physical-room components.

No opening, deferral, or sequencing. Gurobi is imported only when solving.
The primary scheduling objective is never contaminated with realized durations.
"""
from dataclasses import asdict, dataclass
import math
import os
import time

import numpy as np

from .weekly_data import Week

CAPACITY = 480.
TURNOVER = float(os.environ.get('SURGERY_TURNOVER_MINUTES', '30'))
RELAX_REQUIRED_DAYS = os.environ.get('SURGERY_RELAX_REQUIRED_DAYS','0') == '1'
OVERTIME = 1.75


@dataclass
class Result:
    assignment: list | None
    value: float | None          # feasible cost (planner) or objective (max adversary)
    bound: float | None          # MIN lower bound / MAX upper bound
    complete: bool
    status: str
    seconds: float
    primary_complete: bool = False
    tie_complete: bool = False
    primary_seconds: float = 0.
    tie_seconds: float = 0.

    def as_dict(self):
        return asdict(self)


def loads(week: Week, assignment, durations):
    d = np.asarray(durations, float)
    a = np.asarray(assignment, int)
    if d.shape != (week.n,) or a.shape != (week.n,):
        raise ValueError("Assignment and duration lengths must match the component")
    if not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("Durations must be positive and finite")
    if any(not 0 <= int(j) < week.k or int(j) not in week.allowed[int(week.surgeon[i])]
           for i, j in enumerate(a)):
        raise ValueError("Ineligible case-to-room assignment")
    if not RELAX_REQUIRED_DAYS:
        for s, days in enumerate(week.required_days):
            members = np.flatnonzero(week.surgeon == s)
            for wd in days:
                if not any(week.slots[a[i]][0] == wd for i in members):
                    raise ValueError("An observed surgeon-day was emptied")
    counts=np.bincount(a,minlength=week.k)
    return (np.bincount(a,weights=d+TURNOVER,minlength=week.k)
            -TURNOVER*(counts>0))


def evaluate(week: Week, assignment, durations):
    room_load = loads(week, assignment, durations)
    overtime = np.maximum(room_load - CAPACITY, 0.)
    idle = np.maximum(CAPACITY - room_load, 0.)
    a = np.asarray(assignment, int)
    changed = sum(int(a[i] != week.original[i]) for i in range(week.n))
    changed_day = sum(int(week.slots[a[i]][0] != week.slots[week.original[i]][0])
                      for i in range(week.n))
    # A workload diagnostic, NOT a feasibility constraint or surgeon-time cost.
    surgeon_day = {}
    kept_days = {s:set() for s in range(len(week.surgeon_ids))}
    for i in range(week.n):
        kept_days[int(week.surgeon[i])].add(week.slots[a[i]][0])
        key = (int(week.surgeon[i]), week.slots[a[i]][0])
        surgeon_day[key] = surgeon_day.get(key, 0.) + float(durations[i])
    return {
        "cost": float(idle.sum() + OVERTIME * overtime.sum()),
        "idle": float(idle.sum()), "overtime": float(overtime.sum()),
        "max_room_load": float(room_load.max(initial=0)),
        "room_days_over_720": int((room_load > 720).sum()),
        "max_surgeon_case_minutes": max(surgeon_day.values(), default=0.),
        "surgeon_days_over_480": int(sum(v > CAPACITY for v in surgeon_day.values())),
        "emptied_historical_surgeon_days": int(sum(
            wd not in kept_days[s] for s,days in enumerate(week.required_days)
            for wd in days)),
        "moved_cases": int(changed), "moved_weekdays": int(changed_day),
        "used_extra_days": int(sum(
            any(week.surgeon[i] == week.surgeon_ids.index(s)
                and week.slots[a[i]][0] == wd for i in range(week.n))
            for s, wd, _, _ in week.added_days
        )),
    }


def _objective_tolerance(value):
    # Feasibility allowance for post-optimum objective constraints. It is NOT
    # an optimality-gap relaxation. Values are independently recomputed below.
    return min(.1, max(.002, 2e-6 * abs(float(value))))


def _check_reconstruction(solver_value, reconstructed):
    scale = max(1., abs(float(solver_value)), abs(float(reconstructed)))
    delta = abs(float(solver_value) - float(reconstructed))
    allowance = min(.1, max(.002, 2e-6 * scale))
    if not np.isfinite(delta) or delta > allowance:
        raise AssertionError(
            f"Gurobi cost does not match recomputed schedule: "
            f"{solver_value} vs {reconstructed} (difference {delta})")
    return float(reconstructed)


def _model(week, seconds, threads):
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("weekly_" + week.group.replace("-", "_"))
    m.Params.OutputFlag = 0
    m.Params.Threads = threads
    m.Params.Seed = 0
    m.Params.MIPGap = 0
    m.Params.MIPGapAbs = 0
    m.Params.FeasibilityTol = 1e-9
    m.Params.IntFeasTol = 1e-9
    m.Params.OptimalityTol = 1e-9
    m.Params.TimeLimit = max(.001, seconds)

    arcs = [(i, j) for i in range(week.n)
            for j in week.allowed[int(week.surgeon[i])]]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    for i in range(week.n):
        m.addConstr(gp.quicksum(x[i, j] for j in week.allowed[int(week.surgeon[i])]) == 1)
    if not RELAX_REQUIRED_DAYS:
        for s, weekdays in enumerate(week.required_days):
            cases = np.flatnonzero(week.surgeon == s)
            for weekday in weekdays:
                m.addConstr(gp.quicksum(
                    x[int(i), j] for i in cases for j in week.allowed[s]
                    if week.slots[j][0] == weekday
                ) >= 1, name=f"retain_day_{s}_{weekday}")

    by_room = [[] for _ in week.slots]
    for i, j in arcs:
        by_room[j].append(i)
    used = m.addVars(week.k, vtype=GRB.BINARY, name="used")
    for j, cases in enumerate(by_room):
        count = gp.quicksum(x[i, j] for i in cases)
        m.addConstr(count >= used[j])
        m.addConstr(count <= len(cases) * used[j])
    for i in range(week.n):
        x[i, int(week.original[i])].Start = 1.
    return m, x, used, by_room


def _cost(m, x, used, by_room, duration, *, exact, name):
    """J=480K-sum(d)-turnover*n+turnover*used+2.75sum(OT).

    MAX oracles require exact actual-duration overtime, enforced by a binary
    disjunction. The negative predicted-cost term automatically tightens its
    overtime variables, so it needs only the standard epigraph.
    """
    import gurobipy as gp
    from gurobipy import GRB

    d = np.asarray(duration, float)
    overtime = []
    for j, cases in enumerate(by_room):
        excess=(gp.quicksum((float(d[i])+TURNOVER)*x[i,j] for i in cases)
                -TURNOVER*used[j]-CAPACITY)
        ub=max(0.,sum(float(d[i])+TURNOVER for i in cases)
               -TURNOVER-CAPACITY)
        ot = m.addVar(lb=0., ub=ub, name=f"{name}_overtime_{j}")
        m.addConstr(ot >= excess)
        if exact and ub > 0:
            high = m.addVar(vtype=GRB.BINARY, name=f"{name}_high_{j}")
            m.addConstr(ot <= ub * high)
            m.addConstr(ot <= excess + CAPACITY * (1 - high))
        overtime.append(ot)
    return (CAPACITY*len(by_room)-float(d.sum())-TURNOVER*len(d)
            +TURNOVER*used.sum()
            +(OVERTIME+1.)*gp.quicksum(overtime))


def _assignment(week, x):
    return [next(j for j in week.allowed[int(week.surgeon[i])] if x[i, j].X > .5)
            for i in range(week.n)]


def _feasible_duration(week, duration):
    d = np.asarray(duration, float)
    if d.shape != (week.n,) or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("Duration array must contain one positive finite entry per case")
    return d


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

        m.addConstr(primary <= float(m.ObjVal) + _objective_tolerance(m.ObjVal))
        if balance:
            import gurobipy as gp
            peak = m.addVar(lb=0., name="max_predicted_room_load")
            for j, cases in enumerate(by_room):
                total_load = (
                    gp.quicksum((float(d[i])+TURNOVER)*x[i,j] for i in cases)
                    -TURNOVER*used[j]
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
        return Result(a, checked, bound, canonical_complete, str(m.Status),
                      elapsed, True, canonical_complete,
                      primary_elapsed, elapsed-primary_elapsed)
    finally:
        m.dispose()


def gp_sum_moves(x, week):
    import gurobipy as gp
    return gp.quicksum(1 - x[i, int(week.original[i])] for i in range(week.n))


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
