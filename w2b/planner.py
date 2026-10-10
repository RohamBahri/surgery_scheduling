"""Exact fixed-roster W2b follower and loss-augmented scheduling oracle (Gurobi).

Primary cost: Idle + 1.75*Overtime across *all* staffed room sessions,
plus optional case-date movement cost. No room-release decision exists.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from surgery.planner import response
from w2b.data import Week


@dataclass
class Solve:
    assignment: list | None
    cost: float | None
    lower: float | None
    upper: float | None
    optimal: bool
    status: int
    seconds: float

    def as_dict(self):
        return vars(self)


def _reconcile_objective(solver_value, reconstructed, label):
    """Reject material MILP/accounting mismatches; allow numerical MILP tolerances.

    Always report the independently reconstructed feasible objective. Gurobi's
    floating-point incumbent can differ slightly after binary extraction.
    This accounting tolerance is not an optimality-gap allowance.
    """
    scale = max(1., abs(float(solver_value)), abs(float(reconstructed)))
    tolerance = min(.05, max(.001, 2e-6 * scale))
    delta = abs(float(solver_value) - float(reconstructed))
    if not np.isfinite(delta) or delta > tolerance:
        raise AssertionError(
            f"{label}: solver objective {solver_value} vs reconstructed "
            f"{reconstructed} (delta={delta:.9g}, tolerance={tolerance:.9g})")
    return float(reconstructed)


def implement(booked, X, w, alpha, h):
    raw = np.asarray(X @ np.asarray(w, float), float)
    u = np.clip(raw, np.maximum(-180., 1. - booked), 180.)
    return booked + response(u, alpha, h)


def schedule_cost(week: Week, assignment, durations, move_penalty=0.):
    """Independent arithmetic check: vacant staffed rooms incur 480 Idle minutes."""
    slots = np.asarray(assignment, int)
    d = np.asarray(durations, float)
    count = np.bincount(slots, minlength=len(week.slots))
    load = np.bincount(slots, weights=d + 30, minlength=len(week.slots))
    load -= 30 * (count > 0)
    moved = sum(week.slots[j][0] != int(week.original_day[i])
                for i, j in enumerate(slots))
    return float(np.maximum(480 - load, 0).sum()
                 + 1.75 * np.maximum(load - 480, 0).sum()
                 + move_penalty * moved)


def _make_model(week, *, seconds, threads):
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("w2b_" + week.group + "_" + week.monday)
    m.Params.OutputFlag = 0
    m.Params.Threads = threads
    m.Params.TimeLimit = seconds
    m.Params.Seed = 0
    m.Params.MIPGap = 0
    m.Params.MIPGapAbs = 1e-7
    m.Params.FeasibilityTol = 1e-9
    m.Params.IntFeasTol = 1e-9
    m.Params.OptimalityTol = 1e-9

    allowed = [[] for _ in range(week.n)]
    for i, j in week.arcs:
        allowed[i].append(j)
    x = m.addVars(week.arcs, vtype=GRB.BINARY, name="case_slot")
    for i in range(week.n):
        m.addConstr(gp.quicksum(x[i, j] for j in allowed[i]) == 1)

    sessions = sorted({(int(week.surgeon[i]), j) for i, j in week.arcs})
    y = m.addVars(sessions, vtype=GRB.BINARY, name="surgeon_slot")
    surgeon_members = [[] for _ in range(week.n_surgeons)]
    by_session = {key: [] for key in sessions}
    slot_cases = [[] for _ in week.slots]
    for i, j in week.arcs:
        s = int(week.surgeon[i])
        surgeon_members[s].append((i, j))
        by_session[s, j].append(i)
        slot_cases[j].append(i)
        m.addConstr(x[i, j] <= y[s, j])
    for s, j in sessions:
        m.addConstr(y[s, j] <= gp.quicksum(x[i, j] for i in by_session[s, j]))
    for s in range(week.n_surgeons):
        eligible = [j for ss, j in sessions if ss == s]
        m.addConstr(gp.quicksum(y[s, j] for j in eligible)
                    == int(week.days_required[s]))
        for d in week.allowed_days[s]:
            jd = [j for j in eligible if week.slots[j][0] == d]
            if jd:
                m.addConstr(gp.quicksum(y[s, j] for j in jd) <= 1)
    occupied = m.addVars(len(week.slots), vtype=GRB.BINARY, name="occupied")
    for j, cases in enumerate(slot_cases):
        count = gp.quicksum(x[i, j] for i in cases)
        m.addConstr(count <= max(1, len(cases)) * occupied[j])
        m.addConstr(occupied[j] <= count)
    shift = gp.quicksum(x[i, j] for i, j in week.arcs
                        if week.slots[j][0] != int(week.original_day[i]))
    if week.move_budget is not None:
        m.addConstr(shift <= week.move_budget, name="case_date_move_budget")
    return m, x, occupied, slot_cases, shift


def _cost_expr(m, x, occupied, slots_cases, d, shift,
               move_penalty, exact_over, prefix):
    """Return full cost. exact_over is necessary for positive max coefficients."""
    import gurobipy as gp
    from gurobipy import GRB
    overtime = []
    for j, cases in enumerate(slots_cases):
        load = gp.quicksum((float(d[i]) + 30) * x[i, j] for i in cases)
        load -= 30 * occupied[j]
        excess = load - 480
        ub = max(0., sum(float(d[i]) + 30 for i in set(cases)) - 480)
        ot = m.addVar(lb=0, ub=max(0., ub), name=f"{prefix}_ot_{j}")
        m.addConstr(ot >= excess)
        if exact_over:
            high = m.addVar(vtype=GRB.BINARY, name=f"{prefix}_high_{j}")
            m.addConstr(ot <= ub * high)
            # load >= 0, so excess >= -480. This is a tight valid M.
            m.addConstr(ot <= excess + 480 * (1 - high))
        overtime.append(ot)
    return (480 * len(slots_cases) - float(np.sum(d)) - 30 * len(d)
            + 30 * occupied.sum() + 2.75 * gp.quicksum(overtime)
            + float(move_penalty) * shift)


def solve_week(week, durations, *, seconds=120., threads=1,
               move_penalty=0., deterministic_tie=True):
    from gurobipy import GRB
    d = np.asarray(durations, float)
    if len(d) != week.n or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("Invalid planning durations")
    m, x, occupied, cases, shift = _make_model(
        week, seconds=seconds, threads=threads)
    cost = _cost_expr(m, x, occupied, cases, d, shift,
                      move_penalty, False, "planned")
    m.setObjective(cost, GRB.MINIMIZE)
    m.optimize()
    status = int(m.Status)
    elapsed = float(m.Runtime)
    bestbound = float(m.ObjBound) if m.IsMIP and status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else None
    incumbent = float(m.ObjVal) if m.SolCount else None
    assignment = ([next(j for j in range(len(week.slots)) if (i, j) in x
                        and x[i, j].X > .5) for i in range(week.n)]
                  if m.SolCount else None)
    optimal = status == GRB.OPTIMAL
    # Deterministic outcome-blind tie: fewest moved cases, then slot indices.
    # Never use realized durations here.
    if optimal and deterministic_tie:
        m.addConstr(cost <= incumbent + 1e-6)
        index = sum((len(week.slots) * week.n + 1) *
                    int(week.slots[j][0] != week.original_day[i]) * x[i, j]
                    + j * x[i, j] for i, j in week.arcs)
        m.setObjective(index, GRB.MINIMIZE)
        m.Params.TimeLimit = seconds
        m.optimize()
        elapsed += float(m.Runtime)
        optimal = m.Status == GRB.OPTIMAL
        if m.SolCount:
            assignment = [next(j for j in range(len(week.slots))
                               if (i, j) in x and x[i, j].X > .5)
                          for i in range(week.n)]
        status = int(m.Status)
    if assignment is not None:
        checked = schedule_cost(week, assignment, d, move_penalty)
        incumbent = _reconcile_objective(incumbent, checked, "Planner")
        # Primary solver dual bounds are retained, but the verified assignment
        # is the feasible upper bound. Never trust rounded auxiliary variables
        # more than an independent cost recomputation.
        bestbound = min(bestbound, incumbent) if bestbound is not None else None
    m.dispose()
    return Solve(assignment, incumbent, bestbound, incumbent, optimal, status,
                 elapsed)


def solve_adversary(week, predicted, *, gamma=2., seconds=120.,
                    threads=1, move_penalty=0.):
    """Maximize J(z;actual)-gamma J(z;predicted) over ALL W2b schedules."""
    from gurobipy import GRB
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    m, x, occupied, cases, shift = _make_model(
        week, seconds=seconds, threads=threads)
    actual = _cost_expr(m, x, occupied, cases, week.actual, shift,
                        move_penalty, True, "actual")
    plan = _cost_expr(m, x, occupied, cases, predicted, shift,
                      move_penalty, False, "predicted")
    m.setObjective(actual - gamma * plan, GRB.MAXIMIZE)
    m.optimize()
    status = int(m.Status)
    elapsed = float(m.Runtime)
    value = float(m.ObjVal) if m.SolCount else None
    bound = float(m.ObjBound) if m.SolCount or status == GRB.TIME_LIMIT else None
    assignment = None
    if m.SolCount:
        assignment = [next(j for j in range(len(week.slots))
                           if (i, j) in x and x[i, j].X > .5)
                      for i in range(week.n)]
        exact = (schedule_cost(week, assignment, week.actual, move_penalty)
                 - gamma * schedule_cost(week, assignment, predicted, move_penalty))
        value = _reconcile_objective(value, exact, "Loss-augmented oracle")
        # For MAX, independently recomputed feasible cost is a lower bound.
        bound = max(bound, value) if bound is not None else None
    m.dispose()
    return Solve(assignment, value, value, bound, status == GRB.OPTIMAL,
                 status, elapsed)


def plan_hash(week, mode, duration, gamma, move_penalty, tie):
    payload = [week.group, week.monday, week.cases.tolist(), week.slots,
               week.arcs, week.surgeon.tolist(), week.original_day.tolist(),
               week.days_required.tolist(), week.allowed_days, week.move_budget,
               (week.actual.round(8).tolist() if mode == "adversary" else None),
               mode, np.asarray(duration).round(8).tolist(),
               gamma, move_penalty, tie]
    return sha256(json.dumps(payload).encode()).hexdigest()[:24]


def cached_solve(cache_dir, week, duration, *, adversary=False, gamma=2.,
                 seconds=120., threads=1, move_penalty=0., tie=True):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = plan_hash(week, "adversary" if adversary else "planner",
                    duration, gamma, move_penalty, tie)
    file = cache / (key + ".json")
    if file.exists():
        saved = json.loads(file.read_text())
        if saved["optimal"]:
            return Solve(**saved)
    if adversary:
        result = solve_adversary(week, duration, gamma=gamma, seconds=seconds,
                                 threads=threads, move_penalty=move_penalty)
    else:
        result = solve_week(week, duration, seconds=seconds, threads=threads,
                            move_penalty=move_penalty, deterministic_tie=tie)
    file.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return result
