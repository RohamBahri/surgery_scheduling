"""Anchored-experiment exact weekly scheduling solver.

Single implementation for every scheduling decision made by the anchored
experiment: reference proposals, actual deployment, training acceptance,
regret certificates, and validation. The original weekly_planner is left
byte-for-byte unchanged to preserve frozen historical feature matrices.

The feasible integer schedules are unchanged, but deployment now uses a fixed
history-independent hash, not movement and room-index tie stages. Valid
strengthening adds x[i,j] <= used[j] and provably mandatory room-day openings.
Time-limited solves remain uncertified.
"""
import hashlib
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .weekly_planner import (
    CAPACITY, TURNOVER, OVERTIME, Result, _cost, _assignment,
    _feasible_duration, _objective_tolerance, _check_reconstruction,
    evaluate, loads,
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
    A cost proof does not select a unique deployed hash-optimal assignment.
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


@dataclass
class AnchoredResult(Result):
    """Audit trail for the exact deployed selection, not merely its cost."""
    primary_status: str = ""
    hash_status: str = ""
    uniqueness_status: str = ""
    hash_seed: int = 0
    hash_score: int | None = None
    hash_unique: bool = False
    primary_gurobi_seed: int | None = None
    primary_configuration: str = ""
    primary_attempts: tuple = ()
    hash_attempts: tuple = ()
    uniqueness_seconds: float = 0.
    primary_tolerance: float = 0.


def hash_weight(seed, case_id, weekday, room):
    """Stable integer priority; neither performed assignments nor outcomes enter."""
    message = f"{int(seed)}|{int(case_id)}|{int(weekday)}|{room}".encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % (1 << 20)


def hash_score(week, assignment, seed=0):
    return sum(hash_weight(seed, week.case_ids[i], *week.slots[int(j)])
               for i, j in enumerate(assignment))


def _hash_expression(week, x, seed):
    import gurobipy as gp
    return gp.quicksum(
        hash_weight(seed, week.case_ids[i], *week.slots[j]) * x[i, j]
        for i in range(week.n)
        for j in week.allowed[int(week.surgeon[i])]
    )


def solve_week(week, duration, *, seconds=120., threads=1, tie=True,
               hash_seed=None):
    """Certify primary cost, then a unique fixed-hash winner inside its tau set.

    Uncertified time-limited primary, hash, or uniqueness solves cannot be
    passed off as completed deployments. A bounded portfolio of Gurobi
    configurations is used for difficult primary MIPs, without altering the
    feasible set, duration data, or optimality requirements.
    """
    from gurobipy import GRB
    seed = int(os.getenv("SURGERY_ANCHORED_HASH_SEED", "0")
               if hash_seed is None else hash_seed)
    if seed < 0 or not math.isfinite(seconds) or seconds <= 0 or threads < 1:
        raise ValueError("Invalid hash seed, time budget, or thread count")
    d = _feasible_duration(week, duration)
    week.validate()
    start = time.monotonic()
    if all(len(week.allowed[int(s)]) == 1 for s in week.surgeon):
        z = list(map(int, week.original))
        val = evaluate(week, z, d)["cost"]
        return AnchoredResult(z, val, val, True, "ANALYTIC_UNIQUE", 0.,
                              True, True, 0., 0.,
                              primary_status="ANALYTIC_UNIQUE",
                              hash_status="ANALYTIC_UNIQUE",
                              uniqueness_status="UNIQUE_FEASIBLE",
                              hash_seed=seed, hash_score=hash_score(week,z,seed),
                              hash_unique=True)

    cert = conservation_certificate(week, d)
    m, x, used, by_room = _model(week, seconds, threads)
    attempts = []
    hash_attempts = []
    best_bound = None
    primary_z = None
    primary_cost = None
    primary_opt = None
    primary_status = hash_status = uniqueness_status = ""
    config_name = ""
    config_seed = None
    selected = None
    hash_best = None
    hash_unique = False
    uniqueness_elapsed = 0.
    primary_elapsed = 0.
    tol = 0.

    def outcome(status, complete=False, primary_ok=False, tie_ok=False):
        elapsed = time.monotonic() - start
        z = selected if selected is not None else primary_z
        checked = evaluate(week,z,d)["cost"] if z is not None else None
        return AnchoredResult(
            z, checked, best_bound, complete, status, elapsed,
            primary_ok, tie_ok, primary_elapsed, max(0., elapsed-primary_elapsed),
            primary_status=primary_status, hash_status=hash_status,
            uniqueness_status=uniqueness_status, hash_seed=seed,
            hash_score=hash_best, hash_unique=hash_unique,
            primary_gurobi_seed=config_seed, primary_configuration=config_name,
            primary_attempts=tuple(attempts),hash_attempts=tuple(hash_attempts),
            uniqueness_seconds=uniqueness_elapsed, primary_tolerance=tol)

    try:
        primary = _cost(m, x, used, by_room, d, exact=False, name="plan")
        m.setObjective(primary, GRB.MINIMIZE)
        if cert is not None:
            primary_opt = cert["value"]
            primary_z = list(map(int, cert["assignment"]))
            primary_cost = primary_opt
            best_bound = cert["bound"]
            primary_status, config_name = "ANALYTIC_CONSERVATION", "analytic"
        else:
            configs = (("default-0",0,0,-1,-1),
                       ("focused-11",11,3,2,2),
                       ("default-41",41,0,-1,-1),
                       ("focused-73",73,3,2,2))
            cap = float(seconds if not tie else .65*seconds)
            for idx,(name,gseed,focus,cuts,symmetry) in enumerate(configs):
                remaining = cap-(time.monotonic()-start)
                if remaining <= .005:
                    break
                budget = (min(45., remaining*.30) if idx == 0
                          else min(60., remaining*.45) if idx == 1
                          else remaining*.50 if idx == 2 else remaining)
                if idx:
                    m.reset()
                m.Params.Seed = gseed
                m.Params.MIPFocus = focus
                m.Params.Cuts = cuts
                m.Params.Symmetry = symmetry
                # Internal concurrent MIP shares the caller's thread allocation.
                m.Params.ConcurrentMIP = 2 if idx and threads >= 2 else 1
                m.Params.TimeLimit = max(.001, budget)
                m.optimize()
                attempts.append(dict(configuration=name, gurobi_seed=gseed,
                                     status=str(m.Status), seconds=float(m.Runtime)))
                if m.SolCount:
                    z = _assignment(week,x)
                    checked = _check_reconstruction(m.ObjVal,
                                   evaluate(week,z,d)["cost"])
                    if primary_cost is None or checked < primary_cost:
                        primary_z,primary_cost = z,checked
                if m.Status in (GRB.OPTIMAL,GRB.TIME_LIMIT,GRB.SUBOPTIMAL):
                    bound = float(m.ObjBound)
                    if math.isfinite(bound):
                        best_bound = bound if best_bound is None else max(best_bound,bound)
                if m.Status == GRB.OPTIMAL:
                    primary_opt = float(m.ObjVal)
                    primary_status = "OPTIMAL"
                    config_name, config_seed = name, gseed
                    break
            if primary_opt is None:
                primary_elapsed = time.monotonic()-start
                primary_status = attempts[-1]["status"] if attempts else "NO_TIME"
                return outcome("PRIMARY_UNRESOLVED")
        primary_elapsed = time.monotonic()-start
        if best_bound is not None:
            # A numerical dual bound must not exceed a reconstructed feasible
            # primal: this is the conservative input to the regret certificate.
            best_bound = min(float(best_bound),float(primary_cost))
        tol = _objective_tolerance(primary_opt)
        if not tie:
            selected = primary_z
            return outcome("OPTIMAL_PRIMARY", True, True, True)

        m.addConstr(primary <= primary_opt+tol, name="primary_tie_set")
        for ordinal in (0,1):
            priority = _hash_expression(week,x,seed+ordinal)
            m.setObjective(priority, GRB.MINIMIZE)
            m.Params.Cutoff = GRB.INFINITY
            m.Params.TimeLimit = max(.001, seconds-(time.monotonic()-start))
            m.optimize()
            hash_status = str(m.Status)
            hash_attempts.append(dict(stage=ordinal,hash_seed=seed+ordinal,
                                      status=hash_status, seconds=float(m.Runtime)))
            if m.Status != GRB.OPTIMAL or not m.SolCount:
                return outcome("HASH_UNRESOLVED",primary_ok=True)
            selected = _assignment(week,x)
            hash_best = hash_score(week,selected,seed+ordinal)
            if abs(m.ObjVal-hash_best) > .25:
                raise AssertionError("Reconstructed integer hash objective differs")
            # Check uniqueness WITHOUT fixing the hash to its incumbent score.
            # The objective and integer cutoff already prove whether another
            # schedule can attain this score. The extra constraint needlessly
            # makes the exclusion MIP harder on large components.
            winner = list(selected)
            exclude = m.addConstr(
                sum(x[i,int(winner[i])] for i in range(week.n)) <= week.n-1,
                name=f"exclude_hash_{ordinal}")
            m.setObjective(priority, GRB.MINIMIZE)
            m.Params.Cutoff = hash_best+.5
            m.Params.TimeLimit = max(.001, seconds-(time.monotonic()-start))
            t_unique = time.monotonic()
            m.optimize()
            uniqueness_elapsed += time.monotonic()-t_unique
            hash_attempts[-1]['uniqueness_status'] = str(m.Status)
            hash_attempts[-1]['uniqueness_seconds'] = time.monotonic()-t_unique
            if m.Status in (GRB.INFEASIBLE,GRB.CUTOFF):
                hash_unique = True
                uniqueness_status = "UNIQUE_CERTIFIED"
            elif m.Status == GRB.OPTIMAL and m.SolCount:
                if m.ObjVal > hash_best+.5:
                    hash_unique = True
                    uniqueness_status = "UNIQUE_CERTIFIED"
                else:
                    uniqueness_status = "HASH_COLLISION_CERTIFIED"
            else:
                uniqueness_status = "UNIQUE_UNVERIFIED_"+str(m.Status)
            m.remove(exclude)
            m.update()
            m.Params.Cutoff = GRB.INFINITY
            selected = winner
            if hash_unique:
                if evaluate(week,selected,d)["cost"] > primary_opt+tol+1e-6:
                    raise AssertionError("Deployed assignment exceeds primary tie tolerance")
                return outcome("OPTIMAL",True,True,True)
            if uniqueness_status.startswith("UNIQUE_UNVERIFIED"):
                return outcome("HASH_UNIQUENESS_UNRESOLVED",primary_ok=True)
            # A certified collision needs the preceding hash optimum fixed
            # before optimizing any lower-priority hash or canonical fallback.
            m.addConstr(priority <= hash_best, name=f"hash_tie_{ordinal}")
            m.update()

        # Two actual score collisions: deterministic exact lexicographic
        # fallback, restricted to the optima of both hash criteria.
        base = week.k+1
        chunk = max(1,int(math.log(1e6)/math.log(base)))
        for first in range(0,week.n,chunk):
            indices = list(range(first,min(week.n,first+chunk)))
            lex = sum(base**(len(indices)-pos-1)*j*x[i,j]
                      for pos,i in enumerate(indices)
                      for j in week.allowed[int(week.surgeon[i])])
            m.setObjective(lex,GRB.MINIMIZE)
            m.Params.TimeLimit = max(.001,seconds-(time.monotonic()-start))
            m.optimize()
            if m.Status != GRB.OPTIMAL or not m.SolCount:
                uniqueness_status = "LEX_FALLBACK_UNRESOLVED"
                return outcome("HASH_COLLISION_FALLBACK_UNRESOLVED",primary_ok=True)
            selected = _assignment(week,x)
            for i in indices:
                m.addConstr(x[i,int(selected[i])] == 1)
        hash_unique = True
        uniqueness_status = "LEX_FALLBACK_CERTIFIED"
        if evaluate(week,selected,d)["cost"] > primary_opt+tol+1e-6:
            raise AssertionError("Lex fallback exceeds primary tie tolerance")
        return outcome("OPTIMAL",True,True,True)
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
            "room_days_over_720", "room_days_over_960",
            "room_days_over_1440", "changed_days")}
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
            realized_loads = loads(c,planned.assignment,c.actual)
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
                "room_days_over_960": int(np.count_nonzero(realized_loads>960.)),
                "room_days_over_1440": int(np.count_nonzero(realized_loads>1440.)),
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


