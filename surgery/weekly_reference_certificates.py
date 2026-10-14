"""Reference-only exact scheduling certificates.

Not a replacement for the frozen weekly planner: this module is deliberately
outside the feature/source fingerprints. Its only task is to speed rescue of
incomplete reference solves while reproducing the exact same planner outputs.
All original cohort features, eligibility, objectives and tie rules remain frozen.
"""
import time
import numpy as np

from .weekly_planner import (
    CAPACITY, TURNOVER, OVERTIME, _feasible_duration, _objective_tolerance,
    evaluate, solve_week,
)

def conservation_certificate(week, duration):
    """Exact global optimum when all physical room-days are mandatory and
    the historical assignment realizes the aggregate-workload lower bound.

    If every slot is forced open by a required surgeon weekday with only one
    eligible physical room on that weekday, the number of opened slots k and
    total room load W = sum(d) + TURNOVER*(n-k) are constant across feasible
    assignments. Convexity of f(t)=max(-t,0)+OVERTIME*max(t,0) gives
        sum_j f(L_j-CAPACITY) >= f(W-k*CAPACITY).
    If the historical feasible assignment achieves this bound, it is
    primary-optimal. As the unique zero-move assignment, it also wins both
    secondary tie-breaking stages, with no solver calls.

    This is a sufficient certificate only. If it fails, continue with the
    unchanged exact MIP; NEVER infer that the original assignment is optimal.
    """
    d = _feasible_duration(week, duration)
    for j, (weekday, _) in enumerate(week.slots):
        forced = any(
            weekday in required
            and sum(week.slots[slot][0] == weekday for slot in week.allowed[s]) == 1
            and j in week.allowed[s]
            for s, required in enumerate(week.required_days)
        )
        if not forced:
            return None
    total_load = float(np.sum(d)) + TURNOVER * (week.n - week.k)
    diff = total_load - CAPACITY * week.k
    bound = max(-diff, 0.) + OVERTIME * max(diff, 0.)
    original_cost = float(evaluate(week,week.original,d)['cost'])
    tolerance = max(1e-7,1e-9*max(abs(bound),abs(original_cost)))
    if original_cost > bound + tolerance:
        return None
    if original_cost < bound - tolerance:
        raise AssertionError('Aggregate room-load conservation lower bound violated')
    return {'assignment':list(map(int,week.original)),'cost':original_cost,
            'lower_bound':max(0.,bound-tolerance),
            'certificate':'all_slots_forced_and_aggregate_load_bound',
            'mandatory_room_days':week.k,
            'total_room_load':total_load}


def original_wins_tie(week, duration, primary_value):
    """Return True only when zero moves is *provably* an eligible tie choice.

    Used after a certified primary MIP optimum. The unique zero-move
    assignment is the historical one; if it is within the actual objective
    tolerance imposed by this planner, its secondary movement objective is 0
    and the canonical tie stage is unnecessary. Use a tiny strict margin to
    avoid declaring a boundary-feasible schedule eligible because of rounding.
    """
    original_value=float(evaluate(week,week.original,duration)['cost'])
    tol=_objective_tolerance(primary_value)
    guard=1e-8*max(1.,abs(primary_value))
    return original_value <= primary_value+tol-guard


def solve_certified_reference(component, duration, *,
                              seconds=300., threads=1, hindsight=False):
    """Return the same complete deployed-reference schema or remain incomplete.

    Sufficient analytic cost certificate is tried first. Otherwise a single
    primary MILP is solved; if the original zero-move schedule is inside its
    certified tolerance, secondary/canonical optimization is unnecessary.
    The remaining difficult cases fall back to the unchanged true lexicographic
    planner, with no relaxation of solver proof requirements.
    """
    from .anchored_oracles import deployed

    start=time.monotonic()
    d=_feasible_duration(component,duration)
    cert=conservation_certificate(component,d)
    if cert is not None:
        z=cert['assignment']
        real=evaluate(component,z,component.actual)
        return dict(complete=True,status='ANALYTIC_CONSERVATION',
                    assignment=z,predicted_cost=cert['cost'],
                    realized_cost=real['cost'],
                    realized_overtime_minutes=real['overtime'],
                    realized_idle_minutes=real['idle'],
                    moved_cases=real['moved_cases'],
                    room_days_over_720=real['room_days_over_720'],
                    max_realized_room_load=real['max_room_load'],
                    primary_bound=cert['lower_bound'],
                    seconds=time.monotonic()-start)
    # The hindsight oracle requires only primary optimality; no tie stage.
    if hindsight:
        return deployed(component,d,seconds=seconds,threads=threads,
                        hindsight=True)
    primary=solve_week(component,d,seconds=seconds,threads=threads,tie=False)
    if not primary.complete:
        return dict(complete=False,status='PRIMARY_'+primary.status,
                    assignment=primary.assignment,
                    primary_bound=primary.bound,
                    seconds=time.monotonic()-start)
    if original_wins_tie(component,d,primary.value):
        z=list(map(int,component.original))
        planned=evaluate(component,z,d)
        real=evaluate(component,z,component.actual)
        return dict(complete=True,status='OPTIMAL_ORIGINAL_ZERO_MOVE',
                    assignment=z,predicted_cost=planned['cost'],
                    realized_cost=real['cost'],
                    realized_overtime_minutes=real['overtime'],
                    realized_idle_minutes=real['idle'],
                    moved_cases=real['moved_cases'],
                    room_days_over_720=real['room_days_over_720'],
                    max_realized_room_load=real['max_room_load'],
                    primary_bound=primary.bound,
                    seconds=time.monotonic()-start)
    remaining=max(.001,float(seconds)-(time.monotonic()-start))
    return deployed(component,d,seconds=remaining,threads=threads,
                    hindsight=False)
