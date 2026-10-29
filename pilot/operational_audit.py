"""Operational audit for configured-turnover room-day assignment.

Separates aggregate capacity shortfall, necessary eligibility bottlenecks,
and avoidable peak-load concentration. The minimum-peak comparison is a
separate optimization diagnostic, not part of the deployed cost objective.
All claims are qualified by primal/dual bounds and solver status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from surgery.weekly_planner import CAPACITY, loads


def necessary_peak_bounds(week, duration):
    """Valid peak lower bounds from aggregate capacity and eligible subsets."""
    d=np.asarray(duration,float)
    if len(d)!=week.n or np.any(d<=0) or not np.isfinite(d).all():
        raise ValueError("Invalid durations for operational audit")
    from surgery.weekly_planner import TURNOVER
    aggregate=float((d.sum()+TURNOVER*max(0,week.n-week.k))/week.k)
    eligible=[frozenset(int(j) for j in arcs) for arcs in week.allowed]
    candidates=set(eligible)
    candidates.add(frozenset(range(week.k)))
    by_surgeon=np.bincount(week.surgeon,weights=d,
                          minlength=len(eligible))
    case_counts=np.bincount(week.surgeon,minlength=len(eligible))
    bottlenecks=[]
    for slots in candidates:
        if not slots:
            continue
        forced_surgeons=[s for s,arcs in enumerate(eligible)
                         if arcs.issubset(slots)]
        forced=sum(float(by_surgeon[s]) for s in forced_surgeons)
        forced_cases=sum(int(case_counts[s]) for s in forced_surgeons)
        lower_workload=forced+TURNOVER*max(0,forced_cases-len(slots))
        bottlenecks.append(
            (lower_workload/len(slots),len(slots),lower_workload))
    pressure=max((v[0] for v in bottlenecks),default=aggregate)
    return dict(aggregate_peak_lower_bound=aggregate,
                eligibility_peak_lower_bound=float(pressure),
                necessary_peak_lower_bound=float(max(aggregate,pressure)),
                aggregate_overtime_minutes_lower_bound=
                    float(max(0.,d.sum()+
                        TURNOVER*max(0,week.n-week.k)-CAPACITY*week.k)),
                eligibility_excess_peak_lower_bound=
                    float(max(0.,pressure-CAPACITY)))


def solve_minimum_peak(week,duration,*,seconds=60.,threads=1):
    """Proven bounds for min maximum room load over the FULL feasible roster.

    In particular, no primary-cost restriction is imposed: this asks how much
    peak workload is unavoidable due to eligibility and available capacity.
    """
    from gurobipy import GRB
    import gurobipy as gp
    from surgery.weekly_planner import _model, _assignment, TURNOVER
    d=np.asarray(duration,float)
    week.validate()
    original_peak=float(loads(week,week.original,d).max())
    necessary=necessary_peak_bounds(week,d)["necessary_peak_lower_bound"]
    m,x,used,by_room=_model(week,seconds,threads)
    try:
        peak=m.addVar(lb=0.,name="peak_minutes")
        for j,cases in enumerate(by_room):
            m.addConstr(peak>=(
                gp.quicksum((float(d[i])+TURNOVER)*x[i,j] for i in cases)
                -TURNOVER*used[j]))
        m.setObjective(peak,GRB.MINIMIZE)
        m.optimize()
        best=original_peak
        lb=necessary
        if m.SolCount:
            z=_assignment(week,x)
            best=min(best,float(loads(week,z,d).max()))
        if m.Status in (GRB.OPTIMAL,GRB.TIME_LIMIT,GRB.SUBOPTIMAL):
            if np.isfinite(m.ObjBound):
                lb=max(lb,float(m.ObjBound))
        lb=min(lb,best)
        return dict(min_peak_lb=float(lb),min_peak_ub=float(best),
                    min_peak_proven=bool(m.Status==GRB.OPTIMAL),
                    solver_status=int(m.Status),solver_seconds=float(m.Runtime),
                    original_peak=float(original_peak))
    finally:
        m.dispose()


def audit_policy(items,durations,plans,folder,*,seconds=60.,threshold=720.):
    """Log all groups; run the exact minimax-load audit on extreme groups.

    A peak is 'confirmed avoidable' only when a feasible counterfactual with
    strictly smaller peak is available. A peak is 'proven unavoidable'
    only when a valid lower bound equals the chosen peak to tolerance.
    """
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for key,week,rows in items:
        d=np.asarray(durations[rows],float)
        record=plans[key]
        z=record["assignment"]
        p=loads(week,z,d)
        actual=loads(week,z,week.actual)
        chosen=float(p.max())
        actual_peak=float(actual.max())
        bound=necessary_peak_bounds(week,d)
        out=dict(instance=str(key),cases=int(week.n),room_days=int(week.k),
                 planner_complete=bool(record["complete"]),
                 planner_status=str(record["status"]),
                 predicted_peak=chosen,realized_peak=actual_peak,
                 predicted_overtime=float(np.maximum(p-CAPACITY,0.).sum()),
                 realized_overtime=float(np.maximum(actual-CAPACITY,0.).sum()),
                 empty_roster_rooms=int(np.sum(p==0.)),
                 over_720_room_days=int(np.sum(p>720.)),
                 **bound)
        if chosen>threshold or actual_peak>threshold:
            from surgery.weekly_planner import TURNOVER
            payload=repr((key,seconds,d.tolist(),float(TURNOVER),
                          "minpeak_turnover_v2")).encode()
            cache=folder/(hashlib.sha256(payload).hexdigest()+".json")
            if cache.exists():
                result=json.loads(cache.read_text())
            else:
                result=solve_minimum_peak(week,d,seconds=seconds)
                temp=cache.with_suffix(".tmp")
                temp.write_text(json.dumps(result,indent=2))
                temp.replace(cache)
            out.update(result)
            tol=1e-4
            out["avoidable_peak_minutes_at_least"]=max(
                0.,chosen-result["min_peak_ub"])
            out["avoidable_peak_minutes_at_most"]=max(
                0.,chosen-result["min_peak_lb"])
            out["confirmed_avoidable_concentration"]=bool(
                chosen>result["min_peak_ub"]+tol)
            out["unavoidable_at_chosen_peak_proven"]=bool(
                chosen<=result["min_peak_lb"]+tol)
            out["forced_over_capacity_proven"]=bool(
                result["min_peak_lb"]>CAPACITY+tol)
            out["unresolved_minpeak_gap"]=float(
                result["min_peak_ub"]-result["min_peak_lb"])
        # A separate ex-post diagnostic tests whether the REALIZED concentration
        # could be reduced with hindsight. It is never used in learning or
        # deployment and cannot be interpreted as a prospective improvement.
        if actual_peak>threshold:
            from surgery.weekly_planner import TURNOVER
            tag=repr((key,seconds,week.actual.tolist(),float(TURNOVER),
                      "minpeak_actual_turnover_v2")).encode()
            cache=folder/(hashlib.sha256(tag).hexdigest()+".json")
            if cache.exists():
                hindsight=json.loads(cache.read_text())
            else:
                hindsight=solve_minimum_peak(week,week.actual,seconds=seconds)
                temp=cache.with_suffix(".tmp")
                temp.write_text(json.dumps(hindsight,indent=2))
                temp.replace(cache)
            out["hindsight_min_peak_lb"]=hindsight["min_peak_lb"]
            out["hindsight_min_peak_ub"]=hindsight["min_peak_ub"]
            out["hindsight_min_peak_proven"]=hindsight["min_peak_proven"]
            out["hindsight_min_peak_solver_status"]=hindsight["solver_status"]
            out["hindsight_confirmed_avoidable_peak_minutes_at_least"]=float(
                max(0.,actual_peak-hindsight["min_peak_ub"]))
            out["hindsight_forced_peak_over_capacity_proven"]=bool(
                hindsight["min_peak_lb"]>CAPACITY+1e-4)
        entries.append(out)
    flagged=[e for e in entries if "min_peak_lb" in e]
    from surgery.weekly_planner import TURNOVER
    output=dict(model="fixed_roster",turnover_minutes=float(TURNOVER),
                flag_threshold_minutes=float(threshold),
                total_instances=len(entries),
                flagged_instances=len(flagged),
                confirmed_avoidable=sum(e["confirmed_avoidable_concentration"] for e in flagged),
                peak_unavoidable_proven=sum(e["unavoidable_at_chosen_peak_proven"] for e in flagged),
                forced_over_capacity_proven=sum(e["forced_over_capacity_proven"] for e in flagged),
                minpeak_unresolved=sum(not e["min_peak_proven"] for e in flagged),
                hindsight_flagged_instances=sum("hindsight_min_peak_lb" in e for e in entries),
                hindsight_confirmed_avoidable=sum(
                    e.get("hindsight_confirmed_avoidable_peak_minutes_at_least",0.)>1e-4
                    for e in entries),
                aggregate_capacity_shortfall_instances=sum(
                   e["aggregate_overtime_minutes_lower_bound"]>0 for e in entries),
                entries=entries)
    return output
