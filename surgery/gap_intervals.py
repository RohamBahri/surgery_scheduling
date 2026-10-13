"""Safe primal-dual intervals and exact-component reuse for VF-GAP.

For minimization P: lower=ObjBound, upper=feasible ObjVal.
For maximization A: lower=feasible ObjVal, upper=ObjBound.
The gamma*P+A interval is valid even if either MIP reaches TIME_LIMIT.
This module never equates an incumbent with an exact optimum.
"""
import hashlib
import math

import numpy as np

from .weekly_planner import evaluate, solve_week, solve_adversary


def duration_digest(duration):
    a=np.ascontiguousarray(duration,dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _finite(x):
    return x is not None and math.isfinite(float(x))


def _assignment_physical(week, slots):
    return [[int(i), int(week.slots[int(j)][0]), str(week.slots[int(j)][1])]
            for i,j in zip(week.case_ids,slots)]


def _oracle(comp, d, gamma, kind, seconds, threads):
    if kind == "planner":
        res=solve_week(comp,d,seconds=seconds,threads=threads,tie=False)
    else:
        res=solve_adversary(comp,d,gamma=gamma,seconds=seconds,threads=threads)
    row={"status":str(res.status),"seconds":float(res.seconds),
         "complete":bool(res.complete),"bounded":False}
    if res.assignment is None or not _finite(res.value) or not _finite(res.bound):
        row["failure"]="No feasible incumbent or finite solver bound"
        return row
    physical=_assignment_physical(comp,res.assignment)
    predicted=evaluate(comp,res.assignment,d)["cost"]
    actual=evaluate(comp,res.assignment,comp.actual)["cost"]
    recomputed=predicted if kind=="planner" else actual-gamma*predicted
    if abs(recomputed-float(res.value))>0.1:
        raise AssertionError(f"{kind} model and evaluated cost disagree")
    bound=float(res.bound)
    tol=max(.1,2e-6*max(1,abs(recomputed),abs(bound)))
    if kind=="planner":
        if bound>recomputed+tol:
            raise AssertionError("Planner lower bound exceeds incumbent")
        lower=min(bound,recomputed)
        upper=recomputed
    else:
        if bound<recomputed-tol:
            raise AssertionError("Adversarial upper bound below incumbent")
        lower=recomputed
        upper=max(bound,recomputed)
    if res.complete:
        lower=upper=recomputed
    row.update({"bounded":True,"lower":float(lower),"upper":float(upper),
                "assignment":physical,
                "predicted_cost":float(predicted),
                "realized_cost":float(actual)})
    return row


def _merge(old, new, kind):
    """Keep best certified dual bound and compatible best feasible incumbent."""
    if not old or not old.get("bounded"):
        return new
    if not new.get("bounded"):
        return old
    if old.get("complete"):
        return old
    if new.get("complete"):
        return new
    dual=max(old["lower"],new["lower"]) if kind=="planner" else min(old["upper"],new["upper"])
    if kind=="planner":
        incumbent=min((old,new),key=lambda x:x["upper"])
        upper=incumbent["upper"]
        lower=min(dual,upper)
    else:
        incumbent=max((old,new),key=lambda x:x["lower"])
        lower=incumbent["lower"]
        upper=max(dual,lower)
    result=dict(incumbent)
    result["lower"]=float(lower)
    result["upper"]=float(upper)
    result["complete"]=abs(upper-lower)<1e-7
    result["bounded"]=True
    result["seconds"]=old.get("seconds",0)+new.get("seconds",0)
    result["status"]="MERGED_CERTIFIED_BOUNDS"
    return result


def evaluate_component(comp,d,gamma,seconds,threads,previous=None,
                       refine=False):
    stamp=duration_digest(d)
    case_ids=list(map(int,comp.case_ids))
    old=previous if (previous and previous.get("duration_sha256")==stamp
                     and previous.get("case_ids")==case_ids) else {}
    p=old.get("planner")
    a=old.get("adversary")
    if not p or not p.get("bounded") or (refine and not p.get("complete")):
        p=_merge(p,_oracle(comp,d,gamma,"planner",seconds,threads),"planner")
    if not a or not a.get("bounded") or (refine and not a.get("complete")):
        a=_merge(a,_oracle(comp,d,gamma,"adversary",seconds,threads),"adversary")
    bounded=bool(p.get("bounded") and a.get("bounded"))
    complete=bool(p.get("complete") and a.get("complete"))
    output={"case_ids":case_ids,"duration_sha256":stamp,
            "planner":p,"adversary":a,"bounded":bounded,"complete":complete}
    if bounded:
        lower=gamma*p["lower"]+a["lower"]
        upper=gamma*p["upper"]+a["upper"]
        if lower>upper+.02:
            raise AssertionError("VF-GAP interval is inverted")
        if complete:
            lower=upper=gamma*p["upper"]+a["upper"]
        output.update({"lower":float(lower),"upper":float(upper),
                       "uncertainty":max(0.,float(upper-lower)),
                       "realized_planner_cost":float(p["realized_cost"])})
    return output


def interval_week(task):
    """Independent group-week worker. Reuse EXACT previous components."""
    week,duration,digest,gamma,seconds,threads,previous,refine_indices=task
    lookup=dict(zip(map(int,week.case_ids),map(float,duration)))
    old_components=(previous or {}).get("components",[])
    results=[]
    for j,comp in enumerate(week.components()):
        d=np.asarray([lookup[int(cid)] for cid in comp.case_ids],float)
        prior=old_components[j] if j<len(old_components) else None
        results.append(evaluate_component(
            comp,d,gamma,seconds,threads,previous=prior,
            refine=j in refine_indices))
    bounded=all(c["bounded"] for c in results)
    complete=all(c["complete"] for c in results)
    row={"key":week.group+"|"+week.monday,"input_sha256":digest,
         "bounded":bounded,"complete":complete,"components":results,
         "status":"OPTIMAL" if complete else "BOUND_INTERVAL"}
    if bounded:
        lo=sum(c["lower"] for c in results)
        hi=sum(c["upper"] for c in results)
        if lo>hi+.02:
            raise AssertionError("Week VF-GAP bound interval inverted")
        row.update({
            "gap_lower":float(lo),"gap_upper":float(hi),
            "gap_without_hindsight":float(hi) if complete else None,
            "predicted_plan_cost":float(sum(c["planner"]["predicted_cost"] for c in results)),
            "adversary_value":float(sum(c["adversary"]["lower"] for c in results)),
            "realized_plan_cost":float(sum(c["realized_planner_cost"] for c in results)),
            "planner_assignment":[x for c in results for x in c["planner"]["assignment"]],
            "adversary_assignment":[x for c in results for x in c["adversary"]["assignment"]]
        })
    else:
        row["failure"]=[
            {"component":i,
             "planner":c["planner"].get("failure",c["planner"].get("status")),
             "adversary":c["adversary"].get("failure",c["adversary"].get("status"))}
            for i,c in enumerate(results) if not c["bounded"]]
    return row


def gap_interval(row):
    """Backward compatible with the original fully exact week checkpoint."""
    if row.get("bounded") and _finite(row.get("gap_lower")) and _finite(row.get("gap_upper")):
        return float(row["gap_lower"]),float(row["gap_upper"])
    if row.get("complete") and _finite(row.get("gap_without_hindsight")):
        val=float(row["gap_without_hindsight"])
        return val,val
    return None


def worst_components(checkpoint,weeks,count=3):
    """Return (week_key, component_index, weighted interval width)."""
    choices=[]
    for week in weeks:
        key=week.group+"|"+week.monday
        row=checkpoint["weeks"][key]
        for j,component in enumerate(row.get("components",[])):
            if not component.get("complete"):
                width=component.get("uncertainty",float("inf"))
                choices.append((key,j,float(width)))
    choices.sort(key=lambda x:-x[2])
    return choices[:count]
