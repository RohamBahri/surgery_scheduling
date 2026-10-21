"""Pooled, case-weighted out-of-fold reporting with calendar-Monday inference."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np


def summarize(rows, draws=20000, seed=20261021):
    if not rows: raise ValueError("No evaluations to pool")
    arms=sorted(set().union(*(r["methods"] for r in rows)))
    mondays=sorted(set(r["monday"] for r in rows))
    if len(mondays)<2: raise ValueError("At least two independent Mondays required")
    n=sum(int(r["cases"]) for r in rows)
    out=dict(n_cases=n, n_components=len(rows),n_mondays=len(mondays),
             arms={},paired_vs_CASE={},by_site={},by_size={})
    for arm in arms:
        completed=[r for r in rows if r["methods"].get(arm,{}).get("status")=="COMPLETE"]
        full=len(completed)==len(rows)
        val=(sum(np.mean(list(r["methods"][arm]["realized"].values()))
                 for r in completed)/n if full else None)
        out["arms"][arm]=dict(completed_components=len(completed),
             completed_cases=sum(r["cases"] for r in completed),
             unresolved=len(rows)-len(completed),
             mean_realized_cost_per_case=None if val is None else float(val),
             evaluation_wall_hours=sum(r["methods"][arm].get("evaluation_seconds",0.)
                      for r in completed)/3600.)
    if all(r.get("hindsight",{}).get("complete") for r in rows):
        oracle=sum(r["hindsight"]["cost"] for r in rows)/n
        out["hindsight_per_case"]=oracle
        for value in out["arms"].values():
            if value["mean_realized_cost_per_case"] is not None:
                value["regret_per_case"]=value["mean_realized_cost_per_case"]-oracle
    # The stochastic reference is a nondeployable comparison, NOT an advice
    # policy. Keep solver primal/dual bounds separately and withhold point
    # claims when even one reference schedule lacks certification.
    reference=[r.get("saa_reference") for r in rows]
    if any(x is not None for x in reference):
        if not all(x is not None for x in reference):
            raise AssertionError("Inconsistent SAA-reference coverage")
        result=dict(total_components=len(rows),
             certified_components=sum(bool(x.get("certified")) for x in reference),
             feasible_components=sum(x.get("upper") is not None for x in reference),
             bounded_components=sum(x.get("lower") is not None for x in reference),
             solver_hours=sum(float(x.get("seconds",0)) for x in reference)/3600.,
             nondeployable=True)
        if all(x.get("lower") is not None for x in reference):
            result["model_A_lower_per_case"]=sum(x["lower"] for x in reference)/n
        if all(x.get("upper") is not None for x in reference):
            result["model_A_upper_per_case"]=sum(x["upper"] for x in reference)/n
        if all(x.get("assignment") is not None for x in reference):
            result["realized_per_case"]=sum(x["realized"] for x in reference)/n
            for label in ("A","C"):
                result["model_"+label+"_feasible_per_case"]=(
                    sum(x["model"][label] for x in reference)/n)
        if "model_A_lower_per_case" in result and "model_A_upper_per_case" in result:
            result["scenario_A_solver_gap_per_case"]=(
                result["model_A_upper_per_case"]-result["model_A_lower_per_case"])
        out["saa_reference"]=result
        # BOOKED/CASE versus the unrestricted SAA optimum is an opportunity
        # interval, not the stochastic-vs-perfect-information gap.
        if ("model_A_lower_per_case" in result
                and "model_A_upper_per_case" in result
                and not out["arms"].get("CASE",{}).get("unresolved",1)):
            case_model=sum(r["methods"]["CASE"]["model"]["A"] for r in rows)/n
            result["case_vs_saa_opportunity"]=dict(
                model_A_CASE_per_case=case_model,
                lower=case_model-result["model_A_upper_per_case"],
                upper=case_model-result["model_A_lower_per_case"],
                interpretation="Estimated-booking improvement opportunity, "
                  "NOT the value of perfect information.")
        out["inducibility_bounds"]={}
        if "model_A_lower_per_case" in result:
            lower=result["model_A_lower_per_case"]
            for arm in ("PER_WEEK_ROBUST","PW_BOUNDED","PW_ANCHORED","PW_VFCG"):
                if out["arms"].get(arm,{}).get("unresolved",1):continue
                feasible=sum(r["methods"][arm]["model"]["A"] for r in rows)/n
                if feasible < lower-1e-4:
                    raise AssertionError("Deployed scenario cost below SAA bound")
                out["inducibility_bounds"][arm]=dict(
                    deployed_scenario_A_cost_per_case=feasible,
                    gap_lower=0.,gap_upper=max(0.,feasible-lower),
                    statement="Conservative bound on unrestricted advice inducibility "
                       "under three hash rules, not a point estimate or a "
                       "certificate of this method's outer global optimum.")
    size=lambda k: "1" if k==1 else "2-5" if k<=5 else "6-10" if k<=10 else "11-20" if k<=20 else "21-34" if k<=34 else "35+"
    for label,partition in (("by_site",lambda r:r["site"]),("by_size",lambda r:size(r["cases"]))):
        for name in sorted({partition(r) for r in rows}):
            group=[r for r in rows if partition(r)==name]
            denom=sum(r["cases"] for r in group)
            detail=dict(n_cases=denom,n_components=len(group),arms={})
            for arm in arms:
                subset=[r for r in group if r["methods"].get(arm,{}).get("status")=="COMPLETE"]
                avg=(sum(np.mean(list(r["methods"][arm]["realized"].values()))
                    for r in subset)/denom if len(subset)==len(group) else None)
                detail["arms"][arm]=dict(complete=len(subset),unresolved=len(group)-len(subset),
                        cost_per_case=None if avg is None else float(avg))
            out[label][name]=detail
    if out["arms"].get("CASE",{}).get("unresolved",1):
        out["inference_note"]="No inferential contrasts: CASE was incomplete"
        return out
    totals=np.asarray([sum(r["cases"] for r in rows if r["monday"]==m)
                        for m in mondays],float)
    by_week=defaultdict(lambda:defaultdict(float))
    for r in rows:
        for a,rec in r["methods"].items():
            if rec.get("status")=="COMPLETE":
                by_week[r["monday"]][a]+=float(np.mean(list(rec["realized"].values())))
    rng=np.random.default_rng(seed)
    boot=rng.integers(len(mondays),size=(draws,len(mondays)))
    den=totals[boot].sum(axis=1)
    for arm in arms:
        if arm=="CASE" or out["arms"][arm]["unresolved"]: continue
        delta=np.asarray([by_week[m][arm]-by_week[m]["CASE"] for m in mondays])
        est=float(delta.sum()/n)
        samples=delta[boot].sum(axis=1)/den
        p=float((1+np.count_nonzero(np.abs(samples-est)>=abs(est)))/(draws+1))
        out["paired_vs_CASE"][arm]=dict(difference=est,
               ci95=np.quantile(samples,[.025,.975]).tolist(),
               p_centered_bootstrap=p)
    primary=sorted((rec["p_centered_bootstrap"],arm)
                   for arm,rec in out["paired_vs_CASE"].items()
                   if arm in ("SHARED","PER_WEEK"))
    largest=0.
    for rank,(p,arm) in enumerate(primary):
        largest=max(largest,(len(primary)-rank)*p)
        out["paired_vs_CASE"][arm]["p_holm_primary"]=float(min(1.,largest))
    out["inference_note"]=(
        "Case-weighted paired calendar-Monday bootstrap; hash seeds averaged "
        "within the component. Centered-bootstrap p-values are approximate. "
        "Only SHARED and PER_WEEK versus CASE are predeclared primary contrasts "
        "(Holm family of two); other comparisons are exploratory.")
    return out


def pooled_development(root, expected_cases=24506):
    root=Path(root)
    rows=[]
    for fold in (0,1):
        folder=root/f"fold_{fold}"/"full_evaluation"
        paths=sorted(folder.glob("*.json"))
        if not paths: raise RuntimeError(f"Missing evaluation fold {fold}")
        for p in paths:
            row=json.loads(p.read_text());row["fold"]=fold;rows.append(row)
    if sum(row["cases"] for row in rows)!=expected_cases:
        raise RuntimeError("Pooled report withheld: full development cohort missing")
    result=summarize(rows)
    target=root/"pooled_theory_summary.json"
    tmp=target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    tmp.replace(target)
    return result
