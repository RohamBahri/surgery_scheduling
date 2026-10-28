#!/usr/bin/env python3
"""Deployment-only credibility-cliff stress test using frozen response-grid advice.

This NEVER retrains a policy and NEVER re-canonicalizes its displayed advice.
It extends the already completed response-grid 2013 evaluation with six
nonmonotone true-response scenarios. Summaries use one intersected certified
case population for comparisons between saturated and cliff responses.
"""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from pilot.run_nonmonotone_learning import save
from pilot.run_response_scenario_grid import ContentCache
from surgery.response_grid import (
    Scenario,TRAIN_SCENARIOS,EVAL_SCENARIOS,implemented_from_display)
from surgery.weekly_planner import evaluate

POLICIES=tuple("VF__"+s.name for s in TRAIN_SCENARIOS)
METHODS=("BOOKED",*POLICIES)
CLIFF=tuple(Scenario("CLIFF_A"+str(int(100*a))+"_H"+str(int(h)),
            "nonmonotone",a,h)
            for a in (.5,.8) for h in (20.,30.,60.))


def frozen_displays(ctx,grid):
    path=grid/"frozen_advice"/"evaluation_displays.csv"
    if not path.exists():
        raise FileNotFoundError("Frozen policy displays missing: "+str(path))
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    held=ctx["held"];ids=np.asarray(held["case_id"],int)
    booked=np.asarray(held["booked"],float)
    if len(rows)!=len(ids):
        raise AssertionError("Frozen displays case count mismatch")
    if not np.array_equal([int(r["case_id"]) for r in rows],ids):
        raise AssertionError("Frozen displays case order mismatch")
    if not np.allclose([float(r["booked"]) for r in rows],booked,
                       rtol=0.,atol=1e-9):
        raise AssertionError("Frozen displays booked values mismatch")
    return {p:np.asarray([float(r[p]) for r in rows],float) for p in POLICIES}


def verify_cell(file,s,policy,items):
    d=json.loads(file.read_text())
    if d.get("method")!=policy or d.get("evaluation")!=vars(s):
        raise ValueError("Incompatible cached response cell "+str(file))
    entries=d["entries"]
    expected={key for key,_,_ in items}
    if len(entries)!=len(expected) or {r["component"] for r in entries}!=expected:
        raise ValueError("Incomplete or wrong component roster "+str(file))
    return {r["component"]:r for r in entries}


def evaluate_cliff(ctx,grid,out,displays,cache):
    items=ctx["eval"]
    b=np.asarray(ctx["held"]["booked"],float)
    for scenario in CLIFF:
        for policy in METHODS:
            file=out/"evaluation_cells"/scenario.name/(policy+".json")
            if file.exists():
                verify_cell(file,scenario,policy,items)
                continue
            revised=b if policy=="BOOKED" else implemented_from_display(
                b,displays[policy],scenario)
            plans=cache.solve(items,revised,
                "cliff_"+scenario.name+"_"+policy,tie=True)
            entries=[]
            for key,c,rows in items:
                rec=plans[key];cost=evaluate(c,rec["assignment"],c.actual)
                entries.append(dict(component=key,week=c.monday,cases=c.n,
                    method=policy,evaluation_scenario=scenario.name,
                    cost=float(cost["cost"]),complete=bool(rec["complete"]),
                    planner_status=rec["status"]))
            save(file,dict(method=policy,evaluation=vars(scenario),
                           entries=entries))


def read_B(ctx,grid):
    cells={}
    for s in EVAL_SCENARIOS:
        for policy in METHODS:
            path=grid/"evaluation_cells"/s.name/(policy+".json")
            cells[s.name,policy]=verify_cell(path,s,policy,ctx["eval"])
    return cells


def summarize(ctx,grid,out,displays):
    cells=read_B(ctx,grid)
    for s in CLIFF:
        for p in METHODS:
            cells[s.name,p]=verify_cell(
                out/"evaluation_cells"/s.name/(p+".json"),s,p,ctx["eval"])
    common={key for key,_,_ in ctx["eval"]}
    for index in cells.values():
        common&={key for key,r in index.items() if r["complete"]}
    if not common:
        raise RuntimeError("No jointly certified population for cliff comparison")
    keys=sorted(common)
    reference=cells[EVAL_SCENARIOS[0].name,"BOOKED"]
    n=sum(reference[k]["cases"] for k in keys)
    data={}
    for s in (*EVAL_SCENARIOS,*CLIFF):
        data[s.name]={p:sum(cells[s.name,p][k]["cost"] for k in keys)/n
                      for p in METHODS}
    regret={s:{p:data[s][p]-min(data[s][q] for q in POLICIES)
               for p in POLICIES} for s in data}
    filename=out/"cliff_vs_saturated_matched.csv"
    with filename.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["alpha","h","link","scenario",
                         "certified_cases",*METHODS])
        for s in (*EVAL_SCENARIOS,*CLIFF):
            writer.writerow([s.alpha,s.h,s.link,s.name,n,
                             *[data[s.name][p] for p in METHODS]])
    with (out/"cliff_regret_matrix.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["scenario",*POLICIES])
        writer.writerows([[s,*[regret[s][p] for p in POLICIES]]
                          for s in data])
    sensitivity=[]
    for scenario in CLIFF:
        counterpart=next(s for s in EVAL_SCENARIOS
                         if s.alpha==scenario.alpha and s.h==scenario.h)
        for policy in POLICIES:
            display=displays[policy]
            cl=data[scenario.name][policy]
            sat=data[counterpart.name][policy]
            sensitivity.append(dict(
                trained_policy=policy,
                true_alpha=scenario.alpha,true_h=scenario.h,
                fraction_display_beyond_peak=float(np.mean(
                    np.abs(display)>scenario.h+1e-9)),
                saturated_cost=round(sat,8),
                cliff_cost=round(cl,8),
                cliff_minus_saturated=round(cl-sat,8),
                cliff_minus_booked=round(cl-data[scenario.name]["BOOKED"],8),
                cliff_candidate_regret=round(regret[scenario.name][policy],8)))
    with (out/"cliff_policy_sensitivity.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(sensitivity[0]))
        writer.writeheader();writer.writerows(sensitivity)
    report=dict(
        status="EXPLORATORY_CLIFF_STRESS_TEST",
        frozen_display_provenance=str(grid/"frozen_advice"/"evaluation_displays.csv"),
        common_certified_components=len(keys),
        all_components=len(ctx["eval"]),common_certified_cases=n,
        note="2013 repeatedly examined; candidate regret is not global regret",
        scenarios=[vars(s) for s in CLIFF],
        mean_costs=data,regret=regret,
        observations=sensitivity)
    save(out/"cliff_summary.json",report)
    print("CLIFF matched",len(keys),"/",len(ctx["eval"]),
          "components;",n,"cases",flush=True)
    for p in POLICIES:
        rows=[r for r in sensitivity if r["trained_policy"]==p]
        print(p,"maximum cliff-minus-saturated cost/case:",
              round(max(r["cliff_minus_saturated"] for r in rows),4),
              flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--grid",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with args.context.open("rb") as f:ctx=pickle.load(f)
    displays=frozen_displays(ctx,args.grid)
    cache=ContentCache(args.grid,args.cores,args.plan_seconds)
    evaluate_cliff(ctx,args.grid,args.out,displays,cache)
    summarize(ctx,args.grid,args.out,displays)


if __name__=="__main__":
    main()
