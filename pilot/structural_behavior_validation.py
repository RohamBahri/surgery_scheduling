#!/usr/bin/env python3
"""Deployment-only 12-scenario replay of frozen nominal structural vs clean VF.

No refits and NO re-canonicalizing the displayed advice. All B (saturating)
and C (nonmonotone) scenario results use ONE jointly certified 2013 population.
This checks behavioral stress sensitivity of the selected nominal policy;
it does NOT reestablish training-selected minimax regret for new features.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from pilot.run_response_scenario_grid import ContentCache
from pilot.run_nonmonotone_learning import save
from pilot.seed_compare_policies import frozen_case_table,sha,read
from surgery.response_grid import Scenario,EVAL_SCENARIOS,implemented_from_display
from surgery.weekly_planner import evaluate

CLIFF=tuple(Scenario("CLIFF_A"+str(int(a*100))+"_H"+str(int(h)),
                      "nonmonotone",a,h)
             for a in (.5,.8) for h in (20.,30.,60.))
SCENARIOS=(*EVAL_SCENARIOS,*CLIFF)
METHODS=("BOOKED","VF_clean","VF_structural")


def rows_to_costs(path,scenario,label,keys):
    record=read(path)
    if record.get("scenario")!=vars(scenario) or record.get("policy")!=label:
        raise ValueError("Incompatible cached scenario record: "+str(path))
    data=record.get("entries",[])
    if len(data)!=len(keys) or {r["component"] for r in data}!=keys:
        raise ValueError("Incomplete scenario records: "+str(path))
    return {r["component"]:r for r in data}


def run(ctx,clean_folder,structural_folder,out,cores,seconds):
    held=ctx["held"];b=np.asarray(held["booked"],float)
    clean=frozen_case_table(ctx,clean_folder,"clean")
    structural=frozen_case_table(ctx,structural_folder,"structural")
    displays={"VF_clean":clean["vf_display"],
              "VF_structural":structural["vf_display"]}
    items=ctx["eval"]
    keys={k for k,_,_ in items}
    cache=ContentCache(out,cores,seconds)
    rows={}
    # BOOKED is independent of response assumptions and cached once.
    booked_plans=cache.solve(items,b,"behavior_booked",tie=True)
    booked={}
    for key,c,index in items:
        rec=booked_plans[key]
        booked[key]=dict(component=key,cases=int(c.n),week=c.monday,
            complete=bool(rec["complete"]),
            cost=float(evaluate(c,rec["assignment"],c.actual)["cost"]))
    rows["BOOKED"]=booked
    for s in SCENARIOS:
        for label in ("VF_clean","VF_structural"):
            path=out/"evaluation_cells"/s.name/(label+".json")
            if not path.exists():
                reviewed=implemented_from_display(b,displays[label],s)
                solved=cache.solve(items,reviewed,s.name+"_"+label,tie=True)
                entries=[]
                for key,c,index in items:
                    rec=solved[key]
                    entries.append(dict(component=key,cases=int(c.n),
                        week=c.monday,complete=bool(rec["complete"]),
                        cost=float(evaluate(c,rec["assignment"],c.actual)["cost"]),
                        status=rec["status"]))
                save(path,dict(scenario=vars(s),policy=label,entries=entries))
            rows[s.name,label]=rows_to_costs(path,s,label,keys)
    common=set(keys)
    for cell in rows.values():
        common&={k for k,v in cell.items() if v["complete"]}
    if not common:
        raise RuntimeError("No jointly certified 12-scenario population")
    common=sorted(common)
    n=sum(booked[k]["cases"] for k in common)
    costs={"BOOKED":sum(booked[k]["cost"] for k in common)/n}
    results={}
    for s in SCENARIOS:
        vals={label:sum(rows[s.name,label][k]["cost"] for k in common)/n
              for label in ("VF_clean","VF_structural")}
        vals["BOOKED"]=costs["BOOKED"]
        results[s.name]=vals
    filename=out/"matched_12_scenario_costs.csv"
    with filename.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["scenario","link","alpha","h","certified_cases",
                         "BOOKED","VF_clean","VF_structural",
                         "structural_minus_clean"])
        for s in SCENARIOS:
            x=results[s.name]
            writer.writerow([s.name,s.link,s.alpha,s.h,n,
                             x["BOOKED"],x["VF_clean"],x["VF_structural"],
                             x["VF_structural"]-x["VF_clean"]])
    cliff=[]
    for sc in CLIFF:
        sat=next(s for s in EVAL_SCENARIOS
                 if s.alpha==sc.alpha and s.h==sc.h)
        row=dict(alpha=sc.alpha,h=sc.h)
        for label in ("VF_clean","VF_structural"):
            row[label+"_cliff_minus_saturated"]=(
                results[sc.name][label]-results[sat.name][label])
            row[label+"_beyond_true_threshold_fraction"]=float(np.mean(
                abs(displays[label])>sc.h+1e-9))
        cliff.append(row)
    with (out/"cliff_comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(cliff[0]))
        writer.writeheader()
        writer.writerows(cliff)
    adoption=dict(
        overall_costs={label:float(np.mean([results[s.name][label] for s in SCENARIOS]))
                       for label in METHODS},
        per_scenario=results,
        common_certified_components=len(common),common_certified_cases=n,
        all_components=len(items),all_cases=sum(c.n for _,c,_ in items),
        better_than_booked={
            label:sum(results[s.name][label]<costs["BOOKED"] for s in SCENARIOS)
            for label in ("VF_clean","VF_structural")},
        structural_better_than_clean_scenarios=sum(
            results[s.name]["VF_structural"]<results[s.name]["VF_clean"]
            for s in SCENARIOS),
        cliff_details=cliff,
        frozen_display_provenance={
            "clean":sha(clean_folder/"cases_test.csv"),
            "structural":sha(structural_folder/"cases_test.csv")},
        note="Exploratory 2013; 12 response curves hypothesized, not identified. "
             "This is a frozen nominal-policy deployment test, NOT retraining "
             "four scenario-specific structural VF policies.")
    save(out/"behavior_summary.json",adoption)
    return adoption


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--clean",type=Path,required=True)
    ap.add_argument("--structural",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    for attr in ("context","clean","structural","out"):
        setattr(args,attr,getattr(args,attr).resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    contract=dict(context_sha256=sha(args.context),
        clean_display_sha256=sha(args.clean/"cases_test.csv"),
        structural_display_sha256=sha(args.structural/"cases_test.csv"),
        plan_seconds=args.plan_seconds,scenarios=[vars(s) for s in SCENARIOS])
    path=args.out/"behavior_contract.json"
    if path.exists() and read(path)!=contract:
        raise ValueError("Behavior scenario contract changed: use NEW --out")
    save(path,contract)
    with args.context.open("rb") as f:ctx=pickle.load(f)
    result=run(ctx,args.clean,args.structural,args.out,args.cores,args.plan_seconds)
    print("12-scenario common cases:",result["common_certified_cases"],flush=True)
    print("structural better than clean:",
          result["structural_better_than_clean_scenarios"],"/12",flush=True)


if __name__=="__main__":
    main()
