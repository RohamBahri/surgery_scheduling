#!/usr/bin/env python3
"""Hash-seed sensitivity of frozen BOOKED, CASE, joint VF policies.

Freeze the nominal advice ONCE using seed 0 booked-plan features, then
re-solve deployment under hash tie seeds 0,1,2. An observed difference is a
tie-breaking effect, not a change to the policy or to duration predictions.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from pilot.run_nonmonotone_learning import PlannerCache,five_features,save
from surgery.nonmonotone_learning import ResponseDC,response
from surgery.weekly_planner import evaluate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with args.context.open("rb") as f:ctx=pickle.load(f)
    selection=json.loads((args.base/"selected_policy.json").read_text())
    if (selection.get("method")!="JOINT_VF" or selection.get("link")!="saturating"
            or len(selection["vf_theta"])!=82):
        raise ValueError("Expected the frozen five-plan-feature nominal joint VF")
    held=ctx["held"];items=ctx["eval"]
    b=np.asarray(held["booked"],float);X=np.asarray(held["X"],float)
    os.environ["SURGERY_ANCHORED_HASH_SEED"]="0"
    booked=PlannerCache(args.out/"feature_seed0",args.cores,args.plan_seconds)
    draft=booked.solve(items,b,"booked_for_fixed_features",tie=True)
    scale=json.loads((args.base/"vf_features.json").read_text())
    H,_,_=five_features(ctx,items,booked,"eval",
        scaler=(np.asarray(scale["mean"]),np.asarray(scale["std"])),
        plans=draft,schema="baseline")
    response_map=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    case_raw=X@np.asarray(selection["case_coeff"],float)
    vf_raw=np.column_stack([X,H])@np.asarray(selection["vf_theta"],float)
    duration={"BOOKED":b,
              "CASE":np.maximum(1.,b+response(response_map.display(case_raw),.8,30.)),
              "VF":np.maximum(1.,b+response(response_map.display(vf_raw),.8,30.))}
    for seed in (0,1,2):
        folder=args.out/("seed"+str(seed))
        os.environ["SURGERY_ANCHORED_HASH_SEED"]=str(seed)
        cache=PlannerCache(folder,args.cores,args.plan_seconds)
        for name,d in duration.items():
            file=folder/(name+"_cost.json")
            if file.exists():continue
            plans=cache.solve(items,d,"frozen_"+name+"_hash_seed_"+str(seed),tie=True)
            rows=[]
            for key,c,index in items:
                rec=plans[key]
                cost=evaluate(c,rec["assignment"],c.actual)["cost"]
                rows.append(dict(component=key,week=c.monday,cases=c.n,
                    certified=bool(rec["complete"]),cost=float(cost),
                    status=rec["status"]))
            save(file,dict(seed=seed,method=name,entries=rows))
    os.environ["SURGERY_ANCHORED_HASH_SEED"]="0"
    results={}
    common=None
    for seed in (0,1,2):
        for name in duration:
            file=args.out/("seed"+str(seed))/(name+"_cost.json")
            rec=json.loads(file.read_text())["entries"]
            result={r["component"]:r for r in rec}
            results[seed,name]=result
            subset={k for k,v in result.items() if v["certified"]}
            common=subset if common is None else common&subset
    if not common:raise RuntimeError("No common certified seed audit population")
    keys=sorted(common)
    n=sum(results[0,"BOOKED"][k]["cases"] for k in keys)
    output=dict(common_certified_components=len(keys),cases=n,seeds={})
    for seed in (0,1,2):
        output["seeds"][str(seed)]={}
        for label in duration:
            chosen=results[seed,label]
            ref=results[0,label]
            total=sum(chosen[k]["cost"] for k in keys)/n
            delta=np.array([chosen[k]["cost"]-ref[k]["cost"] for k in keys])
            output["seeds"][str(seed)][label]=dict(
                cost_per_case=float(total),
                delta_vs_seed0_per_case=float(
                    sum(delta)/n),
                nonidentical_cost_components=int(np.sum(abs(delta)>1e-7)),
                max_abs_component_cost_difference=float(max(abs(delta),default=0.)))
    save(args.out/"seed_robustness.json",output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":
    main()
