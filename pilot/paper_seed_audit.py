#!/usr/bin/env python3
"""Explicit-seed tie-breaking audit of the frozen nominal joint VF policy.

IMPORTANT: never use PlannerCache.solve for seed 1 or 2. Its worker historically
hardcoded hash_seed=0. This independent worker passes each seed explicitly to
the solver, checks provenance, and caches with the seed in the signature.
Use a NEW output folder (seed_audit_fixed), not the old invalid seed results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from pilot.run_nonmonotone_learning import PlannerCache,five_features,save
from surgery.nonmonotone_learning import ResponseDC,response
from surgery.weekly_planner import evaluate


def explicit_job(job):
    key,c,d,seconds,seed=job
    from surgery.anchored_weekly_solver import solve_week
    result=solve_week(c,d,seconds=seconds,threads=1,tie=True,hash_seed=seed)
    z=(list(map(int,result.assignment))
       if result.assignment is not None else list(map(int,c.original)))
    return key,dict(assignment=z,complete=bool(result.complete),
                    status=str(result.status),hash_seed=seed)


def run_seed(items,duration,folder,label,seed,cores,seconds):
    """Per-component resumability with a duration- and seed-specific signature."""
    cache=folder/"explicit_plans"/("seed"+str(seed))/label
    cache.mkdir(parents=True,exist_ok=True)
    answers={};jobs=[]
    for key,c,rows in items:
        d=np.ascontiguousarray(np.asarray(duration[rows],float),dtype="<f8")
        signature=hashlib.sha256(
            str(key).encode()+d.tobytes()+
            repr((float(seconds),int(seed),"tie")).encode()).hexdigest()
        path=cache/(hashlib.sha256(str(key).encode()).hexdigest()+".json")
        if path.exists():
            record=json.loads(path.read_text())
            if (record.get("signature")==signature and
                    record.get("result",{}).get("hash_seed")==seed):
                answers[key]=record["result"]
                continue
        jobs.append(((key,c,d,seconds,seed),path,signature))
    print(label,"seed",seed,":",len(answers),"cached;",len(jobs),"solves",flush=True)
    def store(reply,path,signature):
        key,data=reply
        if data["hash_seed"]!=seed:
            raise AssertionError("Solver received incorrect hash seed")
        save(path,dict(signature=signature,result=data))
        answers[key]=data
    if cores<=1:
        for job,path,sig in jobs:
            store(explicit_job(job),path,sig)
    elif jobs:
        with ProcessPoolExecutor(max_workers=cores,
                                 mp_context=get_context("spawn")) as pool:
            pending={pool.submit(explicit_job,job):(path,sig)
                     for job,path,sig in jobs}
            for i,future in enumerate(as_completed(pending),1):
                path,sig=pending[future]
                store(future.result(),path,sig)
                if i%100==0 or i==len(pending):
                    print(label,"seed",seed,":",i,"/",len(pending),flush=True)
    return answers


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    # Refuse old invalid results rather than accepting their apparent zero deltas.
    if (args.out/"seed1"/"VF_cost.json").exists():
        raise ValueError("Old invalid seed results in --out; use seed_audit_fixed")
    with args.context.open("rb") as f:ctx=pickle.load(f)
    selection=json.loads((args.base/"selected_policy.json").read_text())
    if (selection.get("method")!="JOINT_VF" or selection.get("link")!="saturating"
            or len(selection["vf_theta"])!=82):
        raise ValueError("Expected frozen five-feature nominal joint VF")
    held=ctx["held"];items=ctx["eval"]
    b=np.asarray(held["booked"],float);X=np.asarray(held["X"],float)
    # The features must be frozen from the original seed-zero booked plan.
    feature_cache=PlannerCache(args.out/"feature_seed0",args.cores,args.plan_seconds)
    draft=feature_cache.solve(items,b,"booked_for_fixed_features",tie=True)
    scaler=json.loads((args.base/"vf_features.json").read_text())
    H,_,_=five_features(ctx,items,feature_cache,"eval",
        scaler=(np.asarray(scaler["mean"]),np.asarray(scaler["std"])),
        plans=draft,schema="baseline")
    m=ResponseDC(b,.8,30.,180.,1.,link="saturating")
    case_raw=X@np.asarray(selection["case_coeff"],float)
    vf_raw=np.column_stack([X,H])@np.asarray(selection["vf_theta"],float)
    duration=dict(
        BOOKED=b,
        CASE=np.maximum(1.,b+response(m.display(case_raw),.8,30.)),
        VF=np.maximum(1.,b+response(m.display(vf_raw),.8,30.)))
    all_cells={}
    common=None
    for seed in (0,1,2):
        for name,d in duration.items():
            plans=run_seed(items,d,args.out,name,seed,args.cores,args.plan_seconds)
            cell={}
            for key,c,rows in items:
                rec=plans[key]
                actual_cost=evaluate(c,rec["assignment"],c.actual)["cost"]
                cell[key]=dict(cost=float(actual_cost),assignment=rec["assignment"],
                    complete=bool(rec["complete"]),cases=c.n,week=c.monday)
            all_cells[seed,name]=cell
            valid={key for key,rec in cell.items() if rec["complete"]}
            common=valid if common is None else common&valid
    if not common:raise RuntimeError("No jointly certified audit cases")
    keys=sorted(common);n=sum(all_cells[0,"BOOKED"][k]["cases"] for k in keys)
    report=dict(valid_explicit_seeds=True,common_certified_components=len(keys),
                common_certified_cases=n,all_components=len(items),seeds={})
    for seed in (0,1,2):
        report["seeds"][str(seed)]={}
        for label in duration:
            here=all_cells[seed,label];ref=all_cells[0,label]
            total=sum(here[k]["cost"] for k in keys)
            differences=np.array([here[k]["cost"]-ref[k]["cost"] for k in keys])
            changed=sum(here[k]["assignment"]!=ref[k]["assignment"] for k in keys)
            report["seeds"][str(seed)][label]=dict(
                cost_per_case=float(total/n),
                difference_from_seed0_per_case=float(differences.sum()/n),
                changed_assignments=changed,
                changed_realized_cost_components=int(sum(abs(differences)>1e-7)),
                max_abs_component_cost_difference=float(max(abs(differences),default=0.)))
    save(args.out/"seed_robustness.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
