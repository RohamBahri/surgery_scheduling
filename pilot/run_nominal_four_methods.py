#!/usr/bin/env python3
"""One resumable nominal UHN experiment: CASE, VF-pDCA, VF-DIRECTIONS, SHARED.

BOOKED is the unchanged reference. Each VF fits THE SAME five-coefficient
schedule-library objective and nominal response, with a different optimizer.
SHARED instead fits the realized cost of planner-selected weekly schedules.
No outcome-bank, behavioral-robustness, or additional VF-regularizer arm.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from pilot.run_nonmonotone_learning import (
    PlannerCache, case_fit, five_features, train_lambda, save,
    record_stage_time)
from surgery.nonmonotone_learning import ResponseDC, response, mm_vf
from surgery.nominal_shared import train_shared
from surgery.weekly_planner import evaluate as planner_evaluate, loads

ARMS=("BOOKED","CASE","VF_PDCA","VF_DIRECTIONS","SHARED")
CASE_L1_SCALE=.75


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scientific_contract(args,root):
    from surgery import nonmonotone_learning as vf_core, nominal_shared
    from pilot import run_nonmonotone_learning as common
    sources=[Path(__file__),Path(vf_core.__file__),Path(nominal_shared.__file__),
             Path(common.__file__)]
    settings=dict(context=digest(args.context.read_bytes()),
                  case_l1_scale=CASE_L1_SCALE,alpha=.8,h=30.,cap=180.,
                  case_steps=args.case_steps,case_radius=100.,
                  vf_outers=args.vf_outers,
                  vf_pdca_steps=args.vf_pdca_steps,
                  vf_radius=64.,vf_lambda_scale=1.,
                  shared_lambda_scale=1.,
                  shared_rounds=args.shared_rounds,
                  plan_seconds=args.plan_seconds,
                  cuts=args.cuts,backend="gurobi",
                  five_features="frozen_case_draft")
    source=digest(b"".join(p.read_bytes() for p in sources))
    file=root/"run_contract.json"
    if file.exists():
        prev=json.loads(file.read_text())
        if any(prev.get(k)!=v for k,v in settings.items()):
            raise ValueError("Changed scientific settings; choose a NEW output folder")
        if prev.get("source")!=source:
            prior=prev.get("source_history",[])
            prior.append(prev.get("source"))
            save(file,dict(**settings,source=source,source_history=prior))
            print("Source update recorded; prior scientific settings retained",flush=True)
    else:
        save(file,dict(**settings,source=source))


def save_case_data(path,data,case_raw,H,theta_map,behavior):
    """A reproducible record for all FOUR trained policies, not just VF."""
    n=len(case_raw)
    scores=dict(CASE=case_raw)
    for label,theta in theta_map.items():
        scores[label]=case_raw+H@np.asarray(theta,float)
    feature_names=("s","q","p","sp","qp")
    header=["case_id","booked","actual"]
    for label in ("CASE","VF_PDCA","VF_DIRECTIONS","SHARED"):
        header+=([label+"_raw",label+"_display",label+"_reviewed"])
    header+=["feature_"+k for k in feature_names]
    d={label:(values,behavior.display(values),behavior.reviewed(values))
       for label,values in scores.items()}
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    with temporary.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(header)
        for i in range(n):
            row=[int(data["case_id"][i]),float(data["booked"][i]),
                 float(data["actual"][i])]
            for label in ("CASE","VF_PDCA","VF_DIRECTIONS","SHARED"):
                row.extend(float(v[i]) for v in d[label])
            writer.writerow(row+H[i].tolist())
    temporary.replace(path)
    print(f"Saved per-case replay: {path} ({n} cases)",flush=True)


def combined_libraries(items,booked_plans,draft_plans,results):
    """Feasible schedules from both VF trainings; never a hindsight oracle."""
    output={}
    for key,c,rows in items:
        choices=[booked_plans[key]["assignment"],draft_plans[key]["assignment"]]
        for state in results.values():
            choices.extend(state["libraries"][key])
        unique={tuple(map(int,z)):list(map(int,z)) for z in choices}
        # A library is only screened after feasibility verification.
        for z in unique.values():
            loads(c,z,c.booked)
        output[key]=list(unique.values())
    return output


def train(ctx,args,root,cache):
    started=time.perf_counter()
    items=ctx["train"]
    data=ctx["fit"]
    booked=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    behavior=ResponseDC(booked,.8,30.,180.,1.)
    coeff=case_fit(ctx,items,behavior,root,
                   args.case_steps,args.cuts,"gurobi",100.,
                   lambda_scale=CASE_L1_SCALE)
    case_raw=np.asarray(data["X"],float)@coeff
    print("Preparing the single frozen CASE draft and five VF features",
          flush=True)
    draft_clock=time.perf_counter()
    H,(mean,std),draft=five_features(ctx,items,case_raw,behavior,cache,"train")
    save(root/"vf_features.json",dict(mean=mean.tolist(),std=std.tolist()))
    booked_plans=cache.solve(items,booked,"train_booked",tie=True)
    record_stage_time(root,"draft_and_booked",time.perf_counter()-draft_clock)
    libraries={key:[booked_plans[key]["assignment"],
                    draft[key]["assignment"]] for key,_,_ in items}
    vf_lam=train_lambda(H,len(items))
    results={}
    for name,method in (("VF_PDCA","pdca"),("VF_DIRECTIONS","directions")):
        started_method=time.perf_counter()
        print(f"START {name}: same objective and lambda={vf_lam:.8f}",flush=True)
        def oracle(parts,durations,k,label=name):
            return cache.solve(parts,durations,
                               label.lower()+"_outer_"+str(k),tie=False)
        state=mm_vf(items,booked,H,case_raw,behavior,actual,
                    root=root/name.lower(),lam=vf_lam,
                    initial_libraries=libraries,
                    iterations=args.vf_outers,radius=64.,
                    pdca_steps=args.vf_pdca_steps,max_cuts=args.cuts,
                    backend="gurobi",solve_plans=oracle,optimizer=method)
        results[name]=state
        record_stage_time(root,name.lower(),
                          time.perf_counter()-started_method)
        print(f"END {name}: {state['status']}; "
              f"iterations={len(state['history'])}",flush=True)
    print("START SHARED: direct realized schedule cost, "
          "hash-seed-0 deployment, train-only decisions",flush=True)
    shared_clock=time.perf_counter()
    union=combined_libraries(items,booked_plans,draft,results)
    shared=train_shared(items,booked,H,case_raw,behavior,cache,
             root=root/"shared",
             vf_thetas=[results["VF_PDCA"]["theta"],
                        results["VF_DIRECTIONS"]["theta"]],
             lam=vf_lam,libraries=union,max_rounds=args.shared_rounds)
    record_stage_time(root,"shared",time.perf_counter()-shared_clock)
    print("END SHARED:",shared["status"],"objective",shared["score"],flush=True)
    policies=dict(
        CASE=coeff.tolist(),VF_PDCA=results["VF_PDCA"]["theta"],
        VF_DIRECTIONS=results["VF_DIRECTIONS"]["theta"],
        SHARED=shared["theta"])
    selected=dict(policies=policies,lambda_case=float(
                       json.loads((root/"case_policy.json").read_text())["lambda_case"]),
                  lambda_vf=float(vf_lam),lambda_shared=float(vf_lam),
                  case_l1_scale=CASE_L1_SCALE,
                  alpha=.8,h=30.,cap=180.,
                  train_cases=len(booked),train_components=len(items),
                  vf_statuses={k:v["status"] for k,v in results.items()},
                  shared_status=shared["status"])
    save(root/"selected_policy.json",selected)
    save_case_data(root/"cases_train.csv",data,case_raw,H,
                   {k:policies[k] for k in ARMS if k not in ("BOOKED","CASE")},
                   behavior)
    record_stage_time(root,"training_total",time.perf_counter()-started)
    return selected


def evaluate(ctx,args,root,cache,selected):
    started=time.perf_counter()
    held=ctx["held"]
    items=ctx["eval"]
    booked=np.asarray(held["booked"],float)
    behavior=ResponseDC(booked,.8,30.,180.,1.)
    weights=selected["policies"]
    raw=np.asarray(held["X"],float)@np.asarray(weights["CASE"],float)
    scale=json.loads((root/"vf_features.json").read_text())
    H,_,_=five_features(ctx,items,raw,behavior,cache,"eval",
                        scaler=(np.asarray(scale["mean"],float),
                                np.asarray(scale["std"],float)))
    scores={label:raw+H@np.asarray(weights[label],float)
            for label in ("VF_PDCA","VF_DIRECTIONS","SHARED")}
    scores["CASE"]=raw
    durations={"BOOKED":booked}
    for name,advice in scores.items():
        durations[name]=behavior.reviewed(advice)
        # Exactly the same minimum display must reproduce the adopted booking.
        implemented=np.maximum(
            booked+response(behavior.display(advice),.8,30.),1.)
        if not np.allclose(implemented,durations[name],atol=1e-8):
            raise AssertionError("Nominal response mismatch: "+name)
    save_case_data(root/"cases_test.csv",held,raw,H,
                   {k:weights[k] for k in
                     ("VF_PDCA","VF_DIRECTIONS","SHARED")},behavior)
    plans={arm:cache.solve(items,durations[arm],
                           "evaluation_"+arm.lower(),tie=True)
           for arm in ARMS}
    rows=[]
    for key,week,indices in items:
        for arm in ARMS:
            rec=plans[arm][key]
            metrics=evaluate_week(week,rec["assignment"])
            rows.append(dict(component=key,site=week.group,week=week.monday,
                       cases=week.n,method=arm,
                       complete=rec["complete"],status=rec["status"],
                       cost=float(metrics["cost"]),
                       idle_minutes=float(metrics["idle"]),
                       overtime_minutes=float(metrics["overtime"]),
                       mae=float(np.abs(week.actual-
                           durations[arm][indices]).sum())))
    save(root/"evaluation_2013.json",dict(
         label="EXPLORATORY_2013_REUSED_IN_DEVELOPMENT",entries=rows))
    report(root)
    record_stage_time(root,"evaluation",time.perf_counter()-started)


def evaluate_week(week,assignment):
    return planner_evaluate(week,assignment,week.actual)


def report(root):
    entries=json.loads((root/"evaluation_2013.json").read_text())["entries"]
    by={}
    for row in entries:
        by.setdefault(row["component"],{})[row["method"]]=row
    common=[x for x in by.values()
            if all(a in x and x[a]["complete"] for a in ARMS)]
    n=sum(x["BOOKED"]["cases"] for x in common)
    output=dict(label="EXPLORATORY_REUSED_2013",
                common_certified_components=len(common),
                all_components=len(by),common_certified_cases=n,
                arm_coverage={arm:dict(
                    certified_components=sum(bool(x.get(arm,{}).get("complete"))
                                             for x in by.values()),
                    certified_cases=sum(x[arm]["cases"] for x in by.values()
                                        if arm in x and x[arm]["complete"]))
                              for arm in ARMS})
    if n:
        for arm in ARMS:
            output[arm]=dict(
                cost_per_case=sum(x[arm]["cost"] for x in common)/n,
                mae_per_case=sum(x[arm]["mae"] for x in common)/n,
                idle_minutes_per_case=sum(x[arm]["idle_minutes"]
                                          for x in common)/n,
                overtime_minutes_per_case=sum(x[arm]["overtime_minutes"]
                                              for x in common)/n)
        weekly={}
        for x in common:
            monday=x["BOOKED"]["week"]
            w=weekly.setdefault(monday,dict(
                n=0,costs={a:0. for a in ARMS}))
            w["n"]+=x["BOOKED"]["cases"]
            for arm in ARMS:
                w["costs"][arm]+=x[arm]["cost"]
        weeks=sorted(weekly)
        denominators=np.array([weekly[w]["n"] for w in weeks],float)
        rng=np.random.default_rng(20261022)
        draws=rng.integers(0,len(weeks),size=(10000,len(weeks)))
        bootstrap_denominator=denominators[draws].sum(axis=1)
        contrasts={}
        for compared,reference in (
                ("CASE","BOOKED"),("VF_PDCA","CASE"),
                ("VF_DIRECTIONS","CASE"),("SHARED","CASE"),
                ("VF_DIRECTIONS","VF_PDCA"),
                ("SHARED","VF_PDCA"),("SHARED","VF_DIRECTIONS")):
            changes=np.array([weekly[w]["costs"][compared]-
                              weekly[w]["costs"][reference] for w in weeks])
            samples=changes[draws].sum(axis=1)/bootstrap_denominator
            contrasts[compared+"_minus_"+reference]=dict(
                difference_per_case=float(changes.sum()/n),
                week_clustered_bootstrap_95=[
                    float(v) for v in np.quantile(samples,[.025,.975])],
                better_weeks=int(np.sum(changes<0)),
                worse_weeks=int(np.sum(changes>0)),
                tied_weeks=int(np.sum(changes==0)))
        output["paired_cost_differences"]=contrasts
    save(root/"summary.json",output)
    print(json.dumps(output,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    parser.add_argument("--out",type=Path,default=Path(
        "artifacts/nominal_four_methods"))
    parser.add_argument("--cores",type=int,default=15)
    parser.add_argument("--plan-seconds",type=float,default=180.)
    parser.add_argument("--case-steps",type=int,default=200)
    parser.add_argument("--vf-outers",type=int,default=30)
    parser.add_argument("--vf-pdca-steps",type=int,default=50)
    parser.add_argument("--shared-rounds",type=int,default=8)
    parser.add_argument("--cuts",type=int,default=160)
    args=parser.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Missing UHN context: "+str(args.context))
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        parser.error("Invalid cores/plan seconds")
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    scientific_contract(args,args.out)
    with args.context.open("rb") as handle:
        ctx=pickle.load(handle)
    print("ONE EXPERIMENT: four learned arms (CASE, VF_PDCA, "
          "VF_DIRECTIONS, SHARED) and BOOKED reference; "
          f"{len(ctx['train'])} train, {len(ctx['eval'])} evaluation "
          "components",flush=True)
    cache=PlannerCache(args.out,args.cores,args.plan_seconds)
    file=args.out/"selected_policy.json"
    if file.exists():
        selected=json.loads(file.read_text())
        print("Using already frozen four policies",flush=True)
    else:
        selected=train(ctx,args,args.out,cache)
    if not (args.out/"summary.json").exists():
        evaluate(ctx,args,args.out,cache,selected)
    else:
        print("Evaluation summary already present; no retuning",flush=True)
    record_stage_time(args.out,"one_command_total",time.perf_counter()-started)


if __name__=="__main__":
    main()
