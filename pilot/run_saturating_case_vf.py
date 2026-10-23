#!/usr/bin/env python3
"""One nominal saturating-link (B) CASE -> VF-pDCA UHN experiment.

Only three evaluation arms: BOOKED, CASE_B and VF_B.
B: raw score has NO casewise [-h,h] constraint; correction =
clip(alpha * raw, -alpha*h, +alpha*h). No attenuation/rejection.
Case durations retain a floor AFTER the response. Features/planner/roster
are inherited unchanged from the existing latest 12h SHARED/VF context.
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

for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot.run_nonmonotone_learning import (
    PlannerCache,case_fit,five_features,train_lambda,save,record_stage_time)
from surgery.nonmonotone_learning import (
    ResponseDC,response_saturating,mm_vf)
from surgery.weekly_planner import evaluate

ARMS=("BOOKED","CASE_B","VF_B")
ALPHA=.8
H=30.
CAP=180.
CASE_LAMBDA_SCALE=.75
VF_LAMBDA_SCALE=.5


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(root,args):
    from pilot import run_nonmonotone_learning as common
    from surgery import nonmonotone_learning as core
    sources=[Path(__file__),Path(common.__file__),Path(core.__file__)]
    provenance=hashlib.sha256(b"".join(f.read_bytes() for f in sources)).hexdigest()
    science=dict(context_sha256=_sha(args.context),
                 response_link="saturating",
                 raw_score_casewise_constraint=False,
                 alpha=ALPHA,h=H,cap=CAP,implemented_duration_floor=1.,
                 case_lambda_scale=CASE_LAMBDA_SCALE,
                 vf_lambda_scale=VF_LAMBDA_SCALE,
                 case_radius=100.,vf_radius=64.,
                 case_steps=args.case_steps,vf_outers=args.vf_outers,
                 vf_pdca_steps=args.vf_pdca_steps,cuts=args.cuts,
                 plan_seconds=args.plan_seconds,
                 planner_hash_seed=0,features="single_case_B_hash_draft",
                 backend="gurobi")
    file=root/"run_contract.json"
    if file.exists():
        previous=json.loads(file.read_text())
        if any(previous.get(k)!=v for k,v in science.items()):
            raise ValueError("Scientific setting mismatch; use a NEW output directory")
        if previous.get("source")!=provenance:
            history=previous.get("source_history",[])
            history.append(previous.get("source"))
            save(file,dict(**science,source=provenance,source_history=history))
            print("Code-source update documented; scientific settings identical",
                  flush=True)
    else:
        save(file,dict(**science,source=provenance))


def save_cases(path,data,case_score,vf_score,features,link):
    """Replay raw unrestricted scores, canonical displays, reviewed bookings."""
    names=("surgeon_spread","procedure_spread","draft_congestion",
           "surgeon_x_congestion","procedure_x_congestion")
    header=["case_id","booked","actual"]
    for name in ("CASE_B","VF_B"):
        header.extend([name+"_raw",name+"_display",name+"_reviewed",
                       name+"_raw_outside_h"])
    header.extend("feature_"+n for n in names)
    raw={"CASE_B":np.asarray(case_score,float),
         "VF_B":np.asarray(vf_score,float)}
    fields={name:(score,link.display(score),link.reviewed(score),
                  (np.abs(score)>H).astype(int))
            for name,score in raw.items()}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(header)
        for i in range(len(case_score)):
            out=[int(data["case_id"][i]),float(data["booked"][i]),
                 float(data["actual"][i])]
            for name in ("CASE_B","VF_B"):
                out.extend(float(v[i]) for v in fields[name][:3])
                out.append(int(fields[name][3][i]))
            writer.writerow(out+np.asarray(features[i],float).tolist())
    temp.replace(path)
    print(f"Saved {len(case_score)} rows: {path}",flush=True)


def train(ctx,args,root,cache):
    started=time.perf_counter()
    items=ctx["train"];data=ctx["fit"]
    booked=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    link=ResponseDC(booked,ALPHA,H,CAP,1.,link="saturating")

    print("CASE(B): unrestricted raw score, exactly saturated response; "
          f"lambda scale {CASE_LAMBDA_SCALE}",flush=True)
    weights=case_fit(ctx,items,link,root,args.case_steps,args.cuts,
                     "gurobi",100.,lambda_scale=CASE_LAMBDA_SCALE)
    case_score=np.asarray(data["X"],float)@weights
    link.verify(case_score)

    print("Computing ONE frozen CASE(B) hash-0 draft and shared features",
          flush=True)
    draft_clock=time.perf_counter()
    X,(mu,sd),draft=five_features(
         ctx,items,case_score,link,cache,"train")
    save(root/"vf_features.json",dict(mean=mu.tolist(),std=sd.tolist()))
    booked_plans=cache.solve(items,booked,"train_booked",tie=True)
    record_stage_time(root,"draft_and_booked",
                      time.perf_counter()-draft_clock)
    libraries={key:[booked_plans[key]["assignment"],
                    draft[key]["assignment"]]
               for key,_,_ in items}

    calibrated=train_lambda(X,len(items))
    vf_lam=VF_LAMBDA_SCALE*calibrated
    print(f"VF(B): library MM-pDCA only, start theta=0; "
          f"train-calibrated lambda={calibrated:.8f}, "
          f"fixed scale={VF_LAMBDA_SCALE}, used={vf_lam:.8f}",
          flush=True)
    vf_started=time.perf_counter()
    def oracle(parts,durations,k):
        return cache.solve(parts,durations,"vf_B_outer_"+str(k),tie=False)
    state=mm_vf(items,booked,X,case_score,link,actual,
                root=root/"vf_b",lam=vf_lam,
                initial_libraries=libraries,iterations=args.vf_outers,
                radius=64.,pdca_steps=args.vf_pdca_steps,
                max_cuts=args.cuts,backend="gurobi",
                solve_plans=oracle,optimizer="pdca")
    record_stage_time(root,"vf_b",time.perf_counter()-vf_started)
    theta=np.asarray(state["theta"],float)
    link.verify(case_score+X@theta)
    policy=dict(case_coeff=weights.tolist(),vf_theta=theta.tolist(),
                case_lambda=json.loads((root/"case_policy.json").read_text())[
                    "lambda_case"],
                vf_lambda=float(vf_lam),
                response_link="saturating",
                training_components=len(items),
                training_cases=len(booked),
                vf_status=state["status"],vf_mm_iterations=len(state["history"]))
    save(root/"selected_policy.json",policy)
    save_cases(root/"cases_train.csv",data,case_score,case_score+X@theta,X,link)
    record_stage_time(root,"training_total",time.perf_counter()-started)
    return policy


def evaluate_2013(ctx,root,cache,selected):
    started=time.perf_counter()
    items=ctx["eval"];data=ctx["held"]
    booked=np.asarray(data["booked"],float)
    link=ResponseDC(booked,ALPHA,H,CAP,1.,link="saturating")
    case_score=np.asarray(data["X"],float)@np.asarray(
         selected["case_coeff"],float)
    normal=json.loads((root/"vf_features.json").read_text())
    X,_,_=five_features(ctx,items,case_score,link,cache,"eval",
                     scaler=(np.asarray(normal["mean"],float),
                             np.asarray(normal["std"],float)))
    vf_score=case_score+X@np.asarray(selected["vf_theta"],float)
    link.verify(case_score);link.verify(vf_score)
    durations=dict(BOOKED=booked,CASE_B=link.reviewed(case_score),
                   VF_B=link.reviewed(vf_score))
    for label,score in (("CASE_B",case_score),("VF_B",vf_score)):
        displayed=link.display(score)
        adopted=np.maximum(booked+response_saturating(displayed,ALPHA,H),1.)
        if not np.allclose(adopted,durations[label],atol=1e-8):
            raise AssertionError("Deployment response discrepancy for "+label)
    save_cases(root/"cases_test.csv",data,case_score,vf_score,X,link)
    plans={label:cache.solve(items,durations[label],
                             "eval_"+label.lower(),tie=True)
           for label in ARMS}
    rows=[]
    for key,c,indices in items:
        for method in ARMS:
            rec=plans[method][key]
            cost=evaluate(c,rec["assignment"],c.actual)
            rows.append(dict(component=key,site=c.group,week=c.monday,
                 cases=c.n,method=method,complete=rec["complete"],
                 status=rec["status"],cost=float(cost["cost"]),
                 idle_minutes=float(cost["idle"]),
                 overtime_minutes=float(cost["overtime"]),
                 mae=float(np.abs(c.actual-durations[method][indices]).sum())))
    save(root/"evaluation_2013.json",dict(
         label="EXPLORATORY_REUSED_2013",entries=rows))
    report(root)
    record_stage_time(root,"evaluation",time.perf_counter()-started)


def report(root):
    entries=json.loads((root/"evaluation_2013.json").read_text())["entries"]
    by={}
    for row in entries:
        by.setdefault(row["component"],{})[row["method"]]=row
    common=[x for x in by.values()
            if all(a in x and x[a]["complete"] for a in ARMS)]
    n=sum(x["BOOKED"]["cases"] for x in common)
    output=dict(status="EXPLORATORY_REUSED_2013",
                response_link="saturating",
                common_certified_components=len(common),
                all_components=len(by),common_certified_cases=n,
                arm_coverage={a:dict(
                    certified_components=sum(bool(x.get(a,{}).get("complete"))
                                             for x in by.values()),
                    certified_cases=sum(x[a]["cases"]
                        for x in by.values() if a in x and x[a]["complete"]))
                    for a in ARMS})
    if n:
        for a in ARMS:
            output[a]=dict(
                cost_per_case=sum(x[a]["cost"] for x in common)/n,
                mae_per_case=sum(x[a]["mae"] for x in common)/n,
                idle_minutes_per_case=sum(x[a]["idle_minutes"]
                                          for x in common)/n,
                overtime_minutes_per_case=sum(x[a]["overtime_minutes"]
                                              for x in common)/n)
        weeks={}
        for x in common:
            monday=x["BOOKED"]["week"]
            wk=weeks.setdefault(monday,dict(n=0,cost={a:0. for a in ARMS}))
            wk["n"]+=x["BOOKED"]["cases"]
            for a in ARMS:
                wk["cost"][a]+=x[a]["cost"]
        keys=sorted(weeks)
        weights=np.array([weeks[k]["n"] for k in keys],float)
        rng=np.random.default_rng(20261022)
        draws=rng.integers(0,len(keys),size=(10000,len(keys)))
        denominator=weights[draws].sum(axis=1)
        output["paired_cost_differences"]={}
        for a,b in (("CASE_B","BOOKED"),("VF_B","CASE_B"),
                    ("VF_B","BOOKED")):
            delta=np.array([weeks[k]["cost"][a]-weeks[k]["cost"][b]
                            for k in keys])
            trials=delta[draws].sum(axis=1)/denominator
            output["paired_cost_differences"][a+"_minus_"+b]=dict(
               estimate=float(delta.sum()/n),
               calendar_week_bootstrap_95=[
                    float(v) for v in np.quantile(trials,[.025,.975])],
               better_weeks=int(np.sum(delta<0)),
               worse_weeks=int(np.sum(delta>0)),
               equal_weeks=int(np.sum(delta==0)),
               resamples=10000)
    save(root/"summary.json",output)
    print(json.dumps(output,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    parser.add_argument("--out",type=Path,default=Path(
        "artifacts/saturating_case_vf"))
    parser.add_argument("--cores",type=int,default=15)
    parser.add_argument("--plan-seconds",type=float,default=180.)
    parser.add_argument("--case-steps",type=int,default=200)
    parser.add_argument("--vf-outers",type=int,default=30)
    parser.add_argument("--vf-pdca-steps",type=int,default=50)
    parser.add_argument("--cuts",type=int,default=160)
    args=parser.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Missing 12h context checkpoint: "+str(args.context))
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        parser.error("Invalid cores/plan seconds")
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    contract(args.out,args)
    with args.context.open("rb") as handle:
        ctx=pickle.load(handle)
    print("B-link experiment:",len(ctx["train"]),"training and",
          len(ctx["eval"]),"evaluation components; "
          "BOOKED, CASE_B, VF_B only",flush=True)
    cache=PlannerCache(args.out,args.cores,args.plan_seconds)
    file=args.out/"selected_policy.json"
    if file.exists():
        selected=json.loads(file.read_text())
        if selected.get("response_link")!="saturating":
            raise ValueError("Saved policies have the wrong response link")
        print("Reusing frozen CASE(B) and VF(B) policies",flush=True)
    else:
        selected=train(ctx,args,args.out,cache)
    if not (args.out/"summary.json").exists():
        evaluate_2013(ctx,args.out,cache,selected)
    else:
        print("Evaluation already saved; no retuning",flush=True)
    record_stage_time(args.out,"one_command_total",time.perf_counter()-started)


if __name__=="__main__":
    main()
