#!/usr/bin/env python3
"""Final response-aware VF pipeline: primary-optimal scheduling, hash-only ties.

One public entry point for training, paired replay, exploratory evaluation
and operational audit. No alternative load-balancing optimization is present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from pilot import _joint_vf_engine as engine
from pilot.operational_audit import audit_policy
from surgery.nonmonotone_learning import ResponseDC
from surgery.weekly_planner import evaluate

DEFAULT_CONTEXT=Path("artifacts/objective_selection_12h/objective_selection_2013/context.pkl")


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("train","replay","evaluate","audit","all"),
                   default="all")
    p.add_argument("--context",type=Path,default=DEFAULT_CONTEXT)
    p.add_argument("--out",type=Path,default=Path("artifacts/vf_final_hash"))
    p.add_argument("--alpha",type=float,default=.8)
    p.add_argument("--naive-start",type=Path,default=None,
                   help="Shared train-only LAD start; bypass CASE training")
    p.add_argument("--h",type=float,default=30.)
    p.add_argument("--link",choices=("saturating","nonmonotone"),
                   default="saturating")
    p.add_argument("--vf-features",choices=("structural",),default="structural")
    p.add_argument("--turnover",type=float,default=0.)
    p.add_argument("--accuracy-mu",type=float,default=2.)
    p.add_argument("--accuracy-budget",type=float,default=.05)
    p.add_argument("--accuracy-gate",choices=("on","off"),default="off")
    p.add_argument("--train-tie",choices=("primary","deployed"),default="primary")
    p.add_argument("--lambda-vf",type=float,default=.1)
    p.add_argument("--cores",type=int,default=15)
    p.add_argument("--plan-seconds",type=float,default=300.)
    p.add_argument("--case-steps",type=int,default=200)
    p.add_argument("--vf-outers",type=int,default=30)
    p.add_argument("--vf-pdca-steps",type=int,default=50)
    p.add_argument("--cuts",type=int,default=80)
    p.add_argument("--radius",type=float,default=100.)
    p.add_argument("--audit-seconds",type=float,default=120.)
    p.add_argument("--audit-threshold",type=float,default=720.)
    a=p.parse_args(argv)
    if not a.context.is_file():
        p.error("Missing private context.pkl: "+str(a.context))
    if not (0<a.alpha<1 and a.h>0):
        p.error("Require alpha in (0,1) and positive response threshold")
    if a.turnover!=0.:
        p.error("The final scientific specification has zero turnover")
    if not all(np.isfinite(z) and z>=0. for z in (
            a.accuracy_mu,a.accuracy_budget,a.lambda_vf)):
        p.error("Accuracy penalties and regularization must be nonnegative and finite")
    if (a.cores<1 or a.plan_seconds<=0 or a.case_steps<1 or
        a.vf_outers<1 or a.vf_pdca_steps<1 or a.cuts<1 or a.radius<=0 or
        a.audit_seconds<=0 or a.audit_threshold<=0):
        p.error("Invalid numerical budget")
    a.context=a.context.resolve()
    if a.naive_start is not None:
        a.naive_start=a.naive_start.resolve()
        if not a.naive_start.is_file():
            p.error("Missing naive initializer: "+str(a.naive_start))
    a.out=a.out.resolve()
    return a


def configure_model(a):
    os.environ["SURGERY_TURNOVER_MINUTES"]="0"
    os.environ["SURGERY_RELAX_REQUIRED_DAYS"]="1"
    from surgery import weekly_planner,anchored_weekly_solver,nonmonotone_learning
    weekly_planner.TURNOVER=0.
    weekly_planner.RELAX_REQUIRED_DAYS=True
    anchored_weekly_solver.TURNOVER=0.
    nonmonotone_learning.TURNOVER=0.


def engine_args(a,stage):
    return ["--stage",stage,"--context",str(a.context),"--out",str(a.out),
        "--link",a.link,"--alpha",str(a.alpha),"--h",str(a.h),
        "--vf-features",a.vf_features,"--vf-lambda-absolute",str(a.lambda_vf),
        "--vf-loading","uniform","--vf-plan-penalty","1",
        "--case-steps",str(a.case_steps),"--vf-outers",str(a.vf_outers),
        "--vf-pdca-steps",str(a.vf_pdca_steps),"--cuts",str(a.cuts),
        "--case-radius",str(a.radius),"--vf-radius",str(a.radius),
        "--backend","gurobi","--cap","180",
        "--cores",str(a.cores),"--plan-seconds",str(a.plan_seconds),
        "--accuracy-mu",str(a.accuracy_mu),
        "--accuracy-budget",str(a.accuracy_budget),
        "--accuracy-gate",a.accuracy_gate,"--train-tie",a.train_tie]+(
        ["--naive-start",str(a.naive_start)] if a.naive_start else [])


def run_engine(a,stage):
    previous=sys.argv
    try:
        sys.argv=[str(Path(engine.__file__).resolve()),*engine_args(a,stage)]
        engine.main()
    finally:
        sys.argv=previous


def verified_context(a):
    contract=read(a.out/"run_contract.json")
    expected=dict(context=file_sha(a.context),alpha=a.alpha,h=a.h,
                  link=a.link,vf_features="structural",
                  vf_lambda_absolute=a.lambda_vf,
                  case_steps=a.case_steps,vf_outers=a.vf_outers,
                  vf_pdca_steps=a.vf_pdca_steps,cuts=a.cuts,
                  case_radius=a.radius,vf_radius=a.radius,
                  plan_seconds=a.plan_seconds,
                  naive_start_sha=(file_sha(a.naive_start) if a.naive_start else None),
                  accuracy_mu=a.accuracy_mu,accuracy_budget=a.accuracy_budget)
    if (contract.get("turnover")!=0. or
        contract.get("relaxed_required_days") is not True or
        contract.get("accuracy_kind")!="displayed_mae" or
        contract.get("accuracy_reference_kind")!="fitted_display_mae_reference" or
        contract.get("accuracy_gate")!=a.accuracy_gate or
        contract.get("vf_training_tie")!=(
            "primary_hash" if a.train_tie=="deployed" else "primary_optimum")):
        raise ValueError("Final hash-only scientific contract mismatch")
    bad={k:(contract.get(k),v) for k,v in expected.items()
         if contract.get(k)!=v}
    if bad:
        raise ValueError("Frozen policy configuration mismatch: "+str(bad))
    policy=read(a.out/"selected_policy.json")
    if (policy.get("method")!="JOINT_VF" or
        policy.get("vf_features")!="structural" or
        policy.get("accuracy_reference_kind")!="fitted_display_mae_reference" or
        policy.get("accuracy_gate")!=a.accuracy_gate or
        policy.get("training_uses_deployment_ties")!=(a.train_tie=="deployed") or
        not np.isclose(policy.get("accuracy_mu",-1),a.accuracy_mu) or
        not np.isclose(policy.get("accuracy_budget",-1),a.accuracy_budget) or
        not np.isclose(policy.get("lambda_vf",-1),a.lambda_vf) or
        len(policy.get("vf_theta",[]))!=83 or
        policy.get("initialization","CASE")!=(
            "NAIVE_LAD" if a.naive_start else "CASE")):
        raise ValueError("Frozen VF policy incompatible with requested settings")
    with a.context.open("rb") as stream:
        context=pickle.load(stream)
    return context,policy


def frozen_durations(a,ctx,split):
    data=ctx["fit" if split=="train" else "held"]
    cases=a.out/("cases_train.csv" if split=="train" else "cases_test.csv")
    with cases.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=len(data["booked"]):
        raise ValueError("Frozen case table length differs from context")
    if not np.array_equal(
            [int(z["case_id"]) for z in rows],data["case_id"]):
        raise ValueError("Frozen case-table order differs from context")
    for name in ("booked","actual"):
        if not np.allclose(
                [float(z[name]) for z in rows],data[name],atol=1e-9,rtol=0):
            raise ValueError(f"Frozen {name} does not match context")
    b=np.asarray(data["booked"],float)
    raw=np.array([float(z["vf_raw"]) for z in rows])
    expected=np.array([float(z["vf_reviewed"]) for z in rows])
    response=ResponseDC(b,a.alpha,a.h,180.,1.,link=a.link)
    reviewed=response.reviewed(raw)
    if not np.allclose(reviewed,expected,atol=1e-8,rtol=0):
        raise ValueError("Frozen VF reviewed bookings differ from model")
    return {"BOOKED":b,"VF":reviewed}


def replay_training(a,ctx):
    ds=frozen_durations(a,ctx,"train")
    items=ctx["train"]
    cache=engine.PlannerCache(a.out,a.cores,a.plan_seconds)
    plans={m:cache.solve(items,ds[m],"train_final_hash_"+m,tie=True)
           for m in ("BOOKED","VF")}
    rows=[]
    for key,c,idx in items:
        p=plans["BOOKED"][key]
        q=plans["VF"][key]
        rows.append(dict(week=str(c.monday),component=str(key),cases=int(c.n),
            booked_cost=float(evaluate(c,p["assignment"],c.actual)["cost"]),
            vf_cost=float(evaluate(c,q["assignment"],c.actual)["cost"]),
            both_certified=bool(p["complete"] and q["complete"]),
            same_assignment=p["assignment"]==q["assignment"]))
    result={}
    for subset,part in (("all_feasible",rows),
        ("common_certified",[z for z in rows if z["both_certified"]])):
        n=sum(z["cases"] for z in part)
        result[subset]=dict(components=len(part),cases=n)
        if n:
            B=sum(z["booked_cost"] for z in part)/n
            V=sum(z["vf_cost"] for z in part)/n
            result[subset].update(BOOKED_cost_per_case=B,
                VF_cost_per_case=V,VF_minus_BOOKED=V-B,
                identical_assignments=sum(z["same_assignment"] for z in part))
    engine.save(a.out/"training_replay_comparison.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def operational_audit(a,ctx):
    ds=frozen_durations(a,ctx,"eval")
    cache=engine.PlannerCache(a.out,a.cores,a.plan_seconds)
    output={}
    for method in ("BOOKED","VF"):
        plans=cache.solve(ctx["eval"],ds[method],
                          "eval_"+method,tie=True)
        result=audit_policy(ctx["eval"],ds[method],plans,
            a.out/"operational_audit_cache"/method,
            seconds=a.audit_seconds,threshold=a.audit_threshold)
        output[method]={k:v for k,v in result.items() if k!="entries"}
        engine.save(a.out/("operational_audit_"+method+".json"),result)
    engine.save(a.out/"operational_audit_summary.json",output)
    print(json.dumps(output,indent=2),flush=True)
    return output


def main(argv=None):
    a=parse_args(argv)
    configure_model(a)
    if a.stage in ("train","all"):
        if (a.out/"selected_policy.json").exists():
            verified_context(a)
            print("Reusing frozen VF policy",a.out,flush=True)
        else:
            run_engine(a,"all")
        if a.stage=="train":
            return
    if a.stage in ("evaluate","all"):
        run_engine(a,"evaluate")
        run_engine(a,"report")
    if a.stage in ("replay","audit","all"):
        ctx,_=verified_context(a)
        if a.stage=="replay":
            replay_training(a,ctx)
        if a.stage in ("audit","all"):
            operational_audit(a,ctx)


if __name__=="__main__":
    main()
