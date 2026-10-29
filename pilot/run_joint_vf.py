#!/usr/bin/env python3
"""Single entry point for structural, response-aware joint VF.

The POLICY is always 77 case + 6 schedule-independent structural features,
trained JOINTLY by schedule-library MM/pDCA and initialized at CASE.
Different assumptions and numerical budgets are configurations of this method.

Examples:
  python -m pilot.run_joint_vf --stage all --out artifacts/structural_main
  python -m pilot.run_joint_vf --stage replay --out artifacts/structural_main
  python -m pilot.run_joint_vf --stage seeds --out artifacts/structural_main
  python -m pilot.run_joint_vf --stage stress --out artifacts/structural_main \
      --true-alpha .5 --true-h 20 --true-link nonmonotone

Historical comparison/sweep code is accessible via old Git commits, not active.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from pilot import _joint_vf_engine as engine
from pilot._seed_solver import run_seed
from surgery.nonmonotone_learning import ResponseDC
from surgery.response_grid import Scenario, implemented_from_display
from surgery.weekly_planner import evaluate

LAMBDA_VF=0.1
DEFAULT_CONTEXT=Path("artifacts/objective_selection_12h/objective_selection_2013/context.pkl")


def read(file):
    return json.loads(Path(file).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage",choices=("all","train","evaluate","probe",
                    "replay","seeds","stress"),default="all")
    p.add_argument("--context",type=Path,default=DEFAULT_CONTEXT)
    p.add_argument("--out",type=Path,default=Path("artifacts/vf_direct_no_turnover"))
    p.add_argument("--alpha",type=float,default=.8,
                   help="Assumed adoption in TRAINING; policy is refitted when changed")
    p.add_argument("--h",type=float,default=30.,
                   help="Training display saturation threshold (minutes)")
    p.add_argument("--link",choices=("saturating","nonmonotone"),
                   default="saturating")
    p.add_argument("--lambda-vf",type=float,default=LAMBDA_VF,
                   help="Absolute VF L1 coefficient (intercept excluded)")
    p.add_argument("--cores",type=int,default=15)
    p.add_argument("--plan-seconds",type=float,default=180.)
    p.add_argument("--case-steps",type=int,default=200)
    p.add_argument("--vf-outers",type=int,default=30)
    p.add_argument("--vf-pdca-steps",type=int,default=50)
    p.add_argument("--cuts",type=int,default=80)
    p.add_argument("--radius",type=float,default=100.)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4])
    p.add_argument("--true-alpha",type=float,default=.8,
                   help="Assumed TRUE adoption for frozen-display stress only")
    p.add_argument("--true-h",type=float,default=30.,
                   help="Assumed TRUE threshold for frozen-display stress only")
    p.add_argument("--true-link",choices=("saturating","nonmonotone"),
                   default="saturating")
    a=p.parse_args(argv)
    if not a.context.is_file():p.error("Missing private context.pkl: "+str(a.context))
    if not(0<a.alpha<1 and a.h>0 and 0<a.true_alpha<1 and a.true_h>0):
        p.error("Response alphas must be in (0,1); thresholds must be positive")
    if not(np.isfinite(a.lambda_vf) and a.lambda_vf>=0):
        p.error("lambda-vf must be finite and nonnegative")
    if a.cores<1 or a.plan_seconds<=0 or a.case_steps<1 or a.vf_outers<1 or a.vf_pdca_steps<1 or a.cuts<1 or a.radius<=0:
        p.error("Invalid compute budget")
    if len(set(a.seeds))!=len(a.seeds) or any(s<0 for s in a.seeds):
        p.error("Hash seeds must be distinct and nonnegative")
    a.context=a.context.resolve()
    a.out=a.out.resolve()
    return a


def engine_args(a,stage):
    return ["--stage",stage,"--context",str(a.context),"--out",str(a.out),
        "--link",a.link,"--alpha",str(a.alpha),"--h",str(a.h),
        "--vf-features","structural","--vf-lambda-absolute",str(a.lambda_vf),
        "--vf-loading","uniform","--vf-plan-penalty","1",
        "--case-steps",str(a.case_steps),"--vf-outers",str(a.vf_outers),
        "--vf-pdca-steps",str(a.vf_pdca_steps),"--cuts",str(a.cuts),
        "--case-radius",str(a.radius),"--vf-radius",str(a.radius),
        "--backend","gurobi","--cap","180",
        "--cores",str(a.cores),"--plan-seconds",str(a.plan_seconds)]


def run_engine(a,stage):
    old=sys.argv
    try:
        sys.argv=[str(Path(engine.__file__).resolve()),*engine_args(a,stage)]
        engine.main()
    finally:
        sys.argv=old


def verified_context(a):
    folder=a.out
    contract=read(folder/"run_contract.json")
    expected=dict(context=file_sha(a.context),alpha=a.alpha,h=a.h,
        link=a.link,vf_features="structural",vf_lambda_absolute=a.lambda_vf,
        case_steps=a.case_steps,vf_outers=a.vf_outers,
        vf_pdca_steps=a.vf_pdca_steps,cuts=a.cuts,
        case_radius=a.radius,vf_radius=a.radius,
        plan_seconds=a.plan_seconds)
    bad={k:(contract.get(k),v) for k,v in expected.items()
         if contract.get(k)!=v}
    if bad:raise ValueError("Frozen policy configuration mismatch: "+str(bad)+
                           ". Use a separate --out for each configuration.")
    selected=read(folder/"selected_policy.json")
    if (selected.get("method")!="JOINT_VF" or
        selected.get("vf_features")!="structural" or
        selected.get("link")!=a.link or
        not np.isclose(selected["lambda_vf"],a.lambda_vf) or
        not np.isclose(selected["alpha"],a.alpha) or
        not np.isclose(selected["h"],a.h) or
        len(selected["vf_theta"])!=83):
        raise ValueError("Saved policy is not the requested structural joint VF")
    with a.context.open("rb") as f:ctx=pickle.load(f)
    return ctx,selected


def case_rows(ctx,root,split):
    data=ctx["fit" if split=="train" else "held"]
    file=root/("cases_train.csv" if split=="train" else "cases_test.csv")
    with file.open(newline="") as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(data["booked"]):
        raise ValueError("Saved case table length differs from context")
    if not np.array_equal([int(r["case_id"]) for r in rows],data["case_id"]):
        raise ValueError("Saved case table case order differs from context")
    for field in ("booked","actual"):
        v=np.array([float(r[field]) for r in rows])
        if not np.allclose(v,data[field],atol=1e-9,rtol=0):
            raise ValueError("Saved "+field+" differs from original context")
    return data,rows


def replay_training(a,ctx):
    """Adapted from user's check_joint_vf_train.py; no hard-coded artifact path."""
    data,rows=case_rows(ctx,a.out,"train")
    booked=np.asarray(data["booked"],float)
    model=ResponseDC(booked,a.alpha,a.h,180.,1.,link=a.link)
    durations={"CASE":np.array([float(r["case_reviewed"]) for r in rows]),
               "VF":np.array([float(r["vf_reviewed"]) for r in rows])}
    for name,col in (("CASE","case_raw"),("VF","vf_raw")):
        raw=np.array([float(r[col]) for r in rows])
        if not np.allclose(model.reviewed(raw),durations[name],atol=1e-8,rtol=0):
            raise ValueError(name+" frozen training reviewed durations disagree with response")
    cache=engine.PlannerCache(a.out,a.cores,a.plan_seconds)
    plans={"CASE":cache.solve(ctx["train"],durations["CASE"],
                               "train_case_draft",tie=True),
           "VF":cache.solve(ctx["train"],durations["VF"],
                             "train_final_vf_tie",tie=True)}
    entries=[]
    for key,c,ix in ctx["train"]:
        p=plans["CASE"][key];q=plans["VF"][key]
        entries.append(dict(component=str(key),cases=int(len(ix)),
            case_cost=float(evaluate(c,p["assignment"],c.actual)["cost"]),
            vf_cost=float(evaluate(c,q["assignment"],c.actual)["cost"]),
            both_certified=bool(p["complete"] and q["complete"]),
            same_assignment=p["assignment"]==q["assignment"]))
    report={}
    for label,part in (
        ("all_feasible_diagnostic",entries),
        ("common_certified",[e for e in entries if e["both_certified"]])):
        n=sum(r["cases"] for r in part)
        report[label]=dict(components=len(part),cases=n)
        if n:
            c=sum(r["case_cost"] for r in part)/n
            v=sum(r["vf_cost"] for r in part)/n
            report[label].update(CASE_cost_per_case=c,VF_cost_per_case=v,
                VF_minus_CASE=v-c,
                VF_better_components=sum(e["vf_cost"]<e["case_cost"]-1e-7 for e in part),
                VF_worse_components=sum(e["vf_cost"]>e["case_cost"]+1e-7 for e in part),
                same_assignment_components=sum(e["same_assignment"] for e in part))
    engine.save(a.out/"training_replay_comparison.json",report)
    print(json.dumps(report,indent=2))
    return report


def frozen_test_durations(a,ctx):
    held,rows=case_rows(ctx,a.out,"eval")
    b=np.asarray(held["booked"],float)
    model=ResponseDC(b,a.alpha,a.h,180.,1.,link=a.link)
    displays={}
    durations={"BOOKED":b}
    for name,col in (("CASE","case"),("VF","vf")):
        raw=np.array([float(r[col+"_raw"]) for r in rows])
        shown=np.array([float(r[col+"_display"]) for r in rows])
        reviewed=np.array([float(r[col+"_reviewed"]) for r in rows])
        if not np.allclose(shown,model.display(raw),atol=1e-8,rtol=0):
            raise ValueError(name+" frozen displayed advice mismatch")
        if not np.allclose(reviewed,model.reviewed(raw),atol=1e-8,rtol=0):
            raise ValueError(name+" frozen reviewed durations mismatch")
        displays[name]=shown
        durations[name]=reviewed
    return b,displays,durations


def paired_weeks(cells,roster,methods,seeds):
    common=sorted(set(roster).intersection(*(
        {key for key,v in cells[s,m].items() if v["complete"]}
        for s in seeds for m in methods)))
    if not common:raise RuntimeError("No jointly certified components")
    weeks=sorted({roster[k]["week"] for k in common})
    ids={w:i for i,w in enumerate(weeks)}
    weights=np.array([roster[k]["cases"] for k in common],float)
    widx=np.array([ids[roster[k]["week"]] for k in common])
    den=np.bincount(widx,weights=weights,minlength=len(weeks))
    n=float(weights.sum())
    totals={}
    for m in methods:
        totals[m]={str(s):sum(cells[s,m][k]["cost"] for k in common)/n
                   for s in seeds}
    comparisons={}
    for candidate,reference in (("VF","CASE"),("VF","BOOKED"),("CASE","BOOKED")):
        numer=np.zeros(len(weeks),float);per_seed={}
        for s in seeds:
            delta=np.array([cells[s,candidate][k]["cost"]-
                            cells[s,reference][k]["cost"] for k in common])
            per_seed[str(s)]=float(delta.sum()/n)
            numer+=np.bincount(widx,weights=delta,minlength=len(weeks))/len(seeds)
        rng=np.random.default_rng(20261028)
        samples=rng.integers(0,len(weeks),size=(10000,len(weeks)))
        boot=numer[samples].sum(1)/den[samples].sum(1)
        comparisons[candidate+"_minus_"+reference]=dict(
            seed_mean=float(numer.sum()/n),per_seed=per_seed,
            paired_calendar_week_bootstrap_95=list(map(float,np.quantile(boot,[.025,.975]))),
            better_weeks=int(sum(numer<0)),worse_weeks=int(sum(numer>0)))
    return dict(common_components=len(common),common_cases=int(n),
        all_components=len(roster),all_cases=sum(q["cases"] for q in roster.values()),
        seeds=list(seeds),cost_per_case=totals,comparisons=comparisons)


def audit_seeds(a,ctx):
    _,_,durations=frozen_test_durations(a,ctx)
    items=ctx["eval"]
    cells={};roster={key:dict(cases=c.n,week=c.monday) for key,c,_ in items}
    for seed in a.seeds:
        for method,d in durations.items():
            plans=run_seed(items,d,a.out/"audit",method,seed,a.cores,a.plan_seconds)
            cells[seed,method]={
                key:dict(cost=float(evaluate(c,plans[key]["assignment"],c.actual)["cost"]),
                         complete=bool(plans[key]["complete"]))
                for key,c,_ in items}
    result=paired_weeks(cells,roster,("BOOKED","CASE","VF"),a.seeds)
    engine.save(a.out/"seed_robustness.json",result)
    print(json.dumps(result,indent=2))
    return result


def audit_stress(a,ctx):
    """Apply one different TRUE-response configuration to frozen policy displays."""
    b,displays,_=frozen_test_durations(a,ctx)
    scenario=Scenario("scenario",""+a.true_link,a.true_alpha,a.true_h)
    ds={"BOOKED":b}
    for name in ("CASE","VF"):
        ds[name]=implemented_from_display(b,displays[name],scenario)
    cache=engine.PlannerCache(a.out,a.cores,a.plan_seconds)
    tag=f"stress_{a.true_link}_a{a.true_alpha:.4f}_h{a.true_h:.4f}"
    plans={m:cache.solve(ctx["eval"],d,tag+"_"+m,tie=True)
           for m,d in ds.items()}
    valid=[(key,c) for key,c,_ in ctx["eval"]
           if all(plans[m][key]["complete"] for m in ds)]
    n=sum(c.n for _,c in valid)
    if n<=0:raise RuntimeError("No jointly certified stress components")
    costs={m:sum(evaluate(c,plans[m][key]["assignment"],c.actual)["cost"]
                 for key,c in valid)/n for m in ds}
    output=dict(training_scenario=dict(alpha=a.alpha,h=a.h,link=a.link),
        deployed_response=vars(scenario),common_cases=n,
        common_components=len(valid),cost_per_case=costs,
        exploratory_2013=True,display_advice_frozen=True)
    name=tag+".json"
    engine.save(a.out/name,output)
    print(json.dumps(output,indent=2))
    return output


def main(argv=None):
    a=parse_args(argv)
    if a.stage in ("all","train"):
        if not (a.out/"selected_policy.json").exists():
            run_engine(a,"all")
        else:
            # Validate contract and policy; do not silently retrain on a mismatch.
            verified_context(a)
            print("Reusing completed structural joint VF policy",a.out)
        if a.stage=="train":return
    if a.stage in ("all","evaluate"):
        # Engine enforces the scientific source/settings checkpoint contract.
        run_engine(a,"evaluate")
        run_engine(a,"report")
    if a.stage in ("all","probe"):
        run_engine(a,"probe")
    if a.stage in ("replay","seeds","stress"):
        ctx,_=verified_context(a)
        if a.stage=="replay":replay_training(a,ctx)
        elif a.stage=="seeds":audit_seeds(a,ctx)
        else:audit_stress(a,ctx)


if __name__=="__main__":
    main()
