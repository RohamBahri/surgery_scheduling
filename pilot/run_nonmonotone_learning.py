#!/usr/bin/env python3
"""Train non-monotone displayed-advice CASE and VF on a frozen UHN context.

Run from repository root. Never updates the old frozen 2013 artifacts.
Use --evaluate only after fitting; the repeatedly inspected 2013 cohort is
retrospective/exploratory, not a fresh untouched holdout.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from surgery.nonmonotone_learning import (
    DCObjective, ResponseDC, Scenario, occupied_room_matrix, pdca, mm_vf)
from surgery.weekly_planner import loads, evaluate
from surgery.advice_comparison import scenarios as duration_scenarios


def save(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def component_key(key):
    return digest(str(key).encode())[:24]


def _solve_task(task):
    key,c,d,seconds,seed,tie=task
    from surgery.anchored_weekly_solver import solve_week
    ans=solve_week(c,np.asarray(d,float),seconds=seconds,threads=1,
                   tie=tie,hash_seed=seed)
    if not ans.complete or ans.assignment is None:
        return key,None,str(ans.status)
    assignment=list(map(int,ans.assignment))
    if abs(evaluate(c,assignment,d)["cost"]-ans.value)>0.11:
        raise AssertionError("Certified plan objective reconstruction failed: "+str(key))
    return key,assignment,"CERTIFIED"


class PlannerCache:
    def __init__(self,root,cores,seconds):
        self.root=Path(root)
        self.cores=int(cores)
        self.seconds=float(seconds)
        if self.cores<1 or self.seconds<=0:
            raise ValueError("Positive cores and planner seconds required")

    def solve(self,items,values,stage,seed=0,tie=False):
        cache_dir=self.root/"plans"/stage/("hash_"+str(seed) if tie else "primary")
        cache_dir.mkdir(parents=True,exist_ok=True)
        answers={}
        pending=[]
        for key,c,rows in items:
            d=np.asarray(values[rows],float)
            sig=digest(str(key).encode()+d.astype("<f8").tobytes()+
                       repr((self.seconds,seed,tie)).encode())
            path=cache_dir/(component_key(key)+".json")
            if path.exists():
                rec=json.loads(path.read_text())
                if rec.get("signature")==sig and rec.get("status")=="CERTIFIED":
                    answers[key]=rec["assignment"]
                    continue
            pending.append((key,c,d,self.seconds,seed,tie,sig,str(path)))
        if pending:
            print(stage,"seed",seed,"certified cached",len(answers),
                  "to solve",len(pending),flush=True)
        if self.cores==1:
            results=((task,_solve_task(task[:6])) for task in pending)
            for task,(key,z,status) in results:
                if z is None:
                    raise RuntimeError("Uncertified exact planner "+str(key)+": "+status)
                save(task[7],dict(signature=task[6],assignment=z,status=status))
                answers[key]=z
        else:
            with ProcessPoolExecutor(max_workers=self.cores,
                                     mp_context=get_context("spawn")) as pool:
                futures={pool.submit(_solve_task,task[:6]):task for task in pending}
                for f in as_completed(futures):
                    task=futures[f]
                    key,z,status=f.result()
                    if z is None:
                        raise RuntimeError("Uncertified exact planner "+str(key)+": "+status)
                    save(task[7],dict(signature=task[6],assignment=z,status=status))
                    answers[key]=z
        if len(answers)!=len(items):
            raise AssertionError("Missing exact planner components")
        return answers


def parse_scenarios(tokens):
    out=[]
    for token in tokens:
        chunks=token.split(":")
        if len(chunks) not in (2,3):
            raise ValueError("Use alpha:h or alpha:h:weight")
        out.append(Scenario(*map(float,chunks)))
    return tuple(out)


def provisional_features(ctx,case_display,cap,nominal,cache,items,old_H,stage,
                         standardize=None):
    """Rebuild congestion AFTER the new CASE fit; no old-CASE draft is reused.

    Historical s/q fields in old_H were standardized out-of-fold. Using them
    as the first two features and recomputing p, s*p, q*p preserves exactly
    the five-dimensional span of the original feature design.
    """
    booked=ctx["fit"]["booked"] if stage=="fit" else ctx["held"]["booked"]
    case_d=ResponseDC(booked,nominal.alpha,nominal.h,cap,
                      canonical_nominal=(nominal.alpha,nominal.h)).reviewed(case_display)
    maps=[cache.solve(items,case_d,stage+"_case_draft",seed=seed,tie=True)
          for seed in (0,1,2)]
    p=np.zeros(len(booked))
    filled=np.zeros(len(booked),bool)
    for key,c,rows in items:
        zs=[np.asarray(m[key],int) for m in maps]
        p[rows]=np.mean([np.clip((loads(c,z,case_d[rows])[z]-480.)/120.,-2.,2.)
                         for z in zs],axis=0)
        filled[rows]=True
    if not filled.all():
        raise AssertionError("Missing congestion feature rows")
    sq=np.asarray(old_H[:,:2],float)
    s,q=sq[:,0],sq[:,1]
    raw=np.column_stack((s,q,p,s*p,q*p))
    if standardize is None:
        mu=raw.mean(axis=0)
        sd=raw.std(axis=0)
        sd=np.where(sd<1e-10,1.,sd)
    else:
        mu,sd=standardize
    return (raw-mu)/sd,(mu,sd)


def prepare_bank(ctx,items,case_display,cap,nominal,draws,seed):
    """Freeze a scenario bank; no evaluation outcomes enter training."""
    from surgery.nonmonotone_learning import ResponseDC
    booked=np.asarray(ctx["fit"]["booked"],float)
    baseline=ResponseDC(booked,nominal.alpha,nominal.h,cap,
             canonical_nominal=(nominal.alpha,nominal.h)).reviewed(case_display)
    bank=np.zeros((draws,len(booked)))
    filled=np.zeros(len(booked),bool)
    for key,c,rows in items:
        bank[:,rows]=duration_scenarios(c,baseline[rows],ctx["pools"],seed,count=draws)
        filled[rows]=True
    if not filled.all() or not np.isfinite(bank).all():
        raise AssertionError("Invalid scenario bank")
    return bank


def evaluated(ctx,weights,theta,mu_sd,cap,nominal,scenarios,cache):
    """Evaluation on reused 2013 cohort; exactly solved or explicitly failed."""
    held=ctx["held"]
    items=ctx["eval"]
    X=np.asarray(held["X"],float)
    b=np.asarray(held["booked"],float)
    raw=X@weights
    H,_=provisional_features(ctx,raw,cap,nominal,cache,items,
                     ctx["H_eval"],"held",standardize=mu_sd)
    records={}
    for j,s in enumerate(scenarios):
        z_book=cache.solve(items,b,"held_BOOKED",seed=0,tie=True)
        case=ResponseDC(b,s.alpha,s.h,cap,canonical_nominal=(nominal.alpha,nominal.h))
        durations={"BOOKED":b,"CASE":case.reviewed(raw),
                    "VF":case.reviewed(raw+H@theta)}
        for name,d in durations.items():
            plans=z_book if name=="BOOKED" else cache.solve(
                items,d,"held_"+name+"_scenario_"+str(j),seed=0,tie=True)
            total=0.
            for key,c,rows in items:
                total+=evaluate(c,plans[key],c.actual)["cost"]
            records.setdefault(name,{})[str(j)]=float(total/len(b))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context",type=Path,
          default=Path("artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    p.add_argument("--out",type=Path,default=Path("artifacts/nonmonotone_case_vf"))
    p.add_argument("--caps",type=float,nargs="+",default=[45.])
    p.add_argument("--scenario",action="append",default=None,
                   help="alpha:h[:weight]; repeated scenarios, equal weights by default")
    p.add_argument("--nominal",default="0.8:30",
                   help="Nominal response defining minimum-magnitude displayed advice")
    p.add_argument("--draws",type=int,default=50)
    p.add_argument("--case-steps",type=int,default=25)
    p.add_argument("--vf-outers",type=int,default=8)
    p.add_argument("--vf-pdca-steps",type=int,default=20)
    p.add_argument("--max-cuts",type=int,default=60)
    p.add_argument("--case-radius",type=float,default=300.)
    p.add_argument("--vf-radius",type=float,default=32.)
    p.add_argument("--case-ridge",type=float,default=.001)
    p.add_argument("--case-l1",type=float,default=.0001)
    p.add_argument("--vf-ridge",type=float,default=.001)
    p.add_argument("--cores",type=int,default=8)
    p.add_argument("--plan-seconds",type=float,default=120.)
    p.add_argument("--backend",choices=("gurobi","scipy"),default="gurobi")
    p.add_argument("--evaluate",action="store_true")
    args=p.parse_args()
    if args.draws<1 or len(set(args.caps))!=len(args.caps):
        raise ValueError("Positive scenario count and distinct display caps required")
    scenarios=parse_scenarios(args.scenario or ["0.8:30","0.6:30","0.4:30","0.8:15"])
    nominal=parse_scenarios([args.nominal])[0]
    if not args.context.exists():
        raise FileNotFoundError("Need the prior exact-context checkpoint: "+str(args.context))
    with args.context.open("rb") as f:
        ctx=pickle.load(f)
    fit=ctx["fit"]
    if len(fit["booked"])!=24506 or len(ctx["held"]["booked"])!=8774:
        raise AssertionError("Unexpected frozen UHN training/evaluation cohort")
    if len(ctx["train"])!=3362 or len(ctx["eval"])!=1125:
        raise AssertionError("Unexpected number of exact planning components")
    from surgery import nonmonotone_learning as module
    manifest=dict(context_sha256=digest(args.context.read_bytes()),
        source_sha256=digest(Path(__file__).read_bytes()+Path(module.__file__).read_bytes()),
        caps=args.caps,scenarios=[vars(s) for s in scenarios],nominal=vars(nominal),
        draws=args.draws,case_steps=args.case_steps,vf_outers=args.vf_outers,
        vf_pdca_steps=args.vf_pdca_steps,max_cuts=args.max_cuts,
        case_radius=args.case_radius,vf_radius=args.vf_radius,
        case_ridge=args.case_ridge,case_l1=args.case_l1,vf_ridge=args.vf_ridge,
        backend=args.backend,plan_seconds=args.plan_seconds,
        note="Full nonmonotone canonical display; 2013 reused/exploratory")
    args.out.mkdir(parents=True,exist_ok=True)
    settings=args.out/"manifest.json"
    if settings.exists() and json.loads(settings.read_text())!=manifest:
        raise RuntimeError("Source or scientific settings changed: choose a new --out directory")
    save(settings,manifest)
    worst_jobs=len(ctx["train"])*len(args.caps)*(
        3+len(scenarios)*(args.vf_outers+1))
    if args.evaluate:
        worst_jobs+=len(ctx["eval"])*(3+1+2*len(scenarios))
    print("PRE-FLIGHT: full-cohort study; up to",worst_jobs,
          "planner-component requests before caching/analytic short-circuits;",
          len(ctx["train"]),"training components;",
          len(scenarios),"response scenarios;",
          args.vf_outers,"VF outer steps. No subset is silently used.",
          flush=True)
    cache=PlannerCache(args.out,args.cores,args.plan_seconds)
    X=np.asarray(fit["X"],float)
    b=np.asarray(fit["booked"],float)
    observed=np.asarray(fit["actual"],float)
    initial=np.asarray(fit["case_w"],float)/nominal.alpha
    results={}
    for cap in args.caps:
        folder=args.out/("cap_"+str(cap).replace(".","_"))
        folder.mkdir(parents=True,exist_ok=True)
        case_path=folder/"case_pdca.json"
        case_obj=DCObjective(b,X,np.zeros(len(b)),scenarios,cap,mode="CASE",
               actual=observed,ridge=args.case_ridge,l1=args.case_l1)
        if case_path.exists():
            case_fit=json.loads(case_path.read_text())
        else:
            case_fit=pdca(case_obj,initial,radius=args.case_radius,
                          outer=args.case_steps,max_cuts=args.max_cuts,
                          backend=args.backend)
            save(case_path,case_fit)
        weights=np.asarray(case_fit["weights"],float)
        raw_case=X@weights
        H,mu_sd=provisional_features(ctx,raw_case,cap,nominal,cache,
                      ctx["train"],ctx["H"],"fit_cap_"+str(cap))
        save(folder/"five_feature_normalization.json",
             dict(mean=mu_sd[0].tolist(),sd=mu_sd[1].tolist()))
        bank=prepare_bank(ctx,ctx["train"],raw_case,cap,nominal,args.draws,20261020)
        def solve_plans(items,durations,j,k):
            return cache.solve(items,durations,"fit_cap_"+str(cap)+
                               "_vf_outer_"+str(k)+"_scenario_"+str(j),
                               seed=0,tie=False)
        state=mm_vf(ctx["train"],b,H,raw_case,scenarios,bank,cap,
                    root=folder,iterations=args.vf_outers,radius=args.vf_radius,
                    ridge=args.vf_ridge,pdca_steps=args.vf_pdca_steps,
                    max_cuts=args.max_cuts,backend=args.backend,
                    canonical_nominal=(nominal.alpha,nominal.h),
                    solve_plans=solve_plans)
        theta=np.asarray(state["theta"],float)
        matrices=[]
        for j,s in enumerate(scenarios):
            d=ResponseDC(b,s.alpha,s.h,cap,
               canonical_nominal=(nominal.alpha,nominal.h)).reviewed(raw_case+H@theta)
            plan=cache.solve(ctx["train"],d,"fit_cap_"+str(cap)+
                             "_vf_final_scenario_"+str(j),seed=0,tie=False)
            S,_=occupied_room_matrix(ctx["train"],plan,len(b))
            matrices.append(S)
        final=DCObjective(b,H,raw_case,scenarios,cap,mode="VF",
             actual_bank=bank,room_matrices=matrices,ridge=args.vf_ridge,
             canonical_nominal=(nominal.alpha,nominal.h))
        result=dict(cap=cap,case_coeff=weights.tolist(),vf_theta=theta.tolist(),
                    case_training_objective=case_obj.value(weights),
                    vf_training_objective=final.value(theta),
                    vf_scenario_objectives=final.scenario_values(theta),
                    case_status=case_fit["status"],vf_status=state["status"],
                    vf_history=state["history"],
                    response_scenarios=[vars(s) for s in scenarios])
        save(folder/"fitted_policy.json",result)
        results[str(cap)]=result
    best=min(results.values(),key=lambda x:x["vf_training_objective"])
    save(args.out/"selected_policy.json",dict(selection="minimum prespecified training VF",
         cap=best["cap"],case_coeff=best["case_coeff"],vf_theta=best["vf_theta"],
         training_vf_objective=best["vf_training_objective"],
         full_train_components=len(ctx["train"]),train_cases=len(b)))
    if args.evaluate:
        selected=best
        norm=json.loads((args.out/("cap_"+str(best["cap"]).replace(".","_"))/
                         "five_feature_normalization.json").read_text())
        held=evaluated(ctx,np.asarray(best["case_coeff"]),np.asarray(best["vf_theta"]),
                       (np.asarray(norm["mean"]),np.asarray(norm["sd"])),
                       best["cap"],nominal,scenarios,cache)
        save(args.out/"retrospective_2013.json",
             dict(label="EXPLORATORY_REUSED_HOLDOUT",cost_per_case=held))
    print(json.dumps(dict(selected_cap=best["cap"],
                          vf_training_objective=best["vf_training_objective"],
                          artifact=str(args.out/"selected_policy.json")),
                     indent=2))


if __name__=="__main__":
    main()
