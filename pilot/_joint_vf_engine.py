#!/usr/bin/env python3
"""Internal engine: structural joint-VF learning, CASE warm start and benchmark.

Invoked through pilot.run_joint_vf. The feature representation is frozen;
behavioral parameters and numerical budgets may vary as explicit configurations.
"""
from __future__ import annotations

import argparse
import csv
import time
import hashlib
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from surgery.nonmonotone_learning import (
    ResponseDC, DCObjective, choose_library_schedules, mm_vf, pdca,
    response, room_matrix, score_loadings)
from surgery.weekly_planner import evaluate, loads
from surgery.vf_features import booked_plan_features, feature_names, standardize_features


def save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def record_stage_time(root,stage,elapsed):
    path=Path(root)/"stage_wall_times.json"
    values=json.loads(path.read_text()) if path.exists() else {}
    values[stage]=float(round(elapsed,3))
    save(path,values)
    print(f"{stage}: {elapsed:.1f} wall seconds",flush=True)


def save_case_table(path,data,case_raw,vf_raw,features,response_map,feature_schema="structural"):
    """Per-case replay file; does not run extra optimizer calls."""
    case_raw=np.asarray(case_raw,float)
    vf_raw=np.asarray(vf_raw,float)
    case_reviewed=response_map.reviewed(case_raw)
    vf_reviewed=response_map.reviewed(vf_raw)
    case_display=response_map.display(case_raw)
    vf_display=response_map.display(vf_raw)
    names=feature_names(feature_schema)
    if np.asarray(features).shape!=(len(case_raw),len(names)):
        raise ValueError("Case-table feature count does not match schema")
    fields=["case_id","booked","actual","case_raw","case_display",
            "case_reviewed","vf_raw","vf_display","vf_reviewed"] + [
            "feature_"+v+"_standardized" for v in names]
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(fields)
        for i in range(len(case_raw)):
            writer.writerow([int(data["case_id"][i]),
                             float(data["booked"][i]),float(data["actual"][i]),
                             float(case_raw[i]),float(case_display[i]),
                             float(case_reviewed[i]),float(vf_raw[i]),
                             float(vf_display[i]),float(vf_reviewed[i]),
                             *np.asarray(features[i],float).tolist()])
    tmp.replace(path)
    print(f"Saved {len(case_raw)} per-case records: {path}",flush=True)


def _plan_job(job):
    key,c,d,seconds,tie=job
    from surgery.anchored_weekly_solver import solve_week
    result=solve_week(c,np.asarray(d,float),seconds=seconds,
                      threads=1,tie=tie,hash_seed=0)
    # A time-limited optimizer may return a sound incumbent without a proof.
    # Even with no incumbent, the recorded original assignment is feasible.
    assignment=(list(map(int,result.assignment))
                if result.assignment is not None
                else list(map(int,c.original)))
    cost=evaluate(c,assignment,d)["cost"]
    bound=result.bound
    if bound is not None and np.isfinite(bound):
        bound=max(0.,min(float(bound),float(cost)))
    else:
        bound=0. # Valid, possibly weak dual lower bound; cost is nonnegative.
    room_loads=loads(c,assignment,d)
    return key,dict(assignment=assignment,cost=float(cost),
                    bound=float(bound),complete=bool(result.complete),
                    status=str(result.status),
                    primary_status=str(getattr(result,"primary_status","")),
                    balance_complete=bool(getattr(result,"balance_complete",False)),
                    balance_objective=getattr(result,"balance_objective",None),
                    balance_seconds=float(getattr(result,"balance_seconds",0.)),
                    hash_status=str(getattr(result,"hash_status","")),
                    uniqueness_status=str(getattr(result,"uniqueness_status","")),
                    max_predicted_load=float(room_loads.max()),
                    squared_loads=float(room_loads@room_loads),
                    empty_roster_rooms=int(np.sum(room_loads==0.)))


class PlannerCache:
    """One source-of-truth weekly planner, with feasible timeout handling."""

    def __init__(self,root,cores,seconds):
        self.root=Path(root)/"plans"
        self.cores=max(1,int(cores))
        self.seconds=float(seconds)

    def solve(self,items,durations,tag,tie=False):
        dest=self.root/tag
        dest.mkdir(parents=True,exist_ok=True)
        results={};pending=[]
        for key,c,rows in items:
            d=np.asarray(durations[rows],float)
            keyhash=sha(str(key).encode())
            signature=sha(str(key).encode()+np.ascontiguousarray(d,dtype="<f8").tobytes()+
                          repr((self.seconds,tie,"accuracy_relaxed_retention_v1",
                                float(__import__("surgery.weekly_planner",
                                fromlist=["TURNOVER"]).TURNOVER),
                                bool(__import__("surgery.weekly_planner",
                                fromlist=["RELAX_REQUIRED_DAYS"]).RELAX_REQUIRED_DAYS))).encode())
            path=dest/(keyhash+".json")
            if path.exists():
                entry=json.loads(path.read_text())
                if entry.get("signature")==signature:
                    results[key]=entry["result"]
                    continue
            pending.append(((key,c,d,self.seconds,tie),path,signature))
        print(tag,":",len(results),"cached,",len(pending),"to solve",flush=True)
        started=time.perf_counter()
        done=0
        def show_progress():
            if done % 100==0 or done==len(pending):
                print(f"{tag}: {done}/{len(pending)} new plans "
                      f"({time.perf_counter()-started:.1f}s)",flush=True)
        if pending and self.cores==1:
            for job,path,sig in pending:
                key,entry=_plan_job(job)
                save(path,dict(signature=sig,result=entry))
                results[key]=entry
                done+=1
                show_progress()
        elif pending:
            with ProcessPoolExecutor(max_workers=self.cores,
                    mp_context=get_context("spawn")) as pool:
                futures={pool.submit(_plan_job,job):(path,sig)
                         for job,path,sig in pending}
                for future in as_completed(futures):
                    key,entry=future.result()
                    path,sig=futures[future]
                    save(path,dict(signature=sig,result=entry))
                    results[key]=entry
                    done+=1
                    show_progress()
        coverage=sum(v["complete"] for v in results.values())
        print(tag,": certified",coverage,"/",len(items),flush=True)
        return results


def train_lambda(design,ncomp,*,intercept=False):
    """Same training-only 2.75*1.1 Rademacher penalty rule for each design.

    Unlike the old generic helper, VF has NO intercept: all five policy
    features must contribute to its lambda calibration.
    """
    design=np.asarray(design,float)
    features=design[:,1:] if intercept else design
    rng=np.random.default_rng(20261013)
    maxima=[]
    for _ in range(10):
        signs=rng.choice((-1.,1.),size=(16,len(design)))
        maxima.extend(np.max(np.abs(signs@features),axis=1))
    return float(2.75*1.1*np.quantile(maxima,.9)/ncomp)


def lad_start(X,errors,lam,components,box):
    """Sparse, globally solved regularized LAD LP (not an untested guess)."""
    n,p=X.shape
    X=sparse.csr_matrix(X)
    I=sparse.eye(n,format="csr")
    Z=sparse.csr_matrix((n,p))
    # variables (w[0:p],t[0:n],v[0:p]); t>=|errors-Xw|,
    # v>=|w|; do not regularize intercept.
    E=sparse.eye(p,format="csr")
    empty=sparse.csr_matrix((p,n))
    A=sparse.vstack((
        sparse.hstack((-X,-I,Z)),
        sparse.hstack((X,-I,Z)),
        sparse.hstack((E,empty,-E)),
        sparse.hstack((-E,empty,-E))
        ),format="csr")
    b=np.r_[-errors,errors,np.zeros(2*p)]
    c=np.r_[np.zeros(p),np.full(n,1./components),
             np.r_[0.,np.full(p-1,lam)]]
    result=linprog(c,A_ub=A,b_ub=b,bounds=(
              [(-box,box)]*p+[(0,None)]*(n+p)),
              method="highs",options={"time_limit":900.})
    return (np.asarray(result.x[:p],float) if result.success else None,
            dict(success=bool(result.success),status=int(result.status),
                 message=str(result.message)))


def case_fit(ctx,items,response_map,folder,steps,cuts,backend,radius,*,
             lambda_scale=1.):
    """One LAD initialization, followed by exact-epigraph CASE pDCA."""
    path=folder/"case_policy.json"
    if path.exists():
        saved=json.loads(path.read_text())
        if (saved.get("link","nonmonotone")!=response_map.link or
                not np.isclose(saved.get("alpha",.8),response_map.alpha) or
                not np.isclose(saved.get("h",30.),response_map.h)):
            raise ValueError("CASE checkpoint fitted for another response scenario")
        if not np.isclose(saved.get("lambda_scale",1.),lambda_scale):
            raise ValueError("CASE checkpoint has a different L1 penalty scale")
        print("CASE: reusing fitted coefficients from",path,
              "objective",saved.get("initial_objective"),"->",
              saved.get("objective"),flush=True)
        return np.asarray(saved["weights"],float)
    started=time.perf_counter()
    data=ctx["fit"]
    X=np.asarray(data["X"],float)
    ncomp=len(items)
    if not 0<lambda_scale<=1.:
        raise ValueError('CASE penalty scale must be in (0,1]')
    lam=lambda_scale*train_lambda(X,ncomp,intercept=True)
    penalty=np.ones(X.shape[1]);penalty[0]=0.
    objective=DCObjective(data["booked"],X,np.zeros(len(X)),response_map,
           mode="CASE",actual=data["actual"],components=ncomp,
           lam=lam,penalize=penalty)
    print("CASE: fitting LAD initializer",flush=True)
    lad_started=time.perf_counter()
    start,lad=lad_start(X,np.asarray(data["actual"])-data["booked"],
                        lam,ncomp,radius)
    print(f"CASE: LAD status {lad['status']}; "
          f"{time.perf_counter()-lad_started:.1f}s",flush=True)
    if start is None:
        # No unrelated, inverse-link CASE policy is used as a substitute.
        # The explicit zero fallback keeps the experiment running and is
        # recorded so a failed LAD solve cannot be mistaken for a LAD start.
        print("CASE LAD incomplete; starting pDCA at zero:",lad["message"],
              flush=True)
        start=np.zeros(X.shape[1])
    initial=objective.value(start)
    result=pdca(objective,start,radius=radius,outer=steps,
                max_cuts=cuts,backend=backend,
                progress_label="CASE",
                progress_path=folder/"case_progress.jsonl")
    w=np.asarray(result["weights"],float)
    print("CASE pDCA: initial",round(initial,6),
          "final",round(float(result["objective"]),6),
          "accepted steps",sum(row["accepted"] for row in result["iterations"]),
          "status",result["status"],flush=True)
    save(path,dict(weights=w.tolist(),initial_objective=float(initial),
            objective=float(result["objective"]),lambda_case=float(lam),
            lad=lad,pdca_status=result["status"],pdca_history=result["iterations"],
            alpha=response_map.alpha,h=response_map.h,cap=response_map.cap,
            link=response_map.link,lambda_scale=lambda_scale))
    record_stage_time(folder,"case",time.perf_counter()-started)
    return w


def five_features(ctx,items,cache,stage,scaler=None,plans=None,
                  schema="structural"):
    """Structural features do not need or inspect a provisional schedule."""
    feature_names(schema)  # reject stale historical feature variants
    data=ctx["fit"] if stage.startswith("train") else ctx["held"]
    booked=np.asarray(data["booked"],float)
    history=np.asarray(ctx["H"] if stage.startswith("train") else ctx["H_eval"],float)
    raw=booked_plan_features(items,booked,None,history,schema="structural")
    scaled,params=standardize_features(raw,scaler)
    return scaled,params,None


def work(ctx,root,args):
    train=ctx["train"]
    data=ctx["fit"]
    b=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    response_map=ResponseDC(b,args.alpha,args.h,args.cap,1.,link=args.link)
    root.mkdir(parents=True,exist_ok=True)
    cache=PlannerCache(root,args.cores,args.plan_seconds)
    if args.stage in ("case","all","draft","vf"):
        coef=case_fit(ctx,train,response_map,root,args.case_steps,
                      args.cuts,args.backend,args.case_radius)
        if args.stage=="case":
            return
        draft_started=time.perf_counter()
        case_raw=np.asarray(data["X"],float)@coef
        H,(mu,sd),_=five_features(
            ctx,train,cache,"train",schema=args.vf_features)
        save(root/"vf_features.json",dict(
            schema=args.vf_features,names=list(feature_names(args.vf_features)),
            mean=mu.tolist(),std=sd.tolist()))
        # No provisional BOOKED/CASE planner solves: exact VF queries only
        # the current policy's planner in each direct-MM iteration.
        record_stage_time(root,"draft_and_booked",time.perf_counter()-draft_started)
        if args.stage=="draft":
            return
        # Direct VF does not use a schedule library; caches only reuse identical solves.
        # CASE initializes VF, but all case and plan coefficients remain free.
        X=np.column_stack([data["X"],H])
        # Diagnose duplicate or nearly duplicate columns before optimization.
        Xcase=np.asarray(data["X"],float)
        xc=Xcase-Xcase.mean(axis=0)
        hc=H-H.mean(axis=0)
        xn=np.sqrt(np.square(xc).sum(axis=0))
        hn=np.sqrt(np.square(hc).sum(axis=0))
        denom=hn[:,None]*xn[None,:]
        corr=np.divide(hc.T@xc,denom,out=np.zeros((H.shape[1],Xcase.shape[1])),
                       where=denom>1e-12)
        audit=[]
        for j,name in enumerate(feature_names(args.vf_features)):
            exact=[int(k) for k in range(Xcase.shape[1])
                   if np.allclose(H[:,j],Xcase[:,k],rtol=1e-10,atol=1e-10)]
            candidates=np.argsort(-np.abs(corr[j]))[:3]
            audit.append(dict(feature=name,exact_case_columns=exact,
                    max_abs_case_correlation=float(np.abs(corr[j]).max()),
                    strongest_case_columns=[dict(column=int(k),correlation=float(corr[j,k]))
                                          for k in candidates]))
        save(root/"vf_feature_diagnostics.json",dict(plan_features=audit))
        start=np.r_[coef,np.zeros(H.shape[1])]
        vf_started=time.perf_counter()
        vf_lam=(args.vf_lambda_absolute if args.vf_lambda_absolute is not None
                else args.vf_lambda_scale*train_lambda(X,len(train),intercept=True))
        penalty=np.ones(X.shape[1]);penalty[0]=0.
        penalty[data["X"].shape[1]:]*=args.vf_plan_penalty
        if args.vf_loading=="score":
            selected=choose_library_schedules(
                train,libraries,response_map.reviewed(X@start))
            S,_=room_matrix(train,selected,len(b))
            objective=DCObjective(b,X,np.zeros_like(b),response_map,
                mode="VF",actual=actual,components=len(train),rooms=S)
            penalty[1:]*=score_loadings(objective,start,train)[1:]
        # Accuracy is measured against actual room durations using the
        # PRE-RESPONSE displayed advice, not the reviewed booking.
        ref_display=b+response_map.display(case_raw)
        ref_errors=actual-ref_display
        accuracy_reference=float((1.75*np.maximum(ref_errors,0.)
                        +np.maximum(-ref_errors,0.)).sum()/len(train))
        def oracle(items,d,k):
            return cache.solve(items,d,"vf_outer_"+str(k),tie=True)
        state=mm_vf(train,b,X,response_map,actual,start=start,root=root,
                 lam=vf_lam,
                 iterations=args.vf_outers,radius=args.vf_radius,
                 pdca_steps=args.vf_pdca_steps,max_cuts=args.cuts,
                 backend=args.backend,solve_plans=oracle,penalize=penalty,
                 accuracy_mu=args.accuracy_mu,
                 accuracy_reference=accuracy_reference,
                 accuracy_budget=args.accuracy_budget,
                 require_complete_deployment=False)
        chosen=np.asarray(state["theta"],float)
        vf_raw=X@chosen
        save(root/"selected_policy.json",dict(
               method="JOINT_VF",case_coeff=coef.tolist(),
               vf_theta=chosen.tolist(),
               lambda_case=json.loads(
                    (root/"case_policy.json").read_text())["lambda_case"],
               lambda_vf=float(vf_lam),vf_penalty=penalty.tolist(),
               vf_lambda_scale=args.vf_lambda_scale,
               vf_plan_penalty=args.vf_plan_penalty,
               vf_loading=args.vf_loading,vf_features=args.vf_features,
               vf_feature_names=list(feature_names(args.vf_features)),link=args.link,
               cap=args.cap,alpha=args.alpha,h=args.h,
               train_components=len(train),train_cases=len(b),
               accuracy_kind="displayed_asymmetric",
               accuracy_mu=float(args.accuracy_mu),
               accuracy_budget=float(args.accuracy_budget),
               accuracy_reference=float(accuracy_reference),
               accuracy_limit=float(accuracy_reference*(1+args.accuracy_budget)),
               accuracy_achieved=float((1.75*np.maximum(
                    actual-b-response_map.display(vf_raw),0.)
                    +np.maximum(b+response_map.display(vf_raw)-actual,0.)
                    ).sum()/len(train)),
               training_uses_deployment_ties=True,
               training_incomplete_tie_counts=[
                   int(row["uncertified_planner_instances"])
                   for row in state["history"]],
               relaxed_required_days=True,turnover_minutes=30.,
               vf_status=state["status"]))
        save_case_table(root/"cases_train.csv",data,case_raw,
                        vf_raw,H,response_map,args.vf_features)
        record_stage_time(root,"vf",time.perf_counter()-vf_started)
        return
    if args.stage=="evaluate":
        selected=json.loads((root/"selected_policy.json").read_text())
        eval_2013(ctx,root,args,cache,selected)
        return
    if args.stage=="probe":
        train_probe(ctx,root,args,cache)
        return
    if args.stage=="report":
        report(ctx,root)
        return
    raise ValueError("Unknown stage")


def eval_2013(ctx,root,args,cache,selected):
    evaluated_started=time.perf_counter()
    held=ctx["held"];items=ctx["eval"]
    b=np.asarray(held["booked"],float)
    c=ResponseDC(b,args.alpha,args.h,args.cap,1.,link=args.link)
    coef=np.asarray(selected["case_coeff"],float)
    theta=np.asarray(selected["vf_theta"],float)
    raw=np.asarray(held["X"],float)@coef
    record=json.loads((root/"vf_features.json").read_text())
    scale=np.asarray(record["mean"]),np.asarray(record["std"])
    schema=record.get("schema","baseline")
    if schema!=args.vf_features:
        raise ValueError("Evaluation VF feature schema differs from trained policy")
    H,_,_=five_features(ctx,items,cache,"eval",scaler=scale,schema=schema)
    vf_raw=np.column_stack([held["X"],H])@theta
    # Evaluate the response assumed for THIS configured policy.
    # Alternative true-response scenarios are separate frozen-display audits.
    ds={"BOOKED":b,"CASE":c.reviewed(raw),"VF":c.reviewed(vf_raw)}
    displayed={"BOOKED":b,"CASE":b+c.display(raw),
               "VF":b+c.display(vf_raw)}
    save_case_table(root/"cases_test.csv",held,raw,vf_raw,H,c,schema)
    for label,r in (("CASE",raw),("VF",vf_raw)):
        if not np.allclose(c.reviewed(r),ds[label],atol=1e-8):
            raise AssertionError("Training target differs from deployed response")
    plans={label:cache.solve(items,d,"eval_"+label,tie=True)
           for label,d in ds.items()}
    rows=[]
    for key,comp,idx in items:
        for label,d in ds.items():
            rec=plans[label][key]
            metrics=evaluate(comp,rec["assignment"],comp.actual)
            row=dict(component=key,week=comp.monday,
                     cases=comp.n,method=label,complete=rec["complete"],
                     status=rec["status"],cost=metrics["cost"],
                     idle_minutes=metrics["idle"],
                     overtime_minutes=metrics["overtime"],
                     emptied_historical_surgeon_days=metrics[
                         "emptied_historical_surgeon_days"],
                     max_surgeon_day_minutes=metrics["max_surgeon_case_minutes"],
                     mae=float(np.abs(comp.actual-d[idx]).sum()),
                     displayed_mae=float(np.abs(
                         comp.actual-displayed[label][idx]).sum()),
                     displayed_asymmetric=float((
                         1.75*np.maximum(comp.actual-displayed[label][idx],0.)
                         +np.maximum(displayed[label][idx]-comp.actual,0.)).sum()),
                     reviewed_asymmetric=float((
                         1.75*np.maximum(comp.actual-d[idx],0.)
                         +np.maximum(d[idx]-comp.actual,0.)).sum()))
            rows.append(row)
    save(root/"evaluation_2013.json",dict(
         label="EXPLORATORY_REUSED_2013_DATA",
         entries=rows))
    report(ctx,root)
    record_stage_time(root,"evaluation",time.perf_counter()-evaluated_started)


def train_probe(ctx,root,args,cache):
    """Realized training cost from frozen final displays; no refitting."""
    data=ctx["fit"];items=ctx["train"]
    booked=np.asarray(data["booked"],float)
    selected=json.loads((root/"selected_policy.json").read_text())
    c=ResponseDC(booked,args.alpha,args.h,args.cap,1.,link=args.link)
    scale=json.loads((root/"vf_features.json").read_text())
    original=cache.solve(items,booked,"train_booked",tie=True)
    schema=scale.get("schema","baseline")
    if schema!=args.vf_features:
        raise ValueError("Probe VF feature schema differs from trained policy")
    H,_,_=five_features(ctx,items,cache,"train",
        scaler=(np.asarray(scale["mean"]),np.asarray(scale["std"])),
        plans=original,schema=schema)
    X=np.asarray(data["X"],float)
    raw=np.column_stack([X,H])@np.asarray(selected["vf_theta"],float)
    reviewed=c.reviewed(raw)
    if not np.allclose(reviewed,c.reviewed(raw),atol=1e-8):
        raise AssertionError("Training and deployment durations differ")
    fitted=cache.solve(items,reviewed,"train_final_vf_tie",tie=True)
    result={}
    for name,plans in (("BOOKED",original),("VF",fitted)):
        certified=[(comp,plans[key]) for key,comp,rows in items
                   if plans[key]["complete"]]
        costs=sum(evaluate(comp,rec["assignment"],comp.actual)["cost"]
                  for comp,rec in certified)
        cases=sum(comp.n for comp,_ in certified)
        result[name]=dict(cost_per_case=float(costs/cases) if cases else None,
            certified_components=len(certified),certified_cases=cases)
    save(root/"train_probe.json",result)
    print(json.dumps(result,indent=2),flush=True)


def report(ctx,root):
    file=root/"evaluation_2013.json"
    if not file.exists():
        print("Training saved; 2013 evaluation has not been requested.",flush=True)
        return
    records=json.loads(file.read_text())["entries"]
    by={}
    for row in records:
        by.setdefault(row["component"],{})[row["method"]]=row
    methods=("BOOKED","CASE","VF")
    eligible=[entry for entry in by.values()
              if all(m in entry and entry[m]["complete"] for m in methods)]
    N=sum(entry["BOOKED"]["cases"] for entry in eligible)
    output=dict(status="EXPLORATORY_REUSED_2013_HOLDOUT",
                common_certified_components=len(eligible),
                all_components=len(by),common_certified_cases=N)
    for method in methods:
        output[method+"_uncertified"]=[
            key for key,entry in by.items()
            if method not in entry or not entry[method]["complete"]]
        if N:
            output[method]=dict(
               cost_per_case=sum(entry[method]["cost"] for entry in eligible)/N,
               mae_per_case=sum(entry[method]["mae"] for entry in eligible)/N,
               idle_minutes_per_case=sum(entry[method]["idle_minutes"] for entry in eligible)/N,
               overtime_minutes_per_case=sum(entry[method]["overtime_minutes"] for entry in eligible)/N,
               displayed_mae_per_case=sum(entry[method].get("displayed_mae",0.) for entry in eligible)/N,
               displayed_asymmetric_per_case=sum(entry[method].get("displayed_asymmetric",0.) for entry in eligible)/N,
               reviewed_asymmetric_per_case=sum(entry[method].get("reviewed_asymmetric",0.) for entry in eligible)/N,
               emptied_historical_surgeon_days=sum(entry[method].get("emptied_historical_surgeon_days",0.) for entry in eligible),
               max_surgeon_day_minutes=max((entry[method].get("max_surgeon_day_minutes",0.) for entry in eligible),default=0.))
    if N:
        weekdata={}
        for entry in eligible:
            monday=entry["BOOKED"]["week"]
            wk=weekdata.setdefault(
                 monday,dict(cases=0,costs={m:0. for m in methods}))
            wk["cases"]+=entry["BOOKED"]["cases"]
            for name in methods:
                wk["costs"][name]+=entry[name]["cost"]
        weeks=sorted(weekdata)
        denom=np.array([weekdata[t]["cases"] for t in weeks],float)
        rng=np.random.default_rng(20261022)
        take=rng.integers(0,len(weeks),size=(10000,len(weeks)))
        resamples=denom[take].sum(axis=1)
        differences={}
        for compared,reference in (("VF","CASE"),("CASE","BOOKED"),("VF","BOOKED")):
            vector=np.array([weekdata[t]["costs"][compared]-
                             weekdata[t]["costs"][reference] for t in weeks])
            distribution=vector[take].sum(axis=1)/resamples
            lo,hi=np.quantile(distribution,[.025,.975])
            differences[compared+"_minus_"+reference]=dict(
                  estimate=float(vector.sum()/N),
                  calendar_week_bootstrap_95=[float(lo),float(hi)],
                  better_weeks=int(np.sum(vector<0)),
                  worse_weeks=int(np.sum(vector>0)),
                  equal_weeks=int(np.sum(vector==0)),
                  resamples=10000)
        output["paired_cost_differences"]=differences
    save(root/"summary.json",output)
    print(json.dumps({k:v for k,v in output.items()
                      if not k.endswith("_uncertified")},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("case","draft","vf","all","evaluate","probe","report"),
                    default="all")
    ap.add_argument("--context",type=Path,default=Path(
       "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/nominal_joint_vf_mm"))
    ap.add_argument("--cap",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=80)
    ap.add_argument("--case-radius",type=float,default=100.)
    ap.add_argument("--vf-radius",type=float,default=100.)
    ap.add_argument("--link",choices=("saturating","nonmonotone"),
                    default="saturating")
    ap.add_argument("--alpha",type=float,default=.8)
    ap.add_argument("--h",type=float,default=30.)
    ap.add_argument("--vf-lambda-scale",type=float,default=1.)
    ap.add_argument("--vf-lambda-absolute",type=float,default=None)
    ap.add_argument("--vf-plan-penalty",type=float,default=1.)
    ap.add_argument("--vf-loading",choices=("uniform","score"),default="uniform")
    ap.add_argument("--accuracy-mu",type=float,default=5.,
                    help="Multiplier of displayed recommendation asymmetric loss")
    ap.add_argument("--accuracy-budget",type=float,default=.05,
                    help="Relative training accuracy degradation permitted vs reference")
    ap.add_argument("--vf-features",choices=("structural",),default="structural")
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--backend",choices=("gurobi",),default="gurobi")
    args=ap.parse_args()
    if not (0<args.alpha<1 and args.h>0 and np.isfinite(args.h)):
        ap.error("Response alpha must be in (0,1) and h must be positive")
    if not (np.isfinite(args.vf_lambda_scale) and args.vf_lambda_scale>0.
            and np.isfinite(args.vf_plan_penalty) and args.vf_plan_penalty>0.):
        ap.error("VF penalty multipliers must be positive and finite")
    if (not np.isfinite(args.accuracy_mu) or args.accuracy_mu<0 or
        not np.isfinite(args.accuracy_budget) or args.accuracy_budget<0):
        raise ValueError("Accuracy multiplier/budget must be finite and nonnegative")
    if args.vf_lambda_absolute is not None and (
            not np.isfinite(args.vf_lambda_absolute) or args.vf_lambda_absolute<0.):
        ap.error("Absolute VF lambda must be finite and nonnegative")
    if args.vf_lambda_absolute is not None and args.vf_lambda_scale!=1.:
        ap.error("Choose either VF lambda scale or an absolute lambda, not both")
    if not args.context.exists():
        raise FileNotFoundError("Context checkpoint unavailable: "+str(args.context))
    with args.context.open("rb") as f:
        ctx=pickle.load(f)
    from surgery import nonmonotone_learning as core
    from surgery import weekly_planner, anchored_weekly_solver, vf_features
    fingerprint=sha(b"".join(Path(mod.__file__).read_bytes() for mod in
        (sys.modules[__name__],core,weekly_planner,anchored_weekly_solver,vf_features)))
    settings=dict(context=sha(args.context.read_bytes()),source=fingerprint,
                  turnover=float(__import__("surgery.weekly_planner",fromlist=["TURNOVER"]).TURNOVER),
                  relaxed_required_days=True,
                  vf_training_tie="squared_load_hash",
                  accuracy_kind="displayed_asymmetric",
                  accuracy_mu=float(args.accuracy_mu),
                  accuracy_budget=float(args.accuracy_budget),
                  alpha=args.alpha,h=args.h,cap=args.cap,case_steps=args.case_steps,
                  vf_outers=args.vf_outers,vf_pdca_steps=args.vf_pdca_steps,
                  cuts=args.cuts,case_radius=args.case_radius,
                  vf_radius=args.vf_radius,link=args.link,
                  vf_lambda_scale=args.vf_lambda_scale,
                  vf_plan_penalty=args.vf_plan_penalty,vf_loading=args.vf_loading,
                  plan_seconds=args.plan_seconds,backend=args.backend)
    # Absent defaults retain compatibility with prior tuning checkpoints.
    settings["vf_features"]="structural"
    if args.vf_lambda_absolute is not None:
        settings["vf_lambda_absolute"]=args.vf_lambda_absolute
    args.out.mkdir(parents=True,exist_ok=True)
    contract=args.out/"run_contract.json"
    if contract.exists():
        previous=json.loads(contract.read_text())
        scientific={k:v for k,v in settings.items() if k!="source"}
        earlier={k:v for k,v in previous.items()
                 if k not in ("source","source_history")}
        if scientific!=earlier:
            raise ValueError("Scientific settings changed; choose a new --out directory")
        if previous.get("source")!=settings["source"]:
            history=previous.get("source_history",[])
            if previous.get("source") is not None:
                history.append(previous["source"])
            settings["source_history"]=history
            save(contract,settings)
            print("Continuing completed CASE checkpoint after code-only "
                  "speed/reporting revision (source history saved)",flush=True)
    else:
        save(contract,settings)
    print("UHN joint VF:",len(ctx["train"]),"train components;",
          len(ctx["eval"]),"evaluation components; nominal response only",
          flush=True)
    work(ctx,args.out,args)


if __name__=="__main__":
    main()
