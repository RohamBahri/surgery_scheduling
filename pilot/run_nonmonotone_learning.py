#!/usr/bin/env python3
"""Reproducible UHN CASE -> provisional plan -> VF library-MM study.

ONE response: alpha=.8, h=30. THREE arms: BOOKED, CASE, VF.
No robustness sweep, donor simulation, synthetic profiles, or old search.
"""
from __future__ import annotations

import argparse
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
    ResponseDC, DCObjective, C_OVER, C_UNDER, mm_vf, pdca, response)
from surgery.weekly_planner import evaluate, loads


def save(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


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
    return key,dict(assignment=assignment,cost=float(cost),
                    bound=float(bound),complete=bool(result.complete),
                    status=str(result.status))


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
                          repr((self.seconds,tie)).encode())
            path=dest/(keyhash+".json")
            if path.exists():
                entry=json.loads(path.read_text())
                if entry.get("signature")==signature:
                    results[key]=entry["result"]
                    continue
            pending.append(((key,c,d,self.seconds,tie),path,signature))
        print(tag,":",len(results),"cached,",len(pending),"to solve",flush=True)
        if pending and self.cores==1:
            for job,path,sig in pending:
                key,entry=_plan_job(job)
                save(path,dict(signature=sig,result=entry))
                results[key]=entry
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


def case_fit(ctx,items,response_map,folder,steps,cuts,backend,radius):
    path=folder/"case_policy.json"
    if path.exists():
        rec=json.loads(path.read_text())
        return np.asarray(rec["weights"],float)
    data=ctx["fit"]
    X=np.asarray(data["X"],float)
    n=len(X)
    ncomp=len(items)
    lam=train_lambda(X,ncomp,intercept=True)
    penalty=np.ones(X.shape[1]);penalty[0]=0.
    obj=DCObjective(data["booked"],X,np.zeros(n),response_map,
          mode="CASE",actual=data["actual"],components=ncomp,
          lam=lam,penalize=penalty)
    # Existing response-aware CASE fit serves as one warm start, not as
    # the new model's fitted weights (its historical link was inverse).
    old=np.asarray(data["case_w"],float)/response_map.alpha
    old=np.clip(old,-radius,radius)
    starts=[("old_inverse_case",old)]
    lad,diagnostic=lad_start(X,np.asarray(data["actual"])-data["booked"],
                            lam,ncomp,radius)
    if lad is not None:
        starts.insert(0,("regularized_LAD",lad))
    logs=[]
    best=(float("inf"),None)
    for name,start in starts:
        rec=pdca(obj,start,radius=radius,outer=steps,max_cuts=cuts,
                 backend=backend)
        weight=np.asarray(rec["weights"],float)
        score=obj.value(weight)
        logs.append(dict(start=name,objective=score,
                         status=rec["status"],history=rec["iterations"]))
        if score<best[0]:
            best=score,weight
    w=best[1]
    save(path,dict(weights=w.tolist(),objective=float(best[0]),
                   lambda_case=lam,initialization=diagnostic,
                   starts=logs,alpha=response_map.alpha,h=response_map.h,
                   cap=response_map.cap,uses_true_nonmonotone_response=True))
    return w


def five_features(ctx,items,case_raw,response_map,cache,stage,scaler=None):
    """ONE frozen CASE draft per physical-room component, hash seed zero."""
    data=ctx["fit"] if stage=="train" else ctx["held"]
    n=len(data["booked"])
    baseline=response_map.reviewed(case_raw)
    draft=cache.solve(items,baseline,stage+"_case_draft",tie=True)
    history=np.asarray(ctx["H"] if stage=="train" else ctx["H_eval"],float)
    # The first two columns retain the original surgeon/procedure history
    # scores; old CASE congestion columns are explicitly discarded.
    s,q=history[:,0],history[:,1]
    p=np.empty(n)
    for key,c,rows in items:
        z=np.asarray(draft[key]["assignment"],int)
        p[rows]=np.clip((loads(c,z,baseline[rows])[z]-480.)/120.,-2.,2.)
    raw=np.column_stack([s,q,p,s*p,q*p])
    if scaler is None:
        mu=raw.mean(axis=0)
        sd=raw.std(axis=0)
        sd=np.maximum(sd,1e-12)
    else:
        mu,sd=scaler
    return (raw-mu)/sd,(mu,sd),draft


def work(ctx,root,args):
    train=ctx["train"]
    data=ctx["fit"]
    b=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    response_map=ResponseDC(b,.8,30.,args.cap,1.)
    root.mkdir(parents=True,exist_ok=True)
    cache=PlannerCache(root,args.cores,args.plan_seconds)
    if args.stage in ("case","all","draft","vf"):
        coef=case_fit(ctx,train,response_map,root,args.case_steps,
                      args.cuts,args.backend,args.case_radius)
        if args.stage=="case":
            return
        case_raw=np.asarray(data["X"],float)@coef
        H,(mu,sd),case_draft=five_features(
                    ctx,train,case_raw,response_map,cache,"train")
        save(root/"vf_features.json",dict(mean=mu.tolist(),std=sd.tolist()))
        booked=cache.solve(train,b,"train_booked",tie=True)
        if args.stage=="draft":
            return
        libraries={key:[entry["assignment"],case_draft[key]["assignment"],
                        list(map(int,c.original))]
                   for key,c,_ in train
                   for entry in [booked[key]]}
        # Feasible schedule certificates are retained on timeouts instead
        # of promoting an unfinished planner solve to "exact".
        vf_lam=train_lambda(H,len(train))
        def oracle(items,d,k):
            return cache.solve(items,d,"vf_outer_"+str(k),tie=False)
        state=mm_vf(train,b,H,case_raw,response_map,actual,root=root,
                 lam=vf_lam,initial_libraries=libraries,
                 iterations=args.vf_outers,radius=args.vf_radius,
                 pdca_steps=args.vf_pdca_steps,max_cuts=args.cuts,
                 backend=args.backend,solve_plans=oracle)
        chosen=np.asarray(state["theta"],float)
        save(root/"selected_policy.json",dict(method="CASE+VF",
               case_coeff=coef.tolist(),vf_theta=chosen.tolist(),
               lambda_case=json.loads((root/"case_policy.json").read_text())["lambda_case"],
               lambda_vf=vf_lam,cap=args.cap,alpha=.8,h=30.,
               train_components=len(train),train_cases=len(b),
               certificate_history=state["history"],
               vf_status=state["status"]))
        return
    if args.stage=="evaluate":
        selected=json.loads((root/"selected_policy.json").read_text())
        eval_2013(ctx,root,args,cache,selected)
        return
    if args.stage=="report":
        report(ctx,root)
        return
    raise ValueError("Unknown stage")


def eval_2013(ctx,root,args,cache,selected):
    held=ctx["held"];items=ctx["eval"]
    b=np.asarray(held["booked"],float)
    c=ResponseDC(b,.8,30.,args.cap,1.)
    coef=np.asarray(selected["case_coeff"],float)
    theta=np.asarray(selected["vf_theta"],float)
    raw=np.asarray(held["X"],float)@coef
    record=json.loads((root/"vf_features.json").read_text())
    scale=np.asarray(record["mean"]),np.asarray(record["std"])
    H,_,_=five_features(ctx,items,raw,c,cache,"eval",scaler=scale)
    ds={"BOOKED":b,"CASE":c.reviewed(raw),"VF":c.reviewed(raw+H@theta)}
    # Verify deployment's smallest display EXACTLY preserves reviewed booking.
    for label,r in (("CASE",raw),("VF",raw+H@theta)):
        displays=c.display(r)
        if not np.allclose(b+response(displays,.8,30.),ds[label],atol=1e-8):
            raise AssertionError("Non-monotone training/deployment mismatch")
    plans={label:cache.solve(items,d,"eval_"+label,tie=True)
           for label,d in ds.items()}
    rows=[]
    for key,comp,idx in items:
        for label,d in ds.items():
            rec=plans[label][key]
            row=dict(component=key,site=comp.group,week=comp.monday,
                     cases=comp.n,method=label,complete=rec["complete"],
                     status=rec["status"],
                     cost=evaluate(comp,rec["assignment"],comp.actual)["cost"],
                     mae=float(np.abs(comp.actual-d[idx]).sum()),
                     assignment=rec["assignment"])
            rows.append(row)
    save(root/"evaluation_2013.json",dict(
         label="EXPLORATORY_REUSED_2013_DATA",
         entries=rows))
    report(ctx,root)


def report(ctx,root):
    file=root/"evaluation_2013.json"
    if not file.exists():
        print("No evaluation yet. Training outputs in",root,flush=True)
        return
    records=json.loads(file.read_text())["entries"]
    by={}
    for r in records:
        by.setdefault(r["component"],{})[r["method"]]=r
    methods=("BOOKED","CASE","VF")
    eligible=[x for x in by.values() if
              all(m in x and x[m]["complete"] for m in methods)]
    N=sum(x["BOOKED"]["cases"] for x in eligible)
    summary=dict(status="EXPLORATORY_REUSED_HOLDOUT",
                 common_certified_components=len(eligible),
                 all_components=len(by),common_certified_cases=N)
    for m in methods:
        if N:
            summary[m]=dict(cost_per_case=sum(x[m]["cost"] for x in eligible)/N,
                            mae_per_case=sum(x[m]["mae"] for x in eligible)/N)
        summary[m+"_uncertified"]=[
            key for key,x in by.items() if m not in x or not x[m]["complete"]]
    if N:
        summary["VF_minus_CASE"]=summary["VF"]["cost_per_case"]-summary["CASE"]["cost_per_case"]
        summary["CASE_minus_BOOKED"]=summary["CASE"]["cost_per_case"]-summary["BOOKED"]["cost_per_case"]
        summary["VF_changed_assignment_fraction"]=sum(
             x["CASE"]["assignment"]!=x["VF"]["assignment"]
             for x in eligible)/len(eligible)
    save(root/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items()
                      if not k.endswith("_uncertified")},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("case","draft","vf","all","evaluate","report"),
                    default="all")
    ap.add_argument("--context",type=Path,default=Path(
       "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/nominal_case_vf_mm"))
    ap.add_argument("--cap",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=80)
    ap.add_argument("--case-radius",type=float,default=100.)
    ap.add_argument("--vf-radius",type=float,default=64.)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--backend",choices=("gurobi","scipy"),default="gurobi")
    args=ap.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Context checkpoint unavailable: "+str(args.context))
    with args.context.open("rb") as f:
        ctx=pickle.load(f)
    from surgery import nonmonotone_learning as core
    fingerprint=sha(Path(__file__).read_bytes()+Path(core.__file__).read_bytes())
    settings=dict(context=sha(args.context.read_bytes()),source=fingerprint,
                  alpha=.8,h=30.,cap=args.cap,case_steps=args.case_steps,
                  vf_outers=args.vf_outers,vf_pdca_steps=args.vf_pdca_steps,
                  cuts=args.cuts,case_radius=args.case_radius,
                  vf_radius=args.vf_radius,
                  plan_seconds=args.plan_seconds,backend=args.backend)
    args.out.mkdir(parents=True,exist_ok=True)
    contract=args.out/"run_contract.json"
    if contract.exists():
        if json.loads(contract.read_text())!=settings:
            raise ValueError("Scientific settings changed; choose a new --out directory")
    else:
        save(contract,settings)
    print("UHN CASE/VF:",len(ctx["train"]),"train components;",
          len(ctx["eval"]),"evaluation components; nominal response only",
          flush=True)
    work(ctx,args.out,args)


if __name__=="__main__":
    main()
