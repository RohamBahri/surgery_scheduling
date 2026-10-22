#!/usr/bin/env python3
"""Frozen-VF mechanism, robustness, ablation and conditional-transfer campaign.

Separate artifacts; never edits the baseline run's context, policies, or shards.
Run with run_vf_followups.sh. Each stage is independently resumable.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ["SURGERY_HASH_SEEDS"]="0"
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"

from surgery.advice_comparison import ALPHA, H, display, scenarios, reviewed
from surgery.theory_exact import KAPPA, reduced, expected_reduced, stop_loss
from surgery.weekly_planner import evaluate
from surgery.objective_selection import train_objective

ROOT_NAME="vf_followups"
STAGES=(("diagnose",900),("response",10800),("tie",5400),
        ("train",9000),("evaluate_variants",7200),
        ("duration",600),("report",600))
POLICIES=("CASE","STAT","VF","SHARED")
RESPONSE_SCENARIOS=(
    ("nominal",.8,30.),("weak",.4,30.),("very_weak",.25,15.),
    ("moderate",.6,30.),("full_adoption",1.,30.),
    ("tight_cap",.8,12.),("heterogeneous",None,None))
VARIANTS={
    "congestion_only":dict(active_features=(2,),coefficient_radius=16.),
    "without_congestion":dict(active_features=(0,1),coefficient_radius=16.),
    "without_interactions":dict(active_features=(0,1,2),coefficient_radius=16.),
    "wider_32":dict(active_features=(0,1,2,3,4),coefficient_radius=32.),
    "observed_outcomes":dict(active_features=(0,1,2,3,4),
                             coefficient_radius=16.,scenario_mode="observed"),
    "leave_out_TGH":dict(active_features=(0,1,2,3,4),coefficient_radius=16.),
    "leave_out_TWH_main":dict(active_features=(0,1,2,3,4),coefficient_radius=16.),
    "leave_out_TWH_day":dict(active_features=(0,1,2,3,4),coefficient_radius=16.)}
SITES={"TGH":"TGH","TWH_main":"TWH-main","TWH_day":"TWH-day"}

def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    tmp.replace(path)

def load(path):
    return json.loads(Path(path).read_text())

def folder(args):
    return args.out/ROOT_NAME

def original(args):
    return args.artifact_root/"objective_selection_2013"

def context(args):
    path=original(args)/"context.pkl"
    with path.open("rb") as f:ctx=pickle.load(f)
    if not Path(ctx["root"]).exists():
        raise RuntimeError("Original context root not found; use the original artifact directory")
    if len(ctx["eval"])!=1125 or len(ctx["train"])!=3362:
        raise AssertionError("Unexpected cohort; follow-up study is not the preregistered 2013 split")
    return ctx

def frozen(args,ctx):
    policies=load(original(args)/"frozen.json")["policy_records"]
    return {"STAT":np.asarray(ctx["beta"],float),
            **{k:np.asarray(policies[k]["coeff"],float) for k in ("VF","SHARED")}}

def record(ctx,ck,arm):
    p=Path(ctx["root"])/"full_evaluation"/"per_arm"/(ck+"__"+arm+".json")
    return load(p) if p.exists() else {"status":"NOT_EVALUATED"}

def expected_durations(ctx,c,rows,coeff,arm):
    from pilot import run_theory_full as full
    base=ctx["held"]["X"]@ctx["held"]["case_w"]
    d0=full.reviewed(c.booked,base[rows])
    if arm=="CASE":return d0
    return full.reviewed_policy(c,rows,ctx["H_eval"],d0,coeff)

def criterion_slack(z,d,bank):
    """Unregularized fixed-plan slack: U-g=D+H, no oracle needed."""
    from surgery.theory_exact import CAP
    comp,assignment=z
    g=expected_reduced(comp,assignment,bank)
    R=expected_reduced(comp,assignment,np.sort(bank,axis=0))
    U=reduced(comp,assignment,d)+KAPPA*stop_loss(d,bank)
    return dict(g=float(g),R=float(R),U=float(U),
                dependence_gap=float(R-g),split_gap=float(U-R),
                upper_slack=float(U-g))

def diagnose(args):
    ctx=context(args);dst=folder(args)/"diagnose";dst.mkdir(parents=True,exist_ok=True)
    base=ctx["held"]["X"]@ctx["held"]["case_w"]
    from pilot import run_theory_full as full
    count=0
    for ck,c,rows in ctx["eval"]:
        path=dst/(ck+".json")
        if path.exists():continue
        d0=full.reviewed(c.booked,base[rows])
        bank=scenarios(c,d0,ctx["pools"],20261022,count=200)
        info=dict(site=c.group,monday=c.monday,n=c.n,methods={})
        for arm in POLICIES:
            rec=record(ctx,ck,arm)
            if rec.get("status")!="COMPLETE":continue
            ass=rec["assignments"]["0"]
            d=np.asarray(rec["duration"],float)
            s=criterion_slack((c,ass),d,bank)
            s["regularizer"]=float(.001/2*np.square(d-d0).sum())
            s["note"]="U is at deployed z; eta needs certified primary Q and is not assumed zero"
            info["methods"][arm]=s
        save(path,info);count+=1
    save(dst/"status.json",dict(new_components=count,source_components=len(ctx["eval"]),
        note="Dependence is comonotonic under estimated case marginals; not observationally identified."))

def stable_fraction(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8],"big")/2**64

def response_case(c,booked,nominal,scenario):
    """Keep the display fixed; vary only the realized response mapping."""
    from surgery.planner import response
    name,alpha,h=scenario
    u=display(booked,nominal)
    if name!="heterogeneous":
        d=np.maximum(1.,booked+response(u,alpha,h))
    else:
        # Synthetic profiles, NOT fitted advice-adoption parameters.
        a=np.asarray([.30+.65*stable_fraction(c.surgeon_ids[int(s)])
                      for s in c.surgeon],float)
        cap=np.asarray([15.+45.*stable_fraction("cap:"+c.surgeon_ids[int(s)])
                        for s in c.surgeon],float)
        effective=np.sign(u)*np.minimum(a*np.abs(u),
                            np.maximum(cap-(1.-a)*np.abs(u),0.))
        d=np.maximum(1.,booked+effective)
    return d

def _response_task(job):
    ck,c,nominal,scenario,solve_seconds=job
    from surgery.anchored_weekly_solver import solve_week
    result=dict(component=ck,site=c.group,monday=c.monday,n=c.n,
                scenario=scenario[0],method={})
    for arm in ("STAT","VF"):
        try:
            d=response_case(c,c.booked,np.asarray(nominal[arm],float),scenario)
            ans=solve_week(c,d,seconds=solve_seconds,threads=1,tie=True,hash_seed=0)
            if not ans.complete or ans.assignment is None:
                result["method"][arm]=dict(status="UNRESOLVED",solver=ans.status)
            else:
                result["method"][arm]=dict(status="COMPLETE",
                    cost=float(evaluate(c,ans.assignment,c.actual)["cost"]),
                    assignment=list(map(int,ans.assignment)),
                    mean_effective_change=float(np.mean(d-c.booked)))
        except Exception as exc:
            result["method"][arm]=dict(status="ERROR",error=repr(exc))
    return result

def run_tasks(items,fn,dst,cores):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    def completed(name):
        path=dst/(name+".json")
        if not path.exists():return False
        old=load(path)
        if "method" in old and isinstance(old["method"],dict):
            return all(x.get("status")=="COMPLETE" for x in old["method"].values())
        if "entries" in old:
            return all(x.get("status")=="CERTIFIED" for x in old["entries"].values())
        return old.get("status")=="COMPLETE"
    todo=[(name,item) for name,item in items if not completed(name)]
    if not todo:return
    with ProcessPoolExecutor(max_workers=min(cores,len(todo)),
                             mp_context=get_context("spawn")) as pool:
        it=iter(todo);active={}
        def submit():
            try:name,item=next(it)
            except StopIteration:return False
            future=pool.submit(fn,item);active[future]=name
            return True
        for _ in range(min(cores,len(todo))):submit()
        processed=0
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for fut in done:
                name=active.pop(fut)
                try:res=fut.result()
                except Exception as exc:res=dict(status="WORKER_ERROR",error=repr(exc))
                save(dst/(name+".json"),res)
                processed+=1
                if processed%100==0:print(dst.name,processed,len(todo),flush=True)
                submit()

def response(args):
    ctx=context(args);coeff=frozen(args,ctx);dst=folder(args)/"response"
    from pilot import run_theory_full as full
    base=ctx["held"]["X"]@ctx["held"]["case_w"]
    tasks=[]
    for ck,c,rows in ctx["eval"]:
        d0=full.reviewed(c.booked,base[rows])
        nominal={m:full.reviewed_policy(c,rows,ctx["H_eval"],d0,coeff[m]).tolist()
                 for m in ("STAT","VF")}
        for name,a,h in RESPONSE_SCENARIOS:
            t=120. if c.n>=25 else 45.
            tasks.append((ck+"__"+name,(ck,c,nominal,(name,a,h),t)))
    run_tasks(tasks,_response_task,dst,args.cores)

def tie_bracket(job):
    """Min/max expected feasible schedule cost inside the fixed predicted-cost tie set."""
    ck,c,d,bank,seconds,*optional=job
    deployed=optional[0] if optional else None
    from surgery.anchored_weekly_solver import _model,solve_week
    from surgery.weekly_planner import _cost,_objective_tolerance,_assignment
    from surgery.theory_exact import TURN,KAPPA,CAP
    import gurobipy as gp
    from gurobipy import GRB
    t0=time.monotonic()
    opt=solve_week(c,d,seconds=min(90.,seconds/3),threads=1,tie=False)
    if not opt.complete:return dict(status="PRIMARY_UNCERTIFIED",component=ck)
    Q=float(opt.value)
    m,x,used,byroom=_model(c,seconds,1)
    try:
        p=_cost(m,x,used,byroom,d,exact=False,name="tie_pred")
        m.addConstr(p<=Q+_objective_tolerance(Q))
        zero=m.addVar(lb=0.,ub=0.)
        value=TURN*used.sum()
        for s,a in enumerate(bank):
            for j,cases in enumerate(byroom):
                expr=gp.quicksum((float(a[i])+TURN)*x[i,j] for i in cases)-TURN*used[j]-CAP
                excess=m.addVar(lb=-CAP,name=f"tie_excess_{s}_{j}")
                m.addConstr(excess==expr)
                overtime=m.addVar(lb=0.,name=f"tie_ot_{s}_{j}")
                m.addGenConstrMax(overtime,[excess,zero])
                value+=(KAPPA/len(bank))*overtime
        eta=(None if deployed is None else
             float(evaluate(c,deployed,d)["cost"]-Q))
        if eta is not None and eta < -1e-4:
            raise AssertionError("Selected hash plan beat certified primary optimum")
        output=dict(status="COMPLETE",component=ck,site=c.group,n=c.n,
                    primary=Q,planner_tolerance_eta=eta,entries={})
        for mode in ("best","worst"):
            time_left=seconds-(time.monotonic()-t0)
            if time_left<1:
                output["entries"][mode]=dict(status="BUDGET_EXHAUSTED");continue
            m.Params.TimeLimit=min(time_left,75.)
            m.setObjective(value,GRB.MINIMIZE if mode=="best" else GRB.MAXIMIZE)
            m.optimize()
            if not m.SolCount:
                output["entries"][mode]=dict(status="NO_INCUMBENT")
            else:
                z=list(map(int,_assignment(c,x)))
                output["entries"][mode]=dict(
                    status="CERTIFIED" if m.Status==GRB.OPTIMAL else "FEASIBLE_UNCERTIFIED",
                    cost=float(expected_reduced(c,z,bank)),
                    realized=float(evaluate(c,z,c.actual)["cost"]),
                    assignment=z)
        return output
    finally:m.dispose()

def tie(args):
    ctx=context(args);dst=folder(args)/"tie"
    # Outcome-blind selection: 16 largest + 48 with VF/STAT distinct plans.
    ev=sorted(ctx["eval"],key=lambda x:-x[1].n)
    largest=ev[:16]
    altered=[v for v in ev if record(ctx,v[0],"VF").get("assignments") !=
                                  record(ctx,v[0],"STAT").get("assignments")]
    sampled=sorted(altered,key=lambda v:stable_fraction(v[0]))[:48]
    chosen={v[0]:v for v in largest+sampled}
    tasks=[]
    from pilot import run_theory_full as full
    base=ctx["held"]["X"]@ctx["held"]["case_w"]
    for ck,c,rows in chosen.values():
        d0=full.reviewed(c.booked,base[rows])
        bank=scenarios(c,d0,ctx["pools"],20261022,count=25)
        for arm in ("STAT","VF"):
            rec=record(ctx,ck,arm)
            if rec.get("status")!="COMPLETE":continue
            d=np.asarray(rec["duration"],float)
            tasks.append((ck+"__"+arm,(ck,c,d,bank,160.,
                          list(map(int,rec["assignments"]["0"])))))
    run_tasks(tasks,tie_bracket,dst,args.cores)
    save(dst/"selection.json",dict(components=len(chosen),tasks=len(tasks),
                rule="16 largest and 48 random changed-plan components; no outcome selection",
                scenario_bank_size=25,not_full_cohort=True))

def train(args):
    ctx=context(args);dst=folder(args)/"train";dst.mkdir(parents=True,exist_ok=True)
    # Never mutate original policy artifacts or original optimization cache.
    local={**ctx,"root":folder(args)}
    for name,cfg in VARIANTS.items():
        output=dst/name
        if (output/"trained.json").exists():continue
        trial={**local}
        if name.startswith("leave_out_"):
            site=SITES[name.removeprefix("leave_out_")]
            trial["train"]=[v for v in ctx["train"] if v[1].group!=site]
        out=train_objective(trial,"VF",{},output,cores=args.cores,
                seconds=1000. if not name.startswith("leave_out_") else 800.,
                solve_seconds=45.,task_seconds=150.,rounds=6,
                library_enrichment=False,**cfg)
        print("trained",name,out["status"],out["theta"],flush=True)

def _variant_task(job):
    ck,c,policy,coeff,base,H,rows,seconds=job
    from pilot import run_theory_full as full
    from surgery.anchored_weekly_solver import solve_week
    try:
        d0=full.reviewed(c.booked,base[rows])
        d=full.reviewed_policy(c,rows,H,d0,coeff)
        sol=solve_week(c,d,seconds=seconds,threads=1,tie=True,hash_seed=0)
        if not sol.complete or sol.assignment is None:
            return dict(status="UNRESOLVED",solver=sol.status,method=policy,n=c.n,
                        site=c.group,monday=c.monday)
        return dict(status="COMPLETE",method=policy,n=c.n,site=c.group,
                    monday=c.monday,cost=float(evaluate(c,sol.assignment,c.actual)["cost"]),
                    assignment=list(map(int,sol.assignment)))
    except Exception as exc:return dict(status="ERROR",error=repr(exc),method=policy,n=c.n)

def evaluate_variants(args):
    ctx=context(args);dst=folder(args)/"evaluate_variants"
    base=ctx["held"]["X"]@ctx["held"]["case_w"];H=ctx["H_eval"]
    items=[]
    for name in VARIANTS:
        f=folder(args)/"train"/name/"trained.json"
        if not f.exists():continue
        coeff=np.asarray(load(f)["coeff"],float)
        for ck,c,rows in ctx["eval"]:
            if name.startswith("leave_out_") and c.group!=SITES[name.removeprefix("leave_out_")]:
                continue
            items.append((ck+"__"+name,(ck,c,name,coeff,base,H,rows,
                                        120. if c.n>=25 else 45.)))
    run_tasks(items,_variant_task,dst,args.cores)

def duration(args):
    """Out-of-period surgeon-level log-error random effects; NOT response adoption."""
    ctx=context(args);dst=folder(args)/"duration"
    pools=ctx["pools"]
    values=np.concatenate([np.asarray(v[:,2],float) for v in pools.values()])
    global_mu=float(values.mean());global_var=max(1e-5,float(values.var()))
    # Fixed predeclared shrinkage strength; avoids tuning on 2013.
    k=20.
    stats={}
    for surgeon,v in pools.items():
        e=np.asarray(v[:,2],float);n=len(e)
        w=n/(n+k)
        mu=w*e.mean()+(1-w)*global_mu
        var=max(1e-5,w*np.var(e)+(1-w)*global_var+
                w*(1-w)*(e.mean()-global_mu)**2)
        stats[str(surgeon)]=(float(mu),float(var),int(n))
    from pilot import run_theory_full as full
    base=ctx["held"]["X"]@ctx["held"]["case_w"]
    sums={name:dict(n=0,nll=0.,covered=0) for name in ("global","surgeon","shrunk")}
    for ck,c,rows in ctx["eval"]:
        d0=full.reviewed(c.booked,base[rows])
        y=np.log(np.maximum(1.,c.actual)/d0)
        for i,s in enumerate(c.surgeon):
            surgeon=str(c.surgeon_ids[int(s)])
            rec=stats.get(surgeon,(global_mu,global_var,0))
            donor=pools.get(surgeon)
            if donor is None:mu_s,var_s=global_mu,global_var
            else:
                a=np.asarray(donor[:,2],float)
                mu_s=float(a.mean());var_s=max(1e-5,float(a.var()))
            for name,(mu,var) in (("global",(global_mu,global_var)),
                                  ("surgeon",(mu_s,var_s)),
                                  ("shrunk",(rec[0],rec[1]))):
                r=sums[name];r["n"]+=1
                r["nll"]+=.5*(np.log(2*np.pi*var)+(float(y[i])-mu)**2/var)
                r["covered"]+=int(abs(float(y[i])-mu)<=1.645*np.sqrt(var))
    for r in sums.values():
        r["mean_nll"]=r["nll"]/r["n"];r["coverage_90"]=r["covered"]/r["n"]
    save(dst/"summary.json",dict(scores=sums,shrinkage_cases=k,
        surgeons_in_history=len(stats),global_mean=global_mu,
        caveat="Historical duration variation is identifiable; advice adoption alpha_s,h_s is not."))

def summarize(args):
    dst=folder(args);ctx=context(args);denom=sum(c.n for _,c,_ in ctx["eval"])
    output=dict(total_2013_cases=denom,stages={},
         note="Exploratory retrospective follow-up; keep original VF method selection frozen.")
    # Non-imputed case-weighted differences; only matched complete comparisons count.
    for stage in ("diagnose","response","tie","train","evaluate_variants","duration"):
        p=dst/stage
        if not p.exists():output["stages"][stage]=dict(status="NOT_RUN");continue
        files=[f for f in p.glob("*.json") if f.name not in
                 ("status.json","selection.json","summary.json")]
        output["stages"][stage]=dict(artifact_files=len(files))
    def paired_from_stage(stage,variants,baseline_arm="STAT"):
        results={}
        for key in variants:
            pairs=[]
            target_site=(SITES[key.removeprefix("leave_out_")]
                         if stage=="evaluate_variants" and key.startswith("leave_out_")
                         else None)
            target_cases=sum(c.n for _,c,_ in ctx["eval"]
                             if target_site is None or c.group==target_site)
            for ck,c,_ in ctx["eval"]:
                if target_site is not None and c.group!=target_site:continue
                if stage=="response":
                    p=dst/stage/(ck+"__"+key+".json")
                    if not p.exists():continue
                    data=load(p);ms=data.get("method",{})
                    if any(ms.get(x,{}).get("status")!="COMPLETE" for x in ("STAT","VF")):
                        continue
                    diff=ms["VF"]["cost"]-ms["STAT"]["cost"]
                else:
                    p=dst/stage/(ck+"__"+key+".json")
                    if not p.exists():continue
                    data=load(p)
                    baseline=record(ctx,ck,baseline_arm)
                    if data.get("status")!="COMPLETE" or baseline.get("status")!="COMPLETE":
                        continue
                    diff=float(data["cost"])-float(baseline["realized"]["0"])
                pairs.append((c.monday,int(c.n),float(diff)))
            n=sum(a for _,a,_ in pairs)
            result=dict(certified_cases=n,target_cases=target_cases,
                        coverage=n/target_cases if target_cases else 0.,
                        paired_change_per_case=None if not n else
                          sum(v for _,_,v in pairs)/n,
                        status="COMPLETE" if n==target_cases else "INCOMPLETE")
            if n and len(set(m for m,_,_ in pairs))>=8:
                weeks=sorted({m for m,_,_ in pairs})
                indexed={w:[a for a in pairs if a[0]==w] for w in weeks}
                vals=np.asarray([sum(v for _,_,v in indexed[w]) for w in weeks])
                counts=np.asarray([sum(n for _,n,_ in indexed[w]) for w in weeks])
                draw=np.random.default_rng(20261022).integers(len(weeks),size=(2000,len(weeks)))
                bootstrap=vals[draw].sum(axis=1)/counts[draw].sum(axis=1)
                result["week_cluster_bootstrap_95"]=[float(x) for x in np.quantile(bootstrap,[.025,.975])]
            results[key]=result
        return results
    output["response_sensitivity"]=paired_from_stage("response",[x[0] for x in RESPONSE_SCENARIOS])
    output["vf_variants"]=paired_from_stage("evaluate_variants",list(VARIANTS))
    output["vf_variants_vs_frozen_VF"]=paired_from_stage(
        "evaluate_variants",list(VARIANTS),baseline_arm="VF")
    d=dst/"duration"/"summary.json"
    if d.exists():output["duration_heterogeneity"]=load(d)
    diag=dst/"diagnose"
    slacks=[]
    if diag.exists():
        for path in diag.glob("*.json"):
            if path.name=="status.json":continue
            rec=load(path)
            ms=rec.get("methods",{})
            if "VF" in ms and "STAT" in ms:
                slacks.append((rec["n"],ms["VF"]["upper_slack"]-ms["STAT"]["upper_slack"],
                               ms["VF"]["dependence_gap"]-ms["STAT"]["dependence_gap"],
                               ms["VF"]["split_gap"]-ms["STAT"]["split_gap"]))
    if slacks:
        a=np.asarray(slacks,float)
        output["slack_variation"]=dict(matched_components=len(a),
            matched_cases=int(a[:,0].sum()),delta_upper_per_case=float(a[:,1].sum()/a[:,0].sum()),
            delta_dependence_per_case=float(a[:,2].sum()/a[:,0].sum()),
            delta_split_per_case=float(a[:,3].sum()/a[:,0].sum()))
    # Tie-bracket widths and completeness are reported independently
    # of the full-cohort policy comparison.
    brackets={}
    for p in (dst/"tie").glob("*.json"):
        if p.name=="selection.json":continue
        rec=load(p)
        if not isinstance(rec.get("entries"),dict):continue
        short=p.stem.rsplit("__",1)
        if len(short)!=2:continue
        arm=short[1]
        q=brackets.setdefault(arm,dict(attempted=0,certified=0,
                                       spread_sum=0.,cases_certified=0,
                                       eta_sum=0.,eta_cases=0))
        q["attempted"]+=1
        if rec.get("planner_tolerance_eta") is not None:
            q["eta_sum"]+=float(rec["planner_tolerance_eta"])
            q["eta_cases"]+=rec["n"]
        lo=rec["entries"].get("best",{})
        hi=rec["entries"].get("worst",{})
        if lo.get("status")=="CERTIFIED" and hi.get("status")=="CERTIFIED":
            q["certified"]+=1
            q["spread_sum"]+=float(hi["cost"])-float(lo["cost"])
            q["cases_certified"]+=rec["n"]
    for q in brackets.values():
        q["mean_bracket_per_case"]=(q["spread_sum"]/q["cases_certified"]
                                   if q["cases_certified"] else None)
        q["mean_eta_per_case"]=(q["eta_sum"]/q["eta_cases"]
                                if q["eta_cases"] else None)
        del q["spread_sum"]
        del q["eta_sum"]
    output["tie_bracket_sample"]=brackets
    output["tie_bracket_caveat"]="Selected 25-scenario component subset, not full-cohort primary-optimal interval"
    # Read per-site response results on the same completed components.
    by_site={}
    for name,_,_ in RESPONSE_SCENARIOS:
        for site in SITES.values():
            rows=[]
            for ck,c,_ in ctx["eval"]:
                if c.group!=site:continue
                p=dst/"response"/(ck+"__"+name+".json")
                if not p.exists():continue
                r=load(p).get("method",{})
                if any(r.get(k,{}).get("status")!="COMPLETE" for k in ("VF","STAT")):continue
                rows.append((c.n,r["VF"]["cost"]-r["STAT"]["cost"]))
            n=sum(x for x,_ in rows)
            by_site[name+"__"+site]=dict(coverage_cases=n,
                paired_change_per_case=(sum(v for _,v in rows)/n if n else None))
    output["response_by_site"]=by_site
    # Recovery-check the nominal-response replay against the frozen exact
    # original policy results; a mismatch invalidates response sensitivity.
    diffs=[]
    for ck,c,_ in ctx["eval"]:
        p=dst/"response"/(ck+"__nominal.json")
        if not p.exists():continue
        r=load(p).get("method",{})
        for k in ("VF","STAT"):
            base=record(ctx,ck,k)
            if r.get(k,{}).get("status")=="COMPLETE" and base.get("status")=="COMPLETE":
                diffs.append(abs(float(r[k]["cost"])-float(base["realized"]["0"])))
    output["nominal_replay"]=dict(compared_arm_components=len(diffs),
        max_cost_discrepancy=max(diffs) if diffs else None,
        verified_exact=bool(diffs and max(diffs)<=1e-6))
    save(dst/"summary.json",output)
    print(json.dumps(output,indent=2))

def stage(args):
    if args.stage=="diagnose":diagnose(args)
    elif args.stage=="response":response(args)
    elif args.stage=="tie":tie(args)
    elif args.stage=="train":train(args)
    elif args.stage=="evaluate_variants":evaluate_variants(args)
    elif args.stage=="duration":duration(args)
    elif args.stage=="report":summarize(args)
    else:raise ValueError(args.stage)

def preflight(args):
    ctx=context(args)
    policies=frozen(args,ctx)
    for label,coeff in policies.items():
        if coeff.shape!=(5,) or not np.isfinite(coeff).all():
            raise RuntimeError("Invalid frozen policy "+label)
    for ck,c,rows in ctx["eval"]:
        for arm in ("CASE","STAT","VF"):
            path=Path(ctx["root"])/"full_evaluation"/"per_arm"/(ck+"__"+arm+".json")
            if not path.exists() or load(path).get("status")!="COMPLETE":
                raise RuntimeError("Missing certified original evaluation: "+ck+" "+arm)
    return ctx

def stage_completeness(name,args):
    """Count saved tasks rather than interpreting a zero-exit stage as exact."""
    p=folder(args)/name
    if name in ("diagnose","response","tie","evaluate_variants"):
        records=[load(x) for x in p.glob("*.json")
                 if x.name not in ("status.json","selection.json","summary.json")]
        if not records:return False
        if name=="response":
            return len(records)==1125*len(RESPONSE_SCENARIOS) and all(
                all(v.get("status")=="COMPLETE" for v in r.get("method",{}).values())
                and len(r.get("method",{}))==2 for r in records)
        if name=="diagnose":return len(records)==1125
        if name=="tie":
            target=load(p/"selection.json")["tasks"] if (p/"selection.json").exists() else 0
            return target>0 and len(records)==target and all(
                len(r.get("entries",{}))==2 and
                all(v.get("status")=="CERTIFIED" for v in r["entries"].values())
                for r in records)
        # Variant task count depends on which policies finished training.
        trained=[k for k in VARIANTS if
                 (folder(args)/"train"/k/"trained.json").exists()]
        ctx=context(args)
        target=sum(len(ctx["eval"]) if not k.startswith("leave_out_")
                   else sum(c.group==SITES[k.removeprefix("leave_out_")]
                            for _,c,_ in ctx["eval"]) for k in trained)
        return target>0 and len(records)==target and all(
            r.get("status")=="COMPLETE" for r in records)
    if name=="train":
        return all((p/k/"trained.json").exists() for k in VARIANTS)
    if name=="duration":return (p/"summary.json").exists()
    return True

def supervisor(args):
    preflight(args)
    dst=folder(args);dst.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parents[1]
    scientific=["pilot/run_vf_followups.py","surgery/objective_selection.py",
       "pilot/run_theory_full.py","surgery/anchored_weekly_solver.py",
       "surgery/advice_comparison.py","surgery/theory_exact.py"]
    contract=dict(source={p:hashlib.sha256((src/p).read_bytes()).hexdigest()
                   for p in scientific},
       original_frozen_sha256=hashlib.sha256((original(args)/"frozen.json").read_bytes()).hexdigest(),
       stages=STAGES,scenarios=RESPONSE_SCENARIOS,variants=VARIANTS,cores=args.cores)
    manifest=dst/"contract.json"
    if manifest.exists() and load(manifest)!=json.loads(json.dumps(contract)):
        raise RuntimeError("Source or frozen policy changed; choose a NEW output root")
    save(manifest,contract)
    begun=time.monotonic()
    status=load(dst/"run_status.json") if (dst/"run_status.json").exists() else {}
    for name,budget in STAGES:
        if name!="report" and status.get(name,{}).get("status")=="COMPLETE" and stage_completeness(name,args):
            continue
        remaining=11.5*3600-(time.monotonic()-begun)
        if remaining<30:break
        cmd=[sys.executable,str(Path(__file__).resolve()),"--stage",name,
             "--artifact-root",str(args.artifact_root),"--out",str(args.out),
             "--cores",str(args.cores)]
        start=time.monotonic()
        with (dst/(name+".log")).open("a") as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                proc.wait(timeout=min(budget,remaining-15))
                state=("COMPLETE" if stage_completeness(name,args) else "PARTIAL"
                       ) if proc.returncode==0 else "FAILED"
            except subprocess.TimeoutExpired:
                state="WALL_LIMIT"
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
        status[name]=dict(status=state,seconds=time.monotonic()-start,
                          exit_code=proc.returncode)
        save(dst/"run_status.json",status)
        print(name,state,flush=True)
    if not (dst/"summary.json").exists():
        try:summarize(args)
        except Exception as exc:save(dst/"summary_error.json",dict(error=repr(exc)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root",type=Path,
        default=Path("artifacts/objective_selection_12h"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/vf_followups"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--stage",choices=[s for s,_ in STAGES])
    args=ap.parse_args()
    if not 1<=args.cores<=15:ap.error("cores must be 1..15")
    if not (original(args)/"frozen.json").exists():ap.error("Original frozen.json missing")
    if args.stage:stage(args)
    else:supervisor(args)

if __name__=="__main__":main()
