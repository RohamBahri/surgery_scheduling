#!/usr/bin/env python3
"""One resumable saturated-response (B) cross-scenario experiment.

Train response-aware CASE and joint decision-aware VF separately
under EACH (alpha,h) in {(.5,30),(.5,60),(.8,30),(.8,60)}.
Train ONE NAIVE no-behavior LAD policy. Evaluate BOOKED, NAIVE and all
eight fitted policies under every saturated response in
{alpha=.5,.8} x {h=20,30,60}. This is 4 fits x 2 methods, NOT 8 VF fits.
The displayed advice of every learned policy is frozen by its TRAINING
scenario; evaluation response maps are applied to this SAME advice.
Never re-canonicalize displays using the evaluation response.
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
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[var]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from pilot.run_nonmonotone_learning import (
    PlannerCache,case_fit,five_features,train_lambda,save,
    record_stage_time,_plan_job)
from surgery.nonmonotone_learning import mm_vf
from surgery.response_grid import (
    TRAIN_SCENARIOS,EVAL_SCENARIOS,check_grid,mapping,
    advice_for_policy,naive_display,implemented_from_display,
    verify_same_scenario,evaluation_policy_labels)
from surgery.weekly_planner import evaluate as planner_evaluate

METHODS=("BOOKED","NAIVE")+evaluation_policy_labels()
CASE_LAMBDA_SCALE=1.0
VF_LAMBDA_SCALE=1.0


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def contract(args,root):
    from surgery import nonmonotone_learning as core,response_grid
    from pilot import run_nonmonotone_learning as common
    code_sources=(Path(__file__),Path(core.__file__),
                  Path(response_grid.__file__),Path(common.__file__))
    provenance=sha_bytes(b"".join(x.read_bytes() for x in code_sources))
    scientific=dict(
        context_sha256=sha_bytes(args.context.read_bytes()),
        train=[vars(x) for x in TRAIN_SCENARIOS],
        evaluation=[vars(x) for x in EVAL_SCENARIOS],
        arms=list(METHODS),case_lambda_scale=CASE_LAMBDA_SCALE,
        vf_lambda_scale=VF_LAMBDA_SCALE,naive_lambda_scale=1.0,
        case_radius=100.,vf_radius=100.,cap=180.,
        case_steps=args.case_steps,vf_outers=args.vf_outers,
        vf_pdca_steps=args.vf_pdca_steps,inner_cuts=args.cuts,
        plan_seconds=args.plan_seconds,cores=args.cores,
        deployment_tie="hash_seed_0",
        advice="freeze_display_under_training_B_link",
        planner_context="latest_objective_selection_12h")
    file=root/"run_contract.json"
    if file.exists():
        recorded=json.loads(file.read_text())
        if any(recorded.get(k)!=v for k,v in scientific.items()):
            raise ValueError("Grid scientific parameters changed: use a NEW output directory")
        if recorded.get("source")!=provenance:
            if (root/"naive_policy.json").exists() or any(
                  (root/"training"/scenario.name/"case_policy.json").exists()
                  for scenario in TRAIN_SCENARIOS):
                raise ValueError("Code changed after training: use a NEW output directory")
            save(file,dict(**scientific,source=provenance))
    else:
        save(file,dict(**scientific,source=provenance))


class ContentCache(PlannerCache):
    """Reuse identical weekly optimization across arms and evaluation scenarios.

    Each solve is cached by component, complete duration vector, tie setting,
    solver budget and hash seed. Unlike tag/key-only caches, this retains
    EVERY distinct candidate's actual certificate for true resumability.
    """
    def solve(self,items,durations,tag,tie=False):
        folder=self.root/"content_addressed"
        folder.mkdir(parents=True,exist_ok=True)
        results={}
        pending=[]
        for key,week,rows in items:
            d=np.ascontiguousarray(np.asarray(durations,float)[rows],
                                   dtype="<f8")
            identity=sha_bytes(
                str(key).encode()+d.tobytes()+
                repr((self.seconds,bool(tie),"hash_seed_0")).encode())
            file=folder/(identity+".json")
            if file.exists():
                saved=json.loads(file.read_text())
                if saved.get("signature")==identity:
                    results[key]=saved["result"]
                    continue
            pending.append(((key,week,d,self.seconds,tie),file,identity))
        print(f"{tag}: {len(results)} cached; {len(pending)} solves",flush=True)
        started=time.perf_counter()
        def remember(key,entry,file,signature):
            save(file,dict(signature=signature,result=entry))
            results[key]=entry
        if pending and self.cores==1:
            for i,(job,file,signature) in enumerate(pending,1):
                key,entry=_plan_job(job)
                remember(key,entry,file,signature)
                if i%100==0 or i==len(pending):
                    print(f"{tag}: {i}/{len(pending)} completed; "
                          f"{time.perf_counter()-started:.1f}s",flush=True)
        elif pending:
            with ProcessPoolExecutor(max_workers=self.cores,
                    mp_context=get_context("spawn")) as pool:
                futures={pool.submit(_plan_job,job):(file,signature)
                         for job,file,signature in pending}
                for i,future in enumerate(as_completed(futures),1):
                    key,entry=future.result()
                    file,signature=futures[future]
                    remember(key,entry,file,signature)
                    if i%100==0 or i==len(pending):
                        print(f"{tag}: {i}/{len(pending)} completed; "
                              f"{time.perf_counter()-started:.1f}s",flush=True)
        print(f"{tag}: certified {sum(x['complete'] for x in results.values())}"
              f"/{len(items)}",flush=True)
        return results


def fit_naive(ctx,root):
    """Response-unaware asymmetric quantile regression, fit ONCE.

    Same under/over-duration error cost (1.75/1), L1 penalty calibration
    and coefficient box as CASE; assumes Xw is adopted directly and
    ignores surgeon response. Deployment is still subjected to the
    actual saturation scenario.
    """
    file=root/"naive_policy.json"
    if file.exists():
        return np.asarray(json.loads(file.read_text())["weights"],float)
    started=time.perf_counter()
    X=sparse.csr_matrix(np.asarray(ctx["fit"]["X"],float))
    booked=np.asarray(ctx["fit"]["booked"],float)
    actual=np.asarray(ctx["fit"]["actual"],float)
    residual=actual-booked
    n,p=X.shape
    ncomp=len(ctx["train"])
    lam=train_lambda(X.toarray(),ncomp,intercept=True)
    I=sparse.eye(n,format="csr")
    E=sparse.eye(p,format="csr")
    Znn=sparse.csr_matrix((n,n))
    Znp=sparse.csr_matrix((n,p))
    Zpn=sparse.csr_matrix((p,n))
    # Variables [w, positive error, negative error, abs(w)].
    # positive error >= residual-Xw; negative error >= Xw-residual.
    A=sparse.vstack((
        sparse.hstack((-X,-I,Znn,Znp)),
        sparse.hstack((X,Znn,-I,Znp)),
        sparse.hstack((E,Zpn,Zpn,-E)),
        sparse.hstack((-E,Zpn,Zpn,-E))
    ),format="csr")
    rhs=np.r_[-residual,residual,np.zeros(2*p)]
    objective=np.r_[np.zeros(p),np.full(n,1.75/ncomp),
                    np.full(n,1./ncomp),
                    np.r_[0.,np.full(p-1,lam)]]
    ans=linprog(objective,A_ub=A,b_ub=rhs,
        bounds=[(-100.,100.)]*p+[(0,None)]*(2*n+p),
        method="highs",options={"time_limit":900.})
    if not ans.success:
        raise RuntimeError("NAIVE asymmetric LP not certified: "+str(ans.message))
    weights=np.asarray(ans.x[:p],float)
    save(file,dict(weights=weights.tolist(),lambda_naive=float(lam),
                   objective=float(ans.fun),solver_status=int(ans.status),
                   trained_with_2013=False,nominal_response_used=False,
                   error_loss="1.75*underprediction + 1.0*overprediction",
                   display_rule="clip(Xw,-180,180) without response inverse"))
    record_stage_time(root,"naive",time.perf_counter()-started)
    print(f"NAIVE: asymmetric LP certified; lambda={lam:.6f}",flush=True)
    return weights

def fit_family(ctx,args,root,cache,scenario,booked_plans):
    """Fit CASE for initialization and benchmark, then joint VF."""
    folder=root/"training"/scenario.name
    folder.mkdir(parents=True,exist_ok=True)
    completed=folder/"frozen_policy.json"
    if completed.exists():
        print(f"{scenario.name}: frozen CASE/VF fit reused",flush=True)
        return json.loads(completed.read_text())
    started=time.perf_counter()
    items=ctx["train"]
    data=ctx["fit"]
    booked=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    model=mapping(booked,scenario)
    print(f"TRAIN {scenario.name}: CASE(B) lambda x1.0",flush=True)
    coeff=case_fit(ctx,items,model,folder,
                   args.case_steps,args.cuts,"gurobi",100.,
                   lambda_scale=CASE_LAMBDA_SCALE)
    raw=np.asarray(data["X"],float)@coeff
    verify_same_scenario(raw,booked,scenario)
    H,(mean,std),_=five_features(
         ctx,items,cache,"train_"+scenario.name,plans=booked_plans)
    save(folder/"vf_features.json",dict(mean=mean.tolist(),std=std.tolist()))
    draft=cache.solve(items,model.reviewed(raw),
                     "train_"+scenario.name+"_case_draft",tie=True)
    libraries={key:[booked_plans[key]["assignment"],
                    draft[key]["assignment"]] for key,_,_ in items}
    X=np.column_stack([data["X"],H])
    start=np.r_[coeff,np.zeros(H.shape[1])]
    vf_lam=VF_LAMBDA_SCALE*train_lambda(X,len(items),intercept=True)
    print(f"TRAIN {scenario.name}: joint VF(B) MM-pDCA, "
          f"lambda={vf_lam:.8f}; initialized at CASE",flush=True)
    def oracle(parts,durations,k):
        return cache.solve(parts,durations,
                           f"train_{scenario.name}_vf_outer_{k}",tie=False)
    state=mm_vf(items,booked,X,model,actual,
                start=start,root=folder/"vf",lam=vf_lam,
                initial_libraries=libraries,
                iterations=args.vf_outers,radius=100.,
                pdca_steps=args.vf_pdca_steps,
                max_cuts=args.cuts,backend="gurobi",
                solve_plans=oracle)
    theta=np.asarray(state["theta"],float)
    verify_same_scenario(X@theta,booked,scenario)
    fitted=dict(scenario=vars(scenario),case_coeff=coeff.tolist(),
                vf_theta=theta.tolist(),
                lambda_case=float(json.loads(
                    (folder/"case_policy.json").read_text())["lambda_case"]),
                lambda_vf=float(vf_lam),
                vf_status=state["status"],
                vf_outer_iterations=len(state["history"]),
                train_cases=len(booked),train_components=len(items))
    save(completed,fitted)
    record_stage_time(root,"train_"+scenario.name,
                      time.perf_counter()-started)
    return fitted


def freeze_displays(ctx,root,cache,naive_weights,policies):
    """Compute EACH trained policy's display once, independent of eval alpha/h."""
    folder=root/"frozen_advice"
    folder.mkdir(parents=True,exist_ok=True)
    data=ctx["held"];items=ctx["eval"]
    booked=np.asarray(data["booked"],float)
    X=np.asarray(data["X"],float)
    displays={"NAIVE":naive_display(X,naive_weights)}
    ids=np.asarray(data["case_id"],int)
    for scenario in TRAIN_SCENARIOS:
        name=scenario.name
        policy=policies[name]
        case_raw=X@np.asarray(policy["case_coeff"],float)
        model=mapping(booked,scenario)
        a_case=verify_same_scenario(case_raw,booked,scenario)
        scaler=json.loads((root/"training"/name/"vf_features.json").read_text())
        H,_,_=five_features(ctx,items,cache,"eval_"+name,
                scaler=(np.asarray(scaler["mean"],float),
                        np.asarray(scaler["std"],float)))
        vf_raw=np.column_stack([X,H])@np.asarray(policy["vf_theta"],float)
        a_vf=verify_same_scenario(vf_raw,booked,scenario)
        displays[f"CASE__{name}"]=a_case
        displays[f"VF__{name}"]=a_vf
    if tuple(["BOOKED",*displays])!=METHODS:
        raise AssertionError("Grid policy order or roster is incorrect")
    path=folder/"evaluation_displays.csv"
    if not path.exists():
        tmp=path.with_suffix(".tmp")
        with tmp.open("w",newline="") as handle:
            w=csv.writer(handle)
            w.writerow(["case_id","booked","actual",*displays])
            for i in range(len(booked)):
                w.writerow([int(ids[i]),float(booked[i]),
                            float(data["actual"][i]),
                            *[float(d[i]) for d in displays.values()]])
        tmp.replace(path)
    return displays


def evaluate_grid(ctx,root,cache,displays):
    """All six scenarios, each with identical ten arms and hash-0 planner."""
    started=time.perf_counter()
    data=ctx["held"];items=ctx["eval"]
    booked=np.asarray(data["booked"],float)
    for scenario in EVAL_SCENARIOS:
        print(f"EVALUATION {scenario.name}: fixed displayed recommendations",
              flush=True)
        target=root/"evaluation_cells"/scenario.name
        target.mkdir(parents=True,exist_ok=True)
        for method in METHODS:
            cell=target/(method+".json")
            if cell.exists():
                continue
            revised=(booked if method=="BOOKED" else
                     implemented_from_display(booked,displays[method],scenario))
            solved=cache.solve(items,revised,
                       "evaluation_"+scenario.name+"_"+method,tie=True)
            records=[]
            for key,week,rows in items:
                rec=solved[key]
                cost=planner_evaluate(week,rec["assignment"],week.actual)
                records.append(dict(component=key,week=week.monday,
                     site=week.group,cases=week.n,method=method,
                     training_scenario=(None if "__" not in method
                                       else method.split("__",1)[1]),
                     evaluation_scenario=scenario.name,
                     cost=float(cost["cost"]),
                     idle_minutes=float(cost["idle"]),
                     overtime_minutes=float(cost["overtime"]),
                     mae_sum=float(np.abs(week.actual-revised[rows]).sum()),
                     complete=bool(rec["complete"]),
                     planner_status=rec["status"]))
            save(cell,dict(method=method,evaluation=vars(scenario),
                           entries=records))
            print(f"Saved {scenario.name}/{method} "
                  f"({sum(x['complete'] for x in records)}/{len(items)} certified)",
                  flush=True)
    record_stage_time(root,"grid_evaluation",
                      time.perf_counter()-started)
    summarize(ctx,root)


def summarize(ctx,root):
    """Paired scenario-by-scenario comparisons on an identical certified set."""
    summary=dict(label="EXPLORATORY_REUSED_2013",
                 model="SATURATING_B_ONLY",
                 train_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
                 evaluation_scenarios=[vars(s) for s in EVAL_SCENARIOS],
                 methods=METHODS,
                 warning="2013 repeatedly viewed in development; exploratory",
                 scenarios={})
    all_valid_sets=[]
    for scenario in EVAL_SCENARIOS:
        folder=root/"evaluation_cells"/scenario.name
        data={}
        for method in METHODS:
            file=folder/(method+".json")
            if not file.exists():
                raise RuntimeError("Incomplete evaluation cell "+str(file))
            rows=json.loads(file.read_text())["entries"]
            data[method]={r["component"]:r for r in rows}
        keys=set(data["BOOKED"])
        if any(set(data[m])!=keys for m in METHODS):
            raise AssertionError("Component roster differs between methods")
        good=sorted(key for key in keys
                    if all(data[m][key]["complete"] for m in METHODS))
        all_valid_sets.append(set(good))
        n=sum(data["BOOKED"][k]["cases"] for k in good)
        if not n:
            raise RuntimeError("No fully certified evaluation components")
        averages={}
        for m in METHODS:
            averages[m]=dict(
                 cost_per_case=sum(data[m][k]["cost"] for k in good)/n,
                 mae_per_case=sum(data[m][k]["mae_sum"] for k in good)/n,
                 idle_minutes_per_case=sum(
                     data[m][k]["idle_minutes"] for k in good)/n,
                 overtime_minutes_per_case=sum(
                     data[m][k]["overtime_minutes"] for k in good)/n,
                 certified_components=sum(data[m][k]["complete"] for k in keys))
        # Paired calendar-week resampling; no fallback from uncertified arms.
        grouped={}
        for key in good:
            ref=data["BOOKED"][key]
            w=grouped.setdefault(ref["week"],dict(
                cases=0,cost={m:0. for m in METHODS}))
            w["cases"]+=ref["cases"]
            for m in METHODS:
                w["cost"][m]+=data[m][key]["cost"]
        weeks=sorted(grouped)
        den=np.array([grouped[w]["cases"] for w in weeks],float)
        rng=np.random.default_rng(20261022)
        draws=rng.integers(0,len(weeks),size=(10000,len(weeks)))
        bootden=den[draws].sum(axis=1)
        contrasts={}
        pairs=[(m,r) for m in METHODS if m not in ("BOOKED","NAIVE")
               for r in ("BOOKED","NAIVE")]
        pairs.append(("NAIVE","BOOKED"))
        pairs += [(f"VF__{s.name}",f"CASE__{s.name}")
                  for s in TRAIN_SCENARIOS]
        for m,r in pairs:
            changes=np.array([grouped[w]["cost"][m]-
                              grouped[w]["cost"][r] for w in weeks])
            samples=changes[draws].sum(axis=1)/bootden
            contrasts[m+"_minus_"+r]=dict(
                estimate=float(changes.sum()/n),
                calendar_week_bootstrap_95=[
                    float(x) for x in np.quantile(samples,[.025,.975])],
                better_weeks=int(np.sum(changes<0)),
                worse_weeks=int(np.sum(changes>0)),
                equal_weeks=int(np.sum(changes==0)))
        summary["scenarios"][scenario.name]=dict(
            alpha=scenario.alpha,h=scenario.h,
            common_certified_components=len(good),
            total_components=len(keys),common_certified_cases=n,
            methods=averages,paired_cost_differences=contrasts)
        print(f"SUMMARY {scenario.name}: "
              f"{len(good)}/{len(keys)} components, {n} cases",flush=True)
    global_valid=set.intersection(*all_valid_sets)
    summary["all_scenarios_common_certified_components"]=len(global_valid)
    if global_valid:
        source=json.loads((root/"evaluation_cells"/EVAL_SCENARIOS[0].name/
                           "BOOKED.json").read_text())["entries"]
        common_cases=sum(r["cases"] for r in source
                         if r["component"] in global_valid)
        summary["all_scenarios_common_certified_cases"]=common_cases
        # Fair between-response comparisons require this SAME denominator.
        common_grid={}
        for scenario in EVAL_SCENARIOS:
            common_grid[scenario.name]={}
            for method in METHODS:
                rows=json.loads((root/"evaluation_cells"/scenario.name/
                                 (method+".json")).read_text())["entries"]
                included=[r for r in rows if r["component"] in global_valid]
                common_grid[scenario.name][method]=dict(
                    cost_per_case=sum(r["cost"] for r in included)/common_cases,
                    mae_per_case=sum(r["mae_sum"] for r in included)/common_cases)
        summary["global_common_grid"]=common_grid
        path=root/"cost_grid_global_matched.csv"
        temp=path.with_suffix(".tmp")
        with temp.open("w",newline="") as handle:
            writer=csv.writer(handle)
            writer.writerow(["evaluation_scenario","alpha","h",
                             "global_common_cases",*METHODS])
            for scenario in EVAL_SCENARIOS:
                writer.writerow([scenario.name,scenario.alpha,scenario.h,
                   common_cases,
                   *[common_grid[scenario.name][m]["cost_per_case"]
                     for m in METHODS]])
        temp.replace(path)
    save(root/"summary.json",summary)
    cost_path=root/"cost_grid.csv"
    mae_path=root/"mae_grid.csv"
    for path,field in ((cost_path,"cost_per_case"),(mae_path,"mae_per_case")):
        tmp=path.with_suffix(".tmp")
        with tmp.open("w",newline="") as handle:
            writer=csv.writer(handle)
            writer.writerow(["evaluation_scenario","alpha","h",
                             "certified_cases",*METHODS])
            for s in EVAL_SCENARIOS:
                row=summary["scenarios"][s.name]
                writer.writerow([s.name,s.alpha,s.h,
                        row["common_certified_cases"],
                        *[row["methods"][m][field] for m in METHODS]])
        tmp.replace(path)
    print("Saved summary.json, cost_grid.csv, mae_grid.csv",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path(
        "artifacts/response_saturation_grid"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=160)
    args=ap.parse_args()
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        ap.error("Invalid core count or planner time limit")
    check_grid()
    if not args.context.exists():
        raise FileNotFoundError("Missing 12h UHN context: "+str(args.context))
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    contract(args,args.out)
    with args.context.open("rb") as handle:
        ctx=pickle.load(handle)
    print(f"SATURATION B GRID: {len(TRAIN_SCENARIOS)} training scenarios, "
          f"{len(EVAL_SCENARIOS)} evaluation scenarios, {len(METHODS)} arms "
          f"({len(ctx['train'])} training components; "
          f"{len(ctx['eval'])} evaluation components)",flush=True)
    cache=ContentCache(args.out,args.cores,args.plan_seconds)
    fit_naive(ctx,args.out)
    booked=cache.solve(ctx["train"],
                       np.asarray(ctx["fit"]["booked"],float),
                       "train_booked",tie=True)
    policies={s.name:fit_family(ctx,args,args.out,cache,s,booked)
              for s in TRAIN_SCENARIOS}
    save(args.out/"selected_policies.json",dict(
        response_link="SATURATING_B",policies=policies,
        training_scenarios=[vars(s) for s in TRAIN_SCENARIOS]))
    naive=np.asarray(json.loads((args.out/"naive_policy.json").read_text())[
                     "weights"],float)
    frozen=freeze_displays(ctx,args.out,cache,naive,policies)
    evaluate_grid(ctx,args.out,cache,frozen)
    record_stage_time(args.out,"one_command_total",
                      time.perf_counter()-start)


if __name__=="__main__":
    main()
