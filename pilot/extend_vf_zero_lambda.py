#!/usr/bin/env python3
"""One-command zero-L1 VF-MM-pDCA experiment on the EXISTING UHN B/C grid.

Reuse and NEVER rewrite trained CASE, original VF, NAIVE, B/C deployment
results, and frozen recommendations. Train four NEW VF policies with lambda=0
using the same B response, CASE coefficients, standardized five features,
initial BOOKED/CASE draft schedule libraries, coefficient box, MM-pDCA and
weekly planner as the original scenario-grid training. Evaluate the four
frozen policies on the original 12 B/C evaluation assumptions.

All outputs are isolated under <existing-root>/vf_zero_extension.
For scientific comparison, the same 14-arm/12-scenario common-certified
sample and paired calendar-week bootstrap are computed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from pilot import run_response_scenario_grid as base
from pilot import extend_response_grid_evaluation as previous
from pilot import extend_vf_ray_grid as ray
from surgery.nonmonotone_learning import mm_vf, library_value
from surgery.response_grid import (
    TRAIN_SCENARIOS,mapping,verify_same_scenario,implemented_from_display)
from surgery.weekly_planner import evaluate as evaluate_week

EXTRA_ARMS=tuple("VF_ZERO__"+s.name for s in TRAIN_SCENARIOS)
METHODS=tuple(base.METHODS)+EXTRA_ARMS
LAMBDA=0.
RADIUS=64.


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_inputs(root,args):
    """The new specification cannot be mixed with previous VF-lambda runs."""
    output=root/"vf_zero_extension"
    output.mkdir(parents=True,exist_ok=True)
    contract=output/"run_contract.json"
    settings=dict(
        source_sha256=hash_file(__file__),
        base_contract_sha256=hash_file(root/"run_contract.json"),
        C_extension_contract_sha256=hash_file(
            root/"response_extension"/"extension_contract.json"),
        policies_sha256=hash_file(root/"selected_policies.json"),
        context_sha256=hash_file(args.context),
        training_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
        evaluation_scenarios=[vars(s) for s in previous.ALL_SCENARIOS],
        original_arms=list(base.METHODS),
        new_arms=list(EXTRA_ARMS),
        coefficient_box=RADIUS,lambda_vf=LAMBDA,
        initialization="theta=0",
        initial_schedules="same BOOKED + CASE(B) draft",
        optimizer="same mm_vf pdca",backend="gurobi",
        vf_outers=args.vf_outers,inner_steps=args.vf_pdca_steps,
        cuts=args.cuts,plan_seconds=args.plan_seconds,
        tie_training_oracle=False,tie_training_direct_cost=True,
        tie_deployment=True)
    if contract.exists():
        if json.loads(contract.read_text())!=settings:
            raise ValueError("Changed zero-lambda experiment settings/code; "
                             "use a different copy of the artifact root.")
    else:
        base.save(contract,settings)
    return output


def require_old_policies(root):
    """A lambda=0 experiment should not secretly refit CASE or plain VF."""
    file=root/"selected_policies.json"
    if not file.exists():
        raise FileNotFoundError("Original fitted B policies are required: "+str(file))
    policies=json.loads(file.read_text())
    if policies.get("response_link")!="SATURATING_B":
        raise ValueError("Only B-trained policies supported")
    got=policies["policies"]
    if set(got)!={s.name for s in TRAIN_SCENARIOS}:
        raise ValueError("Incomplete original CASE/VF policy bank")
    for sc in TRAIN_SCENARIOS:
        if got[sc.name]["scenario"]!=vars(sc):
            raise ValueError("Wrong training scenario: "+sc.name)
    return got


def require_old_results(root,ctx):
    """Original 12x10 cells are comparison data, never overwritten."""
    for sc in previous.ALL_SCENARIOS:
        for method in base.METHODS:
            file=(root/"evaluation_cells"/sc.name/(method+".json")
                  if sc.link=="saturating" else
                  root/"response_extension"/"evaluation_cells"/sc.name/
                  (method+".json"))
            if not file.exists():
                raise FileNotFoundError("First complete original B/C grid: "+str(file))
            previous.check_cell(file,sc,method,ctx["eval"],
                                extension=sc.link=="nonmonotone")


def fit_one(ctx,root,output,cache,scenario,policy,booked_plans,args):
    """Run the SAME MM-pDCA as VF(lambda=1), with lambda literally zero."""
    path=output/"training"/scenario.name
    path.mkdir(parents=True,exist_ok=True)
    frozen=path/"frozen_zero_policy.json"
    if frozen.exists():
        saved=json.loads(frozen.read_text())
        if saved["scenario"]!=vars(scenario) or saved["lambda_vf"]!=0.:
            raise ValueError("Zero-VF training checkpoint mismatch")
        print(f"{scenario.name}: zero-VF fit already completed",flush=True)
        return saved

    started=time.perf_counter()
    booked=np.asarray(ctx["fit"]["booked"],float)
    actual=np.asarray(ctx["fit"]["actual"],float)
    raw=np.asarray(ctx["fit"]["X"],float)@np.asarray(
            policy["case_coeff"],float)
    response=mapping(booked,scenario)
    verify_same_scenario(raw,booked,scenario)
    scaler=json.loads((root/"training"/scenario.name/"vf_features.json").read_text())
    H,_,draft=base.five_features(ctx,ctx["train"],raw,response,cache,
        "train_"+scenario.name,
        scaler=(np.asarray(scaler["mean"],float),
                np.asarray(scaler["std"],float)))
    H=np.asarray(H,float)
    old_theta=np.asarray(policy["vf_theta"],float)
    if H.shape!=(len(booked),len(old_theta)):
        raise ValueError("VF features/coefficient dimensions changed")
    # EXACTLY the same initial schedule-library definition as original VF.
    seed_libraries={
        key:[booked_plans[key]["assignment"],draft[key]["assignment"]]
        for key,_,_ in ctx["train"]}
    def oracle(items,durations,k):
        return cache.solve(items,durations,
              f"vf_zero_{scenario.name}_outer_{k}",tie=False)

    state=mm_vf(ctx["train"],booked,H,raw,response,actual,
        root=path/"vf",lam=0.,initial_libraries=seed_libraries,
        iterations=args.vf_outers,radius=RADIUS,
        pdca_steps=args.vf_pdca_steps,max_cuts=args.cuts,
        backend="gurobi",solve_plans=oracle,optimizer="pdca")
    theta=np.asarray(state["theta"],float)
    if not np.isfinite(theta).all() or abs(theta).max()>RADIUS+1e-7:
        raise AssertionError("VF(lambda=0) violates coefficient box")
    verify_same_scenario(raw+H@theta,booked,scenario)

    # Exact FINITE LIBRARY training surrogate, no regularization.
    d_zero=response.reviewed(raw+H@theta)
    d_plain=response.reviewed(raw+H@old_theta)
    final_library=state["libraries"]
    zero_library_score=library_value(
         ctx["train"],final_library,d_zero,actual,0.,theta)
    plain_library_score=library_value(
         ctx["train"],final_library,d_plain,actual,0.,old_theta)
    # Fair direct-cost comparison with VF-RAY is made on TRAINING data only.
    direct=ray.score_direct_training(
         ctx["train"],d_zero,cache,"train_direct_zero_"+scenario.name)
    selected=dict(
        scenario=vars(scenario),lambda_vf=0.,
        original_lambda_vf=float(policy["lambda_vf"]),
        case_coeff=policy["case_coeff"],
        zero_theta=theta.tolist(),plain_theta=old_theta.tolist(),
        vf_status=state["status"],
        vf_mm_iterations=len(state["history"]),
        max_abs_coefficient=float(abs(theta).max()),
        zero_surrogate=float(zero_library_score),
        plain_surrogate_on_zero_library=float(plain_library_score),
        direct_training=direct,
        elapsed_seconds=round(time.perf_counter()-started,2))
    base.save(frozen,selected)
    print(f"VF_ZERO {scenario.name}: {state['status']}; "
          f"MM iterations={len(state['history'])}; "
          f"direct cost={direct['mean_cost_per_component']}; "
          f"surrogate={zero_library_score:.4f}",flush=True)
    return selected


def train_all(ctx,root,output,cache,policies,args):
    booked=cache.solve(ctx["train"],
        np.asarray(ctx["fit"]["booked"],float),"train_booked",tie=True)
    selected={}
    for scenario in TRAIN_SCENARIOS:
        selected[scenario.name]=fit_one(
            ctx,root,output,cache,scenario,policies[scenario.name],
            booked,args)
    base.save(output/"selected_zero_policies.json",selected)
    return selected


def make_displays(ctx,root,output,cache,selected):
    """Use original training-B canonical displays; freeze before C tests."""
    booked=np.asarray(ctx["held"]["booked"],float)
    X=np.asarray(ctx["held"]["X"],float)
    result={}
    for scenario in TRAIN_SCENARIOS:
        name=scenario.name
        record=selected[name]
        raw=X@np.asarray(record["case_coeff"],float)
        model=mapping(booked,scenario)
        params=json.loads(
            (root/"training"/name/"vf_features.json").read_text())
        H,_,_=base.five_features(ctx,ctx["eval"],raw,model,cache,
            "eval_"+name,
            scaler=(np.asarray(params["mean"],float),
                    np.asarray(params["std"],float)))
        scores=raw+np.asarray(H,float)@np.asarray(
            record["zero_theta"],float)
        result["VF_ZERO__"+name]=verify_same_scenario(
            scores,booked,scenario)
    path=output/"frozen_zero_displays.csv"
    if path.exists():
        with path.open(newline="") as handle:
            prior=list(csv.DictReader(handle))
        if len(prior)!=len(booked):
            raise ValueError("Frozen zero-VF displays have wrong cohort size")
        for i,row in enumerate(prior):
            if int(row["case_id"])!=int(ctx["held"]["case_id"][i]):
                raise ValueError("Frozen zero-VF case IDs differ")
            for label in EXTRA_ARMS:
                if not np.isclose(float(row[label]),result[label][i],
                                  atol=1e-9,rtol=0):
                    raise ValueError("Previously frozen VF0 advice changed")
    else:
        tmp=path.with_suffix(".tmp")
        with tmp.open("w",newline="") as fh:
            writer=csv.writer(fh)
            writer.writerow(["case_id",*EXTRA_ARMS])
            for i,id_ in enumerate(ctx["held"]["case_id"]):
                writer.writerow([int(id_),*[
                    float(result[a][i]) for a in EXTRA_ARMS]])
        tmp.replace(path)
    return result


def evaluate_new(ctx,output,cache,displays):
    """Only four new VF0 arms; never rerun any old BOOKED/NAIVE/CASE/VF arm."""
    start=time.perf_counter()
    items=ctx["eval"]
    booked=np.asarray(ctx["held"]["booked"],float)
    for sc in previous.ALL_SCENARIOS:
        folder=output/"evaluation_cells"/sc.name
        folder.mkdir(parents=True,exist_ok=True)
        for method in EXTRA_ARMS:
            path=folder/(method+".json")
            if path.exists():
                previous.check_cell(path,sc,method,items,extension=True)
                continue
            revised=implemented_from_display(booked,displays[method],sc)
            solved=cache.solve(items,revised,
                    f"eval_zero_{sc.name}_{method}",tie=True)
            entries=[]
            for key,week,rows in items:
                result=solved[key]
                cost=evaluate_week(week,result["assignment"],week.actual)
                entries.append(dict(
                    component=key,site=week.group,week=week.monday,
                    cases=week.n,method=method,
                    training_scenario=method.split("__",1)[1],
                    evaluation_scenario=sc.name,
                    complete=bool(result["complete"]),
                    planner_status=result["status"],
                    cost=float(cost["cost"]),
                    idle_minutes=float(cost["idle"]),
                    overtime_minutes=float(cost["overtime"]),
                    mae_sum=float(np.abs(week.actual-revised[rows]).sum())))
            base.save(path,dict(method=method,evaluation=vars(sc),
                                entries=entries))
            previous.check_cell(path,sc,method,items,extension=True)
            print(f"VF_ZERO {sc.name}/{method}: "
                  f"{sum(x['complete'] for x in entries)}/{len(items)} certified",
                  flush=True)
    base.record_stage_time(output,"zero_evaluation",
                           time.perf_counter()-start)


def summarize(ctx,root,output):
    """Common-certified 14-policy / 12-response table and paired comparisons."""
    items=ctx["eval"]
    all_keys={key for key,_,_ in items}
    case_n={key:w.n for key,w,rows in items}
    week={key:w.monday for key,w,rows in items}
    by={}
    global_common=set(all_keys)
    for sc in previous.ALL_SCENARIOS:
        for method in METHODS:
            if method in base.METHODS:
                path=(root/"evaluation_cells"/sc.name/(method+".json")
                      if sc.link=="saturating" else
                      root/"response_extension"/"evaluation_cells"/sc.name/
                      (method+".json"))
            else:
                path=output/"evaluation_cells"/sc.name/(method+".json")
            lines=previous.check_cell(path,sc,method,items,
                       extension=(sc.link=="nonmonotone"
                                  or method in EXTRA_ARMS))
            by[sc.name,method]={r["component"]:r for r in lines}
        common={k for k in all_keys if all(
             by[sc.name,m][k]["complete"] for m in METHODS)}
        global_common.intersection_update(common)
    if not global_common:
        raise RuntimeError("No common certified cases across 12x14 grid")
    denom=sum(case_n[k] for k in global_common)
    weeks=sorted({week[k] for k in global_common})
    week_cases=np.asarray([
        sum(case_n[k] for k in global_common if week[k]==w)
        for w in weeks],float)
    rng=np.random.default_rng(20261022)
    idx=rng.integers(0,len(weeks),size=(10000,len(weeks)))
    boot_denom=week_cases[idx].sum(axis=1)
    report=dict(status="EXPLORATORY_REUSED_2013",
         primary_response="NONMONOTONE_C",
         secondary_response="SATURATING_B",
         vf_lambda=0.,
         globally_certified_cases=denom,
         globally_certified_components=len(global_common),
         training_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
         methods=list(METHODS),
         per_scenario={},
         interpretation="VF0 uses original pDCA; no numerical ray step.")
    table=[]
    for sc in previous.ALL_SCENARIOS:
        cost={}
        mae={}
        for method in METHODS:
            values=[by[sc.name,method][k] for k in global_common]
            cost[method]=sum(v["cost"] for v in values)/denom
            mae[method]=sum(v["mae_sum"] for v in values)/denom
        contrasts={}
        for tr in TRAIN_SCENARIOS:
            label=tr.name
            for reference in (
                    "CASE__"+label,"VF__"+label,"NAIVE","BOOKED"):
                method="VF_ZERO__"+label
                delta=np.array([
                    sum(by[sc.name,method][k]["cost"]-
                        by[sc.name,reference][k]["cost"]
                        for k in global_common if week[k]==w)
                    for w in weeks])
                sample=delta[idx].sum(axis=1)/boot_denom
                contrasts[method+"_minus_"+reference]=dict(
                     difference_per_case=float(delta.sum()/denom),
                     bootstrap_95=[float(v)
                         for v in np.quantile(sample,[.025,.975])],
                     better_weeks=int(np.sum(delta<0)),
                     worse_weeks=int(np.sum(delta>0)),
                     tied_weeks=int(np.sum(delta==0)))
        report["per_scenario"][sc.name]=dict(
            link=sc.link,alpha=sc.alpha,h=sc.h,
            cost_per_case=cost,mae_per_case=mae,
            contrasts=contrasts)
        table.append([sc.name,sc.link,sc.alpha,sc.h,denom,
                      *[cost[m] for m in METHODS]])
    base.save(output/"summary.json",report)
    path=output/"cost_grid_global_matched.csv"
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as fh:
        writer=csv.writer(fh)
        writer.writerow(["evaluation_scenario","link","alpha","h",
                         "common_cases",*METHODS])
        writer.writerows(table)
    tmp.replace(path)
    print(f"ZERO-L1 VF: common {len(global_common)} components, "
          f"{denom} cases across 12 scenarios and {len(METHODS)} arms",
          flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
         "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path(
         "artifacts/response_saturation_grid"),
         help="Existing complete B/C experiment root (reuses all fits)")
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=160)
    args=ap.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Missing original UHN context")
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        ap.error("Invalid solver configuration")
    start=time.perf_counter()
    previous.validate_original_run(args,args.out)
    with args.context.open("rb") as fh:
        ctx=pickle.load(fh)
    policies=require_old_policies(args.out)
    require_old_results(args.out,ctx)
    output=fixed_inputs(args.out,args)
    cache=base.ContentCache(args.out,args.cores,args.plan_seconds)
    selected=train_all(ctx,args.out,output,cache,policies,args)
    displays=make_displays(ctx,args.out,output,cache,selected)
    evaluate_new(ctx,output,cache,displays)
    summarize(ctx,args.out,output)
    base.record_stage_time(output,"zero_total",
                           time.perf_counter()-start)


if __name__=="__main__":
    main()
