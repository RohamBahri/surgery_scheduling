#!/usr/bin/env python3
"""Train finite-grid VF-RAY step and evaluate on the ORIGINAL twelve-scenario grid.

B-trained CASE/VF coefficients, pDCA histories, response assumptions, fit
features, planner, and old 120 evaluation cells are frozen. The only new
TRAINING decision is one nonnegative ray multiplier per original VF fit.

At each grid s, solve the hash-0 planner on all training components using
B-reviewed durations from case_raw + H @ (s * theta_vf). Rank by the mean
REALIZED operating-room cost of its resulting assignments (no L1 term).
The full-training cost is constant between assignment switches in exact
arithmetic. A finite uniform grid DOES NOT certify continuous-ray optimality.

The grid includes s=0 (CASE) and s=1 (VF), and s_max respects the VF box.
Select smallest s among equal-cost certified grid points. All grid points
must be certified to claim grid optimality, and both endpoints certified
to claim never-worse TRAINING direct cost.

Outputs are entirely under <existing-B-root>/vf_ray_extension/. No existing
train policies, B/C grids, or source fingerprint are rewritten.
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
from pilot import run_response_scenario_grid as grid
from pilot import extend_response_grid_evaluation as previous
from surgery.response_grid import (
    TRAIN_SCENARIOS, advice_for_policy, implemented_from_display,
    mapping, verify_same_scenario)
from surgery.weekly_planner import evaluate as week_evaluate

ARMS=tuple(grid.METHODS)+(tuple("VF_RAY__"+s.name for s in TRAIN_SCENARIOS))
RADIUS=64.


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _as_float_array(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite learned weights or features")
    return x


def ray_points(theta,requested_smax,n_points):
    """A predeclared feasible ray grid containing both original policies."""
    theta=_as_float_array(theta)
    if theta.ndim!=1 or not len(theta):
        raise ValueError("Missing VF coefficient vector")
    if requested_smax<1 or n_points<3:
        raise ValueError("Need s_max >= 1 and at least three grid points")
    magnitude=float(np.max(np.abs(theta)))
    if magnitude>RADIUS+1e-6:
        raise ValueError("Original VF coefficient violates its box")
    if magnitude<1e-12:
        return np.array([0.,1.])
    ceiling=min(float(requested_smax),RADIUS/magnitude)
    if ceiling<1.-1e-9:
        raise ValueError("Ray upper bound excludes unscaled VF")
    ceiling=max(1.,ceiling)
    values=np.linspace(0.,ceiling,n_points)
    # Include both endpoints EXACTLY even if smax is data-dependent.
    return np.unique(np.r_[0.,1.,values])


def scientific_fingerprint(root,args):
    file=root/"vf_ray_extension"/"run_contract.json"
    cfg=dict(
        ray_implementation_sha256=_digest(__file__),
        parent_B_contract_sha256=_digest(root/"run_contract.json"),
        parent_C_extension_sha256=_digest(
            root/"response_extension"/"extension_contract.json"),
        frozen_policy_sha256=_digest(root/"selected_policies.json"),
        frozen_display_sha256=_digest(
            root/"frozen_advice"/"evaluation_displays.csv"),
        context_sha256=_digest(args.context),
        train_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
        eval_scenarios=[vars(s) for s in previous.ALL_SCENARIOS],
        methods=list(ARMS),s_max=float(args.s_max),
        grid_points=int(args.grid_points),coefficient_radius=RADIUS,
        planner_seconds=float(args.plan_seconds),
        train_tie="hash_zero",train_cost="realized_idle+1.75*overtime",
        method="all_grid_steps_fully_certified_one_dimensional_grid",
        grid_optimality="only over sampled grid, not continuous ray",
        experiment_status="exploratory_2013_reused",
    )
    if file.exists():
        if json.loads(file.read_text())!=cfg:
            raise ValueError("Ray scientific inputs changed: use a separate "
                             "output copy; do not mix old ray checkpoints")
    else:
        grid.save(file,cfg)
    return cfg


def training_features(ctx,root,scenario,policy,cache):
    """Recreate original CASE-draft features, from TRAINING data only."""
    data=ctx["fit"]
    booked=_as_float_array(data["booked"])
    case_raw=_as_float_array(data["X"])@_as_float_array(policy["case_coeff"])
    m=mapping(booked,scenario)
    verify_same_scenario(case_raw,booked,scenario)
    file=root/"training"/scenario.name/"vf_features.json"
    scaler=json.loads(file.read_text())
    H,mu_sd,draft=grid.five_features(
        ctx,ctx["train"],case_raw,m,cache,"train_"+scenario.name,
        scaler=(np.asarray(scaler["mean"],float),
                np.asarray(scaler["std"],float)))
    H=_as_float_array(H)
    theta=_as_float_array(policy["vf_theta"])
    if H.shape!=(len(booked),len(theta)):
        raise AssertionError("Recomputed training VF features have wrong shape")
    return booked,case_raw,H,theta,m


def score_direct_training(items,reviews,cache,tag):
    """F_true at one grid point; NEVER penalize or substitute VF surrogate."""
    results=cache.solve(items,reviews,tag,tie=True)
    total=0.
    statuses={}
    missing=[]
    for key,week,rows in items:
        entry=results.get(key)
        if (entry is None or not entry.get("complete")
            or entry.get("assignment") is None):
            missing.append(str(key))
            if entry is not None:
                statuses[str(key)]=str(entry.get("status"))
            continue
        total+=week_evaluate(week,entry["assignment"],week.actual)["cost"]
    return dict(certified_components=len(items)-len(missing),
                total_components=len(items),unresolved=missing,
                unresolved_statuses=statuses,
                total_cost=float(total) if not missing else None,
                mean_cost_per_component=float(total/len(items))
                    if not missing else None)


def optimize_ray(ctx,root,output,cache,scenario,policy,args):
    """One independent pre-specified full-training 1d grid per original VF."""
    folder=output/"training"/scenario.name
    folder.mkdir(parents=True,exist_ok=True)
    booked,case_raw,H,theta,model=training_features(
        ctx,root,scenario,policy,cache)
    grid_s=ray_points(theta,args.s_max,args.grid_points)
    scalar_projection=H@theta
    ledger=[]
    started=time.perf_counter()
    for index,s in enumerate(grid_s):
        path=folder/("s_%03d.json"%index)
        if path.exists():
            row=json.loads(path.read_text())
            if not np.isclose(row["s"],s,rtol=0,atol=1e-12):
                raise ValueError("Ray grid checkpoint step changed")
        else:
            durations=model.reviewed(case_raw+float(s)*scalar_projection)
            row=dict(s=float(s),index=index,elapsed_seconds=None,
                     **score_direct_training(
                         ctx["train"],durations,cache,
                         f"ray_{scenario.name}_{index}"))
            row["elapsed_seconds"]=round(time.perf_counter()-started,2)
            grid.save(path,row)
            print(f"{scenario.name}: point {index+1}/{len(grid_s)} "
                  f"s={s:.5f} cost={row['mean_cost_per_component']} "
                  f"certified={row['certified_components']}/{len(ctx['train'])}",
                  flush=True)
        ledger.append(row)
    complete=[r for r in ledger if r["mean_cost_per_component"] is not None]
    if not complete:
        raise RuntimeError("No fully certified ray point for "+scenario.name)
    best=min(complete,key=lambda r:(r["total_cost"],r["s"]))
    grid_complete=len(complete)==len(ledger)
    at_zero=next(r for r in ledger if abs(r["s"])<1e-12)
    at_one=next(r for r in ledger if abs(r["s"]-1.)<1e-12)
    endpoints_certified=all(
        r["mean_cost_per_component"] is not None for r in (at_zero,at_one))
    never_worse=bool(endpoints_certified and
        best["total_cost"]<=min(at_zero["total_cost"],at_one["total_cost"])+1e-7)
    if endpoints_certified and not never_worse:
        raise AssertionError("Selected point is worse than an endpoint")
    if not grid_complete:
        # A noncertified point is not eligible for a grid-optimality claim.
        print(f"WARNING {scenario.name}: {len(ledger)-len(complete)} "
              "points lack full planner certificates; result is only "
              "best among certified points",flush=True)
    if not endpoints_certified:
        print(f"WARNING {scenario.name}: CASE or VF endpoint uncertified; "
              "no never-worse guarantee can be asserted",flush=True)
    chosen=dict(scenario=vars(scenario),s=float(best["s"]),
        vf_theta_original=theta.tolist(),
        vf_theta_ray=(float(best["s"])*theta).tolist(),
        case_coeff=policy["case_coeff"],
        grid_points=[float(s) for s in grid_s],
        grid_complete=grid_complete,
        endpoints_certified=endpoints_certified,
        never_worse_than_case_and_vf_on_certified_training=endpoints_certified,
        discrete_grid_optimality_certified=grid_complete,
        continuous_ray_optimality_certified=False,
        number_noncertified=len(ledger)-len(complete),
        direct_train_cost_per_component=float(best["mean_cost_per_component"]),
        case_train_cost_per_component=at_zero["mean_cost_per_component"],
        vf_train_cost_per_component=at_one["mean_cost_per_component"],
        mean_cost_denominator="training components",
        objective_includes_L1=False,selection_tie_rule="smallest_s",
        training_ledger=ledger,
        training_elapsed_seconds=round(time.perf_counter()-started,2))
    grid.save(folder/"selection.json",chosen)
    return chosen


def select_all(ctx,root,output,cache,policies,args):
    selections={}
    for scenario in TRAIN_SCENARIOS:
        print("VF-RAY direct-cost selection:",scenario.name,flush=True)
        selections[scenario.name]=optimize_ray(
            ctx,root,output,cache,scenario,policies[scenario.name],args)
    grid.save(output/"selected_rays.json",selections)
    return selections


def frozen_ray_displays(ctx,root,output,cache,policies,selections):
    """Canonical B TRAINING displays; never adapt to C/B evaluation link."""
    output.mkdir(parents=True,exist_ok=True)
    data=ctx["held"]
    booked=_as_float_array(data["booked"])
    X=_as_float_array(data["X"])
    out={}
    rows={}
    for scenario in TRAIN_SCENARIOS:
        label=scenario.name
        policy=policies[label]
        chosen=selections[label]
        raw=X@_as_float_array(policy["case_coeff"])
        m=mapping(booked,scenario)
        scaler=json.loads((root/"training"/label/"vf_features.json").read_text())
        H,_,_=grid.five_features(ctx,ctx["eval"],raw,m,cache,
                   "eval_"+label,
                   scaler=(np.asarray(scaler["mean"],float),
                           np.asarray(scaler["std"],float)))
        projection=_as_float_array(H)@_as_float_array(policy["vf_theta"])
        new_raw=raw+float(chosen["s"])*projection
        shown=verify_same_scenario(new_raw,booked,scenario)
        out["VF_RAY__"+label]=shown
        rows[label]=dict(s=chosen["s"],
                         fraction_above_training_h=float(
                              np.mean(np.abs(new_raw)>scenario.h)),
                         displayed_max=float(np.max(np.abs(shown))))
    if len(out)!=4:
        raise AssertionError("Missing ray policy displays")
    path=output/"frozen_ray_advice.csv"
    if path.exists():
        with path.open(newline="") as fh:
            old=list(csv.DictReader(fh))
        if len(old)!=len(booked):
            raise ValueError("Previously frozen ray advice cohort changed")
        for i,row in enumerate(old):
            if int(row["case_id"])!=int(data["case_id"][i]):
                raise ValueError("Ray advice case IDs changed")
            for arm,val in out.items():
                if not np.isclose(float(row[arm]),val[i],atol=1e-9,rtol=0):
                    raise ValueError("Previously frozen ray displays changed")
    else:
        tmp=path.with_suffix(".tmp")
        with tmp.open("w",newline="") as file:
            writer=csv.writer(file)
            writer.writerow(["case_id",*out])
            for i,id_ in enumerate(data["case_id"]):
                writer.writerow([int(id_),*[
                    float(out[a][i]) for a in out]])
        tmp.replace(path)
    grid.save(output/"ray_display_diagnostics.json",rows)
    return out


def evaluate_ray(ctx,root,output,cache,displays):
    """Solve only 4 new policies x 12 scenarios; old 120 cells remain intact."""
    started=time.perf_counter()
    items=ctx["eval"]
    booked=np.asarray(ctx["held"]["booked"],float)
    for scenario in previous.ALL_SCENARIOS:
        target=output/"evaluation_cells"/scenario.name
        target.mkdir(parents=True,exist_ok=True)
        for method,display in displays.items():
            path=target/(method+".json")
            if path.exists():
                previous.check_cell(path,scenario,method,items,extension=True)
                continue
            reviewed=implemented_from_display(booked,display,scenario)
            plans=cache.solve(items,reviewed,
                 f"eval_ray_{scenario.name}_{method}",tie=True)
            records=[]
            for key,week,indices in items:
                rec=plans[key]
                cost=week_evaluate(week,rec["assignment"],week.actual)
                records.append(dict(
                   component=key,week=week.monday,site=week.group,cases=week.n,
                   method=method,
                   training_scenario=method.split("__",1)[1],
                   evaluation_scenario=scenario.name,
                   complete=bool(rec["complete"]),
                   planner_status=rec["status"],
                   cost=float(cost["cost"]),
                   idle_minutes=float(cost["idle"]),
                   overtime_minutes=float(cost["overtime"]),
                   mae_sum=float(np.abs(
                       week.actual-reviewed[indices]).sum())))
            grid.save(path,dict(method=method,
                               evaluation=vars(scenario),entries=records))
            previous.check_cell(path,scenario,method,items,extension=True)
            print(f"{scenario.name}/{method}: "
                  f"{sum(x['complete'] for x in records)}/{len(items)} "
                  "certified",flush=True)
    grid.record_stage_time(output,"ray_evaluation",
                           time.perf_counter()-started)


def summarize(ctx,root,output):
    """All policies compared on identical sample across all twelve scenarios."""
    items=ctx["eval"]
    methods=ARMS
    keys={key for key,_,_ in items}
    n_by_key={key:week.n for key,week,_ in items}
    week_by_key={key:week.monday for key,week,_ in items}
    rows={}
    good_all=set(keys)
    certified={}
    for sc in previous.ALL_SCENARIOS:
        for m in methods:
            if m in grid.METHODS:
                path=(root/"evaluation_cells"/sc.name/(m+".json")
                      if sc.link=="saturating" else
                      root/"response_extension"/"evaluation_cells"/sc.name/(m+".json"))
            else:
                path=output/"evaluation_cells"/sc.name/(m+".json")
            records=previous.check_cell(path,sc,m,items,
                        extension=(sc.link=="nonmonotone" or m not in grid.METHODS))
            rows[sc.name,m]={r["component"]:r for r in records}
        passed={k for k in keys if all(
            rows[sc.name,m][k]["complete"] for m in methods)}
        certified[sc.name]=passed
        good_all.intersection_update(passed)
    if not good_all:
        raise RuntimeError("No global certified cohort for four VF-RAY arms")
    total_n=sum(n_by_key[k] for k in good_all)
    report=dict(label="EXPLORATORY_REUSED_2013",
                primary_evaluation="NONMONOTONE_C",
                secondary_evaluation="SATURATING_B",
                methods=list(methods),
                global_matched_cases=total_n,
                global_matched_components=len(good_all),
                evaluation_scenarios=[vars(sc) for sc in previous.ALL_SCENARIOS],
                models_trained_under_B=True,
                vf_ray_training_criterion="full-training unregularized direct cost",
                vf_ray_optimality="among sampled certified grid points",
                scenarios={},matched_C_vs_B={})
    week_keys=sorted(set(week_by_key.values()))
    def agg_delta(sc,m,baseline):
        d={w:0. for w in week_keys}
        n={w:0 for w in week_keys}
        for key in good_all:
            w=week_by_key[key]
            n[w]+=n_by_key[key]
            d[w]+=rows[sc.name,m][key]["cost"]-rows[sc.name,baseline][key]["cost"]
        used={w:v for w,v in d.items() if n[w]>0}
        return previous.clustered_interval(used,{w:n[w] for w in used})
    global_grid=[]
    for sc in previous.ALL_SCENARIOS:
        block={}
        for m in methods:
            vals=[rows[sc.name,m][key] for key in good_all]
            block[m]=dict(
                cost_per_case=sum(r["cost"] for r in vals)/total_n,
                mae_per_case=sum(r["mae_sum"] for r in vals)/total_n,
                certified_components=sum(rows[sc.name,m][k]["complete"]
                                         for k in keys))
        contrasts={}
        for train in TRAIN_SCENARIOS:
            tag=train.name
            method="VF_RAY__"+tag
            for baseline in ("CASE__"+tag,"VF__"+tag,"NAIVE","BOOKED"):
                contrasts[method+"_minus_"+baseline]=agg_delta(
                      sc,method,baseline)
        report["scenarios"][sc.name]=dict(
            kind=sc.link,alpha=sc.alpha,h=sc.h,
            globally_matched_methods=block,paired_contrasts=contrasts,
            per_scenario_common_components=len(certified[sc.name]))
        global_grid.append([sc.name,sc.link,sc.alpha,sc.h,total_n,
                           *[block[m]["cost_per_case"] for m in methods]])
    for B in previous.B_SCENARIOS:
        C=next(x for x in previous.C_SCENARIOS
               if x.alpha==B.alpha and x.h==B.h)
        report["matched_C_vs_B"][B.name]={m:
           agg_delta(C,m,"BOOKED")["estimate"]-
           agg_delta(B,m,"BOOKED")["estimate"]
           for m in methods}
    grid.save(output/"summary.json",report)
    path=output/"cost_grid_global_matched.csv"
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["evaluation_scenario","response_link","alpha","h",
                         "common_cases",*methods])
        writer.writerows(global_grid)
    tmp.replace(path)
    print(f"VF-RAY complete: {len(good_all)} components, "
          f"{total_n} cases globally matched across all 12 scenarios "
          "and 14 policy labels",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path(
        "artifacts/response_saturation_grid"),
        help="Existing response_saturation_grid root with trained B policies")
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=160)
    ap.add_argument("--s-max",type=float,default=2.)
    ap.add_argument("--grid-points",type=int,default=13)
    args=ap.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Missing training context "+str(args.context))
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        ap.error("Invalid planner configuration")
    if args.s_max<1 or args.grid_points<3:
        ap.error("Require s-max>=1, grid-points>=3")
    started=time.perf_counter()
    prior=previous.validate_original_run(args,args.out)
    with args.context.open("rb") as file:
        ctx=pickle.load(file)
    cache=grid.ContentCache(args.out,args.cores,args.plan_seconds)
    frozen=previous.existing_policies_or_resume(ctx,args.out,args,cache)
    previous.ensure_B_results(ctx,args.out,cache,frozen)
    folder_C=previous.extension_contract(args.out,prior,
              args.out/"frozen_advice"/"evaluation_displays.csv")
    previous.evaluate_C(ctx,args.out,folder_C,cache,frozen)
    output=args.out/"vf_ray_extension"
    output.mkdir(parents=True,exist_ok=True)
    scientific_fingerprint(args.out,args)
    selected=json.loads((args.out/"selected_policies.json").read_text())["policies"]
    selections=select_all(ctx,args.out,output,cache,selected,args)
    displays=frozen_ray_displays(ctx,args.out,output,cache,selected,selections)
    evaluate_ray(ctx,args.out,output,cache,displays)
    summarize(ctx,args.out,output)
    grid.record_stage_time(output,"ray_total",
                           time.perf_counter()-started)


if __name__=="__main__":
    main()
