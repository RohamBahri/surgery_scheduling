#!/usr/bin/env python3
"""Extend an existing SATURATION-trained UHN response grid with C deployment.

NO new models or penalty settings. Four B-trained CASE and four B-trained VF
policies, one NAIVE and BOOKED are held FIXED. The six original B evaluation
scenarios are preserved as a secondary reference. Six C (non-monotone)
evaluation assumptions, with the same alpha/h combinations, are the new
PRIMARY assessment of behavioral misspecification.

Pass --out pointing to the EXISTING artifacts/response_saturation_grid.
All completed policies, displays, B cells and weekly planner certificates
are reused. The existing run_contract.json and B summary are NOT edited.
This script is deliberately separate from the original runner so its source
fingerprint stays unchanged and a completed scientific fit remains valid.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot import run_response_scenario_grid as old
from surgery.response_grid import (
    TRAIN_SCENARIOS, EVAL_SCENARIOS as B_SCENARIOS,
    Scenario, implemented_from_display,
)
from surgery.weekly_planner import evaluate as evaluate_week

C_SCENARIOS=tuple(
    Scenario("NONMONO_"+s.name.removeprefix("SAT_"),
             "nonmonotone",s.alpha,s.h) for s in B_SCENARIOS)
ALL_SCENARIOS=tuple(B_SCENARIOS)+C_SCENARIOS
METHODS=old.METHODS


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_original_run(args,root):
    """Enforce original scientific contract, not a new training specification."""
    if not (root/"run_contract.json").exists():
        raise FileNotFoundError(
            "The existing B-grid run_contract.json is missing: "+str(root))
    # Does NOT change old contract, and refuses changed training/source.
    old.contract(args,root)
    previous=json.loads((root/"run_contract.json").read_text())
    if previous["arms"]!=list(METHODS):
        raise ValueError("Previous arms do not match the frozen grid")
    if previous["train"]!=[vars(s) for s in TRAIN_SCENARIOS]:
        raise ValueError("Previous B training grid differs")
    if previous["evaluation"]!=[vars(s) for s in B_SCENARIOS]:
        raise ValueError("Original B evaluation grid changed")
    if previous.get("case_lambda_scale")!=1.0:
        raise ValueError("CASE regularization differs from original grid")
    if previous.get("vf_lambda_scale")!=1.0:
        raise ValueError("VF regularization differs from original grid")
    return previous


def read_frozen_displays(path,ctx):
    """Read exactly the recommendations produced under each TRAINING B link."""
    with path.open(newline="") as handle:
        records=list(csv.DictReader(handle))
    expected=["case_id","booked","actual",*METHODS[1:]]
    if not records or list(records[0])!=expected:
        raise ValueError("Frozen-advice file has an unexpected schema")
    data=ctx["held"]
    if len(records)!=len(data["booked"]):
        raise ValueError("Frozen-advice file has a different cohort size")
    ids=np.asarray([int(row["case_id"]) for row in records],int)
    booked=np.asarray([float(row["booked"]) for row in records],float)
    actual=np.asarray([float(row["actual"]) for row in records],float)
    if (not np.array_equal(ids,np.asarray(data["case_id"],int))
            or not np.array_equal(booked,np.asarray(data["booked"],float))
            or not np.array_equal(actual,np.asarray(data["actual"],float))):
        raise ValueError("Frozen advice does not belong to the current cohort")
    displays={arm:np.asarray([float(row[arm]) for row in records],float)
              for arm in METHODS[1:]}
    for arm,u in displays.items():
        if not np.isfinite(u).all() or np.any(abs(u)>180.+1e-8):
            raise ValueError("Invalid displayed advice for "+arm)
        if "__" in arm:
            label=arm.split("__",1)[1]
            train=next(s for s in TRAIN_SCENARIOS if s.name==label)
            if np.any(abs(u)>train.h+1e-8):
                raise ValueError("Displayed advice exceeds training h: "+arm)
    return displays


def existing_policies_or_resume(ctx,root,args,cache):
    """Use all completed prior work; finish missing B fits in same invocation."""
    all_policy=root/"selected_policies.json"
    if all_policy.exists():
        bundle=json.loads(all_policy.read_text())
        if bundle.get("response_link")!="SATURATING_B":
            raise ValueError("Saved training policies use the wrong link")
        policies=bundle["policies"]
        if set(policies)!={s.name for s in TRAIN_SCENARIOS}:
            raise ValueError("Incomplete saved scenario policy bank")
        for s in TRAIN_SCENARIOS:
            entry=policies[s.name]
            if entry["scenario"]!=vars(s):
                raise ValueError("Training scenario mismatch "+s.name)
        naive_file=root/"naive_policy.json"
        if not naive_file.exists():
            raise ValueError("Selected policies exist but NAIVE is missing")
        naive=np.asarray(json.loads(naive_file.read_text())["weights"],float)
        print("Reuse: all eight CASE/VF fits and NAIVE already frozen",flush=True)
    else:
        print("Completing only missing B training checkpoints",flush=True)
        naive=old.fit_naive(ctx,root)
        booked=cache.solve(ctx["train"],
                 np.asarray(ctx["fit"]["booked"],float),
                 "train_booked",tie=True)
        policies={s.name:old.fit_family(ctx,args,root,cache,s,booked)
                  for s in TRAIN_SCENARIOS}
        old.save(all_policy,dict(response_link="SATURATING_B",
             policies=policies,
             training_scenarios=[vars(s) for s in TRAIN_SCENARIOS]))
    path=root/"frozen_advice"/"evaluation_displays.csv"
    if path.exists():
        displays=read_frozen_displays(path,ctx)
        print("Reuse: frozen displayed-advice CSV, no CASE draft re-solves",
              flush=True)
    else:
        print("Constructing missing frozen displays from saved policies",
              flush=True)
        displays=old.freeze_displays(ctx,root,cache,naive,policies)
        read_frozen_displays(path,ctx)
    return displays


def check_cell(file,scenario,method,items,*,extension):
    """Reject stale/partial cells rather than silently treating as certified."""
    saved=json.loads(file.read_text())
    entries=saved.get("entries")
    if (not isinstance(entries,list) or len(entries)!=len(items)
            or saved.get("method")!=method):
        raise ValueError("Invalid or incomplete evaluation cell "+str(file))
    if extension:
        if saved.get("evaluation")!=vars(scenario):
            raise ValueError("Wrong C scenario metadata "+str(file))
    else:
        if saved.get("evaluation")!=vars(scenario):
            raise ValueError("Wrong B scenario metadata "+str(file))
    expected={key for key,_,_ in items}
    ids=[r["component"] for r in entries]
    if set(ids)!=expected or len(ids)!=len(set(ids)):
        raise ValueError("Evaluation cell has missing/duplicate components "+str(file))
    for row in entries:
        if row["method"]!=method or row["evaluation_scenario"]!=scenario.name:
            raise ValueError("Wrong method or scenario in "+str(file))
        if not np.isclose(row["cost"],row["idle_minutes"]+
                          1.75*row["overtime_minutes"],rtol=0,atol=1e-5):
            raise ValueError("Incorrect cost decomposition in "+str(file))
    return entries


def ensure_B_results(ctx,root,cache,displays):
    """Run the original B evaluation ONLY if cells are actually missing."""
    missing=[]
    for scenario in B_SCENARIOS:
        for method in METHODS:
            cell=root/"evaluation_cells"/scenario.name/(method+".json")
            if not cell.exists():
                missing.append(str(cell))
            else:
                check_cell(cell,scenario,method,ctx["eval"],extension=False)
    if missing:
        print(f"B reference: {len(missing)} cells unfinished; "
              "reusing all cached planner certificates for completion",
              flush=True)
        old.evaluate_grid(ctx,root,cache,displays)
        for scenario in B_SCENARIOS:
            for method in METHODS:
                check_cell(root/"evaluation_cells"/scenario.name/
                           (method+".json"),scenario,method,ctx["eval"],
                           extension=False)
    else:
        print("Reuse: all 60 existing B evaluation cells; no B planner solves",
              flush=True)


def extension_contract(root,training_contract,display_csv):
    """Separate immutable extension contract; never rewrite old B contract."""
    folder=root/"response_extension"
    folder.mkdir(parents=True,exist_ok=True)
    settings=dict(
        source_training_contract_sha256=file_hash(root/"run_contract.json"),
        training_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
        old_B_eval_scenarios=[vars(s) for s in B_SCENARIOS],
        new_C_eval_scenarios=[vars(s) for s in C_SCENARIOS],
        methods=list(METHODS),
        frozen_display_sha256=file_hash(display_csv),
        planner_seconds=training_contract["plan_seconds"],
        response_floor=1.,
        evaluation_rule="nonmonotone applied to training-B frozen displays",
        matched_comparison="intersection of all arms and all scenarios",
    )
    file=folder/"extension_contract.json"
    if file.exists():
        previous=json.loads(file.read_text())
        if previous!=settings:
            raise ValueError("Extension inputs/scenarios changed. "
                             "Do not mix incomparable C evaluation cells.")
    else:
        old.save(file,settings)
    return folder


def evaluate_C(ctx,root,folder,cache,displays):
    """Only NEW nonmonotone cells require planner replay."""
    started=time.perf_counter()
    held=ctx["held"];items=ctx["eval"]
    booked=np.asarray(held["booked"],float)
    for scenario in C_SCENARIOS:
        target=folder/"evaluation_cells"/scenario.name
        target.mkdir(parents=True,exist_ok=True)
        for method in METHODS:
            cell=target/(method+".json")
            if cell.exists():
                check_cell(cell,scenario,method,items,extension=True)
                continue
            revised=(booked if method=="BOOKED" else
                     implemented_from_display(booked,displays[method],scenario))
            solved=cache.solve(items,revised,
                 "C_"+scenario.name+"_"+method,tie=True)
            entries=[]
            for key,week,rows in items:
                result=solved[key]
                score=evaluate_week(week,result["assignment"],week.actual)
                entries.append(dict(component=key,week=week.monday,
                   site=week.group,cases=week.n,method=method,
                   training_scenario=(method.split("__",1)[1]
                                      if "__" in method else None),
                   evaluation_scenario=scenario.name,
                   cost=float(score["cost"]),idle_minutes=float(score["idle"]),
                   overtime_minutes=float(score["overtime"]),
                   mae_sum=float(np.abs(week.actual-revised[rows]).sum()),
                   complete=bool(result["complete"]),
                   planner_status=result["status"]))
            old.save(cell,dict(method=method,evaluation=vars(scenario),
                              entries=entries))
            check_cell(cell,scenario,method,items,extension=True)
            print(f"C evaluation {scenario.name}/{method}: "
                  f"{sum(e['complete'] for e in entries)}/{len(items)} "
                  "certified",flush=True)
    old.record_stage_time(folder,"C_eval",time.perf_counter()-started)


def _get_entries(root,extension,scenario,method,items):
    path=(root/"evaluation_cells"/scenario.name/(method+".json")
          if scenario.link=="saturating" else
          extension/"evaluation_cells"/scenario.name/(method+".json"))
    return check_cell(path,scenario,method,items,
                      extension=scenario.link=="nonmonotone")


def clustered_interval(weekly, cases):
    keys=sorted(weekly)
    changes=np.asarray([weekly[k] for k in keys],float)
    n=np.asarray([cases[k] for k in keys],float)
    rng=np.random.default_rng(20261022)
    draw=rng.integers(0,len(keys),size=(10000,len(keys)))
    sims=changes[draw].sum(axis=1)/n[draw].sum(axis=1)
    return dict(estimate=float(changes.sum()/n.sum()),
                bootstrap_95=[float(x) for x in np.quantile(sims,[.025,.975])],
                better_weeks=int(np.sum(changes<0)),
                worse_weeks=int(np.sum(changes>0)),
                equal_weeks=int(np.sum(changes==0)),
                resamples=10000)


def report(ctx,root,folder):
    """C as PRIMARY analysis; B is preserved as comparator and sensitivity."""
    items=ctx["eval"]
    indices={}
    by={}
    certified={}
    expected={key for key,_,_ in items}
    for scenario in ALL_SCENARIOS:
        for method in METHODS:
            records=_get_entries(root,folder,scenario,method,items)
            data={r["component"]:r for r in records}
            if set(data)!=expected:
                raise ValueError("Inconsistent evaluation roster")
            by[scenario.name,method]=data
        certified[scenario.name]={
            key for key in expected if all(by[scenario.name,m][key]["complete"]
                                          for m in METHODS)}
    all_common=set.intersection(*(certified[s.name] for s in ALL_SCENARIOS))
    cases_by_key={key:week.n for key,week,rows in items}
    week_by_key={key:week.monday for key,week,rows in items}
    n_global=sum(cases_by_key[k] for k in all_common)
    if n_global==0:
        raise RuntimeError("No common-certified cases across B and C")
    summary=dict(
        status="EXPLORATORY_REUSED_2013",
        primary_evaluation="NONMONOTONE_C",
        secondary_evaluation="SATURATING_B",
        interpretation=("B is a training assumption; C is a deployment "
                        "stress mechanism, not empirically observed truth"),
        train_scenarios=[vars(s) for s in TRAIN_SCENARIOS],
        evaluation_scenarios=[vars(s) for s in ALL_SCENARIOS],
        arms=list(METHODS),
        all_scenarios_common_certified_components=len(all_common),
        all_scenarios_common_certified_cases=n_global,
        scenarios={},B_vs_C_global_matched={})

    grid=[]
    matched=[]
    for scenario in ALL_SCENARIOS:
        good=certified[scenario.name]
        n=sum(cases_by_key[k] for k in good)
        averages={}
        for method in METHODS:
            values=[by[scenario.name,method][key] for key in good]
            averages[method]=dict(
                cost_per_case=sum(x["cost"] for x in values)/n if n else None,
                mae_per_case=sum(x["mae_sum"] for x in values)/n if n else None,
                idle_minutes_per_case=sum(x["idle_minutes"] for x in values)/n if n else None,
                overtime_minutes_per_case=sum(x["overtime_minutes"] for x in values)/n if n else None,
                certified_components=sum(
                    by[scenario.name,method][key]["complete"] for key in expected),
                certified_cases=sum(
                    by[scenario.name,method][key]["cases"] for key in expected
                    if by[scenario.name,method][key]["complete"]))
        pairs=[("NAIVE","BOOKED")]
        pairs.extend((method,"BOOKED") for method in METHODS[2:])
        pairs.extend((method,"NAIVE") for method in METHODS[2:])
        pairs.extend((f"VF__{s.name}",f"CASE__{s.name}")
                     for s in TRAIN_SCENARIOS)
        diffs={}
        if n:
            case_by_week={}
            for key in good:
                w=week_by_key[key]
                case_by_week[w]=case_by_week.get(w,0)+cases_by_key[key]
            for method,baseline in pairs:
                wk={}
                for key in good:
                    w=week_by_key[key]
                    wk[w]=wk.get(w,0.)+(
                        by[scenario.name,method][key]["cost"]
                        -by[scenario.name,baseline][key]["cost"])
                diffs[method+"_minus_"+baseline]=clustered_interval(
                    wk,case_by_week)
        summary["scenarios"][scenario.name]=dict(
            kind=scenario.link,alpha=scenario.alpha,h=scenario.h,
            common_certified_components=len(good),
            common_certified_cases=n,
            total_components=len(expected),
            methods=averages,paired_cost_differences=diffs)
        grid.append([scenario.name,scenario.link,scenario.alpha,
                     scenario.h,n,*[averages[m]["cost_per_case"]
                                   for m in METHODS]])
        matched.append([scenario.name,scenario.link,scenario.alpha,
                        scenario.h,n_global,
                        *[sum(by[scenario.name,m][k]["cost"]
                              for k in all_common)/n_global for m in METHODS]])

    for B in B_SCENARIOS:
        C=next(s for s in C_SCENARIOS if s.alpha==B.alpha and s.h==B.h)
        panel={}
        counts={}
        for key in all_common:
            w=week_by_key[key]
            counts[w]=counts.get(w,0)+cases_by_key[key]
        for method in METHODS:
            differences={}
            for key in all_common:
                w=week_by_key[key]
                differences[w]=differences.get(w,0.)+(
                    by[C.name,method][key]["cost"]-
                    by[B.name,method][key]["cost"])
            panel[method]=clustered_interval(differences,counts)
        summary["B_vs_C_global_matched"][B.name]=dict(
            B=B.name,C=C.name,common_certified_cases=n_global,
            C_minus_B_by_method=panel)

    old.save(folder/"summary.json",summary)

    def write_grid(path,rows):
        tmp=path.with_suffix(".tmp")
        with tmp.open("w",newline="") as file:
            w=csv.writer(file)
            w.writerow(["evaluation_scenario","link","alpha","h",
                        "common_certified_cases",*METHODS])
            w.writerows(rows)
        tmp.replace(path)
    write_grid(folder/"cost_grid.csv",grid)
    write_grid(folder/"cost_grid_global_matched.csv",matched)
    print("C-primary/B-secondary report saved at",folder,
          f"; all-scenario common cases {n_global}",flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path(
        "artifacts/response_saturation_grid"),
        help="The EXISTING B-grid output directory; do not use a new one")
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--case-steps",type=int,default=200)
    ap.add_argument("--vf-outers",type=int,default=30)
    ap.add_argument("--vf-pdca-steps",type=int,default=50)
    ap.add_argument("--cuts",type=int,default=160)
    args=ap.parse_args()
    if not args.context.exists():
        raise FileNotFoundError("Missing existing UHN context "+str(args.context))
    if not (1<=args.cores<=15 and args.plan_seconds>0):
        ap.error("Invalid solver configuration")
    started=time.perf_counter()
    prior=validate_original_run(args,args.out)
    with args.context.open("rb") as file:
        ctx=pickle.load(file)
    cache=old.ContentCache(args.out,args.cores,args.plan_seconds)
    displays=existing_policies_or_resume(ctx,args.out,args,cache)
    ensure_B_results(ctx,args.out,cache,displays)
    folder=extension_contract(
        args.out,prior,args.out/"frozen_advice"/"evaluation_displays.csv")
    evaluate_C(ctx,args.out,folder,cache,displays)
    summary=report(ctx,args.out,folder)
    old.record_stage_time(folder,"extension_total",
                          time.perf_counter()-started)
    print(json.dumps(dict(primary=summary["primary_evaluation"],
        secondary=summary["secondary_evaluation"],
        common_cases=summary["all_scenarios_common_certified_cases"],
        summary=str(folder/"summary.json")),indent=2),flush=True)


if __name__=="__main__":
    main()
