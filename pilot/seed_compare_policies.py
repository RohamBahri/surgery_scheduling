#!/usr/bin/env python3
"""Five-explicit-seed comparison of already frozen joint VF policies.

Uses ARCHIVED cases_test.csv displays / reviewed durations, never recomputes
features or refits coefficients. Solves the same deployment problem using
explicit hash seeds in pilot.paper_seed_audit.run_seed. Cached plans contain
seed- and duration-specific signatures. 2013 results are EXPLORATORY.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from pilot.paper_seed_audit import run_seed
from pilot.run_nonmonotone_learning import save
from surgery.nonmonotone_learning import ResponseDC,response_saturating
from surgery.weekly_planner import evaluate

METHODS=("BOOKED","CASE","VF_clean","VF_structural")
SEEDS=(0,1,2,3,4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_case_table(ctx,folder,required_schema):
    """Load and verify the exact archived advice against model, cohort and link."""
    selected=read(folder/"selected_policy.json")
    features=read(folder/"vf_features.json")
    if (selected.get("method")!="JOINT_VF" or selected.get("link")!="saturating"
            or features.get("schema")!=required_schema
            or selected.get("vf_features")!=required_schema
            or not np.isclose(selected.get("alpha",0),.8)
            or not np.isclose(selected.get("h",0),30.)):
        raise ValueError(f"Not the frozen {required_schema} nominal joint VF: {folder}")
    with (folder/"cases_test.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    held=ctx["held"];ids=np.asarray(held["case_id"],int)
    booked=np.asarray(held["booked"],float)
    if len(records)!=len(ids) or not np.array_equal(
            np.array([int(r["case_id"]) for r in records]),ids):
        raise AssertionError(f"{folder}: held-out case IDs/order differ")
    def arr(name):
        return np.array([float(r[name]) for r in records],float)
    if not np.allclose(arr("booked"),booked,rtol=0,atol=1e-9):
        raise AssertionError(f"{folder}: booked durations differ")
    if not np.allclose(arr("actual"),held["actual"],rtol=0,atol=1e-9):
        raise AssertionError(f"{folder}: realized duration audit mismatch")
    response=ResponseDC(booked,.8,30.,180.,1.,link="saturating")
    case_raw=arr("case_raw");vf_raw=arr("vf_raw")
    ca=arr("case_display");vf=arr("vf_display")
    # No policy may be silently recomputed with different feature scaling.
    x=np.asarray(held["X"],float)
    if len(selected["case_coeff"])!=x.shape[1]:
        raise AssertionError("CASE coefficient count mismatch")
    if not np.allclose(case_raw,x@np.asarray(selected["case_coeff"]),
                       rtol=0,atol=1e-7):
        raise AssertionError("Stored CASE raw advice differs from frozen coefficients")
    if not np.allclose(ca,response.display(case_raw),rtol=0,atol=1e-8):
        raise AssertionError("Stored CASE display differs from canonical advice")
    if not np.allclose(vf,response.display(vf_raw),rtol=0,atol=1e-8):
        raise AssertionError("Stored VF display differs from canonical advice")
    if not np.allclose(arr("case_reviewed"),response.reviewed(case_raw),
                       rtol=0,atol=1e-8):
        raise AssertionError("Stored CASE reviewed durations differ from response")
    if not np.allclose(arr("vf_reviewed"),response.reviewed(vf_raw),
                       rtol=0,atol=1e-8):
        raise AssertionError("Stored VF reviewed durations differ from response")
    for d,display in ((arr("case_reviewed"),ca),(arr("vf_reviewed"),vf)):
        predicted=np.maximum(1.,booked+response_saturating(display,.8,30.))
        if not np.allclose(d,predicted,rtol=0,atol=1e-8):
            raise AssertionError("Stored deployment response differs from saturating link")
    return dict(case=arr("case_reviewed"),vf=arr("vf_reviewed"),
                case_coeff=np.asarray(selected["case_coeff"],float),
                case_display=ca,vf_display=vf)


def paired_over_seeds(costs,cases,weeks,candidate,reference,seeds,
                      draws=10000,bootstrap_seed=20261028):
    """Resample calendar weeks, keeping each seed's paired components together."""
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    cases=np.asarray(cases,float)
    weeks=list(weeks)
    if len(weeks)==0 or len(cases)!=len(weeks) or np.any(cases<=0):
        raise ValueError("Invalid matched component roster")
    uniq=sorted(set(weeks));index={w:i for i,w in enumerate(uniq)}
    indices=np.array([index[w] for w in weeks])
    numerator=np.zeros(len(uniq),float)
    each={};differences={}
    for seed in seeds:
        diff=np.asarray(costs[seed,candidate],float)-np.asarray(
            costs[seed,reference],float)
        if len(diff)!=len(cases):
            raise ValueError("Mismatched cost and case arrays")
        each[str(seed)]=float(diff.sum()/cases.sum())
        differences[str(seed)]=int(np.sum(np.abs(diff)>1e-7))
        numerator+=np.bincount(indices,weights=diff,minlength=len(uniq))/len(seeds)
    den=np.bincount(indices,weights=cases,minlength=len(uniq))
    rng=np.random.default_rng(bootstrap_seed)
    draw=rng.integers(0,len(uniq),(draws,len(uniq)))
    boot=numerator[draw].sum(axis=1)/den[draw].sum(axis=1)
    return dict(mean_over_seeds=float(numerator.sum()/den.sum()),
        paired_calendar_week_bootstrap_95=[float(t) for t in
                                            np.quantile(boot,[.025,.975])],
        per_seed=each,changed_components_by_seed=differences,
        negative_seed_count=int(sum(x<0 for x in each.values())),
        better_calendar_weeks=int(sum(numerator<0)),
        worse_calendar_weeks=int(sum(numerator>0)),
        zero_calendar_weeks=int(sum(numerator==0)),
        n_calendar_weeks=len(uniq))


def check_seed0_archive(cells,folder,method,common,tolerance=1e-6):
    """Detect changed deployment semantics, not just coincidental aggregate cost."""
    rows=read(folder/"evaluation_2013.json")["entries"]
    archive={r["component"]:r for r in rows
             if r["method"]==method and r["complete"]}
    keys=sorted(set(common)&set(archive))
    deviations={key:cells[0]["cost"][key]-float(archive[key]["cost"])
                for key in keys}
    changed={k:v for k,v in deviations.items() if abs(v)>tolerance}
    return dict(verified_components=len(keys),changed_cost_components=len(changed),
                max_abs_cost_difference=max((abs(v) for v in deviations.values()),default=0.),
                examples=list(changed.items())[:8])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--clean",type=Path,required=True)
    ap.add_argument("--structural",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    if 0 not in args.seeds or len(args.seeds)!=len(set(args.seeds)) or min(args.seeds)<0:
        ap.error("Distinct nonnegative hash seeds including 0 are required")
    for attr in ("context","clean","structural","out"):
        setattr(args,attr,getattr(args,attr).resolve())
    for folder in (args.clean,args.structural):
        for filename in ("selected_policy.json","vf_features.json",
                         "cases_test.csv","evaluation_2013.json"):
            if not (folder/filename).exists():
                raise FileNotFoundError(folder/filename)
    args.out.mkdir(parents=True,exist_ok=True)
    scientific=dict(
        run="frozen_policy_explicit_seed_comparison_v1",
        context_sha256=sha(args.context),
        clean_sha256=sha(args.clean/"selected_policy.json"),
        structural_sha256=sha(args.structural/"selected_policy.json"),
        clean_displays_sha256=sha(args.clean/"cases_test.csv"),
        structural_displays_sha256=sha(args.structural/"cases_test.csv"),
        seeds=args.seeds,plan_seconds=args.plan_seconds)
    contract=args.out/"seed_contract.json"
    if contract.exists() and read(contract)!=scientific:
        raise ValueError("Seed comparison contract changed; use a NEW --out")
    save(contract,scientific)
    with args.context.open("rb") as f:ctx=pickle.load(f)
    clean=frozen_case_table(ctx,args.clean,"clean")
    structural=frozen_case_table(ctx,args.structural,"structural")
    if not np.allclose(clean["case_coeff"],structural["case_coeff"],
                       rtol=0,atol=1e-10):
        raise AssertionError("CASE initializers differ: unsuitable common comparator")
    if not np.allclose(clean["case"],structural["case"],rtol=0,atol=1e-8):
        raise AssertionError("CASE reviewed duration differs between experiments")
    b=np.asarray(ctx["held"]["booked"],float)
    durations=dict(BOOKED=b,CASE=clean["case"],
                   VF_clean=clean["vf"],VF_structural=structural["vf"])
    items=ctx["eval"];cells={}
    for seed in args.seeds:
        for label,d in durations.items():
            plans=run_seed(items,d,args.out,label,seed,args.cores,args.plan_seconds)
            costs={};certified={};assignments={}
            for key,c,rows in items:
                record=plans[key]
                costs[key]=float(evaluate(c,record["assignment"],c.actual)["cost"])
                certified[key]=bool(record["complete"])
                assignments[key]=record["assignment"]
            cells[seed,label]=dict(cost=costs,certified=certified,
                                   assignment=assignments)
    valid=sorted(key for key,c,_ in items
                 if all(cells[s,m]["certified"][key]
                        for s in args.seeds for m in METHODS))
    if not valid:raise RuntimeError("No jointly certified seed comparisons")
    roster={key:(c.n,c.monday) for key,c,_ in items}
    cases=[roster[k][0] for k in valid]
    weeks=[roster[k][1] for k in valid]
    costs={(seed,label):[cells[seed,label]["cost"][k] for k in valid]
           for seed in args.seeds for label in METHODS}
    total_cases=int(sum(cases));all_cases=int(sum(c.n for _,c,_ in items))
    results=dict(
        status="EXPLORATORY_2013_ALREADY_REUSED",
        common_components=len(valid),common_cases=total_cases,
        all_components=len(items),all_cases=all_cases,
        excluded_cases=all_cases-total_cases,seeds=args.seeds,
        certification={str(seed):{label:sum(
            cells[seed,label]["certified"].values()) for label in METHODS}
            for seed in args.seeds},
        costs={label:{str(seed):float(sum(costs[seed,label])/total_cases)
                      for seed in args.seeds} for label in METHODS},
        comparisons={},
        seed0_reproduction={
            "clean":check_seed0_archive(cells[0,"VF_clean"],args.clean,"VF",valid),
            "structural":check_seed0_archive(
                cells[0,"VF_structural"],args.structural,"VF",valid),
            "CASE":check_seed0_archive(cells[0,"CASE"],args.clean,"CASE",valid),
            "BOOKED":check_seed0_archive(cells[0,"BOOKED"],args.clean,"BOOKED",valid)},
        frozen_duration_hashes={
            name:hashlib.sha256(np.asarray(d,dtype="<f8").tobytes()).hexdigest()
            for name,d in durations.items()})
    for name,ref in (("structural_minus_clean","VF_clean"),
                     ("structural_minus_CASE","CASE"),
                     ("structural_minus_BOOKED","BOOKED"),
                     ("clean_minus_CASE","CASE")):
        target="VF_clean" if name=="clean_minus_CASE" else "VF_structural"
        results["comparisons"][name]=paired_over_seeds(
            costs,cases,weeks,target,ref,args.seeds)
    # Do not advertise a false positive if seed-0 deploy differs from archive.
    bad={k:v for k,v in results["seed0_reproduction"].items()
         if v["changed_cost_components"]}
    if bad:
        save(args.out/"seed_reproduction_failure.json",bad)
        raise AssertionError("Seed 0 does not reproduce frozen archived costs; "+
                             "see seed_reproduction_failure.json")
    save(args.out/"seed_comparison.json",results)
    x=results["comparisons"]["structural_minus_clean"]
    print("Seed-averaged structural - clean:",x["mean_over_seeds"],
          "week bootstrap:",x["paired_calendar_week_bootstrap_95"],
          "negative seeds:",x["negative_seed_count"],"/",len(args.seeds),
          "matched cases:",total_cases,flush=True)
    print("Saved "+str(args.out/"seed_comparison.json"),flush=True)


if __name__=="__main__":
    main()
