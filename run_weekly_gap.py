#!/usr/bin/env python3
"""Full 180-group-week VF-GAP training and 54-group-week paired evaluation.

Reuses frozen rich X and the fitted CASE initialization. Never reads 2013
holdout outcomes for training/selection. Runs exact two-oracle certificates
on all 180 training group-weeks for each accepted candidate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from run_weekly_rich import load_matrices, source_fingerprint, verify_bounds
from run_weekly_train import load_weeks, check_week_cases, evaluate_week
from surgery.planner import display_and_duration
from surgery.weekly_planner import solve_adversary, solve_week, evaluate as evaluate_schedule
from surgery.weekly_training import save_json
from surgery.gap_learning import train_gap


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(args):
    meta, frames=load_matrices(args.artifacts)
    settings=json.loads((args.artifacts/"learning_settings.json").read_text())
    if settings["feature_code_sha256"] != source_fingerprint():
        raise ValueError("Existing CASE training used a different feature implementation")
    if (settings["workbook_sha256"] != file_sha(args.workbook)
            or settings["matrix_sha256"] != meta["matrix_sha256"]
            or settings["feature_names"] != meta["feature_names"]
            or settings["train_cases"] != len(frames["train"]["case_id"])
            or len(frames["validation"]["case_id"]) != 5779):
        raise ValueError("Workbook, policy or matrix does not match rich training artifacts")
    if not (np.isclose(args.alpha,settings["alpha"])
            and np.isclose(args.h,settings["h"])):
        raise ValueError("Use the same surgeon-response scenario as CASE training")
    if args.gamma < 1 or not np.isfinite(args.gamma):
        raise ValueError("VF-GAP gamma must be finite and >= 1")
    if not (np.isfinite(args.trust) and args.trust > 0
            and np.isfinite(args.temperature) and args.temperature > 0):
        raise ValueError("Require strictly positive finite trust and temperature")
    models={}
    for name, filename in (
            ("CASE","model_case.json"),("LASSO","lasso_policy.json"),
            ("VF","vf_progress.json")):
        models[name]=json.loads((args.artifacts/filename).read_text())
        if len(models[name]["w" if name != "LASSO" else "weights"]) != len(meta["feature_names"]):
            raise ValueError(f"{name} has different policy dimension")
    if models["CASE"].get("settings") != settings:
        raise ValueError("CASE model does not match learning settings")
    if models["VF"].get("complete") is not True:
        raise ValueError("VF baseline training is not complete")
    if models["VF"].get("settings") != settings:
        raise ValueError("VF model settings differ from CASE")
    if models["LASSO"]["features"] != meta["feature_names"]:
        raise ValueError("LASSO feature order differs")
    init=models[args.initial]["weights" if args.initial=="LASSO" else "w"]
    source_root=Path(__file__).resolve().parent
    trainer_sha=hashlib.sha256()
    for name in ("run_weekly_gap.py","surgery/gap_learning.py"):
        trainer_sha.update(name.encode())
        trainer_sha.update((source_root/name).read_bytes())
    identity={
        "trainer_source_sha256":trainer_sha.hexdigest(),
        "feature_source_sha256":source_fingerprint(),
        "workbook_sha256":file_sha(args.workbook),
        "matrix_sha256":meta["matrix_sha256"],
        "training_group_weeks":180,"validation_group_weeks":54,
        "features":meta["feature_names"],
        "policy":"fixed booked anchor + 74 learned correction coefficients",
        "initial":args.initial,
        "initial_model_sha256":file_sha(args.artifacts/({
            "CASE":"model_case.json","LASSO":"lasso_policy.json",
            "VF":"vf_progress.json"}[args.initial])),
        "alpha":float(args.alpha),"h":float(args.h),
        "gamma":float(args.gamma),
        "lambda_l1":float(settings["lambda_l1"]),
        "max_outer":int(args.max_outer),
        "inner_iterations":int(args.iterations),
        "trust":float(args.trust),
        "temperature":float(args.temperature),
        "loss_normalization":"all 18705 training cases",
        "evaluation_selection":"training-only; validation is assessment",
    }
    args.output.mkdir(parents=True,exist_ok=True)
    destination=args.output/"gap_settings.json"
    if destination.exists():
        if json.loads(destination.read_text())!=identity:
            raise ValueError("VF-GAP run output has different settings or trainer source")
    else:
        save_json(destination,identity)
    return meta,frames,models,identity,np.asarray(init,float)


def fit(args):
    meta,frames,models,ident,initial=initialize(args)
    weeks=load_weeks(args.workbook,meta,"train")
    indices=check_week_cases(weeks,frames["train"])
    if len(weeks)!=180 or len(frames["train"]["case_id"])!=18705:
        raise ValueError("VF-GAP requires all 180 training group-weeks and 18705 cases")
    d=frames["train"]
    result=train_gap(args.output,weeks,indices,
                     d["X"].astype(float),d["booked"],
                     float(ident["lambda_l1"]),initial,ident,
                     args.alpha,args.h,args.gamma,args.max_outer,
                     args.iterations,args.trust,args.temperature,
                     args.seconds,args.threads,args.workers)
    print("VF-GAP training completed:",result["stopping"],flush=True)
    print("Certified training improvement records:",
          [(r["certified_before"]["regularized_gap_without_hindsight_per_case"],
            r["certified_after"]["regularized_gap_without_hindsight_per_case"],
            r["accepted"]) for r in result["history"]],flush=True)


def evaluate_week_gap(task):
    week,durations,seconds,threads,gamma=task
    rows=evaluate_week((week,durations,seconds,threads))
    if not all(r["complete"] for r in rows):
        for row in rows:
            row["vf_gap_bound"]=float("nan")
        return rows
    reviewed=dict(zip(map(int,week.case_ids),map(float,durations["VF_GAP"])))
    adversarial, primary_cost, realized_primary = 0., 0., 0.
    for comp in week.components():
        d=np.array([reviewed[int(i)] for i in comp.case_ids],float)
        primary=solve_week(comp,d,seconds=seconds,threads=threads,tie=False)
        a=solve_adversary(comp,d,gamma=gamma,seconds=seconds,threads=threads)
        if (not primary.complete or primary.assignment is None
                or not a.complete or a.assignment is None):
            for row in rows:
                row["complete"]=False
                row["vf_gap_bound"]=float("nan")
                row["gap_failure"]=(f"{comp.group} {comp.monday}: "
                    f"primary={primary.status}, adversary={a.status}")
            return rows
        primary_cost+=evaluate_schedule(comp,primary.assignment,d)["cost"]
        realized_primary+=evaluate_schedule(comp,primary.assignment,comp.actual)["cost"]
        adversarial+=a.value
    gap_row=next(row for row in rows if row["method"]=="VF_GAP")
    oracle=gap_row["oracle_cost"]
    # This is the mathematical exact primary-value VF-GAP bound.
    bound=gamma*primary_cost+adversarial-oracle
    vf_primary=primary_cost+gap_row["asym_error"]-oracle
    if bound < realized_primary-oracle-.02:
        raise AssertionError("Primary optimum regret exceeds certified VF-GAP")
    if bound > vf_primary+.02:
        raise AssertionError("Exact primary VF-GAP exceeds original VF bound")
    # Deployment applies additional tie-breaking within a small primary
    # tolerance. This bound accounts for that *specific* selected schedule.
    observed_bound=gamma*gap_row["pred_cost"]+adversarial-oracle
    if observed_bound < gap_row["regret"]-.02:
        raise AssertionError("Observed schedule regret exceeds tie-safe GAP bound")
    if primary_cost > gap_row["pred_cost"]+.02:
        raise AssertionError("Tie-breaking schedule cheaper than certified primary")
    for row in rows:
        row["vf_gap_bound"]=bound if row["method"]=="VF_GAP" else float("nan")
        row["vf_gap_bound_tie_safe"]=(observed_bound if row["method"]=="VF_GAP"
                                      else float("nan"))
        row["gap_primary_predicted_cost"]=(primary_cost if row["method"]=="VF_GAP"
                                            else float("nan"))
    return rows


def evaluate(args):
    meta,frames,models,ident,_=initialize(args)
    fitted_path=args.output/"gap_progress.json"
    if not fitted_path.exists():
        raise ValueError("Run --stage train before evaluating VF-GAP")
    model=json.loads(fitted_path.read_text())
    if not model.get("complete") or model["settings"]!=ident:
        raise ValueError("VF-GAP training incomplete or mismatched")
    valid=frames["validation"]
    X=valid["X"].astype(float)
    policies={
        "LASSO":models["LASSO"]["weights"],
        "CASE":models["CASE"]["w"],
        "VF":models["VF"]["w"],
        "VF_GAP":model["w"],
    }
    bookings=valid["booked"]
    predictions={"BOOKED":bookings.copy()}
    for name,w in policies.items():
        _,post=display_and_duration(X@np.asarray(w,float),bookings,args.alpha,args.h)
        predictions[name]=post
    weeks=load_weeks(args.workbook,meta,"validation")
    idx=check_week_cases(weeks,valid)
    if len(weeks)!=54:
        raise ValueError("Require exactly 54 validation group-weeks")
    identity={"training":ident,
              "trained_model_sha256":file_sha(fitted_path),
              "baseline_model_sha256":{
                  key:file_sha(args.artifacts/filename)
                  for key,filename in (
                    ("CASE","model_case.json"),
                    ("VF","vf_progress.json"),
                    ("LASSO","lasso_policy.json"))}}
    settings_file=args.output/"gap_evaluation_settings.json"
    if settings_file.exists():
        if json.loads(settings_file.read_text())!=identity:
            raise ValueError("Validation scores belong to different policies")
    else:
        save_json(settings_file,identity)
    path=args.output/"gap_evaluation.csv"
    records={}
    if path.exists():
        prev=pd.read_csv(path)
        for row in prev.to_dict("records"):
            key=(str(row["group"]),str(row["monday"]),str(row["method"]))
            if key in records:
                raise ValueError("Duplicated validation method-week")
            records[key]=row
    tasks=[]
    for week in weeks:
        rows=[records.get((week.group,week.monday,name)) for name in predictions]
        digests={r.get("oracle_assignment_sha256") for r in rows
                 if r is not None and r.get("complete")==True}
        if (all(r is not None and r.get("complete")==True for r in rows)
                and len(digests)==1 and all(
                    isinstance(d,str) and len(d)==64 for d in digests)
                and np.isfinite(records[(week.group,week.monday,"VF_GAP")].get(
                    "vf_gap_bound",float("nan")))
                and np.isfinite(records[(week.group,week.monday,"VF_GAP")].get(
                    "vf_gap_bound_tie_safe",float("nan")))):
            continue
        for name in predictions:
            records.pop((week.group,week.monday,name),None)
        loc=np.array([idx[int(i)] for i in week.case_ids],int)
        local={name:p[loc] for name,p in predictions.items()}
        tasks.append((week,local,args.seconds,args.threads,args.gamma))

    def save(rows):
        for row in rows:
            records[(row["group"],row["monday"],row["method"])]=row
        df=pd.DataFrame([records[k] for k in sorted(records)])
        tmp=path.with_suffix(".tmp")
        df.to_csv(tmp,index=False)
        tmp.replace(path)
    print(f"Evaluation: {len(tasks)} pending group-weeks, "
          f"{args.workers} worker(s) x {args.threads} Gurobi threads",flush=True)
    if args.workers==1:
        for task in tasks:
            save(evaluate_week_gap(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                                 mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(evaluate_week_gap,task) for task in tasks]
            for future in as_completed(futures):
                save(future.result())
    expected=len(weeks)*len(predictions)
    if (len(records)!=expected or any(r.get("complete")!=True for r in records.values())
            or not all(np.isfinite(r["vf_gap_bound"]) and
                       np.isfinite(r.get("vf_gap_bound_tie_safe",float("nan")))
                       for r in records.values() if r["method"]=="VF_GAP")):
        raise RuntimeError("Validation not fully certified; rerun with larger --seconds")
    summarize_gap(args.output,pd.DataFrame(records.values()))


def summarize_gap(root,df):
    """Bootstrap entire calendar weeks; three sites are paired within each week."""
    result={"methods":{},"by_group":{}}
    for name,frame in df.groupby("method"):
        n=frame.cases.sum()
        result["methods"][name]={
            "cases":int(n),"group_weeks":int(len(frame)),
            "mae":float(frame.abs_error.sum()/n),
            "regret_per_case":float(frame.regret.sum()/n),
            "realized_cost":float(frame.realized_cost.sum()),
            "overtime_minutes":float(frame.overtime.sum()),
            "overloaded_room_days":int(frame.room_days_over_720.sum()),
        }
    for (group,method),frame in df.groupby(["group","method"]):
        result["by_group"].setdefault(group,{})[method]={
            "regret_per_case":float(frame.regret.sum()/frame.cases.sum()),
            "realized_cost":float(frame.realized_cost.sum()),
        }
    result["vf_gap_bound_per_case"]=float(df.loc[df.method=="VF_GAP","vf_gap_bound"].sum()/
                                            df.loc[df.method=="VF_GAP","cases"].sum())
    result["tie_safe_gap_bound_per_case"]=float(
        df.loc[df.method=="VF_GAP","vf_gap_bound_tie_safe"].sum()/
        df.loc[df.method=="VF_GAP","cases"].sum())
    pivot=df.pivot(index=["monday","group"],columns="method",values="realized_cost")
    monthly=pivot.groupby(level="monday").sum()
    weeks=len(monthly)
    if weeks!=18 or set(monthly.columns)!=set(("BOOKED","LASSO","CASE","VF","VF_GAP")):
        raise ValueError("Expected exactly 18 fully paired calendar weeks")
    rng=np.random.default_rng(20261013)
    draws=rng.integers(0,weeks,(20000,weeks))
    result["paired_week_bootstrap"]={}
    for base in ("BOOKED","CASE","LASSO","VF"):
        delta=(monthly["VF_GAP"]-monthly[base]).to_numpy(float)
        denominator=monthly[base].to_numpy(float)
        pct=100*delta[draws].sum(axis=1)/denominator[draws].sum(axis=1)
        result["paired_week_bootstrap"]["VF_GAP_minus_"+base]={
            "cost_change":float(delta.sum()),
            "percent_change":float(100*delta.sum()/denominator.sum()),
            "ci_95_percent":[float(x) for x in np.quantile(pct,[.025,.975])],
            "n_calendar_weeks":weeks,
        }
    result["interpretation"]="Development validation only; 2013 holdout untouched."
    save_json(Path(root)/"gap_evaluation_summary.json",result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook",type=Path)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=("train","evaluate"),required=True)
    p.add_argument("--initial",choices=("CASE","VF","LASSO"),default="CASE")
    p.add_argument("--alpha",type=float,default=.8)
    p.add_argument("--h",type=float,default=30.)
    p.add_argument("--gamma",type=float,default=2.)
    p.add_argument("--max-outer",type=int,default=5)
    p.add_argument("--iterations",type=int,default=150)
    p.add_argument("--trust",type=float,default=6.)
    p.add_argument("--temperature",type=float,default=1.5)
    p.add_argument("--workers",type=int,default=15)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--seconds",type=float,default=300.)
    args=p.parse_args()
    if (args.workers<1 or args.threads<1 or args.workers*args.threads>15 or
        args.max_outer<1 or args.iterations<1 or args.seconds<=0 or
        not 0<args.alpha<=1 or args.h<=0):
        p.error("Require positive budgets, alpha in (0,1], and workers*threads <= 15")
    if not args.artifacts.is_dir():
        p.error("Rich-feature artifacts directory does not exist")
    if args.stage=="train":
        fit(args)
    else:
        evaluate(args)


if __name__=="__main__":
    main()
