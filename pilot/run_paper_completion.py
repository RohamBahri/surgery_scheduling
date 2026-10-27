#!/usr/bin/env python3
"""One-command, unattended, restartable paper finalization experiment.

Does not tune features, response distributions, or L1 using 2013.
Requires UHN context and an existing, frozen lambda025 baseline.
Writes stage logs and summary files under --out. Re-run the identical
command to resume completed/cached stages after an interruption.

Stages: regression tests, nominal theory audit, duplicate-free joint VF,
true case-only VF ablation, 4x6 response-scenario grid, training-selected
minimax/average and fixed-surgeon response stress test, hash tie seeds 0-2,
and one consolidated empirical/theory report.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOMINAL_LAMBDA=.10924129431614196


def read(path):
    return json.loads(path.read_text())


def invoke(name,cmd,root,*,cwd=ROOT):
    root.mkdir(parents=True,exist_ok=True)
    log=root/(name+".log")
    print(f"\n=== {name} ===",flush=True)
    print("$ "+" ".join(map(str,cmd)),flush=True)
    started=time.monotonic()
    with log.open("a") as file:
        file.write("\n=== Resume/start at "+time.strftime("%Y-%m-%d %H:%M:%S")+" ===\n")
        file.flush()
        with subprocess.Popen([str(x) for x in cmd],cwd=cwd,stdout=file,
                              stderr=subprocess.STDOUT,env=os.environ.copy()) as job:
            code=job.wait()
    if code!=0:
        tail=log.read_text(errors="replace").splitlines()[-35:]
        print("\n".join(tail),file=sys.stderr)
        raise RuntimeError(f"{name} failed (exit {code}). Log: {log}")
    print(f"{name} completed; log: {log}",flush=True)


def ensure_initializer(base,out):
    out.mkdir(parents=True,exist_ok=True)
    donor=base/"case_policy.json"
    destination=out/"case_policy.json"
    if not donor.exists():
        raise FileNotFoundError("Missing CASE initializer "+str(donor))
    if destination.exists():
        if destination.read_bytes()!=donor.read_bytes():
            raise RuntimeError("Initializer differs from the frozen baseline: "+str(out))
    else:
        shutil.copy2(donor,destination)


def ablation(args,code,schema,lam):
    folder=args.out/schema
    ensure_initializer(args.base,folder)
    common=[sys.executable,"-u",ROOT/"pilot"/"run_nonmonotone_learning.py",
            "--context",args.context,"--out",folder,"--cores",args.cores,
            "--plan-seconds",args.plan_seconds,"--link","saturating",
            "--vf-features",schema,"--vf-lambda-absolute",repr(lam)]
    for stage,target in (("all","selected_policy.json"),
                         ("evaluate","summary.json"),
                         ("probe","train_probe.json")):
        if (folder/target).exists():
            continue
        invoke("fit_"+schema+"_"+stage,[*common,"--stage",stage],args.out/"logs")


def bootstrap_difference(a,b,*,seed=20261027):
    """Paired calendar-week, case-weighted difference on common components."""
    x=read(a/"evaluation_2013.json")["entries"]
    y=read(b/"evaluation_2013.json")["entries"]
    def select(rows):
        return {r["component"]:r for r in rows if r["method"]=="VF" and r["complete"]}
    ax=select(x);by=select(y)
    keys=sorted(set(ax)&set(by))
    if not keys:raise RuntimeError("No matched VF evaluation components")
    grouped={}
    for key in keys:
        r=ax[key];v=by[key]
        if r["cases"]!=v["cases"] or r["week"]!=v["week"]:
            raise AssertionError("VF populations differ between feature sets")
        z=grouped.setdefault(r["week"],[0.,0.])
        z[0]+=r["cost"]-v["cost"]
        z[1]+=r["cases"]
    weeks=sorted(grouped)
    import numpy as np
    difference=np.asarray([grouped[w][0] for w in weeks])
    counts=np.asarray([grouped[w][1] for w in weeks])
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(weeks),size=(10000,len(weeks)))
    samples=difference[picks].sum(axis=1)/counts[picks].sum(axis=1)
    return dict(cost_difference_per_case=float(difference.sum()/counts.sum()),
        calendar_week_bootstrap_95=[float(v) for v in
                                    np.quantile(samples,[.025,.975])],
        common_components=len(keys),common_cases=int(counts.sum()),
        nonidentical_components=sum(abs(ax[k]["cost"]-by[k]["cost"])>1e-7
                                    for k in keys))


def report(args,lam):
    base=read(args.base/"summary.json")
    audit=read(args.out/"audit"/"numerical_audit.json")
    seed=read(args.out/"seed_audit"/"seed_robustness.json")
    response=read(args.out/"scenarios"/"paper_response_summary.json")
    records={}
    for label in ("clean","case_only"):
        directory=args.out/label
        selected=read(directory/"selected_policy.json")
        evaluation=read(directory/"summary.json")
        train=read(directory/"train_probe.json")
        records[label]=dict(
            features=selected.get("vf_feature_names"),
            parameters=len(selected["vf_theta"]),
            train_cost_per_case=train["VF"]["cost_per_case"],
            test_cost_per_case=evaluation["VF"]["cost_per_case"],
            test_minus_original=bootstrap_difference(directory,args.base),
            test_certified=evaluation["common_certified_components"])
    result=dict(
        status="FINISHED",
        label="Exploratory 2013; re-used during method development",
        baseline_lambda=lam,
        baseline_training_cases=audit["train_cases"],
        baseline_test_cost_per_case=base["VF"]["cost_per_case"],
        booked_test_cost_per_case=base["BOOKED"]["cost_per_case"],
        ablations=records,
        response_selection=response["selection"],
        response_selection_table=response["selection_table"],
        response_heterogeneity=response["heterogeneity"],
        seed_audit=seed["seeds"],
        theory_audit=str(args.out/"audit"/"theory_audit.md"))
    (args.out/"final_results.json").write_text(json.dumps(result,indent=2))
    lines=["# Joint VF final paper checks","",
           "These results are exploratory. The 2013 cohort was used repeatedly "
           "during development, and response scenarios are not estimated surgeon "
           "behavior.","",
           "## Frozen baseline","",
           f"Joint VF, saturating (α=0.8,h=30), λ={lam:.12f}, uniform L1.",
           f"2013 cost/case: {base['VF']['cost_per_case']:.5f}; "
           f"BOOKED {base['BOOKED']['cost_per_case']:.5f}.","",
           "## Feature representation and case-only ablation","",
           "| Policy | Coefficients | Training cost/case | 2013 cost/case | "
           "Difference vs original (95% paired-week CI) |",
           "|---|---:|---:|---:|---|"]
    for label,v in records.items():
        contrast=v["test_minus_original"]
        lines.append(f"| {label} | {v['parameters']} | "
            f"{v['train_cost_per_case']:.3f} | {v['test_cost_per_case']:.3f} | "
            f"{contrast['cost_difference_per_case']:+.3f} "
            f"[{contrast['calendar_week_bootstrap_95'][0]:+.3f},"
            f"{contrast['calendar_week_bootstrap_95'][1]:+.3f}] |")
    lines += ["","## Policy selection under response uncertainty","",
        "The training-period matrix selects candidate policies by minimax "
        "regret and equal-weight mean cost. No probabilities are inferred.",
        "Regret is relative to the four trained VF candidates.",
        "",
        "| Rule | Training response | Test mean cost | Test worst regret | "
        "Scenario losses versus BOOKED |",
        "|---|---|---:|---:|---:|"]
    for row in response["selection_table"]:
        lines.append(f"| {row['selection']} | {row['trained_policy']} | "
          f"{row['test_mean_cost']:.3f} | "
          f"{row['test_max_candidate_regret']:.3f} | "
          f"{6-row['test_scenarios_better_than_booked']}/6 |")
    lines += ["","The 2013 scenario matrix and heterogeneity cost tables are in "
        "`scenarios/`. The training selection is fixed before consulting "
        "2013 scenario outcomes.","",
        "## Numerical and theoretical claims","",
        "See `audit/theory_audit.md` and `audit/claims_register.csv`. "
        "MM descent and certified inner optimization do NOT establish global "
        "optimality or a formally certified stationary point.","",
        "## Hash tie sensitivity","",
        "Frozen BOOKED, CASE and original joint VF advice is identical across "
        "hash seeds; only the scheduler's tie selection changes. "
        "See `seed_audit/seed_robustness.json`.","",
        "## Scope decisions","",
        "No extra feature search, no retrained probabilistic mixture policy, "
        "no fitting of surgeon response parameters from observational data. "
        "The heterogeneous-response experiment is a hypothetical sensitivity "
        "analysis; robust candidate choice is a diagnostic, not a guarantee "
        "against all possible behavioral distributions.",""]
    (args.out/"final_report.md").write_text("\n".join(lines))
    print("\nFinal report: "+str(args.out/"final_report.md"),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--base",type=Path,
        default=Path("artifacts/vf_tuning/lambda025"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/paper_completion"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    args.context=args.context.resolve()
    args.base=args.base.resolve()
    args.out=args.out.resolve()
    if not args.context.exists():
        raise FileNotFoundError("Missing private context checkpoint "+str(args.context))
    if importlib.util.find_spec("gurobipy") is None:
        raise RuntimeError("Gurobi Python package required; run in your working venv")
    selected=read(args.base/"selected_policy.json")
    lam=float(selected["lambda_vf"])
    if (selected.get("method")!="JOINT_VF" or selected.get("link")!="saturating"
            or len(selected["vf_theta"])!=82
            or abs(lam-NOMINAL_LAMBDA)>1e-8):
        raise ValueError("Invalid frozen λ×0.25 joint VF reference")
    args.out.mkdir(parents=True,exist_ok=True)
    provenance=dict(context_sha256=__import__("hashlib").sha256(
       args.context.read_bytes()).hexdigest(),
       frozen_policy_sha256=__import__("hashlib").sha256(
       (args.base/"selected_policy.json").read_bytes()).hexdigest(),
       base_lambda=lam,
       plan_seconds=args.plan_seconds)
    contract=args.out/"paper_contract.json"
    if contract.exists() and read(contract)!=provenance:
        raise ValueError("Paper inputs or solver budget changed; use a new --out")
    contract.write_text(json.dumps(provenance,indent=2))
    invoke("test_suite",
      [sys.executable,"-m","pytest","-q"],args.out/"logs")
    invoke("theory_audit",
      [sys.executable,"-u",ROOT/"pilot"/"paper_theory_audit.py",
       "--base",args.base,"--out",args.out/"audit"],args.out/"logs")
    for name in ("clean","case_only"):
        ablation(args,None,name,lam)
    scenario_root=args.out/"scenarios"
    invoke("four_scenario_grid",
      [sys.executable,"-u",ROOT/"pilot"/"run_response_scenario_grid.py",
       "--context",args.context,"--out",scenario_root,
       "--cores",args.cores,"--plan-seconds",args.plan_seconds,
       "--vf-features","clean","--vf-lambda-absolute",repr(lam)],
      args.out/"logs")
    invoke("response_matrix_and_heterogeneity",
      [sys.executable,"-u","-m","pilot.paper_response_report",
       "--context",args.context,"--grid",scenario_root,
       "--cores",args.cores,"--plan-seconds",args.plan_seconds],
      args.out/"logs")
    invoke("hash_seed_sensitivity",
      [sys.executable,"-u","-m","pilot.paper_seed_audit",
       "--context",args.context,"--base",args.base,
       "--out",args.out/"seed_audit",
       "--cores",args.cores,"--plan-seconds",args.plan_seconds],
      args.out/"logs")
    report(args,lam)


if __name__=="__main__":
    main()
