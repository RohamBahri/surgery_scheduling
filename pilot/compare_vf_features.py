#!/usr/bin/env python3
"""One controlled joint-VF feature comparison against a completed baseline.

Use the same response, absolute L1 coefficient, case initializer, roster,
and 2013 evaluation. Only the plan/context feature schema changes.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

RUNNER=Path(__file__).with_name("run_nonmonotone_learning.py")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=Path("artifacts/vf_tuning/lambda025"))
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out",type=Path,default=Path(
        "artifacts/vf_feature_comparison/context"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--vf-features",choices=("congestion_booked",
        "congestion_clip3","congestion_booked_clip3","context"),
        default="context")
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    policy=json.loads((args.base/"selected_policy.json").read_text())
    if policy.get("link")!="saturating" or policy.get("vf_features","baseline")!="baseline":
        raise ValueError("Reference must be saturating-trained baseline VF")
    lam=float(policy["lambda_vf"])
    if not math.isfinite(lam) or lam<0:
        raise ValueError("Invalid baseline VF lambda")
    if args.out.resolve()==args.base.resolve():
        raise ValueError("Feature comparison must use a separate output directory")
    case_path=args.base/"case_policy.json"
    if not case_path.exists():
        raise FileNotFoundError("Reference case initializer unavailable")
    args.out.mkdir(parents=True,exist_ok=True)
    dest=args.out/"case_policy.json"
    if not dest.exists() and not args.dry_run:
        shutil.copy2(case_path,dest)
    common=["--context",str(args.context),"--out",str(args.out),
            "--cores",str(args.cores),"--plan-seconds",str(args.plan_seconds),
            "--link","saturating","--vf-features",args.vf_features,
            "--vf-lambda-absolute",repr(lam)]
    for stage,result in (("all","selected_policy.json"),
                         ("evaluate","summary.json"),
                         ("probe","train_probe.json")):
        if (args.out/result).exists():
            continue
        cmd=[sys.executable,"-u",str(RUNNER),"--stage",stage,*common]
        print("$"," ".join(cmd),flush=True)
        if not args.dry_run:
            subprocess.run(cmd,check=True)
    if args.dry_run:
        return
    ref=json.loads((args.base/"summary.json").read_text())
    result=json.loads((args.out/"summary.json").read_text())
    train=json.loads((args.out/"train_probe.json").read_text())
    new_policy=json.loads((args.out/"selected_policy.json").read_text())
    if new_policy.get("vf_features")!=args.vf_features:
        raise AssertionError("Unexpected VF feature schema; use another output folder")
    n_case=len(new_policy["case_coeff"])
    n_plan=len(new_policy["vf_feature_names"])
    if len(new_policy["vf_theta"])!=n_case+n_plan:
        raise AssertionError("Joint VF must train all case and plan coefficients")
    outcome=dict(reference=str(args.base),model=str(args.out),lambda_vf=lam,
        vf_features=args.vf_features,
        n_joint_coefficients=len(new_policy["vf_theta"]),
        baseline_2013_cost_per_case=ref["VF"]["cost_per_case"],
        candidate_2013_cost_per_case=result["VF"]["cost_per_case"],
        candidate_minus_baseline_2013_per_case=(
            result["VF"]["cost_per_case"]-ref["VF"]["cost_per_case"]),
        candidate_training_cost_per_case=train["VF"]["cost_per_case"],
        booked_2013_cost_per_case=result["BOOKED"]["cost_per_case"],
        candidate_minus_booked_2013_per_case=(
            result["VF"]["cost_per_case"]-result["BOOKED"]["cost_per_case"]))
    (args.out/"feature_comparison.json").write_text(
        json.dumps(outcome,indent=2,sort_keys=True))
    print(json.dumps(outcome,indent=2),flush=True)

if __name__=="__main__":
    main()
