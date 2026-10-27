#!/usr/bin/env python3
"""Focused VF regularization diagnostic; saturating training, fixed displays.

Every configuration jointly trains all VF coefficients, evaluates the same
three-regime deployed response, and replays its final training assignments.
Use separate output directories so model checkpoints never cross settings.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

RUNNER=Path(__file__).with_name("run_nonmonotone_learning.py")
CONFIGS=(
    ("reference", 1., 1., "uniform"),
    ("lambda010", .1, 1., "uniform"),
    ("lambda025", .25, 1., "uniform"),
    ("lambda050", .5, 1., "uniform"),
    ("lambda4", 4., 1., "uniform"),
    ("plan4", 1., 4., "uniform"),
    ("score", 1., 1., "score"),
)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--out-root",type=Path,default=Path("artifacts/vf_tuning"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--only",nargs="*",choices=[c[0] for c in CONFIGS])
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    rows=[]
    for name,scale,plan,loading in CONFIGS:
        folder=args.out_root/name
        if args.only is None or name in args.only:
            settings=["--context",str(args.context),"--out",str(folder),
                "--cores",str(args.cores),"--plan-seconds",str(args.plan_seconds),
                "--link","saturating","--vf-lambda-scale",str(scale),
                "--vf-plan-penalty",str(plan),"--vf-loading",loading]
            for stage,target in (("all","selected_policy.json"),
                                 ("evaluate","summary.json"),
                                 ("probe","train_probe.json")):
                if (folder/target).exists():
                    continue
                cmd=[sys.executable,"-u",str(RUNNER),"--stage",stage,*settings]
                print("$"," ".join(cmd),flush=True)
                if not args.dry_run:
                    subprocess.run(cmd,check=True)
        # Summarize every completed experiment, even when --only was used.
        if args.dry_run or not all((folder/name).exists() for name in (
                "selected_policy.json","summary.json","train_probe.json",
                "vf_library_progress.json")):
            continue
        policy=json.loads((folder/"selected_policy.json").read_text())
        summary=json.loads((folder/"summary.json").read_text())
        probe=json.loads((folder/"train_probe.json").read_text())
        progress=json.loads((folder/"vf_library_progress.json").read_text())
        theta=policy["vf_theta"];n_case=len(policy["case_coeff"])
        rows.append(dict(
            experiment=name,lambda_scale=scale,plan_penalty=plan,loading=loading,
            train_vf_cost=probe["VF"]["cost_per_case"],
            test_vf_cost=summary["VF"]["cost_per_case"],
            test_booked_cost=summary["BOOKED"]["cost_per_case"],
            test_vf_minus_booked=summary["paired_cost_differences"]
                                   ["VF_minus_BOOKED"]["estimate"],
            case_coefficients_active=sum(abs(x)>1e-3 for x in theta[:n_case]),
            plan_coefficients_active=sum(abs(x)>1e-3 for x in theta[n_case:]),
            congestion_coefficient=theta[n_case+2],
            mm_iterations=len(progress["history"]),mm_status=progress["status"]))
    if rows:
        args.out_root.mkdir(parents=True,exist_ok=True)
        with (args.out_root/"summary.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        print("\nResults (lower cost is better):",flush=True)
        for row in rows:
            print(f"{row['experiment']:>10}: "
                  f"train={row['train_vf_cost']:.3f}, "
                  f"2013={row['test_vf_cost']:.3f}, "
                  f"vs booked={row['test_vf_minus_booked']:+.3f}, "
                  f"congestion coefficient={row['congestion_coefficient']:+.3f}",
                  flush=True)

if __name__=="__main__":
    main()
