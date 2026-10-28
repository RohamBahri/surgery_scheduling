#!/usr/bin/env python3
"""One-command completion of the two missing paper checks.

Requires only the existing paper_completion scenario artifacts and frozen
lambda025 model, not refitting or rerunning the completed scenario grid.
Re-running the same command resumes explicit seed audit and C response cells.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def run(args,stage,cmd):
    log=args.out/(stage+".log")
    print("\n===",stage,"===\n$ "+" ".join(map(str,cmd)),flush=True)
    with log.open("a") as f:
        f.write("\nStarting / resuming "+stage+"\n")
        f.flush()
        code=subprocess.run([str(x) for x in cmd],cwd=ROOT,
                            stdout=f,stderr=subprocess.STDOUT).returncode
    if code:
        tail=log.read_text(errors="replace").splitlines()[-40:]
        print("\n".join(tail),file=sys.stderr)
        raise RuntimeError(stage+" failed; see "+str(log))
    print(stage,"finished;",log,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--base",type=Path,default=Path("artifacts/vf_tuning/lambda025"))
    ap.add_argument("--grid",type=Path,default=Path("artifacts/paper_completion/scenarios"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/paper_completion/followup"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    args.context=args.context.resolve()
    args.base=args.base.resolve()
    args.grid=args.grid.resolve()
    args.out=args.out.resolve()
    for required in (args.context,args.base/"selected_policy.json",
                     args.grid/"frozen_advice"/"evaluation_displays.csv",
                     args.grid/"summary.json"):
        if not required.exists():
            raise FileNotFoundError("Missing completed-run input "+str(required))
    args.out.mkdir(parents=True,exist_ok=True)
    seed_folder=args.out/"seed_audit_fixed"
    cliff_folder=args.out/"cliff"
    if not (seed_folder/"seed_robustness.json").exists():
        run(args,"seed",[sys.executable,"-u","-m","pilot.paper_seed_audit",
            "--context",args.context,"--base",args.base,
            "--out",seed_folder,"--cores",args.cores,
            "--plan-seconds",args.plan_seconds])
    if not (cliff_folder/"cliff_summary.json").exists():
        run(args,"cliff",[sys.executable,"-u","-m","pilot.paper_cliff_audit",
            "--context",args.context,"--grid",args.grid,
            "--out",cliff_folder,"--cores",args.cores,
            "--plan-seconds",args.plan_seconds])
    seed=read(seed_folder/"seed_robustness.json")
    cliff=read(cliff_folder/"cliff_summary.json")
    lines=[
      "# Supplementary validation: seed and credibility-cliff checks","",
      "The original paper_completion seed audit was INVALID: its worker",
      "hardcoded hash_seed=0 for all requested seeds. This new audit passes",
      "each seed directly to solve_week and caches with a seed-specific key.","",
      f"Common certified seed-audit cases: {seed['common_certified_cases']:,};",
      f"components: {seed['common_certified_components']:,}.","",
      "| Seed | Policy | Changed assignments | Changed component costs |",
      "|---:|---|---:|---:|"
    ]
    for n,methods in seed["seeds"].items():
        for policy,r in methods.items():
            lines.append(f"| {n} | {policy} | {r['changed_assignments']} | "
                         f"{r['changed_realized_cost_components']} |")
    lines+=["","## Nonmonotone response / credibility cliff","",
      "All policies are frozen under saturating training; each original",
      "display is evaluated under six true nonmonotone responses without",
      "retraining or recanonicalizing. All B/C figures below are on the",
      "same fully certified component intersection.","",
      f"Common certified cliff cases: {cliff['common_certified_cases']:,};",
      f"components: {cliff['common_certified_components']:,}.","",
      "| Frozen VF policy | Worst C minus matched B cost/case |",
      "|---|---:|"]
    for policy in ("VF__SAT_A50_H30","VF__SAT_A50_H60",
                   "VF__SAT_A80_H30","VF__SAT_A80_H60"):
        p=[r for r in cliff["observations"] if r["trained_policy"]==policy]
        lines.append(f"| {policy} | "
            f"{max(r['cliff_minus_saturated'] for r in p):+.3f} |")
    lines+=["","The costs above are exploratory and depend on assumed response",
      "functions; none are estimates of actual surgeon acceptance.","",
      "See cliff/cliff_policy_sensitivity.csv and",
      "cliff/cliff_vs_saturated_matched.csv for detailed results.",""]
    (args.out/"followup_report.md").write_text("\n".join(lines))
    (args.out/"followup_summary.json").write_text(json.dumps(
        dict(original_seed_audit_valid=False,corrected_seed_audit=seed,
             frozen_advice_cliff_audit=cliff),indent=2))
    print("Follow-up complete:",args.out/"followup_report.md",flush=True)


if __name__=="__main__":
    main()
