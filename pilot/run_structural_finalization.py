#!/usr/bin/env python3
"""One-command, resumable finalization of the frozen structural VF policy.

Step 1: FIVE explicit solver hash seeds for BOOKED/CASE/clean/structural;
        no retraining and no reconstruction of provisional feature plans.
Step 2: Apply an EXPLORATORY, declared practical decision rule.
Step 3: ONLY IF supported, replay the two frozen nominal VF policies under
        twelve saturating/nonmonotone behavioral deployment scenarios.
No feature search, solver change, new VF fitting, or behavioral identification.

Use: caffeinate -i python -u -m pilot.run_structural_finalization
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(0,1,2,3,4)
MIN_COVERAGE=.95
MIN_NEGATIVE_SEEDS=4
MEAN_GAIN=-.4


def read(file):
    return json.loads(Path(file).read_text())


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def reconcile_contract(recorded, expected, failed_seed_log,
                       seed_result_exists=False, behavior_result_exists=False):
    """Permit exactly the observed seed-0 reporting KeyError hotfix.

    No scientific inputs, solver implementation, seed set or source other
    than seed_compare_policies.py may change. Require evidence that the
    earlier run failed IN check_seed0_archive before publishing results.
    Preserve an audit trail of the one accepted code-only repair.
    """
    result=dict(expected)
    previous_history=list(recorded.get("implementation_history",[]))
    old_science={k:v for k,v in recorded.items()
                 if k not in ("implementation_sha256","implementation_history")}
    new_science={k:v for k,v in expected.items()
                 if k!="implementation_sha256"}
    if old_science!=new_science:
        raise ValueError("Frozen finalization inputs or parameters changed: "
                         "use NEW --out")
    prior=recorded.get("implementation_sha256",{})
    current=expected.get("implementation_sha256",{})
    changed={name for name in set(prior)|set(current)
             if prior.get(name)!=current.get(name)}
    if changed:
        log=failed_seed_log.read_text(errors="replace") if failed_seed_log.exists() else ""
        if (changed!={"seed_compare_policies.py"} or
                seed_result_exists or behavior_result_exists or
                "check_seed0_archive" not in log or "KeyError: 0" not in log):
            raise ValueError("Unexpected implementation change after starting "
                             "experiment: use NEW --out")
        previous_history.append(dict(
            reason="Repair check_seed0_archive mistaken cells[0] lookup",
            file="seed_compare_policies.py",
            prior_sha256=prior["seed_compare_policies.py"],
            replacement_sha256=current["seed_compare_policies.py"]))
    if previous_history:
        result["implementation_history"]=previous_history
    return result


def invoke(stage,out,argv):
    log=out/(stage+".log")
    print("\n=== "+stage+" ===",flush=True)
    print("$ "+" ".join(map(str,argv)),flush=True)
    with log.open("a") as f:
        result=subprocess.run([str(x) for x in argv],cwd=ROOT,
                              stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-50:]),
              file=sys.stderr,flush=True)
        raise RuntimeError(stage+" failed; log: "+str(log))
    print(stage+" completed. Log: "+str(log),flush=True)


def decision(seed_result):
    # Five seeds are deterministic perturbations, not five independent data
    # sets; 2013 was reused. This rule is a practical screen, not a p-value
    # controlled confirmatory method-selection guarantee.
    row=seed_result["comparisons"]["structural_minus_clean"]
    coverage=seed_result["common_cases"]/seed_result["all_cases"]
    mean=row["mean_over_seeds"]
    high=row["paired_calendar_week_bootstrap_95"][1]
    negative=row["negative_seed_count"]
    criteria=dict(
        enough_certified_cases=coverage>=MIN_COVERAGE,
        negative_in_at_least_four_seeds=negative>=MIN_NEGATIVE_SEEDS,
        seed_average_gain_at_least_point_four=mean<=MEAN_GAIN,
        paired_week_interval_upper_below_zero=high<0.)
    outcome=all(criteria.values())
    return dict(
        selected_policy=("structural" if outcome else "clean"),
        structural_gain_supported=outcome,
        basis="exploratory practical rule, not pre-registered confirmatory inference",
        thresholds=dict(min_certified_fraction=MIN_COVERAGE,
                        min_negative_seeds=MIN_NEGATIVE_SEEDS,
                        required_mean_difference=MEAN_GAIN,
                        require_bootstrap_upper_below_zero=True),
        actuals=dict(certified_fraction=coverage,
            negative_seeds=negative,mean_difference=mean,
            bootstrap_interval=row["paired_calendar_week_bootstrap_95"]),
        checks=criteria)


def report(out,seed,choice,behavior):
    a=seed["comparisons"]["structural_minus_clean"]
    lines=[
        "# Structural VF: tie-seed and behavioral finalization","",
        "**Interpretation:** exploratory 2013 results; policies were selected",
        "and compared repeatedly on the same cohort. Hash seeds are",
        "deterministic tie-breaking perturbations, not independent test sets.","",
        "## Five explicit hash seeds","",
        f"Common certified: {seed['common_cases']:,}/{seed['all_cases']:,} cases, "
        f"{seed['common_components']:,}/{seed['all_components']:,} components.","",
        "| Seed | BOOKED | CASE | Clean VF | Structural VF | Structural - clean |",
        "|---:|---:|---:|---:|---:|---:|"]
    for i in seed["seeds"]:
        key=str(i)
        old=seed["costs"]["VF_clean"][key]
        new=seed["costs"]["VF_structural"][key]
        lines.append(f"| {i} | {seed['costs']['BOOKED'][key]:.3f} | "
                     f"{seed['costs']['CASE'][key]:.3f} | {old:.3f} | "
                     f"{new:.3f} | {new-old:+.3f} |")
    ci=a["paired_calendar_week_bootstrap_95"]
    lines+=["",f"Structural minus clean: seed-averaged {a['mean_over_seeds']:+.3f} "
        f"cost/case; 95% paired-week bootstrap [{ci[0]:+.3f}, {ci[1]:+.3f}].",
        f"Better in {a['negative_seed_count']}/{len(seed['seeds'])} seeds.",
        "Bootstrapping resamples calendar weeks; all seeds for each sampled",
        "week move together. It does not treat seed results as independent.",
        "","## Frozen architecture decision","",
        f"**Rule-selected policy:** {choice['selected_policy']}.",
        "This rule was formulated after earlier 2013 analyses, so it does",
        "not turn these results into independent model-selection evidence.",
        ""]
    for name,passed in choice["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    if behavior is None:
        lines+=["","The 12-scenario replay was not executed because the",
            "structural-replacement screen did not pass (or --seed-only was set)."]
    else:
        lines+=["","## Twelve behavioral deployment scenarios","",
            f"Common certified population: {behavior['common_certified_cases']:,} "
            f"cases, {behavior['common_certified_components']:,} components.",
            f"Structural beats clean in "
            f"{behavior['structural_better_than_clean_scenarios']}/12 scenarios.",
            f"Structural beats BOOKED in "
            f"{behavior['better_than_booked']['VF_structural']}/12 scenarios.",
            "These tests change the ASSUMED response to frozen displays,",
            "not the training link or policy coefficients. There is no",
            "identification of surgeon behavior. The earlier cross-training",
            "minimax-regret claim does not transfer to structural VF without",
            "retraining response-conditioned structural policies.",
            "See behavioral/matched_12_scenario_costs.csv and",
            "behavioral/cliff_comparison.csv."]
    lines+=["","## Status and limitations","",
        "The original VF MM/pDCA solver, regularization, booked room capacities,",
        "and fixed planning constraints are unchanged. Structural",
        "contextual features never depend on a provisional assignment.",
        "No additional feature models or behavioral training policies were fit.",
        "Certified component omissions must be inspected before paper reporting.",""]
    (out/"finalization_report.md").write_text("\n".join(lines))
    print("\nResult: "+str(out/"finalization_report.md"),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--clean",type=Path,default=Path("artifacts/paper_completion/clean"))
    ap.add_argument("--structural",type=Path,default=Path("artifacts/vf_structural"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/structural_finalization"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    ap.add_argument("--seed-only",action="store_true",
                    help="Only five-seed robustness and decision, no scenarios")
    args=ap.parse_args()
    for attr in ("context","clean","structural","out"):
        setattr(args,attr,getattr(args,attr).resolve())
    for file in (args.context,args.clean/"cases_test.csv",
                 args.structural/"cases_test.csv"):
        if not file.exists():
            raise FileNotFoundError("Missing previous frozen evaluation: "+str(file))
    args.out.mkdir(parents=True,exist_ok=True)
    contract=dict(version="structural_finalization_v1",
        context_sha256=sha(args.context),
        clean_displays_sha256=sha(args.clean/"cases_test.csv"),
        structural_displays_sha256=sha(args.structural/"cases_test.csv"),
        cores=args.cores,plan_seconds=args.plan_seconds,
        seed_set=list(SEEDS),rule=dict(coverage=MIN_COVERAGE,
            negative_seeds=MIN_NEGATIVE_SEEDS,mean_gain=MEAN_GAIN),
        implementation_sha256={
            path.name:sha(path) for path in (
                ROOT/"pilot"/"seed_compare_policies.py",
                ROOT/"pilot"/"structural_behavior_validation.py",
                ROOT/"pilot"/"paper_seed_audit.py",
                ROOT/"surgery"/"anchored_weekly_solver.py",
                ROOT/"surgery"/"response_grid.py")})
    file=args.out/"finalization_contract.json"
    seed_dir=args.out/"seed_comparison"
    result_file=seed_dir/"seed_comparison.json"
    behavior_result=args.out/"behavioral"/"behavior_summary.json"
    if file.exists():
        contract=reconcile_contract(
            read(file),contract,args.out/"explicit_five_seeds.log",
            seed_result_exists=result_file.exists(),
            behavior_result_exists=behavior_result.exists())
    temp=file.with_suffix(".tmp")
    temp.write_text(json.dumps(contract,indent=2,sort_keys=True))
    temp.replace(file)
    if not result_file.exists():
        invoke("explicit_five_seeds",args.out,[
            sys.executable,"-u","-m","pilot.seed_compare_policies",
            "--context",args.context,"--clean",args.clean,
            "--structural",args.structural,"--out",seed_dir,
            "--seeds",*SEEDS,"--cores",args.cores,
            "--plan-seconds",args.plan_seconds])
    seed=read(result_file)
    choice=decision(seed)
    (args.out/"decision.json").write_text(
        json.dumps(choice,indent=2,sort_keys=True))
    print("Rule selected:",choice["selected_policy"],flush=True)
    print("Tests:",choice["checks"],flush=True)
    behavior=None
    if choice["structural_gain_supported"] and not args.seed_only:
        behavior_dir=args.out/"behavioral"
        behavior_file=behavior_dir/"behavior_summary.json"
        if not behavior_file.exists():
            invoke("frozen_twelve_scenario_replay",args.out,[
                sys.executable,"-u","-m","pilot.structural_behavior_validation",
                "--context",args.context,"--clean",args.clean,
                "--structural",args.structural,"--out",behavior_dir,
                "--cores",args.cores,"--plan-seconds",args.plan_seconds])
        behavior=read(behavior_file)
    report(args.out,seed,choice,behavior)


if __name__=="__main__":
    main()
