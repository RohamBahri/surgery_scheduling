#!/usr/bin/env python3
"""ONE controlled, resumable structural-VF experiment.

The ONLY learned change is six schedule-independent contextual features.
All core choices are frozen: joint VF, CASE initializer from nominal baseline,
saturating alpha=.8/h=30, absolute lambda=0.10924129431614196, uniform
L1 except intercept, same MM/pDCA and anchored scheduler (hash seed 0).

No alternative feature search, robust policy retraining, or behavioral
identification. The repeatedly examined 2013 evaluation is EXPLORATORY.

Run from repository root:
  caffeinate -i python -u -m pilot.run_structural_vf
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from pilot.run_nonmonotone_learning import save
from surgery.vf_features import STRUCTURAL_NAMES,structural_features,standardize_features

ROOT=Path(__file__).resolve().parents[1]
LAMBDA=.10924129431614196
SCHEMA="structural"
# Match the fully tested clean-VF run_contract. Do not rely on CLI defaults
# for any scientific setting other than the CPU/process budget.
FROZEN=dict(
    alpha=.8,h=30.,cap=180.,case_steps=200,vf_outers=30,
    vf_pdca_steps=50,cuts=80,case_radius=100.,vf_radius=100.,
    vf_plan_penalty=1.,vf_loading="uniform",backend="gurobi",
    link="saturating",
)


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_baseline(reference):
    """Accept the old implicit baseline schema without weakening other checks.

    Original lambda025 predates the vf_features metadata key: absence means
    baseline (five draft-plan columns). The frozen scientific specification
    and coefficient dimensions still have to match exactly.
    """
    problems=[]
    expected_text={"method":"JOINT_VF","link":"saturating",
                   "vf_loading":"uniform"}
    for key,expected in expected_text.items():
        if reference.get(key)!=expected:
            problems.append(f"{key}={reference.get(key)!r}, expected {expected!r}")
    schema=reference.get("vf_features","baseline")
    if schema!="baseline":
        problems.append(f"vf_features={schema!r}, expected 'baseline'")
    for key,expected in (("alpha",.8),("h",30.),("cap",180.),
                         ("vf_plan_penalty",1.),
                         ("vf_lambda_scale",.25),("lambda_vf",LAMBDA)):
        value=reference.get(key)
        if not isinstance(value,(int,float)) or not math.isclose(
                float(value),expected,rel_tol=1e-10,abs_tol=1e-9):
            problems.append(f"{key}={value!r}, expected {expected!r}")
    case=reference.get("case_coeff")
    theta=reference.get("vf_theta")
    if not isinstance(case,list) or len(case)!=77:
        problems.append("expected 77 frozen case coefficients")
    if not isinstance(theta,list) or len(theta)!=82:
        problems.append("expected 82 frozen joint VF coefficients (77 + 5)")
    if problems:
        raise ValueError("Frozen baseline mismatch: "+"; ".join(problems))
    return True


def audit_features(ctx,out,clean):
    """Training-only diagnostics; no solver, realized data, or plan selection."""
    items=ctx["train"]
    b=np.asarray(ctx["fit"]["booked"],float)
    h=np.asarray(ctx["H"],float)
    x=np.asarray(ctx["fit"]["X"],float)
    raw=structural_features(items,b,h)
    scaled,(mu,sd)=standardize_features(raw)
    # Basic invariance under arbitrary, invalid provisional schedules; the
    # feature recipe must not inspect them.
    from surgery.vf_features import booked_plan_features
    bogus={key:[-99]*c.n for key,c,_ in items}
    against=booked_plan_features(items,b,bogus,h,SCHEMA)
    if not np.array_equal(against,raw):
        raise AssertionError("Structural features depend on provisional assignments")
    if raw.shape!=(len(b),6) or x.shape[0]!=len(b):
        raise AssertionError("Unexpected cohort/feature dimensions")
    rows=[]
    for j,name in enumerate(STRUCTURAL_NAMES):
        col=raw[:,j]
        dup_case=[int(k) for k in range(x.shape[1])
                  if np.allclose(col,x[:,k],rtol=1e-12,atol=1e-12)]
        dup_struct=[STRUCTURAL_NAMES[k] for k in range(j)
                    if np.allclose(col,raw[:,k],rtol=1e-12,atol=1e-12)]
        cx=x-x.mean(axis=0)
        hc=col-col.mean()
        denom=np.linalg.norm(cx,axis=0)*np.linalg.norm(hc)
        corr=np.divide(cx.T@hc,denom,out=np.zeros(x.shape[1]),where=denom>1e-12)
        rows.append(dict(name=name,mean=float(col.mean()),sd=float(col.std()),
            p05=float(np.quantile(col,.05)),median=float(np.median(col)),
            p95=float(np.quantile(col,.95)),
            unique_values=int(np.unique(col).size),
            exact_case_duplicates=dup_case,exact_structural_duplicates=dup_struct,
            max_abs_case_correlation=float(np.max(abs(corr),initial=0.))))
    # One descriptive training-only measure of how much eligibility pressure
    # adds beyond overall load and the surgeon's own load.
    y=raw[:,1]
    A=np.column_stack([np.ones(len(y)),raw[:,0],raw[:,2]])
    beta=np.linalg.lstsq(A,y,rcond=None)[0]
    residual=y-A@beta
    denom=float(np.dot(y-y.mean(),y-y.mean()))
    score=None if denom<=1e-12 else float(1.-np.dot(residual,residual)/denom)
    result=dict(
        design="six_fixed_schedule_independent_context_features",
        n_train_cases=len(b),n_components=len(items),
        feature_names=list(STRUCTURAL_NAMES),features=rows,
        exact_duplicate_found=any(r["exact_case_duplicates"] or
                                  r["exact_structural_duplicates"] for r in rows),
        eligibility_pressure_r2_from_component_and_own=score,
        eligibility_pressure_differs_from_component_fraction=float(np.mean(
            abs(raw[:,1]-raw[:,0])>1e-9)),
        eligibility_pressure_differs_from_own_fraction=float(np.mean(
            abs(raw[:,1]-raw[:,2])>1e-9)),
        scaling_train_only=dict(mean=mu.tolist(),std=sd.tolist()),
        outcome_blind=True,uses_provisional_schedule=False)
    save(out/"structural_feature_audit.json",result)
    if result["exact_duplicate_found"]:
        raise ValueError("Exact duplicate in structural features; see "+
                         str(out/"structural_feature_audit.json"))
    if np.any(np.asarray([row["sd"] for row in rows])<1e-10):
        raise ValueError("Constant structural feature; see feature audit")
    # Compare with old provisional room congestion, on training only.
    old=clean/"cases_train.csv"
    if old.exists():
        with old.open(newline="") as f:
            records=list(csv.DictReader(f))
        if (len(records)==len(b) and
                np.array_equal([int(r["case_id"]) for r in records],
                               np.asarray(ctx["fit"]["case_id"],int))):
            old_y=np.array([float(r["feature_draft_congestion_standardized"])
                            for r in records])
            regression=np.column_stack([np.ones(len(b)),scaled])
            coeff=np.linalg.lstsq(regression,old_y,rcond=None)[0]
            diff=old_y-regression@coeff
            den=float(np.sum((old_y-old_y.mean())**2))
            result["old_draft_congestion_r2_from_structural"]=(
                None if den<=1e-12 else float(1.-np.sum(diff**2)/den))
            save(out/"structural_feature_audit.json",result)
    return result


def component_records(folder,method="VF"):
    data=read(folder/"evaluation_2013.json")
    result={r["component"]:r for r in data["entries"]
            if r["method"]==method and r["complete"]}
    return result


def paired(candidate,reference,draws=10000):
    a=component_records(candidate)
    b=component_records(reference)
    keys=sorted(set(a)&set(b))
    if not keys:raise RuntimeError("No common certified VF evaluation components")
    week={}
    changed=0
    for key in keys:
        x,y=a[key],b[key]
        if x["cases"]!=y["cases"] or x["week"]!=y["week"]:
            raise AssertionError("Matched components have different cases or weeks")
        w=week.setdefault(x["week"],[0.,0.])
        w[0]+=x["cost"]-y["cost"]
        w[1]+=x["cases"]
        if abs(x["cost"]-y["cost"])>1e-7:changed+=1
    weeks=sorted(week)
    numerator=np.array([week[w][0] for w in weeks])
    denominator=np.array([week[w][1] for w in weeks])
    rng=np.random.default_rng(20261028)
    indices=rng.integers(0,len(weeks),size=(draws,len(weeks)))
    samples=numerator[indices].sum(axis=1)/denominator[indices].sum(axis=1)
    return dict(difference_per_case=float(numerator.sum()/denominator.sum()),
        calendar_week_bootstrap_95=[float(x) for x in
                                    np.quantile(samples,[.025,.975])],
        matched_components=len(keys),matched_cases=int(denominator.sum()),
        changed_realized_cost_components=changed,
        better_weeks=int(np.sum(numerator<0)),
        worse_weeks=int(np.sum(numerator>0)),
        equal_weeks=int(np.sum(numerator==0)))


def summary(folder):
    selected=read(folder/"selected_policy.json")
    probe=read(folder/"train_probe.json")
    evaluation=read(folder/"summary.json")
    state=read(folder/"vf_library_progress.json")
    hist=state["history"]
    if not hist:raise ValueError("No VF history "+str(folder))
    last=hist[-1]
    return dict(n_coefficients=len(selected["vf_theta"]),
        features=selected.get("vf_feature_names",[]),
        lambda_vf=float(selected["lambda_vf"]),
        train_cost=float(probe["VF"]["cost_per_case"]),
        test_cost=float(evaluation["VF"]["cost_per_case"]),
        booked_cost=float(evaluation["BOOKED"]["cost_per_case"]),
        certified_components=int(evaluation["common_certified_components"]),
        certified_cases=int(evaluation["common_certified_cases"]),
        final_library_penalized_surrogate=float(
            last["candidate"] if last["accepted"] else last["before"]),
        mm_iterations=len(hist),mm_status=state["status"],
        max_mean_planner_gap=max(float(h["epsilon"]) for h in hist))


def subprocess_stage(label,script,out,extra):
    log=out/(label+".log")
    cmd=[sys.executable,"-u",str(script),*map(str,extra)]
    print("\n"+label+": "+" ".join(cmd),flush=True)
    with log.open("a") as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join(log.read_text(errors="replace").splitlines()[-45:]),
              file=sys.stderr)
        raise RuntimeError(label+" failed; see "+str(log))
    print(label+" passed; log "+str(log),flush=True)


def make_report(out,base,clean,case_only):
    entries={"structural":summary(out),
             "clean":summary(clean),
             "case_only":summary(case_only)}
    selected=read(out/"selected_policy.json")
    if (selected.get("vf_features")!=SCHEMA or len(selected["vf_theta"])!=83
            or len(selected.get("vf_feature_names",[]))!=6
            or not math.isclose(entries["structural"]["lambda_vf"],LAMBDA,
                                abs_tol=1e-9)):
        raise AssertionError("Experiment policy does not match frozen design")
    comparisons={x:paired(out,dir) for x,dir in
                 (("clean",clean),("case_only",case_only),("original",base))}
    audit=read(out/"structural_feature_audit.json")
    result=dict(label="EXPLORATORY_REUSED_2013",
        purpose="one predeclared six-feature schedule-independent replacement",
        no_further_feature_search=True,model_specs=entries,
        comparisons=comparisons,feature_audit=audit)
    save(out/"structural_comparison.json",result)
    lines=[
      "# Structural VF: one controlled feature-representation comparison","",
      "**Evaluation status:** exploratory. The 2013 cohort was examined during "
      "previous method development; do not treat these intervals as an "
      "independent confirmatory feature-selection test.","",
      "All VF fits use saturating response (α=0.8, h=30), absolute "
      f"λ={LAMBDA:.12f}, joint coefficients, uniform penalty and the same "
      "certified scheduling model. Only feature representation varies.","",
      "| VF representation | Parameters | Train cost/case | 2013 cost/case | "
      "Final penalized library surrogate |",
      "|---|---:|---:|---:|---:|"]
    for label,x in entries.items():
        lines.append(f"| {label} | {x['n_coefficients']} | "
                     f"{x['train_cost']:.3f} | {x['test_cost']:.3f} | "
                     f"{x['final_library_penalized_surrogate']:.3f} |")
    lines += ["","## Paired comparisons: structural minus reference","",
      "| Reference | Difference/case | Paired-week bootstrap 95% | "
      "Matched cases | Changed components |",
      "|---|---:|---|---:|---:|"]
    for label,v in comparisons.items():
        lo,hi=v["calendar_week_bootstrap_95"]
        lines.append(f"| {label} | {v['difference_per_case']:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {v['matched_cases']} | "
                     f"{v['changed_realized_cost_components']} |")
    lines += ["","## Training-only feature audit","",
      f"Exact duplicates: {audit['exact_duplicate_found']}.",
      f"Eligibility pressure differs from global pressure for "
      f"{100*audit['eligibility_pressure_differs_from_component_fraction']:.1f}% "
      "of training cases.",
      f"Eligibility pressure differs from own surgeon pressure for "
      f"{100*audit['eligibility_pressure_differs_from_own_fraction']:.1f}% "
      "of training cases.",
      "",
      "The structural features require only bookings, the reconstructed "
      "room-day eligibility graph and historical spread. They use no "
      "provisional assignment, realized duration or tie-breaking seed. "
      "The VF optimizer still constructs feasible scheduling libraries "
      "internally; this is an optimization step, not a recommendation feature.",
      "",
      "The finite-library objective includes the L1 penalty and can vary "
      "with the schedule library; it is not a directly comparable realized-cost "
      "certificate. The 2013 differences are exploratory, not a new "
      "hyperparameter-selection procedure. Existing behavioral-scenario "
      "policies have not been refit.",
      ""]
    if "old_draft_congestion_r2_from_structural" in audit:
        lines.append("Training-only R² explaining old draft congestion from "
            "structural features: "+
            str(round(audit["old_draft_congestion_r2_from_structural"],3)))
    (out/"structural_report.md").write_text("\n".join(lines))
    print("\nReport: "+str(out/"structural_report.md"),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,default=Path(
        "artifacts/objective_selection_12h/objective_selection_2013/context.pkl"))
    ap.add_argument("--base",type=Path,default=Path("artifacts/vf_tuning/lambda025"))
    ap.add_argument("--clean",type=Path,default=Path("artifacts/paper_completion/clean"))
    ap.add_argument("--case-only",type=Path,default=Path(
        "artifacts/paper_completion/case_only"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/vf_structural"))
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    for name in ("context","base","clean","case_only","out"):
        setattr(args,name,getattr(args,name).resolve())
    for file in (args.context,args.base/"selected_policy.json",
                 args.base/"case_policy.json",args.clean/"summary.json",
                 args.case_only/"summary.json"):
        if not file.exists():
            raise FileNotFoundError("Missing prior experiment: "+str(file))
    reference=read(args.base/"selected_policy.json")
    validate_frozen_baseline(reference)
    # Cross-check all method settings against the previous fully tested
    # clean-VF run. Only the feature schema is permitted to change.
    previous=read(args.clean/"run_contract.json")
    differences={k:(previous.get(k),v) for k,v in FROZEN.items()
                 if previous.get(k)!=v}
    if (previous.get("context")!=hash_file(args.context) or
            previous.get("vf_features")!="clean" or
            previous.get("vf_lambda_absolute")!=LAMBDA or differences):
        raise ValueError("Frozen clean-VF training contract mismatch: "+
                         repr(differences))
    if args.out in (args.base,args.clean,args.case_only):
        raise ValueError("New structural experiment requires a distinct output")
    args.out.mkdir(parents=True,exist_ok=True)
    contract=dict(context_sha256=hash_file(args.context),
        frozen_selected_policy_sha256=hash_file(args.base/"selected_policy.json"),
        case_initializer_sha256=hash_file(args.base/"case_policy.json"),
        schema=SCHEMA,lambda_vf=LAMBDA,plan_seconds=args.plan_seconds,
        cores=args.cores,version="structural_vf_1",
        feature_implementation_sha256=hash_file(
            ROOT/"surgery"/"vf_features.py"),
        training_runner_sha256=hash_file(
            ROOT/"pilot"/"run_nonmonotone_learning.py"),
        experiment_runner_sha256=hash_file(Path(__file__)))
    file=args.out/"structural_run_contract.json"
    if file.exists() and read(file)!=contract:
        raise ValueError("Experiment input/contract changed: choose a NEW --out")
    save(file,contract)
    # Lightweight unit tests before any costly solve.
    if not (args.out/"feature_tests_passed.json").exists():
        subprocess_stage("feature_tests",
                        ROOT/"pilot"/"run_structural_tests.py",
                        args.out,[])
        save(args.out/"feature_tests_passed.json",dict(status="passed"))
    audit_file=args.out/"structural_feature_audit.json"
    if not audit_file.exists():
        with args.context.open("rb") as f:ctx=pickle.load(f)
        audit_features(ctx,args.out,args.clean)
    audit=read(audit_file)
    if audit.get("exact_duplicate_found") or any(
            r["sd"]<1e-10 for r in audit["features"]):
        raise ValueError("Structural feature audit failed; see "+str(audit_file))
    initializer=args.out/"case_policy.json"
    if initializer.exists():
        if initializer.read_bytes()!=(args.base/"case_policy.json").read_bytes():
            raise RuntimeError("Existing CASE initializer differs from baseline")
    else:
        shutil.copy2(args.base/"case_policy.json",initializer)
    common=["--context",args.context,"--out",args.out,"--cores",args.cores,
            "--plan-seconds",args.plan_seconds,"--link","saturating",
            "--vf-features",SCHEMA,"--vf-lambda-absolute",repr(LAMBDA),
            "--cap",FROZEN["cap"],"--case-steps",FROZEN["case_steps"],
            "--vf-outers",FROZEN["vf_outers"],
            "--vf-pdca-steps",FROZEN["vf_pdca_steps"],
            "--cuts",FROZEN["cuts"],
            "--case-radius",FROZEN["case_radius"],
            "--vf-radius",FROZEN["vf_radius"],
            "--vf-plan-penalty",FROZEN["vf_plan_penalty"],
            "--vf-loading",FROZEN["vf_loading"],
            "--backend",FROZEN["backend"]]

    for stage,path in (("all","selected_policy.json"),
                       ("evaluate","summary.json"),
                       ("probe","train_probe.json")):
        if not (args.out/path).exists():
            subprocess_stage(stage,
                ROOT/"pilot"/"run_nonmonotone_learning.py",
                args.out,["--stage",stage,*common])
    make_report(args.out,args.base,args.clean,args.case_only)


if __name__=="__main__":
    main()
