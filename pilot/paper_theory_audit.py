#!/usr/bin/env python3
"""Numerical certificate audit and conservative theory-claims register.

This document verifies saved numerical identities and solver reports.
It does not replace a mathematical proof of the scheduling regret bound or
turn local MM/pDCA termination into a global optimality certificate.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def get(path):
    return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run=get(args.base/"selected_policy.json")
    progress=get(args.base/"vf_library_progress.json")
    summary=get(args.base/"summary.json")
    history=progress["history"]
    if not history:
        raise AssertionError("Missing VF outer-iteration records")
    nominal=run["link"]=="saturating" and run["alpha"]==.8 and run["h"]==30.
    dimensions=(len(run["case_coeff"]),len(run["vf_theta"]))
    if dimensions!=(77,82) or not nominal:
        raise AssertionError("Unexpected frozen nominal model")
    if not np.isclose(run["lambda_vf"],.10924129431614196,atol=1e-9):
        raise AssertionError("Frozen L1 regularization no longer matches experiment")
    before=[float(x["before"]) for x in history]
    candidate=[float(x["candidate"]) for x in history]
    eps=[float(x["epsilon"]) for x in history]
    inner=[inner for x in history for inner in x["search"]["inner"]]
    # The exact outer MM objective is evaluated at the current parameters.
    tol=1e-6
    monotone=all(before[i+1]<=before[i]+tol
                 for i in range(len(before)-1))
    safe_surrogate=all(candidate[i]+max(eps[i],tol)>=before[i+1]
                       for i in range(len(before)-1))
    accepted=all((not x["accepted"]) or (x["candidate"]<=x["before"]+tol)
                 for x in history)
    certified_inner=all(x.get("inner_certified") for x in inner)
    complete_schedules=max(eps)<1e-5
    evaluated=summary["common_certified_components"]
    train_cases=run["train_cases"]
    if not (monotone and safe_surrogate and accepted
            and certified_inner and complete_schedules
            and evaluated==summary["all_components"]):
        raise AssertionError("Frozen policy audit did not pass")
    facts=dict(
        train_cases=train_cases,
        train_components=run["train_components"],
        test_components=summary["common_certified_components"],
        test_cases=summary["common_certified_cases"],
        iterations=len(history),
        pDCA_subproblems=len(inner),
        all_inner_certified=certified_inner,
        outer_objective_nonincreasing=monotone,
        verified_majorizer_relation=safe_surrogate,
        monotone_accepted_candidates=accepted,
        planner_gap_epsilon_max=max(eps),
        objective_initial=before[0],
        objective_last=before[-1],
        termination=progress["status"],
        lambda_vf=run["lambda_vf"],
        trained_parameters=len(run["vf_theta"]),
        archive_reference=str(args.base))
    (args.out/"numerical_audit.json").write_text(
        json.dumps(facts,indent=2,sort_keys=True))
    claims=[
        ("Asymmetric error bound","Mathematical identity",
         "For a fixed assignment, realized overtime/idle cost is at most planned "
         "cost plus c_o times underestimated minutes and c_u times overestimated "
         "minutes; derive from convex room-load cost slopes.",
         "Manuscript proof; check fixed-schedule identity tests"),
        ("Finite schedule library","Conditional mathematical guarantee",
         "Minimum planned cost over library is >= true planner optimum; equality "
         "at current iterate requires a globally optimal certified schedule in library.",
         "Library-MM source, planner feasible bounds and recorded epsilon"),
        ("MM descent","Numerically verified for frozen run",
         "Outer before values nonincreasing and majorizer candidate bounds next "
         "true value up to recorded planner numerical tolerance.",
         "vf_library_progress.json, verified per-iterate epsilon"),
        ("pDCA termination","Numerically verified, NOT global stationarity proof",
         "Convex QP subproblems were certified, and relative improvement triggered "
         "termination; do not claim a global solution or formally certified stationary point.",
         "pDCA inner certificates and outer RELATIVE_IMPROVEMENT"),
        ("Regret certificate","Conditional upper bound, not test improvement",
         "Subtract a certified hindsight lower bound from a valid realized cost "
         "upper bound; optimization gaps must be included in the accounting.",
         "Proof and solver bound metadata, not empirical test cost"),
        ("Primary schedule optimality","Solver-dependent numerical certificate",
         "OPTIMAL_PRIMARY/ANALYTIC statuses certify primary cost to stated numerical "
         "tolerances; hash tie stage can select assignments inside a small tau set.",
         "anchored_weekly_solver.py and hash-seed sensitivity test"),
        ("Human adoption","Scenario assumption only",
         "No historical responses to displayed advice; no individual response "
         "parameters are identified from the observational cohort.",
         "Empirical limitations section"),
        ("Cross-scenario robustness","Empirical, conditional on scenario set",
         "Regret is relative to the candidate VF policies and tested scenarios; "
         "no minimax theorem over all possible human-response functions.",
         "Paired same-case cross-scenario cost matrix")
    ]
    file=args.out/"claims_register.csv"
    with file.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["claim","classification","precise_scope","evidence_needed"])
        w.writerows(claims)
    md=[
        "# Theory and implementation audit",
        "",
        "Numerical checks on the frozen nominal λ × 0.25 joint VF:",
        "",
        f"- Training: {facts['train_cases']:,} cases, {facts['train_components']:,} components.",
        f"- Evaluation: {facts['test_cases']:,} common certified cases, "
          f"{facts['test_components']:,} components.",
        f"- VF outer iterations: {len(history)}; certified convex subproblems: {len(inner)}.",
        f"- Outer objective: {before[0]:.5f} to {before[-1]:.5f}; "
          f"maximum mean planner gap: {max(eps):.3g}.",
        f"- Status: {progress['status']}; numerical descent checks passed.",
        "",
        "## Claims and limitations",
        ""
    ]
    for title,status,statement,evidence in claims:
        md.extend([f"### {title} — {status}","",statement,"",
                   "*Evidence:* "+evidence,""])
    md.extend(["## Distinctions for the paper","",
       "A finite-schedule surrogate is a majorizer, not a globally solved "
       "training problem. The lower bound from the planner and the numerical "
       "touching gap are distinct. Matching the bounds at visited iterates "
       "does not establish global optimality of learned coefficients.",
       "",
       "The hash tie tolerance `tau` is a small additive allowance on the "
       "PRIMARY scheduling objective used to select a repeatable assignment. "
       "Seed comparisons test this assignment selection rule, not surgeon behavior.",
       "",
       "The empirical population is the common set certified for all compared "
       "policies; do not silently include uncertified cells using incumbents.",
       "",
       "The archived numerical evidence is not itself a formal proof. Audit "
       "the manuscript formulas and assumptions before submission.",
       ""])
    (args.out/"theory_audit.md").write_text("\n".join(md))
    print(json.dumps(facts,indent=2),flush=True)


if __name__=="__main__":
    main()
