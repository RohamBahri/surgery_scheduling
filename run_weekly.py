#!/usr/bin/env python3
"""Audit or benchmark the supported-weekday weekly planner on the UHN workbook.

Deliberately does NOT train a recommendation policy. This isolates the
scheduling feasible set and the exact ordinary / loss-augmented oracles.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, build_cohort
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, solve_adversary, solve_week


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--stage", choices=("audit", "benchmark"), default="audit")
    p.add_argument("--group", choices=GROUPS, default=None)
    p.add_argument("--weeks", type=int, default=2,
                   help="First N group-weeks, per selected group; 0 means all")
    p.add_argument("--prior-dates", type=int, default=1)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--adversary-seconds", type=float, default=60.)
    p.add_argument("--gamma", type=float, default=2.)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--adversary", action="store_true",
                   help="Also benchmark exact gap-VF loss-augmented oracle at Booked")
    p.add_argument("--output", type=Path, default=Path("artifacts/weekly_supported_days"))
    args = p.parse_args()
    if args.weeks < 0 or args.prior_dates < 1 or args.seconds <= 0 or args.adversary_seconds <= 0:
        p.error("weeks must be >=0, prior-dates >=1, and solve budgets >0")
    if args.gamma < 1 or args.threads < 1:
        p.error("gamma must be >=1 and threads >=1")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory must be new/empty; do not overwrite a prior experiment")
    args.output.mkdir(parents=True, exist_ok=True)

    print("Reading and cleaning UHN workbook...", flush=True)
    raw = pd.read_excel(args.workbook, usecols=COLUMNS)
    cohort, audit, _, _ = build_cohort(raw)
    events = performed_events(raw)
    selected = build_weeks(cohort, events, split="train", group=args.group,
                           prior_dates=args.prior_dates)
    weeks = []
    for group in GROUPS:
        matching = [w for w in selected if w.group == group]
        weeks.extend(matching[:args.weeks] if args.weeks else matching)
    components = [c for w in weeks for c in w.components()]
    sha = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    metadata = {
        "dataset_sha256": sha, "source_branch": "daily-experiment",
        "cohort_train_cases": int(cohort.split.eq("train").sum()),
        "cohort_test_cases": int(cohort.split.eq("test").sum()),
        "selected_group_weeks": len(weeks), "selected_cases": sum(w.n for w in weeks),
        "components": len(components), "largest_component_cases": max((c.n for c in components), default=0),
        "extra_days_offered": sum(len(w.added_days) for w in weeks),
        "prior_dates": args.prior_dates, "all_retained_cases_movable": True,
        "extra_day_rule": "one extra weekday for single-observed-day surgeons with >=2 retained cases; "
                          "prior operating evidence or valid same-week excluded activity; "
                          "room already active in target week",
        "roster": "retrospective active physical room-days, fixed and always costed",
        "cost": {"capacity": 480, "turnover": 30, "idle": 1, "overtime": 1.75},
        "stage": args.stage, "gamma": args.gamma,
        "solver_budget_seconds": args.seconds, "adversary_budget_seconds": args.adversary_seconds,
        "threads": args.threads,
        "cohort_flow": audit["flow"],
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    structure = []
    for w in weeks:
        for k, comp in enumerate(w.components()):
            structure.append({
                "group": w.group, "monday": w.monday, "component": k,
                "cases": comp.n, "surgeons": len(comp.surgeon_ids), "rooms": comp.k,
                "arcs": int(sum(len(comp.allowed[int(s)]) for s in comp.surgeon)),
                "extra_days_offered": len(comp.added_days),
                "original_room_days": len(set(comp.original.tolist())),
                "inflexible": bool(all(len(comp.allowed[int(s)]) == 1 for s in comp.surgeon)),
            })
    pd.DataFrame(structure).to_csv(args.output / "components.csv", index=False)
    print(f"Selected {metadata['selected_cases']} cases, {len(weeks)} group-weeks, "
          f"{len(components)} components, {metadata['extra_days_offered']} optional surgeon-days.",
          flush=True)
    if args.stage == "audit":
        return

    rows = []
    total = len(components)
    for index, comp in enumerate(components):
        key = structure[index]
        booked = solve_week(comp, comp.booked, seconds=args.seconds, threads=args.threads)
        oracle = solve_week(comp, comp.actual, seconds=args.seconds, threads=args.threads,
                            tie=False)
        record = {
            **key,
            "historical_realized_cost": evaluate(comp, comp.original, comp.actual)["cost"],
            "booked_complete": booked.complete,
            "oracle_complete": oracle.complete,
            "booked_seconds": booked.seconds, "oracle_seconds": oracle.seconds,
            "booked_status": booked.status, "oracle_status": oracle.status,
            "booked_predicted_cost": booked.value, "oracle_realized_cost": oracle.value,
            "oracle_realized_lower": oracle.bound,
        }
        if booked.assignment is not None:
            record.update({"booked_" + k: v for k, v in
                           evaluate(comp, booked.assignment, comp.actual).items()})
        if booked.assignment is not None and oracle.value is not None:
            record["booked_regret"] = record["booked_cost"] - oracle.value
            if booked.complete and oracle.complete:
                errors = comp.actual - comp.booked
                theta = (1.75 * np.maximum(errors, 0.) + np.maximum(-errors, 0.)).sum()
                record["vf_bound_at_booked"] = booked.value + float(theta) - oracle.value
        if args.adversary:
            adversary = solve_adversary(comp, comp.booked, gamma=args.gamma,
                                        seconds=args.adversary_seconds, threads=args.threads)
            record.update(adversary_complete=adversary.complete,
                          adversary_status=adversary.status,
                          adversary_seconds=adversary.seconds,
                          adversary_value=adversary.value, adversary_upper=adversary.bound)
            if (adversary.complete and booked.complete and oracle.complete):
                record["gap_vf_bound_at_booked"] = (
                    args.gamma * booked.value + adversary.value - oracle.value
                )
                record["spo_bound_at_booked"] = (
                    args.gamma * evaluate(comp, oracle.assignment, comp.booked)["cost"]
                    + adversary.value - oracle.value
                )
                if record["gap_vf_bound_at_booked"] + .01 < record["booked_regret"]:
                    raise AssertionError("Certified gap-VF bound is below realized regret")
                if record["spo_bound_at_booked"] + .01 < record["gap_vf_bound_at_booked"]:
                    raise AssertionError("SPO upper bound is below gap-VF")
        rows.append(record)
        if (index + 1) % 100 == 0 or index + 1 == total:
            print(f"Completed {index+1}/{total} components", flush=True)
            pd.DataFrame(rows).to_csv(args.output / "benchmark.csv", index=False)

    incomplete = [r for r in rows if not r["booked_complete"] or not r["oracle_complete"]
                  or (args.adversary and not r["adversary_complete"])]
    totals = {
        "components": total,
        "incomplete_components": len(incomplete),
        "historical_realized": sum(r["historical_realized_cost"] for r in rows),
        "booked_realized": sum(r.get("booked_cost", 0) for r in rows),
        "oracle_cost": sum(r.get("oracle_realized_cost") or 0 for r in rows),
        "booked_regret": sum(r.get("booked_regret") or 0 for r in rows),
        "positive_booked_regret_components": sum(r.get("booked_regret", 0) > .01 for r in rows),
        "booked_case_moves": sum(r.get("booked_moved_cases", 0) for r in rows),
        "booked_weekday_moves": sum(r.get("booked_moved_weekdays", 0) for r in rows),
        "booked_extra_days_used": sum(r.get("booked_used_extra_days", 0) for r in rows),
        "solver_seconds_sum": sum(r["booked_seconds"] + r["oracle_seconds"]
                                 + r.get("adversary_seconds", 0) for r in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2), flush=True)
    if incomplete:
        raise SystemExit("INCOMPLETE: see benchmark.csv; unresolved solves are not certified")


if __name__ == "__main__":
    main()
