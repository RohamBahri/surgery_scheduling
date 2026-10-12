#!/usr/bin/env python3
"""Audit severe room loads and compare a cost-preserving load-balancing tie.

Only training weeks are used. Exactly the same fixed physical-room sessions
and eligible case moves are used under both planners. No sequencing is inferred.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, build_cohort
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, loads, solve_week


def audit_component(task):
    key, comp, previous, seconds, threads = task
    baseline = solve_week(comp, comp.booked, seconds=seconds, threads=threads)
    if not baseline.complete:
        raise RuntimeError(f"Uncertified baseline for {key}: {baseline.status}")
    prior_value = float(previous["booked_cost"])
    old_value = evaluate(comp, baseline.assignment, comp.actual)["cost"]
    if abs(old_value - prior_value) > .01:
        raise AssertionError(f"Baseline changed for {key}: {old_value} vs {prior_value}")
    balanced = solve_week(comp, comp.booked, seconds=seconds, threads=threads, balance=True)
    if not balanced.complete:
        raise RuntimeError(f"Uncertified balancing for {key}: {balanced.status}")
    old_pred = evaluate(comp, baseline.assignment, comp.booked)
    new_pred = evaluate(comp, balanced.assignment, comp.booked)
    if new_pred["cost"] > old_pred["cost"] + .01:
        raise AssertionError(f"Balancing worsened primary optimum: {key}")
    old_actual = evaluate(comp, baseline.assignment, comp.actual)
    new_actual = evaluate(comp, balanced.assignment, comp.actual)
    old_l = loads(comp, baseline.assignment, comp.actual)
    new_l = loads(comp, balanced.assignment, comp.actual)
    old_pl = loads(comp, baseline.assignment, comp.booked)
    new_pl = loads(comp, balanced.assignment, comp.booked)
    room_rows = []
    for j, (weekday, room) in enumerate(comp.slots):
        # Strongly forced cases provide a conservative structural lower bound
        # for this room, independent of the optimizer or duration estimates.
        forced = [i for i in range(comp.n)
                  if comp.allowed[int(comp.surgeon[i])] == (j,)]
        forced_load = sum(float(comp.actual[i]) + 30. for i in forced) - (30. if forced else 0.)
        originally = [i for i in range(comp.n) if int(comp.original[i]) == j]
        original_load = sum(float(comp.actual[i]) + 30. for i in originally) - (30. if originally else 0.)
        room_rows.append({
            **key, "weekday": weekday, "room": room,
            "historical_realized_load": original_load,
            "booked_predicted_load": float(old_pl[j]),
            "balanced_predicted_load": float(new_pl[j]),
            "booked_realized_load": float(old_l[j]),
            "balanced_realized_load": float(new_l[j]),
            "strongly_forced_cases": len(forced),
            "strongly_forced_realized_minutes": forced_load,
            "unavoidably_above_720_by_forced_cases": forced_load > 720,
            "historical_long_cases_over_480": sum(comp.actual[i] > 480 for i in originally),
            "cases_baseline": sum(x == j for x in baseline.assignment),
            "cases_balanced": sum(x == j for x in balanced.assignment),
        })
    result = {
        **key,
        "baseline_realized_cost": old_actual["cost"],
        "balanced_realized_cost": new_actual["cost"],
        "realized_cost_change": new_actual["cost"] - old_actual["cost"],
        "baseline_predicted_cost": old_pred["cost"],
        "balanced_predicted_cost": new_pred["cost"],
        "baseline_room_days_over_720": old_actual["room_days_over_720"],
        "balanced_room_days_over_720": new_actual["room_days_over_720"],
        "baseline_max_realized_room_load": old_actual["max_room_load"],
        "balanced_max_realized_room_load": new_actual["max_room_load"],
        "baseline_max_predicted_room_load": old_pred["max_room_load"],
        "balanced_max_predicted_room_load": new_pred["max_room_load"],
        "baseline_surgeon_days_over_480": old_actual["surgeon_days_over_480"],
        "balanced_surgeon_days_over_480": new_actual["surgeon_days_over_480"],
        "baseline_moved_cases": old_actual["moved_cases"],
        "balanced_moved_cases": new_actual["moved_cases"],
        "baseline_solver_seconds": baseline.seconds,
        "balanced_solver_seconds": balanced.seconds,
        "balance_tie_seconds": balanced.tie_seconds,
    }
    return result, room_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Full previous benchmark.csv (not just the summary)")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--threshold", type=float, default=720.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.threads, args.seconds) <= 0 or args.workers * args.threads > 15:
        parser.error("Require positive arguments and workers*threads <= 15")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((args.benchmark.parent / "manifest.json").read_text())
    if old_manifest.get("prior_dates") != 1 or old_manifest.get("selected_cases") != 24506:
        raise ValueError("Audit requires matching full training benchmark (prior_dates=1)")
    digest = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    if digest != old_manifest["dataset_sha256"]:
        raise ValueError("Workbook SHA-256 mismatch")

    raw = pd.read_excel(args.workbook, usecols=COLUMNS)
    cohort, _, _, _ = build_cohort(raw)
    weeks = build_weeks(cohort, performed_events(raw), split="train", prior_dates=1)
    instances = {}
    for week in weeks:
        for j, component in enumerate(week.components()):
            instances[(week.group, week.monday, j)] = component
    previous = pd.read_csv(args.benchmark)
    if len(previous) != len(instances):
        raise ValueError("Full benchmark and reconstructed instance counts differ")
    pending = []
    for row in previous.itertuples(index=False):
        key = (row.group, row.monday, int(row.component))
        if key not in instances:
            raise ValueError(f"Unrecognized benchmark component {key}")
        if int(row.cases) != instances[key].n:
            raise ValueError(f"Case mismatch at {key}")
        if float(row.booked_max_room_load) > args.threshold:
            pending.append((dict(group=row.group, monday=row.monday,
                                 component=int(row.component)),
                            instances[key], {"booked_cost": row.booked_cost},
                            args.seconds, args.threads))
    print(f"Auditing {len(pending)} severe-load components with {args.workers} workers", flush=True)
    results, details = [], []
    if args.workers == 1:
        iterator = map(audit_component, pending)
        for summary, rooms in iterator:
            results.append(summary)
            details.extend(rooms)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(audit_component, task) for task in pending]
            for k, future in enumerate(as_completed(futures), 1):
                summary, rooms = future.result()
                results.append(summary)
                details.extend(rooms)
                if k % 25 == 0 or k == len(futures):
                    print(f"Finished {k}/{len(futures)}", flush=True)
    pd.DataFrame(results).sort_values(["group", "monday", "component"]).to_csv(
        args.output / "components.csv", index=False)
    pd.DataFrame(details).sort_values(["group", "monday", "component", "weekday", "room"]).to_csv(
        args.output / "rooms.csv", index=False)
    report = {
        "audited_components": len(results),
        "primary_cost_preserved": True,
        "max_room_load_booked": max((r["baseline_max_realized_room_load"] for r in results), default=0),
        "max_room_load_balanced": max((r["balanced_max_realized_room_load"] for r in results), default=0),
        "realized_cost_change": sum(r["realized_cost_change"] for r in results),
        "room_days_over_720_baseline": sum(r["baseline_room_days_over_720"] for r in results),
        "room_days_over_720_balanced": sum(r["balanced_room_days_over_720"] for r in results),
        "room_days_with_strongly_forced_load_over_720":
            sum(r["unavoidably_above_720_by_forced_cases"] for r in details),
        "additional_case_moves": sum(r["balanced_moved_cases"] - r["baseline_moved_cases"]
                                     for r in results),
        "balanced_solver_seconds_sum": sum(r["balanced_solver_seconds"] for r in results),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
