#!/usr/bin/env python3
"""Training-only pilot comparing duration losses and exact decision-aware losses.

All objectives select policies from the SAME small, predeclared affine advice
grid. No 2013 holdout cases or outcomes are used. The purpose is to decide
which loss merits a scalable training algorithm, not to claim fitted-model gains.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, build_cohort
from surgery.planner import display_and_duration
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, solve_adversary, solve_week


def _grid(text):
    values = tuple(float(s.strip()) for s in text.split(","))
    if not values or not all(np.isfinite(values)) or len(set(values)) != len(values):
        raise ValueError("Grid points must be distinct finite numbers")
    return values


def _fingerprints(workbook):
    source = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for path in ("run_weekly_learning.py", "surgery/data.py", "surgery/weekly_data.py",
                 "surgery/weekly_planner.py", "surgery/planner.py"):
        code.update(path.encode())
        code.update((source / path).read_bytes())
    return {"dataset_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
            "code_sha256": code.hexdigest()}


def _oracle(week, seconds, threads):
    """Optimize actual cost once per component, shared across all candidates."""
    rows = []
    for c in week.components():
        p = solve_week(c, c.actual, seconds=seconds, threads=threads, tie=False)
        if not p.complete:
            raise RuntimeError(f"Actual oracle unfinished: {week.group}/{week.monday} "
                               f"{p.status}")
        rows.append((list(p.assignment), float(p.value)))
    return rows


def candidate_job(task):
    week, period, offset, slope, alpha, h, gamma, oracles, seconds, threads = task
    advice = offset + slope * ((week.booked - 180.) / 180.)
    _, duration = display_and_duration(advice, week.booked, alpha, h)
    by_case = dict(zip(week.case_ids.tolist(), duration.tolist()))
    totals = {"case_error": 0., "vf": 0., "gap": 0., "spo": 0.,
              "regret": 0., "realized_cost": 0., "oracle_cost": 0.,
              "solver_seconds": 0., "booked_moves": 0, "cases": week.n}
    for component, (oracle_assignment, oracle_value) in zip(week.components(), oracles):
        d = np.asarray([by_case[int(i)] for i in component.case_ids], float)
        p = solve_week(component, d, seconds=seconds, threads=threads)
        q = solve_adversary(component, d, gamma=gamma, seconds=seconds, threads=threads)
        if not p.complete or not q.complete:
            return {"group": week.group, "monday": week.monday, "period": period,
                    "offset": offset, "slope": slope, "complete": False,
                    "reason": f"planner={p.status}, adversary={q.status}",
                    "cases": week.n}
        selected_actual = evaluate(component, p.assignment, component.actual)["cost"]
        predicted_oracle = evaluate(component, oracle_assignment, d)["cost"]
        error = component.actual - d
        theta = float(np.sum(1.75 * np.maximum(error, 0.)
                             + np.maximum(-error, 0.)))
        regret = selected_actual - oracle_value
        vf = p.value + theta - oracle_value
        gap = gamma * p.value + q.value - oracle_value
        spo = gamma * predicted_oracle + q.value - oracle_value
        if min(regret, vf - regret, gap - regret, spo - gap) < -.01:
            raise AssertionError(f"Invalid regret bound: {week.group}/{week.monday}")
        totals["case_error"] += float(np.abs(error).sum())
        totals["vf"] += vf
        totals["gap"] += gap
        totals["spo"] += spo
        totals["regret"] += regret
        totals["realized_cost"] += selected_actual
        totals["oracle_cost"] += oracle_value
        totals["booked_moves"] += evaluate(component, p.assignment, d)["moved_cases"]
        totals["solver_seconds"] += p.seconds + q.seconds
    return {"group": week.group, "monday": week.monday, "period": period,
            "offset": offset, "slope": slope, "complete": True, **totals}


def _write_grid(rows, destination):
    tmp = destination.with_suffix(".tmp")
    pd.DataFrame(rows).sort_values(
        ["group", "period", "monday", "offset", "slope"]
    ).to_csv(tmp, index=False)
    tmp.replace(destination)


def _summarize(grid, offsets, slopes):
    from collections import defaultdict

    metrics = ("case_error", "vf", "gap", "spo", "regret", "realized_cost", "oracle_cost")
    aggregate = defaultdict(lambda: defaultdict(float))
    for row in grid.to_dict("records"):
        if row["complete"] != True:
            continue
        key = (row["group"], row["period"], float(row["offset"]), float(row["slope"]))
        for column in (*metrics, "cases", "booked_moves"):
            aggregate[key][column] += float(row[column])
    selections, validations = [], []
    for group in GROUPS:
        available = [key for key in aggregate if key[0] == group and key[1] == "train"]
        if not available:
            raise ValueError(f"No completed training evaluations for {group}")
        choices = {
            "BOOKED": (0., 0.),
            "SHIFT": None,
            "CASE": None,
            "VF": None,
            "GAP": None,
            "SPO": None,
        }
        for method in ("SHIFT", "CASE", "VF", "GAP", "SPO"):
            name = "case_error" if method in ("SHIFT", "CASE") else method.lower()
            eligible = [k for k in available if method != "SHIFT" or k[3] == 0.]
            selected = min(eligible, key=lambda k: (
                aggregate[k][name], abs(k[2]) + abs(k[3]), k[2], k[3]))
            choices[method] = selected[2:]
        for method, (offset, slope) in choices.items():
            for period in ("train", "validation"):
                key = (group, period, offset, slope)
                if key not in aggregate:
                    raise ValueError(f"Missing {method} candidate in {period}: {key}")
                rec = aggregate[key]
                cases = rec["cases"]
                validations.append({
                    "group": group, "method": method, "period": period,
                    "offset": offset, "slope": slope, "cases": int(cases),
                    **{name + "_per_case": rec[name] / cases for name in metrics},
                    "moved_cases": int(rec["booked_moves"]),
                })
            selections.append({
                "group": group, "method": method, "offset": offset, "slope": slope
            })
    return pd.DataFrame(selections), pd.DataFrame(validations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--start-week-index", type=int, default=20)
    p.add_argument("--train-weeks", type=int, default=3)
    p.add_argument("--validation-weeks", type=int, default=2)
    p.add_argument("--offsets", default="-30,-15,0,15,30")
    p.add_argument("--slopes", default="-15,0,15")
    p.add_argument("--alpha", type=float, default=.8)
    p.add_argument("--h", type=float, default=30.)
    p.add_argument("--gamma", type=float, default=2.)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    offsets, slopes = _grid(args.offsets), _grid(args.slopes)
    if 0. not in offsets or 0. not in slopes:
        p.error("Both grids must include zero for the Booked comparator")
    if (args.start_week_index < 0 or args.train_weeks < 1 or args.validation_weeks < 1
            or args.workers < 1 or args.threads < 1 or args.workers * args.threads > 15
            or args.seconds <= 0 or args.gamma < 1 or not 0 < args.alpha <= 1
            or args.h <= 0):
        p.error("Invalid window, response, solver budget, or 15-core constraint")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("Output directory is nonempty; use a new directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_fingerprints(args.workbook),
        "start_week_index": args.start_week_index,
        "train_weeks": args.train_weeks, "validation_weeks": args.validation_weeks,
        "offsets": offsets, "slopes": slopes, "alpha": args.alpha, "h": args.h,
        "gamma": args.gamma, "seconds": args.seconds, "prior_dates": 1,
        "cost": {"idle": 1, "overtime": 1.75, "turnover": 30, "capacity": 480},
        "policy": "raw advice = offset + slope*(booked-180)/180; displayed advice clipped; response",
        "purpose": "training-only objective comparison; not final model or holdout evaluation",
    }
    manifest_path = args.output / "manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            p.error("Resume manifest mismatch; start a new experiment")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    raw = pd.read_excel(args.workbook, usecols=COLUMNS)
    cohort, _, _, _ = build_cohort(raw)
    weeks = build_weeks(cohort, performed_events(raw), split="train", prior_dates=1)
    selected = []
    for group in GROUPS:
        group_weeks = [w for w in weeks if w.group == group]
        window = group_weeks[args.start_week_index:args.start_week_index
                             +args.train_weeks+args.validation_weeks]
        if len(window) != args.train_weeks+args.validation_weeks:
            raise ValueError(f"Insufficient training weeks for {group}")
        selected.extend((w, "train" if j < args.train_weeks else "validation")
                        for j, w in enumerate(window))
    oracle_by_week = {}
    print("Solving shared realized oracles for selected training-period weeks...", flush=True)
    for week, _ in selected:
        oracle_by_week[(week.group, week.monday)] = _oracle(week, args.seconds, args.threads)

    results_path = args.output / "grid_results.csv"
    existing = []
    if args.resume and results_path.exists():
        # Only certified policy/week computations are reusable. A failed row
        # is replaced by a new attempt rather than poisoning future resumes.
        existing = [row for row in pd.read_csv(results_path).to_dict("records")
                    if row.get("complete") == True]
    done = {(r["group"], r["monday"], float(r["offset"]), float(r["slope"]))
            for r in existing if r.get("complete") == True}
    tasks = [
        (week, period, offset, slope, args.alpha, args.h, args.gamma,
         oracle_by_week[(week.group, week.monday)], args.seconds, args.threads)
        for week, period in selected for offset in offsets for slope in slopes
        if (week.group, week.monday, offset, slope) not in done
    ]
    print(f"{len(tasks)} policy×week evaluations with up to {args.workers} workers", flush=True)
    accumulated = existing[:]
    incomplete = 0
    if args.workers == 1:
        iterator = map(candidate_job, tasks)
        for k, row in enumerate(iterator, 1):
            accumulated.append(row)
            incomplete += not row["complete"]
            if k % 10 == 0 or k == len(tasks):
                _write_grid(accumulated, results_path)
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(candidate_job, item) for item in tasks]
            for k, future in enumerate(as_completed(futures), 1):
                row = future.result()
                accumulated.append(row)
                incomplete += not row["complete"]
                if k % 10 == 0 or k == len(tasks):
                    print(f"Finished {k}/{len(tasks)}", flush=True)
                    _write_grid(accumulated, results_path)
    if incomplete or any(row["complete"] != True for row in accumulated):
        raise SystemExit("Some oracles incomplete; see grid_results.csv and rerun --resume")
    grid = pd.DataFrame(accumulated)
    choices, validation = _summarize(grid, offsets, slopes)
    choices.to_csv(args.output / "policies.csv", index=False)
    validation.to_csv(args.output / "validation.csv", index=False)
    print(validation[validation.period.eq("validation")].to_string(index=False))


if __name__ == "__main__":
    main()
