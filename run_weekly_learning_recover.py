#!/usr/bin/env python3
"""Recover incomplete weekly learning-grid cells without rerunning certified cells.

Keeps the numerical learning/planner sources and candidate grid unchanged.
The difficult jobs run sequentially with up to 15 Gurobi threads each.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, build_cohort
from surgery.weekly_data import build_weeks, performed_events
from run_weekly_learning import (
    _fingerprints, _oracle, _summarize, _write_grid, candidate_job,
)


def _cell_key(row):
    return (str(row["group"]), str(row["monday"]), str(row["period"]),
            float(row["offset"]), float(row["slope"]))


def _read_previous(previous, manifest):
    """Refuse to mix results from different grids or incomplete input files."""
    path = previous / "grid_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    grid = pd.read_csv(path)
    required = {"group", "monday", "period", "offset", "slope", "complete",
                "cases", "case_error", "vf", "gap", "spo", "regret",
                "realized_cost", "oracle_cost", "solver_seconds", "booked_moves"}
    if required - set(grid.columns):
        raise ValueError("Grid is missing columns: " + str(required - set(grid.columns)))
    if grid.empty or grid.duplicated(["group", "monday", "period", "offset", "slope"]).any():
        raise ValueError("Grid is empty or contains duplicate evaluations")
    expected_count = 0
    for group in GROUPS:
        for period, count in (("train", manifest["train_weeks"]),
                              ("validation", manifest["validation_weeks"])):
            expected_count += count * len(manifest["offsets"]) * len(manifest["slopes"])
    if len(grid) != expected_count:
        raise ValueError(f"Expected {expected_count} grid cells; received {len(grid)}")
    all_keys = {_cell_key(row) for row in grid.to_dict("records")}
    if len(all_keys) != expected_count:
        raise ValueError("Grid keys are not unique")
    for row in grid.to_dict("records"):
        if row["group"] not in GROUPS or row["period"] not in ("train", "validation"):
            raise ValueError("Invalid group or period")
        if (row["offset"] not in manifest["offsets"]
                or row["slope"] not in manifest["slopes"]):
            raise ValueError("Grid contains unexpected policy parameter")
        if row["complete"] == True:
            required_numbers = ("case_error", "vf", "gap", "spo", "regret",
                                "realized_cost", "oracle_cost", "solver_seconds", "booked_moves")
            if not all(np.isfinite(float(row[col])) for col in required_numbers):
                raise ValueError("A certified grid row has missing or invalid metrics")
            if (row["gap"] + .01 < row["regret"]
                    or row["vf"] + .01 < row["regret"]
                    or row["spo"] + .01 < row["gap"]):
                raise ValueError("A certified grid row violates regret inequalities")
    return grid


def _weeks(cohort, events, manifest):
    weeks = build_weeks(cohort, events, split="train",
                        prior_dates=int(manifest["prior_dates"]))
    selected = {}
    for group in GROUPS:
        group_weeks = [w for w in weeks if w.group == group]
        start = int(manifest["start_week_index"])
        ntrain = int(manifest["train_weeks"])
        nvalid = int(manifest["validation_weeks"])
        window = group_weeks[start:start + ntrain + nvalid]
        if len(window) != ntrain + nvalid:
            raise ValueError(f"Not enough historical training weeks in {group}")
        for i, week in enumerate(window):
            key = (group, week.monday)
            if key in selected:
                raise AssertionError("Duplicate group-week")
            selected[key] = ("train" if i < ntrain else "validation", week)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--previous", required=True, type=Path,
                        help="Directory containing original manifest.json and grid_results.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="NEW directory for merged complete grid and validation")
    parser.add_argument("--seconds", type=float, default=600.,
                        help="Per solve budget for incomplete jobs")
    parser.add_argument("--threads", type=int, default=15,
                        help="Threads for one difficult solve; maximum total is 15")
    args = parser.parse_args()
    if not 1 <= args.threads <= 15 or args.seconds <= 0:
        parser.error("Require threads from 1 to 15 and seconds > 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    manifest_path = args.previous / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    current = _fingerprints(args.workbook)
    for name, value in current.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"{name} differs from the original experiment; "
                "do not reuse its numerical results across source or workbook changes"
            )
    grid = _read_previous(args.previous, manifest)
    print(f"Checking the {len(grid)} original candidate evaluations...", flush=True)
    raw = pd.read_excel(args.workbook, usecols=COLUMNS)
    cohort, _, _, _ = build_cohort(raw)
    selected = _weeks(cohort, performed_events(raw), manifest)
    old = {_cell_key(row): row for row in grid.to_dict("records")}
    expected = set()
    for (group, monday), (period, week) in selected.items():
        for offset, slope in itertools.product(manifest["offsets"], manifest["slopes"]):
            key = (group, monday, period, float(offset), float(slope))
            expected.add(key)
            if key not in old or int(old[key]["cases"]) != week.n:
                raise ValueError(f"Missing or inconsistent original candidate: {key}")
    if expected != set(old):
        raise ValueError("Original cells disagree with the reconstructed weekly cohort")
    outstanding = sorted((key for key, row in old.items() if row["complete"] != True),
                         key=lambda key: (key[0], key[1], key[3], key[4]))
    print(f"Reusing {len(old)-len(outstanding)} completed evaluations; "
          f"retrying {len(outstanding)} unfinished evaluations "
          f"on one {args.threads}-thread solver at a time.", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        **manifest,
        "recovery": {
            "original_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "original_grid_sha256":
                hashlib.sha256((args.previous / "grid_results.csv").read_bytes()).hexdigest(),
            "recovered_cells": len(outstanding), "threads": args.threads,
            "seconds": args.seconds,
            "original_directory": str(args.previous),
        }
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")

    retries = []
    try:
        for i, key in enumerate(outstanding, 1):
            group, monday, period, offset, slope = key
            _, week = selected[(group, monday)]
            # Only perfect-duration oracles for the two failed group-weeks
            # are recomputed; all certified grid cells are reused as-is.
            print(f"Retry {i}/{len(outstanding)}: {group}, {monday}, "
                  f"offset={offset:g}, slope={slope:g}", flush=True)
            shared_oracle = _oracle(week, args.seconds, args.threads)
            recovered = candidate_job(
                (week, period, offset, slope, float(manifest["alpha"]),
                 float(manifest["h"]), float(manifest["gamma"]), shared_oracle,
                 args.seconds, args.threads))
            if _cell_key(recovered) != key or int(recovered["cases"]) != week.n:
                raise AssertionError("Recovered cell identity or cohort changed")
            old[key] = recovered
            retries.append({"group": group, "monday": monday, "period": period,
                            "offset": offset, "slope": slope,
                            "complete": recovered["complete"],
                            "reason": recovered.get("reason", ""),
                            "solver_seconds": recovered.get("solver_seconds")})
            _write_grid(list(old.values()), args.output / "grid_results.csv")
    finally:
        _write_grid(list(old.values()), args.output / "grid_results.csv")
        pd.DataFrame(retries).to_csv(args.output / "retries.csv", index=False)

    if any(row["complete"] != True for row in old.values()):
        raise SystemExit("At least one candidate remains unfinished; see retries.csv")
    policies, validation = _summarize(pd.DataFrame(list(old.values())),
                                      manifest["offsets"], manifest["slopes"])
    policies.to_csv(args.output / "policies.csv", index=False)
    validation.to_csv(args.output / "validation.csv", index=False)
    print("\nCOMPLETE: matched later-week validation\n", flush=True)
    print(validation[validation.period == "validation"].to_string(index=False))


if __name__ == "__main__":
    main()
