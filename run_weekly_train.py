#!/usr/bin/env python3
"""Full 60-week rich-feature CASE and VF training and 18-week evaluation.

Run from the updated repository with the already-built weekly_rich_v2
matrices; never fits on validation or 2013 held-out outcomes. All 180
training group-weeks are used; all 54 validation group-weeks are evaluated.
CASE and VF are DIFFERENT learned policies over the same 74-dimensional X.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from run_weekly_rich import (load_matrices, read_cohort, source_fingerprint,
                             oracle_assignment_sha256, verify_bounds)
from surgery.data import GROUPS
from surgery.planner import display_and_duration
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, solve_week
from surgery.weekly_training import (fit_case, fit_vf, training_lambda,
                                      asymmetric_error, save_json)


def trainer_source_fingerprint():
    """Independent from frozen feature-code checksum: permits safe X reuse."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("run_weekly_train.py", "surgery/weekly_training.py"):
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def load_weeks(workbook, meta, part):
    raw, cohort = read_cohort(workbook)
    expected = {tuple(t) for t in meta["training_keys" if part == "train" else "validation_keys"]}
    weeks = build_weeks(cohort, performed_events(raw), split="train", prior_dates=1)
    selected = [week for week in weeks if (week.group, week.monday) in expected]
    if len(selected) != len(expected):
        raise ValueError(f"Expected {len(expected)} {part} group-weeks, found {len(selected)}")
    return sorted(selected, key=lambda w: (w.monday, GROUPS.index(w.group)))


def check_week_cases(weeks, data):
    selected = [int(cid) for week in weeks for cid in week.case_ids]
    recorded = [int(cid) for cid in data["case_id"]]
    if len(selected) != len(set(selected)) or set(selected) != set(recorded):
        raise ValueError("The weekly planning roster differs from the feature cohort")
    return {int(cid): j for j, cid in enumerate(data["case_id"])}


def experiment_identity(meta, workbook, alpha, h, lambda_l1, iterations, max_outer,
                        data):
    workbook_hash = hashlib.sha256(workbook.read_bytes()).hexdigest()
    if workbook_hash != meta["workbook_sha256"]:
        raise ValueError("Workbook changed since rich features were built")
    return {
        "feature_code_sha256": source_fingerprint(),
        "trainer_code_sha256": trainer_source_fingerprint(),
        "workbook_sha256": workbook_hash,
        "matrix_sha256": meta["matrix_sha256"],
        "feature_names": meta["feature_names"],
        "train_cases": int(len(data["booked"])),
        "training_group_weeks": 180,
        "alpha": float(alpha), "h": float(h),
        "lambda_l1": float(lambda_l1),
        "iterations_per_majorization": int(iterations),
        "max_outer": int(max_outer),
        "policy": "booked + X@w, with fixed coefficient one",
        "penalty_scope": "all columns except intercept",
        "normalization": "per training case",
    }


def initialize(root, workbook, alpha, h, penalty, iterations, max_outer):
    meta, data = load_matrices(root)
    train = data["train"]
    if (len(train["booked"]) != 18705 or len(data["validation"]["booked"]) != 5779
            or len(meta["feature_names"]) != 74):
        raise ValueError("Unexpected rich-feature cohort or dimension")
    if meta["feature_names"][0] != "intercept" or "booked" in meta["feature_names"]:
        raise ValueError("The booked anchor must be fixed")
    X = train["X"].astype(float)
    lam = (training_lambda(X) if penalty == "auto" else float(penalty))
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("L1 regularization must be finite nonnegative")
    ident = experiment_identity(meta, workbook, alpha, h, lam, iterations, max_outer, train)
    manifest_path = root / "learning_settings.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != ident:
            raise ValueError("Learning settings changed; preserve the old result and use a new directory")
    else:
        save_json(manifest_path, ident)
    return meta, data, ident, X, lam


def train_case(root, workbook, alpha, h, penalty, iterations, max_outer):
    meta, data, settings, X, lam = initialize(
        root, workbook, alpha, h, penalty, iterations, max_outer)
    path = root / "model_case.json"
    if path.exists():
        print("CASE already completed:", path, flush=True)
        return
    baseline = json.loads((root / "lasso_policy.json").read_text())
    if baseline["features"] != meta["feature_names"]:
        raise ValueError("Baseline has a different feature order")
    booked, actual = data["train"]["booked"], data["train"]["actual"]
    result = fit_case(X, booked, actual, alpha, h, lam,
                      baseline["weights"], iterations)
    result["settings"] = settings
    result["train_n"] = len(booked)
    save_json(path, result)
    print(f"CASE fitted on {len(booked)} cases, objective={result['objective']:.6f}", flush=True)


def train_vf(root, workbook, alpha, h, penalty, iterations,
             max_outer, seconds, threads, workers):
    meta, data, settings, X, lam = initialize(
        root, workbook, alpha, h, penalty, iterations, max_outer)
    case_path = root / "model_case.json"
    if not case_path.exists():
        raise ValueError("Run --stage case first")
    case = json.loads(case_path.read_text())
    if case["settings"] != settings:
        raise ValueError("CASE and VF use different settings")
    weeks = load_weeks(workbook, meta, "train")
    index = check_week_cases(weeks, data["train"])
    if len(weeks) != 180:
        raise ValueError(f"VF requires all 180 training group-weeks, found {len(weeks)}")
    state = fit_vf(root, weeks, X, data["train"]["booked"], data["train"]["actual"],
                   index, alpha, h, lam, case["w"], settings,
                   max_outer, iterations, seconds, threads, workers)
    print(f"VF finished {len(state['iterations'])} outer iterations "
          f"with {len(weeks)} training group-weeks per iteration", flush=True)


def evaluate_week(task):
    """Independent week worker, all methods share one hindsight oracle."""
    week, model_durations, seconds, threads = task
    comps = week.components()
    oracle = [solve_week(c, c.actual, seconds=seconds,
                         threads=threads, tie=False) for c in comps]
    oracle_ok = all(o.complete and o.assignment is not None for o in oracle)
    fingerprint = oracle_assignment_sha256(comps, oracle) if oracle_ok else None
    rows = []
    for method, durations in model_durations.items():
        accumulated = {key: 0. for key in (
            "cases", "abs_error", "asym_error", "pred_cost",
            "realized_cost", "oracle_cost", "regret", "vf_bound",
            "idle", "overtime", "moved_cases",
            "room_days_over_720", "changed_days")}
        success = oracle_ok
        full = dict(zip(map(int, week.case_ids), map(float, durations)))
        for c, o in zip(comps, oracle):
            if not success:
                break
            d = np.array([full[int(cid)] for cid in c.case_ids], float)
            planned = solve_week(c, d, seconds=seconds, threads=threads, tie=True)
            if not planned.complete or planned.assignment is None:
                success = False
                break
            pred = evaluate(c, planned.assignment, d)
            real = evaluate(c, planned.assignment, c.actual)
            error = c.actual - d
            asymmetric = float(asymmetric_error(error).sum())
            regret = real["cost"] - o.value
            bound = pred["cost"] + asymmetric - o.value
            verify_bounds(regret, bound, bound, bound)
            quantities = {
                "cases": c.n, "abs_error": np.abs(error).sum(),
                "asym_error": asymmetric, "pred_cost": pred["cost"],
                "realized_cost": real["cost"], "oracle_cost": o.value,
                "regret": regret, "vf_bound": bound, "idle": real["idle"],
                "overtime": real["overtime"], "moved_cases": real["moved_cases"],
                "room_days_over_720": real["room_days_over_720"],
                "changed_days": real["moved_weekdays"]
            }
            for field, val in quantities.items():
                accumulated[field] += float(val)
        rows.append({
            "group": week.group, "monday": week.monday, "method": method,
            "complete": bool(success), "oracle_assignment_sha256": fingerprint,
            **(accumulated if success else {key: np.nan for key in accumulated})
        })
    return rows


def evaluate_models(root, workbook, alpha, h, penalty, iterations,
                    max_outer, seconds, threads, workers):
    meta, data, settings, _, _ = initialize(
        root, workbook, alpha, h, penalty, iterations, max_outer)
    baseline = json.loads((root / "lasso_policy.json").read_text())
    case = json.loads((root / "model_case.json").read_text())
    vf = json.loads((root / "vf_progress.json").read_text())
    if case.get("settings") != settings or vf.get("settings") != settings:
        raise ValueError("Training policies have incompatible settings")
    if vf.get("complete") is not True:
        raise ValueError("VF training has not completed its configured iterations")
    names = meta["feature_names"]
    if baseline["features"] != names:
        raise ValueError("Lasso feature order mismatch")
    policies = {"LASSO": baseline["weights"], "CASE": case["w"], "VF": vf["w"]}
    for name, w in policies.items():
        if len(w) != len(names) or not np.isfinite(w).all():
            raise ValueError(f"Invalid policy {name}")
    validation = data["validation"]
    X, booked = validation["X"].astype(float), validation["booked"]
    predictions = {"BOOKED": booked.copy()}
    for name, weights in policies.items():
        _, reviewed = display_and_duration(X @ np.asarray(weights, float),
                                            booked, alpha, h)
        predictions[name] = reviewed
    weeks = load_weeks(workbook, meta, "validation")
    idx = check_week_cases(weeks, validation)
    if len(weeks) != 54:
        raise ValueError("Evaluation requires exactly 54 validation group-weeks")

    compare_settings = {
        "learn_settings": settings,
        "model_sha256": {
            file: hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in ("lasso_policy.json", "model_case.json", "vf_progress.json")},
        "evaluated_methods": sorted(predictions),
        "prediction_scenario": {"alpha": alpha, "h": h}
    }
    path = root / "evaluation_settings.json"
    if path.exists():
        if json.loads(path.read_text()) != compare_settings:
            raise ValueError("Evaluation policy or settings changed; use a new result directory")
    else:
        save_json(path, compare_settings)
    output = root / "evaluation_all_validation.csv"
    records = {}
    if output.exists():
        old = pd.read_csv(output)
        for record in old.to_dict("records"):
            key = (record["group"], record["monday"], record["method"])
            if key in records:
                raise ValueError("Duplicate method-week in evaluation output")
            records[key] = record

    tasks = []
    for week in weeks:
        matches = [records.get((week.group, week.monday, method))
                   for method in predictions]
        digest = {row.get("oracle_assignment_sha256") for row in matches
                  if row is not None and row.get("complete") == True}
        if (all(row is not None and row.get("complete") == True for row in matches)
                and len(digest) == 1 and all(isinstance(x, str) and len(x) == 64
                                             for x in digest)):
            continue
        for method in predictions:
            records.pop((week.group, week.monday, method), None)
        local = np.array([idx[int(cid)] for cid in week.case_ids])
        duration = {name: pred[local] for name, pred in predictions.items()}
        tasks.append((week, duration, seconds, threads))

    def persist(rows):
        for row in rows:
            records[(row["group"], row["monday"], row["method"])] = row
        temp = output.with_suffix(".tmp")
        pd.DataFrame([records[key] for key in sorted(records)]).to_csv(temp, index=False)
        temp.replace(output)

    print(f"Evaluation: {len(tasks)} pending group-weeks, {workers} worker(s) "
          f"x {threads} Gurobi thread(s)", flush=True)
    if workers == 1:
        for task in tasks:
            rows = evaluate_week(task)
            persist(rows)
            print(rows[0]["group"], rows[0]["monday"], "evaluated", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, max(1,len(tasks))),
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(evaluate_week, task) for task in tasks]
            for future in as_completed(futures):
                rows = future.result()
                persist(rows)
                print(rows[0]["group"], rows[0]["monday"], "evaluated", flush=True)
    all_rows = pd.DataFrame(records.values())
    complete = (len(all_rows) == 54 * len(predictions)
                and all_rows.complete.eq(True).all())
    if not complete:
        raise RuntimeError("Uncertified validation weeks remain. Resume with more --seconds")
    summarize(root, all_rows)


def summarize(root, records, bootstrap_draws=10000):
    """Calendar-week paired bootstrap; do not resample individual cases."""
    if any(records.groupby(["group", "monday"]).method.nunique() != 4):
        raise ValueError("Each group-week must contain four complete methods")
    totals = {}
    for method, frame in records.groupby("method"):
        n = frame.cases.sum()
        totals[method] = {
            "cases": int(n), "group_weeks": int(len(frame)),
            "mae": float(frame.abs_error.sum() / n),
            "realized_cost": float(frame.realized_cost.sum()),
            "realized_cost_per_case": float(frame.realized_cost.sum() / n),
            "regret_per_case": float(frame.regret.sum() / n),
            "vf_bound_per_case": float(frame.vf_bound.sum() / n),
            "overtime_minutes": float(frame.overtime.sum()),
            "overloaded_blocks": int(frame.room_days_over_720.sum())
        }
    group_summary = {}
    for (group, method), frame in records.groupby(["group", "method"]):
        group_summary.setdefault(group, {})[method] = {
            "cases": int(frame.cases.sum()),
            "mae": float(frame.abs_error.sum() / frame.cases.sum()),
            "realized_cost": float(frame.realized_cost.sum()),
            "regret": float(frame.regret.sum())}

    records = records.copy()
    records["delta_cost"] = 0.
    pivot = records.pivot_table(index=["monday", "group"], columns="method",
                                 values="realized_cost", aggfunc="sum")
    pair_week = pivot.reset_index().groupby("monday")[["BOOKED", "LASSO", "CASE", "VF"]].sum()
    rng = np.random.default_rng(20261013)
    book = pair_week.BOOKED.to_numpy()
    uncertainty = {}
    for method in ("LASSO", "CASE", "VF"):
        delta = pair_week[method].to_numpy() - book
        draws = rng.integers(0, len(book), (bootstrap_draws, len(book)))
        pct = 100 * delta[draws].sum(axis=1) / book[draws].sum(axis=1)
        uncertainty[method] = {
            "difference_total": float(delta.sum()),
            "percent_change": float(100 * delta.sum() / book.sum()),
            "calendar_weeks": len(book),
            "bootstrap_ci_95_percent": [float(z) for z in np.quantile(pct, [.025, .975])],
            "bootstrap_draws": bootstrap_draws}
    report = {"methods": totals, "by_group": group_summary,
              "paired_calendar_week_uncertainty": uncertainty,
              "interpretation": "Exploratory development validation only; 2013 test untouched."}
    save_json(root / "evaluation_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--stage", choices=("case", "vf", "evaluate"), required=True)
    p.add_argument("--alpha", type=float, default=.8)
    p.add_argument("--h", type=float, default=30.)
    p.add_argument("--lambda-l1", default="auto")
    p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--max-outer", type=int, default=6)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seconds", type=float, default=180.)
    args = p.parse_args()
    if (args.workers < 1 or args.threads < 1 or args.workers * args.threads > 15
            or args.seconds <= 0 or args.iterations < 1 or args.max_outer < 1
            or not 0 < args.alpha <= 1 or args.h <= 0):
        p.error("Require 1 <= workers*threads <= 15, positive budgets and response parameters")
    if not args.artifacts.is_dir():
        p.error(f"Missing rich-feature artifact directory: {args.artifacts}")
    kw = (args.artifacts, args.workbook, args.alpha, args.h,
          args.lambda_l1, args.iterations, args.max_outer)
    if args.stage == "case":
        train_case(*kw)
    elif args.stage == "vf":
        train_vf(*kw, args.seconds, args.threads, args.workers)
    else:
        evaluate_models(*kw, args.seconds, args.threads, args.workers)


if __name__ == "__main__":
    main()
