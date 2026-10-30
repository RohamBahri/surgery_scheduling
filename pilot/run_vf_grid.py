"""Main behavioral-grid experiment: NAIVE, BOOKED, ORACLE, and direct VF."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot import _joint_vf_engine as engine
from pilot import run_joint_vf as single
from surgery.nonmonotone_learning import ResponseDC
from surgery.weekly_planner import evaluate, loads

METHODS = ("BOOKED", "NAIVE", "VF", "ORACLE")


def scenario_id(link, alpha, h):
    def enc(x):
        return format(x, ".8g").replace(".", "p")
    return f"{link}_a{enc(alpha)}_h{enc(h)}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--context", type=Path, default=single.DEFAULT_CONTEXT)
    p.add_argument("--out", type=Path, default=Path("artifacts/vf_behavior_grid"))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.8])
    p.add_argument("--hs", type=float, nargs="+", default=[20., 30., 60.])
    p.add_argument("--links", nargs="+", choices=("saturating", "nonmonotone"),
                   default=["saturating"])
    p.add_argument("--cores", type=int, default=15)
    p.add_argument("--plan-seconds", type=float, default=300.)
    p.add_argument("--case-steps", type=int, default=200,
                   help="Accuracy-only reference pDCA iterations")
    p.add_argument("--vf-outers", type=int, default=30)
    p.add_argument("--vf-pdca-steps", type=int, default=50)
    p.add_argument("--cuts", type=int, default=80)
    p.add_argument("--radius", type=float, default=100.)
    p.add_argument("--accuracy-mu", type=float, default=2.)
    p.add_argument("--accuracy-budget", type=float, default=.05)
    p.add_argument("--lambda-vf", type=float, default=.1)
    a = p.parse_args(argv)
    if not a.context.is_file():
        p.error("Missing private context checkpoint: " + str(a.context))
    if any(not np.isfinite(v) or not 0 < v < 1 for v in a.alphas):
        p.error("Each alpha must be strictly between zero and one")
    if any(not np.isfinite(v) or v <= 0 for v in a.hs):
        p.error("Each h must be positive and finite")
    if (a.cores < 1 or a.plan_seconds <= 0 or a.case_steps < 1 or
        a.vf_outers < 1 or a.vf_pdca_steps < 1 or a.cuts < 1 or
        a.radius <= 0 or any(not np.isfinite(v) or v < 0 for v in
            (a.accuracy_mu, a.accuracy_budget, a.lambda_vf))):
        p.error("Invalid optimization or accuracy configuration")
    a.context = a.context.resolve()
    a.out = a.out.resolve()
    a.scenarios = [(link, float(alpha), float(h)) for link, alpha, h in
                   itertools.product(dict.fromkeys(a.links),
                                     dict.fromkeys(a.alphas), dict.fromkeys(a.hs))]
    if len({scenario_id(*x) for x in a.scenarios}) != len(a.scenarios):
        p.error("Scenario identifiers collide; supply distinct alpha/h values")
    return a


def contract(a):
    return dict(spec="direct_vf_behavior_grid_v1", context_sha=single.file_sha(a.context),
                scenarios=[dict(link=l, alpha=x, h=h, id=scenario_id(l, x, h))
                           for l, x, h in a.scenarios],
                cores=a.cores, plan_seconds=a.plan_seconds,
                case_steps=a.case_steps, vf_outers=a.vf_outers,
                vf_pdca_steps=a.vf_pdca_steps, cuts=a.cuts, radius=a.radius,
                accuracy_mu=a.accuracy_mu, accuracy_budget=a.accuracy_budget,
                lambda_vf=a.lambda_vf, turnover=0, train_tie="primary",
                deploy_tie="hash_only", vf_features="77_case_plus_6_structural")


def ensure_contract(a):
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "grid_contract.json"
    expected = contract(a)
    if dest.is_file():
        if json.loads(dest.read_text()) != expected:
            raise ValueError("Grid configuration changed; select a different --out")
    else:
        engine.save(dest, expected)


def fit_naive(ctx, a):
    """Train one response-blind, decision-blind regularized LAD predictor."""
    path = a.out / "naive_lad.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("context_sha") != single.file_sha(a.context):
            raise ValueError("Naive predictor belongs to another dataset")
        w = np.asarray(saved["weights"], float)
    else:
        X = np.asarray(ctx["fit"]["X"], float)
        errors = np.asarray(ctx["fit"]["actual"], float) - np.asarray(
            ctx["fit"]["booked"], float)
        lam = engine.train_lambda(X, len(ctx["train"]), intercept=True)
        w, result = engine.lad_start(X, errors, lam, len(ctx["train"]), a.radius)
        if w is None:
            raise RuntimeError("Naive LAD optimizer did not converge: " + str(result))
        engine.save(path, dict(method="NAIVE_LAD", weights=w.tolist(),
                               lambda_lad=lam, optimizer=result,
                               features="77_case_only", context_sha=single.file_sha(a.context),
                               training_cases=len(X)))
    if (w.shape != (np.asarray(ctx["fit"]["X"]).shape[1],) or
        not np.isfinite(w).all() or np.max(np.abs(w)) > a.radius + 1e-8):
        raise ValueError("Invalid saved NAIVE coefficients")
    return w


def single_args(a, scenario, stage):
    link, alpha, h = scenario
    return ["--stage", stage, "--context", str(a.context),
            "--out", str(a.out / "scenarios" / scenario_id(*scenario)),
            "--naive-start", str(a.out / "naive_lad.json"),
            "--link", link, "--alpha", str(alpha), "--h", str(h),
            "--cores", str(a.cores), "--plan-seconds", str(a.plan_seconds),
            "--case-steps", str(a.case_steps), "--vf-outers", str(a.vf_outers),
            "--vf-pdca-steps", str(a.vf_pdca_steps), "--cuts", str(a.cuts),
            "--radius", str(a.radius), "--accuracy-mu", str(a.accuracy_mu),
            "--accuracy-budget", str(a.accuracy_budget),
            "--lambda-vf", str(a.lambda_vf), "--turnover", "0",
            "--train-tie", "primary", "--accuracy-gate", "off"]


def evaluate_rows(ctx, a, naive):
    """Cache common plans once; report all four methods on each scenario."""
    items = ctx["eval"]
    held = ctx["held"]
    b = np.asarray(held["booked"], float)
    y = np.asarray(held["actual"], float)
    X = np.asarray(held["X"], float)
    if len(b) != len(y) or X.shape[0] != len(b):
        raise ValueError("Evaluation context has inconsistent case dimensions")
    cache = engine.PlannerCache(a.out, a.cores, a.plan_seconds)
    shared = {"BOOKED": cache.solve(items, b, "test_BOOKED", tie=True),
              "ORACLE": cache.solve(items, y, "test_ORACLE", tie=True)}
    result = []
    policy_checks = {}
    for scenario in a.scenarios:
        link, alpha, h = scenario
        sid = scenario_id(*scenario)
        root = a.out / "scenarios" / sid
        check_args = single.parse_args(single_args(a, scenario, "evaluate"))
        _, selected = single.verified_context(check_args)
        scale = json.loads((root / "vf_features.json").read_text())
        if scale.get("schema") != "structural":
            raise ValueError("VF requires the frozen structural feature schema")
        H, _, _ = engine.five_features(ctx, items, cache, "eval",
            scaler=(np.asarray(scale["mean"]), np.asarray(scale["std"])),
            schema="structural")
        scores = np.column_stack([X, H]) @ np.asarray(selected["vf_theta"], float)
        naive_scores = X @ naive
        response = ResponseDC(b, alpha, h, 180., 1., link=link)
        d = {"BOOKED": b, "ORACLE": y,
             "NAIVE": response.reviewed(naive_scores),
             "VF": response.reviewed(scores)}
        displays = {"BOOKED": b, "ORACLE": y,
                    "NAIVE": b + response.display(naive_scores),
                    "VF": b + response.display(scores)}
        plans = dict(shared)
        for method in ("NAIVE", "VF"):
            plans[method] = cache.solve(items, d[method],
                                        "test_" + sid + "_" + method, tie=True)
        ratio = float(selected["accuracy_ratio"])
        policy_checks[sid] = dict(alpha=alpha, h=h, link=link,
            training_accuracy_ratio=ratio,
            training_accuracy_met=bool(ratio <= 1 + a.accuracy_budget + 1e-9),
            optimizer_status=selected["vf_status"],
            accuracy_reference=selected["accuracy_reference"],
            initialization=selected["initialization"])
        for key, comp, idx in items:
            for method in METHODS:
                rec = plans[method][key]
                assigned = rec["assignment"]
                measure = evaluate(comp, assigned, comp.actual)
                room_loads = loads(comp, assigned, comp.actual)
                result.append(dict(scenario=sid, link=link, alpha=alpha, h=h,
                    method=method, component=str(key), week=str(comp.monday),
                    cases=int(comp.n), complete=bool(rec["complete"]),
                    status=rec["status"], cost=float(measure["cost"]),
                    idle=float(measure["idle"]),
                    overtime=float(measure["overtime"]),
                    mae=float(np.abs(comp.actual - d[method][idx]).sum()),
                    displayed_mae=float(np.abs(
                        comp.actual - displays[method][idx]).sum()),
                    max_realized_load=float(room_loads.max()),
                    over_720_rooms=int(np.sum(room_loads > 720.)),
                    emptied_historical_surgeon_days=int(
                        measure["emptied_historical_surgeon_days"])))
        engine.save(root / "grid_evaluation.json", dict(
            status="EXPLORATORY_REUSED_2013_TEST", scenario=sid,
            policy=policy_checks[sid],
            entries=[r for r in result if r["scenario"] == sid]))
    return result, policy_checks


def group_components(rows):
    by = defaultdict(dict)
    for r in rows:
        key = (r["scenario"], r["component"])
        if r["method"] in by[key]:
            raise ValueError("Duplicate scenario/method/component entry")
        by[key][r["method"]] = r
    return by


def summarize(rows, checks, a):
    by = group_components(rows)
    scenario_ids = [scenario_id(*s) for s in a.scenarios]
    certified = {
        sid: {key for (ss, key), d in by.items() if ss == sid and
              all(m in d and d[m]["complete"] for m in METHODS)}
        for sid in scenario_ids}
    global_common = set.intersection(*certified.values())
    table = []
    for sid in scenario_ids:
        for sample, keys in (("all_feasible", {k for (ss, k) in by if ss == sid}),
                             ("scenario_certified", certified[sid]),
                             ("grid_common_certified", global_common)):
            subset = [by[(sid, key)] for key in sorted(keys)]
            n = sum(x["BOOKED"]["cases"] for x in subset)
            row = dict(scenario=sid, link=checks[sid]["link"],
                alpha=checks[sid]["alpha"], h=checks[sid]["h"], sample=sample,
                components=len(subset), cases=n,
                training_accuracy_ratio=checks[sid]["training_accuracy_ratio"],
                training_accuracy_met=checks[sid]["training_accuracy_met"])
            for method in METHODS:
                row[method + "_cost_per_case"] = (
                    sum(x[method]["cost"] for x in subset) / n if n else None)
                row[method + "_mae_per_case"] = (
                    sum(x[method]["mae"] for x in subset) / n if n else None)
                row[method + "_idle_per_case"] = (
                    sum(x[method]["idle"] for x in subset) / n if n else None)
                row[method + "_overtime_per_case"] = (
                    sum(x[method]["overtime"] for x in subset) / n if n else None)
                row[method + "_rooms_over_720"] = sum(
                    x[method]["over_720_rooms"] for x in subset)
            if n:
                vf = row["VF_cost_per_case"]
                booked = row["BOOKED_cost_per_case"]
                oracle = row["ORACLE_cost_per_case"]
                row["VF_minus_BOOKED"] = vf - booked
                row["VF_minus_NAIVE"] = vf - row["NAIVE_cost_per_case"]
                row["VF_gap_closed_fraction"] = (
                    (booked - vf) / (booked - oracle)
                    if booked > oracle + 1e-9 else None)
            table.append(row)
    # Paired calendar-week bootstraps on exactly the samples reported in the table.
    rng = np.random.default_rng(20261029)
    confidence = {}
    for sid in scenario_ids:
        confidence[sid] = {}
        for sample, keys in (("scenario_certified", certified[sid]),
                             ("grid_common_certified", global_common)):
            weeks = defaultdict(lambda: defaultdict(float))
            for key in sorted(keys):
                block = by[(sid, key)]
                week = block["BOOKED"]["week"]
                weeks[week]["cases"] += block["BOOKED"]["cases"]
                for m in METHODS:
                    weeks[week][m] += block[m]["cost"]
            ordered = [weeks[k] for k in sorted(weeks)]
            results = {}
            if ordered:
                draws = rng.integers(0, len(ordered), size=(5000, len(ordered)))
                den = np.array([z["cases"] for z in ordered], float)[draws].sum(axis=1)
                for baseline in ("BOOKED", "NAIVE"):
                    diffs = np.array([z["VF"] - z[baseline] for z in ordered])
                    samples = diffs[draws].sum(axis=1) / den
                    results["VF_minus_" + baseline] = dict(
                        estimate=float(diffs.sum() / sum(z["cases"] for z in ordered)),
                        calendar_week_bootstrap_95=np.quantile(
                            samples, [.025, .975]).tolist(),
                        better_weeks=int(np.sum(diffs < 0)),
                        worse_weeks=int(np.sum(diffs > 0)),
                        weeks=len(ordered), cases=int(sum(z["cases"] for z in ordered)))
            confidence[sid][sample] = results
    return table, confidence, dict(
        status="EXPLORATORY_REUSED_2013_TEST", methods=list(METHODS),
        scenarios=scenario_ids,
        certification_by_scenario={s: len(v) for s, v in certified.items()},
        grid_common_certified_components=len(global_common),
        grid_common_certified_cases=(sum(by[(scenario_ids[0], k)]["BOOKED"]["cases"]
                                         for k in global_common)),
        policy_checks=checks, paired_bootstrap=confidence)


def save_grid(a, entries, checks):
    table, intervals, summary = summarize(entries, checks, a)
    engine.save(a.out / "grid_summary.json", summary)
    engine.save(a.out / "grid_table.json", table)
    for name, records in (("grid.csv", table), ("component_results.csv", entries)):
        if not records:
            continue
        with (a.out / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(
                k for r in records for k in r)))
            writer.writeheader()
            writer.writerows(records)
    cols = ("BOOKED", "NAIVE", "VF", "ORACLE")
    lines = ["# VF behavioral-scenario grid", "",
             "Exploratory reused Jan-Jun 2013 test; costs per case, same certified "
             "components across the entire grid. No primary scenario.", "",
             "| Link | Alpha | h (min) | Cases | BOOKED | NAIVE | VF | ORACLE | "
             "VF - NAIVE | Accuracy met? |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in table:
        if row["sample"] != "grid_common_certified":
            continue
        def number(v):
            return f"{v:.3f}" if v is not None else "N/A"
        lines.append("| " + " | ".join([
            row["link"], number(row["alpha"]), number(row["h"]),
            str(row["cases"]), *(number(row[c + "_cost_per_case"]) for c in cols),
            number(row.get("VF_minus_NAIVE")),
            "yes" if row["training_accuracy_met"] else "NO"] ) + " |")
    lines += ["", "Scenario-specific and all-feasible summaries are in grid.csv.",
              "Paired calendar-week confidence intervals and certification counts "
              "are in grid_summary.json."]
    (a.out / "grid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not all(v["training_accuracy_met"] for v in checks.values()):
        print("WARNING: Some scenarios miss the 5% training accuracy safeguard; "
              "inspect grid_summary.json before interpreting them.", flush=True)


def main(argv=None):
    a = parse_args(argv)
    ensure_contract(a)
    single.configure_model(argparse.Namespace(turnover=0.))
    with a.context.open("rb") as stream:
        ctx = pickle.load(stream)
    if a.stage in ("all", "train"):
        fit_naive(ctx, a)
        for scenario in a.scenarios:
            print("\nTRAINING VF", scenario_id(*scenario), flush=True)
            single.main(single_args(a, scenario, "train"))
        if a.stage == "train":
            return
    if a.stage in ("all", "evaluate"):
        naive = fit_naive(ctx, a) if (a.out / "naive_lad.json").exists() else None
        if naive is None:
            raise ValueError("NAIVE predictor missing; run --stage train first")
        entries, checks = evaluate_rows(ctx, a, naive)
        save_grid(a, entries, checks)


if __name__ == "__main__":
    main()
