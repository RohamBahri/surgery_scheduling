#!/usr/bin/env python3
"""Two-fold, full-cohort comparison: CASE, matched statistical, shared and per-week advice.

Example: python pilot/run_advice_comparison.py --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx --out artifacts/advice --cores 15
"""
import os
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["SURGERY_RESEARCH_LINK"] = "inverse"

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize

from pilot.run_asdbs import prepare, get_case
from run_weekly_rich import read_cohort
from surgery.asdbs import components, write
from surgery.advice_comparison import (baseline_job, evaluated_job, init_worker,
                                       residual_history, reviewed, shared_job, LAM)
from surgery.anchored_core import key


def parallel(fn, items, state, cores):
    with ProcessPoolExecutor(max_workers=cores, mp_context=get_context("spawn"),
                             initializer=init_worker, initargs=(state,)) as pool:
        yield from pool.map(fn, items, chunksize=1)


def case_baseline(folder, data):
    folder.mkdir(parents=True, exist_ok=True)
    w = get_case(folder, data)
    if not np.isfinite(w).all():
        raise RuntimeError("CASE fit contains nonfinite coefficients")
    data["case_w"] = w
    return w


def shared_control(data, H):
    """Matched response-aware statistical control, using exactly the five features."""
    base = data["X"] @ data["case_w"]
    booked, actual = data["booked"], data["actual"]
    d0 = reviewed(booked, base)

    def loss(beta):
        d = reviewed(booked, base + H @ beta)
        error = actual - d
        return (np.maximum(error, 0).sum()*1.75
                + np.maximum(-error, 0).sum()
                + LAM/2 * np.sum((d-d0)**2)) / len(booked)

    answer = minimize(loss, np.zeros(H.shape[1]), method="Powell",
                      options=dict(maxiter=150, xtol=1e-4, ftol=1e-7))
    if not np.isfinite(answer.fun):
        raise RuntimeError("Statistical matched control did not fit")
    return np.asarray(answer.x, float)


def fit_features(folder, items, data, names, seconds, cores):
    path = folder / "five_features.npz"
    if path.exists():
        with np.load(path) as old:
            return old["H"], old["mean"], old["scale"]
    H = np.zeros((len(data["booked"]), 5), float)
    seen = np.zeros(len(H), bool)
    state = dict(data=data, names=names, seconds=seconds)
    for i, (_, rows, features) in enumerate(parallel(baseline_job, items, state, cores)):
        if seen[rows].any():
            raise AssertionError("Repeated case in the decomposition")
        H[rows], seen[rows] = features, True
        if (i+1) % 200 == 0:
            print(f"CASE baseline for 5-feature training: {i+1}/{len(items)}", flush=True)
    if not seen.all():
        raise AssertionError("Incomplete feature cohort")
    mu, sd = H.mean(axis=0), H.std(axis=0)
    sd = np.where(sd < 1e-8, 1., sd)
    H = (H-mu)/sd
    np.savez_compressed(path, H=H, mean=mu, scale=sd)
    return H, mu, sd


def test_features(items, data, names, mean, scale, seconds, cores):
    H = np.zeros((len(data["booked"]), 5), float)
    seen = np.zeros(len(H), bool)
    state = dict(data=data, names=names, seconds=seconds)
    for _, rows, features in parallel(baseline_job, items, state, cores):
        H[rows], seen[rows] = (features-mean)/scale, True
    if not seen.all():
        raise AssertionError("Incomplete held-out feature cohort")
    return H


def fit_shared(folder, items, data, H, pools, beta, seconds, cores, rounds, seeds):
    path = folder / "shared.json"
    if path.exists():
        return np.asarray(json.loads(path.read_text())["theta"], float)
    theta = np.zeros(H.shape[1])
    history = []
    for round_id in range(rounds):
        step = 8. / (2**round_id)
        candidates = [beta+theta]
        for j in range(H.shape[1]):
            for sign in (-1., 1.):
                v = theta.copy()
                v[j] += sign*step
                candidates.append(beta+v)
        state = dict(data=data, H=H, pools=pools, candidates=candidates,
                     seeds=seeds, seconds=seconds)
        scores = np.zeros((len(candidates), 2))
        missing = np.zeros(len(candidates), int)
        weekly = {j: {} for j in range(len(candidates))}
        for ck, monday, n, values in parallel(shared_job, items, state, cores):
            for j, v in enumerate(values):
                if v is None:
                    missing[j] += 1
                    continue
                scores[j] += v
                weekly[j][monday] = weekly[j].get(monday, 0.) + (v[1]-values[0][1] if values[0] is not None else 0.)
        valid = [j for j in range(len(candidates)) if missing[j] == 0]
        if 0 not in valid:
            raise RuntimeError("Cannot certify shared-policy reference on every fitting component")
        best = min(valid, key=lambda j: scores[j, 0])
        gain = scores[0, 1]-scores[best, 1]
        sums = np.array(list(weekly[best].values()), float)
        se = (sums.std(ddof=1)*np.sqrt(len(sums)) if len(sums)>1 else np.inf)
        accepted = best != 0 and gain > 1.645*se
        history.append(dict(round=round_id, best_index=best,
                            bank_A_gain=float(scores[0, 0]-scores[best, 0]),
                            bank_B_gain=float(gain), bank_B_se=float(se),
                            accepted=bool(accepted), failed_components=missing.tolist()))
        print(f"Shared round {round_id+1}/{rounds}: {history[-1]}", flush=True)
        if not accepted:
            break
        theta = candidates[best]-beta
    write(path, dict(beta=beta.tolist(), theta=theta.tolist(), history=history))
    return theta


def save_components(folder, items, state, cores):
    folder.mkdir(parents=True, exist_ok=True)
    pending = [item for item in items if not (folder / (item[0]+".json")).exists()]
    total = len(pending)
    for i, (ck, result) in enumerate(parallel(evaluated_job, pending, state, cores)):
        write(folder / (ck+".json"), result)
        if (i+1) % 100 == 0:
            print(f"Evaluation: {i+1}/{total} components", flush=True)
    return [json.loads((folder/(item[0]+".json")).read_text()) for item in items]


def report(rows, seeds, filename):
    from collections import defaultdict
    arms = list(rows[0]["arms"])
    n = sum(row["cases"] for row in rows)
    by_monday = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in rows:
        for name, rec in row["arms"].items():
            for seed in seeds:
                entry = by_monday[row["monday"]][name, seed]
                entry[0] += rec["realized"][str(seed)]
                entry[1] += row["cases"]
    result = dict(n_cases=n, n_components=len(rows),
                  mondays=len(by_monday), seeds=list(seeds), arms={})
    rng = np.random.default_rng(20261020)
    mondays = sorted(by_monday)
    boot = rng.integers(len(mondays), size=(5000, len(mondays)))
    for name in arms:
        diffs = []
        for seed in seeds:
            paired = np.array([[by_monday[m][name,seed][0]-by_monday[m]["CASE",seed][0],
                                by_monday[m][name,seed][1]] for m in mondays], float)
            if not np.array_equal(paired[:, 1], [by_monday[m]["CASE",seed][1] for m in mondays]):
                raise AssertionError("Unmatched cohort or tie rule")
            b = paired[boot]
            samples = b[:, :, 0].sum(axis=1)/b[:, :, 1].sum(axis=1)
            diffs.append(dict(seed=seed, difference=float(paired[:, 0].sum()/n),
                              ci95=np.quantile(samples, [.025, .975]).tolist()))
        result["arms"][name] = dict(per_seed=diffs,
              changed_cases=sum(row["arms"][name]["changed"] for row in rows))
    write(filename, result)
    return result


def run_fold(args, fold, cohort):
    folder = args.out / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    fit, held, trainweeks, evalweeks, fitindex, evalindex, names, spec = prepare(args.workbook, fold)
    if len(fit["booked"]) + len(held["booked"]) != 24506:
        raise AssertionError("Expected the entire 24,506-case development cohort")
    fingerprint = dict(workbook=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                       model_source={p: hashlib.sha256((Path(__file__).resolve().parents[1]/p).read_bytes()).hexdigest()
                                     for p in ("pilot/run_advice_comparison.py", "surgery/advice_comparison.py",
                                               "surgery/anchored_weekly_solver.py", "surgery/rich_features.py")},
                       fold=fold, spec=spec, cores=args.cores, seconds=args.seconds,
                       budget=args.budget, shared_rounds=args.shared_rounds)
    manifest = folder / "settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise RuntimeError("Experiment specification changed; use a fresh output directory")
    write(manifest, fingerprint)
    case_baseline(folder, fit)
    held["case_w"] = fit["case_w"]
    fit_items = components(trainweeks, fitindex)
    eval_items = components(evalweeks, evalindex)
    H, mean, scale = fit_features(folder, fit_items, fit, names, args.seconds, args.cores)
    H_eval = test_features(eval_items, held, names, mean, scale, args.seconds, args.cores)
    fit_frame = cohort[cohort.case_id.isin(fit["case_id"])]
    pools = residual_history(fit_items, fit, fit_frame)
    beta = shared_control(fit, H)
    seeds = (0, 1, 2)
    theta = fit_shared(folder, fit_items, fit, H, pools, beta,
                       args.seconds, args.cores, args.shared_rounds, seeds)
    state = dict(data=held, H=H_eval, pools=pools, beta=beta, theta=theta,
                 seeds=seeds, seconds=args.seconds, budget=args.budget)
    rows = save_components(folder/"components", eval_items, state, args.cores)
    output = report(rows, seeds, folder/"summary.json")
    print(f"Fold {fold}: {output['n_cases']} held-out cases, {output['n_components']} components")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/advice_comparison"))
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--budget", type=float, default=900.)
    parser.add_argument("--shared-rounds", type=int, default=2)
    parser.add_argument("--fold", choices=("0", "1", "both"), default="both")
    args = parser.parse_args()
    if not 1 <= args.cores <= 15 or args.seconds <= 0 or args.budget <= 0:
        parser.error("cores must be 1..15 and time limits positive")
    args.out.mkdir(parents=True, exist_ok=True)
    _, cohort = read_cohort(args.workbook)
    all_rows = []
    for fold in ((0, 1) if args.fold == "both" else (int(args.fold),)):
        all_rows.extend(run_fold(args, fold, cohort))
    if args.fold == "both":
        report(all_rows, (0, 1, 2), args.out/"pooled.json")


if __name__ == "__main__":
    main()
