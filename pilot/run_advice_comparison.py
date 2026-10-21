#!/usr/bin/env python3
"""Two-fold, full-cohort comparison: CASE, matched statistical, shared and per-week advice.

Example: python pilot/run_advice_comparison.py --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx --out artifacts/advice --cores 15
"""
import os
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

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


def case_baseline(folder, data, cores):
    folder.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cores):
        w = get_case(folder, data)
    if not np.isfinite(w).all():
        raise RuntimeError("CASE fit contains nonfinite coefficients")
    data["case_w"] = w
    return w



def out_of_fold_residuals(folder, cohort, weeks, data, cores):
    """Four calendar-week inner folds: fit features and response-aware CASE anew."""
    from surgery.rich_features import RichCorrectionEncoder
    from surgery.weekly_training import fit_fixed
    from pilot.run_pilot import add_context
    from threadpoolctl import threadpool_limits

    path = folder/"residual_oof.npy"
    if path.exists():
        residuals = np.load(path)
        if residuals.shape == data["booked"].shape:
            return residuals
        raise RuntimeError("Stale OOF residual dimensions")
    fitted = cohort[cohort.case_id.isin(data["case_id"])].copy()
    pos = {int(cid): i for i, cid in enumerate(data["case_id"])}
    dates = sorted({w.monday for w in weeks})
    selected_lam = json.loads((folder/"case_policy.json").read_text())["lambda_l1"]
    residuals = np.full(len(pos), np.nan)
    for f in range(4):
        held_days = {d for j, d in enumerate(dates) if j % 4 == f}
        fit_weeks = [w for w in weeks if w.monday not in held_days]
        eval_weeks = [w for w in weeks if w.monday in held_days]
        train_ids = {int(cid) for w in fit_weeks for cid in w.case_ids}
        eval_ids = {int(cid) for w in eval_weeks for cid in w.case_ids}
        inner_train = fitted[fitted.case_id.isin(train_ids)].reset_index(drop=True)
        inner_eval = fitted[fitted.case_id.isin(eval_ids)].reset_index(drop=True)
        enc = RichCorrectionEncoder(history_mode="leave_date_out")
        X, T = enc.fit_transform(inner_train), enc.transform(inner_eval)
        names = list(enc.names)
        train_idx = {int(cid): i for i, cid in enumerate(inner_train.case_id)}
        eval_idx = {int(cid): i for i, cid in enumerate(inner_eval.case_id)}
        X, extra, mu, sd = add_context(X, fit_weeks, train_idx, variant="cohort")
        T, _, _, _ = add_context(T, eval_weeks, eval_idx,
                                variant="cohort", scaler=(mu, sd))
        names += list(extra)
        j = names.index("surgeon_week_booked_minutes")
        for name in ("hist_surgeon_mean_abs_error", "hist_procedure_mean_abs_error"):
            if name in names:
                k = names.index(name)
                X = np.column_stack([X, X[:, k]*X[:, j]])
                T = np.column_stack([T, T[:, k]*T[:, j]])
                names.append(name+"_x_surgeon_week_load")
        with threadpool_limits(limits=cores):
            fit = fit_fixed(X, inner_train.booked.to_numpy(float),
                            inner_train.actual.to_numpy(float), .8, 30.,
                            selected_lam, np.zeros(X.shape[1]), iterations=250)
        forecast = reviewed(inner_eval.booked.to_numpy(float),
                            T @ np.asarray(fit["w"], float))
        residuals[[pos[int(cid)] for cid in inner_eval.case_id]] = (
            np.log(inner_eval.actual.to_numpy(float)/forecast))
        print(f"Cross-fit duration residuals: {f+1}/4 folds", flush=True)
    if not np.isfinite(residuals).all():
        raise AssertionError("Incomplete out-of-fold residual cohort")
    np.save(path, residuals)
    return residuals


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
    """Small shared-policy search: cheap component prescreen, then exact full cohort."""
    path = folder/"shared.json"
    if path.exists():
        return np.asarray(json.loads(path.read_text())["theta"], float)

    rng = np.random.default_rng(20261020)
    flexible = [item for item in items if any(
        len(item[1].allowed[int(s)]) > 1 for s in item[1].surgeon)]
    candidates = flexible or items
    sample = rng.choice(len(candidates), size=min(150, len(candidates)), replace=False)
    screen = [candidates[int(i)] for i in sample]
    theta, history = np.zeros(H.shape[1]), []
    for round_id in range(rounds):
        step = 8./2**round_id
        basis = np.eye(H.shape[1])
        dirs = list(basis)
        for j in range(len(basis)):
            for k in range(j+1, len(basis)):
                dirs.extend(((basis[j]+basis[k])/np.sqrt(2),
                             (basis[j]-basis[k])/np.sqrt(2)))
        proposals = [beta+theta] + [beta+theta+sign*step*v
                                    for v in dirs for sign in (-1., 1.)]

        def assess(candidates, subset):
            state = dict(data=data, H=H, pools=pools, candidates=candidates,
                         seeds=seeds, seconds=seconds, round_id=round_id)
            scores = np.zeros((len(candidates), 2))
            failed = np.zeros(len(candidates), int)
            week_B = {j: {} for j in range(len(candidates))}
            for _, monday, _, values in parallel(shared_job, subset, state, cores):
                if values[0] is None:
                    raise RuntimeError("Uncertified shared-policy reference")
                for j, value in enumerate(values):
                    if value is None:
                        failed[j] += 1
                        continue
                    scores[j] += value
                    week_B[j][monday] = (week_B[j].get(monday, 0.)
                                          + value[1] - values[0][1])
            return scores, failed, week_B

        preview, failed, _ = assess(proposals, screen)
        possible = [j for j in range(1, len(proposals)) if not failed[j]]
        shortlist = [0] + sorted(possible, key=lambda j: preview[j, 0])[:5]
        candidates = [proposals[j] for j in shortlist]
        scores, incomplete, weekly = assess(candidates, items)
        eligible = [j for j in range(len(candidates)) if incomplete[j] == 0]
        best = min(eligible, key=lambda j: scores[j, 0])
        gain = scores[0, 1]-scores[best, 1]
        differences = np.asarray(list(weekly[best].values()))
        se = (differences.std(ddof=1)*np.sqrt(len(differences))
              if len(differences) > 1 else np.inf)
        accepted = bool(best and gain > 1.645*se)
        history.append(dict(round=round_id, proposals=len(proposals),
                            shortlisted=len(shortlist), chosen=shortlist[best],
                            bank_A_gain=float(scores[0, 0]-scores[best, 0]),
                            bank_B_gain=float(gain), bank_B_se=float(se),
                            accepted=accepted, failed_components=incomplete.tolist()))
        print("Shared search:", history[-1], flush=True)
        if accepted:
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
    # Every stratum uses the same case denominator and identical planner tie rules.
    result["by_site"] = {}
    result["by_size"] = {}
    size_label = lambda n: ("fixed/1" if n == 1 else "2-5" if n <= 5
                            else "6-10" if n <= 10 else "11-20" if n <= 20
                            else "21-34" if n <= 34 else "35+")
    for category, label_fn in (("by_site", lambda row: row["site"]),
                               ("by_size", lambda row: size_label(row["cases"]))):
        for label in sorted({label_fn(row) for row in rows}):
            subset = [row for row in rows if label_fn(row) == label]
            den = sum(row["cases"] for row in subset)
            result[category][label] = dict(cases=den, components=len(subset),
                differences={name: {str(seed): sum(
                    row["arms"][name]["realized"][str(seed)] -
                    row["arms"]["CASE"]["realized"][str(seed)]
                    for row in subset)/den for seed in seeds}
                    for name in arms})
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
    case_baseline(folder, fit, args.cores)
    held["case_w"] = fit["case_w"]
    fit_items = components(trainweeks, fitindex)
    eval_items = components(evalweeks, evalindex)
    H, mean, scale = fit_features(folder, fit_items, fit, names, args.seconds, args.cores)
    H_eval = test_features(eval_items, held, names, mean, scale, args.seconds, args.cores)
    fit_frame = cohort[cohort.case_id.isin(fit["case_id"])]
    errors = out_of_fold_residuals(folder, cohort, trainweeks, fit, args.cores)
    pools = residual_history(fit_items, fit, fit_frame, errors)
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



def final_data(workbook):
    """Refit on all pre-2013 data, freezing the model before opening 2013 outcomes."""
    from run_weekly_rich import read_cohort
    from surgery.rich_features import RichCorrectionEncoder
    from surgery.weekly_data import build_weeks, performed_events
    from pilot.run_pilot import add_context

    raw, cohort = read_cohort(workbook)
    train = cohort[cohort.split.eq("train")].reset_index(drop=True)
    test = cohort[cohort.split.eq("test")].reset_index(drop=True)
    if len(train) != 24506 or len(test) != 8774:
        raise AssertionError("2013 evaluation must have 8,774 cases after 24,506 fit cases")
    events = performed_events(raw)
    fitweeks = build_weeks(cohort, events[events.date < "2013-01-01"],
                           split="train", prior_dates=1)
    evalweeks = build_weeks(cohort, events, split="test", prior_dates=1)
    fitindex = {int(v): i for i, v in enumerate(train.case_id)}
    evalindex = {int(v): i for i, v in enumerate(test.case_id)}
    encoder = RichCorrectionEncoder(history_mode="leave_date_out")
    X = encoder.fit_transform(train)
    T = encoder.transform(test)
    names = list(encoder.names)
    X, extra, mu, sd = add_context(X, fitweeks, fitindex, variant="cohort")
    T, _, _, _ = add_context(T, evalweeks, evalindex, variant="cohort", scaler=(mu, sd))
    names.extend(extra)
    j = names.index("surgeon_week_booked_minutes")
    for name in ("hist_surgeon_mean_abs_error", "hist_procedure_mean_abs_error"):
        if name in names:
            k = names.index(name)
            X = np.column_stack([X, X[:, k]*X[:, j]])
            T = np.column_stack([T, T[:, k]*T[:, j]])
            names.append(name+"_x_surgeon_week_load")
    fitted = dict(X=X, booked=train.booked.to_numpy(float),
                  actual=train.actual.to_numpy(float), case_id=train.case_id.to_numpy(int))
    held = dict(X=T, booked=test.booked.to_numpy(float),
                actual=test.actual.to_numpy(float), case_id=test.case_id.to_numpy(int))
    return fitted, held, fitweeks, evalweeks, fitindex, evalindex, names


def run_final(args, cohort):
    """2013 is never used for fitting, tuning, scenario construction or method choice."""
    if not (args.out/"pooled.json").exists():
        raise RuntimeError("Complete both development folds before the frozen 2013 replay")
    folder = args.out/"final_2013"
    folder.mkdir(parents=True, exist_ok=True)
    fit, held, trainweeks, evalweeks, fitindex, evalindex, names = final_data(args.workbook)
    development_hash = hashlib.sha256((args.out/"pooled.json").read_bytes()).hexdigest()
    manifest = folder/"freeze.json"
    frozen = dict(development_sha256=development_hash, cases=8774,
                  seconds=args.seconds, budget=args.budget, shared_rounds=args.shared_rounds,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise RuntimeError("Final 2013 was already opened with other settings")
    write(manifest, frozen)
    case_baseline(folder, fit, args.cores)
    held["case_w"] = fit["case_w"]
    fit_items, eval_items = components(trainweeks, fitindex), components(evalweeks, evalindex)
    H, mu, sd = fit_features(folder, fit_items, fit, names, args.seconds, args.cores)
    T = test_features(eval_items, held, names, mu, sd, args.seconds, args.cores)
    fit_frame = cohort[cohort.case_id.isin(fit["case_id"])]
    errors = out_of_fold_residuals(folder, cohort, trainweeks, fit, args.cores)
    pools = residual_history(fit_items, fit, fit_frame, errors)
    beta = shared_control(fit, H)
    theta = fit_shared(folder, fit_items, fit, H, pools, beta,
                       args.seconds, args.cores, args.shared_rounds, (0, 1, 2))
    state = dict(data=held, H=T, pools=pools, beta=beta, theta=theta,
                 seeds=(0, 1, 2), seconds=args.seconds, budget=args.budget)
    rows = save_components(folder/"components", eval_items, state, args.cores)
    report(rows, (0, 1, 2), folder/"summary.json")
    print("Frozen 2013 replay:", len(held["booked"]), "cases", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/advice_comparison"))
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--budget", type=float, default=900.)
    parser.add_argument("--shared-rounds", type=int, default=10)
    parser.add_argument("--fold", choices=("0", "1", "both"), default="both")
    parser.add_argument("--final-2013", action="store_true", help="One frozen 2013 replay after both development folds")
    args = parser.parse_args()
    os.environ["SURGERY_RESEARCH_LINK"] = "inverse"
    if not 1 <= args.cores <= 15 or args.seconds <= 0 or args.budget <= 0:
        parser.error("cores must be 1..15 and time limits positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.final_2013 and args.fold != "both":
        parser.error("--final-2013 requires both completed development folds")
    _, cohort = read_cohort(args.workbook)
    if args.final_2013:
        run_final(args, cohort)
        return
    all_rows = []
    for fold in ((0, 1) if args.fold == "both" else (int(args.fold),)):
        all_rows.extend(run_fold(args, fold, cohort))
    if args.fold == "both":
        report(all_rows, (0, 1, 2), args.out/"pooled.json")


if __name__ == "__main__":
    main()
