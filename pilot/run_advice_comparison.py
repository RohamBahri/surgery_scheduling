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
                                       residual_history, reviewed, shared_job,
                                       sensitivity_job, LAM)
from surgery.anchored_core import key

SOURCE_DEPENDENCIES = (
    "pilot/run_advice_comparison.py", "surgery/advice_comparison.py",
    "surgery/anchored_weekly_solver.py", "surgery/anchored_core.py",
    "surgery/rich_features.py", "surgery/weekly_planner.py",
    "surgery/asdbs.py", "surgery/weekly_data.py",
    "surgery/data.py", "surgery/planner.py", "surgery/weekly_training.py",
    "pilot/run_asdbs.py", "pilot/run_pilot.py", "run_weekly_rich.py",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
            for p in SOURCE_DEPENDENCIES}


def parallel(fn, items, state, cores):
    if not items:
        return
    workers = min(cores, len(items))
    state = dict(state, threads=max(1, cores//workers))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
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
        d = reviewed(booked, d0 - booked + H @ beta)
        error = actual - d
        return (np.maximum(error, 0).sum()*1.75
                + np.maximum(-error, 0).sum()
                + LAM/2 * np.sum((d-d0)**2)) / len(booked)

    answer = minimize(loss, np.zeros(H.shape[1]), method="Powell",
                      options=dict(maxiter=150, xtol=1e-4, ftol=1e-7))
    if not np.isfinite(answer.fun):
        raise RuntimeError("Statistical matched control did not fit")
    return np.asarray(answer.x, float)



def collect_features(items, data, names, seconds, cores, checkpoint):
    """Resume costly exact CASE feature schedules at component granularity."""
    n = len(data["booked"])
    H, seen = np.zeros((n, 5), float), np.zeros(n, bool)
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as saved:
            H, seen = saved["H"].copy(), saved["seen"].copy()
        if H.shape != (n, 5) or seen.shape != (n,):
            raise RuntimeError("Stale feature-solve checkpoint")
    pending = []
    for item in items:
        rows = item[2]
        if seen[rows].all():
            continue
        if seen[rows].any():
            raise RuntimeError("Partial CASE component in feature checkpoint")
        pending.append(item)
    state = dict(data=data, names=names, seconds=seconds)
    try:
        for j, (_, rows, features) in enumerate(
                parallel(baseline_job, pending, state, cores)):
            if seen[rows].any():
                raise AssertionError("Repeated case in the decomposition")
            H[rows], seen[rows] = features, True
            if (j+1) % 50 == 0 and checkpoint is not None:
                np.savez_compressed(checkpoint, H=H, seen=seen)
                print(f"CASE features: {j+1}/{len(pending)} pending components", flush=True)
    except BaseException:
        if checkpoint is not None:
            np.savez_compressed(checkpoint, H=H, seen=seen)
        raise
    if not seen.all():
        raise AssertionError("Incomplete feature cohort")
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    return H


def fit_features(folder, items, data, names, seconds, cores):
    path = folder/"five_features.npz"
    if path.exists():
        with np.load(path) as old:
            return old["H"], old["mean"], old["scale"]
    H = collect_features(items, data, names, seconds, cores,
                         folder/"five_features_progress.npz")
    mu, sd = H.mean(axis=0), H.std(axis=0)
    sd = np.where(sd < 1e-8, 1., sd)
    H = (H-mu)/sd
    np.savez_compressed(path, H=H, mean=mu, scale=sd)
    return H, mu, sd


def test_features(items, data, names, mean, scale, seconds, cores, folder=None):
    path = folder/"held_features.npz" if folder is not None else None
    if path is not None and path.exists():
        with np.load(path) as saved:
            return saved["H"].copy()
    checkpoint = (folder/"held_features_progress.npz"
                  if folder is not None else None)
    raw = collect_features(items, data, names, seconds, cores, checkpoint)
    H = (raw-mean)/scale
    if path is not None:
        np.savez_compressed(path, H=H)
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
            for ck, monday, _, values in parallel(shared_job, subset, state, cores):
                if values[0] is None:
                    raise RuntimeError(f"Uncertified shared-policy reference: {ck}")
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
    try:
        for i, (ck, result) in enumerate(parallel(evaluated_job, pending, state, cores)):
            write(folder / (ck+".json"), result)
            if (i+1) % 100 == 0:
                print(f"Evaluation: {i+1}/{total} components", flush=True)
    except Exception as exc:
        write(folder/"failure.json", dict(error=str(exc), pending_at_start=total))
        raise
    return [json.loads((folder/(item[0]+".json")).read_text()) for item in items]



def report(rows, seeds, filename):
    """Paired calendar-Monday bootstrap; three hashes form ONE primary estimate."""
    from collections import defaultdict
    if not rows:
        raise ValueError("No evaluated components")
    arms = list(rows[0]["arms"])
    if any(set(row["arms"]) != set(arms) for row in rows):
        raise ValueError("Unmatched policy arms")
    n = sum(row["cases"] for row in rows)
    by_monday = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for row in rows:
        counts[row["monday"]] += row["cases"]
        for name, rec in row["arms"].items():
            for seed in seeds:
                by_monday[row["monday"]][name, seed] += rec["realized"][str(seed)]
    mondays = sorted(counts)
    den = np.asarray([counts[m] for m in mondays], float)
    if sum(den) != n:
        raise AssertionError("Unmatched calendar-week denominators")
    rng = np.random.default_rng(20261020)
    boot = rng.integers(len(mondays), size=(20000, len(mondays)))
    sampled_den = den[boot].sum(axis=1)

    def contrast(a, b):
        diff_seeds = np.asarray([
            [by_monday[m][a, seed] - by_monday[m][b, seed] for m in mondays]
            for seed in seeds
        ], float)
        per_monday = diff_seeds.mean(axis=0)
        effect = float(per_monday.sum() / n)
        samples = per_monday[boot].sum(axis=1) / sampled_den
        # Null-centered cluster bootstrap, two-sided; Holm only for P1/P2.
        p = float((1 + np.count_nonzero(
            np.abs(samples - effect) >= abs(effect))) / (len(samples) + 1))
        return dict(difference=effect,
                    ci95=np.quantile(samples, [.025, .975]).tolist(),
                    p_two_sided=p)

    results = dict(n_cases=n, n_components=len(rows), mondays=len(mondays),
                   seeds=list(seeds), bootstrap_resamples=20000,
                   arms={}, contrasts={})
    for name in arms:
        seed_effects = []
        for seed in seeds:
            delta = np.asarray([
                by_monday[m][name, seed] - by_monday[m]["CASE", seed]
                for m in mondays
            ], float)
            samples = delta[boot].sum(axis=1) / sampled_den
            seed_effects.append(dict(
                seed=seed, difference=float(delta.sum() / n),
                ci95=np.quantile(samples, [.025, .975]).tolist()))
        results["arms"][name] = dict(
            per_seed=seed_effects,
            mean_realized_cost_per_case=float(np.mean([
                sum(by_monday[m][name, seed] for m in mondays) / n
                for seed in seeds])),
            averaged_seeds=contrast(name, "CASE"),
            changed_cases=sum(row["arms"][name]["changed"] for row in rows),
            uncertified_fallback_components=sum(
                str(row["arms"][name].get("status", "")).startswith("UNCERTIFIED")
                for row in rows))

    pairs = {
        "PER_WEEK_REF_minus_CASE": ("PER_WEEK_REF", "CASE"),
        "PER_WEEK_minus_CASE": ("PER_WEEK", "CASE"),
        "SHARED_minus_CASE": ("SHARED", "CASE"),
        "ROBUST_minus_PER_WEEK": ("PER_WEEK_ROBUST", "PER_WEEK"),
        "SHARED_minus_STAT": ("SHARED", "STAT"),
        "PER_WEEK_minus_SHARED": ("PER_WEEK", "SHARED"),
        "PER_WEEK_minus_RISK": ("PER_WEEK", "RISK"),
    }
    for name, (a, b) in pairs.items():
        if a in arms and b in arms:
            results["contrasts"][name] = contrast(a, b)
    # Predeclared primary comparisons: P1 per-week vs CASE; P2 shared vs CASE.
    primary = {"P1": "PER_WEEK_minus_CASE", "P2": "SHARED_minus_CASE"}
    available = sorted(((key, results["contrasts"][val]["p_two_sided"])
                        for key, val in primary.items()
                        if val in results["contrasts"]), key=lambda a: a[1])
    adjusted = 0.
    for rank, (label, raw_p) in enumerate(available):
        adjusted = max(adjusted, min(1., raw_p*(len(available)-rank)))
        results["contrasts"][primary[label]]["holm_p"] = adjusted
        results["contrasts"][primary[label]]["primary"] = label

    def strata(subset):
        count = sum(r["cases"] for r in subset)
        return dict(cases=count, components=len(subset),
                    averaged_seed_differences={
                        name: float(np.mean([
                            sum(row["arms"][name]["realized"][str(seed)] -
                                row["arms"]["CASE"]["realized"][str(seed)]
                                for row in subset) / count
                            for seed in seeds]))
                        for name in arms})
    results["by_fold"] = {
        str(fold): strata([r for r in rows if r.get("fold") == fold])
        for fold in sorted({r["fold"] for r in rows if "fold" in r})
    }
    size_label = lambda x: ("fixed/1" if x == 1 else "2-5" if x <= 5
                            else "6-10" if x <= 10 else "11-20" if x <= 20
                            else "21-34" if x <= 34 else "35+")
    results["by_site"], results["by_size"] = {}, {}
    for category, label_fn in (("by_site", lambda r: r["site"]),
                               ("by_size", lambda r: size_label(r["cases"]))):
        for label in sorted({label_fn(r) for r in rows}):
            subset = [r for r in rows if label_fn(r) == label]
            results[category][label] = strata(subset)
    write(filename, results)
    return results

def run_fold(args, fold, cohort):
    folder = args.out / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    fit, held, trainweeks, evalweeks, fitindex, evalindex, names, spec = prepare(args.workbook, fold)
    if len(fit["booked"]) + len(held["booked"]) != 24506:
        raise AssertionError("Expected the entire 24,506-case development cohort")
    fingerprint = dict(workbook=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                       model_source=source_hashes(),
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
    H_eval = test_features(eval_items, held, names, mean, scale, args.seconds, args.cores,
                           folder=folder)
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
    for row in rows:
        row["fold"] = fold
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
    """Repeatable 2013 evaluation: no one-time seal, separate audited runs."""
    frozen = dict(workbook=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                  cases=8774, cores=args.cores,
                  seconds=args.seconds, budget=args.budget,
                  shared_rounds=args.shared_rounds, source=source_hashes())
    run_id = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]
    folder = args.out/"final_2013"/"runs"/run_id
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/"manifest.json", frozen)
    fit, held, trainweeks, evalweeks, fitindex, evalindex, names = final_data(args.workbook)
    case_baseline(folder, fit, args.cores)
    held["case_w"] = fit["case_w"]
    fit_items, eval_items = components(trainweeks, fitindex), components(evalweeks, evalindex)
    H, mu, sd = fit_features(folder, fit_items, fit, names, args.seconds, args.cores)
    T = test_features(eval_items, held, names, mu, sd, args.seconds, args.cores,
                      folder=folder)
    fit_frame = cohort[cohort.case_id.isin(fit["case_id"])]
    errors = out_of_fold_residuals(folder, cohort, trainweeks, fit, args.cores)
    pools = residual_history(fit_items, fit, fit_frame, errors)
    beta = shared_control(fit, H)
    theta = fit_shared(folder, fit_items, fit, H, pools, beta,
                       args.seconds, args.cores, args.shared_rounds, (0, 1, 2))
    state = dict(data=held, H=T, pools=pools, beta=beta, theta=theta,
                 seeds=(0, 1, 2), seconds=args.seconds, budget=args.budget)
    rows = save_components(folder/"components", eval_items, state, args.cores)
    results = report(rows, (0, 1, 2), folder/"summary.json")
    write(args.out/"final_2013"/"summary.json", results)
    write(args.out/"final_2013"/"latest.json", dict(run_id=run_id, manifest=frozen))
    print(f"2013 evaluation {run_id}: {len(held['booked'])} cases", flush=True)

def decision_date_lookup(cohort):
    """Map case IDs to ISO dates or None; never leak pandas NaN to workers."""
    date_strings = cohort.dtt.dt.strftime("%Y-%m-%d").tolist()
    return dict(zip(
        cohort.case_id.astype(int),
        (day if isinstance(day, str) else None for day in date_strings)))


def run_sensitivity(args, cohort, raw):
    """Independent replay of frozen advice; neither models nor displays are refit."""
    from surgery.weekly_data import build_weeks, performed_events

    events = performed_events(raw)
    events = events[events.date < "2013-01-01"]
    weeks = build_weeks(cohort, events, split="train", prior_dates=1)
    mondays = sorted({w.monday for w in weeks})
    if len(mondays) != 79:
        raise RuntimeError("Changed development Monday cohort")
    if args.final_2013:
        raise ValueError("Development sensitivity and 2013 evaluation are separate runs")
    response_sweep = ((.5, 15.), (.5, 30.), (.8, 15.), (.8, 30.))
    fingerprint = dict(workbook=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                       source=source_hashes(), response_sweep=response_sweep,
                       cutoff="decision_date strictly before Monday",
                       seeds=(0, 1, 2), seconds=args.seconds)
    run_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    root = args.out/"sensitivity"/"runs"/run_id
    root.mkdir(parents=True, exist_ok=True)
    write(root/"manifest.json", fingerprint)
    # pandas can turn missing object dates back into float NaN.
    decision_dates = decision_date_lookup(cohort)
    by_condition = {}
    for fold in (0, 1):
        fold_dir = args.out / f"fold_{fold}"
        manifest = fold_dir/"settings.json"
        if not manifest.exists():
            raise RuntimeError(f"Missing completed fold {fold}: run development first")
        old = json.loads(manifest.read_text())
        # Sensitivity replays SAVED displays; changing the sensitivity code
        # should not invalidate development models. Require identical workbook,
        # cohort reconstruction, and exact planner instead.
        structural = ("surgery/data.py", "surgery/weekly_data.py",
                      "surgery/weekly_planner.py",
                      "surgery/anchored_weekly_solver.py", "surgery/planner.py",
                      "run_weekly_rich.py")
        trained_sources = old.get("model_source", {})
        if (old.get("workbook") != fingerprint["workbook"]
                or any(trained_sources.get(p) != fingerprint["source"].get(p)
                       for p in structural)):
            raise RuntimeError(
                f"Fold {fold}: workbook, roster, or planner has changed; "
                "frozen-display sensitivity cannot safely be replayed")
        test_dates = {monday for i, monday in enumerate(mondays) if i % 2 == fold}
        items = [(key(c), c, None) for week in weeks if week.monday in test_dates
                 for c in week.components()]
        source_dir = fold_dir/"components"
        if any(not (source_dir/(ck+".json")).exists() for ck, _, _ in items):
            raise RuntimeError(f"Incomplete fold {fold} components")
        state = dict(components_dir=source_dir, decision_dates=decision_dates,
                     response_sweep=response_sweep, seconds=args.seconds,
                     seeds=(0, 1, 2))
        saved = root / f"fold_{fold}"
        saved.mkdir(parents=True, exist_ok=True)
        # Reuse completed components from a previous sensitivity attempt
        # when only the runner/sensitivity code changed (e.g. NaN date fix).
        # Keep the original run directory untouched for provenance.
        import shutil
        reused = 0
        for previous in (args.out/"sensitivity"/"runs").iterdir():
            if not previous.is_dir() or previous == root:
                continue
            manifest_path = previous/"manifest.json"
            if not manifest_path.exists():
                continue
            try:
                prev = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                continue
            if any(json.dumps(prev.get(k), sort_keys=True)
                   != json.dumps(fingerprint.get(k), sort_keys=True)
                   for k in ("workbook", "response_sweep", "cutoff", "seeds", "seconds")):
                continue
            prev_sources = prev.get("source", {})
            if any(prev_sources.get(p) != fingerprint["source"].get(p)
                   for p in structural):
                continue
            for ck, c, _ in items:
                target = saved/(ck+".json")
                previous_file = previous/f"fold_{fold}"/(ck+".json")
                if target.exists() or not previous_file.exists():
                    continue
                try:
                    record = json.loads(previous_file.read_text())
                    if (not isinstance(record, dict)
                            or any(value.get("cases") != c.n
                                   for value in record.values())):
                        continue
                except (OSError, ValueError, AttributeError):
                    continue
                shutil.copy2(previous_file, target)
                reused += 1
        if reused:
            print(f"Sensitivity fold {fold}: reused {reused} completed components",
                  flush=True)
        pending = [item for item in items
                   if not (saved/(item[0]+".json")).exists()]
        for j, (ck, scored) in enumerate(parallel(sensitivity_job, pending, state, args.cores)):
            write(saved/(ck+".json"), scored)
            if (j + 1) % 100 == 0:
                print(f"Sensitivity fold {fold}: {j+1}/{len(pending)}", flush=True)
        for ck, _, _ in items:
            for label, record in json.loads((saved/(ck+".json")).read_text()).items():
                record["fold"] = fold
                by_condition.setdefault(label, []).append(record)
    for label, rows in by_condition.items():
        output = report(rows, (0, 1, 2), root/(label+".json"))
        if label == "cutoff_week_start":
            output["cutoff_cases_known"] = sum(row["eligible_at_cutoff"] for row in rows)
            output["cutoff_cases_unknown_date"] = sum(row["unknown_decision_date"] for row in rows)
            output["cutoff_cases_after_or_on_monday"] = (
                output["n_cases"] - output["cutoff_cases_known"]
                - output["cutoff_cases_unknown_date"])
            write(root/(label+".json"), output)
    write(args.out/"sensitivity"/"latest.json",
          dict(run_id=run_id, conditions=list(by_condition), manifest=fingerprint))
    print(f"Sensitivity replay complete: {run_id}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/advice_comparison"))
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--budget", type=float, default=900.)
    parser.add_argument("--shared-rounds", type=int, default=10)
    parser.add_argument("--fold", choices=("0", "1", "both"), default="both")
    parser.add_argument("--final-2013", action="store_true", help="Repeatable 2013 evaluation with run-specific provenance")
    parser.add_argument("--sensitivity-only", action="store_true",
                        help="Rescore frozen development displays under alternative responses and Monday cutoff")
    args = parser.parse_args()
    os.environ["SURGERY_RESEARCH_LINK"] = "inverse"
    if not 1 <= args.cores <= 15 or args.seconds <= 0 or args.budget <= 0:
        parser.error("cores must be 1..15 and time limits positive")
    args.out.mkdir(parents=True, exist_ok=True)
    raw, cohort = read_cohort(args.workbook)
    if args.sensitivity_only:
        run_sensitivity(args, cohort, raw)
        return
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
