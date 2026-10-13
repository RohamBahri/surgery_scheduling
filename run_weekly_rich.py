#!/usr/bin/env python3
"""Rich-feature weekly training and exact, training-period pilot.

Three stages: features -> fit -> pilot. The 2013 holdout is never read into
a fitted feature statistic or used for selection. The pilot evaluates
BOOKED and the fitted LASSO baseline on the SAME weekly instances; VF/GAP/SPO
are reported as certified losses, not misrepresented as separately trained
policies. Existing weekly scheduling/oracle models are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, build_cohort
from surgery.planner import display_and_duration
from surgery.rich_features import RichCorrectionEncoder, attach_preoperative_dates
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, solve_adversary, solve_week

# A source change invalidates previously built matrices and policy coefficients.
# Provenance of the verified rich-feature artifacts produced immediately before
# the paired-oracle resume correction (commit fed01bf). Scientific feature
# inputs have not changed; only pilot orchestration/source fingerprint did.
LEGACY_FEATURE_SOURCE_SHA256 = (
    "dcdc7f7cfaae9cb5fe83a8c8a39da987ea118c9c6a965eb75288f2f2968b0fb9"
)
SOURCE_FILES = ('run_weekly_rich.py', 'surgery/data.py',
                'surgery/rich_features.py', 'surgery/planner.py',
                'surgery/weekly_data.py', 'surgery/weekly_planner.py')


def source_fingerprint():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for filename in SOURCE_FILES:
        h.update(filename.encode())
        h.update((root / filename).read_bytes())
    return h.hexdigest()


def verify_bounds(regret, vf, gap, spo):
    if min(regret, vf - regret, gap - regret, spo - gap) < -0.01:
        raise AssertionError('Certified upper-bound ordering or nonnegative regret violated')


def choose_pilot_weeks(pairs, limit):
    # The initial three-week pilot must cover validation at ALL three sites.
    first = []
    for group in GROUPS:
        choices = sorted((p for p in pairs if p[1] == 'validation'
                          and p[0].group == group), key=lambda p: p[0].monday)
        if choices:
            first.append(choices[0])
    seen = {(w.group, w.monday, part) for w, part in first}
    rest = sorted((p for p in pairs
                   if (p[0].group, p[0].monday, p[1]) not in seen),
                  key=lambda p: (p[1] != 'validation', p[0].monday,
                                 GROUPS.index(p[0].group)))
    result = first + rest
    return result[:limit] if limit else result


def windows(cohort, start, train_weeks, validation_weeks):
    """Case rows and week identities, preserving all three physical-room groups."""
    eligible = cohort.loc[cohort.split.eq("train")].copy()
    eligible["monday"] = eligible.date - pd.to_timedelta(eligible.date.dt.weekday, unit="D")
    train_keys, validation_keys = set(), set()
    for group in GROUPS:
        dates = sorted(eligible.loc[eligible.group.eq(group), "monday"].unique())
        picked = dates[start:start + train_weeks + validation_weeks]
        if len(picked) != train_weeks + validation_weeks:
            raise ValueError(f"Insufficient training-period weeks in {group}")
        keys = [(group, pd.Timestamp(dt).date().isoformat()) for dt in picked]
        train_keys.update(keys[:train_weeks])
        validation_keys.update(keys[train_weeks:])
    keys = list(zip(eligible.group, eligible.monday.dt.strftime("%Y-%m-%d")))
    training = eligible.loc[[key in train_keys for key in keys]].reset_index(drop=True)
    validation = eligible.loc[[key in validation_keys for key in keys]].reset_index(drop=True)
    if training.empty or validation.empty or training.date.max() >= validation.date.min():
        raise ValueError("Require chronological, disjoint training/validation dates across groups")
    if set(training.case_id) & set(validation.case_id):
        raise AssertionError("Train/validation overlap")
    return training, validation, train_keys, validation_keys


def read_cohort(workbook):
    raw = pd.read_excel(workbook, usecols=list(dict.fromkeys([*COLUMNS, "Consult_Date"])))
    cohort, _, _, _ = build_cohort(raw)
    return raw, attach_preoperative_dates(cohort, raw)


def save_features(root, workbook, start, train_weeks, validation_weeks):
    raw, cohort = read_cohort(workbook)
    train, valid, train_keys, validation_keys = windows(
        cohort, start, train_weeks, validation_weeks)
    encoder = RichCorrectionEncoder()
    matrices = {"train": encoder.fit_transform(train), "validation": encoder.transform(valid)}
    matrix_hashes = {}
    for part, frame in (("train", train), ("validation", valid)):
        x = matrices[part]
        if x.shape[0] != len(frame) or not np.isfinite(x).all():
            raise AssertionError(f"Invalid {part} matrix")
        np.savez_compressed(
            root / f"{part}.npz", X=x.astype(np.float32),
            case_id=frame.case_id.to_numpy(np.int64),
            booked=frame.booked.to_numpy(float),
            actual=frame.actual.to_numpy(float),
            group=frame.group.to_numpy(str),
            date=frame.date.dt.strftime("%Y-%m-%d").to_numpy(str),
            monday=frame.monday.dt.strftime("%Y-%m-%d").to_numpy(str))
        matrix_hashes[part] = hashlib.sha256((root / f"{part}.npz").read_bytes()).hexdigest()
    specification = {
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "code_sha256": source_fingerprint(),
        "matrix_sha256": matrix_hashes,
        "start_week_index": start, "train_weeks": train_weeks,
        "validation_weeks": validation_weeks,
        "train_cases": len(train), "validation_cases": len(valid),
        "training_keys": sorted(train_keys),
        "validation_keys": sorted(validation_keys),
        "feature_metadata": asdict(encoder.metadata()),
        "feature_names": list(encoder.names),
        "split": "2011-2012 training period only; 2013 final test outcomes excluded from fitting/scoring",
        "anchor": "booked + X @ w; booked has fixed coefficient one",
        "history": "original leave-one-date-out encoder unchanged"
    }
    (root / "manifest.json").write_text(json.dumps(specification, indent=2) + "\n")
    print(f"Features: train {matrices['train'].shape}, validation {matrices['validation'].shape}", flush=True)


def load_matrices(root):
    meta = json.loads((root / "manifest.json").read_text())
    if meta.get("code_sha256") != source_fingerprint():
        raise ValueError("Scientific code changed since matrices were built; rebuild to a NEW output directory")
    frames = {}
    for part in ("train", "validation"):
        if hashlib.sha256((root / f"{part}.npz").read_bytes()).hexdigest() != meta["matrix_sha256"][part]:
            raise ValueError(f"{part} feature matrix content has changed")
        with np.load(root / f"{part}.npz") as z:
            frames[part] = {name: z[name].copy() for name in z.files}
        if frames[part]["X"].shape[1] != len(meta["feature_names"]):
            raise ValueError("Feature names/matrix mismatch")
    return meta, frames


def error_metrics(actual, prediction):
    err = np.asarray(actual) - np.asarray(prediction)
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias_actual_minus_pred": float(err.mean())}


def fit_lasso(root, penalties):
    """One regularized residual policy; BIC selects L1 penalty on TRAIN only."""
    from sklearn.linear_model import Lasso

    meta, frames = load_matrices(root)
    tr = frames["train"]
    y = tr["actual"] - tr["booked"]
    X = tr["X"].astype(float)
    if meta["feature_names"][0] != "intercept" or "booked" in meta["feature_names"]:
        raise AssertionError("Fixed booking anchor or intercept missing")
    if np.any(np.std(X[:, 1:], axis=0) < 1e-10):
        raise ValueError("Constant penalized feature")
    scores = []
    for alpha in penalties:
        model = Lasso(alpha=alpha, fit_intercept=True, tol=1e-5,
                      max_iter=20000).fit(X[:, 1:], y)
        mse = float(np.mean((y - model.predict(X[:, 1:])) ** 2))
        df = 1 + int(np.count_nonzero(np.abs(model.coef_) > 1e-6))
        bic = len(y) * np.log(max(mse, 1e-12)) + df * np.log(len(y))
        scores.append((bic, alpha, model, df))
    _, selected, model, df = min(scores, key=lambda a: (a[0], a[1]))
    weights = np.r_[model.intercept_, model.coef_]
    results = {"method": "LASSO_MSE", "alpha": selected, "active_coefficients": df - 1,
               "weights": weights.tolist(), "features": meta["feature_names"],
               "selected_using": "training BIC only; no validation/test labels"}
    for part, data in frames.items():
        book = data["booked"]
        actual = data["actual"]
        correction = data["X"].astype(float) @ weights
        _, post = display_and_duration(correction, book, 0.8, 30.)
        results[part] = {
            "booked": error_metrics(actual, book),
            "raw": error_metrics(actual, book + correction),
            "post_alpha_0.8_h_30": error_metrics(actual, post),
        }
    (root / "lasso_policy.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("alpha", "active_coefficients", "train", "validation")},
                     indent=2), flush=True)


def adopt_legacy_artifacts(root, workbook):
    """One-time, fingerprint-checked reuse of pre-oracle-pair artifacts.

    This migration does NOT refit or change the matrices. It updates the
    source-version guard only for the reviewed runner-only change and archives
    old SPO pilot rows, which lack a hindsight assignment fingerprint.
    """
    manifest_path = root / "manifest.json"
    meta = json.loads(manifest_path.read_text())
    old_fingerprint = meta.get("code_sha256")
    if old_fingerprint != LEGACY_FEATURE_SOURCE_SHA256:
        raise ValueError(
            "Only the verified pre-paired-runner artifact version can be adopted; "
            "never bypass code fingerprints for a different feature builder")
    if hashlib.sha256(workbook.read_bytes()).hexdigest() != meta.get("workbook_sha256"):
        raise ValueError("Workbook differs from the artifacts")
    if meta.get("feature_metadata", {}).get("unit_booking_anchor") is not True:
        raise ValueError("Expected fixed booked-duration anchor")
    if meta.get("feature_metadata", {}).get("n_features") != 74:
        raise ValueError("Unexpected historical feature dimension")
    if meta.get("train_cases") != 18705 or meta.get("validation_cases") != 5779:
        raise ValueError("Unexpected development cohort")
    if any(str(name).startswith("calendar_") for name in meta["feature_names"]):
        raise ValueError("Legacy artifact contains removed calendar fields")
    for part in ("train", "validation"):
        path = root / f"{part}.npz"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != meta["matrix_sha256"][part]:
            raise ValueError(f"{part} matrix does not match the original manifest")
        with np.load(path) as z:
            if (z["X"].shape != (meta[part + "_cases"], len(meta["feature_names"]))
                    or not np.isfinite(z["X"]).all()):
                raise ValueError(f"Invalid {part} matrix")
    policy_path = root / "lasso_policy.json"
    policy = json.loads(policy_path.read_text())
    if (policy.get("method") != "LASSO_MSE"
            or policy.get("features") != meta["feature_names"]
            or len(policy.get("weights", [])) != len(meta["feature_names"])):
        raise ValueError("Policy is incompatible with the original feature matrix")
    settings_path = root / "pilot_settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        if (settings.get("workbook_sha256") != meta["workbook_sha256"]
                or settings.get("policy_sha256")
                != hashlib.sha256(policy_path.read_bytes()).hexdigest()):
            raise ValueError("Existing pilot does not match this model/workbook")
    old_backup = root / "manifest_before_oracle_pair_fix.json"
    old_pilot = root / "pilot.csv"
    archived_pilot = root / "pilot_before_oracle_pair_fix.csv"
    if old_backup.exists() or archived_pilot.exists():
        raise ValueError("Migration already attempted; inspect artifacts manually")
    if old_pilot.exists():
        old_pilot.rename(archived_pilot)
    old_backup.write_bytes(manifest_path.read_bytes())
    meta["previous_code_sha256"] = old_fingerprint
    meta["runner_only_migration"] = "fed01bf to paired-oracle pilot runner"
    meta["code_sha256"] = source_fingerprint()
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(meta, indent=2) + "\n")
    temporary.replace(manifest_path)
    print("Verified and reused feature matrices and fitted LASSO policy.", flush=True)
    if archived_pilot.exists():
        print("Prior three-week pilot archived as", archived_pilot, flush=True)
        print("Recompute oracle-matched BOOKED/LASSO pairs with --stage pilot.", flush=True)


def oracle_assignment_sha256(components, oracles):
    """Identify the actual hindsight schedule shared by a week's policies.

    Hash the physical room-days rather than local slot indices, so the
    fingerprint describes the true schedule independent of component order.
    """
    if len(components) != len(oracles) or any(
            not result.complete or result.assignment is None for result in oracles):
        raise ValueError("Only certified hindsight assignments can be fingerprinted")
    assignments = sorted(
        (int(case_id), int(component.slots[int(slot)][0]),
         str(component.slots[int(slot)][1]))
        for component, result in zip(components, oracles)
        for case_id, slot in zip(component.case_ids, result.assignment)
    )
    return hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest()


def paired_week_is_certified(rows, part, group, monday):
    """Only reuse a complete BOOKED/LASSO pair with the SAME hindsight plan."""
    booked = rows.get((part, group, monday, "BOOKED"))
    lasso = rows.get((part, group, monday, "LASSO"))
    if not booked or not lasso:
        return False
    reference = booked.get("oracle_assignment_sha256")
    return (booked.get("complete") == True and lasso.get("complete") == True
            and isinstance(reference, str) and len(reference) == 64
            and lasso.get("oracle_assignment_sha256") == reference)


def score_pilot_week(task):
    """Score both policies against one certified hindsight oracle in one worker."""
    week, period, case_reviews, alpha, h, gamma, seconds, threads = task
    components = week.components()
    hindsight = [solve_week(c, c.actual, seconds=seconds,
                            threads=threads, tie=False) for c in components]
    oracle_ready = all(o.complete for o in hindsight)
    oracle_hash = oracle_assignment_sha256(components, hindsight) if oracle_ready else None
    rows = []
    for method in ("BOOKED", "LASSO"):
        totals = {key: 0. for key in ("cases", "case_error", "vf", "gap", "spo",
                                     "regret", "realized_cost", "oracle_cost")}
        complete = oracle_ready
        for c, o in zip(components, hindsight):
            if not complete:
                break
            reviewed = (c.booked if method == "BOOKED" else
                        np.array([case_reviews[int(i)] for i in c.case_ids], float))
            plan = solve_week(c, reviewed, seconds=seconds, threads=threads)
            adversary = solve_adversary(c, reviewed, gamma=gamma,
                                        seconds=seconds, threads=threads)
            if not plan.complete or not adversary.complete:
                complete = False
                break
            realized = evaluate(c, plan.assignment, c.actual)["cost"]
            predicted_oracle = evaluate(c, o.assignment, reviewed)["cost"]
            error = c.actual - reviewed
            theta = float(np.sum(1.75 * np.maximum(error, 0.) +
                                 np.maximum(-error, 0.)))
            regret = realized - o.value
            vf = plan.value + theta - o.value
            gap = gamma * plan.value + adversary.value - o.value
            spo = gamma * predicted_oracle + adversary.value - o.value
            verify_bounds(regret, vf, gap, spo)
            for key, value in (
                ("cases", c.n), ("case_error", np.abs(error).sum()),
                ("vf", vf), ("gap", gap), ("spo", spo), ("regret", regret),
                ("realized_cost", realized), ("oracle_cost", o.value)
            ):
                totals[key] += float(value)
        rows.append({
            "group": week.group, "monday": week.monday, "period": period,
            "method": method, "complete": complete,
            "oracle_assignment_sha256": oracle_hash,
            **(totals if complete else {key: np.nan for key in totals})
        })
    return rows


def pilot(root, workbook, alpha, h, gamma, seconds, threads, limit,
          workers=1, period="validation"):
    """Certified weekly BOOKED versus LASSO scores, parallel across weeks."""
    if workers * threads > 15 or workers < 1 or threads < 1:
        raise ValueError("CPU budget requires 1 <= workers*threads <= 15")
    meta, frames = load_matrices(root)
    policy_path = root / "lasso_policy.json"
    policy = json.loads(policy_path.read_text())
    if policy["features"] != meta["feature_names"]:
        raise ValueError("Policy features mismatch")
    if hashlib.sha256(workbook.read_bytes()).hexdigest() != meta["workbook_sha256"]:
        raise ValueError("Workbook checksum mismatch")
    weights = np.asarray(policy["weights"], float)
    if not np.isfinite(weights).all():
        raise ValueError("Non-finite policy coefficients")
    reviewed_by_case = {}
    for part in ("train", "validation"):
        data = frames[part]
        _, post = display_and_duration(data["X"].astype(float) @ weights,
                                        data["booked"], alpha, h)
        reviewed_by_case.update((int(case), float(duration))
                                for case, duration in zip(data["case_id"], post))
    raw, cohort = read_cohort(workbook)
    weeks = build_weeks(cohort, performed_events(raw), split="train", prior_dates=1)
    period_by_key = {
        (part, group, monday)
        for part, field in (("train", "training_keys"),
                            ("validation", "validation_keys"))
        for group, monday in meta[field]
    }
    selected = [(w, part) for w in weeks
                for part in ("train", "validation")
                if (part, w.group, w.monday) in period_by_key
                and (period == "all" or part == period)]
    selected = choose_pilot_weeks(selected, limit)
    settings = {
        "workbook_sha256": meta["workbook_sha256"],
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "alpha": alpha, "h": h, "gamma": gamma, "prior_dates": 1,
    }
    identity_path = root / "pilot_settings.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != settings:
            raise ValueError("Pilot settings or fitted policy changed")
    else:
        identity_path.write_text(json.dumps(settings, indent=2) + "\n")
    rows = {}
    output_csv = root / "pilot.csv"
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        for record in previous.to_dict("records"):
            if record.get("complete") == True:
                key = (str(record["period"]), str(record["group"]),
                       str(record["monday"]), str(record["method"]))
                if key in rows:
                    raise ValueError("Duplicate certified pilot cell")
                rows[key] = record
    tasks = []
    for week, part in selected:
        if paired_week_is_certified(rows, part, week.group, week.monday):
            continue
        for method in ("BOOKED", "LASSO"):
            rows.pop((part, week.group, week.monday, method), None)
        case_reviews = {int(i): reviewed_by_case[int(i)] for i in week.case_ids}
        tasks.append((week, part, case_reviews, alpha, h, gamma, seconds, threads))

    def save(scored):
        for row in scored:
            key = (str(row["period"]), str(row["group"]),
                   str(row["monday"]), str(row["method"]))
            rows[key] = row
        temp = output_csv.with_suffix(".tmp")
        pd.DataFrame([rows[k] for k in sorted(rows)]).to_csv(temp, index=False)
        temp.replace(output_csv)

    print(f"{len(tasks)} unscored group-weeks; {workers} workers x "
          f"{threads} Gurobi threads = at most {workers * threads} CPU cores", flush=True)
    if workers == 1:
        for task in tasks:
            scored = score_pilot_week(task)
            save(scored)
            print(f"{scored[0]['period']} {scored[0]['group']} "
                  f"{scored[0]['monday']}: scored pair", flush=True)
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(score_pilot_week, task): task[:2] for task in tasks}
            for future in as_completed(futures):
                scored = future.result()
                save(scored)
                print(f"{scored[0]['period']} {scored[0]['group']} "
                      f"{scored[0]['monday']}: scored pair", flush=True)
    if any(not rows[(part, w.group, w.monday, method)]["complete"]
           for w, part in selected for method in ("BOOKED", "LASSO")):
        raise RuntimeError("Some group-weeks are unfinished; rerun with more time")
    print("Exact scores saved to", output_csv)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--stage", choices=("features", "fit", "pilot", "adopt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-week-index", type=int, default=0)
    parser.add_argument("--train-weeks", type=int, default=60)
    parser.add_argument("--validation-weeks", type=int, default=18)
    parser.add_argument("--penalties", default="0.1,0.2,0.5,1,2,5,10")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--h", type=float, default=30.)
    parser.add_argument("--gamma", type=float, default=2.)
    parser.add_argument("--seconds", type=float, default=120.)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--pilot-period", choices=("validation", "all"), default="validation")
    parser.add_argument("--limit-weeks", type=int, default=3)
    args = parser.parse_args()
    if (args.start_week_index < 0 or args.train_weeks < 1 or args.validation_weeks < 1
            or args.seconds <= 0 or not 1 <= args.threads <= 15 or args.gamma < 1
            or not 0 < args.alpha <= 1 or args.h <= 0 or args.limit_weeks < 0
            or args.workers < 1 or args.workers * args.threads > 15):
        parser.error("Invalid parameter")
    if args.stage == "features":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Feature output must be a new empty directory")
        args.output.mkdir(parents=True, exist_ok=True)
        save_features(args.output, args.workbook, args.start_week_index,
                      args.train_weeks, args.validation_weeks)
    elif args.stage == "adopt":
        adopt_legacy_artifacts(args.output, args.workbook)
    elif args.stage == "fit":
        if (args.output / "lasso_policy.json").exists() or (args.output / "pilot_settings.json").exists():
            parser.error("Fitted policy/pilot already exists; refuse to overwrite it")
        penalties = [float(x) for x in args.penalties.split(",")]
        if not penalties or not all(np.isfinite(a) and a > 0 for a in penalties):
            parser.error("Penalties must be finite positive values")
        fit_lasso(args.output, penalties)
    else:
        pilot(args.output, args.workbook, args.alpha, args.h, args.gamma,
              args.seconds, args.threads, args.limit_weeks,
              workers=args.workers, period=args.pilot_period)


if __name__ == "__main__":
    main()
