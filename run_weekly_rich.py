#!/usr/bin/env python3
"""Rich-feature weekly training and exact, training-period pilot.

Three stages: features -> fit -> pilot. The 2013 holdout is never read into
a fitted feature statistic or used for selection. The pilot evaluates
BOOKED and the fitted LASSO baseline on the SAME weekly instances; VF/GAP/SPO
are reported as certified losses, not misrepresented as separately trained
policies. Existing weekly scheduling/oracle models are unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from surgery.data import COLUMNS, GROUPS, build_cohort
from surgery.planner import display_and_duration
from surgery.rich_features import RichCorrectionEncoder, attach_preoperative_dates
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import evaluate, solve_adversary, solve_week


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
    specification = {
        "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
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
    frames = {}
    for part in ("train", "validation"):
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


def pilot(root, workbook, alpha, h, gamma, seconds, threads, limit):
    """Exact certified per-week loss scores for BOOKED and CASE; no model fitting."""
    meta, frames = load_matrices(root)
    policy = json.loads((root / "lasso_policy.json").read_text())
    if policy["features"] != meta["feature_names"]:
        raise ValueError("Policy features mismatch")
    if hashlib.sha256(workbook.read_bytes()).hexdigest() != meta["workbook_sha256"]:
        raise ValueError("Workbook checksum mismatch")
    weights = np.asarray(policy["weights"], float)
    if not np.isfinite(weights).all():
        raise ValueError("Non-finite policy coefficients")
    allowed = {}
    for part in ("train", "validation"):
        data = frames[part]
        _, post = display_and_duration(data["X"].astype(float) @ weights,
                                        data["booked"], alpha, h)
        for case, duration in zip(data["case_id"], post):
            allowed[int(case)] = float(duration)
    raw, cohort = read_cohort(workbook)
    weeks = build_weeks(cohort, performed_events(raw), split="train", prior_dates=1)
    period_by_key = {
        (part, group, monday)
        for part, field in (("train", "training_keys"), ("validation", "validation_keys"))
        for group, monday in meta[field]
    }
    selected = [(w, part) for w in weeks
                for part in ("train", "validation")
                if (part, w.group, w.monday) in period_by_key]
    selected.sort(key=lambda pair: (pair[1], pair[0].group, pair[0].monday))
    if limit:
        selected = selected[:limit]
    rows = []
    for week, part in selected:
        parts = week.components()
        oracle = [solve_week(c, c.actual, seconds=seconds, threads=threads, tie=False)
                  for c in parts]
        for method in ("BOOKED", "LASSO"):
            aggregate = {key: 0. for key in
                         ("cases", "case_error", "vf", "gap", "spo",
                          "regret", "realized_cost", "oracle_cost")}
            completed = all(o.complete for o in oracle)
            for component, hindsight in zip(parts, oracle):
                if not completed:
                    break
                if method == "BOOKED":
                    reviewed = component.booked
                else:
                    reviewed = np.array([allowed[int(i)] for i in component.case_ids])
                plan = solve_week(component, reviewed, seconds=seconds, threads=threads)
                adversary = solve_adversary(component, reviewed, gamma=gamma,
                                            seconds=seconds, threads=threads)
                if not plan.complete or not adversary.complete:
                    completed = False
                    break
                realized = evaluate(component, plan.assignment, component.actual)["cost"]
                pred_oracle = evaluate(component, hindsight.assignment, reviewed)["cost"]
                error = component.actual - reviewed
                theta = float((1.75 * np.maximum(error, 0.) +
                               np.maximum(-error, 0.)).sum())
                regret = realized - hindsight.value
                vf = plan.value + theta - hindsight.value
                gap = gamma * plan.value + adversary.value - hindsight.value
                spo = gamma * pred_oracle + adversary.value - hindsight.value
                if min(vf - regret, gap - regret, spo - gap) < -0.01:
                    raise AssertionError("Certified upper-bound ordering violated")
                for key, val in (("cases", component.n), ("case_error", np.abs(error).sum()),
                                 ("vf", vf), ("gap", gap), ("spo", spo),
                                 ("regret", regret), ("realized_cost", realized),
                                 ("oracle_cost", hindsight.value)):
                    aggregate[key] += float(val)
            rows.append({"group": week.group, "monday": week.monday,
                         "period": part, "method": method,
                         "complete": completed, **(aggregate if completed else
                                                  {k: np.nan for k in aggregate})})
        pd.DataFrame(rows).to_csv(root / "pilot.csv", index=False)
        print(f"{part} {week.group} {week.monday}: evaluated", flush=True)
    if not all(row["complete"] for row in rows):
        raise RuntimeError("Uncertified weekly jobs; inspect pilot.csv and increase time budget")
    print("Exact scores saved to", root / "pilot.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--stage", choices=("features", "fit", "pilot"), required=True)
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
    parser.add_argument("--limit-weeks", type=int, default=2)
    args = parser.parse_args()
    if (args.start_week_index < 0 or args.train_weeks < 1 or args.validation_weeks < 1
            or args.seconds <= 0 or not 1 <= args.threads <= 15 or args.gamma < 1
            or not 0 < args.alpha <= 1 or args.h <= 0 or args.limit_weeks < 0):
        parser.error("Invalid parameter")
    if args.stage == "features":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Feature output must be a new empty directory")
        args.output.mkdir(parents=True, exist_ok=True)
        save_features(args.output, args.workbook, args.start_week_index,
                      args.train_weeks, args.validation_weeks)
    elif args.stage == "fit":
        if (args.output / "case_policy.json").exists():
            parser.error("Policy already exists; refuse to silently overwrite it")
        penalties = [float(x) for x in args.penalties.split(",")]
        if not penalties or not all(np.isfinite(a) and a > 0 for a in penalties):
            parser.error("Penalties must be finite positive values")
        fit_lasso(args.output, penalties)
    else:
        if (args.output / "pilot.csv").exists():
            parser.error("Pilot already exists; use a new artifact directory")
        pilot(args.output, args.workbook, args.alpha, args.h, args.gamma,
              args.seconds, args.threads, args.limit_weeks)


if __name__ == "__main__":
    main()
