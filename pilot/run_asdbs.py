#!/usr/bin/env python3
"""ASDBS: 79-Monday pre-2013 offline folds; no 2013 outcomes in development."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pilot import run_pilot as core
from pilot.run_fold_pilot import fold_lasso
from surgery.asdbs import train, verify, write, components
from surgery.planner import display_and_duration


def prepare(workbook, fold):
    from run_weekly_rich import read_cohort
    from surgery.weekly_data import build_weeks, performed_events
    from surgery.rich_features import RichCorrectionEncoder

    raw, cohort = read_cohort(workbook)
    pre = cohort[cohort.date < "2013-01-01"].copy()
    if len(pre) != 24506:
        raise RuntimeError(f"Expected 24,506 pre-2013 cases; found {len(pre)}")
    events = performed_events(raw)
    events = events[events.date < "2013-01-01"].copy()
    weeks = build_weeks(cohort, events, split="train", prior_dates=1)
    mondays = sorted({w.monday for w in weeks})
    if len(mondays) != 79:
        raise RuntimeError(f"Expected 79 calendar Mondays, got {len(mondays)}")

    test_dates = {v for j, v in enumerate(mondays) if j % 2 == fold}
    fitting = [w for w in weeks if w.monday not in test_dates]
    evaluation = [w for w in weeks if w.monday in test_dates]
    fit_ids = {int(cid) for w in fitting for cid in w.case_ids}
    eval_ids = {int(cid) for w in evaluation for cid in w.case_ids}
    if fit_ids & eval_ids or len(fit_ids | eval_ids) != len(pre):
        raise RuntimeError("Fold case rosters are incomplete or overlap")
    fit = pre[pre.case_id.isin(fit_ids)].reset_index(drop=True)
    held = pre[pre.case_id.isin(eval_ids)].reset_index(drop=True)

    enc = RichCorrectionEncoder(history_mode="leave_date_out")
    X = enc.fit_transform(fit)
    H = enc.transform(held)
    names = list(enc.names)
    fit_idx = {int(cid): i for i, cid in enumerate(fit.case_id)}
    held_idx = {int(cid): i for i, cid in enumerate(held.case_id)}
    X, extra, mu, sd = core.add_context(X, fitting, fit_idx, variant="cohort")
    H, _, _, _ = core.add_context(H, evaluation, held_idx, variant="cohort",
                                   scaler=(mu, sd))
    names += list(extra)
    # Fixed, explicitly congestion-aware features in BOTH CASE and ASDBS.
    load = names.index("surgeon_week_booked_minutes")
    for dispersion in ("hist_surgeon_mean_abs_error", "hist_procedure_mean_abs_error"):
        if dispersion in names:
            j = names.index(dispersion)
            X = np.column_stack([X, X[:, j] * X[:, load]])
            H = np.column_stack([H, H[:, j] * H[:, load]])
            names.append(dispersion + "_x_surgeon_week_load")

    if not (np.isfinite(X).all() and np.isfinite(H).all() and
            np.allclose(X[:, 0], 1.) and X.shape[1] == H.shape[1]):
        raise RuntimeError("Invalid risk-augmented fold features")
    fit_data = dict(X=np.ascontiguousarray(X, dtype=float),
                    booked=fit.booked.to_numpy(float),
                    actual=fit.actual.to_numpy(float),
                    case_id=fit.case_id.to_numpy(int))
    eval_data = dict(X=np.ascontiguousarray(H, dtype=float),
                     booked=held.booked.to_numpy(float),
                     actual=held.actual.to_numpy(float),
                     case_id=held.case_id.to_numpy(int))
    spec = dict(fold=fold, n_fit=len(fit), n_eval=len(held),
                n_train_mondays=len(mondays)-len(test_dates),
                n_eval_mondays=len(test_dates),
                feature_names=names, history_mode="leave_date_out",
                response_link="inverse", alpha=.8, h=30.,
                lambda0=.001, scenarios_A=50, scenarios_B=50,
                rounds=10, acceptance_z=1.645)
    return (fit_data, eval_data, fitting, evaluation,
            fit_idx, held_idx, names, spec)


def get_case(folder, data):
    policy_file = folder / "case_policy.json"
    if policy_file.exists():
        result = json.loads(policy_file.read_text())
        if result.get("response_link") != "inverse":
            raise RuntimeError("CASE fit was not trained with the inverse link")
        w = np.asarray(result["w"], float)
        if len(w) != data["X"].shape[1]:
            raise RuntimeError("CASE policy feature dimension mismatch")
        return w
    args = argparse.Namespace(out=folder, fold_safe=True, alpha=.8, h=30.,
                              case_inner=650, inner=650,
                              lambda_grid=(.0005,.001,.002,.004,.008,.016,.032))
    fold_lasso(args, data, folder)
    core.case(args, data, dict(response_link="inverse",
                               method="ASDBS_risk_CASE", history="leave_date_out"))
    return np.asarray(json.loads(policy_file.read_text())["w"], float)


def score(folder, fitting, evaluation, fit_idx, eval_idx, train_data, eval_data,
          seconds, workers, spec):
    policy_file = folder / "asdbs_progress.json"
    if not policy_file.exists():
        raise RuntimeError("Fit ASDBS before scoring")
    state = json.loads(policy_file.read_text())
    if not state["complete"]:
        raise RuntimeError("ASDBS fitting is not complete")
    out = {}
    for name, w in (("CASE", np.asarray(state["case_policy"])),
                    ("ASDBS", np.asarray(state["policy"]))):
        out[name] = verify(folder / "heldout_deploy", name, evaluation, eval_idx,
                           eval_data, w, seconds, workers)
    from surgery.anchored_core import key
    rows = []
    for week in evaluation:
        for c in week.components():
            ck = key(c)
            if not out["CASE"][ck]["complete"] or not out["ASDBS"][ck]["complete"]:
                raise RuntimeError("Uncertified held-out schedule: " + ck)
            rows.append(dict(monday=c.monday, site=c.group, cases=c.n,
                             case_cost=out["CASE"][ck]["realized_cost"],
                             asdbs_cost=out["ASDBS"][ck]["realized_cost"]))
    N = len(eval_data["booked"])
    if sum(row["cases"] for row in rows) != N:
        raise RuntimeError("Incomplete held-out evaluation cohort")
    difference = sum(r["asdbs_cost"]-r["case_cost"] for r in rows)/N
    byweek = {}
    for row in rows:
        c, d = byweek.get(row["monday"], (0, 0.))
        byweek[row["monday"]] = (c+row["cases"], d+row["asdbs_cost"]-row["case_cost"])
    # Prespecified paired week bootstrap, evaluation only.
    rng = np.random.default_rng(41020)
    pairs = np.array(list(byweek.values()), dtype=float)
    indices = rng.integers(len(pairs), size=(2000,len(pairs)))
    sampled = pairs[indices]
    estimates = sampled[:, :, 1].sum(axis=1)/sampled[:, :, 0].sum(axis=1)
    report = dict(**spec, n_evaluated_cases=N, case_cost_per_case=
                  sum(x["case_cost"] for x in rows)/N,
                  asdbs_cost_per_case=sum(x["asdbs_cost"] for x in rows)/N,
                  asdbs_minus_case_per_case=difference,
                  calendar_week_bootstrap_95=np.quantile(estimates,[.025,.975]).tolist(),
                  n_weeks_better=sum(v[1]<0 for v in byweek.values()),
                  n_evaluation_weeks=len(byweek),
                  by_site_component=rows,
                  note="Development-fold evaluation; not the separate 2013 test")
    write(folder/"heldout_results.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ("feature_names","by_site_component")},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0,1), required=True)
    parser.add_argument("--stage", choices=("fit","score"), required=True)
    parser.add_argument("--workbook", type=Path,
                        default=Path("data/UHNOperating_RoomScheduling2011-2013.xlsx"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/asdbs_79week_v1"))
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.seconds <= 0 or not 1 <= args.workers <= 15:
        parser.error("Invalid solve budget")
    os.environ["SURGERY_RESEARCH_LINK"] = "inverse"
    os.environ["SURGERY_ANCHORED_HASH_SEED"] = "0"
    _, check = display_and_duration(np.array([10.,40.]),
                                     np.array([200.,200.]),.8,30.)
    if not np.allclose(check,[210.,224.]):
        raise RuntimeError("Inverse response link was not activated")
    folder = args.root/f"fold_{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)
    data, held, weeks, evalweeks, index, evalindex, names, spec = prepare(
        args.workbook,args.fold)
    identity = dict(**spec, workbook_sha256=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                    X_sha256=hashlib.sha256(data["X"].tobytes()).hexdigest())
    mark = folder/"identity.json"
    if mark.exists() and json.loads(mark.read_text()) != identity:
        raise RuntimeError("Changed cohort/features; use a different --root")
    if not mark.exists():
        write(mark, identity)
    if args.stage == "fit":
        wcase = get_case(folder, data)
        train(folder, weeks, index, data, names, wcase,
              seconds=args.seconds, workers=args.workers)
    else:
        score(folder, weeks, evalweeks,index,evalindex,data,held,
              args.seconds,args.workers,spec)


if __name__ == "__main__":
    main()
