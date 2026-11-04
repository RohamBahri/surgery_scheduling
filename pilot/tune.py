"""Training-only regularization screening for the one UHN experiment.

A shortened decision-aware fit is used only for selecting L1 penalties;
final experiment policies are independently refitted with full budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot import experiment, policy_runner, policy_training
from pilot.prepare_data import prepare
from surgery.learning import DCObjective, ResponseDC, pdca


ANCHORS = [("saturating", .5, 30.), ("saturating", .8, 60.)]


def tuning_features(context):
    train_structural, scaling, _ = policy_training.five_features(
        context, context["train"], None, "train")
    validation_structural, _, _ = policy_training.five_features(
        context, context["eval"], None, "eval", scaler=scaling)
    return (np.column_stack([context["fit"]["X"], train_structural]),
            np.column_stack([context["held"]["X"], validation_structural]))


def mean_validation_cost(context, predictions, cache, label, seconds):
    from surgery.weekly_planner import evaluate
    plans = cache.solve(context["eval"], predictions, label, tie=True)
    if not all(plan["complete"] for plan in plans.values()):
        return None
    total = sum(evaluate(component, plans[key]["assignment"], component.actual)["cost"]
                for key, component, positions in context["eval"])
    weeks = len({comp.monday for _, comp, _ in context["eval"]})
    return float(total / weeks)


def tune(workbook: Path, output: Path, lambdas, *, cores=15, seconds=300):
    output.mkdir(parents=True, exist_ok=True)
    context_file = output / "context.pkl"
    if not context_file.exists():
        context = prepare(workbook, context_file, "2012-07-01")
    else:
        import pickle
        with context_file.open("rb") as stream:
            context = pickle.load(stream)
        if context["metadata"]["workbook_sha256"] != hashlib.sha256(
                workbook.read_bytes()).hexdigest():
            raise ValueError("Validation context belongs to a different workbook")
    policy_runner.configure_model(argparse.Namespace(turnover=0.))
    fit_design, validation_design = tuning_features(context)
    booked = np.asarray(context["fit"]["booked"], float)
    actual = np.asarray(context["fit"]["actual"], float)
    validation_booked = np.asarray(context["held"]["booked"], float)
    initialization_args = experiment.parse_args([
        "--context", str(context_file), "--out", str(output),
        "--cores", str(cores), "--plan-seconds", str(seconds)])
    initialization = experiment.fit_initializer(context, initialization_args)
    start = np.r_[initialization, np.zeros(fit_design.shape[1] -
                                           len(initialization))]
    cache = policy_training.PlannerCache(output / "validation_plans", cores, seconds)
    base = mean_validation_cost(
        context, validation_booked, cache, "Booked", seconds)
    if base is None:
        raise RuntimeError("Booked validation plans are not all certified")
    records = []

    for method in ("RESPONSE_AWARE", "DECISION_AWARE"):
        for lam in lambdas:
            for scenario in ANCHORS:
                link, alpha, threshold = scenario
                scenario_name = experiment.scenario_id(*scenario)
                folder = output / "candidates" / method / f"lambda_{lam:g}" / scenario_name
                folder.mkdir(parents=True, exist_ok=True)
                response_fit = ResponseDC(booked, alpha, threshold, 180., 1., link=link)
                response_validation = ResponseDC(
                    validation_booked, alpha, threshold, 180., 1., link=link)
                if method == "RESPONSE_AWARE":
                    candidate_file = folder / "coefficients.json"
                    if candidate_file.exists():
                        weights = np.asarray(json.loads(
                            candidate_file.read_text())["weights"], float)
                    else:
                        objective = DCObjective(
                            booked, fit_design, np.zeros(len(booked)),
                            response_fit, mode="CASE", actual=actual,
                            components=len(context["train"]), lam=lam,
                            penalize=np.r_[0., np.ones(fit_design.shape[1] - 1)],
                            accuracy_mu=2.)
                        fit = pdca(objective, start, radius=100.,
                                   outer=50, backend="gurobi")
                        weights = np.asarray(fit["weights"], float)
                        experiment.save(candidate_file, dict(weights=weights.tolist(),
                                                              status=fit["status"]))
                else:
                    arguments = experiment.policy_args(
                        initialization_args, scenario, 2., folder, "train")
                    option = arguments.index("--lambda-vf")
                    arguments[option + 1] = str(lam)
                    option = arguments.index("--vf-outers")
                    arguments[option + 1] = "6"
                    option = arguments.index("--vf-pdca-steps")
                    arguments[option + 1] = "20"
                    policy_runner.main(arguments)
                    policy = json.loads((folder / "selected_policy.json").read_text())
                    weights = np.asarray(policy["decision_weights"], float)
                reviewed = response_validation.reviewed(validation_design @ weights)
                value = mean_validation_cost(
                    context, reviewed, cache,
                    f"{method}_lambda_{lam:g}_{scenario_name}", seconds)
                records.append(dict(method=method, lambda_value=float(lam),
                    scenario=scenario_name, validation_weekly_cost=value,
                    saving_vs_booked=None if value is None else base-value))
                experiment.save(output / "tuning_progress.json", records)

    selected = {}
    for method in ("RESPONSE_AWARE", "DECISION_AWARE"):
        feasible = {}
        for lam in lambdas:
            matching = [record for record in records if
                        record["method"] == method and
                        record["lambda_value"] == float(lam)]
            if len(matching) == len(ANCHORS) and all(
                    record["validation_weekly_cost"] is not None for record in matching):
                feasible[float(lam)] = float(np.mean([
                    record["validation_weekly_cost"] for record in matching]))
        if not feasible:
            raise RuntimeError(f"No fully certified validation setting for {method}")
        selected[method] = min(feasible, key=lambda lam: (feasible[lam], lam))
    experiment.save(output / "selected_hyperparameters.json", dict(
        lambda_response=selected["RESPONSE_AWARE"],
        lambda_decision=selected["DECISION_AWARE"],
        split_date="2012-07-01",
        anchors=[dict(alpha=s[1], h=s[2]) for s in ANCHORS],
        validation_booked_weekly_cost=base,
        candidates=records))
    print(json.dumps(selected, indent=2))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lambda_tuning"))
    parser.add_argument("--lambdas", type=float, nargs="+", default=[.03, .1, .3])
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--plan-seconds", type=float, default=300.)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 for value in args.lambdas):
        parser.error("L1 penalties must be finite and nonnegative")
    tune(args.workbook, args.out, args.lambdas,
         cores=args.cores, seconds=args.plan_seconds)


if __name__ == "__main__":
    main()
