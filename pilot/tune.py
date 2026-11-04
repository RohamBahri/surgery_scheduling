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


def mean_validation_cost(context, predictions, cache, label, validation_weeks):
    from surgery.weekly_planner import evaluate
    selected_items = [item for item in context["eval"]
                      if item[1].monday in validation_weeks]
    plans = cache.solve(selected_items, predictions, label, tie=True)
    if not all(plan["complete"] for plan in plans.values()):
        return None
    total = sum(evaluate(component, plans[key]["assignment"], component.actual)["cost"]
                for key, component, positions in selected_items)
    weeks = len(validation_weeks)
    return float(total / weeks)


def tune(workbook: Path, output: Path, lambdas, *, cores=15, seconds=300,
         accuracy_mu=2.):
    output.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(accuracy_mu) or accuracy_mu < 0:
        raise ValueError("Tuning accuracy penalty must be nonnegative and finite")
    settings = dict(
        workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
        split_date="2012-07-01", end_date="2013-01-01",
        lambdas=[float(v) for v in lambdas], accuracy_mu=float(accuracy_mu),
        anchors=[dict(alpha=a, h=h) for _, a, h in ANCHORS],
        planner_seconds=float(seconds))
    settings_file = output / "tuning_contract.json"
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        raise ValueError("Tuning settings changed: use a fresh --out directory")
    experiment.save(settings_file, settings)
    context_file = output / "context.pkl"
    if not context_file.exists():
        context = prepare(workbook, context_file, "2012-07-01", "2013-01-01")
    else:
        import pickle
        with context_file.open("rb") as stream:
            context = pickle.load(stream)
        if context["metadata"]["workbook_sha256"] != hashlib.sha256(
                workbook.read_bytes()).hexdigest():
            raise ValueError("Validation context belongs to a different workbook")
        if (context["metadata"].get("split_date") != "2012-07-01" or
                context["metadata"].get("end_date") != "2013-01-01"):
            raise ValueError("Old or incompatible tuning context; use a fresh --out directory")
    # This must hold even when reusing a locally cached tuning context.
    if (context["metadata"].get("split_date") != "2012-07-01" or
            context["metadata"].get("end_date") != "2013-01-01"):
        raise ValueError("Tuning requires July-2012 split and Dec-2012 end")
    if any(component.monday >= "2013-01-01" for _, component, _ in context["eval"]):
        raise AssertionError("2013 scheduling weeks entered tuning validation")
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
    available_weeks = sorted({component.monday for _, component, _ in context["eval"]})
    positions = np.linspace(0, len(available_weeks)-1,
                            num=min(8, len(available_weeks)), dtype=int)
    validation_weeks = {available_weeks[i] for i in positions}
    assert max(validation_weeks) < "2013-01-01", "Tuning validation crossed into 2013"
    base = mean_validation_cost(
        context, validation_booked, cache, "Booked", validation_weeks)
    if base is None:
        raise RuntimeError("Booked validation plans are not all certified")
    records = []
    accuracy_references = {}

    # The decision-aware runs also fit the prediction-only reference; reuse it
    # when screening Response-aware rather than fitting it repeatedly.
    for method in ("DECISION_AWARE", "RESPONSE_AWARE"):
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
                            accuracy_mu=accuracy_mu)
                        fit = pdca(objective, accuracy_references[scenario_name][0], radius=100.,
                                   outer=50, backend="gurobi")
                        weights = np.asarray(fit["weights"], float)
                        experiment.save(candidate_file, dict(weights=weights.tolist(),
                                                              status=fit["status"]))
                else:
                    arguments = experiment.policy_args(
                        initialization_args, scenario, accuracy_mu, folder, "train")
                    option = arguments.index("--lambda-vf")
                    arguments[option + 1] = str(lam)
                    option = arguments.index("--vf-outers")
                    arguments[option + 1] = "6"
                    option = arguments.index("--vf-pdca-steps")
                    arguments[option + 1] = "20"
                    policy_runner.main(arguments)
                    policy = json.loads((folder / "selected_policy.json").read_text())
                    weights = np.asarray(policy["decision_weights"], float)
                    reference_record = json.loads(
                        (folder / "display_accuracy_reference.json").read_text())
                    reference_weights = np.asarray(reference_record["weights"], float)
                    reference_score = float(reference_record["accuracy_per_component"])
                    if scenario_name in accuracy_references:
                        previous_weights, previous_score = accuracy_references[scenario_name]
                        if (not np.allclose(previous_weights, reference_weights) or
                                not np.isclose(previous_score, reference_score)):
                            raise ValueError("Inconsistent accuracy references across L1 candidates")
                    accuracy_references[scenario_name] = (reference_weights, reference_score)
                reference_score = accuracy_references[scenario_name][1]
                training_display = booked + response_fit.display(fit_design @ weights)
                accuracy_ratio = float(
                    np.abs(actual - training_display).sum() /
                    len(context["train"]) / reference_score)
                accuracy_met = accuracy_ratio <= 1.05 + 1e-9
                reviewed = response_validation.reviewed(validation_design @ weights)
                value = mean_validation_cost(
                    context, reviewed, cache,
                    f"{method}_lambda_{lam:g}_{scenario_name}", validation_weeks)
                records.append(dict(method=method, lambda_value=float(lam),
                    scenario=scenario_name, validation_weekly_cost=value,
                    saving_vs_booked=None if value is None else base-value,
                    training_display_accuracy_ratio=accuracy_ratio,
                    accuracy_met=bool(accuracy_met)))
                experiment.save(output / "tuning_progress.json", records)

    selected = {}
    for method in ("RESPONSE_AWARE", "DECISION_AWARE"):
        feasible = {}
        for lam in lambdas:
            matching = [record for record in records if
                        record["method"] == method and
                        record["lambda_value"] == float(lam)]
            if len(matching) == len(ANCHORS) and all(
                    record["validation_weekly_cost"] is not None and record["accuracy_met"]
                    for record in matching):
                feasible[float(lam)] = float(np.mean([
                    record["validation_weekly_cost"] for record in matching]))
        if not feasible:
            raise RuntimeError(
                f"No accuracy-feasible, fully certified L1 setting for {method}; "
                "inspect tuning_progress.json and consider a different --accuracy-mu")
        selected[method] = min(feasible, key=lambda lam: (feasible[lam], lam))
    experiment.save(output / "selected_hyperparameters.json", dict(
        lambda_response=selected["RESPONSE_AWARE"],
        lambda_decision=selected["DECISION_AWARE"],
        split_date="2012-07-01",
        anchors=[dict(alpha=s[1], h=s[2]) for s in ANCHORS],
        validation_booked_weekly_cost=base,
        validation_weeks=sorted(validation_weeks),
        tuning_accuracy_mu=float(accuracy_mu),
        end_date="2013-01-01",
        candidates=records))
    print(json.dumps(selected, indent=2))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lambda_tuning"))
    parser.add_argument("--lambdas", type=float, nargs="+", default=[.03, .1, .3])
    parser.add_argument("--accuracy-mu", type=float, default=2.)
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--plan-seconds", type=float, default=300.)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 for value in args.lambdas):
        parser.error("L1 penalties must be finite and nonnegative")
    tune(args.workbook, args.out, args.lambdas,
         cores=args.cores, seconds=args.plan_seconds, accuracy_mu=args.accuracy_mu)


if __name__ == "__main__":
    main()
