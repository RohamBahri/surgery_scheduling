"""Small fixed-display stress test of behavioral assumptions.

Train nothing. A recommendation chosen under the nominal response is held
fixed while the actual response changes; the ordinary scheduler is rerun.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from pilot import experiment
from pilot import policy_training as training
from surgery.learning import ResponseDC
from surgery.weekly_planner import evaluate


def stress_scenarios(nominal):
    link, alpha, threshold = nominal
    return (
        ("lower_acceptance", max(.1, alpha - .2), threshold, "saturating"),
        ("attenuation_rejection", alpha, threshold / 2., "nonmonotone"),
    )


def cross_evaluate(context, args):
    """Evaluate two representative trained scenarios under two response errors."""
    anchors = [s for s in args.scenarios if (s[1], s[2]) in ((.5, 30.), (.8, 60.))]
    if not anchors:
        anchors = args.scenarios[:2]
    cache = training.PlannerCache(args.out, args.cores, args.plan_seconds)
    booked = np.asarray(context["held"]["booked"], float)
    realized = np.asarray(context["held"]["actual"], float)
    reference = experiment.solve_with_retry(
        cache, context["eval"], booked, "BOOKED", args)
    rows, summary = [], []
    for scenario in anchors:
        sid = experiment.scenario_id(*scenario)
        root, _, policy = experiment.load_decision_policy(args, scenario)
        features = experiment.full_features(
            context, args, scenario, root, context["eval"], "eval")
        nominal = ResponseDC(booked, scenario[1], scenario[2], 180., 1.,
                             link=scenario[0])
        fitted_accuracy = json.loads((root / "display_accuracy_reference.json").read_text())
        fitted_response = json.loads(
            (args.out / "scenarios" / sid / "response_aware.json").read_text())
        fitted = {
            "ACCURACY_TRAINED": np.asarray(fitted_accuracy["weights"], float),
            "RESPONSE_AWARE": np.asarray(fitted_response["weights"], float),
            "DECISION_AWARE": np.asarray(policy["vf_theta"], float),
        }
        # These displays are never changed when the true behavior changes.
        displays = {method: nominal.display(features @ weights)
                    for method, weights in fitted.items()}
        for label, alpha, threshold, link in stress_scenarios(scenario):
            actual_response = ResponseDC(booked, alpha, threshold, 180., 1.,
                                         link=link)
            outcomes = {"BOOKED": booked}
            plans = {"BOOKED": reference}
            for method, fixed_display in displays.items():
                outcomes[method] = actual_response.reviewed_from_display(fixed_display)
                plans[method] = experiment.solve_with_retry(
                    cache, context["eval"], outcomes[method],
                    f"stress_{sid}_{label}_{method}", args)
            weekly = defaultdict(lambda: defaultdict(float))
            all_components = defaultdict(set)
            complete_components = defaultdict(set)
            for key, component, positions in context["eval"]:
                week = str(component.monday)
                all_components[week].add(key)
                for method in outcomes:
                    rec = plans[method][key]
                    if rec["complete"]:
                        complete_components[(week, method)].add(key)
                    realized_cost = evaluate(
                        component, rec["assignment"], component.actual)["cost"]
                    weekly[week][method] += realized_cost
            complete_weeks = [
                week for week in weekly if all(
                    complete_components[(week, method)] == all_components[week]
                    for method in outcomes)
            ]
            for week in sorted(complete_weeks):
                for method in outcomes:
                    rows.append(dict(
                        trained_scenario=sid, stress=label, actual_alpha=alpha,
                        actual_h=threshold, actual_link=link, week=week,
                        method=method, cost=weekly[week][method]))
            for method in fitted:
                advice = displays[method]
                adjusted = outcomes[method] - booked
                acted = np.abs(advice) > 1e-8
                rejection = acted & (np.abs(adjusted) < 1e-8)
                attenuated = acted & (np.abs(adjusted) + 1e-8 <
                                      alpha * np.minimum(np.abs(advice), threshold))
                differences = np.array([weekly[week][method] -
                                        weekly[week]["BOOKED"]
                                        for week in sorted(complete_weeks)])
                summary.append(dict(
                    trained_scenario=sid, stress=label, actual_alpha=alpha,
                    actual_h=threshold, method=method,
                    complete_weeks=len(complete_weeks),
                    mean_weekly_difference=float(differences.mean())
                        if len(differences) else None,
                    rejection_share=float(rejection.mean()),
                    attenuation_share=float(attenuated.mean())))
    experiment.export_rows(args.out / "stress_weekly.csv", rows)
    experiment.export_rows(args.out / "stress_summary.csv", summary)
    return summary
