"""Single-dataset retrospective experiment: train, evaluate, and export components.

All scheduling policies use the same room roster, cost objective, response
assumptions, and deployment tie rule. Paper-facing method names are defined
here; internal mathematical implementations can retain short labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilot import policy_training as training
from pilot import policy_runner
from surgery.learning import DCObjective, ResponseDC, pdca
from surgery.weekly_planner import evaluate, loads


METHODS = (
    "BOOKED", "ACCURACY_TRAINED", "RESPONSE_AWARE",
    "DECISION_AWARE", "PERFECT_ADVICE", "HINDSIGHT",
)
METHOD_NAMES = {
    "BOOKED": "Booked",
    "ACCURACY_TRAINED": "Accuracy-trained",
    "RESPONSE_AWARE": "Response-aware",
    "DECISION_AWARE": "Decision-aware",
    "PERFECT_ADVICE": "Perfect advice",
    "HINDSIGHT": "Hindsight",
}
DEFAULT_SCENARIOS = [(alpha, h) for alpha in (0.5, 0.8)
                     for h in (20., 30., 60.)] + [(0.3, 15.), (0.8, 90.)]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(path)


def scenario_id(link, alpha, h):
    def token(number):
        return format(float(number), ".8g").replace(".", "p")
    return f"{link}_a{token(alpha)}_h{token(h)}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "cross-evaluate"),
                        default="all")
    parser.add_argument("--context", type=Path, default=policy_runner.DEFAULT_CONTEXT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/paper_experiment"))
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Optional alpha:h pairs (e.g. 0.5:30). Defaults to 8 prespecified scenarios.")
    parser.add_argument("--cores", type=int, default=15)
    parser.add_argument("--plan-seconds", type=float, default=300.)
    parser.add_argument("--retry-seconds", type=float, default=900.)
    parser.add_argument("--case-steps", type=int, default=200)
    parser.add_argument("--vf-outers", type=int, default=30)
    parser.add_argument("--vf-pdca-steps", type=int, default=50)
    parser.add_argument("--cuts", type=int, default=80)
    parser.add_argument("--radius", type=float, default=100.)
    parser.add_argument("--accuracy-budget", type=float, default=.05)
    parser.add_argument("--lambda-decision", type=float, default=.1)
    parser.add_argument("--lambda-response", type=float, default=.1)
    parser.add_argument("--mu-candidates", type=float, nargs="+",
                        default=[2., 4., 8.])
    parser.add_argument("--tuning", type=Path, default=None,
                        help="Training-only selected_hyperparameters.json from pilot.tune")
    args = parser.parse_args(argv)
    if args.tuning is not None:
        selected = json.loads(args.tuning.read_text())
        args.lambda_decision = float(selected["lambda_decision"])
        args.lambda_response = float(selected["lambda_response"])
    if not args.context.is_file():
        parser.error(f"Study context missing: {args.context}; run python -m pilot.prepare_data")
    if args.cores < 1 or args.plan_seconds <= 0 or args.retry_seconds < args.plan_seconds:
        parser.error("Invalid planning time or worker count")
    if min(args.case_steps, args.vf_outers, args.vf_pdca_steps, args.cuts) < 1:
        parser.error("Training iteration limits must be positive")
    if min(args.radius, args.accuracy_budget + 1, args.lambda_decision + 1,
           args.lambda_response + 1) <= 0 or not all(np.isfinite(x) for x in (
               args.radius, args.accuracy_budget, args.lambda_decision, args.lambda_response)):
        parser.error("Invalid training settings")
    if any(not np.isfinite(mu) or mu < 0 for mu in args.mu_candidates):
        parser.error("Accuracy penalties must be finite and nonnegative")
    pairs = DEFAULT_SCENARIOS if args.scenarios is None else [
        tuple(float(v) for v in entry.split(":")) for entry in args.scenarios]
    if not pairs or any(len(pair) != 2 or not 0 < pair[0] < 1 or pair[1] <= 0
                        for pair in pairs):
        parser.error("Scenarios must be alpha:h with 0<alpha<1 and h>0")
    args.scenarios = [("saturating", alpha, h) for alpha, h in dict.fromkeys(pairs)]
    args.mu_candidates = sorted(set(args.mu_candidates))
    args.context = args.context.resolve()
    args.out = args.out.resolve()
    return args


def contract(args):
    return {
        "schema": "paper_experiment_v2",
        "context_sha256": hashlib.sha256(args.context.read_bytes()).hexdigest(),
        "methods": list(METHODS),
        "scenarios": [dict(link=link, alpha=alpha, h=h)
                      for link, alpha, h in args.scenarios],
        "cores": args.cores, "plan_seconds": args.plan_seconds,
        "retry_seconds": args.retry_seconds, "case_steps": args.case_steps,
        "vf_outers": args.vf_outers, "vf_pdca_steps": args.vf_pdca_steps,
        "cuts": args.cuts, "radius": args.radius,
        "accuracy_budget": args.accuracy_budget,
        "lambda_decision": args.lambda_decision,
        "lambda_response": args.lambda_response,
        "mu_candidates": args.mu_candidates,
        "tuning_file_sha256": (hashlib.sha256(args.tuning.read_bytes()).hexdigest()
                               if args.tuning is not None else None),
        "turnover": 0, "deploy_tie": "hash_only",
    }


def prepare_output(args):
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "experiment_contract.json"
    expected = contract(args)
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError("Experiment settings changed: use a fresh output directory")
    save(path, expected)


def fit_initializer(context, args):
    """Shared 77-feature LAD fit used ONLY as an optimizer initialization."""
    path = args.out / "lad_initializer.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["context_sha256"] != contract(args)["context_sha256"]:
            raise ValueError("Initializer belongs to another training dataset")
        return np.asarray(record["weights"], float)
    fit = context["fit"]
    X = np.asarray(fit["X"], float)
    residual = np.asarray(fit["actual"]) - np.asarray(fit["booked"])
    penalty = training.train_lambda(X, len(context["train"]), intercept=True)
    coefficients, status = training.lad_start(
        X, residual, penalty, len(context["train"]), args.radius)
    if coefficients is None:
        raise RuntimeError(f"LAD initialization failed: {status}")
    save(path, dict(weights=coefficients.tolist(),
                    context_sha256=contract(args)["context_sha256"],
                    optimization=status))
    return coefficients


def policy_args(args, scenario, mu, root, stage):
    link, alpha, h = scenario
    return [
        "--stage", stage, "--context", str(args.context),
        "--out", str(root), "--initial-weights", str(args.out / "lad_initializer.json"),
        "--link", link, "--alpha", str(alpha), "--h", str(h),
        "--cores", str(args.cores), "--plan-seconds", str(args.plan_seconds),
        "--case-steps", str(args.case_steps), "--vf-outers", str(args.vf_outers),
        "--vf-pdca-steps", str(args.vf_pdca_steps), "--cuts", str(args.cuts),
        "--radius", str(args.radius), "--accuracy-mu", str(mu),
        "--accuracy-budget", str(args.accuracy_budget),
        "--lambda-vf", str(args.lambda_decision),
        "--turnover", "0", "--train-tie", "primary", "--accuracy-gate", "off",
    ]


def root_for(args, scenario, mu):
    return args.out / "scenarios" / scenario_id(*scenario) / (
        "mu_" + format(mu, ".8g").replace(".", "p"))


def train_decision_policy(args, scenario):
    """Train candidates until the first feasible accuracy multiplier is found."""
    scenario_dir = args.out / "scenarios" / scenario_id(*scenario)
    selection = scenario_dir / "selected.json"
    if selection.exists():
        return json.loads(selection.read_text())
    initial_reference = None
    candidates = []
    for mu in args.mu_candidates:
        root = root_for(args, scenario, mu)
        root.mkdir(parents=True, exist_ok=True)
        if initial_reference is not None and not (root / "display_accuracy_reference.json").exists():
            # Same scenario, features and accuracy reference for every mu.
            import shutil
            shutil.copyfile(initial_reference, root / "display_accuracy_reference.json")
        policy_runner.main(policy_args(args, scenario, mu, root, "train"))
        policy = json.loads((root / "selected_policy.json").read_text())
        reference = root / "display_accuracy_reference.json"
        initial_reference = reference
        ratio = float(policy["accuracy_ratio"])
        record = dict(mu=mu, accuracy_ratio=ratio,
                      accuracy_met=ratio <= 1 + args.accuracy_budget + 1e-9,
                      status=policy["decision_status"])
        candidates.append(record)
        if record["accuracy_met"]:
            selected = dict(mu=mu, candidates=candidates, accuracy_met=True)
            save(selection, selected)
            return selected
    selected = dict(mu=args.mu_candidates[-1], candidates=candidates, accuracy_met=False)
    save(selection, selected)
    return selected


def load_decision_policy(args, scenario):
    selected = json.loads((args.out / "scenarios" / scenario_id(*scenario) /
                           "selected.json").read_text())
    mu = selected["mu"]
    root = root_for(args, scenario, mu)
    check_args = policy_runner.parse_args(policy_args(args, scenario, mu, root, "evaluate"))
    _, policy = policy_runner.verified_context(check_args)
    return root, selected, policy


def full_features(context, args, scenario, root, items, partition):
    """The same 83 features for all learned comparators."""
    case_features = np.asarray(context["fit" if partition == "train" else "held"]["X"])
    scaling = json.loads((root / "features.json").read_text())
    structural, _, _ = training.five_features(
        context, items, None, partition,
        scaler=(np.asarray(scaling["mean"]), np.asarray(scaling["std"])),
        schema="structural")
    return np.column_stack([case_features, structural])


def fit_response_policy(context, args, scenario, root):
    """Case-local asymmetric loss and displayed-accuracy safeguard; no planner training."""
    path = args.out / "scenarios" / scenario_id(*scenario) / "response_aware.json"
    if path.exists():
        return json.loads(path.read_text())
    response = ResponseDC(context["fit"]["booked"], scenario[1], scenario[2],
                          180., 1., link=scenario[0])
    design = full_features(context, args, scenario, root, context["train"], "train")
    reference = json.loads((root / "display_accuracy_reference.json").read_text())
    fitted_reference = float(reference["accuracy_per_component"])
    starting = np.asarray(reference["weights"], float)
    candidates = []
    for mu in args.mu_candidates:
        objective = DCObjective(
            context["fit"]["booked"], design,
            np.zeros(len(design)), response, mode="CASE",
            actual=context["fit"]["actual"], components=len(context["train"]),
            lam=args.lambda_response,
            penalize=np.r_[0., np.ones(design.shape[1]-1)],
            accuracy_mu=mu)
        result = pdca(objective, starting, radius=args.radius,
                      outer=args.case_steps, backend="gurobi")
        weights = np.asarray(result["weights"], float)
        displayed = context["fit"]["booked"] + response.display(design @ weights)
        ratio = float(np.abs(context["fit"]["actual"] - displayed).sum() /
                      len(context["train"]) / fitted_reference)
        record = dict(mu=mu, ratio=ratio, status=result["status"],
                      weights=weights.tolist())
        candidates.append({key: value for key, value in record.items() if key != "weights"})
        if ratio <= 1 + args.accuracy_budget + 1e-9:
            output = dict(**record, candidates=candidates, accuracy_met=True)
            save(path, output)
            return output
        starting = weights
    output = dict(**record, candidates=candidates, accuracy_met=False)
    save(path, output)
    return output


def solve_with_retry(cache, items, durations, tag, args):
    plans = cache.solve(items, durations, tag, tie=True)
    unresolved = [(key, comp, idx) for key, comp, idx in items if not plans[key]["complete"]]
    if unresolved and args.retry_seconds > args.plan_seconds:
        fallback = training.PlannerCache(args.out / "retry", args.cores, args.retry_seconds)
        retried = fallback.solve(unresolved, durations, tag, tie=True)
        for key, _, _ in unresolved:
            if retried[key]["complete"]:
                plans[key] = retried[key]
    return plans


def evaluate_scenario(context, args, scenario, shared, cache):
    link, alpha, h = scenario
    sid = scenario_id(*scenario)
    root, selected, policy = load_decision_policy(args, scenario)
    response = ResponseDC(context["held"]["booked"], alpha, h, 180., 1., link=link)
    booked = np.asarray(context["held"]["booked"], float)
    realized = np.asarray(context["held"]["actual"], float)
    X = full_features(context, args, scenario, root, context["eval"], "eval")
    accuracy_fit = json.loads((root / "display_accuracy_reference.json").read_text())
    response_fit = json.loads((args.out / "scenarios" / sid /
                               "response_aware.json").read_text())
    scores = {
        "ACCURACY_TRAINED": X @ np.asarray(accuracy_fit["weights"], float),
        "RESPONSE_AWARE": X @ np.asarray(response_fit["weights"], float),
        "DECISION_AWARE": X @ np.asarray(policy["decision_weights"], float),
    }
    displays = {method: response.display(score) for method, score in scores.items()}
    displays["PERFECT_ADVICE"] = np.clip((realized - booked)/alpha, -h, h)
    displays["BOOKED"] = np.zeros_like(booked)
    displays["HINDSIGHT"] = realized - booked
    durations = {
        method: response.reviewed_from_display(display)
        for method, display in displays.items() if method not in ("HINDSIGHT",)
    }
    durations["BOOKED"] = booked
    durations["HINDSIGHT"] = realized
    plans = dict(shared)
    for method in ("ACCURACY_TRAINED", "RESPONSE_AWARE", "DECISION_AWARE",
                   "PERFECT_ADVICE"):
        plans[method] = solve_with_retry(
            cache, context["eval"], durations[method], f"{sid}_{method}", args)
    rows = []
    for key, component, positions in context["eval"]:
        for method in METHODS:
            solved = plans[method][key]
            assignment = solved["assignment"]
            metrics = evaluate(component, assignment, component.actual)
            planned = evaluate(component, assignment, durations[method][positions])
            realized_loads = loads(component, assignment, component.actual)
            duration_error = component.actual - durations[method][positions]
            asymmetric_error = float(
                (1.75 * np.maximum(duration_error, 0.)
                 + np.maximum(-duration_error, 0.)).sum())
            upper_bound = float(planned["cost"] + asymmetric_error)
            if upper_bound + 1e-5 < metrics["cost"]:
                raise AssertionError("Realized cost exceeds fixed-schedule error bound")
            rows.append(dict(
                scenario=sid, alpha=alpha, h=h, link=link,
                method=method, method_name=METHOD_NAMES[method],
                component=str(key), group=str(component.group),
                week=str(component.monday), cases=int(component.n),
                complete=bool(solved["complete"]), status=solved["status"],
                cost=float(metrics["cost"]), planned_cost=float(planned["cost"]),
                asymmetric_error=asymmetric_error, loss_bound=upper_bound,
                idle=float(metrics["idle"]), overtime=float(metrics["overtime"]),
                mae=float(np.abs(component.actual-durations[method][positions]).sum()),
                displayed_mae=float(np.abs(component.actual -
                    booked[positions]-displays[method][positions]).sum()),
                max_realized_load=float(realized_loads.max()),
                over_720_rooms=int((realized_loads > 720).sum())))
    return rows, dict(decision_accuracy_ratio=float(policy["accuracy_ratio"]),
                      decision_accuracy_met=bool(selected["accuracy_met"]),
                      response_accuracy_ratio=float(response_fit["ratio"]),
                      response_accuracy_met=bool(response_fit["accuracy_met"]),
                      selected_mu_decision=float(selected["mu"]),
                      selected_mu_response=float(response_fit["mu"]))


def complete_week_keys(rows, scenario):
    """Keep entire calendar weeks; never report a partial week as a week."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["scenario"] == scenario:
            grouped[row["week"]][row["component"]][row["method"]] = row
    complete = {}
    for week, blocks in grouped.items():
        if all(set(METHODS) == set(methods) and
               all(methods[m]["complete"] for m in METHODS)
               for methods in blocks.values()):
            complete[week] = blocks
    return complete


def summarize(rows, scenarios, checks):
    by_scenario = {scenario_id(*scenario): complete_week_keys(rows, scenario_id(*scenario))
                   for scenario in scenarios}
    common_weeks = set.intersection(*(set(weeks) for weeks in by_scenario.values()))
    if not common_weeks:
        raise ValueError("No complete certified calendar weeks across the grid")
    rng = np.random.default_rng(20261103)
    table, bootstraps, weekly_rows = [], {}, []
    for scenario in scenarios:
        sid = scenario_id(*scenario)
        week_data = {}
        for week in sorted(common_weeks):
            policy_totals = {method: defaultdict(float) for method in METHODS}
            for component in by_scenario[sid][week].values():
                for method in METHODS:
                    rec = component[method]
                    for quantity in ("cost", "planned_cost", "idle", "overtime", "mae",
                                     "displayed_mae"):
                        policy_totals[method][quantity] += rec[quantity]
            week_data[week] = policy_totals
            for method in METHODS:
                weekly_rows.append(dict(scenario=sid, week=week, method=method,
                    **{key: float(value) for key, value in policy_totals[method].items()}))
        number_weeks = len(week_data)
        values = {m: np.array([week_data[w][m]["cost"] for w in sorted(week_data)])
                  for m in METHODS}
        counts = sum(row["cases"] for row in rows if row["scenario"] == sid and
                     row["method"] == "BOOKED" and row["week"] in common_weeks)
        result = dict(scenario=sid, alpha=scenario[1], h=scenario[2],
                      weeks=number_weeks, cases=counts, **checks[sid])
        for method in METHODS:
            for quantity in ("cost", "planned_cost", "idle", "overtime"):
                result[f"{method}_{quantity}_per_week"] = float(np.mean([
                    week_data[w][method][quantity] for w in week_data]))
            result[f"{method}_mae_per_case"] = float(
                sum(week_data[w][method]["mae"] for w in week_data)/counts)
            result[f"{method}_displayed_mae_per_case"] = float(
                sum(week_data[w][method]["displayed_mae"] for w in week_data)/counts)
        denominator = values["BOOKED"].mean() - values["HINDSIGHT"].mean()
        result["DECISION_AWARE_gap_closed_fraction"] = (
            float((values["BOOKED"].mean()-values["DECISION_AWARE"].mean())/denominator)
            if denominator > 1e-9 else None)
        draws = rng.integers(0, number_weeks, size=(5000, number_weeks))
        bootstraps[sid] = {}
        for baseline in ("BOOKED", "ACCURACY_TRAINED", "RESPONSE_AWARE",
                         "PERFECT_ADVICE"):
            differences = values["DECISION_AWARE"] - values[baseline]
            bootstraps[sid][baseline] = dict(
                mean_weekly_difference=float(differences.mean()),
                confidence_95=np.quantile(differences[draws].mean(axis=1),
                                          [.025, .975]).tolist(),
                better_weeks=int((differences < 0).sum()),
                worse_weeks=int((differences > 0).sum()),
                weeks=number_weeks)
        table.append(result)
    return table, bootstraps, weekly_rows, sorted(common_weeks)


def export_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(
            field for row in rows for field in row)))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_grid(context, args):
    cache = training.PlannerCache(args.out, args.cores, args.plan_seconds)
    booked = np.asarray(context["held"]["booked"], float)
    realized = np.asarray(context["held"]["actual"], float)
    shared = {
        "BOOKED": solve_with_retry(cache, context["eval"], booked, "BOOKED", args),
        "HINDSIGHT": solve_with_retry(cache, context["eval"], realized, "HINDSIGHT", args),
    }
    rows, checks = [], {}
    for scenario in args.scenarios:
        sid = scenario_id(*scenario)
        current, check = evaluate_scenario(context, args, scenario, shared, cache)
        rows.extend(current)
        checks[sid] = check
    table, uncertainty, weekly_rows, weeks = summarize(rows, args.scenarios, checks)
    export_rows(args.out / "component_results.csv", rows)
    export_rows(args.out / "weekly_results.csv", weekly_rows)
    export_rows(args.out / "main_results.csv", table)
    save(args.out / "paired_bootstrap.json", uncertainty)
    save(args.out / "coverage.json", dict(
        total_components=len(context["eval"]), common_complete_weeks=weeks,
        total_calendar_weeks=len({c.monday for _, c, _ in context["eval"]}),
        certified_by_scenario={
            scenario_id(*sc): sum(r["complete"] for r in rows
                if r["scenario"] == scenario_id(*sc) and r["method"] == "DECISION_AWARE")
            for sc in args.scenarios},
        response_accuracy_safeguard=checks))
    return table


def main(argv=None):
    args = parse_args(argv)
    prepare_output(args)
    policy_runner.configure_model(argparse.Namespace(turnover=0.))
    with args.context.open("rb") as input_file:
        context = pickle.load(input_file)
    if args.stage in ("train", "all"):
        fit_initializer(context, args)
        for scenario in args.scenarios:
            selected = train_decision_policy(args, scenario)
            root = root_for(args, scenario, selected["mu"])
            fit_response_policy(context, args, scenario, root)
    if args.stage in ("evaluate", "all"):
        evaluate_grid(context, args)
    if args.stage == "cross-evaluate":
        from pilot.response_stress import cross_evaluate
        cross_evaluate(context, args)


if __name__ == "__main__":
    main()
