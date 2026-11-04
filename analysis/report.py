"""Turn frozen component results into paper tables, figures, and numbers.json.

Run after training and evaluation; no optimizer is imported or invoked here.
All scheduling costs are reported as calendar-week totals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot.experiment import METHODS, METHOD_NAMES


def json_safe(value):
    """Convert pandas/numpy scalars and missing values to strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen result {path}")
    return pd.read_csv(path)


def save_figure(fig, folder, name):
    folder.mkdir(exist_ok=True, parents=True)
    for suffix in ("svg", "png"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def descriptive_statistics(cases, tables, figures):
    booked = cases.booked.to_numpy(float)
    actual = cases.actual.to_numpy(float)
    discrepancy = booked - actual
    overbooked = discrepancy[discrepancy > 0]
    underbooked = -discrepancy[discrepancy < 0]
    output = {
        "cases": int(len(cases)),
        "overbooking_share": float(np.mean(discrepancy > 0)),
        "conditional_mean_overbooking_minutes": float(overbooked.mean()),
        "conditional_mean_underbooking_minutes": float(underbooked.mean()),
        "underbooking_share": float(np.mean(discrepancy < 0)),
        "mean_absolute_error_minutes": float(np.abs(discrepancy).mean()),
        "median_absolute_error_minutes": float(np.median(np.abs(discrepancy))),
        "mean_booked_minutes": float(booked.mean()),
        "mean_realized_minutes": float(actual.mean()),
        "rounded_30_minute_bookings_share": float(np.mean(
            np.isclose((booked - 1) % 30, 0))),
    }
    table = cases.groupby(["group", "split"], as_index=False).agg(
        cases=("case_id", "count"), mean_booked=("booked", "mean"),
        mean_realized=("actual", "mean"))
    table.to_csv(tables / "cohort_by_site.csv", index=False)

    # Measured on the raw retained booking in minutes, not the +1 model floor.
    quartiles = pd.qcut(cases.booked, 4, labels=False, duplicates="drop")
    detail = cases.assign(quartile=quartiles,
        absolute_error=np.abs(discrepancy)).groupby(
            "quartile", as_index=False).agg(
                cases=("case_id", "count"), mean_absolute_error=("absolute_error", "mean"),
                mean_booking=("booked", "mean"))
    detail.to_csv(tables / "booking_error_by_quartile.csv", index=False)

    # The validation period is never used to fit these descriptive summaries.
    train = cases.loc[cases.split.eq("train")].copy()
    test = cases.loc[cases.split.eq("test")].copy()
    for frame in (train, test):
        frame["signed_error"] = frame.booked - frame.actual
    key = ["surgeon", "procedure"]
    earlier = train.groupby(key).signed_error.agg(["mean", "size"]).reset_index()
    later = test.groupby(key).signed_error.agg(["mean", "size"]).reset_index()
    pairs = earlier.merge(later, on=key, suffixes=("_train", "_test"))
    pairs = pairs[(pairs.size_train >= 3) & (pairs.size_test >= 3)]
    output["surgeon_procedure_pairs_for_persistence"] = int(len(pairs))
    if len(pairs) >= 2:
        output["surgeon_procedure_error_correlation"] = float(
            pairs.mean_train.corr(pairs.mean_test))
        output["surgeon_procedure_same_sign_share"] = float(np.mean(
            np.sign(pairs.mean_train) == np.sign(pairs.mean_test)))
    pd.DataFrame([dict(metric=key, value=value) for key, value in output.items()
                  ]).to_csv(tables / "data_statistics.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.hist(actual - booked, bins=70)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Realized minus booked duration (minutes)", ylabel="Cases",
           title="Booking error distribution")
    save_figure(fig, figures, "booking_errors")
    return output


def conceptual_figures(folder):
    """Paper-ready study-timing and response-model diagrams."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from surgery.learning import response, response_saturating

    steps = ["Surgeon\nbooking", "Displayed\nadvice", "Surgeon\nreview",
             "Weekly\nplanner", "Realized\ncost"]
    fig, ax = plt.subplots(figsize=(10.4, 2.0))
    ax.set(xlim=(0, 10.4), ylim=(0, 2))
    ax.axis("off")
    for index, title in enumerate(steps):
        left = .13 + index * 2.08
        ax.add_patch(FancyBboxPatch((left, .55), 1.72, .9,
            boxstyle="round,pad=0.07", facecolor="white",
            edgecolor="black", linewidth=1.1))
        ax.text(left + .86, 1.0, title, ha="center", va="center", fontsize=10)
        if index < len(steps) - 1:
            ax.annotate("", xy=(left + 2.0, 1.0),
                        xytext=(left + 1.81, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.3))
    save_figure(fig, folder, "workflow")

    grid = np.linspace(-110, 110, 500)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(grid, response_saturating(grid, .8, 30.),
            label="Saturating response")
    ax.plot(grid, response(grid, .8, 30.),
            label="Attenuation and rejection")
    ax.axhline(0, color="black", linewidth=.6)
    ax.axvline(0, color="black", linewidth=.6)
    ax.set(xlabel="Displayed correction (minutes)",
           ylabel="Implemented correction (minutes)",
           title="Illustrative surgeon response (adoption 0.8, threshold 30)")
    ax.legend(frameon=False)
    save_figure(fig, folder, "surgeon_response")


def tuning_figures(run, output):
    path = run / "selected_hyperparameters.json"
    if not path.exists():
        return
    selected = json.loads(path.read_text())
    candidates = pd.DataFrame(selected["candidates"])
    candidates.to_csv(output / "tables" / "lambda_validation.csv", index=False)
    pd.DataFrame([dict(method=key, selected_lambda=value)
                  for key, value in selected.items()
                  if key.startswith("lambda_")]).to_csv(
                      output / "tables" / "selected_lambdas.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    means = candidates.groupby(["method", "lambda_value"], as_index=False)[
        "validation_weekly_cost"].mean()
    for method, sample in means.groupby("method"):
        ax.plot(sample.lambda_value, sample.validation_weekly_cost,
                marker="o", label=method.replace("_", " ").title())
    ax.set(xlabel="L1 coefficient penalty",
           ylabel="Mean weekly validation cost",
           title="Training-period regularization screening")
    ax.legend(frameon=False)
    save_figure(fig, output / "figures", "lambda_validation")


def convergence_figure(run, output):
    chosen = run / "scenarios" / "saturating_a0p5_h30" / "selected.json"
    if not chosen.exists():
        return
    mu = json.loads(chosen.read_text())["mu"]
    suffix = format(float(mu), ".8g").replace(".", "p")
    progress = chosen.parent / f"mu_{suffix}" / "vf_direct_progress.json"
    if not progress.exists():
        return
    history = json.loads(progress.read_text()).get("history", [])
    if not history:
        return
    pd.DataFrame(history).drop(columns=["inner"], errors="ignore").to_csv(
        output / "tables" / "decision_training_convergence.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot([v["outer"] for v in history],
            [v["majorizer_after"] for v in history], marker="o")
    ax.set(xlabel="Majorization iteration",
           ylabel="Penalized fixed-schedule majorizer",
           title="Decision-aware training diagnostics (adoption 0.5, h=30)")
    save_figure(fig, output / "figures", "training_convergence")


def main_tables(main, components, weekly, intervals, output):
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(exist_ok=True, parents=True)
    main.to_csv(tables / "main_results.csv", index=False)
    weekly.to_csv(tables / "weekly_costs.csv", index=False)

    scenarios = []
    for row in main.to_dict("records"):
        for method in METHODS:
            scenarios.append(dict(scenario=row["scenario"], alpha=row["alpha"],
                h=row["h"], method=METHOD_NAMES[method],
                mean_weekly_cost=row[f"{method}_cost_per_week"],
                mean_weekly_overtime=row[f"{method}_overtime_per_week"],
                mean_weekly_idle=row[f"{method}_idle_per_week"],
                post_review_mae=row[f"{method}_mae_per_case"],
                mean_weekly_saving_vs_booked=(
                    row["BOOKED_cost_per_week"]-row[f"{method}_cost_per_week"])))
    long = pd.DataFrame(scenarios)
    long.to_csv(tables / "scenario_method_results.csv", index=False)

    bounds = []
    for scenario, comparison in intervals.items():
        for baseline, values in comparison.items():
            bounds.append(dict(scenario=scenario,
                comparison=f"Decision-aware - {METHOD_NAMES[baseline]}",
                weekly_difference=values["mean_weekly_difference"],
                confidence_low=values["confidence_95"][0],
                confidence_high=values["confidence_95"][1],
                weeks_better=values["better_weeks"],
                weeks_worse=values["worse_weeks"]))
    pd.DataFrame(bounds).to_csv(tables / "paired_weekly_intervals.csv", index=False)

    # Report site-specific results from the exact same complete calendar weeks.
    selected_weeks = set(weekly.week.astype(str))
    selected = components.loc[components.week.astype(str).isin(selected_weeks)]
    site = selected.groupby(["scenario", "group", "method", "week"], as_index=False)[
        ["cost", "overtime", "idle"]].sum()
    site_summary = site.groupby(["scenario", "group", "method"], as_index=False)[
        ["cost", "overtime", "idle"]].mean()
    site_summary.rename(columns={
        "cost": "mean_weekly_cost", "overtime": "mean_weekly_overtime",
        "idle": "mean_weekly_idle"}, inplace=True)
    site_summary["method_name"] = site_summary.method.map(METHOD_NAMES)
    site_summary.to_csv(tables / "site_results.csv", index=False)
    return long, site_summary


def main_figures(long, weekly, output):
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, group in long.groupby("method", sort=False):
        ax.scatter(group.post_review_mae, group.mean_weekly_cost, label=name,
                   s=42, alpha=.8)
    ax.set(xlabel="Post-review mean absolute error (minutes per case)",
           ylabel="Mean total realized scheduling cost per week",
           title="Prediction accuracy versus scheduling performance")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save_figure(fig, figures, "accuracy_vs_weekly_cost")

    savings = long.loc[long.method.eq("Decision-aware")]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for alpha, group in savings.groupby("alpha"):
        ax.plot(group.h, group.mean_weekly_saving_vs_booked,
                marker="o", label=f"Adoption {alpha:g}")
    ax.axhline(0, linewidth=.9, color="black")
    ax.set(xlabel="Response threshold h (minutes)",
           ylabel="Mean weekly cost saving versus Booked",
           title="Cost savings across response scenarios")
    ax.legend(frameon=False)
    save_figure(fig, figures, "weekly_savings_by_threshold")

    anchor = long.loc[(np.isclose(long.alpha, .5)) &
                      (np.isclose(long.h, 30.))].copy()
    if not anchor.empty:
        anchor = anchor.set_index("method").reindex([METHOD_NAMES[m] for m in METHODS])
        fig, ax = plt.subplots(figsize=(8.3, 4.3))
        xpos = np.arange(len(anchor))
        ax.bar(xpos, anchor.mean_weekly_idle, label="Idle minutes")
        ax.bar(xpos, anchor.mean_weekly_overtime,
               bottom=anchor.mean_weekly_idle, label="Overtime minutes")
        ax.set_xticks(xpos, anchor.index, rotation=25, ha="right")
        ax.set(ylabel="Mean total minutes per week",
               title="Realized capacity utilization (adoption 0.5, threshold 30)")
        ax.legend(frameon=False)
        save_figure(fig, figures, "overtime_idle")

    anchor_id = "saturating_a0p5_h30"
    sub = weekly.loc[weekly.scenario.eq(anchor_id)]
    if not sub.empty:
        pairs = sub.pivot(index="week", columns="method", values="cost")
        if {"BOOKED", "DECISION_AWARE"}.issubset(pairs):
            diff = pairs.BOOKED - pairs.DECISION_AWARE
            fig, ax = plt.subplots(figsize=(8.0, 3.7))
            ax.bar(np.arange(len(diff)), diff)
            ax.axhline(0, linewidth=1, color="black")
            ax.set(xlabel="Evaluation calendar weeks (chronological)",
                   ylabel="Booked minus Decision-aware weekly cost",
                   title="Paired weekly operational improvements")
            save_figure(fig, figures, "weekly_cost_differences")


def report(run: Path, context: Path, output: Path):
    main = read_csv(run / "main_results.csv")
    weekly = read_csv(run / "weekly_results.csv")
    components = read_csv(run / "component_results.csv")
    intervals = json.loads((run / "paired_bootstrap.json").read_text())
    output.mkdir(exist_ok=True, parents=True)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    long, site = main_tables(main, components, weekly, intervals, output)
    main_figures(long, weekly, output)
    conceptual_figures(figures)
    tuning_figures(run, output)
    convergence_figure(run, output)

    numbers = {
        "evaluation_status": "Exploratory: Jan-Jun 2013 was inspected during development",
        "sample": dict(calendar_weeks=int(weekly.week.nunique()),
                       cases=int(main.cases.iloc[0]),
                       scenarios=int(len(main)),
                       methods=list(METHOD_NAMES.values())),
        "scenarios": main.to_dict("records"),
        "weekly_uncertainty": intervals,
    }
    retained = context.parent / "retained_cases.csv"
    if retained.exists():
        cases = pd.read_csv(retained, parse_dates=["date"])
        numbers["data"] = descriptive_statistics(cases, tables, figures)
    stress = run / "stress_summary.csv"
    if stress.exists():
        read_csv(stress).to_csv(tables / "response_stress.csv", index=False)
    coverage = run / "coverage.json"
    if coverage.exists():
        numbers["certification"] = json.loads(coverage.read_text())
    (output / "numbers.json").write_text(json.dumps(
        json_safe(numbers), indent=2, sort_keys=True, allow_nan=False))
    with (output / "summary.md").open("w") as file:
        file.write("# Surgery scheduling: reproducible results\n\n")
        file.write(f"Calendar weeks: {numbers['sample']['calendar_weeks']}; "
                   f"cases: {numbers['sample']['cases']}.\n\n")
        file.write("Scheduling costs are calendar-week totals; MAE is per case.\n\n")
        file.write("The 2013 evaluation was inspected during development and is exploratory.\n")
    print(f"Generated CSV tables, SVG/PNG figures, and numbers.json in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--context", type=Path, default=Path("artifacts/context.pkl"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report(args.run, args.context,
           args.out if args.out is not None else args.run / "publication")


if __name__ == "__main__":
    main()
