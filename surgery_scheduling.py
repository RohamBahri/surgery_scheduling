# =============================================================================
# IMPORTS
# =============================================================================
from datetime import timedelta
import numpy as np
from copy import deepcopy
from datetime import timedelta

from src.config import PARAMS
from src.data_processing import load_data, split_data, compute_block_capacity
from src.predictors import train_lasso_predictor, train_knn_predictor, train_lasso_asym
from src.scheduling_utils import (
    select_surgeries,
    extract_schedule,
    evaluate_schedule_actual_costs,
)
from src.data_processing import attach_pred
from src.stochastic_utils import build_empirical_distributions, sample_scenarios
from src.solver_utils import (
    solve_saa_model,
    solve_deterministic_model,
    solve_predictive_model,
    solve_clairvoyant_model,
)
from src.output import (
    init_output,
    append_horizon,
    print_console_summary,
    save_detailed,
    save_aggregate
)



# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main() -> None:
    """multiple horizons elective surgery scheduling experiment with optional SAA."""

    print("\n=== Elective Surgery Scheduling – 8 h blocks ===")

    # 1) Load & split data
    df_all = load_data(PARAMS)
    df_warm, df_pool, horizon_start = split_data(df_all, PARAMS)

    # 2) Build empirical distributions (for SAA)
    proc_samples, all_samples = build_empirical_distributions(df_warm, PARAMS)

    # 3) Decide whether to run SAA at all
    run_saa = PARAMS.get("run_saa", False)
    # 4) Prepare output
    save_results = not PARAMS.get("debug_mode", False)
    if save_results:
        out = {
            "config": {
                "saa_scenarios":      PARAMS["saa_scenarios"],
                "NUM_HORIZONS":       PARAMS["NUM_HORIZONS"],
            },
            "horizons": []
        }

    # 5) Fast-debug override
    if PARAMS.get("debug_mode", False):
        PARAMS.update({
            "run_saa":      False,
            "gurobi_timelimit":   10,
            "gurobi_mipgap":      0.10,
            "NUM_HORIZONS":       1,
        })
        print("[DEBUG] Running in fast-debug mode")

    # 6) Train predictive models
    lasso_model       = train_lasso_predictor(df_warm, PARAMS)
    lasso_asym_model  = train_lasso_asym(df_warm, PARAMS)
    knn_model         = train_knn_predictor(df_warm, PARAMS)

    H            = PARAMS["planning_horizon_days"]
    NUM_HORIZONS = PARAMS["NUM_HORIZONS"]

    # 7) Define which methods to run & report
    tags = []
    if run_saa:
        tags.append("SAA")
    tags += ["Det", "Lasso", "LassoAsym", "KNN", "Oracle"]

    # helper for console output
    def brief(obj, kpi):
        if obj is None:
            return "plan=NA"
        return (
            f"plan={obj:.0f} | act={kpi['total_actual_cost']:.0f}, "
            f"sch={kpi['scheduled']}, rej={kpi['rejected']}, "
            f"OT={kpi['overtime_min_total']:.0f}m, "
            f"idle={kpi['idle_min_total']:.0f}m"
        )

    # 8) horizon loop
    for h in range(NUM_HORIZONS):
        horizon_end = horizon_start + timedelta(days=H - 1)
        mask = (
            (df_pool["actual_start"].dt.date >= horizon_start.date()) &
            (df_pool["actual_start"].dt.date <= horizon_end.date())
        )
        df_week = df_pool[mask]
        if df_week.empty:
            print(f"\n--- Horizon {h+1}: no data; stopping. ---")
            break

        print(
            f"\n--- Horizon {h+1}/{NUM_HORIZONS}  "
            f"{horizon_start.date()}–{horizon_end.date()}  "
            f"| ORs={df_week['operating_room'].nunique()} ---"
        )

        # a) compute capacity & select surgeries
        day_blocks      = compute_block_capacity(df_week, {**PARAMS, "_horizon_start_date": horizon_start})
        base            = select_surgeries(df_pool, horizon_start, PARAMS)
        surgeries_map   = {
            "Det":       (solve_deterministic_model, base),
            "Lasso":     (solve_predictive_model,  attach_pred(deepcopy(base), lasso_model, df_pool)),
            "LassoAsym": (solve_predictive_model,  attach_pred(deepcopy(base), lasso_asym_model, df_pool)),
            "KNN":       (solve_predictive_model,  attach_pred(deepcopy(base), knn_model, df_pool)),
            "Oracle":    (solve_clairvoyant_model,  base),
        }

        results = {}

        # b) optional SAA
        if run_saa:
            scen_mat  = sample_scenarios(base, proc_samples, all_samples, PARAMS)
            res_saa   = solve_saa_model(base, day_blocks, PARAMS, scen_mat)
            sch_saa   = extract_schedule(res_saa["model"], base, PARAMS, True)
            kpi_saa   = evaluate_schedule_actual_costs(sch_saa, day_blocks, PARAMS)
            results["SAA"] = {"res": res_saa, "kpi": kpi_saa}

        # c) run other methods
        for tag, (solver, surg) in surgeries_map.items():
            res = solver(surg, day_blocks, PARAMS)
            sch = extract_schedule(res["model"], surg, PARAMS, True)
            kpi = evaluate_schedule_actual_costs(sch, day_blocks, PARAMS)
            results[tag] = {"res": res, "kpi": kpi}

        # d) Console summary
        for tag in tags:
            obj = results[tag]["res"]["obj"]
            kpi = results[tag]["kpi"]
            print(f"{tag:<10}: {brief(obj, kpi)}")

        # e) Save per-horizon detail
        if save_results:
            append_horizon(
                out,
                horizon_idx = h + 1,
                start_date  = horizon_start.date(),
                results     = results
            )

        # f) advance to next
        df_pool      = df_pool[df_pool["actual_start"].dt.date > horizon_end.date()]
        horizon_start += timedelta(days=H)

    # 9) Aggregate & print summary
    def stats(xs):
        return (
            f"mean={np.mean(xs):.0f}, "
            f"median={np.median(xs):.0f}, "
            f"min={np.min(xs):.0f}, "
            f"max={np.max(xs):.0f}"
        )

    if save_results and out["horizons"]:
        print_console_summary(tags, out)
        save_detailed(out, PARAMS["output_file"])
        save_aggregate(
            out,
            PARAMS.get("aggregated_output_file", "agg_results.json"),
            methods = tags
        )

if __name__ == "__main__":
    main()