# =============================================================================
# IMPORTS
# =============================================================================
from datetime import timedelta
import numpy as np
import pandas as pd
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


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main() -> None:
    """Rolling‑horizon elective surgery scheduling experiment."""

    print("\n=== Elective Surgery Scheduling – 8 h blocks ===")

    # ------------------------------------------------------------------
    # 1)  Load data & split into warm‑up vs. rolling pool
    # ------------------------------------------------------------------
    df_all = load_data(PARAMS)
    df_warm, df_pool, horizon_start = split_data(df_all, PARAMS)

    # ------------------------------------------------------------------
    # 2)  Empirical duration distributions (for SAA sampling)
    # ------------------------------------------------------------------
    proc_samples, all_samples = build_empirical_distributions(df_warm, PARAMS)

    # ------------------------------------------------------------------
    # 3)  Optional fast‑debug override
    # ------------------------------------------------------------------
    if PARAMS.get("debug_mode", False):
        PARAMS.update(
            {
                "saa_scenarios": 1,
                "gurobi_timelimit": 10,
                "gurobi_mipgap": 0.10,
                "NUM_HORIZONS": 1,
            }
        )
        print("[DEBUG] Running in fast‑debug mode")

    # ------------------------------------------------------------------
    # 4)  Train predictors
    # ------------------------------------------------------------------
    lasso_model = train_lasso_predictor(df_warm, PARAMS)
    # lasso_asym_model = train_lasso_asym(df_warm, PARAMS)
    knn_model = train_knn_predictor(df_warm, PARAMS)

    H = PARAMS["planning_horizon_days"]
    NUM_HORIZONS = PARAMS["NUM_HORIZONS"]

    # ------------------------------------------------------------------
    # 5)  KPI containers
    # ------------------------------------------------------------------
    tags = (
        "SAA",
        "Det",
        "Lasso",
        # "LassoAsym",
        "KNN",
        "Oracle",
    )
    plan_totals = {t: [] for t in tags}
    actual_totals = {t: [] for t in tags}
    idle_totals = {t: [] for t in tags}
    ot_totals = {t: [] for t in tags}
    runtime_totals = {t: [] for t in tags}

    # ------------------------------------------------------------------
    # 6)  Rolling‑horizon loop
    # ------------------------------------------------------------------
    for h in range(NUM_HORIZONS):
        horizon_end = horizon_start + timedelta(days=H - 1)
        mask_week = (df_pool["actual_start"].dt.date >= horizon_start.date()) & (
            df_pool["actual_start"].dt.date <= horizon_end.date()
        )
        df_week = df_pool[mask_week]

        if df_week.empty:
            print(f"\n--- Horizon {h+1}: no data in pool; stopping. ---")
            break

        print(
            f"\n--- Horizon {h+1}/{NUM_HORIZONS}  "
            f"{horizon_start.date()}–{horizon_end.date()}  "
            f"| ORs={df_week['operating_room'].nunique()} ---"
        )

        # 6a) Available OR blocks per day
        day_blocks = compute_block_capacity(
            df_week, {**PARAMS, "_horizon_start_date": horizon_start}
        )

        # 6b) Surgery selection ONCE
        surgeries_base = select_surgeries(df_pool, horizon_start, PARAMS)
        surgeries_lasso = attach_pred(deepcopy(surgeries_base), lasso_model, df_pool)
        # surgeries_lasso_as = attach_pred(deepcopy(surgeries_base), lasso_asym_model, df_pool)
        surgeries_knn = attach_pred(deepcopy(surgeries_base), knn_model, df_pool)

        # 6c) Scenario sampling for SAA
        scen_mat = sample_scenarios(surgeries_base, proc_samples, all_samples, PARAMS)

        # 6d) Optimise each model --------------------------------------------
        res_saa = solve_saa_model(surgeries_base, day_blocks, PARAMS, scen_mat)
        res_det = solve_deterministic_model(surgeries_base, day_blocks, PARAMS)
        res_las = solve_predictive_model(surgeries_lasso, day_blocks, PARAMS)
        # res_las_a = solve_predictive_model(surgeries_lasso_as, day_blocks, PARAMS)
        res_knn = solve_predictive_model(surgeries_knn, day_blocks, PARAMS)
        res_orc = solve_clairvoyant_model(surgeries_base, day_blocks, PARAMS)

        # 6e) Evaluate schedules ---------------------------------------------
        sch_saa = extract_schedule(res_saa["model"], surgeries_base, PARAMS, True)
        sch_det = extract_schedule(res_det["model"], surgeries_base, PARAMS, True)
        sch_las = extract_schedule(res_las["model"], surgeries_lasso, PARAMS, True)
        # sch_las_a = extract_schedule(res_las_a["model"], surgeries_lasso_as, PARAMS, True)
        sch_knn = extract_schedule(res_knn["model"], surgeries_knn, PARAMS, True)
        sch_orc = extract_schedule(res_orc["model"], surgeries_base, PARAMS, True)

        kpi_saa = evaluate_schedule_actual_costs(sch_saa, day_blocks, PARAMS)
        kpi_det = evaluate_schedule_actual_costs(sch_det, day_blocks, PARAMS)
        kpi_las = evaluate_schedule_actual_costs(sch_las, day_blocks, PARAMS)
        # kpi_las_a = evaluate_schedule_actual_costs(sch_las_a, day_blocks, PARAMS)
        kpi_knn = evaluate_schedule_actual_costs(sch_knn, day_blocks, PARAMS)
        kpi_orc = evaluate_schedule_actual_costs(sch_orc, day_blocks, PARAMS)

        # 6f) Brief console output -------------------------------------------
        def brief(obj, kpi):
            val = "NA" if obj is None else f"{obj:.0f}"
            return (
                f"plan={val} | act={kpi['total_actual_cost']:.0f}, "
                f"sch={kpi['scheduled']}, rej={kpi['rejected']}, "
                f"OT={kpi['overtime_min_total']:.0f} m, "
                f"idle={kpi['idle_min_total']:.0f} m"
            )

        print("SAA        :", brief(res_saa["obj"], kpi_saa))
        print("Det        :", brief(res_det["obj"], kpi_det))
        print("Lasso      :", brief(res_las["obj"], kpi_las))
        # print("Lasso‑Asym :", brief(res_las_a["obj"], kpi_las_a))
        print("KNN        :", brief(res_knn["obj"], kpi_knn))
        print("Oracle     :", brief(res_orc["obj"], kpi_orc))

        # 6g) KPI collection --------------------------------------------------
        for tag, res, kpi in zip(
            tags,
            (
                res_saa,
                res_det,
                res_las,
                # res_las_a,
                res_knn,
                res_orc,
            ),
            (
                kpi_saa,
                kpi_det,
                kpi_las,
                # kpi_las_a,
                kpi_knn,
                kpi_orc,
            ),
        ):
            plan_totals[tag].append(res["obj"])
            actual_totals[tag].append(kpi["total_actual_cost"])
            idle_totals[tag].append(kpi["idle_min_total"])
            ot_totals[tag].append(kpi["overtime_min_total"])
            runtime_totals[tag].append(res["model"].Runtime)

        # 6h) Advance pool & horizon -----------------------------------------
        df_pool = df_pool[df_pool["actual_start"].dt.date > horizon_end.date()]
        horizon_start += timedelta(days=H)

    # ------------------------------------------------------------------
    # 7)  Aggregate summary
    # ------------------------------------------------------------------
    def stats(xs):
        return (
            f"mean={np.mean(xs):.0f}, "
            f"median={np.median(xs):.0f}, "
            f"min={np.min(xs):.0f}, "
            f"max={np.max(xs):.0f}"
        )

    print("\n=== Summary over horizons ===")
    for tag in tags:
        if plan_totals[tag]:
            print(f"{tag} planned obj : {stats(plan_totals[tag])}")
            print(f"{tag} actual obj  : {stats(actual_totals[tag])}")
            print(f"{tag} idle        : {stats(idle_totals[tag])}")
            print(f"{tag} overtime    : {stats(ot_totals[tag])}")
            print(f"{tag} runtime     : {stats(runtime_totals[tag])}")


if __name__ == "__main__":
    main()


"""
=== Summary over horizons ===
SAA planned obj : mean=216918, median=205798, min=194960, max=267710
SAA actual obj  : mean=275491, median=240465, min=233055, max=366145
SAA idle        : mean=7654, median=7368, min=6665, max=8528
SAA overtime    : mean=8956, median=8752, min=5355, max=12116
SAA runtime     : mean=1200, median=1200, min=1200, max=1200
Det planned obj : mean=285956, median=300085, min=214395, max=334125
Det actual obj  : mean=191097, median=186965, min=139005, max=239925
Det idle        : mean=5470, median=5273, min=4217, max=6784
Det overtime    : mean=9053, median=8949, min=5109, max=12769
Det runtime     : mean=5, median=6, min=3, max=6
Lasso planned obj : mean=54260, median=51295, min=33765, max=80245
Det runtime     : mean=5, median=6, min=3, max=6
Lasso planned obj : mean=54260, median=51295, min=33765, max=80245
Lasso actual obj  : mean=173215, median=173695, min=127615, max=228795
Lasso idle        : mean=4444, median=4462, min=2819, max=5668
Lasso overtime    : mean=7085, median=6617, min=4433, max=10110
Lasso actual obj  : mean=173215, median=173695, min=127615, max=228795
Lasso idle        : mean=4444, median=4462, min=2819, max=5668
Lasso overtime    : mean=7085, median=6617, min=4433, max=10110
Lasso overtime    : mean=7085, median=6617, min=4433, max=10110
Lasso runtime     : mean=275, median=8, min=6, max=1200
KNN planned obj : mean=36144, median=39320, min=18920, max=54485
KNN actual obj  : mean=152680, median=148205, min=122935, max=206240
KNN idle        : mean=4201, median=3838, min=3514, max=5211
KNN overtime    : mean=6271, median=6474, min=3432, max=9318
KNN runtime     : mean=818, median=859, min=90, max=1200
Oracle planned obj : mean=34830, median=33060, min=21255, max=55790
Oracle actual obj  : mean=34830, median=33060, min=21255, max=55790
Oracle idle        : mean=956, median=1527, min=0, max=1634
Oracle overtime    : mean=832, median=710, min=256, max=1682
Oracle runtime     : mean=501, median=44, min=26, max=1201
"""
