"""Main script for the elective surgery scheduling workflow."""

import logging
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np

from src.config import CONFIG
from src.constants import (
    DataColumns,
    JSONKeys,
    LoggingConstants,
)
from src.data_processing import (
    attach_pred,
    compute_block_capacity,
    load_data,
    split_data,
)
from src.output import (
    append_horizon_results,
    initialize_output_structure,
    print_console_summary,
    save_aggregated_results,
    save_detailed_results,
)
from src.predictors import (
    make_theta_predictor,
    train_knn_predictor,
    train_lasso_asym,
    train_lasso_predictor,
    train_xgboost_predictor,
    tune_knn_k_for_optimization,
)
from src.scheduling_utils import (
    evaluate_schedule_actual_costs,
    extract_schedule,
    select_surgeries,
)
from src.solver_utils import (
    solve_clairvoyant_model,
    solve_deterministic_model,
    solve_predictive_model,
    solve_saa_model,
    solve_knn_model,
    solve_utilization_maximization_model,
    solve_balanced_utilization_model,
)
from src.stochastic_utils import sample_scenarios, build_empirical_distributions

# --- Setup Project Root and Logger ---
ROOT_DIR = Path(__file__).resolve().parent

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format=LoggingConstants.LOG_FORMAT,
    datefmt=LoggingConstants.LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),  # Log to console
    ],
)
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)

logging.getLogger("gurobipy").setLevel(logging.WARNING)

# Type alias for solver functions used in the main loop
SolverFunctionType = Callable[
    [List[Dict[str, Any]], Dict[int, int], Any, Any], Dict[str, Any]
]


def _generate_brief_console_output(
    planned_objective: Optional[float], kpi_results: Dict[str, Any]
) -> str:
    """Generates a brief summary string for console output for one method's results."""
    if kpi_results is None or not kpi_results:
        return "plan=NA | kpi_results=NA"

    obj_str = f"{planned_objective:.0f}" if planned_objective is not None else "NA"

    # Safely get KPI values with defaults
    actual_cost = kpi_results.get("total_actual_cost", float("nan"))
    scheduled_count = kpi_results.get("scheduled_count", "NA")
    rejected_count = kpi_results.get("rejected_count", "NA")
    overtime_total = kpi_results.get("overtime_minutes_total", float("nan"))
    idle_total = kpi_results.get("idle_minutes_total", float("nan"))

    return (
        f"plan={obj_str} | "
        f"act={actual_cost:.0f}, "
        f"sch={scheduled_count}, rej={rejected_count}, "
        f"OT={overtime_total:.0f}m, idle={idle_total:.0f}m"
    )


def main() -> None:
    """Main function to run the elective surgery scheduling experiment."""

    logger.info("Starting elective surgery scheduling run")

    # --- 1. Load & Split Data ---
    logger.info("Loading and splitting data...")
    all_surgeries_df = load_data(CONFIG)
    df_warm_up, df_pool_initial, horizon_start_date_initial = split_data(
        all_surgeries_df, CONFIG
    )

    # Make mutable copies for the loop
    current_pool_df = df_pool_initial.copy()
    current_horizon_start_date = horizon_start_date_initial

    # --- 2. Build Empirical Distributions (for SAA) ---
    logger.info("Building empirical distributions for SAA...")
    procedure_duration_samples, all_pooled_duration_samples = (
        build_empirical_distributions(df_warm_up, CONFIG)
    )

    # --- 3. Configuration & Output Setup ---
    run_saa_flag = CONFIG.saa.run_saa
    is_debug_mode = CONFIG.debug_mode
    save_results_flag = not is_debug_mode

    # Update CONFIG if in debug mode
    if is_debug_mode:
        logger.warning(
            "Running in DEBUG MODE. Overriding some parameters for a faster run."
        )
        CONFIG.saa.run_saa = False  # Often SAA is slow, disable in debug
        CONFIG.gurobi.timelimit = CONFIG.gurobi.timelimit_debug
        CONFIG.gurobi.mipgap = CONFIG.gurobi.mipgap_debug
        CONFIG.data.num_horizons = CONFIG.gurobi.num_horizons_debug
        run_saa_flag = CONFIG.saa.run_saa  # Re-fetch after potential override

    # Initialize output structure
    output_data_structure: Dict[str, Any] = {}
    if save_results_flag:
        output_data_structure = initialize_output_structure(
            CONFIG.saa.scenarios, CONFIG.data.num_horizons
        )
        logger.info(
            f"Results will be saved. SAA Scenarios: {output_data_structure[JSONKeys.CONFIG].get(JSONKeys.CONFIG_SAA_SCENARIOS, 'N/A')}, "
            f"Num Horizons: {output_data_structure[JSONKeys.CONFIG].get(JSONKeys.CONFIG_NUM_HORIZONS, 'N/A')}."
        )
    else:
        logger.info(
            "Results will NOT be saved (debug_mode is ON or save_results is False)."
        )

    # --- 4. Train Predictive Models ---
    logger.info("Training predictive models...")
    lasso_model = train_lasso_predictor(df_warm_up, CONFIG)
    lasso_asym_model = train_lasso_asym(df_warm_up, CONFIG)
    knn_model = train_knn_predictor(df_warm_up, CONFIG)
    xgboost_model = train_xgboost_predictor(df_warm_up, CONFIG)

    # Tune K for KNN solver
    logger.info("Tuning K parameter for KNN optimization...")
    optimal_k = tune_knn_k_for_optimization(df_warm_up, CONFIG)
    CONFIG.ml.knn_neighbors = optimal_k
    logger.info(f"Optimal K for KNN optimization: {optimal_k}")

    # Load multiple theta predictors for different alpha values
    theta_predictors: Dict[str, Callable[[Dict[str, Any]], float]] = {}
    theta_dir = ROOT_DIR / Path(CONFIG.data.theta_path).parent
    theta_alpha_files = list(theta_dir.glob("theta_alpha_*.json"))

    if theta_alpha_files:
        logger.info(
            f"Found {len(theta_alpha_files)} theta files for different alpha values."
        )
        for theta_file in sorted(theta_alpha_files):
            # Extract alpha value from filename
            alpha_str = theta_file.stem.replace("theta_alpha_", "")
            try:
                alpha_val = float(alpha_str)
                predictor = make_theta_predictor(theta_file, all_surgeries_df)
                if predictor is not None:
                    theta_predictors[f"Integrated_{alpha_str}"] = predictor
                    logger.info(f"Created theta predictor for alpha={alpha_val}")
            except ValueError:
                logger.warning(
                    f"Could not parse alpha value from filename: {theta_file}"
                )
    else:
        # Fallback to original single theta file
        theta_json_file_path = ROOT_DIR / CONFIG.data.theta_path
        if theta_json_file_path.exists():
            logger.info(
                f"Found single theta JSON at {theta_json_file_path}. Creating theta predictor."
            )
            theta_predictor_func = make_theta_predictor(
                theta_json_file_path, all_surgeries_df
            )
            if theta_predictor_func is not None:
                theta_predictors["Integrated"] = theta_predictor_func
        else:
            logger.warning(f"No theta JSON files found. Skipping integrated models.")

    # --- 5. Define Methods to Run & Report ---
    method_tags_ordered: List[str] = []
    if run_saa_flag:
        method_tags_ordered.append("SAA")
    # Add all integrated models (sorted by alpha value for consistent ordering)
    for integrated_method_name in sorted(theta_predictors.keys()):
        method_tags_ordered.append(integrated_method_name)
    method_tags_ordered.extend(
        ["Det", "Lasso", "LassoAsym", "KNN", "XGBoost", "UtilMax", "BalancedUtil", "Oracle"]
    )

    planning_horizon_duration_days = CONFIG.data.planning_horizon_days
    num_horizons_to_run = CONFIG.data.num_horizons

    # --- 6. Horizon Loop ---
    logger.info(f"Starting simulation for {num_horizons_to_run} horizons.")
    for h_idx in range(num_horizons_to_run):
        current_horizon_end_date = current_horizon_start_date + timedelta(
            days=planning_horizon_duration_days - 1
        )

        # Filter df_pool for surgeries whose actual_start falls within the current horizon window
        horizon_mask = (
            current_pool_df[DataColumns.ACTUAL_START].dt.normalize().dt.date
            >= current_horizon_start_date.normalize().date()
        ) & (
            current_pool_df[DataColumns.ACTUAL_START].dt.normalize().dt.date
            <= current_horizon_end_date.normalize().date()
        )
        df_surgeries_for_this_horizon_ref = current_pool_df[horizon_mask]

        if df_surgeries_for_this_horizon_ref.empty:
            logger.info(
                f"\n--- Horizon {h_idx+1}: No surgery data available in pool for this period. Stopping. ---"
            )
            break

        num_operating_rooms_active = df_surgeries_for_this_horizon_ref[
            DataColumns.OPERATING_ROOM
        ].nunique()
        logger.info(
            f"\n--- Horizon {h_idx+1}/{num_horizons_to_run} | "
            f"{current_horizon_start_date.date()} to {current_horizon_end_date.date()} | "
            f"ORs Active (hist): {num_operating_rooms_active} ---"
        )

        # a) Compute daily block capacity and select surgeries for this horizon
        daily_block_capacities = compute_block_capacity(
            df_surgeries_for_this_horizon_ref, CONFIG, current_horizon_start_date
        )

        # Select surgeries from the current full pool that are candidates for this horizon
        base_surgeries_list_for_horizon = select_surgeries(
            current_pool_df, current_horizon_start_date, CONFIG
        )
        if not base_surgeries_list_for_horizon:
            logger.warning(
                f"Horizon {h_idx+1}: No surgeries selected for scheduling. Advancing to next horizon."
            )
            # Advance to next horizon period
            current_pool_df = current_pool_df[
                current_pool_df[DataColumns.ACTUAL_START].dt.normalize().dt.date
                > current_horizon_end_date.normalize().date()
            ]
            current_horizon_start_date += timedelta(days=planning_horizon_duration_days)
            continue

        # b) Prepare surgery lists for each predictive model
        surgeries_map_for_solvers: Dict[str, Tuple[Callable, List[Dict[str, Any]]]] = {
            "Det": (
                solve_deterministic_model,
                deepcopy(base_surgeries_list_for_horizon),
            ),
            "Lasso": (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    lasso_model,
                    current_pool_df,
                ),
            ),
            "LassoAsym": (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    lasso_asym_model,
                    current_pool_df,
                ),
            ),
            "Oracle": (
                solve_clairvoyant_model,
                deepcopy(base_surgeries_list_for_horizon),
            ),
        }

        if xgboost_model is not None:
            surgeries_map_for_solvers["XGBoost"] = (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    xgboost_model,
                    current_pool_df,
                ),
            )

            surgeries_map_for_solvers["UtilMax"] = (
                solve_utilization_maximization_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    xgboost_model,
                    current_pool_df,
                ),
            )

            surgeries_map_for_solvers["BalancedUtil"] = (
                solve_balanced_utilization_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    xgboost_model,
                    current_pool_df,
                ),
            )
        else:
            logger.warning(
                "Horizon %d: XGBoost model unavailable; skipping XGBoost, UtilMax, and BalancedUtil solvers.",
                h_idx + 1,
            )

        # Add all integrated models to surgeries_map_for_solvers
        for integrated_method_name, theta_predictor in theta_predictors.items():
            surgeries_map_for_solvers[integrated_method_name] = (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    theta_predictor,
                    current_pool_df,
                ),
            )

        # c) Store results for this horizon
        horizon_method_results: Dict[str, Dict[str, Any]] = {}

        # d) Optional SAA model run
        if "SAA" in method_tags_ordered:
            logger.info(f"Horizon {h_idx+1}: Running SAA model...")
            saa_scenario_matrix = sample_scenarios(
                base_surgeries_list_for_horizon,
                procedure_duration_samples,
                all_pooled_duration_samples,
                CONFIG,
            )
            saa_solver_result = solve_saa_model(
                base_surgeries_list_for_horizon,
                daily_block_capacities,
                CONFIG,
                saa_scenario_matrix,
            )
            if saa_solver_result and saa_solver_result.get("model"):
                saa_schedule_df = extract_schedule(
                    saa_solver_result["model"], base_surgeries_list_for_horizon, True
                )
                saa_kpi_results = evaluate_schedule_actual_costs(
                    saa_schedule_df, daily_block_capacities, CONFIG
                )
                horizon_method_results["SAA"] = {
                    "res": saa_solver_result,
                    "kpi": saa_kpi_results,
                }
            else:
                logger.warning(
                    f"SAA model failed to solve for horizon {h_idx+1}. Skipping SAA results."
                )
                horizon_method_results["SAA"] = {
                    "res": {"obj": None},
                    "kpi": {},
                }

        # Handle KNN separately
        if "KNN" in method_tags_ordered:
            logger.info(f"Horizon {h_idx+1}: Running KNN model...")
            knn_solver_result = solve_knn_model(
                deepcopy(base_surgeries_list_for_horizon),
                daily_block_capacities,
                CONFIG,
                historical_data=df_warm_up,  # Pass warm-up data as historical data
            )

            if knn_solver_result and knn_solver_result.get("model"):
                knn_schedule_df = extract_schedule(
                    knn_solver_result["model"], base_surgeries_list_for_horizon, True
                )
                knn_kpi_results = evaluate_schedule_actual_costs(
                    knn_schedule_df, daily_block_capacities, CONFIG
                )
                
                # Log KNN debug info if available
                if "knn_debug" in knn_solver_result:
                    debug_info = knn_solver_result["knn_debug"]
                    logger.debug(
                        "KNN details | K=%s | Unique neighbors=%s | Coverage=%.1f%% | Scenario mean=%.1f",
                        debug_info["k_used"],
                        debug_info["unique_neighbors"],
                        debug_info["neighbor_coverage"] * 100,
                        debug_info["scenario_mean"],
                    )
                
                horizon_method_results["KNN"] = {
                    "res": knn_solver_result,
                    "kpi": knn_kpi_results,
                }
            else:
                logger.warning(f"KNN model failed to solve for horizon {h_idx+1}. Skipping KNN results.")
                horizon_method_results["KNN"] = {
                    "res": {"obj": None},
                    "kpi": {},
                }

        # e) Run other deterministic/predictive models
        for method_tag, (
            solver_func,
            surgeries_for_method,
        ) in surgeries_map_for_solvers.items():
            if not surgeries_for_method:
                logger.warning(
                    f"Horizon {h_idx+1}: No surgeries to process for method {method_tag}. Skipping."
                )
                horizon_method_results[method_tag] = {"res": {"obj": None}, "kpi": {}}
                continue

            logger.info(f"Horizon {h_idx+1}: Running {method_tag} model...")

            # The solver functions take (surgeries, day_blocks, config)
            solver_result = solver_func(
                surgeries_for_method, daily_block_capacities, CONFIG
            )

            if solver_result and solver_result.get("model"):
                schedule_df = extract_schedule(
                    solver_result["model"], surgeries_for_method, True
                )
                kpi_results = evaluate_schedule_actual_costs(
                    schedule_df, daily_block_capacities, CONFIG
                )
                horizon_method_results[method_tag] = {
                    "res": solver_result,
                    "kpi": kpi_results,
                }
            else:
                logger.warning(
                    f"{method_tag} model failed to solve for horizon {h_idx+1}. Skipping its results."
                )
                horizon_method_results[method_tag] = {
                    "res": {"obj": None},
                    "kpi": {},
                }

        # f) Console summary for this horizon
        logger.info(f"--- Horizon {h_idx+1} Summary ---")
        for tag_to_print in method_tags_ordered:
            if tag_to_print in horizon_method_results:
                planned_obj = horizon_method_results[tag_to_print]["res"].get("obj")
                kpis = horizon_method_results[tag_to_print]["kpi"]
                logger.info(
                    f"  {tag_to_print:<12}: {_generate_brief_console_output(planned_obj, kpis)}"
                )
            else:
                logger.info(f"  {tag_to_print:<12}: Results not available.")

        # g) Save per-horizon detailed results
        if save_results_flag:
            total_blocks = sum(daily_block_capacities.values())
            append_horizon_results(
                output_data_structure,
                horizon_index=h_idx + 1,
                horizon_start_date=current_horizon_start_date.date(),
                total_blocks=total_blocks,
                per_method_results=horizon_method_results,
            )

        # h) Advance to the next planning period
        current_pool_df = current_pool_df[
            current_pool_df[DataColumns.ACTUAL_START].dt.normalize().dt.date
            > current_horizon_end_date.normalize().date()
        ]
        current_horizon_start_date += timedelta(days=planning_horizon_duration_days)

    # --- 7. Aggregate & Save Final Results ---
    if save_results_flag and output_data_structure.get(JSONKeys.HORIZONS):
        logger.info("Aggregating results and saving output files...")
        print_console_summary(method_tags_ordered, output_data_structure)

        detailed_output_path = CONFIG.data.output_file
        save_detailed_results(output_data_structure, detailed_output_path)

        aggregated_output_path = CONFIG.data.aggregated_output_file
        save_aggregated_results(
            output_data_structure,
            aggregated_output_path,
            method_tags_to_aggregate=method_tags_ordered,
        )
    elif not output_data_structure.get(JSONKeys.HORIZONS):
        logger.info("No horizons were processed. Skipping final summary and saving.")
    else:
        logger.info(
            "Summary and saving skipped as per configuration (debug_mode or save_results=False)."
        )

    logger.info(
        "Cost configuration | Overtime: $%.2f/min | Idle: $%.2f/min",
        CONFIG.costs.overtime_per_min,
        CONFIG.costs.idle_per_min,
    )
    logger.info("Experiment finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in main execution: {e}", exc_info=True
        )
