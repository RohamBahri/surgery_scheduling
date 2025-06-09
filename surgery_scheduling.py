# surgery_scheduling.py
"""
Main script for running elective surgery scheduling experiments.

This script loads and processes surgery data, trains predictive models,
simulates scheduling over multiple horizons using various methods (deterministic,
predictive, SAA, clairvoyant), evaluates the schedules, and outputs results.
"""
import logging
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np


from src.config import PARAMS
from src.constants import (
    COL_ACTUAL_START,
    COL_OPERATING_ROOM,
    DEFAULT_LOGGER_NAME,
    HORIZON_START_DATE_PARAM_KEY,
    JSON_KEY_CONFIG,
    JSON_KEY_CONFIG_NUM_HORIZONS,
    JSON_KEY_CONFIG_SAA_SCENARIOS,
    JSON_KEY_HORIZONS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
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
    # solve_saa_model,
    solve_saa_benders as solve_saa_model,
)
from src.stochastic_utils import sample_scenarios, build_empirical_distributions

# --- Setup Project Root and Logger ---
ROOT_DIR = Path(__file__).resolve().parent

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),  # Log to console
        # logging.FileHandler("surgery_scheduling.log") # Optional: log to file
    ],
)
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

logging.getLogger("gurobipy").setLevel(logging.WARNING)

# Type alias for solver functions used in the main loop
SolverFunctionType = Callable[
    [List[Dict[str, Any]], Dict[int, int], Dict[str, Any], Any], Dict[str, Any]
]


def _generate_brief_console_output(
    planned_objective: Optional[float], kpi_results: Dict[str, Any]
) -> str:
    """Generates a brief summary string for console output for one method's results."""
    if kpi_results is None or not kpi_results:  # Check if kpi_results is empty or None
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

    logger.info(
        "\n" + "=" * 30 + " Elective Surgery Scheduling – 8h Blocks " + "=" * 30
    )

    # --- 1. Load & Split Data ---
    logger.info("Loading and splitting data...")
    all_surgeries_df = load_data(PARAMS)
    df_warm_up, df_pool_initial, horizon_start_date_initial = split_data(
        all_surgeries_df, PARAMS
    )

    # Make mutable copies for the loop
    current_pool_df = df_pool_initial.copy()
    current_horizon_start_date = horizon_start_date_initial

    # --- 2. Build Empirical Distributions (for SAA) ---
    logger.info("Building empirical distributions for SAA...")
    procedure_duration_samples, all_pooled_duration_samples = (
        build_empirical_distributions(df_warm_up, PARAMS)
    )

    # --- 3. Configuration & Output Setup ---
    run_saa_flag = PARAMS.get("run_saa", False)
    is_debug_mode = PARAMS.get("debug_mode", False)
    save_results_flag = not is_debug_mode

    # Update PARAMS if in debug mode (original logic)
    if is_debug_mode:
        logger.warning(
            "Running in DEBUG MODE. Overriding some parameters for a faster run."
        )
        PARAMS.update(
            {
                "run_saa": False,  # Often SAA is slow, disable in debug
                "gurobi_timelimit": PARAMS.get(
                    "gurobi_timelimit_debug", 10
                ),  # Use specific debug keys or defaults
                "gurobi_mipgap": PARAMS.get("gurobi_mipgap_debug", 0.10),
                "NUM_HORIZONS": PARAMS.get(
                    "num_horizons_debug", 1
                ),  # Control number of horizons in debug
            }
        )
        run_saa_flag = PARAMS["run_saa"]  # Re-fetch after potential override
        # Gurobi params in PARAMS will be picked up by set_gurobi_model_parameters

    # Initialize output structure
    output_data_structure: Dict[str, Any] = {}
    if save_results_flag:
        output_data_structure = initialize_output_structure(
            PARAMS.get("saa_scenarios", 0), PARAMS.get("NUM_HORIZONS", 0)
        )
        logger.info(
            f"Results will be saved. SAA Scenarios: {output_data_structure[JSON_KEY_CONFIG].get(JSON_KEY_CONFIG_SAA_SCENARIOS, 'N/A')}, "
            f"Num Horizons: {output_data_structure[JSON_KEY_CONFIG].get(JSON_KEY_CONFIG_NUM_HORIZONS, 'N/A')}."
        )
    else:
        logger.info(
            "Results will NOT be saved (debug_mode is ON or save_results is False)."
        )

    # --- 4. Train Predictive Models ---
    logger.info("Training predictive models...")
    lasso_model = train_lasso_predictor(df_warm_up, PARAMS)
    lasso_asym_model = train_lasso_asym(df_warm_up, PARAMS)
    knn_model = train_knn_predictor(df_warm_up, PARAMS)

    theta_predictor_func: Optional[Callable[[Dict[str, Any]], float]] = None
    theta_json_file_path = ROOT_DIR / PARAMS["theta_path"]
    if theta_json_file_path.exists():
        logger.info(
            f"Found theta JSON at {theta_json_file_path}. Creating theta predictor."
        )
        # Pass all_surgeries_df for scaling reference, as it was used in training run_integrated.py
        theta_predictor_func = make_theta_predictor(
            theta_json_file_path, all_surgeries_df
        )
    else:
        logger.warning(
            f"Theta JSON file {theta_json_file_path} not found. Skipping 'Integrated' (theta) model."
        )

    # --- 5. Define Methods to Run & Report ---
    # These tags are used for structuring results and console output
    # Order here defines console output order.
    method_tags_ordered: List[str] = []
    if run_saa_flag:
        method_tags_ordered.append("SAA")
    if theta_predictor_func is not None:
        method_tags_ordered.append("Integrated")
    method_tags_ordered.extend(["Det", "Lasso", "LassoAsym", "KNN", "Oracle"])

    planning_horizon_duration_days = PARAMS["planning_horizon_days"]
    num_horizons_to_run = PARAMS["NUM_HORIZONS"]  # Can be overridden by debug mode

    # --- 6. Horizon Loop ---
    logger.info(f"Starting simulation for {num_horizons_to_run} horizons.")
    for h_idx in range(num_horizons_to_run):
        current_horizon_end_date = current_horizon_start_date + timedelta(
            days=planning_horizon_duration_days - 1
        )

        # Filter df_pool for surgeries whose actual_start falls within the current horizon window
        # Using normalized dates for robust comparison
        horizon_mask = (
            current_pool_df[COL_ACTUAL_START].dt.normalize().dt.date
            >= current_horizon_start_date.normalize().date()
        ) & (
            current_pool_df[COL_ACTUAL_START].dt.normalize().dt.date
            <= current_horizon_end_date.normalize().date()
        )
        df_surgeries_for_this_horizon_ref = current_pool_df[
            horizon_mask
        ]  # This is just for OR count

        if df_surgeries_for_this_horizon_ref.empty:
            logger.info(
                f"\n--- Horizon {h_idx+1}: No surgery data available in pool for this period. Stopping. ---"
            )
            break

        num_operating_rooms_active = df_surgeries_for_this_horizon_ref[
            COL_OPERATING_ROOM
        ].nunique()
        logger.info(
            f"\n--- Horizon {h_idx+1}/{num_horizons_to_run} | "
            f"{current_horizon_start_date.date()} to {current_horizon_end_date.date()} | "
            f"ORs Active (hist): {num_operating_rooms_active} ---"
        )

        # a) Compute daily block capacity and select surgeries for this horizon
        # Pass the original horizon start date to compute_block_capacity via params
        params_for_capacity = {
            **PARAMS,
            HORIZON_START_DATE_PARAM_KEY: current_horizon_start_date,
        }
        daily_block_capacities = compute_block_capacity(
            df_surgeries_for_this_horizon_ref, params_for_capacity
        )

        # Select surgeries from the *current full pool* that are *candidates* for this horizon
        # `select_surgeries` will internally filter by date based on `current_horizon_start_date`
        base_surgeries_list_for_horizon = select_surgeries(
            current_pool_df, current_horizon_start_date, PARAMS
        )
        if not base_surgeries_list_for_horizon:
            logger.warning(
                f"Horizon {h_idx+1}: No surgeries selected for scheduling. Advancing to next horizon."
            )
            # Advance to next horizon period
            current_pool_df = current_pool_df[
                current_pool_df[COL_ACTUAL_START].dt.normalize().dt.date
                > current_horizon_end_date.normalize().date()
            ]  # Remove processed part from pool
            current_horizon_start_date += timedelta(days=planning_horizon_duration_days)
            continue

        # b) Prepare surgery lists for each predictive model
        # `attach_pred` modifies the list in-place (or returns modified list)
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
            "KNN": (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    knn_model,
                    current_pool_df,
                ),
            ),
            "Oracle": (
                solve_clairvoyant_model,
                deepcopy(base_surgeries_list_for_horizon),
            ),
        }

        if theta_predictor_func is not None:
            surgeries_map_for_solvers["Integrated"] = (
                solve_predictive_model,
                attach_pred(
                    deepcopy(base_surgeries_list_for_horizon),
                    theta_predictor_func,
                    current_pool_df,
                ),
            )

        # c) Store results for this horizon
        horizon_method_results: Dict[str, Dict[str, Any]] = {}

        # d) Optional SAA model run
        if "SAA" in method_tags_ordered:  # Check if SAA is actually supposed to run
            logger.info(f"Horizon {h_idx+1}: Running SAA model...")
            # SAA uses its own scenario matrix, not a single prediction.
            # Base surgeries for SAA should be the same as for deterministic.
            # Ensure `base_surgeries_list_for_horizon` has 'booked_min' and 'proc_id'
            saa_scenario_matrix = sample_scenarios(
                base_surgeries_list_for_horizon,  # Use the original list for SAA
                procedure_duration_samples,
                all_pooled_duration_samples,
                PARAMS,
            )
            # SAA model solver function signature needs scenario_matrix
            # Modifying solve_saa_model to accept it as last arg if not already.
            # solve_saa_model(surgeries, day_blocks, params, scen_mat)
            saa_solver_result = solve_saa_model(
                base_surgeries_list_for_horizon,
                daily_block_capacities,
                PARAMS,
                saa_scenario_matrix,
            )
            if saa_solver_result and saa_solver_result.get("model"):
                saa_schedule_df = extract_schedule(
                    saa_solver_result["model"], base_surgeries_list_for_horizon, True
                )
                saa_kpi_results = evaluate_schedule_actual_costs(
                    saa_schedule_df, daily_block_capacities, PARAMS
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
                }  # Placeholder for missing result

        # e) Run other deterministic/predictive models
        for method_tag, (
            solver_func,
            surgeries_for_method,
        ) in surgeries_map_for_solvers.items():
            if (
                not surgeries_for_method
            ):  # Skip if attach_pred returned empty (e.g., model was None)
                logger.warning(
                    f"Horizon {h_idx+1}: No surgeries to process for method {method_tag}. Skipping."
                )
                horizon_method_results[method_tag] = {"res": {"obj": None}, "kpi": {}}
                continue

            logger.info(f"Horizon {h_idx+1}: Running {method_tag} model...")
            # Ensure solver_func is not None (e.g. if a predictor failed to train)
            # This check should be implicitly handled if surgeries_for_method is correctly prepared

            # The solver functions might need **kwargs if they accept lp_relax etc.
            # For now, assuming they take (surgeries, day_blocks, params)
            solver_result = solver_func(
                surgeries_for_method, daily_block_capacities, PARAMS
            )

            if solver_result and solver_result.get("model"):
                schedule_df = extract_schedule(
                    solver_result["model"], surgeries_for_method, True
                )
                kpi_results = evaluate_schedule_actual_costs(
                    schedule_df, daily_block_capacities, PARAMS
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
                }  # Placeholder

        # f) Console summary for this horizon
        logger.info(f"--- Horizon {h_idx+1} Summary ---")
        for tag_to_print in method_tags_ordered:
            if tag_to_print in horizon_method_results:
                planned_obj = horizon_method_results[tag_to_print]["res"].get("obj")
                kpis = horizon_method_results[tag_to_print]["kpi"]
                logger.info(
                    f"  {tag_to_print:<10}: {_generate_brief_console_output(planned_obj, kpis)}"
                )
            else:
                logger.info(f"  {tag_to_print:<10}: Results not available.")

        # g) Save per-horizon detailed results
        if save_results_flag:
            append_horizon_results(
                output_data_structure,
                horizon_index=h_idx + 1,
                horizon_start_date=current_horizon_start_date.date(),
                per_method_results=horizon_method_results,
            )

        # h) Advance to the next planning period
        # Remove processed part of the pool (surgeries up to current_horizon_end_date)
        current_pool_df = current_pool_df[
            current_pool_df[COL_ACTUAL_START].dt.normalize().dt.date
            > current_horizon_end_date.normalize().date()
        ]
        current_horizon_start_date += timedelta(days=planning_horizon_duration_days)

    # --- 7. Aggregate & Save Final Results ---
    if save_results_flag and output_data_structure.get(
        JSON_KEY_HORIZONS
    ):  # Check if any horizons were processed
        logger.info("Aggregating results and saving output files...")
        print_console_summary(method_tags_ordered, output_data_structure)

        detailed_output_path = PARAMS.get("output_file", "outputs/results.json")
        save_detailed_results(output_data_structure, detailed_output_path)

        aggregated_output_path = PARAMS.get(
            "aggregated_output_file", "outputs/agg_results.json"
        )
        save_aggregated_results(
            output_data_structure,
            aggregated_output_path,
            method_tags_to_aggregate=method_tags_ordered,
        )
    elif not output_data_structure.get(JSON_KEY_HORIZONS):
        logger.info("No horizons were processed. Skipping final summary and saving.")
    else:  # Not save_results_flag
        logger.info(
            "Summary and saving skipped as per configuration (debug_mode or save_results=False)."
        )

    logger.info("\n" + "=" * 30 + " Experiment Finished " + "=" * 30)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in main execution: {e}", exc_info=True
        )

"""
project/
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── data_processing.py
│   ├── solver_utils.py
│   ├── predictors.py
│   ├── scheduling_utils.py
│   └── stochastic_utils.py
│   └── output.py
│   └── __init__.py
├── integrated/
│   ├── run_integrated.py
│   ├── master_problem.py
│   ├── sub_problem.py
│   ├── callbacks.py
│   └── __init__.py
└── surgery_scheduling.py
"""
