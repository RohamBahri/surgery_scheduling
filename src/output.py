"""
Handles the formatting, summarization, and saving of experiment results
for the surgery scheduling application.
"""
import json
import logging
from datetime import date
from pathlib import Path # For type hinting and path operations
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_LOGGER_NAME,
    JSON_KEY_ACTUAL_COST,
    JSON_KEY_AGGREGATE,
    JSON_KEY_AVG_ACTUAL_COST,
    JSON_KEY_AVG_IDLE_MIN,
    JSON_KEY_AVG_OVERTIME_MIN,
    JSON_KEY_AVG_PLANNED_COST,
    JSON_KEY_AVG_RUNTIME_SEC,
    JSON_KEY_CONFIG,
    JSON_KEY_HORIZON_INDEX,
    JSON_KEY_HORIZONS,
    JSON_KEY_IDLE_MIN,
    JSON_KEY_OVERTIME_MIN,
    JSON_KEY_PLANNED_COST,
    JSON_KEY_REJECTED_COUNT,
    JSON_KEY_RUNTIME_SEC,
    JSON_KEY_SCHEDULED_COUNT,
    JSON_KEY_START_DATE,
    JSON_KEY_STATUS,
    JSON_KEY_TOTAL_ACTUAL_COST,
    JSON_KEY_TOTAL_IDLE_MIN,
    JSON_KEY_TOTAL_OVERTIME_MIN,
    JSON_KEY_TOTAL_PLANNED_COST,
)

# Setup logger
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def initialize_output_structure(
    num_saa_scenarios: int, num_horizons_planned: int
) -> Dict[str, Any]:
    """Initializes the main dictionary structure for storing results.

    Args:
        num_saa_scenarios: Number of SAA scenarios configured for the run.
        num_horizons_planned: Total number of horizons planned for the experiment.

    Returns:
        An initialized dictionary for collecting results.
    """
    output_dict = {
        JSON_KEY_CONFIG: {
            "saa_scenarios_configured": num_saa_scenarios,
            "num_horizons_planned": num_horizons_planned,
            # Add other global config items if needed
        },
        JSON_KEY_HORIZONS: [], # List to store per-horizon results
    }
    logger.debug("Initialized output data structure.")
    return output_dict


def append_horizon_results(
    output_data_struct: Dict[str, Any],
    horizon_index: int,
    horizon_start_date: date,
    per_method_results: Dict[str, Dict[str, Any]], # e.g. {"SAA": {"res":..., "kpi":...}, "Det": ...}
) -> None:
    """Appends results for a single horizon to the main output structure.

    Args:
        output_data_struct: The main dictionary holding all results.
        horizon_index: The 1-based index of the current horizon.
        horizon_start_date: The start date of the current horizon.
        per_method_results: A dictionary where keys are method tags (e.g., "SAA",
            "Det") and values are dictionaries containing 'res' (solver result)
            and 'kpi' (evaluated KPIs) for that method.
    """
    horizon_entry: Dict[str, Any] = {
        JSON_KEY_HORIZON_INDEX: horizon_index,
        JSON_KEY_START_DATE: horizon_start_date.isoformat(),
    }

    for method_tag, results_dict in per_method_results.items():
        solver_result = results_dict.get("res", {}) # Gurobi model result
        kpi_results = results_dict.get("kpi", {})   # Evaluated schedule costs/stats

        # Gracefully handle missing keys, e.g. if a model failed to solve
        model_status = solver_result.get("status", "N/A")
        # Gurobi model object is usually under 'model' key in solver_result
        gurobi_model_obj = solver_result.get("model")
        model_runtime = getattr(gurobi_model_obj, "Runtime", -1.0) if gurobi_model_obj else -1.0
        
        # Ensure planned objective value is numeric, or None if not available
        planned_obj_val = solver_result.get("obj")
        if planned_obj_val is not None:
            try:
                planned_obj_val = float(planned_obj_val)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert planned objective '{planned_obj_val}' to float for method {method_tag}, horizon {horizon_index}. Storing as is.")
        

        horizon_entry[method_tag] = {
            JSON_KEY_STATUS: model_status,
            JSON_KEY_PLANNED_COST: planned_obj_val,
            JSON_KEY_ACTUAL_COST: kpi_results.get("total_actual_cost"), # From evaluate_schedule
            JSON_KEY_OVERTIME_MIN: kpi_results.get("overtime_minutes_total"),
            JSON_KEY_IDLE_MIN: kpi_results.get("idle_minutes_total"),
            JSON_KEY_SCHEDULED_COUNT: kpi_results.get("scheduled_count"),
            JSON_KEY_REJECTED_COUNT: kpi_results.get("rejected_count"),
            JSON_KEY_RUNTIME_SEC: model_runtime,
        }
    
    output_data_struct[JSON_KEY_HORIZONS].append(horizon_entry)
    logger.debug(f"Appended results for horizon {horizon_index} ({horizon_start_date}).")


def _generate_statistic_string(data_array: List[float]) -> str:
    """Helper to create a string summary of mean, median, min, max."""
    if not data_array or all(x is None for x in data_array): # Handle empty or all None
        return "data=NA"
    
    # Filter out None values before calculating numpy stats
    valid_data = [x for x in data_array if x is not None]
    if not valid_data: # If filtering results in empty list
        return "data=NA (all None)"

    return (
        f"mean={np.mean(valid_data):.0f}, median={np.median(valid_data):.0f}, "
        f"min={np.min(valid_data):.0f}, max={np.max(valid_data):.0f} "
        f"(n={len(valid_data)})" # Add count of valid data points
    )

def print_console_summary(
    method_tags_to_summarize: List[str], output_data_struct: Dict[str, Any]
) -> None:
    """Prints a summary of results over all horizons to the console.

    Args:
        method_tags_to_summarize: List of method tags (e.g., ["SAA", "Det"])
            for which to print summaries.
        output_data_struct: The main results dictionary containing horizon data.
    """
    logger.info("\n" + "=" * 20 + " Summary Over Horizons " + "=" * 20)
    
    horizons_data = output_data_struct.get(JSON_KEY_HORIZONS, [])
    if not horizons_data:
        logger.info("No horizon data available to summarize.")
        return

    for method_tag in method_tags_to_summarize:
        # Extract data for this method, handling potential missing keys or None values
        planned_costs = [h.get(method_tag, {}).get(JSON_KEY_PLANNED_COST) for h in horizons_data]
        actual_costs = [h.get(method_tag, {}).get(JSON_KEY_ACTUAL_COST) for h in horizons_data]
        idle_minutes = [h.get(method_tag, {}).get(JSON_KEY_IDLE_MIN) for h in horizons_data]
        overtime_minutes = [h.get(method_tag, {}).get(JSON_KEY_OVERTIME_MIN) for h in horizons_data]
        runtimes_sec = [h.get(method_tag, {}).get(JSON_KEY_RUNTIME_SEC) for h in horizons_data]

        # Log using info for summaries
        logger.info(f"--- Method: {method_tag} ---")
        logger.info(f"  Planned Objective : {_generate_statistic_string(planned_costs)}")
        logger.info(f"  Actual Cost       : {_generate_statistic_string(actual_costs)}")
        logger.info(f"  Idle Minutes      : {_generate_statistic_string(idle_minutes)}")
        logger.info(f"  Overtime Minutes  : {_generate_statistic_string(overtime_minutes)}")
        logger.info(f"  Runtime (sec)     : {_generate_statistic_string(runtimes_sec)}")
    logger.info("=" * (40 + len(" Summary Over Horizons ")) + "\n")


def save_detailed_results(
    output_data_struct: Dict[str, Any], output_file_path: Union[str, Path]
) -> None:
    """Saves the detailed per-horizon results to a JSON file.

    Args:
        output_data_struct: The main results dictionary.
        output_file_path: Path to save the JSON file.
    """
    path_obj = Path(output_file_path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(path_obj, "w") as f:
            json.dump(output_data_struct, f, indent=2, cls=NpEncoder) # Use NpEncoder for numpy types
        logger.info(f"Detailed results successfully written to: {path_obj}")
    except IOError as e:
        logger.error(f"IOError saving detailed results to {path_obj}: {e}", exc_info=True)
    except TypeError as e:
        logger.error(f"TypeError during JSON serialization for detailed results: {e}. Check for non-serializable types.", exc_info=True)


def save_aggregated_results(
    output_data_struct: Dict[str, Any],
    aggregated_output_file_path: Union[str, Path],
    method_tags_to_aggregate: List[str],
) -> None:
    """Calculates and saves aggregated (sum, average) results to a JSON file.

    Args:
        output_data_struct: The main results dictionary.
        aggregated_output_file_path: Path to save the aggregated JSON file.
        method_tags_to_aggregate: List of method tags to include in aggregation.
    """
    aggregated_data_struct: Dict[str, Any] = {
        JSON_KEY_CONFIG: output_data_struct.get(JSON_KEY_CONFIG, {}),
        JSON_KEY_AGGREGATE: {},
    }
    
    horizons_data = output_data_struct.get(JSON_KEY_HORIZONS, [])
    num_horizons_completed = len(horizons_data)

    if num_horizons_completed == 0:
        logger.warning(
            "No horizon data to aggregate. Aggregated results file will be minimal."
        )
        # Still save the structure with config, but aggregate part will be empty
        path_obj = Path(aggregated_output_file_path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w") as f:
                json.dump(aggregated_data_struct, f, indent=2)
            logger.info(f"Saved minimal aggregated results (no horizon data) to: {path_obj}")
        except IOError as e:
            logger.error(f"IOError saving minimal aggregated results to {path_obj}: {e}", exc_info=True)
        return

    for method_tag in method_tags_to_aggregate:
        # Helper to extract values for a field, converting None to 0 for sum/mean
        def get_values_for_field(field_key: str) -> List[float]:
            raw_values = [h.get(method_tag, {}).get(field_key) for h in horizons_data]
            # Convert None to 0 for summation/mean, or filter them out if preferred.
            # Current original logic implies Nones are treated as 0 for sum, and affect count for mean.
            # To be safe, let's filter Nones for sum and mean calculations on valid data points.
            return [float(v) for v in raw_values if v is not None]


        planned_costs_vals = get_values_for_field(JSON_KEY_PLANNED_COST)
        actual_costs_vals = get_values_for_field(JSON_KEY_ACTUAL_COST)
        overtime_vals = get_values_for_field(JSON_KEY_OVERTIME_MIN)
        idle_vals = get_values_for_field(JSON_KEY_IDLE_MIN)
        runtime_vals = get_values_for_field(JSON_KEY_RUNTIME_SEC)
        
        # Calculate sums and averages, handling cases where no valid data exists for a metric
        aggregated_metrics_for_method = {
            JSON_KEY_TOTAL_PLANNED_COST: sum(planned_costs_vals) if planned_costs_vals else None,
            JSON_KEY_AVG_PLANNED_COST: np.mean(planned_costs_vals) if planned_costs_vals else None,
            JSON_KEY_TOTAL_ACTUAL_COST: sum(actual_costs_vals) if actual_costs_vals else None,
            JSON_KEY_AVG_ACTUAL_COST: np.mean(actual_costs_vals) if actual_costs_vals else None,
            JSON_KEY_TOTAL_OVERTIME_MIN: sum(overtime_vals) if overtime_vals else None,
            JSON_KEY_AVG_OVERTIME_MIN: np.mean(overtime_vals) if overtime_vals else None,
            JSON_KEY_TOTAL_IDLE_MIN: sum(idle_vals) if idle_vals else None,
            JSON_KEY_AVG_IDLE_MIN: np.mean(idle_vals) if idle_vals else None,
            JSON_KEY_AVG_RUNTIME_SEC: np.mean(runtime_vals) if runtime_vals else None,
        }
        aggregated_data_struct[JSON_KEY_AGGREGATE][method_tag] = aggregated_metrics_for_method

    path_obj = Path(aggregated_output_file_path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(aggregated_data_struct, f, indent=2, cls=NpEncoder) # Use NpEncoder
        logger.info(f"Aggregated results successfully written to: {path_obj}")
    except IOError as e:
        logger.error(f"IOError saving aggregated results to {path_obj}: {e}", exc_info=True)
    except TypeError as e:
        logger.error(f"TypeError during JSON serialization for aggregated results: {e}. Check for non-serializable types.", exc_info=True)


class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types.
    
    Handles common NumPy types like int64, float64, and ndarray by converting
    them to their Python equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp): # Handle pandas Timestamps if they appear
            return obj.isoformat()
        if pd.isna(obj): # Handle pd.NA, which is not directly JSON serializable
             return None
        return super(NpEncoder, self).default(obj)