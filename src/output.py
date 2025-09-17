"""
Handles the formatting, summarization, and saving of experiment results
for the surgery scheduling application.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.constants import JSONKeys, LoggingConstants

# Setup logger
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


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
        JSONKeys.CONFIG: {
            JSONKeys.CONFIG_SAA_SCENARIOS: num_saa_scenarios,
            JSONKeys.CONFIG_NUM_HORIZONS: num_horizons_planned,
        },
        JSONKeys.HORIZONS: [],
    }
    logger.debug("Initialized output data structure.")
    return output_dict


def append_horizon_results(
    output_data_struct: Dict[str, Any],
    horizon_index: int,
    horizon_start_date: date,
    per_method_results: Dict[str, Dict[str, Any]],
    total_blocks: int = 0,  
) -> None:
    """Appends results for a single horizon to the main output structure.

    Args:
        output_data_struct: The main dictionary holding all results.
        horizon_index: The 1-based index of the current horizon.
        horizon_start_date: The start date of the current horizon.
        per_method_results: A dictionary where keys are method tags and
            values are dictionaries containing 'res' and 'kpi' for that method.
        total_blocks: Total number of available blocks for this horizon.
    """
    horizon_entry: Dict[str, Any] = {
        JSONKeys.HORIZON_INDEX: horizon_index,
        JSONKeys.START_DATE: horizon_start_date.isoformat(),
        JSONKeys.TOTAL_BLOCKS: total_blocks,  
    }

    for method_tag, results_dict in per_method_results.items():
        solver_result = results_dict.get("res", {})
        kpi_results = results_dict.get("kpi", {})

        model_status = solver_result.get("status", "N/A")
        gurobi_model_obj = solver_result.get("model")
        model_runtime = (
            getattr(gurobi_model_obj, "Runtime", -1.0) if gurobi_model_obj else -1.0
        )

        planned_obj_val = solver_result.get("obj")
        if planned_obj_val is not None:
            try:
                planned_obj_val = float(planned_obj_val)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert planned objective '{planned_obj_val}' to float for method {method_tag}, horizon {horizon_index}. Storing as is."
                )

        horizon_entry[method_tag] = {
            JSONKeys.STATUS: model_status,
            JSONKeys.PLANNED_COST: planned_obj_val,
            JSONKeys.ACTUAL_COST: kpi_results.get("total_actual_cost"),
            JSONKeys.OVERTIME_MIN: kpi_results.get("overtime_minutes_total"),
            JSONKeys.IDLE_MIN: kpi_results.get("idle_minutes_total"),
            JSONKeys.SCHEDULED_COUNT: kpi_results.get("scheduled_count"),
            JSONKeys.REJECTED_COUNT: kpi_results.get("rejected_count"),
            JSONKeys.RUNTIME_SEC: model_runtime,
        }

    output_data_struct[JSONKeys.HORIZONS].append(horizon_entry)
    logger.debug(
        f"Appended results for horizon {horizon_index} ({horizon_start_date})."
    )


def _generate_statistic_string(data_array: List[float]) -> str:
    """Helper to create a string summary of mean, median, min, max."""
    if not data_array or all(x is None for x in data_array):
        return "data=NA"

    valid_data = [x for x in data_array if x is not None]
    if not valid_data:
        return "data=NA (all None)"

    return (
        f"mean={np.mean(valid_data):.0f}, median={np.median(valid_data):.0f}, "
        f"min={np.min(valid_data):.0f}, max={np.max(valid_data):.0f} "
        f"(n={len(valid_data)})"
    )


def print_console_summary(
    method_tags_to_summarize: List[str], output_data_struct: Dict[str, Any]
) -> None:
    """Prints a summary of results over all horizons to the console.

    Args:
        method_tags_to_summarize: List of method tags for summary.
        output_data_struct: The main results dictionary containing horizon data.
    """
    logger.info("\n" + "=" * 20 + " Summary Over Horizons " + "=" * 20)

    horizons_data = output_data_struct.get(JSONKeys.HORIZONS, [])
    if not horizons_data:
        logger.info("No horizon data available to summarize.")
        return

    for method_tag in method_tags_to_summarize:
        # Extract data for this method
        planned_costs = [
            h.get(method_tag, {}).get(JSONKeys.PLANNED_COST) for h in horizons_data
        ]
        actual_costs = [
            h.get(method_tag, {}).get(JSONKeys.ACTUAL_COST) for h in horizons_data
        ]
        idle_minutes = [
            h.get(method_tag, {}).get(JSONKeys.IDLE_MIN) for h in horizons_data
        ]
        overtime_minutes = [
            h.get(method_tag, {}).get(JSONKeys.OVERTIME_MIN) for h in horizons_data
        ]
        runtimes_sec = [
            h.get(method_tag, {}).get(JSONKeys.RUNTIME_SEC) for h in horizons_data
        ]

        logger.info(f"--- Method: {method_tag} ---")
        logger.info(
            f"  Planned Objective : {_generate_statistic_string(planned_costs)}"
        )
        logger.info(f"  Actual Cost       : {_generate_statistic_string(actual_costs)}")
        logger.info(f"  Idle Minutes      : {_generate_statistic_string(idle_minutes)}")
        logger.info(
            f"  Overtime Minutes  : {_generate_statistic_string(overtime_minutes)}"
        )
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
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(output_data_struct, f, indent=2, cls=NpEncoder)
        logger.info(f"Detailed results successfully written to: {path_obj}")
    except IOError as e:
        logger.error(
            f"IOError saving detailed results to {path_obj}: {e}", exc_info=True
        )
    except TypeError as e:
        logger.error(
            f"TypeError during JSON serialization for detailed results: {e}. "
            "Check for non-serializable types.",
            exc_info=True,
        )


def save_aggregated_results(
    output_data_struct: Dict[str, Any],
    aggregated_output_file_path: Union[str, Path],
    method_tags_to_aggregate: List[str],
) -> None:
    """Calculates and saves aggregated (sum, average, median) results to a JSON file."""
    aggregated_data_struct: Dict[str, Any] = {
        JSONKeys.CONFIG: output_data_struct.get(JSONKeys.CONFIG, {}),
        JSONKeys.AGGREGATE: {},
    }

    horizons_data = output_data_struct.get(JSONKeys.HORIZONS, [])
    if not horizons_data:
        logger.warning("No horizon data to aggregate; saving minimal structure.")
        Path(aggregated_output_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(aggregated_output_file_path, "w") as f:
            json.dump(aggregated_data_struct, f, indent=2)
        return

    for tag in method_tags_to_aggregate:

        def collect(key: str) -> List[float]:
            vals = [h.get(tag, {}).get(key) for h in horizons_data]
            return [float(v) for v in vals if v is not None]

        # Gather all metrics
        planned = collect(JSONKeys.PLANNED_COST)
        actual = collect(JSONKeys.ACTUAL_COST)
        ot = collect(JSONKeys.OVERTIME_MIN)
        idle = collect(JSONKeys.IDLE_MIN)
        runsec = collect(JSONKeys.RUNTIME_SEC)
        sched = collect(JSONKeys.SCHEDULED_COUNT)
        rej = collect(JSONKeys.REJECTED_COUNT)

        # Build the aggregated block
        m = {}
        m[JSONKeys.MEDIAN_PLANNED_COST] = float(np.median(planned)) if planned else None
        m[JSONKeys.AVG_PLANNED_COST] = float(np.mean(planned)) if planned else None
        m[JSONKeys.MEDIAN_ACTUAL_COST] = float(np.median(actual)) if actual else None
        m[JSONKeys.AVG_ACTUAL_COST] = float(np.mean(actual)) if actual else None
        m[JSONKeys.MEDIAN_OVERTIME_MIN] = float(np.median(ot)) if ot else None
        m[JSONKeys.AVG_OVERTIME_MIN] = float(np.mean(ot)) if ot else None
        m[JSONKeys.MEDIAN_IDLE_MIN] = float(np.median(idle)) if idle else None
        m[JSONKeys.AVG_IDLE_MIN] = float(np.mean(idle)) if idle else None
        m[JSONKeys.MEDIAN_SCHEDULED] = float(np.median(sched)) if sched else None
        m[JSONKeys.AVG_SCHEDULED] = float(np.mean(sched)) if sched else None
        m[JSONKeys.MEDIAN_REJECTED] = float(np.median(rej)) if rej else None
        m[JSONKeys.AVG_REJECTED] = float(np.mean(rej)) if rej else None
        m[JSONKeys.AVG_RUNTIME_SEC] = float(np.mean(runsec)) if runsec else None

        aggregated_data_struct[JSONKeys.AGGREGATE][tag] = m

    # Save to disk
    path_obj = Path(aggregated_output_file_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w") as f:
        json.dump(aggregated_data_struct, f, indent=2, cls=NpEncoder)
    logger.info(f"Aggregated results written to: {path_obj}")


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)
