"""
Utilities for processing and evaluating surgery schedules.
Includes functions to extract schedules from optimization models,
evaluate costs based on actual durations, and select surgeries for
planning horizons.
"""

import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import gurobipy as gp  # For type hinting Model
import numpy as np
import pandas as pd

from src.constants import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    COL_ACTUAL_DUR_MIN,
    COL_ACTUAL_START,
    COL_ASSIGNED_BLOCK,
    COL_ASSIGNED_DAY,
    COL_BOOKED_MIN,
    COL_BOOKED_MIN,
    COL_PROCEDURE_DURATION_MIN,
    COL_MAIN_PROCEDURE_ID,
    COL_PROC_ID,
    COL_SURGERY_ID,
    COL_ACTUAL_DUR_MIN,
    COL_ASSIGNED_BLOCK,
    COL_ASSIGNED_DAY,
    COL_PROC_ID,
    COL_SURGERY_ID,
    DEFAULT_GUROBI_MODEL_NAME,
    DEFAULT_LOGGER_NAME,
    GUROBI_BINARY_THRESHOLD,
    GUROBI_VAR_R_PREFIX,
    GUROBI_VAR_X_PREFIX,
    MIN_PROCEDURE_DURATION,
    REGEX_NUMERICAL_VALUES,
    UNKNOWN_CATEGORY,
)
from src.data_processing import add_time_features


# Setup logger
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def extract_schedule(
    model: Optional[gp.Model],
    surgeries_input_list: List[Dict[str, Any]],
    # params: Dict[str, Any], # params was unused, removing for now
    include_block_column: bool = True,
) -> pd.DataFrame:
    """Extracts a surgery schedule from a solved Gurobi optimization model.

    Converts Gurobi decision variable values (x[i,d,b] for assignment,
    r[i] for rejection) into a structured pandas DataFrame.

    Args:
        model: The solved Gurobi model object. Can be None if no solution.
        surgeries_input_list: A list of surgery dictionaries, in the same order
            as they were indexed (0 to N-1) in the Gurobi model. Each dict must
            contain at least 'id' (original surgery identifier), 'proc_id',
            'booked_min', and 'actual_dur_min'.
        include_block_column: If True, the output DataFrame includes an
            'AssignedBlock' column.

    Returns:
        A pandas DataFrame representing the schedule. Columns include:
        - 'SurgeryID': Original surgery identifier.
        - 'ProcID': Procedure ID.
        - 'AssignedDay': Day index (0 to H-1) or "Rejected".
        - 'AssignedBlock': Block index (0 to B-1) or None/pd.NA if rejected
                           or if `include_block_column` is False.
        - 'BookedMin': Booked duration.
        - 'ActualDurMin': Actual duration.
    """
    output_columns: List[str] = [
        COL_SURGERY_ID,
        COL_PROC_ID,
        COL_ASSIGNED_DAY,
        COL_BOOKED_MIN,
        COL_ACTUAL_DUR_MIN,  # Using constant COL_BOOKED_MIN
    ]
    if include_block_column:
        output_columns.append(COL_ASSIGNED_BLOCK)

    model_name_for_log = DEFAULT_GUROBI_MODEL_NAME
    if model and hasattr(model, "ModelName") and model.ModelName:
        model_name_for_log = model.ModelName

    if model is None or model.SolCount == 0:
        logger.warning(
            f"Model '{model_name_for_log}' is None or has no solution. "
            "Returning empty schedule DataFrame."
        )
        return pd.DataFrame(columns=output_columns)

    # surgery_idx -> (day_idx, block_idx) or "Rejected"
    assignments: Dict[int, Any] = {}
    # Pre-compile regex for extracting numbers from variable names
    numerical_pattern = re.compile(REGEX_NUMERICAL_VALUES)

    logger.debug(f"Extracting schedule from Gurobi model '{model_name_for_log}'.")
    for var in model.getVars():
        try:
            var_value = var.X
        except AttributeError:  # Should not happen if SolCount > 0
            logger.warning(
                f"Variable {var.VarName} has no solution value '.X'. Skipping."
            )
            continue

        # Consider values < GUROBI_BINARY_THRESHOLD (e.g., 0.5) as not selected for binary vars
        if abs(var_value) < GUROBI_BINARY_THRESHOLD:
            continue

        var_name = var.VarName
        # Extract all integer sequences from the variable name
        indices_str = numerical_pattern.findall(var_name)
        if not indices_str:  # No numbers found, not a structured variable like x[i,d,b]
            continue

        extracted_indices = list(map(int, indices_str))

        if var_name.startswith(GUROBI_VAR_X_PREFIX) and len(extracted_indices) >= 3:
            # Assignment variable: x[i, d, b]
            surgery_idx, day_idx, block_idx = extracted_indices[:3]
            assignments[surgery_idx] = (day_idx, block_idx)
        elif var_name.startswith(GUROBI_VAR_R_PREFIX) and extracted_indices:
            # Rejection variable: r[i]
            surgery_idx = extracted_indices[0]
            # Only mark as rejected if not already assigned by an x-variable
            # (A model inconsistency if both x[i,d,b]=1 and r[i]=1 for same i)
            assignments.setdefault(surgery_idx, "Rejected")

    schedule_rows_list: List[Dict[str, Any]] = []
    unassigned_surgeries_count = 0

    for i, surgery_data in enumerate(surgeries_input_list):
        assigned_slot = assignments.get(i)

        row_data_dict: Dict[str, Any] = {
            COL_SURGERY_ID: surgery_data.get(
                "id", i
            ),  # Fallback to index 'i' if 'id' is missing
            COL_PROC_ID: surgery_data["proc_id"],
            COL_ASSIGNED_DAY: None,  # Initialize
            COL_BOOKED_MIN: surgery_data[COL_BOOKED_MIN],
            COL_ACTUAL_DUR_MIN: surgery_data[
                COL_ACTUAL_DUR_MIN
            ],  # Assuming this key exists
        }
        if include_block_column:
            row_data_dict[COL_ASSIGNED_BLOCK] = (
                pd.NA
            )  # Initialize with pd.NA for nullable int

        if assigned_slot is None:
            # Surgery was neither explicitly scheduled nor rejected by Gurobi vars.
            # This could indicate an issue with model constraints (e.g., sum(x_i) + r_i = 1 violated)
            # or an incomplete Gurobi solution.
            logger.warning(
                f"Surgery index {i} (ID: {surgery_data.get('id', 'N/A')}) "
                f"had no Gurobi assignment (x=0, r=0). It will be excluded from the schedule."
            )
            unassigned_surgeries_count += 1
            continue  # Skip this surgery from the output schedule

        if assigned_slot == "Rejected":
            row_data_dict[COL_ASSIGNED_DAY] = "Rejected"
            # COL_ASSIGNED_BLOCK remains pd.NA
        else:  # It's a tuple (day_idx, block_idx)
            day_idx_assigned, block_idx_assigned = (
                assigned_slot  # These are already integers
            )
            row_data_dict[COL_ASSIGNED_DAY] = day_idx_assigned
            if include_block_column:
                row_data_dict[COL_ASSIGNED_BLOCK] = block_idx_assigned

        schedule_rows_list.append(row_data_dict)

    if unassigned_surgeries_count > 0:
        logger.warning(
            f"Model '{model_name_for_log}': {unassigned_surgeries_count}/{len(surgeries_input_list)} "
            "surgeries had no assignment/rejection variable set to 1. They were excluded."
        )

    schedule_df = pd.DataFrame(schedule_rows_list, columns=output_columns)

    # Ensure 'AssignedBlock' can hold integers and pandas' NA (nullable integer type)
    if include_block_column and COL_ASSIGNED_BLOCK in schedule_df.columns:
        try:
            schedule_df[COL_ASSIGNED_BLOCK] = schedule_df[COL_ASSIGNED_BLOCK].astype(
                pd.Int64Dtype()
            )
        except Exception as e:
            logger.warning(
                f"Could not cast '{COL_ASSIGNED_BLOCK}' to Int64Dtype: {e}. "
                "Column dtype might be object or float, which could affect downstream logic."
            )
    logger.debug(f"Extracted schedule with {len(schedule_df)} entries.")
    return schedule_df


def evaluate_schedule_actual_costs(
    schedule_df: pd.DataFrame,
    daily_block_capacities: Dict[int, int],  # Renamed from day_blocks for clarity
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes operational costs (overtime, idle, rejection) for a schedule.

    Uses actual surgery durations for cost calculation.

    Args:
        schedule_df: DataFrame of the schedule, typically from `extract_schedule`.
            Expected columns: 'AssignedDay', 'AssignedBlock' (optional),
            'ActualDurMin', 'BookedMin'.
        daily_block_capacities: Dict mapping day index (0 to H-1) to the number
            of available OR blocks on that day.
        params: Configuration dictionary. Must include:
            - 'planning_horizon_days' (int)
            - 'block_size_minutes' (int)
            - 'cost_overtime_per_min' (float)
            - 'cost_idle_per_min' (float)
            - 'cost_rejection_per_case' (float)

    Returns:
        A dictionary containing KPIs:
        - 'total_actual_cost': Sum of overtime, idle, and rejection costs.
        - 'scheduled_count': Number of successfully scheduled surgeries.
        - 'rejected_count': Number of rejected surgeries.
        - 'overtime_minutes_total': Total overtime minutes.
        - 'idle_minutes_total': Total idle time minutes.
        - 'overtime_matrix': 2D numpy array (H x max_blocks) of overtime per block.
        - 'idle_matrix': 2D numpy array (H x max_blocks) of idle time per block.

    Raises:
        KeyError: If required parameters are missing from `params`.
    """
    try:
        horizon_days = params["planning_horizon_days"]
        block_duration_minutes = params["block_size_minutes"]
        cost_overtime_per_min = params["cost_overtime_per_min"]
        cost_idle_per_min = params["cost_idle_per_min"]
        cost_rejection_per_case = params["cost_rejection_per_case"]
    except KeyError as e:
        logger.error(f"Missing parameter for schedule cost evaluation: {e}")
        raise

    # Determine max blocks per day for matrix shapes
    max_blocks_in_any_day = 0
    if daily_block_capacities and daily_block_capacities.values():
        max_blocks_in_any_day = max(daily_block_capacities.values())
    if horizon_days > 0 and max_blocks_in_any_day == 0:
        # If horizon exists but no blocks defined, matrices should still have 1 column
        # to avoid issues with empty arrays if some logic expects at least one block.
        max_blocks_in_any_day = 1

    # Initialize capacity, overtime, and idle time matrices
    if horizon_days > 0:
        # capacity_matrix stores the capacity (in minutes) of each block
        capacity_matrix = np.zeros((horizon_days, max_blocks_in_any_day))
        for day_idx, num_blocks_on_day in daily_block_capacities.items():
            if 0 <= day_idx < horizon_days:  # Ensure day_idx is within horizon
                # Fill capacity for actual blocks available on this day, up to matrix width
                for block_idx in range(min(num_blocks_on_day, max_blocks_in_any_day)):
                    capacity_matrix[day_idx, block_idx] = block_duration_minutes

        overtime_matrix = np.zeros_like(capacity_matrix)
        # Idle time starts as full capacity for all defined blocks
        idle_matrix = capacity_matrix.copy()
    else:  # No planning horizon, implies no capacity or operations
        capacity_matrix = np.array([[]], dtype=float)  # Ensure 2D for consistency
        overtime_matrix = np.array([[]], dtype=float)
        idle_matrix = np.array([[]], dtype=float)

    total_rejection_cost: float = 0.0
    num_scheduled_surgeries: int = 0
    num_rejected_surgeries: int = 0
    num_skipped_schedule_rows: int = 0

    has_block_column = COL_ASSIGNED_BLOCK in schedule_df.columns

    # Handle edge cases for empty schedule or zero horizon
    if schedule_df.empty and horizon_days == 0:  # No schedule, no horizon = zero costs
        pass  # All costs and counts remain zero
    elif schedule_df.empty and horizon_days > 0:  # No schedule, but horizon exists
        # All defined capacity is idle. No surgeries means no rejections.
        pass  # overtime_matrix is already zeros, idle_matrix reflects full capacity
    elif horizon_days == 0 and not schedule_df.empty:
        logger.warning(
            "Schedule is not empty, but planning horizon is 0. "
            "All non-rejected surgeries in schedule will be treated as rejected for cost."
        )
        for _, row_data in schedule_df.iterrows():
            # Use COL_BOOKED_MIN for rejection cost as per original logic for r[i] obj term
            total_rejection_cost += cost_rejection_per_case * row_data[COL_BOOKED_MIN]
            num_rejected_surgeries += 1
    # Main processing loop only if H > 0 and schedule_df is not empty
    elif horizon_days > 0 and not schedule_df.empty:
        for _, row_data in schedule_df.iterrows():
            assigned_day_val = row_data[COL_ASSIGNED_DAY]

            if assigned_day_val == "Rejected":
                total_rejection_cost += (
                    cost_rejection_per_case * row_data[COL_BOOKED_MIN]
                )
                num_rejected_surgeries += 1
                continue

            # Process scheduled surgeries
            if not isinstance(assigned_day_val, (int, np.integer)):
                logger.debug(
                    f"Skipping row: AssignedDay '{assigned_day_val}' is not an integer. "
                    f"SurgeryID: {row_data.get(COL_SURGERY_ID, 'N/A')}."
                )
                num_skipped_schedule_rows += 1
                continue

            day_idx_assigned = int(assigned_day_val)

            if not (0 <= day_idx_assigned < horizon_days):
                logger.debug(
                    f"Skipping row: AssignedDay {day_idx_assigned} is out of horizon "
                    f"[0, {horizon_days-1}). SurgeryID: {row_data.get(COL_SURGERY_ID, 'N/A')}."
                )
                num_skipped_schedule_rows += 1
                continue

            if has_block_column:
                assigned_block_val = row_data.get(COL_ASSIGNED_BLOCK)
                block_idx_assigned: Optional[int] = None

                if pd.notna(assigned_block_val):  # Handles None, np.nan, pd.NA
                    try:
                        block_idx_assigned = int(
                            float(assigned_block_val)
                        )  # float() handles "20.0"
                    except (ValueError, TypeError):
                        logger.debug(
                            f"Skipping row: AssignedBlock '{assigned_block_val}' not convertible to int. "
                            f"SurgeryID: {row_data.get(COL_SURGERY_ID, 'N/A')}."
                        )
                        num_skipped_schedule_rows += 1
                        continue

                # Check if block_idx is valid for the given day_idx
                num_blocks_on_assigned_day = daily_block_capacities.get(
                    day_idx_assigned, 0
                )
                if block_idx_assigned is None or not (
                    0 <= block_idx_assigned < num_blocks_on_assigned_day
                ):
                    logger.debug(
                        f"Skipping row: AssignedBlock {block_idx_assigned} is invalid for Day {day_idx_assigned} "
                        f"(capacity: {num_blocks_on_assigned_day} blocks). SurgeryID: {row_data.get(COL_SURGERY_ID, 'N/A')}."
                    )
                    num_skipped_schedule_rows += 1
                    continue

                # All checks pass, account for the surgery in the block
                actual_duration = row_data[COL_ACTUAL_DUR_MIN]

                # Ensure indices are within bounds of ot_matrix/idle_matrix
                if (
                    day_idx_assigned < overtime_matrix.shape[0]
                    and block_idx_assigned < overtime_matrix.shape[1]
                ):

                    # Reduce idle time by the portion of duration covered by it
                    time_covered_by_idle = min(
                        idle_matrix[day_idx_assigned, block_idx_assigned],
                        actual_duration,
                    )
                    idle_matrix[
                        day_idx_assigned, block_idx_assigned
                    ] -= time_covered_by_idle

                    # Any remaining duration is overtime
                    overtime_for_this_surgery = actual_duration - time_covered_by_idle
                    if overtime_for_this_surgery > 0:
                        overtime_matrix[
                            day_idx_assigned, block_idx_assigned
                        ] += overtime_for_this_surgery

                    num_scheduled_surgeries += 1
                else:
                    # This should ideally not be reached if matrix creation and index checks are correct
                    logger.warning(
                        f"Matrix index out of bounds for Day {day_idx_assigned}, Block {block_idx_assigned}. "
                        f"SurgeryID: {row_data.get(COL_SURGERY_ID, 'N/A')}. This indicates an internal logic issue."
                    )
                    num_skipped_schedule_rows += 1
                    continue
            else:  # Day-level accounting (if no 'AssignedBlock' column)
                logger.warning(
                    f"'{COL_ASSIGNED_BLOCK}' column missing. Day-level cost accounting is not fully "
                    "implemented in this version. Skipping surgery cost calculation."
                )
                num_skipped_schedule_rows += 1

    if num_skipped_schedule_rows > 0:
        logger.warning(
            f"Skipped {num_skipped_schedule_rows} rows from schedule_df during cost "
            "evaluation due to invalid/missing day/block assignments or data issues."
        )

    total_operational_cost = (
        overtime_matrix.sum() * cost_overtime_per_min
        + idle_matrix.sum() * cost_idle_per_min
    )
    final_total_actual_cost = total_operational_cost + total_rejection_cost

    return {
        "total_actual_cost": final_total_actual_cost,
        "scheduled_count": num_scheduled_surgeries,
        "rejected_count": num_rejected_surgeries,
        "overtime_minutes_total": float(overtime_matrix.sum()),
        "idle_minutes_total": float(idle_matrix.sum()),
        "overtime_matrix": overtime_matrix,
        "idle_matrix": idle_matrix,
    }


def select_surgeries(
    df_scheduling_pool: pd.DataFrame,
    horizon_start_timestamp: pd.Timestamp,
    params: Dict[str, Any],
    predictor_model: Optional[Any] = None,  # e.g., scikit-learn model
) -> List[Dict[str, Any]]:
    """Selects surgeries for a planning horizon and optionally predicts durations.

    Filters surgeries from `df_scheduling_pool` that fall within the
    `planning_horizon_days` starting from `horizon_start_timestamp`.
    If `predictor_model` is provided, it's used to predict surgery durations.

    Args:
        df_scheduling_pool: DataFrame of all available surgeries. Must include
            'actual_start', 'main_procedure_id', 'booked_time_minutes',
            'procedure_duration_min', and feature columns if predictor is used.
        horizon_start_timestamp: Start timestamp of the planning horizon.
        params: Configuration dictionary. Must contain 'planning_horizon_days'.
        predictor_model: Optional trained ML model with a `.predict(X)` method.

    Returns:
        List of surgery dictionaries for the horizon. Each dict includes 'id'
        (original index), 'proc_id', 'booked_min', 'actual_dur_min', and
        'predicted_dur_min' (if predictor_model is used).

    Raises:
        KeyError: If 'planning_horizon_days' is missing or if feature columns
            are missing when a predictor model is provided.
    """
    try:
        planning_horizon_days = params["planning_horizon_days"]
    except KeyError as e:
        logger.error(f"Missing 'planning_horizon_days' in params for select_surgeries.")
        raise

    horizon_start_date_norm = horizon_start_timestamp.normalize().date()
    # Horizon end is inclusive of the H-th day. Day 0 to Day H-1.
    horizon_end_date_norm = (
        horizon_start_timestamp.normalize() + timedelta(days=planning_horizon_days - 1)
    ).date()

    # Filter surgeries within the planning horizon date range
    # Ensure 'actual_start' is datetime before filtering
    if (
        COL_ACTUAL_START not in df_scheduling_pool.columns
        or not pd.api.types.is_datetime64_any_dtype(
            df_scheduling_pool[COL_ACTUAL_START]
        )
    ):
        logger.error(
            f"'{COL_ACTUAL_START}' is missing or not datetime in df_scheduling_pool."
        )
        # Depending on strictness, could raise error or return empty list
        return []

    surgeries_in_horizon_df = df_scheduling_pool[
        (
            df_scheduling_pool[COL_ACTUAL_START].dt.normalize().dt.date
            >= horizon_start_date_norm
        )
        & (
            df_scheduling_pool[COL_ACTUAL_START].dt.normalize().dt.date
            <= horizon_end_date_norm
        )
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    if surgeries_in_horizon_df.empty:
        logger.info(
            f"No surgeries found in scheduling pool for horizon "
            f"{horizon_start_date_norm} to {horizon_end_date_norm}."
        )
        return []

    # Prepare features and predict if a predictor_model is provided
    if predictor_model is not None:
        # Ensure temporal features are present for the predictor
        # add_time_features modifies DataFrame in-place or returns modified copy
        surgeries_in_horizon_df = add_time_features(surgeries_in_horizon_df)

        # Check for all required feature columns in the subset
        # ALL_FEATURE_COLS is assumed to be defined (e.g., from constants.py)
        missing_features = [
            col
            for col in ALL_FEATURE_COLS
            if col not in surgeries_in_horizon_df.columns
        ]
        if missing_features:
            msg = f"Missing feature columns for prediction: {missing_features}"
            logger.error(msg)
            raise KeyError(msg)

        X_to_predict = surgeries_in_horizon_df[ALL_FEATURE_COLS].copy()
        # Ensure categorical features are strings and handle NaNs for OneHotEncoder
        for cat_col in CATEGORICAL_FEATURE_COLS:
            X_to_predict[cat_col] = (
                X_to_predict[cat_col].astype(str).fillna(UNKNOWN_CATEGORY)
            )

        try:
            predictions = predictor_model.predict(X_to_predict)
            # Clip predictions to be at least MIN_PROCEDURE_DURATION and round
            surgeries_in_horizon_df["predicted_dur_min"] = np.maximum(
                predictions, MIN_PROCEDURE_DURATION
            ).round()
            logger.debug(
                f"Generated predictions for {len(surgeries_in_horizon_df)} surgeries."
            )
        except Exception as e:
            logger.error(
                f"Error during prediction in select_surgeries: {e}", exc_info=True
            )
            # Fallback: no "predicted_dur_min" column or fill with NaN/booked_min
            surgeries_in_horizon_df["predicted_dur_min"] = np.nan

    # Construct the list of surgery dictionaries
    selected_surgeries_list: List[Dict[str, Any]] = []
    for original_idx, row_data in surgeries_in_horizon_df.iterrows():
        surgery_dict: Dict[str, Any] = {
            "id": original_idx,  # Original index from df_scheduling_pool
            "proc_id": row_data[COL_MAIN_PROCEDURE_ID],
            COL_BOOKED_MIN: row_data[
                COL_BOOKED_MIN
            ],  # Use consistent booked_min key
            # Ensure actual_dur_min is present, otherwise problem for Oracle/evaluation
            COL_ACTUAL_DUR_MIN: row_data[COL_PROCEDURE_DURATION_MIN],
        }
        if "predicted_dur_min" in row_data and pd.notna(row_data["predicted_dur_min"]):
            surgery_dict["predicted_dur_min"] = row_data["predicted_dur_min"]
        # If prediction failed or wasn't run, 'predicted_dur_min' might be missing or NaN.
        # Downstream solvers using 'predicted_dur_min' must handle its potential absence
        # or have a fallback (e.g., to COL_BOOKED_MIN).

        selected_surgeries_list.append(surgery_dict)

    logger.info(
        f"Selected {len(selected_surgeries_list)} surgeries for {planning_horizon_days}-day "
        f"horizon starting {horizon_start_date_norm}."
    )
    return selected_surgeries_list
