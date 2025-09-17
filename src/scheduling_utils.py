"""
Utilities for processing and evaluating surgery schedules with graduated overtime costs.
Includes functions to extract schedules from optimization models,
evaluate costs based on actual durations, and select surgeries for
planning horizons.
"""

import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import gurobipy as gp
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.constants import (
    DataColumns,
    FeatureColumns,
    ScheduleColumns,
    DomainConstants,
    GurobiConstants,
    LoggingConstants,
)
from src.data_processing import add_time_features

# Setup logger
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


def extract_schedule(
    model: Optional[gp.Model],
    surgeries_input_list: List[Dict[str, Any]],
    include_block_column: bool = True,
) -> pd.DataFrame:
    """Extracts a surgery schedule from a solved Gurobi optimization model.

    Converts Gurobi decision variable values (x[i,d,b] for assignment,
    r[i] for rejection) into a structured pandas DataFrame.

    Args:
        model: The solved Gurobi model object. Can be None if no solution.
        surgeries_input_list: A list of surgery dictionaries, in the same order
            as they were indexed (0 to N-1) in the Gurobi model. Each dict must
            contain at least 'id', 'proc_id', 'booked_min', and 'actual_dur_min'.
        include_block_column: If True, the output DataFrame includes an
            'AssignedBlock' column.

    Returns:
        A pandas DataFrame representing the schedule.
    """
    output_columns: List[str] = [
        ScheduleColumns.SURGERY_ID,
        ScheduleColumns.PROC_ID,
        ScheduleColumns.ASSIGNED_DAY,
        DataColumns.BOOKED_MIN,
        ScheduleColumns.ACTUAL_DUR_MIN,
    ]
    if include_block_column:
        output_columns.append(ScheduleColumns.ASSIGNED_BLOCK)

    model_name_for_log = GurobiConstants.DEFAULT_MODEL_NAME
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
    numerical_pattern = re.compile(GurobiConstants.REGEX_NUMERICAL_VALUES)

    logger.debug(f"Extracting schedule from Gurobi model '{model_name_for_log}'.")
    for var in model.getVars():
        try:
            var_value = var.X
        except AttributeError:
            logger.warning(
                f"Variable {var.VarName} has no solution value '.X'. Skipping."
            )
            continue

        if abs(var_value) < GurobiConstants.BINARY_THRESHOLD:
            continue

        var_name = var.VarName
        indices_str = numerical_pattern.findall(var_name)
        if not indices_str:
            continue

        extracted_indices = list(map(int, indices_str))

        if (
            var_name.startswith(GurobiConstants.VAR_X_PREFIX)
            and len(extracted_indices) >= 3
        ):
            surgery_idx, day_idx, block_idx = extracted_indices[:3]
            assignments[surgery_idx] = (day_idx, block_idx)
        elif var_name.startswith(GurobiConstants.VAR_R_PREFIX) and extracted_indices:
            surgery_idx = extracted_indices[0]
            assignments.setdefault(surgery_idx, "Rejected")

    schedule_rows_list: List[Dict[str, Any]] = []
    unassigned_surgeries_count = 0

    for i, surgery_data in enumerate(surgeries_input_list):
        assigned_slot = assignments.get(i)

        row_data_dict: Dict[str, Any] = {
            ScheduleColumns.SURGERY_ID: surgery_data.get("id", i),
            ScheduleColumns.PROC_ID: surgery_data["proc_id"],
            ScheduleColumns.ASSIGNED_DAY: None,
            DataColumns.BOOKED_MIN: surgery_data[DataColumns.BOOKED_MIN],
            ScheduleColumns.ACTUAL_DUR_MIN: surgery_data[
                ScheduleColumns.ACTUAL_DUR_MIN
            ],
        }
        if include_block_column:
            row_data_dict[ScheduleColumns.ASSIGNED_BLOCK] = pd.NA

        if assigned_slot is None:
            logger.warning(
                f"Surgery index {i} (ID: {surgery_data.get('id', 'N/A')}) "
                f"had no Gurobi assignment (x=0, r=0). It will be excluded from the schedule."
            )
            unassigned_surgeries_count += 1
            continue

        if assigned_slot == "Rejected":
            row_data_dict[ScheduleColumns.ASSIGNED_DAY] = "Rejected"
        else:
            day_idx_assigned, block_idx_assigned = assigned_slot
            row_data_dict[ScheduleColumns.ASSIGNED_DAY] = day_idx_assigned
            if include_block_column:
                row_data_dict[ScheduleColumns.ASSIGNED_BLOCK] = block_idx_assigned

        schedule_rows_list.append(row_data_dict)

    if unassigned_surgeries_count > 0:
        logger.warning(
            f"Model '{model_name_for_log}': {unassigned_surgeries_count}/{len(surgeries_input_list)} "
            "surgeries had no assignment/rejection variable set to 1. They were excluded."
        )

    schedule_df = pd.DataFrame(schedule_rows_list, columns=output_columns)

    # Ensure 'AssignedBlock' can hold integers and pandas' NA
    if include_block_column and ScheduleColumns.ASSIGNED_BLOCK in schedule_df.columns:
        try:
            schedule_df[ScheduleColumns.ASSIGNED_BLOCK] = schedule_df[
                ScheduleColumns.ASSIGNED_BLOCK
            ].astype(pd.Int64Dtype())
        except Exception as e:
            logger.warning(
                f"Could not cast '{ScheduleColumns.ASSIGNED_BLOCK}' to Int64Dtype: {e}. "
                "Column dtype might be object or float, which could affect downstream logic."
            )
    logger.debug(f"Extracted schedule with {len(schedule_df)} entries.")
    return schedule_df


def evaluate_schedule_actual_costs(
    schedule_df: pd.DataFrame,
    daily_block_capacities: Dict[int, int],
    config: AppConfig,
) -> Dict[str, Any]:
    """Computes operational costs (overtime, idle, rejection) for a schedule.

    Uses actual surgery durations for cost calculation with single overtime rate.
    This function evaluates what actually happened - overtime is NOT capped here.

    Args:
        schedule_df: DataFrame of the schedule, typically from `extract_schedule`.
        daily_block_capacities: Dict mapping day index to number of blocks.
        config: Application configuration containing cost and capacity settings.

    Returns:
        A dictionary containing KPIs.
    """
    horizon_days = config.data.planning_horizon_days
    block_duration_minutes = config.operating_room.block_size_minutes
    cost_overtime_per_min = config.costs.overtime_per_min
    cost_idle_per_min = config.costs.idle_per_min
    cost_rejection_per_case = config.costs.rejection_per_case

    # Determine max blocks per day for matrix shapes
    max_blocks_in_any_day = 0
    if daily_block_capacities and daily_block_capacities.values():
        max_blocks_in_any_day = max(daily_block_capacities.values())
    if horizon_days > 0 and max_blocks_in_any_day == 0:
        max_blocks_in_any_day = 1

    # Initialize capacity, overtime, and idle time matrices
    if horizon_days > 0:
        capacity_matrix = np.zeros((horizon_days, max_blocks_in_any_day))
        for day_idx, num_blocks_on_day in daily_block_capacities.items():
            if 0 <= day_idx < horizon_days:
                for block_idx in range(min(num_blocks_on_day, max_blocks_in_any_day)):
                    capacity_matrix[day_idx, block_idx] = block_duration_minutes

        overtime_matrix = np.zeros_like(capacity_matrix)
        idle_matrix = capacity_matrix.copy()
    else:
        capacity_matrix = np.array([[]], dtype=float)
        overtime_matrix = np.array([[]], dtype=float)
        idle_matrix = np.array([[]], dtype=float)

    total_rejection_cost: float = 0.0
    total_overtime_cost: float = 0.0
    total_idle_cost: float = 0.0
    num_scheduled_surgeries: int = 0
    num_rejected_surgeries: int = 0
    num_skipped_schedule_rows: int = 0

    has_block_column = ScheduleColumns.ASSIGNED_BLOCK in schedule_df.columns

    # Handle edge cases for empty schedule or zero horizon
    if schedule_df.empty and horizon_days == 0:
        pass  # All costs and counts remain zero
    elif schedule_df.empty and horizon_days > 0:
        pass  # overtime_matrix is already zeros, idle_matrix reflects full capacity
    elif horizon_days == 0 and not schedule_df.empty:
        logger.warning(
            "Schedule is not empty, but planning horizon is 0. "
            "All non-rejected surgeries in schedule will be treated as rejected for cost."
        )
        for _, row_data in schedule_df.iterrows():
            total_rejection_cost += (
                cost_rejection_per_case * row_data[DataColumns.BOOKED_MIN]
            )
            num_rejected_surgeries += 1
    elif horizon_days > 0 and not schedule_df.empty:
        for _, row_data in schedule_df.iterrows():
            assigned_day_val = row_data[ScheduleColumns.ASSIGNED_DAY]

            if assigned_day_val == "Rejected":
                total_rejection_cost += (
                    cost_rejection_per_case * row_data[DataColumns.BOOKED_MIN]
                )
                num_rejected_surgeries += 1
                continue

            # Process scheduled surgeries
            if not isinstance(assigned_day_val, (int, np.integer)):
                logger.debug(
                    f"Skipping row: AssignedDay '{assigned_day_val}' is not an integer. "
                    f"SurgeryID: {row_data.get(ScheduleColumns.SURGERY_ID, 'N/A')}."
                )
                num_skipped_schedule_rows += 1
                continue

            day_idx_assigned = int(assigned_day_val)

            if not (0 <= day_idx_assigned < horizon_days):
                logger.debug(
                    f"Skipping row: AssignedDay {day_idx_assigned} is out of horizon "
                    f"[0, {horizon_days-1}). SurgeryID: {row_data.get(ScheduleColumns.SURGERY_ID, 'N/A')}."
                )
                num_skipped_schedule_rows += 1
                continue

            if has_block_column:
                assigned_block_val = row_data.get(ScheduleColumns.ASSIGNED_BLOCK)
                block_idx_assigned: Optional[int] = None

                if pd.notna(assigned_block_val):
                    try:
                        block_idx_assigned = int(float(assigned_block_val))
                    except (ValueError, TypeError):
                        logger.debug(
                            f"Skipping row: AssignedBlock '{assigned_block_val}' not convertible to int. "
                            f"SurgeryID: {row_data.get(ScheduleColumns.SURGERY_ID, 'N/A')}."
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
                        f"(capacity: {num_blocks_on_assigned_day} blocks). SurgeryID: {row_data.get(ScheduleColumns.SURGERY_ID, 'N/A')}."
                    )
                    num_skipped_schedule_rows += 1
                    continue

                # All checks pass, account for the surgery in the block
                actual_duration = row_data[ScheduleColumns.ACTUAL_DUR_MIN]

                # Ensure indices are within bounds
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

                    # Any remaining duration is overtime (NOT CAPPED - this is evaluation)
                    overtime_for_this_surgery = actual_duration - time_covered_by_idle
                    if overtime_for_this_surgery > 0:
                        overtime_matrix[
                            day_idx_assigned, block_idx_assigned
                        ] += overtime_for_this_surgery

                    num_scheduled_surgeries += 1
                else:
                    logger.warning(
                        f"Matrix index out of bounds for Day {day_idx_assigned}, Block {block_idx_assigned}. "
                        f"SurgeryID: {row_data.get(ScheduleColumns.SURGERY_ID, 'N/A')}. This indicates an internal logic issue."
                    )
                    num_skipped_schedule_rows += 1
                    continue
            else:  # Day-level accounting
                logger.warning(
                    f"'{ScheduleColumns.ASSIGNED_BLOCK}' column missing. Day-level cost accounting is not fully "
                    "implemented in this version. Skipping surgery cost calculation."
                )
                num_skipped_schedule_rows += 1

    if num_skipped_schedule_rows > 0:
        logger.warning(
            f"Skipped {num_skipped_schedule_rows} rows from schedule_df during cost "
            "evaluation due to invalid/missing day/block assignments or data issues."
        )

    # Calculate final costs (single overtime rate, not capped)
    total_overtime_cost = overtime_matrix.sum() * cost_overtime_per_min
    total_idle_cost = idle_matrix.sum() * cost_idle_per_min

    total_operational_cost = total_overtime_cost + total_idle_cost
    final_total_actual_cost = total_operational_cost + total_rejection_cost

    return {
        "total_actual_cost": final_total_actual_cost,
        "scheduled_count": num_scheduled_surgeries,
        "rejected_count": num_rejected_surgeries,
        "overtime_minutes_total": float(overtime_matrix.sum()),
        "idle_minutes_total": float(idle_matrix.sum()),
        "overtime_cost": total_overtime_cost,
        "idle_cost": total_idle_cost,
        "rejection_cost": total_rejection_cost,
        "overtime_matrix": overtime_matrix,
        "idle_matrix": idle_matrix,
    }


def select_surgeries(
    df_scheduling_pool: pd.DataFrame,
    horizon_start_timestamp: pd.Timestamp,
    config: AppConfig,
    predictor_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Selects surgeries for a planning horizon and optionally predicts durations.

    Filters surgeries from `df_scheduling_pool` that fall within the
    `planning_horizon_days` starting from `horizon_start_timestamp`.
    If `predictor_model` is provided, it's used to predict surgery durations.

    Args:
        df_scheduling_pool: DataFrame of all available surgeries.
        horizon_start_timestamp: Start timestamp of the planning horizon.
        config: Application configuration containing horizon settings.
        predictor_model: Optional trained ML model with a `.predict(X)` method.

    Returns:
        List of surgery dictionaries for the horizon.
    """
    planning_horizon_days = config.data.planning_horizon_days

    horizon_start_date_norm = horizon_start_timestamp.normalize().date()
    horizon_end_date_norm = (
        horizon_start_timestamp.normalize() + timedelta(days=planning_horizon_days - 1)
    ).date()

    # Filter surgeries within the planning horizon date range
    if (
        DataColumns.ACTUAL_START not in df_scheduling_pool.columns
        or not pd.api.types.is_datetime64_any_dtype(
            df_scheduling_pool[DataColumns.ACTUAL_START]
        )
    ):
        logger.error(
            f"'{DataColumns.ACTUAL_START}' is missing or not datetime in df_scheduling_pool."
        )
        return []

    surgeries_in_horizon_df = df_scheduling_pool[
        (
            df_scheduling_pool[DataColumns.ACTUAL_START].dt.normalize().dt.date
            >= horizon_start_date_norm
        )
        & (
            df_scheduling_pool[DataColumns.ACTUAL_START].dt.normalize().dt.date
            <= horizon_end_date_norm
        )
    ].copy()

    if surgeries_in_horizon_df.empty:
        logger.info(
            f"No surgeries found in scheduling pool for horizon "
            f"{horizon_start_date_norm} to {horizon_end_date_norm}."
        )
        return []

    # Prepare features and predict if a predictor_model is provided
    if predictor_model is not None:
        surgeries_in_horizon_df = add_time_features(surgeries_in_horizon_df)

        missing_features = [
            col
            for col in FeatureColumns.ALL
            if col not in surgeries_in_horizon_df.columns
        ]
        if missing_features:
            msg = f"Missing feature columns for prediction: {missing_features}"
            logger.error(msg)
            raise KeyError(msg)

        X_to_predict = surgeries_in_horizon_df[FeatureColumns.ALL].copy()
        # Ensure categorical features are strings and handle NaNs
        for cat_col in FeatureColumns.CATEGORICAL:
            X_to_predict[cat_col] = (
                X_to_predict[cat_col]
                .astype(str)
                .fillna(DomainConstants.UNKNOWN_CATEGORY)
            )

        try:
            predictions = predictor_model.predict(X_to_predict)
            surgeries_in_horizon_df["predicted_dur_min"] = np.maximum(
                predictions, DomainConstants.MIN_PROCEDURE_DURATION
            ).round()
            logger.debug(
                f"Generated predictions for {len(surgeries_in_horizon_df)} surgeries."
            )
        except Exception as e:
            logger.error(
                f"Error during prediction in select_surgeries: {e}", exc_info=True
            )
            surgeries_in_horizon_df["predicted_dur_min"] = np.nan

    # Construct the list of surgery dictionaries
    selected_surgeries_list: List[Dict[str, Any]] = []
    for original_idx, row_data in surgeries_in_horizon_df.iterrows():
        surgery_dict: Dict[str, Any] = {
            "id": original_idx,
            "proc_id": row_data[DataColumns.MAIN_PROCEDURE_ID],
            DataColumns.BOOKED_MIN: row_data[DataColumns.BOOKED_MIN],
            ScheduleColumns.ACTUAL_DUR_MIN: row_data[
                DataColumns.PROCEDURE_DURATION_MIN
            ],
        }
        if "predicted_dur_min" in row_data and pd.notna(row_data["predicted_dur_min"]):
            surgery_dict["predicted_dur_min"] = row_data["predicted_dur_min"]

        selected_surgeries_list.append(surgery_dict)

    logger.info(
        f"Selected {len(selected_surgeries_list)} surgeries for {planning_horizon_days}-day "
        f"horizon starting {horizon_start_date_norm}."
    )
    return selected_surgeries_list
