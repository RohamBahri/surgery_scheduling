from datetime import timedelta
import pandas as pd
import numpy as np
import re

from .data_processing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    add_time_features,
)


def extract_schedule(
    model,  
    surgeries: list[dict],
    params: dict, 
    include_block: bool = True
) -> pd.DataFrame:
    """
    Extracts the surgery schedule from a solved Gurobi optimization model.

    Converts Gurobi decision variable values (x[i,d,b] for assignment, 
    r[i] for rejection) into a structured pandas DataFrame. 
    Ensures that day and block indices are integers.

    Parameters
    ----------
    model : gurobipy.Model
        The solved Gurobi model object.
    surgeries : list of dict
        A list of surgery dictionaries, in the same order as they were
        indexed (0 to N-1) in the Gurobi model. Each dictionary must contain
        at least 'id' (original surgery identifier), 'proc_id', 
        'booked_min', and 'actual_dur_min'.
    params : dict
        The parameter dictionary (currently used for logging model name, optional).
    include_block : bool, optional
        If True (default), the output DataFrame will include an 'AssignedBlock'
        column. If False, only 'AssignedDay' is included.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the schedule. Columns include:
        - SurgeryID: The original identifier of the surgery.
        - ProcID: The procedure ID.
        - AssignedDay: The day index (0 to H-1) to which the surgery is
                       assigned, or the string "Rejected".
        - AssignedBlock: The block index (0 to B-1) within the assigned day,
                         or None if rejected or if include_block is False.
                         This column will be of object type to accommodate
                         integers and None, preventing automatic float conversion
                         if rejections are present.
        - BookedMin: Booked duration of the surgery.
        - ActualDurMin: Actual duration of the surgery.
    
    Notes
    -----
    - It is crucial that the `surgeries` list matches the indexing used when
      defining variables in the Gurobi model.
    - The function assumes variable names like "x[i,d,b]" and "r[i]".
    - `AssignedBlock` is intentionally typed to handle `None` for rejected cases
      without forcing the column to float, which can cause issues with strict
      integer checks later.
    """
    output_cols = ["SurgeryID", "ProcID", "AssignedDay", "BookedMin", "ActualDurMin"]
    if include_block:
        output_cols.append("AssignedBlock")

    model_name_for_log = "GurobiModel"
    if hasattr(model, 'ModelName') and model.ModelName: 
        model_name_for_log = model.ModelName
    
    if model is None or model.SolCount == 0:
        print(f"[extract_schedule] Model '{model_name_for_log}' is None or has no solution. Returning empty DataFrame.")
        return pd.DataFrame(columns=output_cols)

    assignment = {}  # surgery_idx -> (day_idx, block_idx) or "Rejected"
    num_pat = re.compile(r"\d+") # Pre-compile regex for extracting numbers

    # Iterate through Gurobi variables to find assignments and rejections
    for var in model.getVars():
        try:
            var_value = var.X
        except AttributeError: # Should not happen if SolCount > 0
            continue # Skip if .X is not available

        if abs(var_value) < 0.5: # Consider values < 0.5 as not selected
            continue

        name = var.VarName
        extracted_indices = list(map(int, num_pat.findall(name)))

        if name.startswith("x") and len(extracted_indices) >= 3:
            # Assignment variable: x[i, d, b]
            surgery_idx, day_idx, block_idx = extracted_indices[:3]
            assignment[surgery_idx] = (day_idx, block_idx) # Store as tuple of ints
        elif name.startswith("r") and extracted_indices:
            # Rejection variable: r[i]
            surgery_idx = extracted_indices[0]
            # Only mark as rejected if not already assigned by an x-variable
            # (which would be a model inconsistency if both x and r are 1)
            assignment.setdefault(surgery_idx, "Rejected")

    # Construct the DataFrame from the assignments
    schedule_rows = []
    unassigned_in_loop_count = 0
    for i, surg_data in enumerate(surgeries):
        assigned_slot = assignment.get(i)

        row_data = {
            "SurgeryID": surg_data.get("id", i), # Fallback to index if 'id' is missing
            "ProcID": surg_data["proc_id"],
            "AssignedDay": None, # Initialize
            "BookedMin": surg_data["booked_min"],
            "ActualDurMin": surg_data["actual_dur_min"],
        }
        if include_block:
            row_data["AssignedBlock"] = None # Initialize for object type column

        if assigned_slot is None:
            # This surgery was neither explicitly scheduled nor rejected by Gurobi vars
            # This can happen if a constraint sum(x_i) + r_i = 1 is violated or Gurobi solution is incomplete.
            # Forcing it to "Rejected" might hide underlying issues.
            # Current behavior: It will be missing from assignment, leading to "Unassigned" below.
            # Consider how to handle this case: error, warning, or default to rejected.
            # For now, it will be implicitly handled by `assigned_slot` being None.
            # This surgery won't be added to schedule_rows if we want to strictly reflect solver output.
            # However, usually, we want to account for all surgeries.
            # Forcing to "Rejected" if no assignment is found by Gurobi vars for robustness:
            # print(f"[extract_schedule] Warning: Surgery index {i} (ID: {surg_data.get('id', 'N/A')}) "
            #       f"had no Gurobi assignment (x=0, r=0). Marking as 'ImpliedRejected'.")
            # row_data["AssignedDay"] = "ImpliedRejected" 
            # schedule_rows.append(row_data)
            # unassigned_in_loop_count +=1
            # continue # Or, if we want to skip it and report:
            print(f"[extract_schedule] Warning: Surgery index {i} (ID: {surg_data.get('id', 'N/A')}) "
                  f"had no decision (x=0, r=0) from Gurobi variables. Skipping from schedule output.")
            unassigned_in_loop_count += 1
            continue


        if assigned_slot == "Rejected":
            row_data["AssignedDay"] = "Rejected"
            # AssignedBlock remains None
        else: # It's a tuple (day_idx, block_idx)
            day_idx, block_idx = assigned_slot # These are already integers
            row_data["AssignedDay"] = day_idx
            if include_block:
                row_data["AssignedBlock"] = block_idx
        
        schedule_rows.append(row_data)
    
    if unassigned_in_loop_count > 0:
         print(f"[extract_schedule] Model '{model_name_for_log}': {unassigned_in_loop_count}/{len(surgeries)} "
               "surgeries had no assignment/rejection variable set to 1 in the Gurobi solution.")

    # Create DataFrame, explicitly set dtype for AssignedBlock if it exists
    # to handle mix of int and None without upcasting to float.
    schedule_df = pd.DataFrame(schedule_rows, columns=output_cols)
    
    if include_block and "AssignedBlock" in schedule_df.columns:
        # Ensure AssignedBlock can hold integers and pandas' NA, not float NaN
        # This helps prevent type issues in downstream functions.
        # pd.Int64Dtype() is the nullable integer type.
        try:
            schedule_df["AssignedBlock"] = schedule_df["AssignedBlock"].astype(pd.Int64Dtype())
        except Exception as e:
            # Fallback if Int64Dtype() fails (e.g., very old pandas or unexpected data)
            # It might remain object or get converted to float if NAs are present.
            # The per-row conversion in evaluate_schedule_actual_costs will be the safety net.
            print(f"[extract_schedule] Warning: Could not cast 'AssignedBlock' to Int64Dtype: {e}. "
                  "Column dtype might be object or float.")


    return schedule_df


def evaluate_schedule_actual_costs(
    schedule_df: pd.DataFrame,
    day_blocks: dict[int, int],
    params: dict,
) -> dict:
    """
    Computes actual operational costs (overtime, idle time, rejection) for a
    given schedule, using the true (actual) surgery durations.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        The schedule, typically output from `extract_schedule`.
        Expected columns: 'AssignedDay', 'AssignedBlock' (optional),
        'ActualDurMin', 'BookedMin'.
        'AssignedDay' contains day indices (int) or "Rejected" (str).
        'AssignedBlock' should contain block indices (int) or be missing/None.
    day_blocks : dict[int, int]
        A dictionary mapping each day index (0 to H-1) to the number of
        available OR blocks on that day.
    params : dict
        A dictionary containing configuration parameters, including:
        - 'planning_horizon_days' (H)
        - 'block_size_minutes'
        - 'cost_overtime_per_min'
        - 'cost_idle_per_min'
        - 'cost_rejection_per_case'

    Returns
    -------
    dict
        A dictionary containing various Key Performance Indicators (KPIs):
        - 'total_actual_cost': The sum of overtime, idle, and rejection costs.
        - 'scheduled': Count of surgeries successfully scheduled and costed.
        - 'rejected': Count of surgeries explicitly marked as "Rejected".
        - 'overtime_min_total': Total minutes of overtime across all blocks.
        - 'idle_min_total': Total minutes of idle time across all blocks.
        - 'overtime_matrix': A 2D numpy array (H x max_blocks) of overtime per block.
        - 'idle_matrix': A 2D numpy array (H x max_blocks) of idle time per block.
    """
    H = params["planning_horizon_days"]
    blk_min = params["block_size_minutes"]
    c_ot = params["cost_overtime_per_min"]
    c_idle = params["cost_idle_per_min"]
    c_rej = params["cost_rejection_per_case"]

    # Determine max blocks per day for matrix shape robustly
    max_b_per_day = 0
    if day_blocks and day_blocks.values():
        max_b_per_day = max(day_blocks.values())
    if H > 0 and max_b_per_day == 0 : # If horizon exists but no blocks, ensure matrices have 1 col
        max_b_per_day = 1 
    if H == 0: # No planning horizon, no capacity.
        max_b_per_day = 0


    # Initialize capacity, overtime, and idle time matrices
    # Ensure H is not zero before creating matrices of this shape.
    if H > 0:
        cap_matrix = np.zeros((H, max_b_per_day))
        for d_idx, num_d_blocks in day_blocks.items():
            if 0 <= d_idx < H: # Ensure day index is within horizon
                # Iterate up to the minimum of actual blocks for the day or matrix width
                for b_idx in range(min(num_d_blocks, max_b_per_day)):
                    cap_matrix[d_idx, b_idx] = blk_min
        ot_matrix = np.zeros_like(cap_matrix)
        idle_matrix = cap_matrix.copy() # Start with all blocks being fully idle
    else: # No planning horizon
        cap_matrix = np.array([[]]) # Empty 2D array
        ot_matrix = np.array([[]])
        idle_matrix = np.array([[]])


    rejection_cost_total = 0.0
    num_scheduled_processed = 0
    num_rejected_processed = 0
    num_skipped_rows = 0

    # Check if 'AssignedBlock' column exists for block-level accounting
    has_block_column = "AssignedBlock" in schedule_df.columns

    if schedule_df.empty and H == 0: # No schedule and no horizon means zero costs
        pass # Costs will remain zero
    elif schedule_df.empty and H > 0: # No schedule but horizon exists, all capacity is idle
        rejection_cost_total = 0.0 # No surgeries to reject
        # Idle time will be sum of all cap_matrix, ot will be 0.
        # num_scheduled_processed and num_rejected_processed remain 0.
    elif H == 0 and not schedule_df.empty: # Horizon is 0, but schedule has entries (should not happen)
        print("[evaluate_schedule_actual_costs] Warning: Schedule DataFrame is not empty, but planning horizon is 0. All scheduled cases effectively rejected.")
        for _, row_data in schedule_df.iterrows():
            if row_data["AssignedDay"] != "Rejected": # Count non-rejected as implicitly rejected due to no capacity
                 rejection_cost_total += c_rej * row_data["BookedMin"] # Or a different penalty
            else: # Already rejected
                 rejection_cost_total += c_rej * row_data["BookedMin"]
            num_rejected_processed +=1 # Count all as rejected in this edge case
    
    # Main processing loop only if H > 0 and schedule_df is not empty
    elif H > 0 and not schedule_df.empty:
        for _, row_data in schedule_df.iterrows():
            assigned_day = row_data["AssignedDay"]
            
            if assigned_day == "Rejected":
                rejection_cost_total += c_rej * row_data["BookedMin"]
                num_rejected_processed += 1
                continue

            # Process scheduled surgeries
            # Ensure assigned_day is an integer (it should be if not "Rejected")
            if not isinstance(assigned_day, (int, np.integer)):
                # print(f"  [Skipped in eval] Row with SurgID {row_data.get('SurgeryID', 'N/A')}: "
                #       f"AssignedDay '{assigned_day}' is not an integer. Skipping.")
                num_skipped_rows += 1
                continue
            
            day_idx = int(assigned_day) # Should be safe now

            if not (0 <= day_idx < H): # Check if day is within planning horizon
                # print(f"  [Skipped in eval] Row with SurgID {row_data.get('SurgeryID', 'N/A')}: "
                #       f"AssignedDay {day_idx} is out of horizon [0, {H-1}). Skipping.")
                num_skipped_rows += 1
                continue

            if has_block_column:
                raw_block_val = row_data.get("AssignedBlock")
                block_idx = None
                if pd.notna(raw_block_val): # Handles None and pd.NA
                    try:
                        # Convert to float first (to handle "20.0") then to int
                        block_idx = int(float(raw_block_val))
                    except (ValueError, TypeError):
                        # print(f"  [Skipped in eval] Row with SurgID {row_data.get('SurgeryID', 'N/A')}: "
                        #       f"Block value '{raw_block_val}' could not be converted to int. Skipping.")
                        num_skipped_rows += 1
                        continue # Skip this surgery if block is invalid
                
                # Check if block_idx is valid for the given day_idx
                if block_idx is None or not (0 <= block_idx < day_blocks.get(day_idx, 0)):
                    # print(f"  [Skipped in eval] Row with SurgID {row_data.get('SurgeryID', 'N/A')}: "
                    #       f"AssignedBlock {block_idx} is invalid for Day {day_idx} (capacity: {day_blocks.get(day_idx,0)}). Skipping.")
                    num_skipped_rows += 1
                    continue

                # If all checks pass, account for the surgery in the block
                actual_duration = row_data["ActualDurMin"]
                
                # Ensure indices are within bounds of ot_matrix/idle_matrix
                if day_idx < ot_matrix.shape[0] and block_idx < ot_matrix.shape[1]:
                    used_idle_time = min(idle_matrix[day_idx, block_idx], actual_duration)
                    idle_matrix[day_idx, block_idx] -= used_idle_time
                    overtime_duration = actual_duration - used_idle_time
                    ot_matrix[day_idx, block_idx] += overtime_duration
                    num_scheduled_processed += 1
                else:
                    # This should not happen if day_idx/block_idx checks above and matrix creation are correct
                    # print(f"  [Skipped in eval - Matrix Index] Row with SurgID {row_data.get('SurgeryID', 'N/A')}: "
                    #       f"Day {day_idx} or Block {block_idx} out of bounds for cost matrices. Skipping.")
                    num_skipped_rows += 1
                    continue

            else: # Day-level accounting (fallback if no 'AssignedBlock' column)
                # This path should ideally not be taken if extract_schedule works as intended.
                # For simplicity, this example assumes block-level data is always present.
                # If day-level accounting is needed, it would sum durations per day and compare to daily capacity.
                print("[evaluate_schedule_actual_costs] Warning: 'AssignedBlock' column missing, day-level accounting not fully implemented here.")
                num_skipped_rows +=1 # Count as skipped for now

    if num_skipped_rows > 0:
        print(f"[evaluate_schedule_actual_costs] Warning: Skipped {num_skipped_rows} rows from schedule_df during cost calculation due to invalid day/block assignments or data issues.")

    total_op_cost = ot_matrix.sum() * c_ot + idle_matrix.sum() * c_idle
    final_total_cost = total_op_cost + rejection_cost_total

    return {
        "total_actual_cost": final_total_cost,
        "scheduled": num_scheduled_processed,
        "rejected": num_rejected_processed,
        "overtime_min_total": float(ot_matrix.sum()),
        "idle_min_total": float(idle_matrix.sum()),
        "overtime_matrix": ot_matrix,
        "idle_matrix": idle_matrix,
    }


def select_surgeries(
    df_schedule_pool: pd.DataFrame,
    horizon_start: pd.Timestamp,
    params: dict,
    predictor=None, # Model object for predictions
) -> list[dict]:
    """
    Selects elective surgeries scheduled to start within the given planning horizon.
    Optionally attaches predicted durations if a predictor model is provided.

    Parameters
    ----------
    df_schedule_pool : pd.DataFrame
        DataFrame containing all surgeries available for scheduling.
        Must include 'actual_start', 'main_procedure_id', 
        'booked_time_minutes', 'procedure_duration_min', and feature columns
        if a predictor is used.
    horizon_start : pd.Timestamp
        The starting timestamp of the planning horizon.
    params : dict
        Parameter dictionary, must contain 'planning_horizon_days'.
        Also used for feature column names if predictor is active.
    predictor : model object, optional
        A trained machine learning model with a `.predict(X)` method
        for predicting surgery durations. If None, no predictions are made.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary represents a selected surgery
        and contains keys like 'id', 'proc_id', 'booked_min', 'actual_dur_min',
        and 'predicted_dur_min' (if predictor is used).
        The 'id' corresponds to the index of the surgery in `df_schedule_pool`.
    """
    H = params["planning_horizon_days"]
    # Horizon end is inclusive of the H-th day.
    # If H=7, horizon_start is Day 0, horizon_end is Day 6.
    horizon_end_date = (horizon_start + timedelta(days=H - 1)).date()
    horizon_start_date = horizon_start.date()

    # Filter surgeries that fall within the planning horizon date range
    pool_in_horizon = df_schedule_pool[
        (df_schedule_pool["actual_start"].dt.date >= horizon_start_date) &
        (df_schedule_pool["actual_start"].dt.date <= horizon_end_date)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if pool_in_horizon.empty:
        # print(
        #     f"[select_surgeries] No surgeries in pool for horizon "
        #     f"{horizon_start_date}–{horizon_end_date}."
        # )
        return []

    # Prepare features for prediction if a predictor is provided
    if predictor is not None:
        # Ensure temporal features are present for the predictor
        pool_in_horizon = add_time_features(pool_in_horizon)
        
        # Check for all required feature columns
        # ALL_FEATURES is assumed to be defined (e.g., from data_processing.py)
        missing_features = [col for col in ALL_FEATURES if col not in pool_in_horizon.columns]
        if missing_features:
            raise KeyError(f"[select_surgeries] Missing feature columns for prediction: {missing_features}")

        X_to_predict = pool_in_horizon[ALL_FEATURES].copy()
        # Ensure categorical features are strings and handle NaNs for OneHotEncoder
        for cat_col in CATEGORICAL_FEATURES: # Assumed to be defined
            X_to_predict[cat_col] = X_to_predict[cat_col].astype(str).fillna("Unknown")
        
        try:
            predictions = predictor.predict(X_to_predict).clip(min=1.0).round()
            pool_in_horizon["predicted_dur_min"] = predictions
        except Exception as e:
            print(f"[select_surgeries] Error during prediction: {e}. Predicted durations will not be available.")
            pool_in_horizon["predicted_dur_min"] = np.nan # Or some fallback

    # Construct the list of surgery dictionaries
    surgeries_for_horizon = []
    for original_idx, row_data in pool_in_horizon.iterrows():
        surgery_dict = {
            "id": original_idx, # Original index from df_schedule_pool
            "proc_id": row_data["main_procedure_id"],
            "booked_min": row_data["booked_time_minutes"],
            "actual_dur_min": row_data["procedure_duration_min"],
        }
        if predictor is not None and "predicted_dur_min" in row_data:
            predicted_val = row_data["predicted_dur_min"]
            if pd.notna(predicted_val):
                surgery_dict["predicted_dur_min"] = predicted_val
            # else:
                # If prediction failed or resulted in NaN, key won't be added
                # or could be set to a fallback like booked_min
                # surgery_dict["predicted_dur_min"] = row_data["booked_time_minutes"] 
        surgeries_for_horizon.append(surgery_dict)

    # print(
    #     f"[select_surgeries] Selected {len(surgeries_for_horizon)} surgeries "
    #     f"for {H}-day horizon starting {horizon_start_date}."
    # )
    return surgeries_for_horizon