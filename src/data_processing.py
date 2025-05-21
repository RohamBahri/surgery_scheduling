import pandas as pd
import sys
from datetime import timedelta
import numpy as np # Added for np.floor


NUMERIC_FEATURES = [
    "booked_time_minutes",
    "week_of_year",
    "month",
    "year",
]

CATEGORICAL_FEATURES = [
    "patient_type",
    "main_procedure_id",
    "surgeon_code",
    "case_service",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds week_of_year, month, year in‑place and returns df."""
    df["week_of_year"] = df["actual_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["actual_start"].dt.month.astype(int)
    df["year"] = df["actual_start"].dt.year.astype(int)
    return df


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(params):
    """
    Reads the raw Excel file, cleans obvious data issues, and returns a tidy
    DataFrame of elective surgeries with snake_case columns, computed timestamps,
    and durations.

    Assumes params contains:
      - excel_file_path
      - min_samples_procedure
      - min_samples_surgeon
      - min_samples_service
    """

    path = params["excel_file_path"]
    print(f"[load_data] Reading {path}")
    # 1) Read raw data, only file‐not‐found and read errors handled here
    try:
        df = pd.read_excel(
            path,
            usecols=[
                "Patient_Type",
                "Case_Service",
                "Main_Procedure",
                "Main_Procedure_Id",
                "Operating_Room",
                "Consult_Date",
                "Decision_Date",
                "Booked Time (Minutes)",
                "Enter Room Date",
                "Enter Room Time",
                "Actual Start Date",
                "Actual Start Time",
                "Actual Stop Date",
                "Actual Stop Time",
                "Leave Room Date",
                "Leave Room Time",
                "CMG",
                "CMG Description",
                "Anaesthetic_Type_Given",
                "Start_Delay_Reason",
                "Patient_Disposition from OR",
                "Admit Recovery Date",
                "Admit Recovery Time",
                "Leave Recovery Date",
                "Leave Recovery Time",
                "Recovery_Time_Mins",
                "Case_Cancelled_Reason",
                "Case Cancel Date",
                "Case Cancel Time",
                "Patient_ID",
                "Surgeon",
                "Surgeon_Code",
            ],
        )
    except FileNotFoundError:
        sys.exit(f"[load_data] ERROR – file not found: {path}")
    except Exception as e:
        sys.exit(f"[load_data] ERROR while reading Excel: {e}")

    # 2) Normalize column names to snake_case
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )

    # 3) Filter out records without an actual start date
    df = df[df["actual_start_date"].notna()]

    # 4) Keep only OR names starting with "OR"
    df = df[df["operating_room"].str.startswith("OR", na=False)]

    # 5) Exclude emergencies
    df = df[df["patient_type"] != "EMERGENCY PATIENT"]

    # 6) Parse consult and decision dates
    df["consult_date"] = pd.to_datetime(df["consult_date"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

    # 7) Build timestamps from date + time columns
    def to_ts(date_col, time_col):
        # Ensure df is the one being modified, not a global one if this were nested differently
        # For Series, .dt.strftime works on datetime-like Series.
        # If df[date_col] is already datetime, this is fine. If not, to_datetime first.
        d_series = pd.to_datetime(df[date_col], errors="coerce")
        # Filter out NaT before strftime to avoid errors/warnings
        valid_dates_mask = d_series.notna()
        
        # Initialize result series
        ts_series = pd.Series([pd.NaT] * len(df), index=df.index)

        # Process only valid dates
        if valid_dates_mask.any():
            d_str = d_series[valid_dates_mask].dt.strftime("%Y-%m-%d")
            t_str = df.loc[valid_dates_mask, time_col].fillna("00:00:00").astype(str)
            ts_series[valid_dates_mask] = pd.to_datetime(d_str + " " + t_str, errors="coerce")
        return ts_series


    df["enter_room"] = to_ts("enter_room_date", "enter_room_time")
    df["actual_start"] = to_ts("actual_start_date", "actual_start_time")
    df["actual_stop"] = to_ts("actual_stop_date", "actual_stop_time")
    df["leave_room"] = to_ts("leave_room_date", "leave_room_time")
    df["admit_recovery"] = to_ts("admit_recovery_date", "admit_recovery_time")
    df["leave_recovery"] = to_ts("leave_recovery_date", "leave_recovery_time")

    # 8) Compute durations in minutes
    df["preparation_duration_min"] = (
        (df["actual_start"] - df["enter_room"]).dt.total_seconds().div(60).clip(lower=0)
    )

    df["procedure_duration_min"] = (
        (df["actual_stop"] - df["actual_start"]).dt.total_seconds().div(60)
    )

    # 9) Drop the original date/time columns
    df = df.drop(
        [
            "enter_room_date",
            "enter_room_time",
            "actual_start_date",
            "actual_start_time",
            "actual_stop_date",
            "actual_stop_time",
            "leave_room_date",
            "leave_room_time",
            "admit_recovery_date",
            "admit_recovery_time",
            "leave_recovery_date",
            "leave_recovery_time",
        ],
        axis=1,
    )

    # 10) Filter out invalid procedure durations
    df = df[df["procedure_duration_min"] >= 0]

    # 11) Rare‐category recoding using thresholds in params
    proc_thr = params["min_samples_procedure"]
    surg_thr = params["min_samples_surgeon"]
    serv_thr = params["min_samples_service"]

    # Ensure columns are object/string type BEFORE conditional assignment
    df["main_procedure_id"] = df["main_procedure_id"].astype(object)
    df["surgeon_code"] = df["surgeon_code"].astype(object)
    df["case_service"] = df["case_service"].astype(object)

    proc_counts = df["main_procedure_id"].value_counts()
    df.loc[
        df["main_procedure_id"].map(proc_counts).lt(proc_thr),
        "main_procedure_id",
    ] = "Other"

    surg_counts = df["surgeon_code"].value_counts()
    df.loc[
        df["surgeon_code"].map(surg_counts).lt(surg_thr),
        "surgeon_code",
    ] = "Other"

    serv_counts = df["case_service"].value_counts()
    df.loc[
        df["case_service"].map(serv_counts).lt(serv_thr),
        "case_service",
    ] = "Other"

    # 12) Final tidy‐up
    df = df.sort_values("actual_start").reset_index(drop=True)
    print(f"[load_data] Cleaned elective dataset contains {len(df)} rows.")

    return df


# =============================================================================
# DATA SPLITTING
# =============================================================================
def split_data(df: pd.DataFrame, params: dict):
    """
    Split the cleaned data into:
      • warm‑up period  – the first <warmup_weeks> whole weeks, starting Monday
      • scheduling pool – observations that follow
    """
    warmup_weeks = params.get("warmup_weeks")
    horizon_days = params.get("planning_horizon_days")
    if warmup_weeks is None or horizon_days is None:
        raise KeyError(
            "split_data requires 'warmup_weeks' and 'planning_horizon_days' in params"
        )

    # ------------------------------------------------------------------ #
    # 1)  Find the first Monday ≥ min(actual_start) and floor to midnight
    # ------------------------------------------------------------------ #
    earliest_ts = pd.to_datetime(df["actual_start"].min()).normalize()
    offset_days = earliest_ts.weekday()  # Monday = 0 … Sunday = 6
    first_monday = earliest_ts - pd.Timedelta(days=offset_days)

    # Discard any records before that Monday (user requested)
    df = df[df["actual_start"] >= first_monday].copy()

    # ------------------------------------------------------------------ #
    # 2)  Warm‑up end & horizon start, both at 00:00 on a Monday
    # ------------------------------------------------------------------ #
    warmup_end = first_monday + pd.Timedelta(weeks=warmup_weeks) - pd.Timedelta(days=1)
    horizon_start = warmup_end + pd.Timedelta(days=1)

    # ------------------------------------------------------------------ #
    # 3)  Ensure at least one horizon fits after warm‑up
    # ------------------------------------------------------------------ #
    # Ensure df["actual_start"] has valid timestamps for max()
    valid_starts = df["actual_start"].dropna()
    if valid_starts.empty:
        raise ValueError("No valid 'actual_start' timestamps found in the DataFrame for splitting.")
        
    last_possible_start = valid_starts.max() - pd.Timedelta(days=horizon_days - 1)
    if horizon_start > last_possible_start:
        warmup_end = last_possible_start - pd.Timedelta(days=1)
        horizon_start = last_possible_start
        print("[split_data] Warm‑up period clipped to fit the dataset.")

    # ------------------------------------------------------------------ #
    # 4)  Slice the DataFrame
    # ------------------------------------------------------------------ #
    df_warm = df[df["actual_start"] < horizon_start].copy()
    df_pool = df[df["actual_start"] >= horizon_start].copy()

    if df_warm.empty:
        print("[split_data] WARNING – warm‑up data is empty.")
    if df_pool.empty:
        print("[split_data] WARNING – scheduling‑pool data is empty.")

    print(
        f"[split_data] Warm‑up end: {warmup_end.date()}  |  "
        f"pool start: {horizon_start.date()}  |  "
        f"warm‑up rows: {len(df_warm)}  |  pool rows: {len(df_pool)}"
    )
    return df_warm, df_pool, horizon_start


# =============================================================================
# DYNAMIC CAPACITY ALLOCATION
# =============================================================================
def compute_block_capacity(df_week: pd.DataFrame, params: dict) -> dict[int, int]:
    """
    Return the number of 8‑hour OR blocks available on each day of the
    planning horizon, based on distinct operating rooms active that day in
    df_week, optionally reduced by a specified percentage.

    Parameters
    ----------
    df_week : pd.DataFrame
        DataFrame of surgeries for the current horizon window. Must contain
        'operating_room' (str) and 'actual_start' (Timestamp) columns.
        Used to determine historical capacity.
    params : dict
        Configuration dictionary. Must contain:
        - 'planning_horizon_days' (int): The number of days in the horizon.
        - '_horizon_start_date' (pd.Timestamp): The start date of the horizon.
        - 'capacity_reduction_percentage' (float): Percentage (0.0 to 1.0)
          by which to reduce the historical block capacity. 0.0 means no reduction.
        - 'min_blocks_after_reduction' (int): The minimum number of blocks
          a day can have after reduction is applied.

    Returns
    -------
    dict[int, int]
        A dictionary mapping the day index (0 to H-1) to the calculated
        number of available OR blocks for that day.
    """
    H = params["planning_horizon_days"]
    start_date = params["_horizon_start_date"].normalize()
    reduction_percentage = params.get("capacity_reduction_percentage", 0.0)
    min_blocks_after_reduction = params.get("min_blocks_after_reduction", 0)

    if not (0.0 <= reduction_percentage <= 1.0):
        raise ValueError(
            f"capacity_reduction_percentage must be between 0.0 and 1.0, got {reduction_percentage}"
        )

    if df_week.empty:
        # If df_week is empty, no historical surgeries for this period,
        # implying 0 historical ORs active. Thus, 0 blocks.
        print(f"[compute_block_capacity] df_week is empty for horizon starting {start_date.date()}. "
              "Returning 0 blocks for all days.")
        return {d: 0 for d in range(H)}

    # --- 1) Sanity checks for required columns in df_week ---
    required_cols = {"operating_room", "actual_start"}
    missing_cols = required_cols - set(df_week.columns)
    if missing_cols:
        raise KeyError(f"compute_block_capacity: df_week is missing columns: {missing_cols}")

    # --- 2) Prepare df_week for processing ---
    df_current_week = df_week.copy()
    df_current_week["actual_start"] = pd.to_datetime(df_current_week["actual_start"], errors="coerce")
    # Drop rows where actual_start could not be parsed, as they can't be grouped by date
    df_current_week.dropna(subset=["actual_start"], inplace=True)
    
    if df_current_week.empty: # If all rows had invalid actual_start
        print(f"[compute_block_capacity] df_week has no valid 'actual_start' dates for horizon starting {start_date.date()}. "
              "Returning 0 blocks for all days.")
        return {d: 0 for d in range(H)}


    # --- 3) Count distinct ORs per calendar day (historical capacity) ---
    historical_blocks_per_date = (
        df_current_week.groupby(df_current_week["actual_start"].dt.normalize())["operating_room"]
        .nunique()
        .astype(int)
    )

    # --- 4) Map to horizon day indices and apply reduction ---
    day_blocks_final = {}
    print(f"[compute_block_capacity] Horizon: {start_date.date()}–{(start_date + timedelta(days=H-1)).date()}, "
          f"Reduction: {reduction_percentage*100:.1f}%")

    for d_idx in range(H):
        current_date_in_horizon = start_date + pd.Timedelta(days=d_idx)
        # Get historical blocks for this specific date, default to 0 if date not in historical_blocks_per_date
        num_historical_blocks = historical_blocks_per_date.get(current_date_in_horizon, 0)
        
        # Apply reduction
        reduced_blocks_float = num_historical_blocks * (1.0 - reduction_percentage)
        
        # Floor the result to ensure reduction is effective, then ensure it's not below min_blocks
        num_reduced_blocks_int = max(min_blocks_after_reduction, np.floor(reduced_blocks_float).astype(int))
        
        day_blocks_final[d_idx] = num_reduced_blocks_int
        
        if reduction_percentage > 0.0:
            print(f"  Day {d_idx} ({current_date_in_horizon.date()}): Historical Blocks={num_historical_blocks} "
                  f"-> Reduced Blocks={num_reduced_blocks_int}")
        else:
            print(f"  Day {d_idx} ({current_date_in_horizon.date()}): Blocks={num_historical_blocks}")
            
    return day_blocks_final


def attach_pred(surg_list: list[dict], model, df_pool: pd.DataFrame):
    """
    Attaches predicted durations to a list of surgery dictionaries.

    For each surgery in `surg_list`, this function retrieves its features
    from `df_pool`, uses the provided `model` to predict its duration,
    and adds/updates the "predicted_dur_min" key in the surgery's dictionary.

    Parameters
    ----------
    surg_list : list[dict]
        A list of surgery dictionaries. Each dictionary must contain an 'id'
        key corresponding to its index in `df_pool`.
    model : scikit-learn compatible model
        A trained model object with a `.predict(X)` method. If `model` is None,
        the function returns `surg_list` unmodified.
    df_pool : pd.DataFrame
        The DataFrame containing all available surgeries and their features,
        indexed such that `df_pool.loc[surgery_id]` retrieves the correct row.
        It must contain all features required by the model.

    Returns
    -------
    list[dict]
        The input `surg_list` with "predicted_dur_min" added/updated for
        each surgery dictionary.

    Notes
    -----
    - The order of `surg_list` is preserved.
    - Uses `add_time_features` to ensure temporal features are available.
    - Predictions are clipped to a minimum of 1.0 and rounded.
    - If a `pred_debug/output.json` exists, it will be overwritten.
    """
    if not surg_list: # Handle empty list early
        return []
        
    if model is None:
        # This path is for deterministic/oracle models that don't use ML predictions
        # for planning. The 'predicted_dur_min' key might not be needed or will be
        # handled differently by the solver wrappers for those cases.
        return surg_list

    # 1) Get IDs and ensure they are valid for iloc/loc
    surgery_ids = [s["id"] for s in surg_list]
    
    # Create a DataFrame of features for the selected surgeries, maintaining order
    # Using .loc[list_of_ids] preserves the order of the list_of_ids
    try:
        features_df = df_pool.loc[surgery_ids].copy()
    except KeyError as e:
        print(f"[attach_pred] Error: Some surgery IDs not found in df_pool. {e}")
        # Fallback or error handling:
        # Option 1: Raise error
        # raise
        # Option 2: Return list as is, possibly with warning
        print("[attach_pred] Warning: Could not retrieve all features; returning list without predictions.")
        return surg_list


    # 2) Ensure temporal features exist and get the feature matrix X
    features_df = add_time_features(features_df) # Adds 'week_of_year', 'month', 'year'

    # Verify all necessary features are present
    # ALL_FEATURES should be defined globally or passed via params
    missing_model_features = [col for col in ALL_FEATURES if col not in features_df.columns]
    if missing_model_features:
        raise KeyError(f"[attach_pred] DataFrame for prediction is missing features: {missing_model_features}")

    X_predict = features_df[ALL_FEATURES].copy()

    # 3) Preprocess categorical features (cast to string, fill NaNs)
    # CATEGORICAL_FEATURES should be defined globally or passed
    for cat_col in CATEGORICAL_FEATURES:
        X_predict[cat_col] = X_predict[cat_col].astype(str).fillna("Unknown")

    # 4) Predict durations, clip, and round
    try:
        predicted_durations = model.predict(X_predict).clip(min=1.0).round()
    except Exception as e:
        print(f"[attach_pred] Error during model.predict(): {e}")
        # Fallback: e.g., use booked_time_minutes or skip adding predictions
        # For now, let's just not add the key if prediction fails for all
        for sg_dict in surg_list:
            sg_dict.pop("predicted_dur_min", None) # Remove if it was there from a previous failed attempt
        return surg_list


    # 5) Attach predictions back to the original list of dictionaries
    # This ensures that surg_list (which might be a deepcopy) gets updated.
    for i, sg_dict in enumerate(surg_list):
        sg_dict["predicted_dur_min"] = predicted_durations[i]


    # --- Optional Debug Output ---
    # This part writes the surg_list (now with predictions) to a JSON file.
    # Useful for inspecting the predictions alongside other surgery data.
    # Consider making this conditional via a debug flag in params.
    # if params.get("debug_attach_pred_output", False):
    #     import json
    #     from pathlib import Path
    #     out_dir = Path("pred_debug")
    #     out_dir.mkdir(exist_ok=True)
    #     # Create a more unique filename if running multiple horizons/experiments
    #     # For example, using a timestamp or a counter.
    #     model_name_part = model.__class__.__name__ if hasattr(model, "__class__") else "unknown_model"
    #     fname = out_dir / f"predictions_{model_name_part}.json" 
    #     try:
    #         with open(fname, "w", encoding="utf-8") as f:
    #             json.dump(surg_list, f, indent=2, default=str) # default=str for non-serializable like Timestamps
    #         print(f"[attach_pred] Debug predictions saved to {fname}")
    #     except Exception as e_json:
    #         print(f"[attach_pred] Error saving debug JSON: {e_json}")
    # --- End Optional Debug Output ---

    return surg_list