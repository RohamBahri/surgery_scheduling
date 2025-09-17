"""
Handles data loading, cleaning, splitting, feature engineering, and preparation
for surgery scheduling models.
"""

import logging
import sys
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig, CONFIG
from src.constants import (
    DataColumns,
    FeatureColumns,
    DomainConstants,
    InternalKeys,
    LoggingConstants,
)

# Setup logger
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds week_of_year, month, year features in-place from 'actual_start'.

    Args:
        df: DataFrame to modify. Must have an 'actual_start' column of
            datetime type.

    Returns:
        The DataFrame with added time features.
    """
    if DataColumns.ACTUAL_START not in df.columns:
        raise KeyError(
            f"DataFrame must contain '{DataColumns.ACTUAL_START}' for time feature engineering."
        )
    if not pd.api.types.is_datetime64_any_dtype(df[DataColumns.ACTUAL_START]):
        logger.warning(
            f"'{DataColumns.ACTUAL_START}' column is not datetime. Attempting conversion. "
            "This might fail or lead to incorrect features if conversion is problematic."
        )
        df[DataColumns.ACTUAL_START] = pd.to_datetime(
            df[DataColumns.ACTUAL_START], errors="coerce"
        )
        if df[DataColumns.ACTUAL_START].isnull().any():
            logger.error(
                f"Coercion of '{DataColumns.ACTUAL_START}' to datetime resulted in NaT values. "
                "Time features might be incorrect or cause errors."
            )

    df[DataColumns.WEEK_OF_YEAR] = (
        df[DataColumns.ACTUAL_START].dt.isocalendar().week.astype(int)
    )
    df[DataColumns.MONTH] = df[DataColumns.ACTUAL_START].dt.month.astype(int)
    df[DataColumns.YEAR] = df[DataColumns.ACTUAL_START].dt.year.astype(int)

    return df


def load_data(config: AppConfig) -> pd.DataFrame:
    """Reads the raw Excel file, cleans data, and computes time-based features.

    Performs several cleaning steps:
    - Normalizes column names to snake_case.
    - Filters out records without actual start dates.
    - Keeps only OR names starting with a specific prefix.
    - Excludes emergency patients.
    - Parses date and time columns into proper datetime objects.
    - Computes durations (preparation, procedure).
    - Drops original separate date/time columns.
    - Filters out invalid (e.g., negative) procedure durations.
    - Recodes rare categories for procedures, surgeons, and services to 'Other'.
    - Sorts data by actual start time.

    Args:
        config: Application configuration containing data settings.

    Returns:
        A tidy DataFrame of elective surgeries.

    Raises:
        SystemExit: If the Excel file is not found or cannot be read.
        KeyError: If essential parameters for rare category recoding are missing.
    """
    excel_path = config.data.excel_file_path
    logger.info(f"Reading data from Excel file: {excel_path}")

    # Columns to use from the Excel file, specified by their original names
    excel_cols_to_use = [
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
    ]

    try:
        df = pd.read_excel(excel_path, usecols=excel_cols_to_use)
    except FileNotFoundError:
        logger.error(f"Excel file not found at path: {excel_path}")
        sys.exit(f"[load_data] ERROR – file not found: {excel_path}")
    except Exception as e:
        logger.error(f"Error reading Excel file {excel_path}: {e}")
        sys.exit(f"[load_data] ERROR while reading Excel: {e}")

    # Normalize column names to snake_case
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_|_$", "", regex=True)
    )
    logger.debug("Normalized column names to snake_case.")

    # Filter out records without an actual start date
    df = df[df[DataColumns.ACTUAL_START_DATE].notna()]

    # Keep only OR names starting with specified prefix
    df = df[
        df[DataColumns.OPERATING_ROOM].str.startswith(
            DomainConstants.OR_ROOM_PREFIX, na=False
        )
    ]

    # Exclude emergencies
    df = df[df[DataColumns.PATIENT_TYPE] != DomainConstants.EMERGENCY_PATIENT_CATEGORY]
    logger.debug(
        f"Filtered data: removed rows with no actual start date, "
        f"non-'{DomainConstants.OR_ROOM_PREFIX}' rooms, and '{DomainConstants.EMERGENCY_PATIENT_CATEGORY}'."
    )

    # Parse consult and decision dates
    df[DataColumns.CONSULT_DATE] = pd.to_datetime(
        df[DataColumns.CONSULT_DATE], errors="coerce"
    )
    df[DataColumns.DECISION_DATE] = pd.to_datetime(
        df[DataColumns.DECISION_DATE], errors="coerce"
    )

    # Helper to build timestamps from date + time columns
    def to_timestamp_series(date_col_name: str, time_col_name: str) -> pd.Series:
        date_series = pd.to_datetime(df[date_col_name], errors="coerce")
        valid_dates_mask = date_series.notna()
        ts_series = pd.Series([pd.NaT] * len(df), index=df.index)

        if valid_dates_mask.any():
            date_str = date_series[valid_dates_mask].dt.strftime("%Y-%m-%d")
            time_str = (
                df.loc[valid_dates_mask, time_col_name]
                .fillna(DomainConstants.DEFAULT_TIME_STR)
                .astype(str)
            )
            time_str_cleaned = time_str.apply(
                lambda t: t.split(".")[0] if isinstance(t, str) and "." in t else t
            )

            ts_series[valid_dates_mask] = pd.to_datetime(
                date_str + " " + time_str_cleaned, errors="coerce"
            )
        return ts_series

    df[DataColumns.ENTER_ROOM] = to_timestamp_series(
        DataColumns.ENTER_ROOM_DATE, DataColumns.ENTER_ROOM_TIME
    )
    df[DataColumns.ACTUAL_START] = to_timestamp_series(
        DataColumns.ACTUAL_START_DATE, DataColumns.ACTUAL_START_TIME
    )
    df[DataColumns.ACTUAL_STOP] = to_timestamp_series(
        DataColumns.ACTUAL_STOP_DATE, DataColumns.ACTUAL_STOP_TIME
    )
    df[DataColumns.LEAVE_ROOM] = to_timestamp_series(
        DataColumns.LEAVE_ROOM_DATE, DataColumns.LEAVE_ROOM_TIME
    )
    logger.debug("Converted date and time columns to datetime objects.")

    # Compute durations in minutes
    df[DataColumns.PREPARATION_DURATION_MIN] = (
        (df[DataColumns.ACTUAL_START] - df[DataColumns.ENTER_ROOM])
        .dt.total_seconds()
        .div(DomainConstants.SECONDS_PER_MINUTE)
        .clip(lower=0)
    )
    df[DataColumns.PROCEDURE_DURATION_MIN] = (
        (df[DataColumns.ACTUAL_STOP] - df[DataColumns.ACTUAL_START])
        .dt.total_seconds()
        .div(DomainConstants.SECONDS_PER_MINUTE)
    )
    logger.debug("Computed preparation and procedure durations.")

    # Drop the original date/time columns
    cols_to_drop_after_ts_creation = [
        DataColumns.ENTER_ROOM_DATE,
        DataColumns.ENTER_ROOM_TIME,
        DataColumns.ACTUAL_START_DATE,
        DataColumns.ACTUAL_START_TIME,
        DataColumns.ACTUAL_STOP_DATE,
        DataColumns.ACTUAL_STOP_TIME,
        DataColumns.LEAVE_ROOM_DATE,
        DataColumns.LEAVE_ROOM_TIME,
    ]
    df = df.drop(columns=cols_to_drop_after_ts_creation)
    logger.debug(
        f"Dropped original date/time string columns: {cols_to_drop_after_ts_creation}"
    )

    # Filter out invalid procedure durations
    df = df[df[DataColumns.PROCEDURE_DURATION_MIN] >= 0]
    logger.debug("Filtered out records with negative procedure durations.")

    # Rare-category recoding
    proc_threshold = config.data.min_samples_procedure
    surg_threshold = config.data.min_samples_surgeon
    serv_threshold = config.data.min_samples_service

    # Ensure columns are object/string type before conditional assignment
    categorical_cols_for_recode = [
        DataColumns.MAIN_PROCEDURE_ID,
        DataColumns.SURGEON_CODE,
        DataColumns.CASE_SERVICE,
    ]
    for col in categorical_cols_for_recode:
        df[col] = df[col].astype(object)

    def recode_rare_categories(
        series: pd.Series, threshold: int, name: str
    ) -> pd.Series:
        series = series.copy()
        counts = series.value_counts()
        rare_mask = series.map(counts).lt(threshold)
        series.loc[rare_mask] = DomainConstants.OTHER_CATEGORY
        logger.debug(
            f"Recoded rare categories in '{name}' using threshold {threshold}. "
            f"Number of '{DomainConstants.OTHER_CATEGORY}' entries: {series.eq(DomainConstants.OTHER_CATEGORY).sum()}"
        )
        return series

    df[DataColumns.MAIN_PROCEDURE_ID] = recode_rare_categories(
        df[DataColumns.MAIN_PROCEDURE_ID], proc_threshold, DataColumns.MAIN_PROCEDURE_ID
    )
    df[DataColumns.SURGEON_CODE] = recode_rare_categories(
        df[DataColumns.SURGEON_CODE], surg_threshold, DataColumns.SURGEON_CODE
    )
    df[DataColumns.CASE_SERVICE] = recode_rare_categories(
        df[DataColumns.CASE_SERVICE], serv_threshold, DataColumns.CASE_SERVICE
    )

    # Final tidy-up
    df = df.sort_values(DataColumns.ACTUAL_START).reset_index(drop=True)
    logger.info(f"Cleaned elective surgery dataset contains {len(df)} rows.")
    return df


def split_data(
    df: pd.DataFrame, config: AppConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Splits data into warm-up and scheduling pool based on specified weeks.

    The warm-up period starts from the first Monday on or after the earliest
    surgery's 'actual_start' date and lasts for 'warmup_weeks'. The scheduling
    pool contains all subsequent surgeries.

    Args:
        df: DataFrame of cleaned surgeries. Must contain 'actual_start'.
        config: Application configuration containing warmup and horizon settings.

    Returns:
        A tuple containing:
        - df_warm: DataFrame for the warm-up period.
        - df_pool: DataFrame for the scheduling pool.
        - horizon_start_date: The first Monday marking the start of the pool.

    Raises:
        ValueError: If 'actual_start' has no valid timestamps or is empty.
    """
    warmup_weeks = config.data.warmup_weeks
    horizon_days = config.data.planning_horizon_days

    if df.empty or DataColumns.ACTUAL_START not in df.columns:
        logger.error(
            f"Input DataFrame for split_data is empty or missing '{DataColumns.ACTUAL_START}' column."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.Timestamp("1970-01-01")

    # Ensure 'actual_start' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[DataColumns.ACTUAL_START]):
        df[DataColumns.ACTUAL_START] = pd.to_datetime(
            df[DataColumns.ACTUAL_START], errors="coerce"
        )

    # Find the first Monday >= min(actual_start) and floor to midnight
    earliest_ts = pd.to_datetime(df[DataColumns.ACTUAL_START].min()).normalize()
    if pd.isna(earliest_ts):
        raise ValueError(
            f"Could not determine the earliest timestamp from '{DataColumns.ACTUAL_START}'. "
            "The column might contain all NaT values after conversion."
        )
    offset_days = earliest_ts.weekday()  # Monday = 0, ..., Sunday = 6
    first_monday = earliest_ts - pd.Timedelta(days=offset_days)

    # Discard records before this first Monday
    df = df[df[DataColumns.ACTUAL_START] >= first_monday].copy()
    if df.empty:
        logger.warning(
            "No data remaining after filtering to start from the first Monday."
        )
        return pd.DataFrame(), pd.DataFrame(), first_monday

    # Warm-up end & horizon start, both at 00:00 on a Monday
    warmup_end_date = (
        first_monday + pd.Timedelta(weeks=warmup_weeks) - pd.Timedelta(days=1)
    )
    pool_start_date = warmup_end_date + pd.Timedelta(days=1)  # This is the next Monday

    # Ensure at least one horizon fits after warm-up
    valid_starts = df[DataColumns.ACTUAL_START].dropna()
    if valid_starts.empty:
        raise ValueError(
            f"No valid '{DataColumns.ACTUAL_START}' timestamps found after initial filtering."
        )

    # The last possible day a horizon_days long period can start
    last_possible_horizon_start = valid_starts.max().normalize() - pd.Timedelta(
        days=horizon_days - 1
    )

    if pool_start_date > last_possible_horizon_start:
        logger.warning(
            "Warm-up period extends beyond available data for a full horizon. "
            "Clipping warm-up period to fit."
        )
        pool_start_date = last_possible_horizon_start
        warmup_end_date = pool_start_date - pd.Timedelta(days=1)
        if pool_start_date < first_monday:
            logger.error(
                "Cannot define a valid warm-up and pool period with current data and parameters."
            )
            return pd.DataFrame(), pd.DataFrame(), first_monday

    # Slice the DataFrame
    df_warm = df[df[DataColumns.ACTUAL_START] < pool_start_date].copy()
    df_pool = df[df[DataColumns.ACTUAL_START] >= pool_start_date].copy()

    if df_warm.empty:
        logger.warning("Warm-up data split resulted in an empty DataFrame.")
    if df_pool.empty:
        logger.warning("Scheduling pool data split resulted in an empty DataFrame.")

    logger.info(
        f"Data split complete. Warm-up end: {warmup_end_date.date()} | "
        f"Pool start: {pool_start_date.date()} | "
        f"Warm-up rows: {len(df_warm)} | Pool rows: {len(df_pool)}"
    )
    return df_warm, df_pool, pool_start_date


def compute_block_capacity(
    df_week: pd.DataFrame, config: AppConfig, horizon_start_date: pd.Timestamp
) -> Dict[int, int]:
    """Calculates OR block capacity for each day in a planning horizon.

    Capacity is based on the number of distinct operating rooms active on each
    day in `df_week` (historical data), potentially reduced by a percentage.

    Args:
        df_week: DataFrame of surgeries for the current horizon window.
            Must contain 'operating_room' and 'actual_start'.
        config: Application configuration containing capacity settings.
        horizon_start_date: Start date of the planning horizon.

    Returns:
        A dictionary mapping day index (0 to H-1) to the number of
        available OR blocks for that day.

    Raises:
        KeyError: If required columns in `df_week` are missing.
        ValueError: If 'capacity_reduction_percentage' is out of bounds.
    """
    planning_horizon_days = config.data.planning_horizon_days
    horizon_start_dt = horizon_start_date.normalize()
    reduction_percentage = config.operating_room.capacity_reduction_percentage
    min_blocks_after_reduction = config.operating_room.min_blocks_after_reduction

    if not (0.0 <= reduction_percentage <= 1.0):
        msg = (
            f"capacity_reduction_percentage must be between 0.0 and 1.0, "
            f"got {reduction_percentage}"
        )
        logger.error(msg)
        raise ValueError(msg)

    required_cols = {DataColumns.OPERATING_ROOM, DataColumns.ACTUAL_START}
    missing_cols = required_cols - set(df_week.columns)
    if missing_cols:
        msg = f"df_week is missing required columns for capacity computation: {missing_cols}"
        logger.error(msg)
        raise KeyError(msg)

    if df_week.empty:
        logger.warning(
            f"df_week is empty for horizon starting {horizon_start_dt.date()}. "
            "Returning 0 blocks for all days."
        )
        return {d_idx: 0 for d_idx in range(planning_horizon_days)}

    df_current_week = df_week.copy()
    df_current_week[DataColumns.ACTUAL_START] = pd.to_datetime(
        df_current_week[DataColumns.ACTUAL_START], errors="coerce"
    )
    df_current_week.dropna(subset=[DataColumns.ACTUAL_START], inplace=True)

    if df_current_week.empty:
        logger.warning(
            f"df_week has no valid '{DataColumns.ACTUAL_START}' dates for horizon starting "
            f"{horizon_start_dt.date()}. Returning 0 blocks for all days."
        )
        return {d_idx: 0 for d_idx in range(planning_horizon_days)}

    # Count distinct ORs per calendar day (historical capacity)
    historical_blocks_per_date = (
        df_current_week.groupby(
            df_current_week[DataColumns.ACTUAL_START].dt.normalize()
        )[DataColumns.OPERATING_ROOM]
        .nunique()
        .astype(int)
    )

    daily_block_capacity: Dict[int, int] = {}
    horizon_end_date = horizon_start_dt + timedelta(days=planning_horizon_days - 1)
    logger.info(
        f"Computing block capacity for horizon: {horizon_start_dt.date()}–"
        f"{horizon_end_date.date()}, Reduction: {reduction_percentage*100:.1f}%"
    )

    for d_idx in range(planning_horizon_days):
        current_date_in_horizon = horizon_start_dt + pd.Timedelta(days=d_idx)
        num_historical_blocks = historical_blocks_per_date.get(
            current_date_in_horizon, 0
        )

        reduced_blocks_float = num_historical_blocks * (1.0 - reduction_percentage)
        num_reduced_blocks_int = max(
            min_blocks_after_reduction, np.floor(reduced_blocks_float).astype(int)
        )
        daily_block_capacity[d_idx] = num_reduced_blocks_int

        log_msg_detail = (
            f"  Day {d_idx} ({current_date_in_horizon.date()}): "
            f"Historical ORs={num_historical_blocks}"
        )
        if reduction_percentage > 0.0:
            log_msg_detail += f" -> Reduced Blocks={num_reduced_blocks_int}"
        else:
            log_msg_detail += f" -> Blocks={num_reduced_blocks_int}"
        logger.debug(log_msg_detail)

    return daily_block_capacity


def build_feature_vector(
    surgery_dict: Dict[str, Any],
    df_pool_full_data: pd.DataFrame,
    feature_names_ordered: List[str],
) -> np.ndarray:
    """Constructs a 1D feature vector for a single surgery using standardized scaling.

    The output vector's length and order exactly match `feature_names_ordered`,
    which is typically derived from the training pipeline's feature set.
    Numeric features are always scaled using standardization (z-score).

    Args:
        surgery_dict: Dictionary representing the surgery. Must contain 'id'
            (index in `df_pool_full_data`). Can optionally override features.
        df_pool_full_data: The full DataFrame for scaling parameters and lookup.
        feature_names_ordered: List of all feature names in exact order
            expected by the ML model.

    Returns:
        A 1D NumPy array representing the feature vector.

    Raises:
        KeyError: If surgery 'id' not in `df_pool_full_data` or if required
            base features are missing.
        AssertionError: If the generated vector length mismatch `feature_names_ordered`.
    """
    surgery_id = surgery_dict["id"]
    try:
        surgery_row_from_pool = df_pool_full_data.loc[surgery_id]
    except KeyError:
        logger.error(f"Surgery ID {surgery_id} not found in df_pool_full_data.")
        raise

    # Numeric features: booked_time, week, month, year
    booked_duration = float(
        surgery_dict.get(
            DataColumns.BOOKED_MIN, surgery_row_from_pool[DataColumns.BOOKED_MIN]
        )
    )

    actual_start_ts = pd.to_datetime(
        surgery_dict.get(
            DataColumns.ACTUAL_START, surgery_row_from_pool[DataColumns.ACTUAL_START]
        ),
        errors="raise",
    )

    numeric_raw_values = np.array(
        [
            booked_duration,
            actual_start_ts.isocalendar().week,
            actual_start_ts.month,
            actual_start_ts.year,
        ],
        dtype=float,
    )

    # Ensure calendar columns exist in df_pool_full_data for mu/sigma cache calculation
    time_feature_cols_for_cache = [
        DataColumns.WEEK_OF_YEAR,
        DataColumns.MONTH,
        DataColumns.YEAR,
    ]
    if not all(col in df_pool_full_data.columns for col in time_feature_cols_for_cache):
        logger.debug("Adding time features to df_pool_full_data for scaling cache.")
        df_pool_full_data_copy = df_pool_full_data.copy()
        df_pool_full_data_copy = add_time_features(df_pool_full_data_copy)

    # Always use standardization scaling
    if InternalKeys.BFV_MU_SIGMA_CACHE_KEY not in df_pool_full_data.attrs:
        logger.debug(
            f"'{InternalKeys.BFV_MU_SIGMA_CACHE_KEY}' not in df_pool_full_data.attrs. Calculating mu/sigma."
        )
        cols_for_scaling_stats = FeatureColumns.NUMERIC

        temp_df_for_stats = df_pool_full_data.copy()
        if not all(
            col in temp_df_for_stats.columns
            for col in [DataColumns.WEEK_OF_YEAR, DataColumns.MONTH, DataColumns.YEAR]
        ):
            temp_df_for_stats = add_time_features(temp_df_for_stats)

        mu = temp_df_for_stats[cols_for_scaling_stats].mean().to_numpy()
        sd = (
            temp_df_for_stats[cols_for_scaling_stats]
            .std(ddof=0)
            .replace(0, 1)
            .to_numpy()
        )
        df_pool_full_data.attrs[InternalKeys.BFV_MU_SIGMA_CACHE_KEY] = (mu, sd)
        logger.debug(
            f"Cached mu/sigma for numeric features using columns: {cols_for_scaling_stats}"
        )

    mu_cached, sd_cached = df_pool_full_data.attrs[InternalKeys.BFV_MU_SIGMA_CACHE_KEY]
    numeric_vector_scaled = (numeric_raw_values - mu_cached) / sd_cached

    # Categorical features (one-hot encoded part)
    num_numeric_feats = len(FeatureColumns.NUMERIC)
    categorical_feature_names_ordered = feature_names_ordered[num_numeric_feats:]

    # Construct the "active" one-hot encoded feature names for this surgery
    active_categorical_feature_flags = set()
    for cat_col_base_name in FeatureColumns.CATEGORICAL:
        value = surgery_dict.get(
            cat_col_base_name, surgery_row_from_pool[cat_col_base_name]
        )
        active_categorical_feature_flags.add(f"{cat_col_base_name}_{value}")

    categorical_vector = np.fromiter(
        (
            1.0 if cat_feat_name in active_categorical_feature_flags else 0.0
            for cat_feat_name in categorical_feature_names_ordered
        ),
        dtype=float,
        count=len(categorical_feature_names_ordered),
    )

    final_feature_vector = np.concatenate([numeric_vector_scaled, categorical_vector])
    if final_feature_vector.size != len(feature_names_ordered):
        msg = (
            f"Feature vector length mismatch. Expected {len(feature_names_ordered)}, "
            f"got {final_feature_vector.size}. Surgery ID: {surgery_id}."
        )
        logger.error(msg)
        raise AssertionError(msg)
    return final_feature_vector


def attach_pred(
    surgery_list: List[Dict[str, Any]],
    model_or_function: Any,
    df_pool_reference: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Attaches 'predicted_dur_min' to each surgery dict in a list.

    Predictions are generated using the provided model or function.
    If `model_or_function` is a scikit-learn model, it expects a DataFrame of
    features. If it's a custom function, it's assumed to take a surgery
    dictionary and return a prediction.

    Args:
        surgery_list: List of surgery dictionaries. Each must have an 'id'
            key corresponding to an index in `df_pool_reference`.
        model_or_function: A trained scikit-learn model (with a .predict method)
            or a callable function.
        df_pool_reference: DataFrame used to source features.

    Returns:
        The input `surgery_list` with 'predicted_dur_min' added/updated in each
        dictionary. Predictions are clipped to be >= MIN_PROCEDURE_DURATION.

    Raises:
        KeyError: If surgery IDs from `surgery_list` are not found in
            `df_pool_reference` when building features for an sklearn model.
    """    
    if not surgery_list or model_or_function is None:
        logger.debug(
            "attach_pred: No surgeries or no model provided, returning list as is."
        )
        return surgery_list

    predictions_array: np.ndarray

    # Determine if model_or_function is an sklearn-like model or a custom function
    is_sklearn_model = hasattr(model_or_function, "predict") and callable(
        getattr(model_or_function, "predict")
    )

    if is_sklearn_model:
        logger.debug("Using scikit-learn compatible model for predictions.")
        surgery_ids = [s["id"] for s in surgery_list]
        try:
            features_df_subset = df_pool_reference.loc[surgery_ids].copy()
        except KeyError as e:
            missing_ids = list(set(surgery_ids) - set(df_pool_reference.index))
            logger.error(
                f"attach_pred: Unknown surgery IDs in df_pool_reference: {missing_ids}"
            )
            raise KeyError(f"[attach_pred] Unknown surgery IDs: {e}") from e

        # Ensure necessary features are present
        features_df_subset = add_time_features(features_df_subset)

        # Ensure categorical features are string type and fill NaNs
        for cat_col in FeatureColumns.CATEGORICAL:
            if cat_col in features_df_subset.columns:
                features_df_subset[cat_col] = (
                    features_df_subset[cat_col]
                    .astype(str)
                    .fillna(DomainConstants.UNKNOWN_CATEGORY)
                )
            else:
                logger.warning(
                    f"Categorical feature '{cat_col}' not found in subset for prediction."
                )

        # Select all required features
        feature_columns_for_model = [
            col for col in FeatureColumns.ALL if col in features_df_subset.columns
        ]
        if len(feature_columns_for_model) != len(FeatureColumns.ALL):
            logger.warning(
                f"Not all expected features ({FeatureColumns.ALL}) were found in the "
                f"DataFrame subset for prediction. Using: {feature_columns_for_model}"
            )

        X_features = features_df_subset[feature_columns_for_model]
        predictions_array = model_or_function.predict(X_features)
    else:  # Custom function
        logger.debug("Using custom callable function for predictions.")
        predictions_list = [model_or_function(surg) for surg in surgery_list]
        predictions_array = np.array(predictions_list, dtype=float)

    # Post-process predictions: clip at min duration, round, convert to float
    processed_predictions = (
        np.maximum(predictions_array, DomainConstants.MIN_PROCEDURE_DURATION)
        .round()
        .astype(float)
    )

    # Additional trimming to reasonable bounds
    max_reasonable_duration = CONFIG.operating_room.block_size_minutes * 2  # 2x block size max
    processed_predictions = np.clip(
        processed_predictions, 
        DomainConstants.MIN_PROCEDURE_DURATION,
        max_reasonable_duration
    )
    
    # Round to nearest 5 minutes for cleaner optimization
    processed_predictions = np.round(processed_predictions / 5) * 5

    # Attach predictions to the surgery dictionaries
    for surgery_dict, pred_value in zip(surgery_list, processed_predictions, strict=True):
        surgery_dict["predicted_dur_min"] = float(pred_value)  # Ensure clean float

    logger.debug(f"Attached predictions to {len(surgery_list)} surgeries.")
    return surgery_list
