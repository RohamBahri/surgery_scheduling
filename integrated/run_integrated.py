# --- Make project root importable when run as a script from integrated/ ---
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple # Ensure Tuple and List are imported

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------------------------------

import json
import logging
import pandas as pd
import numpy as np # For initial_theta_idx if coefs is numpy array

from src.config import PARAMS
from src.constants import (
    ALL_FEATURE_COLS,
    DEFAULT_LOGGER_NAME,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    COL_PROCEDURE_DURATION_MIN, # Needed for prepare_weekly_data...
    MIN_PROCEDURE_DURATION,     # Needed for prepare_weekly_data...
    COL_ACTUAL_START,           # Needed for prepare_weekly_data...
    HORIZON_START_DATE_PARAM_KEY,# Needed for prepare_weekly_data...
    NUMERIC_FEATURE_COLS,       # For fallback feature name generation
    CATEGORICAL_FEATURE_COLS,   # For fallback feature name generation
    UNKNOWN_CATEGORY            # For fallback feature name generation
)
from src.data_processing import (
    load_data,
    split_data,
    compute_block_capacity, # Used in prepare_weekly_data...
    add_time_features # For robust feature name generation fallback
)
from src.scheduling_utils import select_surgeries # Used in prepare_weekly_data...
from src.predictors import train_lasso_predictor
from integrated.master_problem import build_benders_master_problem
from integrated.callbacks import BendersThetaLearningCallback # Corrected class name
from integrated.sub_problem import solve_weekly_subproblem # Correct function name
from src.solver_utils import set_gurobi_model_parameters # For master model params

# Type alias for the enhanced weekly data tuple
EnhancedWeeklyDataTypeForRun = Tuple[List[Dict[str, Any]], Dict[int, int], Dict[Any, float]]

logger = logging.getLogger(DEFAULT_LOGGER_NAME)

def prepare_weekly_data_for_benders_training(
    df_full_training_dataset: pd.DataFrame,
    params_config: Dict[str, Any]
) -> List[EnhancedWeeklyDataTypeForRun]:
    if df_full_training_dataset.empty:
        logger.error("Training dataset for Benders is empty. Cannot prepare weekly data.")
        return []
        
    earliest_ts_in_training_data = pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START].min()).normalize()
    offset_days = earliest_ts_in_training_data.weekday()
    first_monday_of_training_period = earliest_ts_in_training_data - pd.Timedelta(days=offset_days)
    
    num_training_weeks = params_config.get("warmup_weeks", 52)
    weekly_benders_data: List[EnhancedWeeklyDataTypeForRun] = []

    logger.info(f"Preparing weekly data for {num_training_weeks} training weeks for GTSM-Benders.")
    for week_num in range(num_training_weeks):
        current_week_start_date = first_monday_of_training_period + pd.Timedelta(weeks=week_num)
        
        params_for_selection = {**params_config, "planning_horizon_days": 7}
        surgeries_for_week_list = select_surgeries(
            df_full_training_dataset, current_week_start_date, params_for_selection
        )

        if not surgeries_for_week_list:
            logger.debug(f"Training Week {week_num+1} (starting {current_week_start_date.date()}) has no surgeries. Skipping data prep for this week.")
            # Still might need to add an empty entry if master problem expects W eta vars
            # For now, skipping means num_training_weeks fed to master might be less than PARAMS["warmup_weeks"]
            # This should be fine if master_problem takes len(weekly_data_list)
            continue

        current_week_end_date = current_week_start_date + pd.Timedelta(days=6)
        historical_activity_for_this_week_df = df_full_training_dataset[
            (pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START]).dt.normalize().dt.date >= current_week_start_date.normalize().date()) &
            (pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START]).dt.normalize().dt.date <= current_week_end_date.normalize().date())
        ]

        params_for_capacity = {
            **params_config,
            HORIZON_START_DATE_PARAM_KEY: current_week_start_date,
            "planning_horizon_days": 7
        }
        daily_blocks_for_week = compute_block_capacity(
            historical_activity_for_this_week_df, params_for_capacity
        )
        
        actual_durations_map_for_week: Dict[Any, float] = {}
        valid_surgeries_for_this_week_list = [] # Only include surgeries for which we have actuals
        for surg_dict in surgeries_for_week_list:
            surg_id = surg_dict['id']
            try:
                actual_duration = float(df_full_training_dataset.loc[surg_id, COL_PROCEDURE_DURATION_MIN])
                actual_durations_map_for_week[surg_id] = max(MIN_PROCEDURE_DURATION, actual_duration)
                valid_surgeries_for_this_week_list.append(surg_dict) # Add to valid list
            except KeyError:
                logger.warning(f"Could not find surgery ID {surg_id} in df_full_training_dataset for actual duration. Excluding from this week's training data.")
            except (ValueError, TypeError) as e:
                 logger.warning(f"Invalid actual duration for surgery ID {surg_id}: {e}. Excluding from this week's training data.")
        
        if not valid_surgeries_for_this_week_list: # If all surgeries were filtered out due to missing actuals
            logger.debug(f"Training Week {week_num+1}: No valid surgeries with actual durations. Skipping.")
            continue
            
        # Use valid_surgeries_for_this_week_list from now on for this week
        weekly_benders_data.append(
            (valid_surgeries_for_this_week_list, daily_blocks_for_week, actual_durations_map_for_week)
        )
        logger.debug(
            f"Prepared data for Benders training week {week_num+1}: "
            f"{len(valid_surgeries_for_this_week_list)} surgeries, "
            f"Blocks sum: {sum(daily_blocks_for_week.values())}, "
            f"Actual durations mapped: {len(actual_durations_map_for_week)}."
        )
    return weekly_benders_data


def main() -> None: # Renamed main function
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    gurobipy_internal_logger = logging.getLogger('gurobipy')
    gurobipy_internal_logger.setLevel(logging.WARNING)

    logger.info("Starting integrated Benders GTSM θ-learning run.")

    df_all = load_data(PARAMS)
    df_train, _, _ = split_data(df_all, PARAMS)
    if df_train.empty:
        logger.error("Training split (df_train) is empty. Aborting.")
        return

    # Pass df_train to prepare_weekly_data_for_benders_training
    weekly_data_list_for_training = prepare_weekly_data_for_benders_training(df_train, PARAMS)
    if not weekly_data_list_for_training:
        logger.error("No weekly data prepared for Benders training. Aborting.")
        return
    logger.info(f"Prepared {len(weekly_data_list_for_training)} weekly data scenarios for Benders training.")

    logger.info("Warm-starting θ via LASSO regression on df_train.")
    lasso_pipe = train_lasso_predictor(df_train, PARAMS)
    
    initial_theta_idx: Dict[int, float]
    feat_names: List[str]

    if lasso_pipe is None:
        logger.warning("LASSO training failed. Trying to get feature names via preprocessor for zero-theta warm-start.")
        # Fallback to get feature names (as in previous version)
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            df_train_processed = df_train.copy()
            df_train_processed = add_time_features(df_train_processed)
            for cat_col in CATEGORICAL_FEATURE_COLS:
                df_train_processed[cat_col] = df_train_processed[cat_col].astype(str).fillna(UNKNOWN_CATEGORY)
            temp_preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), NUMERIC_FEATURE_COLS),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURE_COLS)],
                remainder="drop", USET_DISPLAY="diagram"
            ) # Added USET_DISPLAY
            temp_preprocessor.fit(df_train_processed[ALL_FEATURE_COLS]) # Use ALL_FEATURE_COLS
            feat_names = list(temp_preprocessor.get_feature_names_out())
            if not feat_names: raise ValueError("Fallback preprocessor yielded no feature names.")
            initial_theta_idx = {j: 0.0 for j in range(len(feat_names))}
            logger.info(f"Using zero-theta warm-start with {len(feat_names)} features from fallback.")
        except Exception as e_feat:
            logger.error(f"Critical error getting feature names for theta warm-start: {e_feat}. Aborting.")
            return
    else:
        pre = lasso_pipe.named_steps["preprocessor"]
        feat_names = list(pre.get_feature_names_out())
        coefs = lasso_pipe.named_steps["regressor"].coef_
        if isinstance(coefs, (float, int)): # Handle single feature case from LASSO
            coefs = [coefs]
        initial_theta_idx = {j: float(coefs[j]) if j < len(coefs) else 0.0 for j in range(len(feat_names))}
        logger.info(f"Warm-start θ (indexed) initialized with {len(initial_theta_idx)} features from LASSO.")


    logger.info("Building Benders master problem (GTSM approach).")
    # Pass weekly_data_list_for_training to master problem
    master_model, Z_vars, R_vars, theta_master_vars, eta_master_vars = build_benders_master_problem(
        weekly_data_list=weekly_data_list_for_training, # Corrected variable name
        num_features=len(feat_names),
        initial_theta=initial_theta_idx
    )
    set_gurobi_model_parameters(master_model, PARAMS) # Set Gurobi params for master

    # Initial cuts are usually beneficial but complex to set up for GTSM without Z_bar, R_bar.
    # Relying on callback for now.
    master_model.update()
    # master_model.printStats() # Can be very verbose for large master
    logger.info("Master problem ready; invoking Benders callback.")

    callback = BendersThetaLearningCallback(
        weekly_data_list=weekly_data_list_for_training, # Corrected variable name
        Z_master_vars=Z_vars,
        R_master_vars=R_vars,
        theta_master_vars=theta_master_vars,
        eta_master_vars=eta_master_vars,
        solve_weekly_subproblem_fn=solve_weekly_subproblem,
        df_pool_reference=df_train.copy(), # Use df_train as the reference for feature building context
        ordered_feature_names=feat_names, 
        params_config=PARAMS,
    )
    master_model.optimize(callback)

    if master_model.SolCount > 0:
        learned_theta_by_name = {feat_names[j]: theta_master_vars[j].X for j in theta_master_vars}
        theta_path = Path(PARAMS["theta_path"])
        theta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(theta_path, "w") as f:
            json.dump(learned_theta_by_name, f, indent=2)
        logger.info(f"Learned global θ (GTSM) saved to {theta_path}. Master Obj: {master_model.ObjVal:.2f}")
    else:
        logger.error("No feasible solution found for GTSM master problem; θ not learned.")

    logger.info("Integrated Benders GTSM θ-learning run complete.")

if __name__ == "__main__":
    main()