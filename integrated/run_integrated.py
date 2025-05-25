# --- Make project root importable when run as a script from integrated/ ---
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------------------------------

import json
import logging
import pandas as pd
from typing import Any, Dict, List, Tuple

import gurobipy as gp

from src.config import PARAMS
from src.constants import (
    COL_PROCEDURE_DURATION_MIN,
    MIN_PROCEDURE_DURATION,
    COL_ACTUAL_START,
    HORIZON_START_DATE_PARAM_KEY,
    DEFAULT_LOGGER_NAME,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)
from src.data_processing import load_data, split_data, compute_block_capacity
from src.predictors import train_lasso_predictor
from src.scheduling_utils import select_surgeries
from src.solver_utils import set_gurobi_model_parameters
from integrated.master_problem import build_benders_master_problem
from integrated.callbacks import BendersThetaLearningCallback

# Type alias for one week's data: 
# ( surgeries:list of dict, daily_blocks:day→#blocks, actual_map:id→actual_dur )
EnhancedWeeklyDataTypeForRun = Tuple[
    List[Dict[str, Any]],
    Dict[int, int],
    Dict[Any, float],
]

logger = logging.getLogger(DEFAULT_LOGGER_NAME)

def prepare_weekly_data_for_benders_training(
    df_full_training_dataset: pd.DataFrame,
    params_config: Dict[str, Any]
) -> List[EnhancedWeeklyDataTypeForRun]:
    """
    Split df_full_training_dataset into W weekly scenarios.
    Each week w yields:
      - surgeries_for_week_list: List[dict] from select_surgeries()
      - daily_blocks_for_week: Dict[day_index→block_count] from compute_block_capacity()
      - actual_durations_map_for_week: Dict[surgery_id→actual_duration_min]
    Weeks with zero surgeries or missing actuals are skipped.
    """
    if df_full_training_dataset.empty:
        logger.error("Training dataset is empty; cannot prepare weekly data.")
        return []

    # 1) Align to first Monday
    ts_min = pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START].min()).normalize()
    first_monday = ts_min - pd.Timedelta(days=ts_min.weekday())

    num_weeks = params_config.get("warmup_weeks", 52)
    weekly_data: List[EnhancedWeeklyDataTypeForRun] = []
    logger.info(f"Preparing {num_weeks} training weeks for Benders.")

    for week_num in range(num_weeks):
        week_start = first_monday + pd.Timedelta(weeks=week_num)
        # a) select surgeries in this 7-day window
        sel_params = {**params_config, "planning_horizon_days": 7}
        surgeries = select_surgeries(df_full_training_dataset, week_start, sel_params)
        if not surgeries:
            logger.debug(f"Week {week_num+1}: no surgeries → skipping.")
            continue

        # b) compute historical block counts
        week_end = week_start + pd.Timedelta(days=6)
        hist_df = df_full_training_dataset[
            (pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START]).dt.normalize().dt.date
             >= week_start.date())
            &
            (pd.to_datetime(df_full_training_dataset[COL_ACTUAL_START]).dt.normalize().dt.date
             <= week_end.date())
        ]
        cap_params = {
            **params_config,
            HORIZON_START_DATE_PARAM_KEY: week_start,
            "planning_horizon_days": 7,
        }
        daily_blocks = compute_block_capacity(hist_df, cap_params)

        # c) build actual-duration map & filter surgeries missing actuals
        actual_map: Dict[Any, float] = {}
        valid_surgeries: List[Dict[str, Any]] = []
        for surg in surgeries:
            sid = surg["id"]
            try:
                d = float(df_full_training_dataset.loc[sid, COL_PROCEDURE_DURATION_MIN])
                actual_map[sid] = max(MIN_PROCEDURE_DURATION, d)
                valid_surgeries.append(surg)
            except Exception:
                logger.warning(f"Week {week_num+1}: missing/invalid actual for ID {sid}, skipping surgery.")

        if not valid_surgeries:
            logger.debug(f"Week {week_num+1}: no valid surgeries after actual-duration check → skipping.")
            continue

        weekly_data.append((valid_surgeries, daily_blocks, actual_map))
        logger.debug(
            f"Week {week_num+1}: "
            f"{len(valid_surgeries)} surgeries, "
            f"{sum(daily_blocks.values())} blocks total."
        )

    return weekly_data

def main() -> None:
    """
    1) Load & split data
    2) LASSO warm-start for θ
    3) Prepare weekly_data
    4) Build Benders master
    5) MIP start (all-reject + θ⁰)
    6) Attach LazyCallback
    7) Optimize
    8) Dump θ JSON
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logger.info("Starting integrated Benders θ-learning.")

    # 1) data
    df_all = load_data(PARAMS)
    df_train, _, _ = split_data(df_all, PARAMS)
    if df_train.empty:
        logger.error("No warm-up data; exiting.")
        return
    
    # 2) θ⁰ via LASSO
    lasso_pipe = train_lasso_predictor(df_train, PARAMS)
    if lasso_pipe is None:
        logger.error("LASSO failed; aborting.")
        return
    pre = lasso_pipe.named_steps["preprocessor"]
    feat_names = list(pre.get_feature_names_out())
    coefs     = lasso_pipe.named_steps["regressor"].coef_
    initial_theta = {j: float(coefs[j]) for j in range(len(feat_names))}

    # 3) weekly scenarios
    weekly_data = prepare_weekly_data_for_benders_training(df_train, PARAMS)
    if not weekly_data:
        logger.error("No weekly scenarios prepared; aborting.")
        return

    # 4) build master
    model, Z_vars, R_vars, theta_vars, eta_vars = build_benders_master_problem(
        weekly_data, len(feat_names), initial_theta
    )

    # 5) trivial MIP start: reject-all, no Z, warm θ
    for key, var in R_vars.items(): var.Start = 1
    for key, var in Z_vars.items(): var.Start = 0
    for j, val in initial_theta.items(): theta_vars[j].Start = val
    model.update()

    # 6) attach callback
    callback = BendersThetaLearningCallback(
        master_model     = model,
        weekly_data      = weekly_data,
        Z_vars           = Z_vars,
        R_vars           = R_vars,
        theta_vars       = theta_vars,
        eta_vars         = eta_vars,
        df_pool_reference= df_train,
        feature_names    = feat_names,
        params           = PARAMS,
    )
    model.Params.LazyConstraints = 1

    # 7) solve
    model.optimize(callback.callback)

    # 8) write θ JSON
    theta_out = {feat_names[j]: theta_vars[j].X for j in sorted(theta_vars)}
    out_path = PARAMS["theta_path"]
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(theta_out, f, indent=2)
    logger.info(f"Wrote learned θ to {out_path}")

if __name__ == "__main__":
    main()
