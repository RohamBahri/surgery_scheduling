"""
Core Gurobi model building and solving utilities for surgery scheduling.
Includes implementations for deterministic, predictive, clairvoyant, SAA,
and integrated optimization models.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pandas as pd

from src.config import PARAMS  # For solver parameters and debug mode
from src.constants import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    COL_ACTUAL_DUR_MIN,  # Assuming it's added to constants.py
    COL_BOOKED_MIN,
    DEFAULT_LOGGER_NAME,
    GUROBI_MIP_FOCUS_FEASIBILITY,
    GUROBI_OUTPUT_FLAG_SILENT,
    GUROBI_PRESOLVE_AUTO,
    GUROBI_THREADS_ALL_CORES,
    GUROBI_VAR_IT_PREFIX,
    GUROBI_VAR_OT_PREFIX,
    GUROBI_VAR_R_PREFIX,
    GUROBI_VAR_THETA_PREFIX,
    GUROBI_VAR_X_PREFIX,
    GUROBI_VAR_ZSPILL_PREFIX,
    MAX_OVERTIME_MINUTES_PER_BLOCK,
    NUMERIC_FEATURE_COLS,
    SCALING_RAW,  # Default scaling for _build_feature_matrix if not specified
    SCALING_STD,
    GUROBI_VAR_BLK_ABS_ERR_PREFIX,  # For integrated model
    GUROBI_VAR_BLK_SIGNED_ERR_PREFIX,  # For integrated model
)
from src.data_processing import add_time_features  # For _build_feature_matrix

logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def set_gurobi_model_parameters(
    model: gp.Model,
    params_config: Dict[str, Any] = PARAMS,
    override_timelimit: Optional[float] = None,
    override_mipgap: Optional[float] = None,
    override_heuristics: Optional[float] = None,
    override_output_flag: Optional[int] = None,
    override_threads: Optional[int] = None,
) -> None:
    """Sets common Gurobi parameters for a model.

    Uses defaults from `params_config` (typically `src.config.PARAMS`),
    with specific overrides possible. `debug_mode` in `params_config`
    influences default time limit, MIP gap, and heuristics.

    Args:
        model: The Gurobi model to configure.
        params_config: Dictionary containing base Gurobi settings and debug_mode flag.
        override_timelimit: Specific time limit in seconds.
        override_mipgap: Specific MIP gap.
        override_heuristics: Specific heuristics aggressiveness (0 to 1).
        override_output_flag: Gurobi OutputFlag (0=silent, 1=verbose).
        override_threads: Number of threads for Gurobi (0=auto).
    """
    is_debug_mode = params_config.get("debug_mode", False)

    # Default Gurobi parameters from config or fallback values
    default_timelimit = params_config.get("gurobi_timelimit", 600)
    default_mipgap = params_config.get("gurobi_mipgap", 0.01)
    default_heuristics = params_config.get("gurobi_heuristics", 0.05)
    default_output_flag = params_config.get(
        "gurobi_output_flag", GUROBI_OUTPUT_FLAG_SILENT
    )
    default_threads = params_config.get("gurobi_threads", GUROBI_THREADS_ALL_CORES)
    default_presolve = params_config.get("gurobi_presolve", GUROBI_PRESOLVE_AUTO)
    default_mip_focus = params_config.get("gurobi_mipfocus", 0)  # 0 = balanced

    # Apply debug mode adjustments if active
    if is_debug_mode:
        logger.debug(
            f"Gurobi debug mode active for model {model.ModelName}. Applying fast-solve settings."
        )
        # Override defaults with debug-specific values if not explicitly provided in PARAMS
        # Or, if PARAMS already has debug-specific values, those will be used by default_... above.
        # This logic gives precedence to PARAMS if specific gurobi_xxx_debug keys were set.
        # For simplicity, we'll use fixed debug values here if is_debug_mode is true,
        # unless specific overrides are passed to this function.
        effective_timelimit = 60.0
        effective_mipgap = 0.20
        effective_heuristics = 0.50
        effective_mip_focus = (
            GUROBI_MIP_FOCUS_FEASIBILITY  # Focus on finding feasible solutions quickly
        )
    else:
        effective_timelimit = default_timelimit
        effective_mipgap = default_mipgap
        effective_heuristics = default_heuristics
        effective_mip_focus = default_mip_focus

    # Apply function-level overrides
    if override_timelimit is not None:
        effective_timelimit = override_timelimit
    if override_mipgap is not None:
        effective_mipgap = override_mipgap
    if override_heuristics is not None:
        effective_heuristics = override_heuristics

    effective_output_flag = (
        override_output_flag
        if override_output_flag is not None
        else default_output_flag
    )
    effective_threads = (
        override_threads if override_threads is not None else default_threads
    )

    # Set parameters on the Gurobi model
    model.setParam(GRB.Param.TimeLimit, effective_timelimit)
    model.setParam(GRB.Param.MIPGap, effective_mipgap)
    model.setParam(GRB.Param.Heuristics, effective_heuristics)
    model.setParam(GRB.Param.OutputFlag, effective_output_flag)
    model.setParam(GRB.Param.Threads, effective_threads)
    model.setParam(
        GRB.Param.Presolve, default_presolve
    )  # Presolve usually best left to auto or from PARAMS
    if (
        effective_mip_focus != 0 or is_debug_mode
    ):  # Only set MIPFocus if non-default or debug
        model.setParam(GRB.Param.MIPFocus, effective_mip_focus)

    # logger.debug(
    #     f"Gurobi params for model {model.ModelName}: TimeLimit={effective_timelimit:.1f}s, "
    #     f"MIPGap={effective_mipgap:.3f}, Heuristics={effective_heuristics:.2f}, "
    #     f"OutputFlag={effective_output_flag}, Threads={effective_threads}, "
    #     f"Presolve={default_presolve}, MIPFocus={effective_mip_focus}."
    # )


def _add_single_case_spillover_constraints(
    model: gp.Model,
    assignment_vars: Dict[Tuple[int, int, int], gp.Var],  # x[i,d,b]
    surgery_durations: Dict[int, float],  # duration for surgery i
    block_tuples_list: List[Tuple[int, int]],  # (day_idx, block_idx)
    block_capacity_minutes: float,
) -> None:
    """Adds constraints to allow one surgery per block to exceed block capacity.

    This models a "spillover" case where a single long surgery can run over
    the nominal block time, and its full duration is counted against capacity
    for overtime calculation purposes (handled by main capacity constraints).
    This helper adds the binary variables and linking constraints.

    Args:
        model: The Gurobi model.
        assignment_vars: Gurobi variables for surgery assignments (x_idb).
        surgery_durations: Dictionary mapping surgery index to its duration.
        block_tuples_list: List of (day, block) tuples where capacity applies.
        block_capacity_minutes: Nominal capacity of a block in minutes.
    """
    num_surgeries = len(surgery_durations)

    # z_spill[i, d, b]: Binary, 1 if surgery i is the designated spillover case in block (d,b)
    z_spill_vars = model.addVars(
        [(i, d, b) for (d, b) in block_tuples_list for i in range(num_surgeries)],
        vtype=GRB.BINARY,
        name=GUROBI_VAR_ZSPILL_PREFIX,
    )

    for day_idx, block_idx in block_tuples_list:
        # At most one surgery can be the spillover case in a given block
        model.addConstr(
            quicksum(z_spill_vars[i, day_idx, block_idx] for i in range(num_surgeries))
            <= 1,
            name=f"one_spill_limit_{day_idx}_{block_idx}",
        )

        for surg_idx in range(num_surgeries):
            # If surgery i is a spillover in (d,b), it must be assigned to (d,b)
            # z_spill[i,d,b] <= x[i,d,b]
            model.addConstr(
                z_spill_vars[surg_idx, day_idx, block_idx]
                <= assignment_vars[surg_idx, day_idx, block_idx],
                name=f"link_z_x_{surg_idx}_{day_idx}_{block_idx}",
            )

        # Capacity constraint with spillover: sum(dur_i * x_idb) <= C + sum(dur_i * z_idb)
        # This means the total scheduled duration can exceed C by at most the duration
        # of the single designated spillover case.
        # The OT/IT variables then correctly capture deviation from C.
        model.addConstr(
            quicksum(
                surgery_durations[i] * assignment_vars[i, day_idx, block_idx]
                for i in range(num_surgeries)
            )
            <= block_capacity_minutes
            + quicksum(
                surgery_durations[i] * z_spill_vars[i, day_idx, block_idx]
                for i in range(num_surgeries)
            ),
            name=f"capacity_with_spillover_{day_idx}_{block_idx}",
        )
    logger.debug("Added single-case spillover constraints to the model.")


def solve_saa_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    params_config: Dict[str, Any],
    scenario_duration_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Solves the Sample Average Approximation (SAA) model.

    This is a two-stage stochastic programming model where first-stage decisions
    are surgery assignments (x_idb) and rejections (r_i). Second-stage (recourse)
    variables are overtime and idle time, calculated for each scenario based on
    sampled durations.

    Args:
        surgeries_info: List of surgery data dictionaries. Each must contain
            `COL_BOOKED_MIN` and optionally 'predicted_dur_min' for initial spillover logic.
        daily_block_counts: Maps day index to the number of blocks available.
        params_config: Main configuration dictionary. Used for costs, horizon length,
            number of SAA scenarios, block size, and Gurobi settings.
        scenario_duration_matrix: Numpy array (num_surgeries x num_scenarios)
            of sampled surgery durations for each scenario.

    Returns:
        Dictionary with "obj" (objective value), "status" (Gurobi status),
        and "model" (the solved Gurobi model object). Returns obj=None if no solution.
    """
    num_surgeries = len(surgeries_info)
    planning_horizon_days = params_config["planning_horizon_days"]
    num_saa_scenarios = params_config["saa_scenarios"]
    block_size_min = params_config["block_size_minutes"]
    cost_rejection = params_config["cost_rejection_per_case"]
    cost_overtime = params_config["cost_overtime_per_min"]
    cost_idle = params_config["cost_idle_per_min"]

    model_saa = gp.Model("SAA_Scheduling")
    set_gurobi_model_parameters(model_saa, params_config)

    # --- First-stage variables ---
    # x_vars[i, d, b]: surgery i assigned to block b on day d
    x_vars = model_saa.addVars(
        [
            (i, d, b)
            for i in range(num_surgeries)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        vtype=GRB.BINARY,
        name=GUROBI_VAR_X_PREFIX,
    )
    # r_vars[i]: surgery i is rejected
    r_vars = model_saa.addVars(
        num_surgeries, vtype=GRB.BINARY, name=GUROBI_VAR_R_PREFIX
    )

    # --- Assignment constraints (first-stage) ---
    for i in range(num_surgeries):
        model_saa.addConstr(
            quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            )
            + r_vars[i]
            == 1,
            name=f"assign_or_reject_{i}",
        )

    # Durations for spillover constraints (can use booked or predicted)
    # Using predicted if available, else booked, for the deterministic part of spillover.
    # The scenarios use their own sampled durations.
    spillover_durations: Dict[int, float] = {
        i: surg.get("predicted_dur_min", surg[COL_BOOKED_MIN])
        for i, surg in enumerate(surgeries_info)
    }

    all_block_tuples = [
        (d, b)
        for d in range(planning_horizon_days)
        for b in range(daily_block_counts.get(d, 0))
    ]
    if all_block_tuples:  # Only add if there are blocks
        _add_single_case_spillover_constraints(
            model_saa, x_vars, spillover_durations, all_block_tuples, block_size_min
        )

    # --- Second-stage (recourse) variables and constraints ---
    total_expected_recourse_cost = gp.LinExpr()

    for k_scen in range(num_saa_scenarios):
        # ot_scen[d, b, k]: overtime in block (d,b) for scenario k
        ot_scen_vars = model_saa.addVars(
            [
                (d, b)
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            ],
            lb=0.0,
            name=f"{GUROBI_VAR_OT_PREFIX}_scen{k_scen}",
        )
        # it_scen[d, b, k]: idle time in block (d,b) for scenario k
        it_scen_vars = model_saa.addVars(
            [
                (d, b)
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            ],
            lb=0.0,
            name=f"{GUROBI_VAR_IT_PREFIX}_scen{k_scen}",
        )

        for d_day in range(planning_horizon_days):
            for b_block in range(daily_block_counts.get(d_day, 0)):
                # Sum of sampled durations for surgeries assigned to this block in this scenario
                sum_sampled_durations_in_block = quicksum(
                    scenario_duration_matrix[i, k_scen] * x_vars[i, d_day, b_block]
                    for i in range(num_surgeries)
                )

                # Overtime constraint
                model_saa.addConstr(
                    sum_sampled_durations_in_block - block_size_min
                    <= ot_scen_vars[d_day, b_block],
                    name=f"ot_calc_{d_day}_{b_block}_scen{k_scen}",
                )
                # Idle time constraint
                model_saa.addConstr(
                    block_size_min - sum_sampled_durations_in_block
                    <= it_scen_vars[d_day, b_block],
                    name=f"it_calc_{d_day}_{b_block}_scen{k_scen}",
                )

        scenario_recourse_cost = (
            cost_overtime * ot_scen_vars.sum() + cost_idle * it_scen_vars.sum()
        )
        total_expected_recourse_cost += (
            1.0 / num_saa_scenarios
        ) * scenario_recourse_cost

    # --- Objective function ---
    rejection_cost_total = cost_rejection * quicksum(
        surgeries_info[i][COL_BOOKED_MIN] * r_vars[i] for i in range(num_surgeries)
    )
    model_saa.setObjective(
        rejection_cost_total + total_expected_recourse_cost, GRB.MINIMIZE
    )

    logger.info(
        f"Optimizing SAA model ({num_surgeries} surgeries, {num_saa_scenarios} scenarios)..."
    )
    model_saa.optimize()

    obj_val = model_saa.ObjVal if model_saa.SolCount > 0 else None
    if obj_val is None:
        logger.warning("SAA model optimization did not find a feasible solution.")
    else:
        logger.info(
            f"SAA model optimized. Objective: {obj_val:.2f}, Status: {model_saa.Status}"
        )

    return {"obj": obj_val, "status": model_saa.Status, "model": model_saa}


def _solve_single_stage_deterministic_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    params_config: Dict[str, Any],
    duration_key_for_model: str,  # e.g. COL_BOOKED_MIN, "predicted_dur_min", COL_ACTUAL_DUR_MIN
    model_name_suffix: str,
    is_lp_relaxation: bool = False,
) -> Dict[str, Any]:
    """Core solver for single-stage deterministic assignment models.

    This function builds and solves models like "Deterministic" (using booked times),
    "Predictive" (using ML predictions), or "Clairvoyant" (using actual durations).

    Args:
        surgeries_info: List of surgery data. Each dict must contain `COL_BOOKED_MIN`
            and the `duration_key_for_model`.
        daily_block_counts: Maps day index to number of blocks.
        params_config: Main configuration dictionary for costs, block size, etc.
        duration_key_for_model: The key in surgery_info dicts to use for durations
            in the model's capacity constraints.
        model_name_suffix: Suffix for the Gurobi model name (e.g., "Booked", "Pred").
        is_lp_relaxation: If True, relaxes binary variables to continuous.

    Returns:
        Dictionary with "obj", "status", and "model". Obj is None if no solution.
    """
    num_surgeries = len(surgeries_info)
    planning_horizon_days = params_config["planning_horizon_days"]
    block_size_min = params_config["block_size_minutes"]
    cost_rejection = params_config["cost_rejection_per_case"]
    cost_overtime = params_config["cost_overtime_per_min"]
    cost_idle = params_config["cost_idle_per_min"]

    model_det = gp.Model(f"SingleStage_{model_name_suffix}")
    set_gurobi_model_parameters(model_det, params_config)

    var_type = GRB.CONTINUOUS if is_lp_relaxation else GRB.BINARY

    # --- Decision Variables ---
    x_vars = model_det.addVars(
        [
            (i, d, b)
            for i in range(num_surgeries)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        vtype=var_type,
        name=GUROBI_VAR_X_PREFIX,
    )
    r_vars = model_det.addVars(num_surgeries, vtype=var_type, name=GUROBI_VAR_R_PREFIX)
    ot_vars = model_det.addVars(
        [
            (d, b)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        lb=0.0,
        name=GUROBI_VAR_OT_PREFIX,
    )
    it_vars = model_det.addVars(
        [
            (d, b)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        lb=0.0,
        name=GUROBI_VAR_IT_PREFIX,
    )

    # --- Constraints ---
    # 1. Assignment or rejection for each surgery
    for i in range(num_surgeries):
        model_det.addConstr(
            quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            )
            + r_vars[i]
            == 1,
            name=f"assign_or_reject_{i}",
        )

    # 2. Block capacity, overtime, and idle time calculation
    for d_day in range(planning_horizon_days):
        for b_block in range(daily_block_counts.get(d_day, 0)):
            sum_durations_in_block = quicksum(
                surgeries_info[i][duration_key_for_model] * x_vars[i, d_day, b_block]
                for i in range(num_surgeries)
            )
            model_det.addConstr(
                sum_durations_in_block - block_size_min <= ot_vars[d_day, b_block],
                name=f"ot_calc_{d_day}_{b_block}",
            )
            model_det.addConstr(
                block_size_min - sum_durations_in_block <= it_vars[d_day, b_block],
                name=f"it_calc_{d_day}_{b_block}",
            )
            # Optional: Cap on overtime per block
            model_det.addConstr(
                ot_vars[d_day, b_block] <= MAX_OVERTIME_MINUTES_PER_BLOCK,
                name=f"ot_cap_{d_day}_{b_block}",
            )

    # 3. Spillover constraints
    model_durations: Dict[int, float] = {
        i: surg[duration_key_for_model] for i, surg in enumerate(surgeries_info)
    }
    all_block_tuples = [
        (d, b)
        for d in range(planning_horizon_days)
        for b in range(daily_block_counts.get(d, 0))
    ]
    if all_block_tuples:
        _add_single_case_spillover_constraints(
            model_det, x_vars, model_durations, all_block_tuples, block_size_min
        )

    # --- Objective Function ---
    rejection_cost_total = cost_rejection * quicksum(
        surgeries_info[i][COL_BOOKED_MIN] * r_vars[i] for i in range(num_surgeries)
    )
    operational_cost = cost_overtime * ot_vars.sum() + cost_idle * it_vars.sum()
    model_det.setObjective(rejection_cost_total + operational_cost, GRB.MINIMIZE)

    relax_msg = " (LP relaxation)" if is_lp_relaxation else ""
    logger.info(
        f"Optimizing {model_det.ModelName}{relax_msg} for {num_surgeries} surgeries..."
    )
    model_det.optimize()

    obj_val = model_det.ObjVal if model_det.SolCount > 0 else None
    if obj_val is None:
        logger.warning(
            f"{model_det.ModelName}{relax_msg} optimization did not find a feasible solution."
        )
    else:
        logger.info(
            f"{model_det.ModelName}{relax_msg} optimized. Objective: {obj_val:.2f}, Status: {model_det.Status}"
        )

    return {"obj": obj_val, "status": model_det.Status, "model": model_det}


def solve_deterministic_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    params_config: Dict[str, Any],
    **kwargs,  # Catches lp_relax, return_model implicitly if passed from old signature
) -> Dict[str, Any]:
    """Solves the deterministic model using booked surgery durations."""
    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        params_config,
        duration_key_for_model=COL_BOOKED_MIN,
        model_name_suffix="BookedTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )


def solve_predictive_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    params_config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Solves the predictive model using ML-predicted surgery durations."""
    # Ensure 'predicted_dur_min' key exists or fall back
    for surg in surgeries_info:
        if "predicted_dur_min" not in surg:
            logger.warning(
                f"Surgery id {surg.get('id','N/A')} missing 'predicted_dur_min'. "
                f"Falling back to '{COL_BOOKED_MIN}' for this surgery in predictive model."
            )
            surg["predicted_dur_min"] = surg[COL_BOOKED_MIN]

    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        params_config,
        duration_key_for_model="predicted_dur_min",
        model_name_suffix="PredictedTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )


def solve_clairvoyant_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    params_config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    if not (surgeries_info and COL_ACTUAL_DUR_MIN in surgeries_info[0]):
        logger.error(
            f"Clairvoyant model: Cannot find duration key '{COL_ACTUAL_DUR_MIN}' "
            "in surgery data. Ensure `select_surgeries` provides this key."
        )
        return {
            "obj": None,
            "status": "Error_Missing_Actual_Duration_Key",
            "model": None,
        }

    clairvoyant_duration_key_to_use = COL_ACTUAL_DUR_MIN

    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        params_config,
        duration_key_for_model=clairvoyant_duration_key_to_use,  # Use the constant
        model_name_suffix="ClairvoyantActualTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )
