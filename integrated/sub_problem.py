from typing import Any, Dict, List, Tuple 

import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np 

from src.config import PARAMS
from src.constants import (
    COL_ACTUAL_DUR_MIN,
    COL_BOOKED_MIN,
    MIN_PROCEDURE_DURATION,
    GUROBI_VAR_OT_PREFIX, 
    GUROBI_VAR_IT_PREFIX, 
    GUROBI_VAR_U_LEARN_ABS_ERROR_PREFIX, 
    MAX_OVERTIME_MINUTES_PER_BLOCK, 
)
from src.data_processing import build_feature_vector
from src.solver_utils import set_gurobi_model_parameters 

def solve_weekly_subproblem(
    surgeries_info: List[Dict[str, Any]],
    daily_blocks_info: Dict[int, int], # Renamed for clarity
    theta_values_by_index: Dict[int, float], # Renamed for clarity (integer indexed)
    Z_bar_assignments: Dict[Tuple[int, int], int], # (surg_idx_in_week, flat_block_idx) -> 0 or 1
    R_bar_rejections: Dict[int, int], # (surg_idx_in_week) -> 0 or 1 (not used in this LP)
    df_pool_reference: pd.DataFrame,
    ordered_feature_names: List[str], # Semantic feature names
    params_config: Dict[str, Any], # Should be PARAMS
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Solve the LP subproblem for one week given fixed assignments Z_bar and theta.
    Returns:
      - Q_val: optimal overtime+idle cost,
      - grads: subgradients g_{i,b} for each (surgery i, flat_block_idx b).
    """

    num_surgeries = len(surgeries_info)
    num_features = len(ordered_feature_names) # Should match len(theta_values_by_index)

    # 1. Build predicted (p_i) & actual (d_i) durations
    predicted_durations_p_i: List[float] = [] # p_i for i = 0..num_surgeries-1
    actual_durations_d_i: List[float] = []    # d_i for i = 0..num_surgeries-1

    for i_idx_in_week, surg_dict in enumerate(surgeries_info):
        feature_vector = build_feature_vector(
            surgery_dict=surg_dict,
            df_pool_full_data=df_pool_reference,
            feature_names_ordered=ordered_feature_names,
            scaling_method=params_config["feature_scaling"],
        )
        
        # Calculate predicted duration using integer-indexed theta_values_by_index
        # Ensure feature_vector is also effectively indexed 0..num_features-1 by build_feature_vector
        pred_duration_raw = sum(theta_values_by_index[j] * feature_vector[j] for j in range(num_features))
        predicted_durations_p_i.append(max(MIN_PROCEDURE_DURATION, pred_duration_raw))
        
        actual_duration = float(surg_dict.get(COL_ACTUAL_DUR_MIN, surg_dict.get(COL_BOOKED_MIN)))
        actual_durations_d_i.append(actual_duration)


    # 2. Flatten blocks in sorted-day order to get a list of (day, block_in_day) tuples
    # and a flat index b = 0 ... B-1
    flat_block_tuples: List[Tuple[int,int]] = [] # Stores (day_original, block_in_day_original)
    for day_orig_idx in sorted(daily_blocks_info.keys()):
        for b_in_day_orig_idx in range(daily_blocks_info[day_orig_idx]):
            flat_block_tuples.append((day_orig_idx, b_in_day_orig_idx))
    
    num_flat_blocks_B = len(flat_block_tuples)

    # 3. Build Gurobi LP model for the subproblem
    sub_model = gp.Model(f"Subproblem_LP_Week") # Add week index if available for better naming
    sub_model.Params.OutputFlag = 0

    # Use centralized Gurobi parameter setting for subproblems
    subproblem_gurobi_settings = params_config.get("gurobi_subproblem_milp_settings", {})
    set_gurobi_model_parameters(sub_model, subproblem_gurobi_settings) # Pass only relevant settings

    block_capacity_minutes = params_config["block_size_minutes"]
    cost_overtime_param = params_config["cost_overtime_per_min"]
    cost_idle_param = params_config["cost_idle_per_min"]
    epsilon_mae_param = params_config["epsilon_block_mae"]

    # Decision Variables for this LP
    OT_vars = sub_model.addVars(range(num_flat_blocks_B), lb=0.0, name=GUROBI_VAR_OT_PREFIX)
    IT_vars = sub_model.addVars(range(num_flat_blocks_B), lb=0.0, name=GUROBI_VAR_IT_PREFIX)
    u_learn_vars = sub_model.addVars(range(num_flat_blocks_B), lb=0.0, name=GUROBI_VAR_U_LEARN_ABS_ERROR_PREFIX)

    # Store constraint objects for dual extraction
    capacity_constraints = {}
    learning_pos_constraints = {}
    learning_neg_constraints = {}
    learning_epsilon_constraints = {} # Though duals of u <= eps are not directly in subgradient

    for b_flat_idx in range(num_flat_blocks_B):
        # Identify surgeries assigned to this flat block b_flat_idx by Z_bar_assignments
        # Z_bar_assignments keys are (surgery_idx_in_week, flat_block_idx)
        surgeries_assigned_to_this_block_indices = [
            i_idx for (i_idx, assigned_b_flat_idx), is_assigned_val in Z_bar_assignments.items()
            if assigned_b_flat_idx == b_flat_idx and is_assigned_val == 1
        ]

        sum_pred_dur_in_block = quicksum(predicted_durations_p_i[i_idx] for i_idx in surgeries_assigned_to_this_block_indices)
        
        capacity_constraints[b_flat_idx] = sub_model.addConstr(
            sum_pred_dur_in_block - block_capacity_minutes == OT_vars[b_flat_idx] - IT_vars[b_flat_idx],
            name=f"cap_sub_{b_flat_idx}",
        )
        # Optional overtime cap
        sub_model.addConstr(OT_vars[b_flat_idx] <= MAX_OVERTIME_MINUTES_PER_BLOCK, name=f"ot_cap_sub_{b_flat_idx}")


        # Learning Constraint (MAE linearization)
        sum_signed_errors_in_block = quicksum(
            (predicted_durations_p_i[i_idx] - actual_durations_d_i[i_idx])
            for i_idx in surgeries_assigned_to_this_block_indices
        )
        
        learning_pos_constraints[b_flat_idx] = sub_model.addConstr(
            sum_signed_errors_in_block <= u_learn_vars[b_flat_idx], name=f"learn_pos_sub_{b_flat_idx}"
        )
        learning_neg_constraints[b_flat_idx] = sub_model.addConstr(
            -sum_signed_errors_in_block <= u_learn_vars[b_flat_idx], name=f"learn_neg_sub_{b_flat_idx}"
        )
        learning_epsilon_constraints[b_flat_idx] = sub_model.addConstr( # Storing this though its dual isn't in g_it
            u_learn_vars[b_flat_idx] <= epsilon_mae_param, name=f"learn_eps_sub_{b_flat_idx}"
        )

    # Subproblem Objective: minimize OT and IT costs
    sub_model.setObjective(
        quicksum(cost_overtime_param * OT_vars[b_flat_idx] + cost_idle_param * IT_vars[b_flat_idx] for b_flat_idx in range(num_flat_blocks_B)),
        GRB.MINIMIZE
    )

    sub_model.optimize()

    if sub_model.Status != GRB.OPTIMAL:
        # If subproblem is infeasible (likely due to learning constraint + fixed Z_bar),
        # this indicates the master's (theta, Z_bar) combination is problematic for this week.
        return float("inf"), {key_z_bar: 0.0 for key_z_bar in Z_bar_assignments.keys()}

    optimal_Q_value = sub_model.ObjVal

    # 4. Extract Duals for subgradient calculation
    # Duals (Pi values) are associated with constraints
    lambda_duals = {b_flat_idx: capacity_constraints[b_flat_idx].Pi for b_flat_idx in capacity_constraints}
    mu_pos_duals = {b_flat_idx: learning_pos_constraints[b_flat_idx].Pi for b_flat_idx in learning_pos_constraints}
    mu_neg_duals = {b_flat_idx: learning_neg_constraints[b_flat_idx].Pi for b_flat_idx in learning_neg_constraints}
    # Duals for u_learn_vars[b] <= epsilon_mae_param are not directly part of g_it for Z_it.

    # 5. Compute Subgradients g_{i,b_flat}
    # The subgradient g_{i,b} measures how Q_val changes if Z_bar[i,b] flips from 0 to 1 (or 1 to 0).
    # It's calculated for all (i,b) pairs that *could* be in Z_bar_assignments.
    subgradients: Dict[Tuple[int, int], float] = {}
    for i_idx_in_week in range(num_surgeries):
        for b_flat_idx in range(num_flat_blocks_B):
            # Coefficient of Z_bar[i_idx_in_week, b_flat_idx] in capacity constraint is p_i[i_idx_in_week]
            # Coefficient of Z_bar[i_idx_in_week, b_flat_idx] in learning_pos constraint is (p_i - d_i)
            # Coefficient of Z_bar[i_idx_in_week, b_flat_idx] in learning_neg constraint is -(p_i - d_i)
            
            g_val = (lambda_duals.get(b_flat_idx, 0.0) * predicted_durations_p_i[i_idx_in_week] +
                     mu_pos_duals.get(b_flat_idx, 0.0) * (predicted_durations_p_i[i_idx_in_week] - actual_durations_d_i[i_idx_in_week]) +
                     mu_neg_duals.get(b_flat_idx, 0.0) * (-(predicted_durations_p_i[i_idx_in_week] - actual_durations_d_i[i_idx_in_week]))
            )
            subgradients[(i_idx_in_week, b_flat_idx)] = g_val
            
    return optimal_Q_value, subgradients