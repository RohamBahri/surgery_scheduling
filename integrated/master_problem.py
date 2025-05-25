from typing import Any
import gurobipy as gp
from gurobipy import GRB, quicksum

# Assuming PARAMS and relevant column name constants are in these modules
from src.config import PARAMS
from src.constants import (
    COL_BOOKED_MIN,
    GUROBI_VAR_THETA_PREFIX, # Assuming you'll add this if not present
    GUROBI_VAR_ABS_THETA_PREFIX, # Assuming you'll add this if not present
    GUROBI_VAR_ETA_PREFIX, # Assuming you'll add this if not present
    GUROBI_VAR_R_PREFIX, # Assuming you'll add this if not present
    GUROBI_VAR_X_PREFIX, # Assuming you'll add this if not present (for Z vars)
)

# Define temporary constants if not in constants.py yet
if "GUROBI_VAR_THETA_PREFIX" not in globals(): GUROBI_VAR_THETA_PREFIX = "theta"
if "GUROBI_VAR_ABS_THETA_PREFIX" not in globals(): GUROBI_VAR_ABS_THETA_PREFIX = "abs_theta"
if "GUROBI_VAR_ETA_PREFIX" not in globals(): GUROBI_VAR_ETA_PREFIX = "eta"
if "GUROBI_VAR_R_PREFIX" not in globals(): GUROBI_VAR_R_PREFIX = "R" # Master R vars are R_s_i
if "GUROBI_VAR_X_PREFIX" not in globals(): GUROBI_VAR_X_PREFIX = "Z" # Master Z vars are Z_s_i_b


def build_benders_master_problem(
    weekly_data_list: list[tuple[list[dict], dict[int, int], dict[Any, float]]], # Updated type hint
    num_features: int,
    initial_theta: dict[int, float] = None,
) -> tuple[gp.Model, dict, dict, dict, dict]:
    """
    Construct the integrated Benders master MIP.

    Args:
      weekly_data_list: list of (surgeries_info, daily_blocks, actual_durations_map)
                        per historical week.
      num_features: number of features for the regression vector theta.
      initial_theta: Optional dictionary mapping feature index j to warm-start value.

    Returns:
      model: the Gurobi Model
      Z_vars: dict mapping (s, i, b) -> binary var Z_{s,i,b} for week s
      R_vars: dict mapping (s, i)   -> binary var R_{s,i} for week s
      theta_vars: dict mapping j -> continuous var theta_j
      eta_vars: dict mapping s -> continuous var eta_s
    """
    model = gp.Model("Benders_Master_Theta_Learning")
    model.Params.OutputFlag = 0

    # Solver parameters from PARAMS
    model.Params.TimeLimit = PARAMS["gurobi_timelimit"]
    model.Params.MIPGap = PARAMS["gurobi_mipgap"]
    model.Params.Threads = PARAMS["gurobi_threads"]
    model.Params.OutputFlag = PARAMS["gurobi_output_flag"]
    model.Params.LazyConstraints = 1 # Essential for Benders

    num_training_weeks = len(weekly_data_list)

    # Decision variables
    theta_vars = model.addVars(
        range(num_features), lb=-GRB.INFINITY, ub=GRB.INFINITY, name=GUROBI_VAR_THETA_PREFIX
    )
    if initial_theta:
        for j, val in initial_theta.items():
            if j in theta_vars: # Check if index is valid
                theta_vars[j].Start = val

    abs_theta_penalty_vars = model.addVars(range(num_features), lb=0.0, name=GUROBI_VAR_ABS_THETA_PREFIX)
    for j in range(num_features):
        model.addConstr(abs_theta_penalty_vars[j] >= theta_vars[j], name=f"t_pos_{j}")
        model.addConstr(abs_theta_penalty_vars[j] >= -theta_vars[j], name=f"t_neg_{j}")

    eta_vars = model.addVars(range(num_training_weeks), lb=0.0, name=GUROBI_VAR_ETA_PREFIX)

    Z_vars = {}
    R_vars = {}
    for s_idx, (surgeries_in_week, daily_blocks_in_week, _) in enumerate(weekly_data_list): # Unpack correctly
        num_block_slots_in_week = sum(daily_blocks_in_week.values())
        for i_idx in range(len(surgeries_in_week)):
            R_vars[s_idx, i_idx] = model.addVar(vtype=GRB.BINARY, name=f"{GUROBI_VAR_R_PREFIX}_{s_idx}_{i_idx}")
            for b_idx_flat in range(num_block_slots_in_week):
                Z_vars[s_idx, i_idx, b_idx_flat] = model.addVar(vtype=GRB.BINARY, name=f"{GUROBI_VAR_X_PREFIX}_{s_idx}_{i_idx}_{b_idx_flat}")

    # Assignment constraints
    for s_idx, (surgeries_in_week, daily_blocks_in_week, _) in enumerate(weekly_data_list):
        num_block_slots_in_week = sum(daily_blocks_in_week.values())
        for i_idx in range(len(surgeries_in_week)):
            model.addConstr(
                quicksum(Z_vars[s_idx, i_idx, b_idx_flat] for b_idx_flat in range(num_block_slots_in_week)) + R_vars[s_idx, i_idx] == 1,
                name=f"assign_week{s_idx}_surg{i_idx}",
            )

    # Objective
    cost_rejection_param = PARAMS["cost_rejection_per_case"]
    l1_regularization_param = PARAMS["benders_lambda_l1_theta"]

    rejection_cost_expr = quicksum(
        cost_rejection_param * surgeries_in_week[i_idx][COL_BOOKED_MIN] * R_vars[s_idx, i_idx]
        for s_idx, (surgeries_in_week, _, _) in enumerate(weekly_data_list)
        for i_idx in range(len(surgeries_in_week))
    )
    recourse_cost_expr = quicksum(eta_vars[s_idx] for s_idx in range(num_training_weeks))
    l1_penalty_expr = l1_regularization_param * quicksum(abs_theta_penalty_vars[j] for j in range(num_features))

    model.setObjective(rejection_cost_expr + recourse_cost_expr + l1_penalty_expr, GRB.MINIMIZE)
    model.update()

    return model, Z_vars, R_vars, theta_vars, eta_vars