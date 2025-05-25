"""
Master problem module for integrated Benders decomposition.
Builds the two-stage master MIP with assignment, rejection, theta, and eta vars.
"""

from typing import Any, Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB, quicksum
from src.config import PARAMS


def build_benders_master_problem(
    weekly_data_list: List[Tuple[List[Dict[str, Any]], List[int], Dict[int, float]]],
    num_features: int,
    params: Dict[str, Any],
    initial_theta: Dict[int, float] = None,
) -> Tuple[
    gp.Model,
    Dict[Tuple[int, int, int], gp.Var],  # Z_vars
    Dict[Tuple[int, int], gp.Var],  # R_vars
    Dict[int, gp.Var],  # theta_vars
    Dict[int, gp.Var],  # eta_vars
]:
    """
    Construct the Benders master problem.

    Args:
        weekly_data_list: Sequence of (surgeries, blocks, actual_map) for each scenario week.
            - surgeries: list of surgery dicts for that week
            - blocks:    list of block capacities (minutes) for each block index
            - actual_map: dict mapping surgery['id'] -> realized duration
        num_features:     Number of features in regression (p).
        params:           Parameter dict containing keys:
                          - "cost_rejection_per_case"
                          - "benders_lambda_l1_theta"
        initial_theta:    Optional warm-start values mapping feature index -> theta.

    Returns:
        model:      Gurobi Model for the master problem.
        Z_vars:     Mapping (week, surgery_index, block_index) -> binary Var.
        R_vars:     Mapping (week, surgery_index) -> binary reject Var.
        theta_vars: Mapping feature_index -> continuous theta Var.
        eta_vars:   Mapping week -> continuous eta Var for recourse cost.
    """
    # Initialize model
    model = gp.Model("BendersMaster")
    # Master parameters (could be set here)
    # model.Params.OutputFlag = 0

    # Extract parameters
    c_rej = PARAMS["cost_rejection_per_case"]
    lambda_l1 = PARAMS["benders_lambda_l1_theta"]

    # 1) Decision variables
    Z_vars: Dict[Tuple[int, int, int], gp.Var] = {}
    R_vars: Dict[Tuple[int, int], gp.Var] = {}
    eta_vars: Dict[int, gp.Var] = {}

    for s, (surgeries, blocks, _) in enumerate(weekly_data_list):
        # Rejection var per surgery
        for i, surg in enumerate(surgeries):
            R_vars[(s, i)] = model.addVar(vtype=GRB.BINARY, name=f"R_s{s}_i{i}")
            # Assignment var per block index
            for b in range(len(blocks)):
                Z_vars[(s, i, b)] = model.addVar(
                    vtype=GRB.BINARY, name=f"Z_s{s}_i{i}_b{b}"
                )
        # Recourse cost var
        eta_vars[s] = model.addVar(lb=0.0, name=f"eta_s{s}")

    # Theta variables and L1 proxies
    theta_vars: Dict[int, gp.Var] = {
        j: model.addVar(lb=-GRB.INFINITY, name=f"theta_{j}")
        for j in range(num_features)
    }
    abs_theta_vars: Dict[int, gp.Var] = {
        j: model.addVar(lb=0, name=f"abs_theta_{j}") for j in theta_vars
    }

    model.update()

    # 2) Constraints
    # 2a) Assignment: sum_b Z + R = 1
    for s, (surgeries, blocks, _) in enumerate(weekly_data_list):
        for i in range(len(surgeries)):
            model.addConstr(
                quicksum(Z_vars[(s, i, b)] for b in range(len(blocks))) + R_vars[(s, i)]
                == 1,
                name=f"assign_s{s}_i{i}",
            )
    # 2b) Recourse cost cuts: eta_s >= Q_s(...) added via callback
    # No static cuts here; will use lazy constraints

    # 2c) L1 linkage for theta
    for j in theta_vars:
        model.addConstr(theta_vars[j] <= abs_theta_vars[j], name=f"l1_pos_{j}")
        model.addConstr(-theta_vars[j] <= abs_theta_vars[j], name=f"l1_neg_{j}")

    # 3) Objective
    # Rejection cost + eta recourse + lambda * ||theta||_1
    obj_rej = quicksum(
        c_rej * surgeries[i]["booked_time_minutes"] * R_vars[(s, i)]
        for s, (surgeries, _, _) in enumerate(weekly_data_list)
        for i in range(len(surgeries))
    )
    obj_rec = quicksum(eta_vars[s] for s in eta_vars)
    obj_reg = lambda_l1 * quicksum(abs_theta_vars[j] for j in abs_theta_vars)

    model.setObjective(obj_rej + obj_rec + obj_reg, GRB.MINIMIZE)
    model.update()

    # 4) Warm-start theta if available
    if initial_theta is not None:
        for j, val in initial_theta.items():
            if j in theta_vars:
                theta_vars[j].Start = val

    return model, Z_vars, R_vars, theta_vars, eta_vars
