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
    daily_blocks_info: Dict[int, int],
    theta_values_by_index: Dict[int, float],
    Z_bar_assignments: Dict[Tuple[int, int], int],
    R_bar_rejections: Dict[int, int],
    df_pool_reference: pd.DataFrame,
    ordered_feature_names: List[str],
    params_config: Dict[str, Any],
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """
    Solve the LP subproblem for one week given fixed assignments Z_bar and theta.
    Returns:
      - Q_val: optimal overtime + idle cost,
      - subgradients g[(i, b)] for each surgery i and flat‐block index b.
    """

    # --- 1. Compute predicted p_i and actual d_i for each surgery ---
    num_surgeries = len(surgeries_info)
    num_features = len(ordered_feature_names)

    predicted_durations_p_i: List[float] = []
    actual_durations_d_i:   List[float] = []

    for i, surg in enumerate(surgeries_info):
        fv = build_feature_vector(
            surgery_dict=surg,
            df_pool_full_data=df_pool_reference,
            feature_names_ordered=ordered_feature_names,
            scaling_method=params_config["feature_scaling"],
        )
        raw_pred = sum(theta_values_by_index[j] * fv[j] for j in range(num_features))
        predicted = max(MIN_PROCEDURE_DURATION, raw_pred)
        predicted_durations_p_i.append(predicted)

        actual = float(surg.get(COL_ACTUAL_DUR_MIN, surg.get(COL_BOOKED_MIN)))
        actual_durations_d_i.append(actual)

    # --- 2. Flatten the per-day block counts into a single list of flat blocks ---
    flat_block_tuples: List[Tuple[int, int]] = []
    for day_idx in sorted(daily_blocks_info):
        for b in range(daily_blocks_info[day_idx]):
            flat_block_tuples.append((day_idx, b))
    B = len(flat_block_tuples)

    # --- 3. Build & configure the Gurobi LP subproblem ---
    m = gp.Model("Subproblem_LP_Week")
    subparams = params_config.get("gurobi_subproblem_milp_settings", {})
    set_gurobi_model_parameters(m, subparams)

    C = params_config["block_size_minutes"]
    c_ot = params_config["cost_overtime_per_min"]
    c_it = params_config["cost_idle_per_min"]
    eps = params_config["epsilon_block_mae"]

    OT = m.addVars(range(B), lb=0.0, name=GUROBI_VAR_OT_PREFIX)
    IT = m.addVars(range(B), lb=0.0, name=GUROBI_VAR_IT_PREFIX)
    U  = m.addVars(range(B), lb=0.0, name=GUROBI_VAR_U_LEARN_ABS_ERROR_PREFIX)

    cap_cons = {}
    pos_cons = {}
    neg_cons = {}

    for b in range(B):
        # which surgeries are assigned to flat block b?
        assigned_idxs = [
            i for (i, fb), val in Z_bar_assignments.items() if fb == b and val == 1
        ]

        # capacity balance: sum p_i - C = OT - IT
        expr_p = quicksum(predicted_durations_p_i[i] for i in assigned_idxs)
        cap_cons[b] = m.addConstr(
            expr_p - C == OT[b] - IT[b], name=f"cap_{b}"
        )
        m.addConstr(OT[b] <= MAX_OVERTIME_MINUTES_PER_BLOCK, name=f"otcap_{b}")

        # error bounds: |∑(p_i - d_i)| ≤ U[b]
        expr_err = quicksum(
            (predicted_durations_p_i[i] - actual_durations_d_i[i]) for i in assigned_idxs
        )
        pos_cons[b] = m.addConstr(expr_err <= U[b], name=f"err_pos_{b}")
        neg_cons[b] = m.addConstr(-expr_err <= U[b], name=f"err_neg_{b}")

        # enforce U[b] ≤ eps
        m.addConstr(U[b] <= eps, name=f"err_eps_{b}")

    # objective: minimize overtime + idle and prediction error
    w_pred = params_config["prediction_error_weight"]
    m.setObjective(
        quicksum(c_ot * OT[b] + c_it * IT[b] for b in range(B))
      + w_pred * quicksum(U[b] for b in range(B)),
        GRB.MINIMIZE
    )

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        # infeasible or unbounded → signal with infinite cost and zero gradient
        return float("inf"), {k: 0.0 for k in Z_bar_assignments}

    Q_val = m.ObjVal

    # --- 4. Extract duals for subgradient computation ---
    lambda_duals = {b: cap_cons[b].Pi for b in cap_cons}
    mu_pos      = {b: pos_cons[b].Pi for b in pos_cons}
    mu_neg      = {b: neg_cons[b].Pi for b in neg_cons}

    # --- 5. Compute g_{i,b} for all (i,b) ---
    grads: Dict[Tuple[int,int], float] = {}
    for i in range(num_surgeries):
        for b in range(B):
            # derivative contributions from capacity and error constraints
            dp = predicted_durations_p_i[i]
            dd = actual_durations_d_i[i]
            g = (
                lambda_duals.get(b, 0.0) * dp
                + mu_pos .get(b, 0.0) * (dp - dd)
                + mu_neg .get(b, 0.0) * (-(dp - dd))
            )
            grads[(i, b)] = g

    return Q_val, grads
