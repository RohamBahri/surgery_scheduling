# --- Make project root importable when run as a script from integrated/ ---
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------------------------------
import logging
import time
import json

import pandas as pd
import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum

from src.config import PARAMS
from src.data_processing import build_feature_vector, load_data, split_data
from src.predictors import train_lasso_predictor
from src.constants import COL_BOOKED_MIN
from integrated.run_integrated import prepare_weekly_data_for_benders_training


def extract_lasso_coefs(lasso_pipe):
    """
    Pull feature names and coefficients out of the fitted LASSO pipeline.
    Mirrors run_integrated.py behavior.
    Returns (feat_names: List[str], initial_theta: Dict[int, float]).
    """
    if lasso_pipe is None:
        raise RuntimeError("LASSO training failed; cannot extract coefficients.")
    pre = lasso_pipe.named_steps["preprocessor"]
    feat_names = list(pre.get_feature_names_out())
    coefs = lasso_pipe.named_steps["regressor"].coef_
    # handle single-feature edge-case
    if isinstance(coefs, (float, int)):
        coefs = [coefs]
    # map each feature index to its coef (or zero if missing)
    initial_theta = {
        j: float(coefs[j]) if j < len(coefs) else 0.0 for j in range(len(feat_names))
    }
    logging.getLogger("MonolithicYear").info(
        f"Extracted {len(feat_names)} features from LASSO for warm start."
    )
    return feat_names, initial_theta


def build_monolithic_model(weekly_data, feat_names, df_pool_reference):
    """
    Monolithic MIP over all weeks, with |sum_i (pred_i - actual_i)*Z| ≤ ε per block.
    """
    model = gp.Model("MonolithicScheduler")
    model.Params.TimeLimit = PARAMS["time_limit_monolithic"]
    model.Params.OutputFlag = 1

    # θ_j and its L1 proxy
    theta = {
        j: model.addVar(lb=-GRB.INFINITY, name=f"theta_{j}")
        for j in range(len(feat_names))
    }
    abs_theta = {j: model.addVar(lb=0, name=f"abs_theta_{j}") for j in theta}

    # Assignment binaries Z[w,i,b] and rejection R[w,i]
    Z = {}
    R = {}
    for w, (surgeries, blocks, actual_map) in enumerate(weekly_data):
        for i, surg in enumerate(surgeries):
            R[w, i] = model.addVar(vtype=GRB.BINARY, name=f"R_w{w}_i{i}")
            for b in range(len(blocks)):
                Z[w, i, b] = model.addVar(vtype=GRB.BINARY, name=f"Z_w{w}_i{i}_b{b}")

    # OT/IT per block
    OT = {}
    IT = {}
    U = {}
    for w, (_, blocks, _) in enumerate(weekly_data):
        for b in range(len(blocks)):
            OT[w, b] = model.addVar(lb=0, name=f"OT_w{w}_b{b}")
            IT[w, b] = model.addVar(lb=0, name=f"IT_w{w}_b{b}")
            # U slack for block‐level absolute error
            U[w, b]  = model.addVar(lb=0, name=f"U_w{w}_b{b}")
    

    model.update()

    # 1) Each surgery: assign to exactly one block or reject
    for w, (surgeries, blocks, _) in enumerate(weekly_data):
        for i in range(len(surgeries)):
            model.addConstr(
                quicksum(Z[w, i, b] for b in range(len(blocks))) + R[w, i] == 1,
                name=f"assign_w{w}_i{i}",
            )

    # 2) Capacity balance per block
    for w, (surgeries, blocks, _) in enumerate(weekly_data):
        for b, cap in enumerate(blocks):
            expr = LinExpr()
            for i, surg in enumerate(surgeries):
                # feature vector and pred duration
                p_i = build_feature_vector(
                    surgery_dict=surg,
                    df_pool_full_data=df_pool_reference,
                    feature_names_ordered=feat_names,
                    scaling_method=PARAMS["feature_scaling"],
                )
                for j, xij in enumerate(p_i):
                    expr += xij * theta[j] * Z[w, i, b]
            model.addConstr(
                expr - cap == OT[w, b] - IT[w, b], name=f"capacity_w{w}_b{b}"
            )

    # 3) Absolute-sum error bound per block: |∑(pred - actual)*Z| ≤ ε
    eps = PARAMS["epsilon_block_mae"]
    for w, (surgeries, blocks, actual_map) in enumerate(weekly_data):
        for b in range(len(blocks)):
            err_sum = LinExpr()
            for i, surg in enumerate(surgeries):
                # pred minus actual
                p_i = build_feature_vector(
                    surgery_dict=surg,
                    df_pool_full_data=df_pool_reference,
                    feature_names_ordered=feat_names,
                    scaling_method=PARAMS["feature_scaling"],
                )
                di = actual_map[surg["id"]]
                # build (pred-actual)*Z term
                term = LinExpr()
                for j, xij in enumerate(p_i):
                    term += xij * theta[j]
                term.addConstant(-di)
                err_sum += term * Z[w, i, b]
            model.addConstr(err_sum <= eps, name=f"error_pos_w{w}_b{b}")
            model.addConstr(-err_sum <= eps, name=f"error_neg_w{w}_b{b}")

    # 4) L1 linking for θ
    for j in theta:
        model.addConstr(theta[j] <= abs_theta[j], name=f"l1pos_{j}")
        model.addConstr(-theta[j] <= abs_theta[j], name=f"l1neg_{j}")

    # 4½) Prediction‐error cost: sum of block‐level absolute errors
    #          U[w,b] already equals |∑ᵢ (predᵢ−actᵢ)*Z[w,i,b]| ≤ ε
    prediction_cost = quicksum(
        U[w, b]
        for w, (_, blocks, _) in enumerate(weekly_data)
        for b in range(len(blocks))
    )

    model.update()

    # Objective: rejection + OT/IT + λ‖θ‖₁
    rej_cost = quicksum(
        PARAMS["cost_rejection_per_case"] * surg[COL_BOOKED_MIN] * R[w, i]
        for w, (surgeries, _, _) in enumerate(weekly_data)
        for i, surg in enumerate(surgeries)
    )
    otit_cost = quicksum(
        PARAMS["cost_overtime_per_min"] * OT[w, b]
        + PARAMS["cost_idle_per_min"] * IT[w, b]
        for w, (_, blocks, _) in enumerate(weekly_data)
        for b in range(len(blocks))
    )
    l1_cost = PARAMS["benders_lambda_l1_theta"] * quicksum(
        abs_theta[j] for j in abs_theta
    )
    PRED_COST_WEIGHT = PARAMS["prediction_error_weight"]

    model.setObjective(
        rej_cost + otit_cost + l1_cost + (PRED_COST_WEIGHT * prediction_cost),
        GRB.MINIMIZE,
    )
    model.update()

    return model, theta, Z, R


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MonolithicYear")

    # 1) Load & preprocess full OR dataset

    df_raw = load_data(PARAMS)  # has 'actual_start', etc.
    df_train, df_pool, pool_start_date = split_data(df_raw, PARAMS)

    # df_train = df_train.head(50)

    # 2) Train LASSO on the full warm-up DataFrame
    lasso_pipe = train_lasso_predictor(df_train, PARAMS)

    # 3) Prepare weekly_data for the big MIP
    weekly_data = prepare_weekly_data_for_benders_training(df_train, PARAMS)
    if not weekly_data:
        raise RuntimeError("No weekly data generated for the monolithic experiment.")

    # 4) Extract feature names & θ⁰
    feat_names, initial_theta = extract_lasso_coefs(lasso_pipe)

    # 3) Build the monolithic model
    t0 = time.time()
    model, theta_vars, Z_vars, R_vars = build_monolithic_model(
        weekly_data, feat_names, df_train
    )

    # 5) Warm-start θ
    for j, val in initial_theta.items():
        theta_vars[j].Start = val

    # 6) Optimize
    # print progress every time Gurobi explores 1,000 nodes (default is 1)
    model.Params.DisplayInterval = 1000

    # also write a full log file you can tail on disk
    model.Params.LogFile = "monolithic.log"

    # ensure console output is on
    model.Params.OutputFlag = 1

    model.optimize()
    elapsed = time.time() - t0
    logger.info(
        f"Monolithic solve finished in {elapsed:.1f}s. " f"Nodes={model.NodeCount}, "
    )

    # 7) Save learned θ directly into the JSON file that surgery_scheduling expects

    # Build a dict mapping feature→value
    theta_json = {
        feat_names[j]: theta_vars[j].x
        for j in sorted(theta_vars)
    }

    theta_path = PARAMS["theta_path"]
    # ensure parent directory exists
    from pathlib import Path
    Path(theta_path).parent.mkdir(parents=True, exist_ok=True)

    # Write out
    with open(theta_path, "w") as f:
        json.dump(theta_json, f, indent=2)

    logger.info(f"Saved learned θ JSON to {theta_path}")


if __name__ == "__main__":
    main()
