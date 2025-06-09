# --- Make project root importable when run as a script from integrated/ ---
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------------------------------


def main() -> None:
    # --------------------------------------------------------------
    # 0) imports  (same as before, plus json at end)
    # --------------------------------------------------------------
    import gurobipy as gp
    from gurobipy import GRB, quicksum
    import numpy as np, json
    from pathlib import Path
    import pandas as pd

    from src.config import PARAMS
    from src.data_processing import (
        load_data,
        split_data,
        add_time_features,
        compute_block_capacity,
    )
    from src.predictors import _prepare_features_target, _create_sklearn_preprocessor
    from src.constants import (
        ALL_FEATURE_COLS,
        COL_BOOKED_MIN,
        GUROBI_VAR_X_PREFIX,
        GUROBI_VAR_R_PREFIX,
        GUROBI_VAR_OT_PREFIX,
        GUROBI_VAR_IT_PREFIX,
        MAX_OVERTIME_MINUTES_PER_BLOCK,
    )
    from src.solver_utils import set_gurobi_model_parameters
    from sklearn.linear_model import Lasso

    # --------------------------------------------------------------
    # 1) data & warm-up sets
    # --------------------------------------------------------------
    df = load_data(PARAMS)
    df = add_time_features(df)
    df_warm, df_pool, horizon_start = split_data(df, PARAMS)

    X_warm, y_warm = _prepare_features_target(df_warm)
    preproc = _create_sklearn_preprocessor()
    Xw = preproc.fit_transform(X_warm)
    yw = y_warm.to_numpy()

    # Lasso warm-start for θ
    theta_init = (
        Lasso(alpha=PARAMS["integrated_lambda"], fit_intercept=False, max_iter=10_000)
        .fit(Xw, yw)
        .coef_
    )

    # --------------------------------------------------------------
    # 2) build one big planning set for P weeks
    # --------------------------------------------------------------
    P = PARAMS["num_planning_weeks"]  
    all_rows = []  # raw DataFrame rows
    week_of = []  # parallel list of week s
    blocks = []  # (s, day, blk)
    start0 = horizon_start  # first planning Monday

    for s in range(P):
        w_start = start0 + np.timedelta64(7 * s, "D")
        w_end = w_start + np.timedelta64(6, "D")

        df_week = df_pool[
            (df_pool["actual_start"].dt.normalize() >= w_start.normalize())
            & (df_pool["actual_start"].dt.normalize() <= w_end.normalize())
        ]
        if df_week.empty:
            continue  # skip empty weeks

        # append surgeries
        all_rows.append(df_week)
        week_of.extend([s] * len(df_week))

        # compute capacity for this week
        daily_cnt = compute_block_capacity(
            df_week, {**PARAMS, "_internal_horizon_start_date": w_start}
        )
        blocks.extend(
            [(s, day, blk) for day, cnt in daily_cnt.items() for blk in range(cnt)]
        )

    if not all_rows:
        print("No surgeries in the planning horizon – nothing to solve.")
        return

    df_plan_all = pd.concat(all_rows, ignore_index=True)
    N_plan = len(df_plan_all)

    X_plan = df_plan_all[ALL_FEATURE_COLS]
    bp = df_plan_all[COL_BOOKED_MIN].to_numpy()
    Xp = preproc.transform(X_plan)
    week_of = np.array(week_of)  # numpy for fast masking

    # --------------------------------------------------------------
    # 3) dimensions & constants
    # --------------------------------------------------------------
    Nw, p = Xw.shape
    C = PARAMS["block_size_minutes"]
    MIN_DUR = PARAMS["min_procedure_duration"]
    MAX_DUR = C + MAX_OVERTIME_MINUTES_PER_BLOCK

    ALPHA = PARAMS["integrated_alpha"]
    LAMBDA = PARAMS["integrated_lambda"]
    c_r = PARAMS["cost_rejection_per_case"]
    c_o = PARAMS["cost_overtime_per_min"]
    c_i = PARAMS["cost_idle_per_min"]

    # --------------------------------------------------------------
    # 4) model
    # --------------------------------------------------------------
    m = gp.Model("MultiWeek_JointLearning")
    set_gurobi_model_parameters(m, PARAMS)

    theta = m.addMVar(p, lb=-GRB.INFINITY, name="theta")
    d_var = m.addMVar(N_plan, lb=MIN_DUR, ub=MAX_DUR, name="d")
    e = m.addMVar(Nw, lb=0.0, name="e")
    u = m.addMVar(p, lb=0.0, name="u")

    # binaries / auxiliaries
    Z = m.addVars(
        (
            (i, s, day, blk)
            for i in range(N_plan)
            for (s, day, blk) in blocks
            if week_of[i] == s
        ),
        vtype=GRB.BINARY,
        name=GUROBI_VAR_X_PREFIX,
    )
    R = m.addVars(range(N_plan), vtype=GRB.BINARY, name=GUROBI_VAR_R_PREFIX)
    ot = m.addVars(blocks, lb=0.0, name=GUROBI_VAR_OT_PREFIX)
    it_ = m.addVars(blocks, lb=0.0, name=GUROBI_VAR_IT_PREFIX)
    Y = m.addVars(
        (
            (i, s, day, blk)
            for i in range(N_plan)
            for (s, day, blk) in blocks
            if week_of[i] == s
        ),
        lb=0.0,
        name="Y",
    )
    m.update()

    # ----- MIP start: reject all (optional) -----------------------
    for i in range(N_plan):
        R[i].Start = 1
    for key, z in Z.items():
        z.Start = 0
        Y[key].Start = 0
    for j in range(p):
        theta[j].Start = float(theta_init[j])

    # ------------- constraints -----------------------------------
    # prediction link
    m.addConstr(d_var == Xp @ theta, name="pred_link")

    # warm-up absolute value
    for t in range(Nw):
        m.addConstr(e[t] >= Xw[t, :] @ theta - yw[t])
        m.addConstr(e[t] >= -Xw[t, :] @ theta + yw[t])

    # ℓ1
    m.addConstr(u >= theta)
    m.addConstr(u >= -theta)

    # assignment or rejection
    for i in range(N_plan):
        s = week_of[i]
        m.addConstr(
            quicksum(Z[i, s, day, blk] for (ss, day, blk) in blocks if ss == s) + R[i]
            == 1
        )

    # indicators + block balance
    for i in range(N_plan):
        s = week_of[i]
        for ss, day, blk in blocks:
            if ss != s:
                continue
            m.addGenConstrIndicator(
                Z[i, s, day, blk], True, Y[i, s, day, blk] == d_var[i]
            )
            m.addGenConstrIndicator(Z[i, s, day, blk], False, Y[i, s, day, blk] == 0)

    for s, day, blk in blocks:
        m.addConstr(
            quicksum(Y[i, s, day, blk] for i in range(N_plan) if week_of[i] == s)
            + it_[s, day, blk]
            - ot[s, day, blk]
            == C
        )

    # ------------- objective -------------------------------------
    pred_term = e.sum() + LAMBDA * u.sum()
    sched_term = (
        c_r * quicksum(bp[i] * R[i] for i in range(N_plan))
        + c_o * ot.sum()
        + c_i * it_.sum()
    )
    m.setObjective((1 - ALPHA) * pred_term + ALPHA * sched_term, GRB.MINIMIZE)

    # ------------- solve -----------------------------------------
    m.Params.OutputFlag = 1  # turn solver logging on
    m.Params.LogToConsole = 1  # ensure it goes to stdout
    m.Params.MIPFocus = 1
    m.Params.Cuts = 2
    m.Params.Presolve = 2
    m.Params.Method = 2
    m.optimize()

    # ------------- output ----------------------------------------
    if m.SolCount > 0:
        total_cost = (
            c_r * sum(bp[i] * R[i].X for i in range(N_plan))
            + c_o * ot.sum().getValue()
            + c_i * it_.sum().getValue()
        )
        print(f"Scheduling cost across {P} weeks: {total_cost:,.2f}")

        # save θ

        def _clean_feat_name(s: str) -> str:
            if s.startswith("num__"):
                return s.replace("num__", "", 1)
            if s.startswith("cat__"):
                # cat__case_service_UROLOGY  ->  case_service=UROLOGY
                base = s.replace("cat__", "", 1)
                col, *rest = base.split("_", 1)
                return f"{col}={rest[0]}" if rest else col
            return s

        feat_names = list(preproc.get_feature_names_out())
        clean_names = [_clean_feat_name(n) for n in feat_names]
        theta_out = {clean_names[j]: float(theta[j].X) for j in range(len(feat_names))}

        out_path = Path(PARAMS["theta_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(theta_out, out_path.open("w"), indent=2)
        print(f"θ saved to {out_path}")
    else:
        print("Solver status:", m.Status)


if __name__ == "__main__":
    main()
