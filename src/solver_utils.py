import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np


def _add_single_spillover_cap(m, x, dur, block_tuples, cap=480):
    # z[i, d, b] – binary “case i is spill‑over in block (d,b)”
    z = m.addVars(
        [(i, d, b) for (d, b) in block_tuples for i in dur],
        vtype=GRB.BINARY,
        name="zspill",
    )

    for d, b in block_tuples:
        m.addConstr(quicksum(z[i, d, b] for i in dur) <= 1, name=f"one_spill_{d}_{b}")

        for i in dur:
            m.addConstr(z[i, d, b] <= x[i, d, b], name=f"link_zx_{i}_{d}_{b}")

        m.addConstr(
            quicksum(dur[i] * x[i, d, b] for i in dur)
            <= cap + quicksum(dur[i] * z[i, d, b] for i in dur),
            name=f"cap_with_spill_{d}_{b}",
        )


def solve_saa_model(surgeries, day_blocks, params, scen_matrix):
    """
    Two-stage Sample Average Approximation with shared first-stage x/r
    and block-level OT/idle in the recourse.

    Parameters
    ----------
    surgeries : list of dict
        Each dict must contain keys "booked_min".
    day_blocks : dict[int, int]
        Mapping day index -> number of blocks that day.
    scen_matrix : np.ndarray, shape (N, K)
        Each column k is a vector of sampled durations.

    Returns
    -------
    dict with keys "obj", "status", "model"
    """
    N = len(surgeries)
    H = params["planning_horizon_days"]
    K = params["saa_scenarios"]
    blk_min = params["block_size_minutes"]
    c_rej = params["cost_rejection_per_case"]
    c_ot = params["cost_overtime_per_min"]
    c_idle = params["cost_idle_per_min"]

    m = gp.Model("SAA")
    set_grb_params(m, params)

    # First‐stage vars: x[i,d,b] and r[i]
    x = m.addVars(
        [
            (i, d, b)
            for i in range(N)
            for d in range(H)
            for b in range(day_blocks.get(d, 0))
        ],
        vtype=GRB.BINARY,
        name="x",
    )
    r = m.addVars(N, vtype=GRB.BINARY, name="r")

    # Assignment or rejection
    for i in range(N):
        m.addConstr(
            gp.quicksum(
                x[i, d, b] for d in range(H) for b in range(day_blocks.get(d, 0))
            )
            + r[i]
            == 1,
            name=f"assign_{i}",
        )

    # ---------- prepare durations dict --------------------------
    dur = {
        i: surg.get("predicted_dur_min", surg["booked_min"])
        for i, surg in enumerate(surgeries)
    }

    block_tuples = [(d, b) for d in range(H) for b in range(day_blocks[d])]
    # ---------- add single‑spillover cap ------------------------
    _add_single_spillover_cap(m, x, dur, block_tuples, params["block_size_minutes"])
    # -----------------------------------------------------------------

    # Scenario recourse
    recourse = 0.0
    for k in range(K):
        ot = m.addVars(
            [(d, b, k) for d in range(H) for b in range(day_blocks.get(d, 0))],
            lb=0.0,
            name=f"ot_{k}",
        )
        it = m.addVars(
            [(d, b, k) for d in range(H) for b in range(day_blocks.get(d, 0))],
            lb=0.0,
            name=f"it_{k}",
        )

        # Capacity constraints per block
        for d in range(H):
            for b in range(day_blocks.get(d, 0)):
                # sum of sampled durations in this block
                m.addConstr(
                    gp.quicksum(scen_matrix[i, k] * x[i, d, b] for i in range(N))
                    - blk_min
                    <= ot[d, b, k],
                    name=f"ot_constr_{d}_{b}_{k}",
                )
                m.addConstr(
                    blk_min
                    - gp.quicksum(scen_matrix[i, k] * x[i, d, b] for i in range(N))
                    <= it[d, b, k],
                    name=f"it_constr_{d}_{b}_{k}",
                )

        recourse += (c_ot * ot.sum() + c_idle * it.sum()) / K

    # Rejection cost
    rej = c_rej * gp.quicksum(surgeries[i]["booked_min"] * r[i] for i in range(N))

    # Objective
    m.setObjective(rej + recourse, GRB.MINIMIZE)
    m.optimize()

    # ------------------------------------------------------------------
    # print(f"\n--- Solver Debug: SAA Model ---")
    # print(f"Model: {m.ModelName}, N_surgeries: {N}")
    # print(f"Status: {m.Status}, SolCount: {m.SolCount}")
    # if m.SolCount > 0:
    #     print(f"ObjVal: {m.ObjVal:.2f}, MIPGap: {m.MIPGap:.4f}, Runtime: {m.Runtime:.2f}s")

    #     # Save model and solution for external inspection
    #     model_name_prefix = "saa"
    #     m.write(f"debug_{model_name_prefix}.lp")
    #     m.write(f"debug_{model_name_prefix}.sol")

    # else:
    #     print("No solution found by Gurobi.")
    # print("--- End Solver Debug ---\n")
    # ------------------------------------------------------------------

    return {"obj": m.ObjVal, "status": m.Status, "model": m}


def solve_deterministic_model(surgeries, day_blocks, params):
    """Plans with BOOKED times (hospital reality benchmark)."""
    return _solve_single_stage(surgeries, day_blocks, params, mode="booked")


def solve_predictive_model(surgeries, day_blocks, params):
    """Plans with PREDICTED durations."""
    return _solve_single_stage(surgeries, day_blocks, params, mode="pred")


def solve_clairvoyant_model(surgeries, day_blocks, params):
    """Oracle model that knows ACTUAL durations in advance."""
    return _solve_single_stage(surgeries, day_blocks, params, mode="actual")


def _solve_single_stage(
    surgeries: list[dict],
    day_blocks: dict[int, int],
    params: dict,
    *,
    mode: str = "booked",
):
    """
    Core single‑stage assignment model.

    Parameters
    ----------
    surgeries
        Each surgery dict must contain the keys
        "booked_min", "predicted_dur_min", and "actual_dur_min".
    day_blocks
        Mapping {day_idx: number_of_blocks_that_day}.
    params
        Configuration dictionary (costs, block length, etc.).
    mode : {"booked", "pred", "actual"}, default "booked"
        Which duration the optimiser should treat as known:
        - "booked"  → use the time booked by the OR clerk (baseline deterministic).
        - "pred"    → use machine‑learning prediction (learning‑augmented).
        - "actual"  → clairvoyant lower bound that sees the realised duration.
    """
    if mode not in {"booked", "pred", "actual"}:
        raise ValueError(f"mode must be 'booked', 'pred', or 'actual' (got {mode})")

    # ------------------------------------------------------------------
    # Shorthand aliases
    # ------------------------------------------------------------------
    N = len(surgeries)
    H = params["planning_horizon_days"]
    blk_min = params["block_size_minutes"]
    c_rej = params["cost_rejection_per_case"]
    c_ot = params["cost_overtime_per_min"]
    c_idle = params["cost_idle_per_min"]

    # ------------------------------------------------------------------
    # Build MIP
    # ------------------------------------------------------------------
    m = gp.Model(f"single_stage_{mode}")
    set_grb_params(m, params)

    # x[i,d,b] = 1 if surgery i assigned to block b on day d
    x = m.addVars(
        [
            (i, d, b)
            for i in range(N)
            for d in range(H)
            for b in range(day_blocks.get(d, 0))
        ],
        vtype=GRB.BINARY,
        name="x",
    )

    # r[i] = 1 if surgery i is rejected
    r = m.addVars(N, vtype=GRB.BINARY, name="r")

    # ot[d,b], it[d,b] = overtime / idle minutes for each block
    ot = m.addVars(
        [(d, b) for d in range(H) for b in range(day_blocks.get(d, 0))],
        lb=0.0,
        name="ot",
    )
    it = m.addVars(
        [(d, b) for d in range(H) for b in range(day_blocks.get(d, 0))],
        lb=0.0,
        name="it",
    )

    # ------------------------------------------------------------------
    # 1) Each surgery either scheduled exactly once or rejected
    # ------------------------------------------------------------------
    for i in range(N):
        m.addConstr(
            gp.quicksum(
                x[i, d, b] for d in range(H) for b in range(day_blocks.get(d, 0))
            )
            + r[i]
            == 1,
            name=f"assign_{i}",
        )

    # ------------------------------------------------------------------
    # 2) Block capacity, overtime & idle
    # ------------------------------------------------------------------
    for d in range(H):
        for b in range(day_blocks.get(d, 0)):
            dur_sum = gp.quicksum(
                (
                    surgeries[i]["predicted_dur_min"]
                    if mode == "pred"
                    else (
                        surgeries[i]["actual_dur_min"]
                        if mode == "actual"
                        else surgeries[i]["booked_min"]
                    )
                )
                * x[i, d, b]
                for i in range(N)
            )
            # dur_sum  - blk <= ot
            m.addConstr(dur_sum - blk_min <= ot[d, b], name=f"ot_{d}_{b}")
            # blk - dur_sum <= it
            m.addConstr(blk_min - dur_sum <= it[d, b], name=f"it_{d}_{b}")
            # ot + it <= 120
            m.addConstr(ot[d, b] <= 120, name=f"ot_cap_{d}_{b}")

    # ---------- prepare durations dict --------------------------
    if mode == "pred":
        dur = {i: surg["predicted_dur_min"] for i, surg in enumerate(surgeries)}
    elif mode == "actual":
        dur = {i: surg["actual_dur_min"] for i, surg in enumerate(surgeries)}
    else:  # "booked"
        dur = {i: surg["booked_min"] for i, surg in enumerate(surgeries)}

    block_tuples = [(d, b) for d in range(H) for b in range(day_blocks[d])]

    # ---------- add single‑spillover cap ------------------------
    _add_single_spillover_cap(m, x, dur, block_tuples, params["block_size_minutes"])
    # -----------------------------------------------------------------
    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    obj = (
        c_rej * gp.quicksum(surgeries[i]["booked_min"] * r[i] for i in range(N))
        + c_ot * ot.sum()
        + c_idle * it.sum()
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    m.optimize()

    # ------------------------------------------------------------------
    # print(f"\n--- Solver Debug: {mode} Model ---")
    # print(f"Model: {m.ModelName}, N_surgeries: {N}")
    # print(f"Status: {m.Status}, SolCount: {m.SolCount}")
    # if m.SolCount > 0:
    #     print(f"ObjVal: {m.ObjVal:.2f}, MIPGap: {m.MIPGap:.4f}, Runtime: {m.Runtime:.2f}s")

    #     # Save model and solution for external inspection
    #     model_name_prefix = mode
    #     m.write(f"debug_{model_name_prefix}.lp")
    #     m.write(f"debug_{model_name_prefix}.sol")

    # else:
    #     print("No solution found by Gurobi.")
    # print("--- End Solver Debug ---\n")
    # ------------------------------------------------------------------

    return {
        "obj": m.ObjVal,
        "status": m.status,
        "model": m,
    }


def set_grb_params(model, p):
    """Assign common Gurobi parameters from PARAMS dict."""
    model.Params.OutputFlag = p["gurobi_output_flag"]
    model.Params.TimeLimit = p["gurobi_timelimit"]
    model.Params.MIPGap = p["gurobi_mipgap"]
    model.Params.OutputFlag = p["gurobi_output_flag"]
    model.Params.Heuristics = p["gurobi_heuristics"]
    model.Params.Presolve = 2  # model.Params.RINS = p["gurobi_rins"]
    model.Params.Threads = p["gurobi_threads"]
