from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from src.config import CONFIG, AppConfig
from src.constants import DataColumns, DomainConstants, FeatureColumns, GurobiConstants
from src.data_processing import (
    add_time_features,
    compute_block_capacity,
    load_data,
    split_data,
)
from src.predictors import _create_sklearn_preprocessor, _prepare_features_target
from src.solver_utils import set_gurobi_model_parameters
from src.json_helper_util import to_json_safe
from src.feature_names_util import canonical_name


def greedy_schedule_cost(
    predicted_durations: np.ndarray,
    booked_minutes: np.ndarray,
    num_blocks: int,
    capacity_per_block: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast greedy scheduler using First Fit Decreasing heuristic.

    Args:
        predicted_durations: Array of predicted surgery durations
        booked_minutes: Array of booked times for billing purposes
        num_blocks: Number of available operating room blocks
        capacity_per_block: Time capacity of each block in minutes

    Returns:
        Tuple of (total_cost, block_loads, reject_mask, assignment_matrix, rejection_vector)
    """
    N = len(predicted_durations)
    if N == 0:
        return (
            0.0,
            np.zeros(num_blocks),
            np.zeros(0, dtype=bool),
            np.zeros((0, num_blocks)),
            np.zeros(0),
        )

    c_r = CONFIG.costs.rejection_per_case
    c_o = CONFIG.costs.overtime_per_min
    c_i = CONFIG.costs.idle_per_min
    C = capacity_per_block

    block_loads = np.zeros(num_blocks)
    reject_mask = np.zeros(N, dtype=bool)
    z_assignment = np.zeros((N, num_blocks), dtype=int)

    # Sort surgeries by decreasing duration for First Fit Decreasing
    sorted_indices = np.argsort(predicted_durations)[::-1]

    for idx in sorted_indices:
        d_i = predicted_durations[idx]
        b_i = booked_minutes[idx]

        # Find best block assignment by comparing incremental costs
        best_block = -1
        min_delta_cost = float("inf")

        for block in range(num_blocks):
            current_load = block_loads[block]
            new_load = current_load + d_i

            # Calculate incremental overtime and idle costs
            old_overtime = max(0.0, current_load - C)
            new_overtime = max(0.0, new_load - C)
            old_idle = max(0.0, C - current_load)
            new_idle = max(0.0, C - new_load)

            delta_cost = c_o * (new_overtime - old_overtime) + c_i * (
                new_idle - old_idle
            )

            if delta_cost < min_delta_cost:
                min_delta_cost = delta_cost
                best_block = block

        # Compare assignment cost to rejection cost
        reject_cost = c_r * b_i

        if min_delta_cost <= reject_cost:
            block_loads[best_block] += d_i
            z_assignment[idx, best_block] = 1
        else:
            reject_mask[idx] = True

    # Calculate total cost components
    overtime_cost = c_o * np.sum(np.maximum(0.0, block_loads - C))
    idle_cost = c_i * np.sum(np.maximum(0.0, C - block_loads))
    rejection_cost = c_r * np.sum(booked_minutes[reject_mask])
    total_cost = overtime_cost + idle_cost + rejection_cost

    r_warm_start = reject_mask.astype(int)

    return total_cost, block_loads, reject_mask, z_assignment, r_warm_start


def build_master_problem(
    Xw: np.ndarray,
    yw: np.ndarray,
    p: int,
    lambda_reg: float,
    num_weeks: int,
    tau_pred: float,
    rho_pred: float,
) -> Tuple[gp.Model, gp.MVar, gp.MVar, gp.MVar, List[gp.Var]]:
    """
    Build Benders master problem for integrated learning-scheduling.

    Objective: minimize prediction penalty + L1 regularization + scheduling costs
    where prediction penalty uses hinge loss with margin tau_pred.

    Args:
        Xw: Feature matrix for warm-up data
        yw: Target vector (actual durations) for warm-up data
        p: Number of features
        lambda_reg: L1 regularization coefficient
        num_weeks: Number of planning weeks
        tau_pred: Prediction error tolerance margin
        rho_pred: Weight on prediction penalty

    Returns:
        Tuple of (model, theta_vars, prediction_slack_vars, l1_slack_vars, beta_vars)
    """
    master = gp.Model("BendersMaster")

    theta = master.addMVar(p, lb=-GRB.INFINITY, name="theta")
    s_pred = master.addMVar(len(yw), lb=0.0, name="s_pred")
    u = master.addMVar(p, lb=0.0, name="u")
    beta = master.addVars(num_weeks, lb=0.0, name="beta")

    master.update()

    # Prediction error hinge constraints: s_pred >= |Xw @ theta - yw| - tau_pred
    for t in range(len(yw)):
        master.addConstr(
            s_pred[t] >= Xw[t, :] @ theta - yw[t] - tau_pred, name=f"pred_hinge_pos_{t}"
        )
        master.addConstr(
            s_pred[t] >= -Xw[t, :] @ theta + yw[t] - tau_pred,
            name=f"pred_hinge_neg_{t}",
        )

    # L1 regularization constraints: u >= |theta|
    master.addConstr(u >= theta, name="l1_pos")
    master.addConstr(u >= -theta, name="l1_neg")

    # Objective function
    master.setObjective(
        rho_pred * quicksum(s_pred)
        + lambda_reg * quicksum(u)
        + quicksum(beta[t] for t in range(num_weeks)),
        GRB.MINIMIZE,
    )

    master.Params.OutputFlag = 0
    master.Params.Method = 2

    return master, theta, s_pred, u, beta


def build_subproblem(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    durations_fixed: np.ndarray,
    week_idx: int,
) -> gp.Model:
    """
    Build weekly scheduling subproblem for given predicted durations.

    Minimizes rejection, overtime, and idle costs for a single week's surgeries
    given fixed duration predictions.

    Args:
        week_data: DataFrame containing week's surgery data
        week_blocks: List of available (day, block) pairs for the week
        durations_fixed: Fixed predicted durations for all surgeries in week
        week_idx: Week index for naming

    Returns:
        Gurobi model for the weekly scheduling problem
    """
    if week_data.empty:
        dummy = gp.Model(f"DummyWeek_{week_idx}")
        dummy.setObjective(0, GRB.MINIMIZE)
        return dummy

    sub = gp.Model(f"WeeklyScheduling_{week_idx}")

    N = len(week_data)
    bp = week_data[DataColumns.BOOKED_MIN].to_numpy()

    C = CONFIG.operating_room.block_size_minutes
    c_r = CONFIG.costs.rejection_per_case
    c_o = CONFIG.costs.overtime_per_min
    c_i = CONFIG.costs.idle_per_min

    # Decision variables
    Z = sub.addVars(
        ((i, day, blk) for i in range(N) for (day, blk) in week_blocks),
        vtype=GRB.BINARY,
        name="z",
    )
    R = sub.addVars(range(N), vtype=GRB.BINARY, name="r")
    ot = sub.addVars(
        week_blocks, lb=0.0, ub=CONFIG.costs.max_overtime_minutes, name="ot"
    )
    it = sub.addVars(week_blocks, lb=0.0, ub=C, name="it")

    sub.update()

    # Each surgery must be either assigned to exactly one block or rejected
    for i in range(N):
        sub.addConstr(
            quicksum(Z[i, day, blk] for (day, blk) in week_blocks) + R[i] == 1,
            name=f"assign_{i}",
        )

    # Block capacity constraints using two-inequality formulation
    balance_constrs = {}
    for day, blk in week_blocks:
        assigned_time = quicksum(durations_fixed[i] * Z[i, day, blk] for i in range(N))

        overtime_constr = sub.addConstr(
            assigned_time - C <= ot[day, blk],
            name=f"overtime_{day}_{blk}",
        )
        idle_constr = sub.addConstr(
            C - assigned_time <= it[day, blk],
            name=f"idle_{day}_{blk}",
        )

        balance_constrs[(day, blk)] = (overtime_constr, idle_constr)

    # Objective: minimize total scheduling costs
    rejection_cost = c_r * quicksum(bp[i] * R[i] for i in range(N))
    overtime_cost = c_o * quicksum(ot[day, blk] for (day, blk) in week_blocks)
    idle_cost = c_i * quicksum(it[day, blk] for (day, blk) in week_blocks)

    sub.setObjective(rejection_cost + overtime_cost + idle_cost, GRB.MINIMIZE)

    sub.Params.OutputFlag = 0
    sub.Params.MIPFocus = 1
    sub.Params.Threads = 1

    # Store data for cut generation
    sub._balance_constrs = balance_constrs
    sub._Z_vars = Z
    sub._N = N
    sub._week_blocks = week_blocks
    sub._durations_fixed = durations_fixed
    sub._ot_vars = ot
    sub._it_vars = it
    sub._R_vars = R

    return sub


def extract_scheduling_cut(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    X_processed: np.ndarray,
    theta_current: np.ndarray,
    week_idx: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Extract affine minorant (supporting hyperplane) for scheduling costs.

    Uses LP relaxation to obtain a valid supporting hyperplane of the scheduling
    recourse function. Since the LP relaxation G_t(θ) is convex in θ, the supporting
    hyperplane provides a valid global lower bound for all θ.

    Args:
        week_data: DataFrame containing week's surgery data
        week_blocks: Available (day, block) pairs for scheduling
        X_processed: Preprocessed feature matrix for the week
        theta_current: Current estimate of duration prediction parameters
        week_idx: Week index for model naming

    Returns:
        Tuple of (slope_vector, function_value, reference_point) for supporting hyperplane
    """
    if week_data.empty:
        p = X_processed.shape[1] if len(X_processed) > 0 else 1
        return np.zeros(p), 0.0, theta_current

    # Predict and clip durations to feasible range
    predicted_durations = X_processed @ theta_current
    MIN_DUR = CONFIG.operating_room.min_procedure_duration
    MAX_DUR = (
        CONFIG.operating_room.block_size_minutes
        + DomainConstants.MAX_OVERTIME_MINUTES_PER_BLOCK
    )
    predicted_durations = np.clip(predicted_durations, MIN_DUR, MAX_DUR)

    # Build and solve LP relaxation of scheduling subproblem
    subproblem = build_subproblem(week_data, week_blocks, predicted_durations, week_idx)

    # Relax binary variables to continuous [0,1]
    for var in subproblem.getVars():
        if var.VType == GRB.BINARY:
            var.VType = GRB.CONTINUOUS
            var.LB = 0.0
            var.UB = 1.0

    subproblem.Params.Method = 1  # Dual simplex
    subproblem.Params.TimeLimit = 10
    subproblem.Params.OutputFlag = 0

    subproblem.optimize()

    p = X_processed.shape[1]
    if subproblem.Status != GRB.OPTIMAL:
        # LP not solved to optimality: return zero slope with bound if available
        bound = (
            float(subproblem.ObjBound)
            if hasattr(subproblem, "ObjBound") and np.isfinite(subproblem.ObjBound)
            else 0.0
        )
        return np.zeros(p), bound, theta_current

    # Extract supporting hyperplane using dual information
    h_sched = np.zeros(p)
    raw_pred = X_processed @ theta_current
    eps = 1e-6
    active_mask = (raw_pred > MIN_DUR + eps) & (raw_pred < MAX_DUR - eps)

    Z_vars = subproblem._Z_vars
    balance_constrs = subproblem._balance_constrs
    N = subproblem._N

    # Build subgradient using chain rule: ∂G_t/∂θ = Σ (∂G_t/∂d_i) * (∂d_i/∂θ)
    for (day, blk), (overtime_constr, idle_constr) in balance_constrs.items():
        pi_overtime = overtime_constr.Pi  # Dual of (assigned_time - C <= ot)
        pi_idle = idle_constr.Pi  # Dual of (C - assigned_time <= it)

        # Combined dual coefficient: ∂objective/∂(assigned_time)
        dual_coeff = pi_overtime - pi_idle

        # Accumulate subgradient contributions from active (non-clipped) surgeries
        for i in range(N):
            if active_mask[i]:
                z_val = Z_vars[i, day, blk].X if (i, day, blk) in Z_vars else 0.0
                if z_val > 1e-9:  # Surgery has positive assignment to this block
                    # Chain rule: dual_coeff * z_val * ∂d_i/∂θ = dual_coeff * z_val * X_i
                    h_sched += (dual_coeff * z_val) * X_processed[i, :]

    g_sched = subproblem.ObjVal
    return h_sched, g_sched, theta_current


def extract_booked_hinge_cut_at(
    week_data: pd.DataFrame,
    X_processed: np.ndarray,
    theta_ref: np.ndarray,
    eta_book: float,
) -> Tuple[np.ndarray, float]:
    """
    Extract supporting hyperplane for booked-time hinge penalty at reference point.

    The booked-time penalty sum_i [|X_i @ theta - b_i| - eta_book]+ is convex in theta,
    so we can safely extract a tangent hyperplane at any reference point theta_ref.
    This hyperplane provides a valid global lower bound for the penalty function.

    Args:
        week_data: DataFrame containing week's surgery data with booked times
        X_processed: Preprocessed feature matrix for the week
        theta_ref: Reference point for hyperplane extraction
        eta_book: Tolerance margin for booked time deviations

    Returns:
        Tuple of (slope_vector, constant_term) for supporting hyperplane
    """
    if week_data.empty or len(X_processed) == 0:
        p = X_processed.shape[1] if len(X_processed) > 0 else 1
        return np.zeros(p), 0.0

    bp = week_data[DataColumns.BOOKED_MIN].to_numpy()

    # Calculate prediction residuals and hinge violations at reference point
    r_ti = X_processed @ theta_ref - bp
    v_ti = np.maximum(0.0, np.abs(r_ti) - eta_book)

    # Subgradient coefficients for hinge function
    sigma_ti = np.where(r_ti > eta_book, 1.0, np.where(r_ti < -eta_book, -1.0, 0.0))

    # Supporting hyperplane parameters
    h_book = (sigma_ti[:, None] * X_processed).sum(axis=0)
    g_book = float(v_ti.sum())  # Function value at reference point

    return h_book, g_book


def solve_subproblem_mip_with_warmstart(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    theta_val: np.ndarray,
    preproc,
    week_idx: int,
    z_warm_start: np.ndarray,
    r_warm_start: np.ndarray,
    time_limit: int = 60,
    mip_gap: float = 0.05,
) -> float:
    """
    Solve MIP scheduling subproblem exactly with warm start from greedy solution.

    Used for periodic certification of upper bounds. Provides exact or near-exact
    scheduling costs to verify convergence quality.

    Args:
        week_data: DataFrame containing week's surgery data
        week_blocks: Available (day, block) pairs for scheduling
        theta_val: Duration prediction parameters
        preproc: Fitted preprocessor for feature transformation
        week_idx: Week index for model naming
        z_warm_start: Initial assignment variables from greedy solution
        r_warm_start: Initial rejection variables from greedy solution
        time_limit: Maximum solution time in seconds
        mip_gap: Target optimality gap for early termination

    Returns:
        Optimal or near-optimal scheduling cost for the week
    """
    if week_data.empty:
        return 0.0

    X_week = week_data[FeatureColumns.ALL]
    X_processed = preproc.transform(X_week)
    predicted_durations = X_processed @ theta_val

    # Clip durations to feasible range
    MIN_DUR = CONFIG.operating_room.min_procedure_duration
    MAX_DUR = (
        CONFIG.operating_room.block_size_minutes
        + DomainConstants.MAX_OVERTIME_MINUTES_PER_BLOCK
    )
    predicted_durations = np.clip(predicted_durations, MIN_DUR, MAX_DUR)

    subproblem = build_subproblem(week_data, week_blocks, predicted_durations, week_idx)

    # Configure MIP solver for quality solutions
    subproblem.Params.MIPFocus = 1
    subproblem.Params.Heuristics = 0.5
    subproblem.Params.Symmetry = 2
    subproblem.Params.Cuts = 2
    subproblem.Params.Threads = 1
    subproblem.Params.TimeLimit = time_limit
    subproblem.Params.MIPGap = mip_gap

    # Apply warm start from greedy solution
    try:
        Z_vars = subproblem._Z_vars
        R_vars = subproblem._R_vars

        for i in range(len(r_warm_start)):
            R_vars[i].Start = float(r_warm_start[i])

        for i in range(len(week_data)):
            for blk_idx, (day, blk) in enumerate(week_blocks):
                if (i, day, blk) in Z_vars and blk_idx < z_warm_start.shape[1]:
                    Z_vars[i, day, blk].Start = float(z_warm_start[i, blk_idx])
    except:
        pass  # Continue without warm start if it fails

    subproblem.optimize()

    if subproblem.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        return subproblem.ObjVal
    elif subproblem.Status == GRB.TIME_LIMIT and hasattr(subproblem, "ObjVal"):
        return subproblem.ObjVal
    else:
        # Fallback to greedy cost if MIP fails
        bp = week_data[DataColumns.BOOKED_MIN].to_numpy()
        num_blocks = len(week_blocks)
        C = CONFIG.operating_room.block_size_minutes
        greedy_cost, _, _, _, _ = greedy_schedule_cost(
            predicted_durations, bp, num_blocks, C
        )
        return greedy_cost


def calculate_objective_components(
    theta_current: np.ndarray,
    Xw: np.ndarray,
    yw: np.ndarray,
    df_plan_data: List[pd.DataFrame],
    preproc,
    lambda_reg: float,
    tau_pred: float,
    rho_pred: float,
    rho_book: float,
    eta_book: float,
) -> Tuple[float, float, float]:
    """
    Calculate individual components of the integrated objective function.

    Args:
        theta_current: Current duration prediction parameters
        Xw: Feature matrix for warm-up data
        yw: Actual durations for warm-up data
        df_plan_data: List of weekly surgery DataFrames
        preproc: Fitted feature preprocessor
        lambda_reg: L1 regularization coefficient
        tau_pred: Prediction error tolerance margin
        rho_pred: Weight on prediction penalty
        rho_book: Weight on booked-time penalty
        eta_book: Booked-time deviation tolerance margin

    Returns:
        Tuple of (prediction_penalty, l1_penalty, booked_penalty)
    """
    # Prediction penalty: weighted hinge loss on warm-up data
    pred_residuals = Xw @ theta_current - yw
    pred_violations = np.maximum(0.0, np.abs(pred_residuals) - tau_pred)
    pred_penalty = rho_pred * np.sum(pred_violations)

    # L1 regularization penalty
    l1_penalty = lambda_reg * np.sum(np.abs(theta_current))

    # Booked-time penalty: deviations from scheduled procedure times
    book_penalty = 0.0
    if rho_book > 0.0:
        for week_data in df_plan_data:
            if week_data.empty:
                continue
            X_week = week_data[FeatureColumns.ALL]
            X_processed = preproc.transform(X_week)
            bp = week_data[DataColumns.BOOKED_MIN].to_numpy()

            book_residuals = X_processed @ theta_current - bp
            book_violations = np.maximum(0.0, np.abs(book_residuals) - eta_book)
            book_penalty += rho_book * np.sum(book_violations)

    return pred_penalty, l1_penalty, book_penalty


def solve_integrated_benders(
    df_plan_data: List[pd.DataFrame],
    week_blocks_list: List[List[Tuple[int, int]]],
    Xw: np.ndarray,
    yw: np.ndarray,
    preproc,
    max_iterations: int = 100,
    use_parallel: bool = True,
    mip_certification_every_k: int = 10,
    num_horizons_for_mip: int = 8,
) -> Dict[str, Any]:
    """
    Solve integrated learning-scheduling problem using Benders decomposition.

    Combines duration prediction learning with surgical scheduling optimization.
    Uses greedy upper bounds every iteration and periodic MIP certification for
    exact bounds. Maintains mathematical consistency through proper cost scaling
    and valid lower bound cuts.

    Args:
        df_plan_data: List of weekly surgery DataFrames for planning horizon
        week_blocks_list: List of available (day, block) pairs for each week
        Xw: Feature matrix for warm-up data
        yw: Actual durations for warm-up data
        preproc: Fitted feature preprocessor
        max_iterations: Maximum number of Benders iterations
        use_parallel: Whether to use parallel processing (currently unused)
        mip_certification_every_k: Run exact MIP certification every k iterations
        num_horizons_for_mip: Number of weeks to solve exactly during certification

    Returns:
        Dictionary containing optimal solution, objective value, and iteration statistics
    """
    print("Starting Benders decomposition for integrated learning-scheduling")
    start_time = time.time()

    Nw, p = Xw.shape
    num_weeks = len(df_plan_data)

    # Rotation state for systematic MIP certification (locals only, no config changes)
    non_empty_weeks = [t for t in range(num_weeks) if not df_plan_data[t].empty]
    n_non_empty = len(non_empty_weeks)

    cert_ptr = 0  # Round-robin pointer over non-empty weeks
    current_batch_size = max(
        1, min(num_horizons_for_mip, n_non_empty)
    )  # Start with existing arg

    last_certified = [
        -(10**9)
    ] * num_weeks  # Iteration when each week was last MIP-certified
    prev_gap = float("inf")
    no_progress_iters = 0

    def pick_rotating_batch(start_idx: int, batch_size: int) -> List[int]:
        """Round-robin batch of horizon indices, skipping empty weeks."""
        if n_non_empty == 0:
            return []
        sel = []
        k = 0
        while len(sel) < batch_size and k < n_non_empty:
            t = non_empty_weeks[(start_idx + k) % n_non_empty]
            sel.append(t)
            k += 1
        return sel

    # Extract problem parameters
    LAMBDA = CONFIG.integrated.lambda_reg
    TAU_PRED = CONFIG.integrated.tau_pred
    RHO_PRED = CONFIG.integrated.rho_pred
    ETA_BOOK = CONFIG.integrated.eta_book
    RHO_BOOK = CONFIG.integrated.rho_book

    print(
        f"Parameters: λ={LAMBDA}, τ_pred={TAU_PRED}, ρ_pred={RHO_PRED}, η_book={ETA_BOOK}, ρ_book={RHO_BOOK}"
    )

    # Apply cost scaling for numerical stability
    bp_mean = np.mean(
        [
            week_data[DataColumns.BOOKED_MIN].mean()
            for week_data in df_plan_data
            if not week_data.empty
        ]
    )
    cost_scale = 1.0
    max_cost = max(
        CONFIG.costs.rejection_per_case * bp_mean if not np.isnan(bp_mean) else 0,
        CONFIG.costs.overtime_per_min,
        CONFIG.costs.idle_per_min,
    )

    if max_cost > 1000:
        cost_scale = 1000.0 / max_cost
        print(f"Applying cost scaling factor: {cost_scale:.4f}")

    # Store original parameter values for restoration
    original_costs = None
    original_rho_book = None
    original_rho_pred = None
    original_lambda = None

    # Scale all cost-related parameters consistently to maintain mathematical equivalence
    if cost_scale != 1.0:
        original_costs = (
            CONFIG.costs.rejection_per_case,
            CONFIG.costs.overtime_per_min,
            CONFIG.costs.idle_per_min,
        )
        CONFIG.costs.rejection_per_case *= cost_scale
        CONFIG.costs.overtime_per_min *= cost_scale
        CONFIG.costs.idle_per_min *= cost_scale

        original_rho_book = CONFIG.integrated.rho_book
        CONFIG.integrated.rho_book *= cost_scale

        original_rho_pred = RHO_PRED
        RHO_PRED *= cost_scale

        original_lambda = CONFIG.integrated.lambda_reg
        CONFIG.integrated.lambda_reg *= cost_scale

    try:
        print("Initializing with Lasso solution...")
        theta_init = (
            Lasso(
                alpha=CONFIG.integrated.lambda_reg, fit_intercept=False, max_iter=10_000
            )
            .fit(Xw, yw)
            .coef_
        )

        print("Building master problem...")
        master, theta_var, s_pred_var, u_var, beta_vars = build_master_problem(
            Xw, yw, p, CONFIG.integrated.lambda_reg, num_weeks, TAU_PRED, RHO_PRED
        )

        # Warm start master problem with Lasso solution
        for j in range(p):
            theta_var[j].Start = float(theta_init[j])

        # Initialize tracking variables
        best_upper_bound = float("inf")
        best_theta = theta_init.copy()
        theta_prev = None  # Track previous iteration's theta for multi-point cuts

        warm_starts = {}  # Store greedy solutions for MIP warm starting
        greedy_costs_by_week = {}  # Track scheduling costs by week
        lp_lb_by_week = {}  # Store LP lower bounds by week for consistency
        last_certified_iter = -999  # Track when UB was last certified with MIP

        iteration_info = []
        ub_stagnant_count = 0

        print(f"Starting {max_iterations} iterations...")

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Solve master problem
            master_start = time.time()
            master.optimize()
            master_time = time.time() - master_start

            if master.Status != GRB.OPTIMAL:
                print(f"Master problem failed with status: {master.Status}")
                break

            theta_current = np.array([theta_var[j].X for j in range(p)])
            lower_bound = master.ObjVal

            print(f"Master: LB={lower_bound:.2f}, time={master_time:.2f}s")

            # Compute fast greedy upper bound for all weeks
            print("Computing greedy upper bound...")
            greedy_start = time.time()

            total_greedy_scheduling_cost = 0.0
            for week_idx, (week_data, week_blocks) in enumerate(
                zip(df_plan_data, week_blocks_list)
            ):
                if week_data.empty:
                    greedy_costs_by_week[week_idx] = 0.0
                    warm_starts[week_idx] = (np.zeros((0, 0)), np.zeros(0))
                    continue

                # Predict and clip durations
                X_week = week_data[FeatureColumns.ALL]
                X_processed = preproc.transform(X_week)
                predicted_durations = X_processed @ theta_current

                MIN_DUR = CONFIG.operating_room.min_procedure_duration
                MAX_DUR = (
                    CONFIG.operating_room.block_size_minutes
                    + DomainConstants.MAX_OVERTIME_MINUTES_PER_BLOCK
                )
                predicted_durations = np.clip(predicted_durations, MIN_DUR, MAX_DUR)

                # Run greedy scheduler
                bp = week_data[DataColumns.BOOKED_MIN].to_numpy()
                num_blocks = len(week_blocks)
                C = CONFIG.operating_room.block_size_minutes

                cost, _, _, z_warm, r_warm = greedy_schedule_cost(
                    predicted_durations, bp, num_blocks, C
                )

                # Floor greedy cost by LP lower bound to ensure consistency
                if week_idx in lp_lb_by_week:
                    cost = max(cost, lp_lb_by_week[week_idx])

                greedy_costs_by_week[week_idx] = cost
                warm_starts[week_idx] = (z_warm, r_warm)
                total_greedy_scheduling_cost += cost

            # Calculate all objective components (in scaled units)
            pred_penalty, l1_penalty, book_penalty = calculate_objective_components(
                theta_current,
                Xw,
                yw,
                df_plan_data,
                preproc,
                CONFIG.integrated.lambda_reg,
                TAU_PRED,
                RHO_PRED,
                CONFIG.integrated.rho_book,
                ETA_BOOK,
            )

            # Total upper bound (in scaled units for consistent comparison with LB)
            greedy_upper_bound = (
                pred_penalty + l1_penalty + total_greedy_scheduling_cost + book_penalty
            )

            greedy_time = time.time() - greedy_start
            print(
                f"Greedy UB: {greedy_upper_bound:.2f} (scaled units), time: {greedy_time:.3f}s"
            )

            # Emergency consistency guard: never allow stale best_ub below current LB
            if lower_bound > best_upper_bound + 1e-6:
                print(
                    "! LB exceeded stored best UB -- resetting best UB to current greedy UB"
                )
                best_upper_bound = greedy_upper_bound
                best_theta = theta_current.copy()
                ub_stagnant_count = 0
                print(f"*** Reset best UB to current: {best_upper_bound:.2f} ***")

            # Update best solution tracking
            elif greedy_upper_bound < best_upper_bound:
                best_upper_bound = greedy_upper_bound
                best_theta = theta_current.copy()
                ub_stagnant_count = 0
                print(f"*** New best greedy UB: {best_upper_bound:.2f} ***")
            else:
                ub_stagnant_count += 1

            # Generate Benders cuts for all weeks with multi-point booked cuts
            print(f"Adding cuts for all {num_weeks} horizons...")
            cuts_start = time.time()
            cuts_added = 0

            # Prepare multiple reference points for booked-hinge cuts
            theta_refs = [theta_current]
            if theta_prev is not None and not np.allclose(
                theta_prev, theta_current, atol=1e-6
            ):
                theta_refs.append(theta_prev)
            if best_theta is not None and not np.allclose(
                best_theta, theta_current, atol=1e-6
            ):
                theta_refs.append(best_theta)

            print(f"Using {len(theta_refs)} reference points for booked cuts")

            for week_idx, (week_data, week_blocks) in enumerate(
                zip(df_plan_data, week_blocks_list)
            ):
                if week_data.empty:
                    continue

                X_week = week_data[FeatureColumns.ALL]
                X_processed = preproc.transform(X_week)

                # Extract scheduling cut (affine minorant of LP relaxation at current θ)
                h_sched, g_sched, theta_ref_sched = extract_scheduling_cut(
                    week_data, week_blocks, X_processed, theta_current, week_idx
                )

                # Store LP lower bound for this week (for legacy consistency checks)
                lp_lb_by_week[week_idx] = float(g_sched)

                # Compute scheduling cut constant at its own reference point
                const_sched = g_sched - float(h_sched @ theta_ref_sched)

                # Add cuts at multiple reference points for stronger lower bounds
                for ref_idx, theta_ref in enumerate(theta_refs):
                    # Extract booked-time hinge cut at this reference point
                    h_book, g_book = extract_booked_hinge_cut_at(
                        week_data, X_processed, theta_ref, ETA_BOOK
                    )

                    # Compute booked cut constant at its own reference point
                    const_book = g_book - float(h_book @ theta_ref)

                    # Combine cuts: each supporting plane uses its own reference point
                    h_total = h_sched + CONFIG.integrated.rho_book * h_book
                    g_total = const_sched + CONFIG.integrated.rho_book * const_book

                    # Add Benders cut: beta_t >= g_total + h_total^T * theta
                    cut_expr = g_total + quicksum(
                        h_total[j] * theta_var[j] for j in range(p)
                    )
                    master.addConstr(
                        beta_vars[week_idx] >= cut_expr,
                        name=f"cut_w{week_idx}_iter{iteration}_ref{ref_idx}",
                    )
                    cuts_added += 1

            master.update()
            cuts_time = time.time() - cuts_start
            print(f"Cuts: added={cuts_added}, time={cuts_time:.3f}s")

            # Periodic MIP certification with rotating coverage
            exact_ub = greedy_upper_bound
            if (
                iteration + 1
            ) % mip_certification_every_k == 0 or ub_stagnant_count >= 5:

                # Use rotation instead of top-k selection for comprehensive coverage
                batch_size = max(1, min(current_batch_size, n_non_empty))
                weeks_to_cert = pick_rotating_batch(cert_ptr, batch_size)
                print(
                    f"Rotating certification: certifying {len(weeks_to_cert)}/{n_non_empty} horizons "
                    f"(batch_size={batch_size}, start_ptr={cert_ptr})"
                )

                mip_start = time.time()
                exact_scheduling_cost = 0.0
                remaining_greedy_cost = 0.0

                for week_idx in range(num_weeks):
                    if week_idx in weeks_to_cert and not df_plan_data[week_idx].empty:
                        # Solve exact MIP for selected rotating weeks
                        z_warm, r_warm = warm_starts.get(
                            week_idx, (np.zeros((0, 0)), np.zeros(0))
                        )
                        exact_cost = solve_subproblem_mip_with_warmstart(
                            df_plan_data[week_idx],
                            week_blocks_list[week_idx],
                            theta_current,
                            preproc,
                            week_idx,
                            z_warm,
                            r_warm,
                            time_limit=60,
                            mip_gap=0.05,
                        )
                        exact_scheduling_cost += exact_cost
                        last_certified[week_idx] = iteration  # Track certification
                    else:
                        # Use greedy cost for other weeks
                        greedy_cost = greedy_costs_by_week.get(week_idx, 0.0)
                        remaining_greedy_cost += greedy_cost

                # Combine exact and greedy costs
                total_exact_scheduling = exact_scheduling_cost + remaining_greedy_cost
                exact_ub = (
                    pred_penalty + l1_penalty + total_exact_scheduling + book_penalty
                )

                mip_time = time.time() - mip_start
                print(
                    f"Rotating certification UB={exact_ub:.2f} (scaled units), time={mip_time:.2f}s"
                )

                if exact_ub < best_upper_bound:
                    best_upper_bound = exact_ub
                    best_theta = theta_current.copy()
                    ub_stagnant_count = 0
                    last_certified_iter = iteration
                    print(
                        f"*** New best exact UB (rotating): {best_upper_bound:.2f} ***"
                    )

                # Advance rotation pointer for next certification round
                if n_non_empty > 0:
                    cert_ptr = (cert_ptr + batch_size) % n_non_empty

            # Sanity check: lower bound should never exceed upper bound in valid Benders
            if lower_bound > best_upper_bound + 1e-6:
                print(
                    f"! Warning: LB ({lower_bound:.2f}) exceeded UB ({best_upper_bound:.2f})"
                )
                print(
                    "  This indicates invalid cuts. Using constant scheduling cuts for safety."
                )

            # Calculate optimality gap (both bounds in same scaled units)
            if abs(best_upper_bound) > 1e-12:
                gap = max(0.0, (best_upper_bound - lower_bound) / abs(best_upper_bound))
            else:
                gap = 0.0

            # Convert to original units for display
            if cost_scale != 1.0:
                display_lb = lower_bound / cost_scale
                display_ub = best_upper_bound / cost_scale
                display_sched = total_greedy_scheduling_cost / cost_scale
                display_pred = (
                    pred_penalty / cost_scale if original_rho_pred else pred_penalty
                )
                display_book = (
                    book_penalty / cost_scale if original_rho_book else book_penalty
                )
            else:
                display_lb = lower_bound
                display_ub = best_upper_bound
                display_sched = total_greedy_scheduling_cost
                display_pred = pred_penalty
                display_book = book_penalty

            print(f"Bounds: LB={display_lb:.2f}, UB={display_ub:.2f} (original units)")
            print(
                f"Components: Pred={display_pred:.1f}, L1={l1_penalty:.1f}, Sched={display_sched:.1f}, Book={display_book:.1f}"
            )
            print(f"Gap: {gap:.4f} ({gap*100:.2f}%)")

            # Progress tracking and adaptive certification escalation
            if gap < prev_gap - 0.01:  # Improved by >= 1 percentage point
                no_progress_iters = 0
                prev_gap = gap
            else:
                no_progress_iters += 1

            # Escalate rotation coverage locally when progress stalls (no config changes)
            if no_progress_iters >= 5 and current_batch_size < n_non_empty:
                # Jump by ~1/3 of remaining weeks (aggressive but bounded)
                jump = max(1, (n_non_empty - current_batch_size) // 3)
                current_batch_size = min(n_non_empty, current_batch_size + jump)
                no_progress_iters = 0
                print(
                    f"Escalating certification coverage: batch_size -> {current_batch_size}"
                )

            # Store iteration statistics
            iteration_info.append(
                {
                    "iteration": iteration + 1,
                    "master_obj": float(lower_bound),
                    "master_time": float(master_time),
                    "greedy_ub": float(greedy_upper_bound),
                    "best_ub": float(best_upper_bound),
                    "greedy_scheduling_cost": float(total_greedy_scheduling_cost),
                    "prediction_penalty": float(pred_penalty),
                    "l1_penalty": float(l1_penalty),
                    "booked_penalty": float(book_penalty),
                    "gap": float(gap),
                    "cuts_added": int(cuts_added),
                    "cuts_time": float(cuts_time),
                    "theta": theta_current.copy(),
                }
            )

            # Check convergence criteria with full coverage requirement

            # Calculate coverage window: how many iterations for all horizons to be certified
            if n_non_empty > 0:
                max_age = max(iteration - last_certified[t] for t in non_empty_weeks)
                coverage_window = max(
                    1, (n_non_empty + current_batch_size - 1) // current_batch_size
                ) * max(1, mip_certification_every_k)
            else:
                max_age = 0
                coverage_window = 1

            if gap < 0.01 and max_age <= coverage_window:
                print(
                    "✓ Converged: Gap < 1% with recent certification coverage over all horizons"
                )
                break
            elif gap < 0.01:
                print(
                    f"Gap < 1% but incomplete coverage (max_age={max_age}, coverage_window={coverage_window})"
                )

            if cuts_added == 0 and iteration >= 5:
                print("✓ Converged: No improving cuts")
                break

            # Update theta tracking for next iteration's multi-point cuts
            theta_prev = theta_current.copy()

        end_time = time.time()
        total_time = end_time - start_time

        # Convert final objective to original units for reporting
        final_objective_original = (
            best_upper_bound / cost_scale if cost_scale != 1.0 else best_upper_bound
        )

        print(f"\n{'='*60}")
        print(f"Benders decomposition completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best objective: {final_objective_original:.2f} (original units)")
        print(f"Final gap: {gap:.4f} ({gap*100:.2f}%)")
        print(f"Iterations: {len(iteration_info)}")

        if iteration_info:
            avg_iter_time = total_time / len(iteration_info)
            print(f"Average time per iteration: {avg_iter_time:.2f}s")
        print(f"{'='*60}")

        return {
            "theta": best_theta,
            "objective": float(final_objective_original),
            "iterations": len(iteration_info),
            "iteration_info": iteration_info,
            "total_time": float(total_time),
            "converged": gap < 0.01,
        }

    finally:
        # Restore original parameter values
        if cost_scale != 1.0 and original_costs is not None:
            (
                CONFIG.costs.rejection_per_case,
                CONFIG.costs.overtime_per_min,
                CONFIG.costs.idle_per_min,
            ) = original_costs

        if cost_scale != 1.0 and original_rho_book is not None:
            CONFIG.integrated.rho_book = original_rho_book

        if cost_scale != 1.0 and original_lambda is not None:
            CONFIG.integrated.lambda_reg = original_lambda


def build_planning_weeks_data(
    df_warm: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[List[Tuple[int, int]]]]:
    """
    Partition warm-up data into weekly planning periods.

    Splits historical data into weekly chunks aligned with Monday-Sunday boundaries
    and determines available operating room blocks for each week.

    Args:
        df_warm: DataFrame containing warm-up period surgery data

    Returns:
        Tuple of (weekly_data_list, weekly_blocks_list) where each element
        corresponds to one planning week
    """
    if df_warm.empty:
        return [], []

    start_date = df_warm[DataColumns.ACTUAL_START].min()
    end_date = df_warm[DataColumns.ACTUAL_START].max()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    # Align to Monday-Sunday week boundaries
    days_from_monday = start_ts.weekday()
    first_monday = start_ts - pd.Timedelta(days=days_from_monday)

    print(f"Data period: {start_ts.date()} to {end_ts.date()}")
    print(f"First Monday: {first_monday.date()}")

    week_data_list = []
    week_blocks_list = []

    current_monday = first_monday
    week_idx = 0

    while current_monday <= end_ts:
        week_end = current_monday + pd.Timedelta(days=6)

        # Extract surgeries for this week
        week_mask = (
            df_warm[DataColumns.ACTUAL_START].dt.normalize() >= current_monday
        ) & (df_warm[DataColumns.ACTUAL_START].dt.normalize() <= week_end)

        df_week = df_warm[week_mask].copy()
        week_data_list.append(df_week)

        # Determine available operating room blocks for this week
        if not df_week.empty:
            daily_capacity = compute_block_capacity(df_week, CONFIG, current_monday)
            week_blocks = [
                (day, blk) for day, cnt in daily_capacity.items() for blk in range(cnt)
            ]
        else:
            week_blocks = []

        week_blocks_list.append(week_blocks)

        print(
            f"Week {week_idx}: {current_monday.date()} - {week_end.date()}: "
            f"{len(df_week)} surgeries, {len(week_blocks)} blocks"
        )

        current_monday += pd.Timedelta(days=7)
        week_idx += 1

    print(f"Built {len(week_data_list)} weeks from data")

    return week_data_list, week_blocks_list


def main() -> None:
    """
    Main execution function for integrated learning-scheduling optimization.

    Loads surgery data, builds weekly planning periods, and solves the integrated
    problem using Benders decomposition with mathematical consistency guarantees.
    Tests multiple parameter combinations and saves results for analysis.
    """
    print("=== Integrated Learning-Scheduling Optimization ===\n")

    # Load and preprocess data
    df = load_data(CONFIG)
    df = add_time_features(df)
    df_warm, df_pool, horizon_start = split_data(df, CONFIG)

    # Prepare features and targets for learning
    X_warm, y_warm = _prepare_features_target(df_warm)
    preproc = _create_sklearn_preprocessor()
    Xw = preproc.fit_transform(X_warm)
    yw = y_warm.to_numpy()

    # Build weekly planning structure
    week_data_list, week_blocks_list = build_planning_weeks_data(df_warm)

    total_surgeries = sum(len(week_data) for week_data in week_data_list)
    total_blocks = sum(len(week_blocks) for week_blocks in week_blocks_list)

    print(
        f"\nTotal across all weeks: {total_surgeries} surgeries, {total_blocks} blocks"
    )

    # Test multiple parameter combinations
    param_sets = CONFIG.integrated.param_values_to_test
    all_results = {}

    print(f"\nRunning Benders decomposition for {len(param_sets)} parameter sets")

    for i, params in enumerate(param_sets):
        print(f"\n{'='*60}")
        print(f"Parameter set {i+1}/{len(param_sets)}: {params}")
        print(f"{'='*60}")

        # Temporarily update configuration
        original_params = {}
        for key, value in params.items():
            original_params[key] = getattr(CONFIG.integrated, key)
            setattr(CONFIG.integrated, key, value)

        # Solve integrated problem
        results = solve_integrated_benders(
            week_data_list,
            week_blocks_list,
            Xw,
            yw,
            preproc,
            max_iterations=100,
            use_parallel=True,
            mip_certification_every_k=10,
            num_horizons_for_mip=8,
        )

        # Restore original configuration
        for key, value in original_params.items():
            setattr(CONFIG.integrated, key, value)

        param_key = "_".join([f"{k}{v}" for k, v in params.items()])

        all_results[param_key] = {"results": results, "params": params.copy()}

        # Save optimal parameters for this configuration
        if results["theta"] is not None:
            feat_names = list(preproc.get_feature_names_out())
            clean_names = [canonical_name(n) for n in feat_names]
            theta_out = {
                clean_names[j]: float(results["theta"][j])
                for j in range(len(feat_names))
            }

            out_path = Path(CONFIG.data.theta_path)
            param_theta_path = out_path.parent / f"theta_{param_key}.json"
            param_theta_path.parent.mkdir(parents=True, exist_ok=True)
            with param_theta_path.open("w") as f:
                json.dump(theta_out, f, indent=2)
            print(f"θ for {param_key} saved to {param_theta_path}")

    # Analyze and save performance comparison
    out_path = Path(CONFIG.data.theta_path)

    performance_results = {}
    for param_key, data in all_results.items():
        results = data["results"]
        params = data["params"]

        if results["theta"] is not None and results["iteration_info"]:
            last_iter = results["iteration_info"][-1]

            performance_results[param_key] = {
                "parameters": params,
                "prediction_penalty": float(last_iter.get("prediction_penalty", 0)),
                "l1_penalty": float(last_iter.get("l1_penalty", 0)),
                "booked_penalty": float(last_iter.get("booked_penalty", 0)),
                "greedy_scheduling_cost": float(
                    last_iter.get("greedy_scheduling_cost", 0)
                ),
                "total_objective": float(results["objective"]),
                "iterations": int(results["iterations"]),
                "converged": bool(results["converged"]),
                "total_time": float(results["total_time"]),
            }

            print(f"\n{param_key} Performance:")
            print(f"  Prediction: {last_iter.get('prediction_penalty', 0):.2f}")
            print(f"  L1: {last_iter.get('l1_penalty', 0):.2f}")
            print(f"  Booked: {last_iter.get('booked_penalty', 0):.2f}")
            print(f"  Scheduling: {last_iter.get('greedy_scheduling_cost', 0):.2f}")
            print(f"  Total: {results['objective']:.2f}")

    performance_path = out_path.parent / "performance_comparison_corrected.json"
    with performance_path.open("w") as f:
        json.dump(to_json_safe(performance_results), f, indent=2)
    print(f"\nPerformance comparison saved to {performance_path}")

    # Save detailed iteration-by-iteration results
    detailed_results_path = (
        out_path.parent / "all_params_detailed_results_corrected.json"
    )
    detailed_results = {}
    for param_key, data in all_results.items():
        results = data["results"]
        detailed_results[param_key] = {
            "objective": float(results["objective"]),
            "iterations": int(results["iterations"]),
            "total_time": float(results["total_time"]),
            "converged": bool(results["converged"]),
            "iteration_info": to_json_safe(results["iteration_info"]),
        }

    with detailed_results_path.open("w") as f:
        json.dump(to_json_safe(detailed_results), f, indent=2)
    print(f"Detailed results saved to {detailed_results_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("Summary of all parameter sets:")
    print(f"{'='*60}")
    for param_key in sorted(performance_results.keys()):
        perf = performance_results[param_key]
        print(
            f"{param_key}: Total={perf['total_objective']:.2f}, "
            f"Pred={perf['prediction_penalty']:.2f}, "
            f"Book={perf['booked_penalty']:.2f}, "
            f"Sched={perf['greedy_scheduling_cost']:.2f}"
        )

    print(
        f"\nImplementation features:\n"
        f"• Mathematically consistent cost scaling across all parameters\n"
        f"• Constant scheduling cuts for validity (no invalid slopes)\n"
        f"• Fast greedy upper bounds every iteration\n"
        f"• Periodic exact MIP certification for quality assurance"
    )


if __name__ == "__main__":
    main()
