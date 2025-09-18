from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

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


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def calculate_schedule_cost(
    predicted_durations: np.ndarray,
    booked_minutes: np.ndarray,
    num_blocks: int,
    capacity_per_block: float,
    rejection_cost: float,
    overtime_cost: float,
    idle_cost: float,
    enable_optimization: bool = True,
    max_optimization_time: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate scheduling cost using a greedy heuristic with optional refinement."""
    num_cases = len(predicted_durations)
    if num_cases == 0:
        return (
            0.0,
            np.zeros(num_blocks),
            np.zeros(0, dtype=bool),
            np.zeros((0, num_blocks)),
            np.zeros(0),
        )

    capacity = capacity_per_block
    block_loads = np.zeros(num_blocks)
    reject_mask = np.zeros(num_cases, dtype=bool)
    z_assignment = np.zeros((num_cases, num_blocks), dtype=int)

    sorted_indices = np.argsort(predicted_durations)[::-1]

    for idx in sorted_indices:
        duration = predicted_durations[idx]
        booked = booked_minutes[idx]

        best_block = -1
        min_delta_cost = float("inf")
        best_leftover = float("inf")

        for block in range(num_blocks):
            current_load = block_loads[block]
            new_load = current_load + duration

            old_overtime = max(0.0, current_load - capacity)
            new_overtime = max(0.0, new_load - capacity)
            old_idle = max(0.0, capacity - current_load)
            new_idle = max(0.0, capacity - new_load)

            delta_cost = overtime_cost * (new_overtime - old_overtime) + idle_cost * (
                new_idle - old_idle
            )

            leftover = capacity - new_load if new_load <= capacity else -1.0

            if delta_cost < min_delta_cost or (
                delta_cost == min_delta_cost and 0 <= leftover < best_leftover
            ):
                min_delta_cost = delta_cost
                best_block = block
                best_leftover = leftover

        reject_penalty = rejection_cost * booked

        if min_delta_cost <= reject_penalty:
            block_loads[best_block] += duration
            z_assignment[idx, best_block] = 1
        else:
            reject_mask[idx] = True

    if enable_optimization and max_optimization_time > 0:
        start_time = time.time()
        improved = True

        while improved and (time.time() - start_time) < max_optimization_time:
            improved = False
            rejected_indices = np.where(reject_mask)[0]
            for idx in rejected_indices:
                duration = predicted_durations[idx]
                booked = booked_minutes[idx]
                reject_penalty = rejection_cost * booked

                best_improvement = 0.0
                best_target_block = -1
                best_eject_idx = -1

                for block in range(num_blocks):
                    current_load = block_loads[block]
                    new_load_with_insertion = current_load + duration

                    old_overtime = max(0.0, current_load - capacity)
                    new_overtime = max(0.0, new_load_with_insertion - capacity)
                    old_idle = max(0.0, capacity - current_load)
                    new_idle = max(0.0, capacity - new_load_with_insertion)

                    insertion_cost = overtime_cost * (new_overtime - old_overtime) + idle_cost * (
                        new_idle - old_idle
                    )

                    direct_improvement = reject_penalty - insertion_cost
                    if direct_improvement > best_improvement:
                        best_improvement = direct_improvement
                        best_target_block = block
                        best_eject_idx = -1

                    assigned_in_block = np.where(z_assignment[:, block] == 1)[0]
                    for eject_idx in assigned_in_block:
                        duration_eject = predicted_durations[eject_idx]
                        booked_eject = booked_minutes[eject_idx]

                        new_load_after_swap = current_load - duration_eject + duration
                        swap_overtime = max(0.0, new_load_after_swap - capacity)
                        swap_idle = max(0.0, capacity - new_load_after_swap)

                        swap_cost = overtime_cost * swap_overtime + idle_cost * swap_idle
                        original_block_cost = overtime_cost * old_overtime + idle_cost * old_idle
                        eject_penalty = rejection_cost * booked_eject

                        swap_improvement = (reject_penalty + original_block_cost) - (
                            eject_penalty + swap_cost
                        )

                        if swap_improvement > best_improvement:
                            best_improvement = swap_improvement
                            best_target_block = block
                            best_eject_idx = eject_idx

                if best_improvement > 1e-6:
                    if best_eject_idx == -1:
                        block_loads[best_target_block] += duration
                        z_assignment[idx, best_target_block] = 1
                        reject_mask[idx] = False
                    else:
                        duration_eject = predicted_durations[best_eject_idx]
                        block_loads[best_target_block] = (
                            block_loads[best_target_block] - duration_eject + duration
                        )
                        z_assignment[best_eject_idx, best_target_block] = 0
                        z_assignment[idx, best_target_block] = 1
                        reject_mask[idx] = False
                        reject_mask[best_eject_idx] = True

                    improved = True
                    break

    overtime_total = overtime_cost * np.sum(np.maximum(0.0, block_loads - capacity))
    idle_total = idle_cost * np.sum(np.maximum(0.0, capacity - block_loads))
    rejection_total = rejection_cost * np.sum(booked_minutes[reject_mask])
    total_cost = overtime_total + idle_total + rejection_total

    warm_start_rejects = reject_mask.astype(int)

    return total_cost, block_loads, reject_mask, z_assignment, warm_start_rejects


def build_optimization_model(
    Xw: np.ndarray,
    yw: np.ndarray,
    p: int,
    lambda_reg: float,
    num_weeks: int,
    tau_pred: float,
    rho_pred: float,
    theta_init: np.ndarray,
) -> Tuple[
    gp.Model,
    gp.MVar,
    gp.MVar,
    gp.MVar,
    List[gp.Var],
    gp.MVar,
    gp.MVar,
    gp.Constr,
    gp.LinExpr,
]:
    """Build master problem for integrated optimization with trust-region stabilization."""
    master = gp.Model("IntegratedOptimizationMaster")

    # Original variables
    theta = master.addMVar(p, lb=-GRB.INFINITY, name="theta")
    s_pred = master.addMVar(len(yw), lb=0.0, name="s_pred")
    u = master.addMVar(p, lb=0.0, name="u")
    beta = master.addVars(num_weeks, lb=0.0, name="beta")

    # Trust region variables for L1 proximity to the current center
    trp = master.addMVar(p, lb=0.0, name="trp")  # positive part
    trn = master.addMVar(p, lb=0.0, name="trn")  # negative part

    master.update()

    # Original constraints
    # Prediction error hinge constraints
    for t in range(len(yw)):
        master.addConstr(
            s_pred[t] >= Xw[t, :] @ theta - yw[t] - tau_pred, name=f"pred_hinge_pos_{t}"
        )
        master.addConstr(
            s_pred[t] >= -Xw[t, :] @ theta + yw[t] - tau_pred,
            name=f"pred_hinge_neg_{t}",
        )

    # L1 regularization constraints
    master.addConstr(u >= theta, name="l1_pos")
    master.addConstr(u >= -theta, name="l1_neg")

    # Trust region constraints keep theta close to the provided center point
    tr_pos_constrs = []
    tr_neg_constrs = []
    for j in range(p):
        tr_pos_constrs.append(
            master.addConstr(theta[j] - trp[j] <= theta_init[j], name=f"tr_pos_{j}")
        )
        tr_neg_constrs.append(
            master.addConstr(-theta[j] - trn[j] <= -theta_init[j], name=f"tr_neg_{j}")
        )

    # Original objective expression (will become level constraint)
    orig_obj = (
        rho_pred * quicksum(s_pred)
        + lambda_reg * quicksum(u)
        + quicksum(beta[t] for t in range(num_weeks))
    )

    # Level constraint (RHS will be updated each iteration)
    level_constr = master.addConstr(orig_obj <= GRB.INFINITY, name="level")

    # New objective: minimize L1 distance from the center point
    master.setObjective(quicksum(trp) + quicksum(trn), GRB.MINIMIZE)

    master.Params.OutputFlag = 0
    master.Params.Method = 1  # Standardized to dual simplex

    # Store trust region constraints for easy RHS updates
    master._tr_pos_constrs = tr_pos_constrs
    master._tr_neg_constrs = tr_neg_constrs

    return master, theta, s_pred, u, beta, trp, trn, level_constr, orig_obj


def build_subproblem(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    durations_fixed: np.ndarray,
    week_idx: int,
    C: float,
    c_r: float,
    c_o: float,
    c_i: float,
    max_overtime: float,
) -> gp.Model:
    """Build weekly scheduling subproblem for given predicted durations.

    This is the canonical version - only one definition to avoid conflicts.
    """
    if week_data.empty:
        dummy = gp.Model(f"DummyWeek_{week_idx}")
        dummy.setObjective(0, GRB.MINIMIZE)
        return dummy

    sub = gp.Model(f"WeeklyScheduling_{week_idx}")

    N = len(week_data)
    bp = week_data[DataColumns.BOOKED_MIN].to_numpy()

    # Decision variables
    Z = sub.addVars(
        ((i, day, blk) for i in range(N) for (day, blk) in week_blocks),
        vtype=GRB.BINARY,
        name="z",
    )
    R = sub.addVars(range(N), vtype=GRB.BINARY, name="r")
    ot = sub.addVars(week_blocks, lb=0.0, ub=max_overtime, name="ot")
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
    sub.Params.Method = 1  # Standardized to dual simplex

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


def generate_scheduling_cut(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    X_processed: np.ndarray,
    theta_ref: np.ndarray,
    week_idx: int,
    C: float,
    c_r: float,
    c_o: float,
    c_i: float,
    max_overtime: float,
    min_dur: float,
    max_dur: float,
    use_affine_cuts: bool = False,
) -> Tuple[np.ndarray, float]:
    """Generate a supporting hyperplane for scheduling costs at the reference point."""
    if week_data.empty:
        p = X_processed.shape[1] if len(X_processed) > 0 else 1
        return np.zeros(p), 0.0

    # Predict and clip durations to feasible range
    predicted_durations = X_processed @ theta_ref
    predicted_durations = np.clip(predicted_durations, min_dur, max_dur)

    # Build and solve LP relaxation of scheduling subproblem
    subproblem = build_subproblem(
        week_data,
        week_blocks,
        predicted_durations,
        week_idx,
        C,
        c_r,
        c_o,
        c_i,
        max_overtime,
    )

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
        # LP not solved to optimality - return safe constant cut
        bound = (
            float(subproblem.ObjBound)
            if hasattr(subproblem, "ObjBound") and np.isfinite(subproblem.ObjBound)
            else 0.0
        )
        return np.zeros(p), bound

    # Function value at reference point
    g_sched_ref = subproblem.ObjVal

    if not use_affine_cuts:
        # Safe constant cut: slope = 0, standard intercept = f(θ_ref)
        return np.zeros(p), g_sched_ref

    # Affine cut using LP duals (use with caution)
    h_sched = np.zeros(p)
    raw_pred = X_processed @ theta_ref
    eps = 1e-6
    active_mask = (raw_pred > min_dur + eps) & (raw_pred < max_dur - eps)

    Z_vars = subproblem._Z_vars
    balance_constrs = subproblem._balance_constrs
    N = subproblem._N

    # Build subgradient using chain rule
    for (day, blk), (overtime_constr, idle_constr) in balance_constrs.items():
        pi_overtime = overtime_constr.Pi
        pi_idle = idle_constr.Pi
        dual_coeff = pi_overtime - pi_idle

        for i in range(N):
            if active_mask[i]:
                z_val = Z_vars[i, day, blk].X if (i, day, blk) in Z_vars else 0.0
                if z_val > 1e-9:
                    h_sched += (dual_coeff * z_val) * X_processed[i, :]

    # Standard supporting hyperplane: β ≥ b + h^T θ where b = f(θ_ref) - h^T θ_ref
    b_standard = float(g_sched_ref - h_sched @ theta_ref)

    return h_sched, b_standard


def extract_booked_hinge_cut_at(
    week_data: pd.DataFrame,
    X_processed: np.ndarray,
    theta_ref: np.ndarray,
    eta_book: float,
) -> Tuple[np.ndarray, float]:
    """
    Extract valid supporting hyperplane for booked-time hinge penalty at theta_ref.

    Returns:
        Tuple of (slope_vector, standard_intercept)
        Standard intercept b where cut is: β ≥ b + h^T θ
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

    # Function value at reference point
    g_book_ref = float(v_ti.sum())

    # Standard supporting hyperplane: β ≥ b + h^T θ where b = f(θ_ref) - h^T θ_ref
    b_standard = float(g_book_ref - h_book @ theta_ref)

    return h_book, b_standard


def solve_subproblem_mip_with_warmstart(
    week_data: pd.DataFrame,
    week_blocks: List[Tuple[int, int]],
    theta_val: np.ndarray,
    preproc,
    week_idx: int,
    z_warm_start: np.ndarray,
    r_warm_start: np.ndarray,
    C: float,
    c_r: float,
    c_o: float,
    c_i: float,
    max_overtime: float,
    min_dur: float,
    max_dur: float,
    time_limit: int = 60,
    mip_gap: float = 0.05,
) -> float:
    """Solve MIP scheduling subproblem exactly with warm start.

    CANONICAL IMPLEMENTATION - unified interface.
    """
    if week_data.empty:
        return 0.0

    X_week = week_data[FeatureColumns.ALL]
    X_processed = preproc.transform(X_week)
    predicted_durations = X_processed @ theta_val

    # Clip durations to feasible range
    predicted_durations = np.clip(predicted_durations, min_dur, max_dur)

    subproblem = build_subproblem(
        week_data,
        week_blocks,
        predicted_durations,
        week_idx,
        C,
        c_r,
        c_o,
        c_i,
        max_overtime,
    )

    # Configure MIP solver
    subproblem.Params.MIPFocus = 1
    subproblem.Params.Heuristics = 0.5
    subproblem.Params.Symmetry = 2
    subproblem.Params.Cuts = 2
    subproblem.Params.Threads = 1
    subproblem.Params.TimeLimit = time_limit
    subproblem.Params.MIPGap = mip_gap

    # Apply warm start
    try:
        Z_vars = subproblem._Z_vars
        R_vars = subproblem._R_vars

        for i in range(len(r_warm_start)):
            if i in R_vars:
                R_vars[i].Start = float(r_warm_start[i])

        for i in range(len(week_data)):
            for blk_idx, (day, blk) in enumerate(week_blocks):
                if (i, day, blk) in Z_vars and blk_idx < z_warm_start.shape[1]:
                    Z_vars[i, day, blk].Start = float(z_warm_start[i, blk_idx])
    except:
        pass

    subproblem.optimize()

    if subproblem.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        return subproblem.ObjVal
    elif subproblem.Status == GRB.TIME_LIMIT and hasattr(subproblem, "ObjVal"):
        return subproblem.ObjVal
    else:
        # Fallback to greedy cost
        bp = week_data[DataColumns.BOOKED_MIN].to_numpy()
        num_blocks = len(week_blocks)
        greedy_cost, _, _, _, _ = calculate_schedule_cost(
            predicted_durations,
            bp,
            num_blocks,
            C,
            c_r,
            c_o,
            c_i,
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
    """Calculate individual components of the integrated objective function."""
    # Prediction penalty
    pred_residuals = Xw @ theta_current - yw
    pred_violations = np.maximum(0.0, np.abs(pred_residuals) - tau_pred)
    pred_penalty = rho_pred * np.sum(pred_violations)

    # L1 regularization penalty
    l1_penalty = lambda_reg * np.sum(np.abs(theta_current))

    # Booked-time penalty
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


def create_cut_signature(h_total: np.ndarray, b_total: float) -> Tuple:
    """Create a signature for cut deduplication."""
    return (
        tuple(np.round(h_total, 8)),
        round(float(b_total), 6),
    )


def manage_cut_pool(
    cut_pool: Dict[int, Set[Tuple]], max_cuts_per_week: int = 5000
) -> None:
    """Manage cut pool size to prevent memory bloat."""
    for week_idx in cut_pool:
        if len(cut_pool[week_idx]) > max_cuts_per_week:
            # Remove oldest signatures (simple FIFO approximation)
            cut_list = list(cut_pool[week_idx])
            cut_pool[week_idx] = set(cut_list[-max_cuts_per_week // 2 :])


def solve_integrated_optimization(
    df_plan_data: List[pd.DataFrame],
    week_blocks_list: List[List[Tuple[int, int]]],
    Xw: np.ndarray,
    yw: np.ndarray,
    preproc,
    max_iterations: int = 100,
    use_parallel: bool = True,
    mip_certification_every_k: int = 10,
    num_horizons_for_mip: int = 8,
    use_affine_sched_cuts: bool = False,
) -> Dict[str, Any]:
    """Run the integrated prediction and scheduling optimization loop."""
    logger.info("Starting integrated optimization run")
    start_time = time.time()

    try:
        Nw, p = Xw.shape
        num_weeks = len(df_plan_data)

        # Initialize stabilization parameters
        non_empty_weeks = [t for t in range(num_weeks) if not df_plan_data[t].empty]
        n_non_empty = len(non_empty_weeks)

        # Age tracking for certification
        last_certified = [-1000] * num_weeks

        # Cut management with proper deduplication
        cut_pool: Dict[int, Set[Tuple]] = defaultdict(
            set
        )  # week_idx -> set of cut signatures

        # Stabilization parameters
        level_factor = 0.7
        update_rate = 0.3

        # Extract and scale problem parameters (no CONFIG mutation)
        LAMBDA_ORIG = CONFIG.integrated.lambda_reg
        TAU_PRED_ORIG = CONFIG.integrated.tau_pred
        RHO_PRED_ORIG = CONFIG.integrated.rho_pred
        ETA_BOOK_ORIG = CONFIG.integrated.eta_book
        RHO_BOOK_ORIG = CONFIG.integrated.rho_book

        # Operating room and cost parameters
        C = CONFIG.operating_room.block_size_minutes
        c_r_base = CONFIG.costs.rejection_per_case
        c_o_base = CONFIG.costs.overtime_per_min
        c_i_base = CONFIG.costs.idle_per_min
        max_overtime = CONFIG.costs.max_overtime_minutes
        min_dur = CONFIG.operating_room.min_procedure_duration
        max_dur = (
            CONFIG.operating_room.block_size_minutes
            + DomainConstants.MAX_OVERTIME_MINUTES_PER_BLOCK
        )

        logger.info("Starting optimization with %d planning periods", num_weeks)

        # Apply cost scaling to local variables only (no CONFIG mutation)
        bp_mean = np.mean(
            [
                week_data[DataColumns.BOOKED_MIN].mean()
                for week_data in df_plan_data
                if not week_data.empty
            ]
        )
        cost_scale = 1.0
        max_cost = max(
            c_r_base * bp_mean if not np.isnan(bp_mean) else 0,
            c_o_base,
            c_i_base,
        )

        if max_cost > 1000:
            cost_scale = 1000.0 / max_cost
            logger.info("Applying cost scaling factor %.4f", cost_scale)

        # Scale all parameters consistently in local variables
        c_r = c_r_base * cost_scale
        c_o = c_o_base * cost_scale
        c_i = c_i_base * cost_scale
        rho_book = RHO_BOOK_ORIG * cost_scale
        rho_pred = RHO_PRED_ORIG * cost_scale
        lambda_reg = LAMBDA_ORIG * cost_scale
        tau_pred = TAU_PRED_ORIG  # No scaling needed for tolerance
        eta_book = ETA_BOOK_ORIG  # No scaling needed for tolerance

        logger.info("Initializing model coefficients with Lasso solution")
        theta_init = (
            Lasso(alpha=lambda_reg, fit_intercept=False, max_iter=10_000)
            .fit(Xw, yw)
            .coef_
        )

        logger.info("Building optimization model")
        (
            master,
            theta_var,
            s_pred_var,
            u_var,
            beta_vars,
            trp_var,
            trn_var,
            level_constr,
            orig_obj,
        ) = build_optimization_model(
            Xw, yw, p, lambda_reg, num_weeks, tau_pred, rho_pred, theta_init
        )

        model_center = theta_init.copy()
        reference_point = theta_init.copy()

        # Initialize tracking variables
        best_upper_bound = float("inf")
        best_theta = theta_init.copy()
        theta_prev = None

        warm_starts = {}
        greedy_costs_by_week = {}
        lp_lb_by_week = {}

        iteration_info = []
        no_progress_iters = 0
        prev_gap = float("inf")

        # Certification parameters
        cert_budget = 180.0  # seconds
        full_sweep_triggered = False

        logger.info(
            "Running optimization loop for %d iterations", max_iterations
        )

        for iteration in range(max_iterations):
            logger.info("Iteration %d/%d", iteration + 1, max_iterations)

            # Update level constraint RHS (between LB and UB)
            if iteration > 0:  # Skip first iteration (no meaningful bounds yet)
                if best_upper_bound < float("inf") and lower_bound is not None:
                    level_value = float(
                        lower_bound + level_factor * (best_upper_bound - lower_bound)
                    )
                    level_constr.RHS = level_value
                    logger.info("Updated target level to %.2f", level_value)

            # Update trust region center and constraints
            if iteration > 0:
                model_center = (1 - update_rate) * model_center + update_rate * theta_current

                # Update trust region constraint RHS
                for j in range(p):
                    master._tr_pos_constrs[j].RHS = model_center[j]
                    master._tr_neg_constrs[j].RHS = -model_center[j]

            # Update reference point
            if iteration > 0:
                reference_point = 0.7 * reference_point + 0.3 * theta_current

            # Solve master problem with level infeasibility guard
            master_start = time.time()
            master.optimize()

            # Handle level constraint infeasibility
            if master.Status == GRB.INFEASIBLE:
                logger.info("Relaxing level constraint due to infeasibility")
                level_constr.RHS = GRB.INFINITY  # Disable level constraint this round
                master.optimize()

            master_time = time.time() - master_start

            if master.Status != GRB.OPTIMAL:
                logger.error("Master problem failed with status %s", master.Status)
                break

            theta_current = np.array([theta_var[j].X for j in range(p)])

            # Initialize lower bound for first iteration (rigorous LB computed after cuts)
            if iteration == 0:
                lower_bound = None

            logger.info("Master problem solved in %.2fs", master_time)

            # Compute scheduling upper bound
            logger.info("Estimating scheduling upper bound")
            greedy_start = time.time()

            total_greedy_scheduling_cost = 0.0
            for week_idx, (week_data, week_blocks) in enumerate(
                zip(df_plan_data, week_blocks_list)
            ):
                if week_data.empty:
                    greedy_costs_by_week[week_idx] = 0.0
                    warm_starts[week_idx] = (np.zeros((0, 0)), np.zeros(0))
                    continue

                X_week = week_data[FeatureColumns.ALL]
                X_processed = preproc.transform(X_week)
                predicted_durations = X_processed @ theta_current
                predicted_durations = np.clip(predicted_durations, min_dur, max_dur)

                bp = week_data[DataColumns.BOOKED_MIN].to_numpy()
                num_blocks = len(week_blocks)

                # Estimate scheduling cost using local parameters
                cost, _, _, z_warm, r_warm = calculate_schedule_cost(
                    predicted_durations,
                    bp,
                    num_blocks,
                    C,
                    c_r,
                    c_o,
                    c_i,
                    enable_optimization=True,
                )

                # Floor by LP lower bound for consistency
                if week_idx in lp_lb_by_week:
                    cost = max(cost, lp_lb_by_week[week_idx])

                greedy_costs_by_week[week_idx] = cost
                warm_starts[week_idx] = (z_warm, r_warm)
                total_greedy_scheduling_cost += cost

            pred_penalty, l1_penalty, book_penalty = calculate_objective_components(
                theta_current,
                Xw,
                yw,
                df_plan_data,
                preproc,
                lambda_reg,
                tau_pred,
                rho_pred,
                rho_book,
                eta_book,
            )

            greedy_upper_bound = (
                pred_penalty + l1_penalty + total_greedy_scheduling_cost + book_penalty
            )

            greedy_time = time.time() - greedy_start
            logger.info(
                "Scheduling upper bound %.2f calculated in %.3fs",
                greedy_upper_bound,
                greedy_time,
            )

            # Emergency consistency check
            if lower_bound is not None and lower_bound > best_upper_bound + 1e-6:
                logger.warning("Lower bound exceeded stored best upper bound; resetting")
                best_upper_bound = greedy_upper_bound
                best_theta = theta_current.copy()
                logger.info("Reset best upper bound to %.2f", best_upper_bound)

            # Update best solution
            if greedy_upper_bound < best_upper_bound:
                best_upper_bound = greedy_upper_bound
                best_theta = theta_current.copy()
                no_progress_iters = 0
                logger.info("New best upper bound: %.2f", best_upper_bound)
            else:
                no_progress_iters += 1

            # Generate cuts using multiple reference points
            logger.info("Adding scheduling cuts")
            cuts_start = time.time()
            cuts_added = 0

            # Multiple reference points for stronger cuts
            theta_refs = [theta_current]
            if theta_prev is not None and not np.allclose(
                theta_prev, theta_current, atol=1e-6
            ):
                theta_refs.append(theta_prev)
            if not np.allclose(best_theta, theta_current, atol=1e-6):
                theta_refs.append(best_theta)
            if not np.allclose(reference_point, theta_current, atol=1e-6):
                theta_refs.append(reference_point)

            for week_idx, (week_data, week_blocks) in enumerate(
                zip(df_plan_data, week_blocks_list)
            ):
                if week_data.empty:
                    continue

                X_week = week_data[FeatureColumns.ALL]
                X_processed = preproc.transform(X_week)

                # Extract scheduling cut at current point for LB tracking
                h_sched_current, b_sched_current = generate_scheduling_cut(
                    week_data,
                    week_blocks,
                    X_processed,
                    theta_current,
                    week_idx,
                    C,
                    c_r,
                    c_o,
                    c_i,
                    max_overtime,
                    min_dur,
                    max_dur,
                    use_affine_sched_cuts,
                )

                # For LB tracking, compute LP value at current point
                if use_affine_sched_cuts:
                    lp_lb_by_week[week_idx] = float(
                        b_sched_current + h_sched_current @ theta_current
                    )
                else:
                    lp_lb_by_week[week_idx] = float(
                        b_sched_current
                    )  # Constant cut value

                # Generate cuts at multiple reference points
                for ref_idx, theta_ref in enumerate(theta_refs):
                    # Scheduling cut at this reference point
                    if np.allclose(theta_ref, theta_current, atol=1e-6):
                        # Use already computed cut at current point
                        h_sched, b_sched = h_sched_current, b_sched_current
                    else:
                        # Compute cut at different reference point
                        h_sched, b_sched = generate_scheduling_cut(
                            week_data,
                            week_blocks,
                            X_processed,
                            theta_ref,
                            week_idx,
                            C,
                            c_r,
                            c_o,
                            c_i,
                            max_overtime,
                            min_dur,
                            max_dur,
                            use_affine_sched_cuts,
                        )

                    # Booked-time hinge cut at this reference point
                    h_book, b_book = extract_booked_hinge_cut_at(
                        week_data, X_processed, theta_ref, eta_book
                    )

                    # Combine cuts using standard intercepts
                    h_total = h_sched + rho_book * h_book
                    b_total = b_sched + rho_book * b_book

                    # Skip near-zero cuts
                    if np.linalg.norm(h_total) <= 1e-10 and abs(b_total) <= 1e-10:
                        continue

                    # Check if cut is violated using current β values
                    beta_current = beta_vars[week_idx].X
                    cut_value = float(b_total + h_total @ theta_current)
                    if cut_value <= beta_current + 1e-6:
                        continue  # Skip non-violated cut

                    # Check for near-duplicates using standardized signature
                    cut_signature = create_cut_signature(h_total, b_total)
                    if cut_signature in cut_pool[week_idx]:
                        continue  # Skip duplicate

                    # Add the cut: β ≥ b + h^T θ
                    cut_expr = b_total + quicksum(
                        h_total[j] * theta_var[j] for j in range(p)
                    )
                    master.addConstr(
                        beta_vars[week_idx] >= cut_expr,
                        name=f"cut_w{week_idx}_iter{iteration}_ref{ref_idx}",
                    )

                    # Track the cut
                    cut_pool[week_idx].add(cut_signature)
                    cuts_added += 1

            # Manage cut pool size to prevent memory bloat
            manage_cut_pool(cut_pool)

            master.update()
            cuts_time = time.time() - cuts_start
            logger.info("Cuts added: %d (%.3fs)", cuts_added, cuts_time)

            # Compute rigorous master lower bound
            if iteration >= 1 and cuts_added > 0:  # Only if we have meaningful cuts
                # Temporarily solve master with original objective for true LB
                saved_obj = master.getObjective()
                saved_rhs = level_constr.RHS

                level_constr.RHS = GRB.INFINITY  # Disable level constraint
                master.setObjective(orig_obj, GRB.MINIMIZE)
                master.optimize()

                if master.Status == GRB.OPTIMAL:
                    lower_bound = master.ObjVal  # Rigorous LB
                else:
                    # Fallback to component-based LB if master fails
                    pred_penalty_lb, l1_penalty_lb, book_penalty_lb = (
                        calculate_objective_components(
                            theta_current,
                            Xw,
                            yw,
                            df_plan_data,
                            preproc,
                            lambda_reg,
                            tau_pred,
                            rho_pred,
                            rho_book,
                            eta_book,
                        )
                    )
                    current_scheduling_lb = sum(lp_lb_by_week.values())
                    lower_bound = (
                        pred_penalty_lb
                        + l1_penalty_lb
                        + book_penalty_lb
                        + current_scheduling_lb
                    )

                # Restore stabilized formulation
                master.setObjective(quicksum(trp_var) + quicksum(trn_var), GRB.MINIMIZE)
                level_constr.RHS = saved_rhs
                master.update()
            else:
                # First iteration or no cuts: use component-based LB
                pred_penalty_lb, l1_penalty_lb, book_penalty_lb = (
                    calculate_objective_components(
                        theta_current,
                        Xw,
                        yw,
                        df_plan_data,
                        preproc,
                        lambda_reg,
                        tau_pred,
                        rho_pred,
                        rho_book,
                        eta_book,
                    )
                )
                current_scheduling_lb = (
                    sum(lp_lb_by_week.values()) if lp_lb_by_week else 0.0
                )
                lower_bound = (
                    pred_penalty_lb
                    + l1_penalty_lb
                    + book_penalty_lb
                    + current_scheduling_lb
                )

            logger.info("Updated lower bound: %.2f", lower_bound)

            # Age-aware MIP certification with time budget
            exact_ub = greedy_upper_bound

            # Check if we should do full sweep (first time gap < 5%)
            current_gap = (
                (best_upper_bound - lower_bound) / abs(best_upper_bound)
                if abs(best_upper_bound) > 1e-12
                else 0.0
            )
            if current_gap < 0.05 and not full_sweep_triggered:
                logger.info("Gap below 5%%; running full validation sweep")
                full_sweep_triggered = True

                # Certify ALL non-empty weeks
                mip_start = time.time()
                exact_scheduling_cost = 0.0

                for week_idx in non_empty_weeks:
                    if df_plan_data[week_idx].empty:
                        continue

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
                        C,
                        c_r,
                        c_o,
                        c_i,
                        max_overtime,
                        min_dur,
                        max_dur,
                        time_limit=60,
                        mip_gap=0.02,  # Tighter gap for full sweep
                    )
                    exact_scheduling_cost += exact_cost
                    last_certified[week_idx] = iteration

                exact_ub = (
                    pred_penalty + l1_penalty + exact_scheduling_cost + book_penalty
                )
                mip_time = time.time() - mip_start
                logger.info(
                    "Full validation upper bound %.2f computed in %.2fs",
                    exact_ub,
                    mip_time,
                )

                if exact_ub < best_upper_bound:
                    best_upper_bound = exact_ub
                    best_theta = theta_current.copy()
                    no_progress_iters = 0

            elif (
                iteration + 1
            ) % mip_certification_every_k == 0 or no_progress_iters >= 5:
                # Regular age-aware certification with budget
                ages = [(iteration - last_certified[t], t) for t in non_empty_weeks]
                ages.sort(reverse=True)  # Oldest first

                selected_weeks = []
                budget_remaining = cert_budget

                for age, week_idx in ages:
                    if budget_remaining <= 0:
                        break
                    selected_weeks.append(week_idx)
                    budget_remaining -= 60  # Rough estimate per week

                if selected_weeks:
                    logger.info(
                        "Running targeted validation on %d weeks (budget %.0fs)",
                        len(selected_weeks),
                        cert_budget,
                    )

                    mip_start = time.time()
                    exact_scheduling_cost = 0.0
                    remaining_greedy_cost = 0.0
                    spent = 0.0

                    for week_idx in range(num_weeks):
                        if (
                            week_idx in selected_weeks
                            and not df_plan_data[week_idx].empty
                        ):
                            if spent >= cert_budget:
                                break

                            t0 = time.time()
                            time_limit = min(60, math.ceil(cert_budget - spent))

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
                                C,
                                c_r,
                                c_o,
                                c_i,
                                max_overtime,
                                min_dur,
                                max_dur,
                                time_limit=time_limit,
                                mip_gap=0.05,
                            )

                            exact_scheduling_cost += exact_cost
                            last_certified[week_idx] = iteration
                            spent += time.time() - t0
                        else:
                            remaining_greedy_cost += greedy_costs_by_week.get(
                                week_idx, 0.0
                            )

                    total_exact_scheduling = (
                        exact_scheduling_cost + remaining_greedy_cost
                    )
                    exact_ub = (
                        pred_penalty
                        + l1_penalty
                        + total_exact_scheduling
                        + book_penalty
                    )

                    mip_time = time.time() - mip_start
                    logger.info(
                        "Targeted validation upper bound %.2f computed in %.2fs",
                        exact_ub,
                        mip_time,
                    )

                    if exact_ub < best_upper_bound:
                        best_upper_bound = exact_ub
                        best_theta = theta_current.copy()
                        no_progress_iters = 0

            # Gap calculation and convergence check
            if abs(best_upper_bound) > 1e-12 and lower_bound is not None:
                gap = max(0.0, (best_upper_bound - lower_bound) / abs(best_upper_bound))
            else:
                gap = 0.0

            # Convert to original units for display (restore all scaled parameters)
            if cost_scale != 1.0:
                display_lb = (
                    lower_bound / cost_scale if lower_bound is not None else 0.0
                )
                display_ub = best_upper_bound / cost_scale
                display_sched = total_greedy_scheduling_cost / cost_scale
                display_pred = pred_penalty / cost_scale
                display_book = book_penalty / cost_scale
                display_l1 = l1_penalty / cost_scale
            else:
                display_lb = lower_bound if lower_bound is not None else 0.0
                display_ub = best_upper_bound
                display_sched = total_greedy_scheduling_cost
                display_pred = pred_penalty
                display_book = book_penalty
                display_l1 = l1_penalty

            logger.info(
                "Bounds: LB=%.2f, UB=%.2f | Components: Pred=%.1f, L1=%.1f, Sched=%.1f, Book=%.1f",
                display_lb,
                display_ub,
                display_pred,
                display_l1,
                display_sched,
                display_book,
            )
            logger.info("Gap: %.4f (%.2f%%)", gap, gap * 100)

            # Progress tracking
            if gap < prev_gap - 0.01:
                no_progress_iters = 0
                prev_gap = gap
            else:
                no_progress_iters += 1

            # Store iteration info
            iteration_info.append(
                {
                    "iteration": iteration + 1,
                    "master_obj": (
                        float(lower_bound) if lower_bound is not None else 0.0
                    ),
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

            # Convergence criteria
            max_age = (
                max(iteration - last_certified[t] for t in non_empty_weeks)
                if non_empty_weeks
                else 0
            )
            coverage_window = 3 * max(
                1, mip_certification_every_k
            )  # Allow some staleness

            if gap < 0.01 and max_age <= coverage_window:
                logger.info("Converged: gap below 1%% with recent validation coverage")
                break
            elif gap < 0.01:
                logger.info(
                    "Gap below 1%% but waiting for validation coverage (max age %d)",
                    max_age,
                )

            if cuts_added == 0 and iteration >= 5:
                logger.info("Converged: no new cuts added")
                break

            # Update theta tracking
            theta_prev = theta_current.copy()

        end_time = time.time()
        total_time = end_time - start_time

        # Convert final objective to original units for reporting (ensure all scaling restored)
        final_objective_original = (
            best_upper_bound / cost_scale if cost_scale != 1.0 else best_upper_bound
        )

        logger.info("Optimization completed in %.2fs", total_time)
        logger.info("Best objective value: %.2f", final_objective_original)
        logger.info("Final gap: %.4f (%.2f%%)", gap, gap * 100)
        logger.info("Iterations run: %d", len(iteration_info))
        if iteration_info:
            avg_iter_time = total_time / len(iteration_info)
            logger.info("Average iteration time: %.2fs", avg_iter_time)
        logger.info(
            "Scheduling cuts used: %s",
            "Affine" if use_affine_sched_cuts else "Constant",
        )

        return {
            "theta": best_theta,
            "objective": float(final_objective_original),
            "iterations": len(iteration_info),
            "iteration_info": iteration_info,
            "total_time": float(total_time),
            "converged": gap < 0.01,
        }

    except Exception as e:
        logger.exception("Error in solve_integrated_optimization")
        return {
            "theta": None,
            "objective": None,
            "iterations": 0,
            "iteration_info": [],
            "total_time": 0.0,
            "converged": False,
            "error": str(e),
        }


def build_planning_weeks_data(
    df_warm: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[List[Tuple[int, int]]]]:
    """Partition warm-up data into weekly planning periods."""
    if df_warm.empty:
        return [], []

    start_date = df_warm[DataColumns.ACTUAL_START].min()
    end_date = df_warm[DataColumns.ACTUAL_START].max()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    # Align to Monday-Sunday week boundaries
    days_from_monday = start_ts.weekday()
    first_monday = start_ts - pd.Timedelta(days=days_from_monday)

    logger.info(
        "Planning window: %s to %s", start_ts.date(), end_ts.date()
    )
    logger.info("First planning Monday: %s", first_monday.date())

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

        logger.info(
            "Week %d: %s-%s | Surgeries: %d | Blocks: %d",
            week_idx,
            current_monday.date(),
            week_end.date(),
            len(df_week),
            len(week_blocks),
        )

        current_monday += pd.Timedelta(days=7)
        week_idx += 1

    logger.info("Prepared %d planning weeks", len(week_data_list))
    return week_data_list, week_blocks_list


def main() -> None:
    """Run the integrated optimization experiment."""
    print("Surgery Scheduling Optimization")
    print("Starting optimization...\n")

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
        f"Planning summary: {total_surgeries} surgeries across {total_blocks} blocks"
    )

    # Test multiple parameter combinations
    param_sets = CONFIG.integrated.param_values_to_test
    all_results = {}

    print(f"Evaluating {len(param_sets)} parameter sets\n")

    for i, params in enumerate(param_sets):
        print(f"Parameter set {i + 1}/{len(param_sets)}: {params}")

        # Temporarily update configuration
        original_params = {}
        for key, value in params.items():
            original_params[key] = getattr(CONFIG.integrated, key)
            setattr(CONFIG.integrated, key, value)

        # Solve integrated problem
        results = solve_integrated_optimization(
            week_data_list,
            week_blocks_list,
            Xw,
            yw,
            preproc,
            max_iterations=100,
            use_parallel=True,
            mip_certification_every_k=10,
            num_horizons_for_mip=8,
            use_affine_sched_cuts=True,
        )

        # Restore original configuration
        for key, value in original_params.items():
            setattr(CONFIG.integrated, key, value)

        param_key = "_".join([f"{k}{v}" for k, v in params.items()])
        all_results[param_key] = {"results": results, "params": params.copy()}

        # Save optimal parameters
        if results.get("theta") is not None:
            feat_names = list(preproc.get_feature_names_out())
            clean_names = [canonical_name(n) for n in feat_names]
            theta_out = {
                clean_names[j]: float(results["theta"][j])
                for j in range(len(feat_names))
            }

            out_path = Path(CONFIG.data.theta_path)
            param_theta_path = out_path.parent / f"theta_{param_key}_enhanced.json"
            param_theta_path.parent.mkdir(parents=True, exist_ok=True)
            with param_theta_path.open("w") as f:
                json.dump(theta_out, f, indent=2)
            logger.info("Saved parameter coefficients for %s to %s", param_key, param_theta_path)

    # Save performance comparison
    out_path = Path(CONFIG.data.theta_path)
    performance_results = {}

    for param_key, data in all_results.items():
        results = data["results"]
        params = data["params"]

        if results.get("theta") is not None and results.get("iteration_info"):
            last_iter = results["iteration_info"][-1]

            performance_results[param_key] = {
                "parameters": params,
                "prediction_penalty": float(last_iter.get("prediction_penalty", 0)),
                "l1_penalty": float(last_iter.get("l1_penalty", 0)),
                "booked_penalty": float(last_iter.get("booked_penalty", 0)),
                "greedy_scheduling_cost": float(
                    last_iter.get("greedy_scheduling_cost", 0)
                ),
                "total_objective": float(results.get("objective", 0)),
                "iterations": int(results.get("iterations", 0)),
                "converged": bool(results.get("converged", False)),
                "total_time": float(results.get("total_time", 0)),
            }

            logger.info(
                "%s performance | Prediction: %.2f | L1: %.2f | Booked: %.2f | Scheduling: %.2f | Total: %.2f",
                param_key,
                last_iter.get("prediction_penalty", 0),
                last_iter.get("l1_penalty", 0),
                last_iter.get("booked_penalty", 0),
                last_iter.get("greedy_scheduling_cost", 0),
                results.get("objective", 0),
            )

    performance_path = out_path.parent / "performance_comparison_enhanced.json"
    with performance_path.open("w") as f:
        json.dump(to_json_safe(performance_results), f, indent=2)
    logger.info("Performance comparison saved to %s", performance_path)

    # Save detailed results
    detailed_results_path = (
        out_path.parent / "all_params_detailed_results_enhanced.json"
    )
    detailed_results = {}
    for param_key, data in all_results.items():
        results = data["results"]
        detailed_results[param_key] = {
            "objective": float(results.get("objective", 0)),
            "iterations": int(results.get("iterations", 0)),
            "total_time": float(results.get("total_time", 0)),
            "converged": bool(results.get("converged", False)),
            "iteration_info": to_json_safe(results.get("iteration_info", [])),
        }

    with detailed_results_path.open("w") as f:
        json.dump(to_json_safe(detailed_results), f, indent=2)
    logger.info("Detailed results saved to %s", detailed_results_path)

    # Print final summary
    print("\nOptimization summary by parameter set:")
    for param_key in sorted(performance_results.keys()):
        perf = performance_results[param_key]
        print(
            f"{param_key}: Total={perf['total_objective']:.2f}, "
            f"Iterations={perf['iterations']}, Time={perf['total_time']:.1f}s"
        )


if __name__ == "__main__":
    main()
