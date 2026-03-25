"""Restricted Master Problem — Gurobi MILP implementation.

Solves the RMP over the active column pools while jointly optimizing
recommendation weights and follower-optimal column selectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.bilevel.config import BilevelConfig
from src.core.column import ScheduleColumn
from src.core.config import CostConfig
from src.core.types import BlockId
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData

logger = logging.getLogger(__name__)


@dataclass
class RMPResult:
    w: np.ndarray
    selected_columns: Dict[int, int]
    value_functions: Dict[int, float]
    objective: float
    solve_time: float
    status: str


def _compute_big_m(week_data: WeekRecommendationData, costs: CostConfig, turnover: float) -> float:
    """Conservative upper bound on predicted cost for any feasible column response."""
    U = week_data.U_bounds
    n_cases = week_data.n_cases
    candidates = week_data.calendar.candidates

    max_activation = sum(b.activation_cost for b in candidates)
    max_deferral = costs.deferral_per_case * n_cases

    if not candidates:
        return max_activation + max_deferral

    c_max = max((b.capacity_minutes for b in candidates), default=480.0)
    max_load = float(U.sum()) + turnover * max(n_cases - 1, 0)
    max_ot = costs.overtime_per_minute * max(max_load - c_max, 0.0)
    max_idle = costs.idle_per_minute * c_max

    return max_activation + max_deferral + len(candidates) * (max_ot + max_idle)


def _compute_mae_base(week_data_list: List[WeekRecommendationData]) -> float:
    """MAE of bookings versus realized durations across training weeks."""
    total_abs_err = 0.0
    total_cases = 0
    for wd in week_data_list:
        total_abs_err += float(np.sum(np.abs(wd.realized - wd.bookings)))
        total_cases += wd.n_cases
    if total_cases == 0:
        return 1.0
    return total_abs_err / total_cases


def solve_restricted_master(
    week_data_list: List[WeekRecommendationData],
    column_pools: Dict[int, List[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    config: BilevelConfig,
    costs: CostConfig,
    turnover: float,
) -> RMPResult:
    """Solve the restricted master problem MILP."""
    _ = recommendation_model
    t0 = perf_counter()

    n_weeks = len(week_data_list)
    if n_weeks == 0:
        return RMPResult(
            w=np.zeros(0),
            selected_columns={},
            value_functions={},
            objective=0.0,
            solve_time=0.0,
            status="EMPTY",
        )

    feature_dim = week_data_list[0].features.shape[1]
    total_cases = sum(wd.n_cases for wd in week_data_list)

    realized_costs: Dict[int, List[float]] = {}
    for wd in week_data_list:
        pool = column_pools[wd.week_index]
        realized_costs[wd.week_index] = [col.compute_cost(wd.realized, costs, turnover) for col in pool]

    big_m: Dict[int, float] = {wd.week_index: _compute_big_m(wd, costs, turnover) for wd in week_data_list}

    mae_base = _compute_mae_base(week_data_list)
    e_pred_max = config.credibility_eta * mae_base

    model = gp.Model("RMP")
    model.Params.TimeLimit = config.master_time_limit
    model.Params.MIPGap = config.master_mip_gap
    model.Params.OutputFlag = 0
    model.Params.Threads = 0

    w_vars = model.addVars(feature_dim, lb=-config.w_max, ub=config.w_max, vtype=GRB.CONTINUOUS, name="w")

    delta_rec: Dict[tuple[int, int], gp.Var] = {}
    delta_post: Dict[tuple[int, int], gp.Var] = {}
    d_post: Dict[tuple[int, int], gp.Var] = {}
    lam: Dict[tuple[int, int, int], gp.Var] = {}
    e_plus: Dict[tuple[int, int], gp.Var] = {}
    e_minus: Dict[tuple[int, int], gp.Var] = {}
    phi: Dict[int, gp.Var] = {}
    y_sel: Dict[tuple[int, int], gp.Var] = {}
    p_var: Dict[tuple[int, int], gp.Var] = {}
    o_plus: Dict[tuple[int, int, BlockId], gp.Var] = {}
    o_minus: Dict[tuple[int, int, BlockId], gp.Var] = {}

    for wd in week_data_list:
        t = wd.week_index
        for i in range(wd.n_cases):
            lb_rec = float(wd.L_bounds[i] - wd.bookings[i])
            ub_rec = float(wd.U_bounds[i] - wd.bookings[i])
            delta_rec[t, i] = model.addVar(lb=lb_rec, ub=ub_rec, vtype=GRB.CONTINUOUS, name=f"drec_{t}_{i}")
            delta_post[t, i] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"dpost_{t}_{i}")
            d_post[t, i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"dpostabs_{t}_{i}")
            e_plus[t, i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"ep_{t}_{i}")
            e_minus[t, i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"em_{t}_{i}")
            for m in range(5):
                lam[t, i, m] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"lam_{t}_{i}_{m}")

        pool = column_pools[t]
        phi[t] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"phi_{t}")
        for g_idx, col in enumerate(pool):
            y_sel[t, g_idx] = model.addVar(vtype=GRB.BINARY, name=f"ysel_{t}_{g_idx}")
            p_var[t, g_idx] = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"P_{t}_{g_idx}")
            for bid in col.v_open:
                name_suffix = f"{bid.day_index}_{bid.site}_{bid.room}"
                o_plus[t, g_idx, bid] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"op_{t}_{g_idx}_{name_suffix}")
                o_minus[t, g_idx, bid] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"om_{t}_{g_idx}_{name_suffix}")

    for wd in week_data_list:
        t = wd.week_index
        x = wd.features
        for i in range(wd.n_cases):
            model.addConstr(
                delta_rec[t, i] == quicksum(float(x[i, j]) * w_vars[j] for j in range(feature_dim)),
                name=f"rec_{t}_{i}",
            )

            sos2_case = wd.sos2_data[i]
            knot_x = sos2_case.knot_x
            knot_y = sos2_case.knot_y
            model.addConstr(
                delta_rec[t, i] == quicksum(lam[t, i, m] * float(knot_x[m]) for m in range(5)),
                name=f"sos2x_{t}_{i}",
            )
            model.addConstr(
                delta_post[t, i] == quicksum(lam[t, i, m] * float(knot_y[m]) for m in range(5)),
                name=f"sos2y_{t}_{i}",
            )
            model.addConstr(quicksum(lam[t, i, m] for m in range(5)) == 1.0, name=f"sos2sum_{t}_{i}")
            model.addSOS(GRB.SOS_TYPE2, [lam[t, i, m] for m in range(5)])

            model.addConstr(d_post[t, i] == float(wd.bookings[i]) + delta_post[t, i], name=f"dpost_def_{t}_{i}")
            model.addConstr(
                float(wd.realized[i]) - d_post[t, i] == e_plus[t, i] - e_minus[t, i],
                name=f"cred_split_{t}_{i}",
            )

        pool = column_pools[t]
        model.addConstr(quicksum(y_sel[t, g_idx] for g_idx in range(len(pool))) == 1, name=f"sel_{t}")

        for g_idx, col in enumerate(pool):
            fixed_cost = col.get_fixed_cost_component(costs)
            ot_idle_expr = gp.LinExpr(0.0)

            for bid in col.v_open:
                case_ids = col.cases_in_block(bid)
                n_in_block = len(case_ids)
                cap = col.block_capacities[bid]
                turnover_const = turnover * max(n_in_block - 1, 0)
                load_expr = quicksum(d_post[t, i] for i in case_ids) + turnover_const

                model.addConstr(o_plus[t, g_idx, bid] >= load_expr - cap, name=f"ot_{t}_{g_idx}_{bid.day_index}_{bid.room}")
                model.addConstr(o_minus[t, g_idx, bid] >= cap - load_expr, name=f"idle_{t}_{g_idx}_{bid.day_index}_{bid.room}")

                ot_idle_expr += costs.overtime_per_minute * o_plus[t, g_idx, bid]
                ot_idle_expr += costs.idle_per_minute * o_minus[t, g_idx, bid]

            model.addConstr(p_var[t, g_idx] == fixed_cost + ot_idle_expr, name=f"Pdef_{t}_{g_idx}")
            model.addConstr(phi[t] <= p_var[t, g_idx], name=f"vf_lb_{t}_{g_idx}")
            model.addConstr(
                p_var[t, g_idx] <= phi[t] + big_m[t] * (1 - y_sel[t, g_idx]),
                name=f"vf_ub_{t}_{g_idx}",
            )

    if total_cases > 0:
        model.addConstr(
            (1.0 / total_cases)
            * quicksum(
                e_plus[wd.week_index, i] + e_minus[wd.week_index, i]
                for wd in week_data_list
                for i in range(wd.n_cases)
            )
            <= e_pred_max,
            name="credibility",
        )

    obj = quicksum(
        y_sel[wd.week_index, g_idx] * realized_costs[wd.week_index][g_idx]
        for wd in week_data_list
        for g_idx in range(len(column_pools[wd.week_index]))
    )
    model.setObjective(obj * (1.0 / n_weeks), GRB.MINIMIZE)

    model.optimize()
    solve_time = perf_counter() - t0

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status_name = status_map.get(model.Status, str(model.Status))

    if model.SolCount == 0:
        logger.warning("RMP returned no solution (status=%s). Falling back to w=0.", status_name)
        return _fallback_zero_weights(week_data_list, column_pools, costs, turnover, config, solve_time, status_name)

    w_opt = np.array([w_vars[j].X for j in range(feature_dim)], dtype=float)
    selected_columns: Dict[int, int] = {}
    value_functions: Dict[int, float] = {}

    for wd in week_data_list:
        t = wd.week_index
        value_functions[t] = float(phi[t].X)
        pool_len = len(column_pools[t])
        selected = next((g for g in range(pool_len) if y_sel[t, g].X > 0.5), 0)
        selected_columns[t] = selected

    objective = float(model.ObjVal)
    logger.info("RMP solved: status=%s obj=%.2f |w|_inf=%.4f time=%.1fs", status_name, objective, float(np.max(np.abs(w_opt))), solve_time)

    return RMPResult(
        w=w_opt,
        selected_columns=selected_columns,
        value_functions=value_functions,
        objective=objective,
        solve_time=solve_time,
        status=status_name,
    )


def _fallback_zero_weights(
    week_data_list: List[WeekRecommendationData],
    column_pools: Dict[int, List[ScheduleColumn]],
    costs: CostConfig,
    turnover: float,
    config: BilevelConfig,
    solve_time: float,
    status_name: str,
) -> RMPResult:
    """Emergency fallback: evaluate all pools at w=0."""
    feature_dim = week_data_list[0].features.shape[1] if week_data_list else 0
    w = np.zeros(feature_dim, dtype=float)

    selected_columns: Dict[int, int] = {}
    value_functions: Dict[int, float] = {}
    realized_total = 0.0

    for wd in week_data_list:
        t = wd.week_index
        pool = column_pools[t]
        predicted = [col.compute_cost(wd.bookings, costs, turnover) for col in pool]
        realized = [col.compute_cost(wd.realized, costs, turnover) for col in pool]
        phi_val = float(min(predicted))
        value_functions[t] = phi_val

        tol = config.convergence_tol
        feasible = [g for g, p in enumerate(predicted) if p <= phi_val + tol]
        chosen = min(feasible, key=lambda g: realized[g])
        selected_columns[t] = chosen
        realized_total += realized[chosen]

    objective = realized_total / max(len(week_data_list), 1)
    return RMPResult(
        w=w,
        selected_columns=selected_columns,
        value_functions=value_functions,
        objective=objective,
        solve_time=solve_time,
        status=f"FALLBACK_{status_name}",
    )
