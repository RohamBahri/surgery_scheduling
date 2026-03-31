"""Native-variable restricted compact master for VFCG."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.core.column import ScheduleColumn, extract_column_from_model
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_weekly_optimistic
from src.vfcg.cuts import add_reference_predicted_cost_rhs
from src.vfcg.oracle import build_case_records_from_week_data

logger = logging.getLogger(__name__)


@dataclass
class NativeMasterResult:
    weights: np.ndarray
    schedules_by_week: dict[int, ScheduleColumn]
    objective: float
    bound: float
    gap: float
    solve_time: float
    status: str
    predicted_costs_by_week: dict[int, float]
    is_fallback: bool = False


def _status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status, str(status))


def _compute_mae_base(week_data_list: list[WeekRecommendationData]) -> float:
    errs = []
    for wd in week_data_list:
        errs.extend(np.abs(np.asarray(wd.realized, dtype=float) - np.asarray(wd.bookings, dtype=float)).tolist())
    if not errs:
        return 0.0
    return float(np.mean(errs))


def solve_native_master(
    week_data_list: list[WeekRecommendationData],
    reference_sets: dict[int, list[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    config: Config,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
) -> NativeMasterResult:
    t0 = time.perf_counter()
    n_weeks = len(week_data_list)
    feat_dim = recommendation_model.feature_dim if hasattr(recommendation_model, "feature_dim") else week_data_list[0].features.shape[1]

    model = gp.Model("vfcg_native_master")
    model.Params.TimeLimit = config.vfcg.master_time_limit
    model.Params.MIPGap = config.vfcg.master_mip_gap
    model.Params.Threads = solver_cfg.threads
    model.Params.OutputFlag = 1 if solver_cfg.verbose else 0

    w = model.addVars(feat_dim, lb=-config.vfcg.w_max, ub=config.vfcg.w_max, name="w")

    z_vars: dict[int, dict[tuple[int, object], gp.Var]] = {}
    r_vars: dict[int, dict[int, gp.Var]] = {}
    v_vars: dict[int, dict[object, gp.Var]] = {}
    y_vars: dict[int, dict[object, gp.Var]] = {}
    d_post_vars: dict[int, dict[int, gp.Var]] = {}

    predicted_cost_exprs: dict[int, gp.LinExpr] = {}
    realized_cost_exprs: dict[int, gp.LinExpr] = {}

    abs_err_terms = []

    # McCormick products are exact at integer solutions; LP relaxation can be weak.
    for t, wd in enumerate(week_data_list):
        blocks = list(wd.calendar.block_ids)
        z_t: dict[tuple[int, object], gp.Var] = {}
        r_t = model.addVars(wd.n_cases, vtype=GRB.BINARY, name=f"r_t{t}")
        v_t = model.addVars(blocks, vtype=GRB.BINARY, name=f"v_t{t}")
        y_t = model.addVars(blocks, vtype=GRB.BINARY, name=f"y_t{t}")

        # Recommendation pipeline variables.
        dpost_t: dict[int, gp.Var] = {}
        for i in range(wd.n_cases):
            dpost_t[i] = model.addVar(lb=float(wd.L_bounds[i]), ub=float(wd.U_bounds[i]), name=f"d_post_t{t}_i{i}")
            delta_rec = model.addVar(lb=float(wd.L_bounds[i] - wd.bookings[i]), ub=float(wd.U_bounds[i] - wd.bookings[i]), name=f"delta_rec_t{t}_i{i}")
            delta_post = model.addVar(lb=-GRB.INFINITY, name=f"delta_post_t{t}_i{i}")

            model.addConstr(delta_rec == quicksum(float(wd.features[i, j]) * w[j] for j in range(feat_dim)), name=f"delta_rec_link_t{t}_i{i}")

            sos = wd.sos2_data[i]
            lam = model.addVars(len(sos.knot_x), lb=0.0, name=f"lam_t{t}_i{i}")
            model.addConstr(quicksum(lam[k] for k in range(len(sos.knot_x))) == 1.0, name=f"lam_sum_t{t}_i{i}")
            model.addConstr(delta_rec == quicksum(float(sos.knot_x[k]) * lam[k] for k in range(len(sos.knot_x))), name=f"sos_x_t{t}_i{i}")
            model.addConstr(delta_post == quicksum(float(sos.knot_y[k]) * lam[k] for k in range(len(sos.knot_y))), name=f"sos_y_t{t}_i{i}")
            model.addSOS(GRB.SOS_TYPE2, [lam[k] for k in range(len(sos.knot_x))])
            model.addConstr(dpost_t[i] == float(wd.bookings[i]) + delta_post, name=f"d_post_def_t{t}_i{i}")

            e_plus = model.addVar(lb=0.0, name=f"e_plus_t{t}_i{i}")
            e_minus = model.addVar(lb=0.0, name=f"e_minus_t{t}_i{i}")
            model.addConstr(float(wd.realized[i]) - dpost_t[i] == e_plus - e_minus, name=f"cred_err_t{t}_i{i}")
            abs_err_terms.append(e_plus + e_minus)

        for i in range(wd.n_cases):
            elig = list(wd.case_eligible_blocks.get(i, []))
            if not elig:
                model.addConstr(r_t[i] == 1.0, name=f"force_defer_t{t}_i{i}")
                continue
            for bid in elig:
                z_t[i, bid] = model.addVar(vtype=GRB.BINARY, name=f"z_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}")
            model.addConstr(quicksum(z_t[i, bid] for bid in elig) + r_t[i] == 1.0, name=f"assign_t{t}_i{i}")
            for bid in elig:
                model.addConstr(z_t[i, bid] <= v_t[bid], name=f"zv_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}")

        mu = {}
        alpha_plus = model.addVars(blocks, lb=0.0, name=f"alpha_plus_t{t}")
        alpha_minus = model.addVars(blocks, lb=0.0, name=f"alpha_minus_t{t}")
        gamma_plus = model.addVars(blocks, lb=0.0, name=f"gamma_plus_t{t}")
        gamma_minus = model.addVars(blocks, lb=0.0, name=f"gamma_minus_t{t}")

        for bid in blocks:
            eligible_i = [i for i in range(wd.n_cases) if (i, bid) in z_t]
            m_bid = len(eligible_i)
            n_bid = quicksum(z_t[i, bid] for i in eligible_i) if eligible_i else 0.0
            if m_bid == 0:
                model.addConstr(v_t[bid] == 0.0, name=f"close_block_t{t}_{bid.day_index}_{bid.site}_{bid.room}")
                model.addConstr(y_t[bid] == 0.0, name=f"close_used_t{t}_{bid.day_index}_{bid.site}_{bid.room}")
            else:
                model.addConstr(n_bid <= m_bid * y_t[bid], name=f"use_up_t{t}_{bid.day_index}_{bid.site}_{bid.room}")
                model.addConstr(n_bid >= y_t[bid], name=f"use_low_t{t}_{bid.day_index}_{bid.site}_{bid.room}")
                model.addConstr(y_t[bid] <= v_t[bid], name=f"yv_t{t}_{bid.day_index}_{bid.site}_{bid.room}")

            mu_sum = 0.0
            for i in eligible_i:
                l_i = float(wd.L_bounds[i])
                u_i = float(wd.U_bounds[i])
                mu[i, bid] = model.addVar(lb=0.0, name=f"mu_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}")
                z = z_t[i, bid]
                d = dpost_t[i]
                model.addConstr(mu[i, bid] >= l_i * z)
                model.addConstr(mu[i, bid] <= u_i * z)
                model.addConstr(mu[i, bid] >= d - u_i * (1 - z))
                model.addConstr(mu[i, bid] <= d - l_i * (1 - z))
                mu_sum += mu[i, bid]

            pred_load = mu_sum + turnover * (n_bid - y_t[bid])
            real_load = quicksum(float(wd.realized[i]) * z_t[i, bid] for i in eligible_i) + turnover * (n_bid - y_t[bid])
            cap = next(b.capacity_minutes for b in wd.calendar.candidates if b.id == bid)
            model.addConstr(alpha_plus[bid] >= pred_load - cap * v_t[bid])
            model.addConstr(alpha_minus[bid] >= cap * v_t[bid] - pred_load)
            model.addConstr(gamma_plus[bid] >= real_load - cap * v_t[bid])
            model.addConstr(gamma_minus[bid] >= cap * v_t[bid] - real_load)

        activation = quicksum(next(b.activation_cost for b in wd.calendar.candidates if b.id == bid) * v_t[bid] for bid in blocks)
        deferral = costs.deferral_per_case * quicksum(r_t[i] for i in range(wd.n_cases))
        predicted_cost_exprs[t] = activation + deferral + costs.overtime_per_minute * quicksum(alpha_plus[bid] for bid in blocks) + costs.idle_per_minute * quicksum(alpha_minus[bid] for bid in blocks)
        realized_cost_exprs[t] = activation + deferral + costs.overtime_per_minute * quicksum(gamma_plus[bid] for bid in blocks) + costs.idle_per_minute * quicksum(gamma_minus[bid] for bid in blocks)

        for h, ref_col in enumerate(reference_sets.get(int(wd.week_index), [])):
            rhs = add_reference_predicted_cost_rhs(
                model=model,
                week_data=wd,
                d_post_vars=dpost_t,
                reference_schedule=ref_col,
                costs=costs,
                capacity_cfg=capacity_cfg,
                turnover=turnover,
                prefix=f"ref_t{t}_h{h}",
            )
            model.addConstr(predicted_cost_exprs[t] <= rhs, name=f"follower_cut_t{t}_h{h}")

        z_vars[t] = z_t
        r_vars[t] = r_t
        v_vars[t] = v_t
        y_vars[t] = y_t
        d_post_vars[t] = dpost_t

    n_cases_total = sum(wd.n_cases for wd in week_data_list)
    mae_base = _compute_mae_base(week_data_list)
    e_pred_max = config.vfcg.credibility_eta * mae_base
    if n_cases_total > 0:
        model.addConstr((1.0 / n_cases_total) * quicksum(abs_err_terms) <= e_pred_max + 1e-9, name="credibility")

    obj = (1.0 / max(1, n_weeks)) * quicksum(realized_cost_exprs[t] for t in range(n_weeks))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    status_name = _status_name(model.Status)
    no_incumbent = model.SolCount == 0
    bad_status = model.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD} or (model.Status == GRB.TIME_LIMIT and no_incumbent)

    if bad_status or no_incumbent:
        logger.warning("Native master returned no usable primal; using safe fallback (status=%s).", status_name)
        weights = np.zeros(feat_dim, dtype=float)
        schedules: dict[int, ScheduleColumn] = {}
        pred_costs: dict[int, float] = {}
        fallback_realized_costs: list[float] = []
        for wd in week_data_list:
            cases = build_case_records_from_week_data(wd)
            planning_durations = np.asarray(wd.bookings, dtype=float)
            realized_durations = np.asarray(wd.realized, dtype=float)
            col, predicted_cost, _, fallback_status, _ = solve_weekly_optimistic(
                cases=cases,
                planning_durations=planning_durations,
                realized_durations=realized_durations,
                calendar=wd.calendar,
                costs=costs,
                solver_cfg=solver_cfg,
                case_eligible_blocks=wd.case_eligible_blocks,
                turnover=turnover,
                model_name=f"fallback_week_{wd.week_index}",
                tol=config.vfcg.convergence_tol or 1e-6,
            )
            if col is None:
                raise RuntimeError(f"Fallback optimistic solve failed for week {wd.week_index} (status={fallback_status})")
            week_key = int(wd.week_index)
            schedules[week_key] = col
            pred_costs[week_key] = float(predicted_cost)
            fallback_realized_costs.append(float(col.compute_cost(realized_durations, costs, turnover)))

        fallback_objective = float(np.mean(fallback_realized_costs)) if fallback_realized_costs else float("inf")

        return NativeMasterResult(
            weights=weights,
            schedules_by_week=schedules,
            objective=fallback_objective,
            bound=fallback_objective,
            gap=float("nan"),
            solve_time=time.perf_counter() - t0,
            status=f"{status_name}_FALLBACK",
            predicted_costs_by_week=pred_costs,
            is_fallback=True,
        )

    weights = np.array([w[j].X for j in range(feat_dim)], dtype=float)
    schedules: dict[int, ScheduleColumn] = {}
    pred_costs: dict[int, float] = {}
    for t, wd in enumerate(week_data_list):
        col = extract_column_from_model(model, wd.n_cases, wd.calendar, z_vars[t], v_vars[t], y_vars[t], r_vars[t])
        schedules[int(wd.week_index)] = col
        pred_costs[int(wd.week_index)] = float(predicted_cost_exprs[t].getValue())

    gap = float(model.MIPGap) if hasattr(model, "MIPGap") else float("nan")
    return NativeMasterResult(
        weights=weights,
        schedules_by_week=schedules,
        objective=float(model.ObjVal),
        bound=float(model.ObjBound),
        gap=gap,
        solve_time=time.perf_counter() - t0,
        status=status_name,
        predicted_costs_by_week=pred_costs,
        is_fallback=False,
    )
