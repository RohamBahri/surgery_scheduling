"""Discrete displayed-duration restricted compact master for VFCG."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.core.column import ScheduleColumn, extract_column_from_model
from src.core.config import CapacityConfig, Config, CostConfig, SolverConfig
from src.estimation.recommendation import DiscreteDisplayCaseData, RecommendationModel, WeekRecommendationData
from src.solvers.deterministic import solve_weekly_optimistic
from src.vfcg.cuts import add_reference_predicted_cost_rhs
from src.vfcg.oracle import build_case_records_from_week_data

logger = logging.getLogger(__name__)


@dataclass
class NativeMasterResult:
    weights: np.ndarray
    schedules_by_week: dict[int, ScheduleColumn]
    objective: float
    realized_objective: float
    credibility_mae: float
    credibility_slack: float
    bound: float
    gap: float
    solve_time: float
    status: str
    predicted_costs_by_week: dict[int, float]
    is_fallback: bool = False


@dataclass
class NativeMasterStart:
    """Structured MIP start for the restricted compact master.

    The start must be feasible for the current cut set. In later VFCG
    iterations this is typically formed by keeping the previous master weights
    and replacing any violated-week schedule with that week's oracle-optimal
    schedule under those same weights.
    """

    weights: np.ndarray
    schedules_by_week: dict[int, ScheduleColumn]


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


def _build_default_master_start(
    week_data_list: list,
    reference_sets: dict[int, list[ScheduleColumn]],
    feat_dim: int,
) -> NativeMasterStart:
    """Construct the initial booking-based seed for the first master solve."""
    schedules: dict[int, ScheduleColumn] = {}
    for wd in week_data_list:
        week_key = int(wd.week_index)
        ref_list = reference_sets.get(week_key, [])
        if not ref_list:
            raise RuntimeError(f"No reference schedule available for week {week_key}.")
        schedules[week_key] = ref_list[0]
    return NativeMasterStart(weights=np.zeros(feat_dim, dtype=float), schedules_by_week=schedules)


def _project_weights_to_master_feasible_region(
    week_data_list: list[WeekRecommendationData],
    weights: np.ndarray,
    w_bound: float,
) -> np.ndarray:
    """Project a candidate warm-start weight vector onto the master-feasible region.

    The native master enforces, for every case, the raw linear recommendation
        delta_rec = x^T w
    together with the explicit bounds
        L - b <= delta_rec <= U - b.

    We therefore solve a small LP that finds the closest weight vector in L1
    distance subject to all master-feasibility constraints.
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    feat_dim = int(weights.shape[0])
    if feat_dim == 0 or not week_data_list:
        return weights.copy()

    proj = gp.Model("vfcg_warmstart_projection")
    proj.Params.OutputFlag = 0
    proj.Params.Threads = 1

    w_proj = proj.addVars(feat_dim, lb=-float(w_bound), ub=float(w_bound), name="w_proj")
    dev = proj.addVars(feat_dim, lb=0.0, name="dev")

    for j in range(feat_dim):
        proj.addConstr(w_proj[j] - float(weights[j]) <= dev[j], name=f"dev_pos_{j}")
        proj.addConstr(float(weights[j]) - w_proj[j] <= dev[j], name=f"dev_neg_{j}")

    for t, wd in enumerate(week_data_list):
        X = np.asarray(wd.features, dtype=float)
        if X.size == 0:
            continue
        lower = np.asarray(wd.L_bounds, dtype=float) - np.asarray(wd.bookings, dtype=float)
        upper = np.asarray(wd.U_bounds, dtype=float) - np.asarray(wd.bookings, dtype=float)
        for i in range(wd.n_cases):
            expr = quicksum(float(X[i, j]) * w_proj[j] for j in range(feat_dim))
            proj.addConstr(expr >= float(lower[i]), name=f"warm_lb_t{t}_i{i}")
            proj.addConstr(expr <= float(upper[i]), name=f"warm_ub_t{t}_i{i}")

    proj.setObjective(quicksum(dev[j] for j in range(feat_dim)), GRB.MINIMIZE)
    proj.optimize()

    if proj.SolCount == 0:
        logger.warning("Warm-start weight projection failed; reverting to zero weights.")
        return np.zeros(feat_dim, dtype=float)

    projected = np.array([w_proj[j].X for j in range(feat_dim)], dtype=float)
    return projected




def _compute_post_review_or_bookings(
    recommendation_model: RecommendationModel,
    weights: np.ndarray,
    wd: WeekRecommendationData,
) -> np.ndarray:
    if hasattr(recommendation_model, "compute_post_review"):
        return np.asarray(recommendation_model.compute_post_review(weights, wd), dtype=float)
    return np.asarray(wd.bookings, dtype=float)


def _repair_master_start_for_model_feasibility(
    start: NativeMasterStart,
    week_data_list: list[WeekRecommendationData],
    reference_sets: dict[int, list[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    costs: CostConfig,
    turnover: float,
    w_bound: float,
) -> NativeMasterStart:
    """Repair a warm start so that it is feasible for the native master model."""
    raw_weights = np.asarray(start.weights, dtype=float).reshape(-1)
    repaired_weights = _project_weights_to_master_feasible_region(
        week_data_list=week_data_list,
        weights=raw_weights,
        w_bound=w_bound,
    )

    if np.allclose(repaired_weights, raw_weights, atol=1e-9, rtol=0.0):
        return start

    logger.info(
        "Repaired warm-start weights to satisfy raw master recommendation bounds (max |Δw|=%.6f).",
        float(np.max(np.abs(repaired_weights - raw_weights))) if repaired_weights.size else 0.0,
    )

    repaired_schedules: dict[int, ScheduleColumn] = {}
    for wd in week_data_list:
        week_key = int(wd.week_index)
        ref_list = reference_sets.get(week_key, [])
        if not ref_list:
            raise RuntimeError(f"No reference schedules available for week {week_key}.")
        d_post = _compute_post_review_or_bookings(recommendation_model, repaired_weights, wd)
        repaired_schedules[week_key] = min(
            ref_list,
            key=lambda col: float(col.compute_cost(d_post, costs, turnover)),
        )

    return NativeMasterStart(
        weights=repaired_weights,
        schedules_by_week=repaired_schedules,
    )


def _singleton_display_data(wd: WeekRecommendationData, i: int) -> DiscreteDisplayCaseData:
    booking = float(wd.bookings[i])
    L_bound = float(wd.L_bounds[i])
    U_bound = float(wd.U_bounds[i])
    return DiscreteDisplayCaseData(
        case_index=i,
        profile_id=0,
        booking=booking,
        L_bound=L_bound,
        U_bound=U_bound,
        grid_values=np.asarray([booking], dtype=float),
        delta_rec_lb=np.asarray([L_bound - booking], dtype=float),
        delta_rec_ub=np.asarray([U_bound - booking], dtype=float),
    )


def _ensure_discrete_display_data(
    wd: WeekRecommendationData,
    recommendation_model: RecommendationModel,
) -> list[DiscreteDisplayCaseData]:
    discrete = getattr(wd, "discrete_display_data", [])
    if len(discrete) == wd.n_cases:
        return discrete
    if hasattr(recommendation_model, "build_discrete_display_data"):
        discrete = recommendation_model.build_discrete_display_data(wd)
        wd.discrete_display_data = discrete
        return discrete
    discrete = [_singleton_display_data(wd, i) for i in range(wd.n_cases)]
    wd.discrete_display_data = discrete
    return discrete


def _nearest_display_index(case_data: DiscreteDisplayCaseData, target_d_post: float) -> int:
    grid = np.asarray(case_data.grid_values, dtype=float)
    booking = float(case_data.booking)
    best_idx = 0
    best_key = (float("inf"), float("inf"), float("inf"))
    for idx, g_val in enumerate(grid):
        key = (abs(float(g_val) - target_d_post), abs(float(g_val) - booking), abs(float(g_val)))
        if key < best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _inject_structured_mip_start(
    model: gp.Model,
    w_vars,
    z_vars: dict,
    r_vars: dict,
    v_vars: dict,
    y_vars: dict,
    u_vars: dict,
    x_vars: dict,
    d_post_vars: dict,
    delta_rec_vars: dict,
    week_data_list: list,
    recommendation_model: RecommendationModel,
    feat_dim: int,
    start: NativeMasterStart,
) -> None:
    """Inject a structured, cut-feasible MIP start into the master."""
    _ = model
    weights = np.asarray(start.weights, dtype=float)
    if weights.shape != (feat_dim,):
        raise ValueError(f"Warm-start weight dimension mismatch: got {weights.shape}, expected {(feat_dim,)}")

    for j in range(feat_dim):
        w_vars[j].Start = float(weights[j])

    for t, wd in enumerate(week_data_list):
        week_key = int(wd.week_index)
        schedule = start.schedules_by_week.get(week_key)
        if schedule is None:
            raise ValueError(f"Warm start missing schedule for week {week_key}.")

        discrete_data = _ensure_discrete_display_data(wd, recommendation_model)
        delta_rec = np.asarray(wd.features, dtype=float) @ weights
        d_post = _compute_post_review_or_bookings(recommendation_model, weights, wd)

        for i in range(wd.n_cases):
            if i in delta_rec_vars.get(t, {}):
                delta_rec_vars[t][i].Start = float(delta_rec[i])
            if i in d_post_vars.get(t, {}):
                d_post_vars[t][i].Start = float(d_post[i])

            chosen_idx = _nearest_display_index(discrete_data[i], float(d_post[i]))
            for g_idx in range(len(discrete_data[i].grid_values)):
                u_var = u_vars.get(t, {}).get((i, g_idx))
                if u_var is not None:
                    u_var.Start = 1.0 if g_idx == chosen_idx else 0.0

        for i in range(wd.n_cases):
            is_deferred = i in schedule.z_defer
            assigned_bid = None
            if not is_deferred:
                for (ii, bid), val in schedule.z_assign.items():
                    if ii == i and val > 0.5:
                        assigned_bid = bid
                        break

            if i in r_vars.get(t, {}):
                r_vars[t][i].Start = 1.0 if is_deferred else 0.0

            chosen_idx = _nearest_display_index(discrete_data[i], float(d_post[i]))
            for (ii, bid), var in z_vars.get(t, {}).items():
                if ii != i:
                    continue
                var.Start = 0.0 if is_deferred else (1.0 if bid == assigned_bid else 0.0)

            for (ii, bid, g_idx), var in x_vars.get(t, {}).items():
                if ii != i:
                    continue
                var.Start = 1.0 if (not is_deferred and bid == assigned_bid and g_idx == chosen_idx) else 0.0

        for bid in wd.calendar.block_ids:
            if bid in v_vars.get(t, {}):
                v_vars[t][bid].Start = 1.0 if bid in schedule.v_open else 0.0
            if bid in y_vars.get(t, {}):
                y_vars[t][bid].Start = 1.0 if bid in schedule.y_used else 0.0

    logger.info("Structured MIP start injected for %d weeks.", len(week_data_list))


def solve_native_master(
    week_data_list: list[WeekRecommendationData],
    reference_sets: dict[int, list[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    config: Config,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    solver_cfg: SolverConfig,
    turnover: float,
    warm_start: NativeMasterStart | None = None,
) -> NativeMasterResult:
    t0 = time.perf_counter()
    n_weeks = len(week_data_list)
    feat_dim = recommendation_model.feature_dim if hasattr(recommendation_model, "feature_dim") else week_data_list[0].features.shape[1]

    model = gp.Model("vfcg_native_master")
    model.Params.TimeLimit = config.vfcg.master_time_limit
    model.Params.MIPGap = config.vfcg.master_mip_gap
    model.Params.Threads = solver_cfg.threads
    model.Params.OutputFlag = 1 if solver_cfg.verbose else 0
    model.Params.MIPFocus = 2
    model.Params.Symmetry = 2
    model.Params.Cuts = 2

    w = model.addVars(feat_dim, lb=-config.vfcg.w_max, ub=config.vfcg.w_max, name="w")

    z_vars: dict[int, dict[tuple[int, object], gp.Var]] = {}
    r_vars: dict[int, dict[int, gp.Var]] = {}
    v_vars: dict[int, dict[object, gp.Var]] = {}
    y_vars: dict[int, dict[object, gp.Var]] = {}
    u_vars: dict[int, dict[tuple[int, int], gp.Var]] = {}
    x_vars: dict[int, dict[tuple[int, object, int], gp.Var]] = {}
    d_post_vars: dict[int, dict[int, gp.Var]] = {}
    delta_rec_vars: dict[int, dict[int, gp.Var]] = {}

    predicted_cost_exprs: dict[int, gp.LinExpr] = {}
    realized_cost_exprs: dict[int, gp.LinExpr] = {}

    abs_err_terms = []

    for t, wd in enumerate(week_data_list):
        discrete_data = _ensure_discrete_display_data(wd, recommendation_model)
        blocks = list(wd.calendar.block_ids)
        z_t: dict[tuple[int, object], gp.Var] = {}
        r_t = model.addVars(wd.n_cases, vtype=GRB.BINARY, name=f"r_t{t}")
        v_t = model.addVars(blocks, vtype=GRB.BINARY, name=f"v_t{t}")
        y_t = model.addVars(blocks, vtype=GRB.BINARY, name=f"y_t{t}")
        u_t: dict[tuple[int, int], gp.Var] = {}
        x_t: dict[tuple[int, object, int], gp.Var] = {}

        dpost_t: dict[int, gp.Var] = {}
        delta_rec_t: dict[int, gp.Var] = {}
        for i in range(wd.n_cases):
            case_data = discrete_data[i]
            grid_vals = np.asarray(case_data.grid_values, dtype=float)
            global_lb = float(wd.L_bounds[i] - wd.bookings[i])
            global_ub = float(wd.U_bounds[i] - wd.bookings[i])

            dpost_t[i] = model.addVar(lb=float(np.min(grid_vals)), ub=float(np.max(grid_vals)), name=f"d_post_t{t}_i{i}")
            delta_rec = model.addVar(lb=global_lb, ub=global_ub, name=f"delta_rec_t{t}_i{i}")
            delta_rec_t[i] = delta_rec

            model.addConstr(
                delta_rec == quicksum(float(wd.features[i, j]) * w[j] for j in range(feat_dim)),
                name=f"delta_rec_link_t{t}_i{i}",
            )

            for g_idx, g_val in enumerate(grid_vals):
                u_var = model.addVar(vtype=GRB.BINARY, name=f"u_t{t}_i{i}_g{g_idx}")
                u_t[(i, g_idx)] = u_var

                pre_lb = float(case_data.delta_rec_lb[g_idx])
                pre_ub = float(case_data.delta_rec_ub[g_idx])
                lower_relax = max(0.0, pre_lb - global_lb)
                upper_relax = max(0.0, global_ub - pre_ub)
                model.addConstr(
                    delta_rec >= pre_lb - lower_relax * (1.0 - u_var),
                    name=f"display_lb_t{t}_i{i}_g{g_idx}",
                )
                model.addConstr(
                    delta_rec <= pre_ub + upper_relax * (1.0 - u_var),
                    name=f"display_ub_t{t}_i{i}_g{g_idx}",
                )

            model.addConstr(
                quicksum(u_t[(i, g_idx)] for g_idx in range(len(grid_vals))) == 1.0,
                name=f"display_pick_t{t}_i{i}",
            )
            model.addConstr(
                dpost_t[i] == quicksum(float(grid_vals[g_idx]) * u_t[(i, g_idx)] for g_idx in range(len(grid_vals))),
                name=f"d_post_def_t{t}_i{i}",
            )

            e_plus = model.addVar(lb=0.0, name=f"e_plus_t{t}_i{i}")
            e_minus = model.addVar(lb=0.0, name=f"e_minus_t{t}_i{i}")
            model.addConstr(float(wd.realized[i]) - dpost_t[i] == e_plus - e_minus, name=f"cred_err_t{t}_i{i}")
            abs_err_terms.append(e_plus + e_minus)

        for i in range(wd.n_cases):
            elig = list(wd.case_eligible_blocks.get(i, []))
            case_data = discrete_data[i]
            if not elig:
                model.addConstr(r_t[i] == 1.0, name=f"force_defer_t{t}_i{i}")
                continue

            for bid in elig:
                z_t[i, bid] = model.addVar(vtype=GRB.BINARY, name=f"z_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}")
                model.addConstr(z_t[i, bid] <= v_t[bid], name=f"zv_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}")

                assign_sum = []
                for g_idx in range(len(case_data.grid_values)):
                    x_var = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}_g{g_idx}",
                    )
                    x_t[(i, bid, g_idx)] = x_var
                    model.addConstr(
                        x_var <= u_t[(i, g_idx)],
                        name=f"xu_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}_g{g_idx}",
                    )
                    assign_sum.append(x_var)

                model.addConstr(
                    z_t[i, bid] == quicksum(assign_sum),
                    name=f"zx_t{t}_i{i}_{bid.day_index}_{bid.site}_{bid.room}",
                )

            model.addConstr(quicksum(z_t[i, bid] for bid in elig) + r_t[i] == 1.0, name=f"assign_t{t}_i{i}")

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

            pred_load = quicksum(
                float(discrete_data[i].grid_values[g_idx]) * x_t[(i, bid, g_idx)]
                for i in eligible_i
                for g_idx in range(len(discrete_data[i].grid_values))
            ) + turnover * (n_bid - y_t[bid])
            real_load = quicksum(float(wd.realized[i]) * z_t[i, bid] for i in eligible_i) + turnover * (n_bid - y_t[bid])
            cap = next(b.capacity_minutes for b in wd.calendar.candidates if b.id == bid)
            model.addConstr(alpha_plus[bid] >= pred_load - cap * v_t[bid])
            model.addConstr(alpha_minus[bid] >= cap * v_t[bid] - pred_load)
            model.addConstr(gamma_plus[bid] >= real_load - cap * v_t[bid])
            model.addConstr(gamma_minus[bid] >= cap * v_t[bid] - real_load)

        activation = quicksum(next(b.activation_cost for b in wd.calendar.candidates if b.id == bid) * v_t[bid] for bid in blocks)
        deferral = costs.deferral_per_case * quicksum(r_t[i] for i in range(wd.n_cases))
        predicted_cost_exprs[t] = (
            activation
            + deferral
            + costs.overtime_per_minute * quicksum(alpha_plus[bid] for bid in blocks)
            + costs.idle_per_minute * quicksum(alpha_minus[bid] for bid in blocks)
        )
        realized_cost_exprs[t] = (
            activation
            + deferral
            + costs.overtime_per_minute * quicksum(gamma_plus[bid] for bid in blocks)
            + costs.idle_per_minute * quicksum(gamma_minus[bid] for bid in blocks)
        )

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
        u_vars[t] = u_t
        x_vars[t] = x_t
        d_post_vars[t] = dpost_t
        delta_rec_vars[t] = delta_rec_t

    n_cases_total = sum(wd.n_cases for wd in week_data_list)
    mae_base = _compute_mae_base(week_data_list)
    e_pred_max = config.vfcg.credibility_eta * mae_base
    credibility_mae_expr = (
        (1.0 / n_cases_total) * quicksum(abs_err_terms)
        if n_cases_total > 0
        else gp.LinExpr(0.0)
    )
    credibility_slack_var: gp.Var | None = None
    if n_cases_total > 0:
        if config.vfcg.credibility_mode == "hard":
            model.addConstr(credibility_mae_expr <= e_pred_max + 1e-9, name="credibility")
        elif config.vfcg.credibility_mode == "elastic_penalty":
            credibility_slack_var = model.addVar(lb=0.0, name="credibility_slack")
            model.addConstr(
                credibility_mae_expr <= e_pred_max + credibility_slack_var + 1e-9,
                name="credibility_elastic",
            )
        else:
            raise ValueError(f"Unknown credibility_mode={config.vfcg.credibility_mode!r}")

    realized_obj = (1.0 / max(1, n_weeks)) * quicksum(realized_cost_exprs[t] for t in range(n_weeks))
    full_obj = realized_obj
    if credibility_slack_var is not None:
        full_obj = full_obj + float(config.vfcg.credibility_penalty_rho) * credibility_slack_var
    model.setObjective(full_obj, GRB.MINIMIZE)

    start_to_use = warm_start
    if start_to_use is None:
        start_to_use = _build_default_master_start(week_data_list, reference_sets, feat_dim)
    start_to_use = _repair_master_start_for_model_feasibility(
        start=start_to_use,
        week_data_list=week_data_list,
        reference_sets=reference_sets,
        recommendation_model=recommendation_model,
        costs=costs,
        turnover=turnover,
        w_bound=config.vfcg.w_max,
    )

    try:
        _inject_structured_mip_start(
            model=model,
            w_vars=w,
            z_vars=z_vars,
            r_vars=r_vars,
            v_vars=v_vars,
            y_vars=y_vars,
            u_vars=u_vars,
            x_vars=x_vars,
            d_post_vars=d_post_vars,
            delta_rec_vars=delta_rec_vars,
            week_data_list=week_data_list,
            recommendation_model=recommendation_model,
            feat_dim=feat_dim,
            start=start_to_use,
        )
    except Exception:
        if warm_start is not None:
            logger.exception("Provided warm start could not be injected; reverting to default booking-based seed.")
            _inject_structured_mip_start(
                model=model,
                w_vars=w,
                z_vars=z_vars,
                r_vars=r_vars,
                v_vars=v_vars,
                y_vars=y_vars,
                u_vars=u_vars,
                x_vars=x_vars,
                d_post_vars=d_post_vars,
                delta_rec_vars=delta_rec_vars,
                week_data_list=week_data_list,
                recommendation_model=recommendation_model,
                feat_dim=feat_dim,
                start=_build_default_master_start(week_data_list, reference_sets, feat_dim),
            )
        else:
            raise

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
        abs_error_total = 0.0
        total_cases = 0
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
            d_post = _compute_post_review_or_bookings(recommendation_model, weights, wd)
            abs_error_total += float(np.sum(np.abs(realized_durations - d_post)))
            total_cases += int(wd.n_cases)

        fallback_realized_objective = (
            float(np.mean(fallback_realized_costs)) if fallback_realized_costs else float("inf")
        )
        fallback_credibility_mae = abs_error_total / max(1, total_cases)
        if config.vfcg.credibility_mode == "elastic_penalty":
            fallback_credibility_slack = max(0.0, fallback_credibility_mae - e_pred_max)
            fallback_objective = (
                fallback_realized_objective
                + float(config.vfcg.credibility_penalty_rho) * fallback_credibility_slack
            )
        else:
            fallback_credibility_slack = 0.0
            fallback_objective = fallback_realized_objective

        return NativeMasterResult(
            weights=weights,
            schedules_by_week=schedules,
            objective=fallback_objective,
            realized_objective=fallback_realized_objective,
            credibility_mae=fallback_credibility_mae,
            credibility_slack=fallback_credibility_slack,
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
    realized_objective = float(realized_obj.getValue())
    credibility_mae = float(credibility_mae_expr.getValue()) if n_cases_total > 0 else 0.0
    credibility_slack = float(credibility_slack_var.X) if credibility_slack_var is not None else 0.0
    return NativeMasterResult(
        weights=weights,
        schedules_by_week=schedules,
        objective=float(model.ObjVal),
        realized_objective=realized_objective,
        credibility_mae=credibility_mae,
        credibility_slack=credibility_slack,
        bound=float(model.ObjBound),
        gap=gap,
        solve_time=time.perf_counter() - t0,
        status=status_name,
        predicted_costs_by_week=pred_costs,
        is_fallback=False,
    )
