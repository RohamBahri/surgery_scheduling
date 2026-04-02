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
    A regression-derived seed can violate those bounds, in which case its MIP
    start is invalid even if the *clipped* recommendation pipeline is feasible.

    We therefore solve a small LP that finds the closest weight vector in L1
    distance subject to all master-feasibility constraints. Since w=0 is always
    feasible in this model family, this projection should always succeed.
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


def _repair_master_start_for_model_feasibility(
    start: NativeMasterStart,
    week_data_list: list[WeekRecommendationData],
    reference_sets: dict[int, list[ScheduleColumn]],
    recommendation_model: RecommendationModel,
    costs: CostConfig,
    turnover: float,
    w_bound: float,
) -> NativeMasterStart:
    """Repair a warm start so that it is feasible for the native master model.

    If the proposed weights violate any raw master recommendation bound
    constraint, we project them onto the master-feasible region. When that
    happens, we also refresh the per-week schedule seed by selecting, from the
    current reference set, the cheapest reference schedule under the repaired
    weights. This guarantees satisfaction of the follower-cut bundle for the
    repaired start.
    """
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
        d_post = np.asarray(recommendation_model.compute_post_review(repaired_weights, wd), dtype=float)
        repaired_schedules[week_key] = min(
            ref_list,
            key=lambda col: float(col.compute_cost(d_post, costs, turnover)),
        )

    return NativeMasterStart(
        weights=repaired_weights,
        schedules_by_week=repaired_schedules,
    )


def _set_sos2_lambda_start(lam_dict, knot_x: np.ndarray, target_x: float, tol: float = 1e-9) -> None:
    """Seed an SOS2 representation for a given scalar target.

    The representation uses either a single knot or a convex combination of two
    adjacent knots so that the model's SOS2 equalities are respected by the
    start.
    """
    xs = np.asarray(knot_x, dtype=float)
    n_knots = len(xs)

    for k in range(n_knots):
        lam_dict[k].Start = 0.0

    if n_knots == 0:
        return
    if n_knots == 1:
        lam_dict[0].Start = 1.0
        return

    if target_x <= xs[0] + tol:
        lam_dict[0].Start = 1.0
        return
    if target_x >= xs[-1] - tol:
        lam_dict[n_knots - 1].Start = 1.0
        return

    exact = np.where(np.isclose(xs, target_x, atol=tol, rtol=0.0))[0]
    if len(exact) > 0:
        lam_dict[int(exact[0])].Start = 1.0
        return

    right = int(np.searchsorted(xs, target_x, side="left"))
    left = max(0, right - 1)

    x_left = float(xs[left])
    x_right = float(xs[right])
    if abs(x_right - x_left) <= tol:
        lam_dict[left].Start = 1.0
        return

    theta = (float(target_x) - x_left) / (x_right - x_left)
    theta = min(1.0, max(0.0, theta))
    lam_dict[left].Start = 1.0 - theta
    lam_dict[right].Start = theta


def _inject_structured_mip_start(
    model: gp.Model,
    w_vars,
    z_vars: dict,
    r_vars: dict,
    v_vars: dict,
    y_vars: dict,
    d_post_vars: dict,
    delta_rec_vars: dict,
    delta_post_vars: dict,
    lam_vars: dict,
    week_data_list: list,
    recommendation_model: RecommendationModel,
    feat_dim: int,
    start: NativeMasterStart,
) -> None:
    """Inject a structured, cut-feasible MIP start into the master.

    The supplied schedules must already be feasible for the current follower-cut
    set. Later VFCG iterations should therefore pass the previous master weights
    together with oracle-optimal schedules on newly violated weeks.
    """
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

        delta_rec = np.asarray(wd.features, dtype=float) @ weights
        d_post = np.asarray(recommendation_model.compute_post_review(weights, wd), dtype=float)
        delta_post = d_post - np.asarray(wd.bookings, dtype=float)

        for i in range(wd.n_cases):
            if i in delta_rec_vars.get(t, {}):
                delta_rec_vars[t][i].Start = float(delta_rec[i])
            if i in delta_post_vars.get(t, {}):
                delta_post_vars[t][i].Start = float(delta_post[i])
            if i in d_post_vars.get(t, {}):
                d_post_vars[t][i].Start = float(d_post[i])
            lam_dict = lam_vars.get(t, {}).get(i)
            if lam_dict is not None:
                _set_sos2_lambda_start(lam_dict, wd.sos2_data[i].knot_x, float(delta_rec[i]))

        for i in range(wd.n_cases):
            is_deferred = i in schedule.z_defer
            if i in r_vars.get(t, {}):
                r_vars[t][i].Start = 1.0 if is_deferred else 0.0
            for (ii, bid), var in z_vars.get(t, {}).items():
                if ii != i:
                    continue
                var.Start = 0.0 if is_deferred else (1.0 if schedule.z_assign.get((i, bid), 0.0) > 0.5 else 0.0)

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
    model.Params.MIPFocus = 1 
    model.Params.Symmetry = 2

    w = model.addVars(feat_dim, lb=-config.vfcg.w_max, ub=config.vfcg.w_max, name="w")

    z_vars: dict[int, dict[tuple[int, object], gp.Var]] = {}
    r_vars: dict[int, dict[int, gp.Var]] = {}
    v_vars: dict[int, dict[object, gp.Var]] = {}
    y_vars: dict[int, dict[object, gp.Var]] = {}
    d_post_vars: dict[int, dict[int, gp.Var]] = {}
    delta_rec_vars: dict[int, dict[int, gp.Var]] = {}
    delta_post_vars: dict[int, dict[int, gp.Var]] = {}

    lam_vars: dict[int, dict[int, object]] = {}

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
        delta_rec_t: dict[int, gp.Var] = {}
        delta_post_t: dict[int, gp.Var] = {}
        for i in range(wd.n_cases):
            dpost_t[i] = model.addVar(lb=float(wd.L_bounds[i]), ub=float(wd.U_bounds[i]), name=f"d_post_t{t}_i{i}")
            delta_rec = model.addVar(lb=float(wd.L_bounds[i] - wd.bookings[i]), ub=float(wd.U_bounds[i] - wd.bookings[i]), name=f"delta_rec_t{t}_i{i}")
            delta_post = model.addVar(lb=-GRB.INFINITY, name=f"delta_post_t{t}_i{i}")
            delta_rec_t[i] = delta_rec
            delta_post_t[i] = delta_post

            model.addConstr(delta_rec == quicksum(float(wd.features[i, j]) * w[j] for j in range(feat_dim)), name=f"delta_rec_link_t{t}_i{i}")

            sos = wd.sos2_data[i]
            lam = model.addVars(len(sos.knot_x), lb=0.0, name=f"lam_t{t}_i{i}")
            model.addConstr(quicksum(lam[k] for k in range(len(sos.knot_x))) == 1.0, name=f"lam_sum_t{t}_i{i}")
            model.addConstr(delta_rec == quicksum(float(sos.knot_x[k]) * lam[k] for k in range(len(sos.knot_x))), name=f"sos_x_t{t}_i{i}")
            model.addConstr(delta_post == quicksum(float(sos.knot_y[k]) * lam[k] for k in range(len(sos.knot_y))), name=f"sos_y_t{t}_i{i}")
            model.addSOS(GRB.SOS_TYPE2, [lam[k] for k in range(len(sos.knot_x))])

            lam_vars.setdefault(t, {})[i] = lam

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
        delta_rec_vars[t] = delta_rec_t
        delta_post_vars[t] = delta_post_t

    n_cases_total = sum(wd.n_cases for wd in week_data_list)
    mae_base = _compute_mae_base(week_data_list)
    e_pred_max = config.vfcg.credibility_eta * mae_base
    if n_cases_total > 0:
        model.addConstr((1.0 / n_cases_total) * quicksum(abs_err_terms) <= e_pred_max + 1e-9, name="credibility")

    obj = (1.0 / max(1, n_weeks)) * quicksum(realized_cost_exprs[t] for t in range(n_weeks))
    model.setObjective(obj, GRB.MINIMIZE)

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
            d_post_vars=d_post_vars,
            delta_rec_vars=delta_rec_vars,
            delta_post_vars=delta_post_vars,
            lam_vars=lam_vars,
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
                d_post_vars=d_post_vars,
                delta_rec_vars=delta_rec_vars,
                delta_post_vars=delta_post_vars,
                lam_vars=lam_vars,
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
