"""Follower-optimality cut helpers for VFCG."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, CostConfig
from src.estimation.recommendation import WeekRecommendationData


def compute_reference_big_m(
    week_data: WeekRecommendationData,
    turnover: float,
) -> float:
    max_case = float(max(week_data.U_bounds)) if week_data.n_cases > 0 else 0.0
    max_cases_block = max((len(v) for v in week_data.case_eligible_blocks.values()), default=1)
    max_capacity = max((b.capacity_minutes for b in week_data.calendar.candidates), default=0.0)
    return max_case * max(1, max_cases_block) + abs(turnover) * week_data.n_cases + max_capacity


def add_reference_predicted_cost_rhs(
    model: gp.Model,
    week_data: WeekRecommendationData,
    d_post_vars: dict[int, gp.Var],
    reference_schedule: ScheduleColumn,
    costs: CostConfig,
    capacity_cfg: CapacityConfig,
    turnover: float,
    prefix: str,
) -> LinExpr:
    _ = capacity_cfg
    _ = week_data

    activation_const = sum(reference_schedule.block_activation_costs.get(bid, 0.0) for bid in reference_schedule.v_open)
    deferral_const = costs.deferral_per_case * len(reference_schedule.z_defer)

    overtime_vars = []
    idle_vars = []

    for bid in reference_schedule.v_open:
        assigned_cases = reference_schedule.cases_in_block(bid)
        fixed_used = 1.0 if bid in reference_schedule.y_used else 0.0

        reference_load = quicksum(d_post_vars[i] for i in assigned_cases) + turnover * (len(assigned_cases) - fixed_used)
        capacity_val = float(reference_schedule.block_capacities[bid])

        ot = model.addVar(lb=0.0, name=f"{prefix}_ot_{bid.day_index}_{bid.site}_{bid.room}")
        idle = model.addVar(lb=0.0, name=f"{prefix}_idle_{bid.day_index}_{bid.site}_{bid.room}")
        overtime_vars.append(ot)
        idle_vars.append(idle)

        over_expr = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_over_expr_{bid.day_index}_{bid.site}_{bid.room}")
        idle_expr = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_idle_expr_{bid.day_index}_{bid.site}_{bid.room}")
        model.addConstr(over_expr == reference_load - capacity_val, name=f"{prefix}_over_link_{bid.day_index}_{bid.site}_{bid.room}")
        model.addConstr(idle_expr == capacity_val - reference_load, name=f"{prefix}_idle_link_{bid.day_index}_{bid.site}_{bid.room}")

        model.addGenConstrMax(ot, [over_expr], constant=0.0, name=f"{prefix}_ot_max_{bid.day_index}_{bid.site}_{bid.room}")
        model.addGenConstrMax(idle, [idle_expr], constant=0.0, name=f"{prefix}_idle_max_{bid.day_index}_{bid.site}_{bid.room}")

    return (
        activation_const
        + deferral_const
        + costs.overtime_per_minute * quicksum(overtime_vars)
        + costs.idle_per_minute * quicksum(idle_vars)
    )
