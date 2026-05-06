from __future__ import annotations

import gurobipy as gp
import numpy as np

from src.core.column import ScheduleColumn
from src.core.config import CapacityConfig, CostConfig
from src.core.types import BlockCalendar, CandidateBlock
from src.estimation.recommendation import SOS2CaseData, WeekRecommendationData
from src.vfcg.cuts import add_reference_predicted_cost_rhs


def _week() -> WeekRecommendationData:
    b1 = CandidateBlock(0, "TGH", "OR1", 100.0, 30.0)
    b2 = CandidateBlock(0, "TGH", "OR2", 120.0, 40.0)
    return WeekRecommendationData(
        week_index=0,
        n_cases=2,
        features=np.zeros((2, 1), dtype=float),
        bookings=np.array([90.0, 60.0]),
        realized=np.array([95.0, 70.0]),
        L_bounds=np.array([60.0, 40.0]),
        U_bounds=np.array([140.0, 120.0]),
        surgeon_codes=["S1", "S2"],
        sos2_data=[
            SOS2CaseData(0, 0, np.array([-10.0, 0.0, 10.0]), np.array([-5.0, 0.0, 5.0]), 90.0, 60.0, 140.0),
            SOS2CaseData(1, 0, np.array([-10.0, 0.0, 10.0]), np.array([-5.0, 0.0, 5.0]), 60.0, 40.0, 120.0),
        ],
        case_eligible_blocks={0: [b1.id], 1: [b1.id, b2.id]},
        calendar=BlockCalendar([b1, b2]),
    )


def test_rhs_matches_analytic_reference_cost_for_fixed_d_post() -> None:
    wd = _week()
    bid = wd.calendar.candidates[0].id
    ref = ScheduleColumn(
        z_assign={(0, bid): 1.0, (1, bid): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({bid}),
        y_used=frozenset({bid}),
        n_cases=2,
        block_capacities={b.id: b.capacity_minutes for b in wd.calendar.candidates},
        block_activation_costs={b.id: b.activation_cost for b in wd.calendar.candidates},
    )

    model = gp.Model("cuts_test_rhs")
    model.Params.OutputFlag = 0
    d0 = model.addVar(lb=0.0, ub=200.0, name="d0")
    d1 = model.addVar(lb=0.0, ub=200.0, name="d1")
    model.addConstr(d0 == 100.0)
    model.addConstr(d1 == 80.0)

    rhs = add_reference_predicted_cost_rhs(
        model=model,
        week_data=wd,
        d_post_vars={0: d0, 1: d1},
        reference_schedule=ref,
        costs=CostConfig(overtime_per_minute=2.0, idle_per_minute=1.0, deferral_per_case=1000.0),
        capacity_cfg=CapacityConfig(),
        turnover=0.0,
        prefix="rhs",
    )
    model.setObjective(rhs, gp.GRB.MINIMIZE)
    model.optimize()

    analytic = ref.compute_cost(np.array([100.0, 80.0]), CostConfig(overtime_per_minute=2.0, idle_per_minute=1.0, deferral_per_case=1000.0), 0.0)
    assert np.isclose(model.ObjVal, analytic)


def test_rhs_handles_deferrals_and_unused_open_blocks() -> None:
    wd = _week()
    b1 = wd.calendar.candidates[0].id
    b2 = wd.calendar.candidates[1].id
    ref = ScheduleColumn(
        z_assign={(0, b1): 1.0},
        z_defer=frozenset({1}),
        v_open=frozenset({b1, b2}),
        y_used=frozenset({b1}),
        n_cases=2,
        block_capacities={b.id: b.capacity_minutes for b in wd.calendar.candidates},
        block_activation_costs={b.id: b.activation_cost for b in wd.calendar.candidates},
    )

    model = gp.Model("cuts_test_defer")
    model.Params.OutputFlag = 0
    d0 = model.addVar(lb=0.0, ub=200.0, name="d0")
    d1 = model.addVar(lb=0.0, ub=200.0, name="d1")
    model.addConstr(d0 == 100.0)
    model.addConstr(d1 == 80.0)

    rhs = add_reference_predicted_cost_rhs(
        model=model,
        week_data=wd,
        d_post_vars={0: d0, 1: d1},
        reference_schedule=ref,
        costs=CostConfig(overtime_per_minute=2.0, idle_per_minute=1.0, deferral_per_case=100.0),
        capacity_cfg=CapacityConfig(),
        turnover=0.0,
        prefix="rhs2",
    )
    model.setObjective(rhs, gp.GRB.MINIMIZE)
    model.optimize()

    analytic = ref.compute_cost(np.array([100.0, 80.0]), CostConfig(overtime_per_minute=2.0, idle_per_minute=1.0, deferral_per_case=100.0), 0.0)
    assert np.isclose(model.ObjVal, analytic)


def test_unopened_reference_blocks_create_no_auxiliaries() -> None:
    wd = _week()
    b1 = wd.calendar.candidates[0].id
    ref = ScheduleColumn(
        z_assign={(0, b1): 1.0, (1, b1): 1.0},
        z_defer=frozenset(),
        v_open=frozenset({b1}),
        y_used=frozenset({b1}),
        n_cases=2,
        block_capacities={b.id: b.capacity_minutes for b in wd.calendar.candidates},
        block_activation_costs={b.id: b.activation_cost for b in wd.calendar.candidates},
    )

    model = gp.Model("cuts_test_aux")
    model.Params.OutputFlag = 0
    d0 = model.addVar(lb=0.0, ub=200.0, name="d0")
    d1 = model.addVar(lb=0.0, ub=200.0, name="d1")
    _ = add_reference_predicted_cost_rhs(
        model=model,
        week_data=wd,
        d_post_vars={0: d0, 1: d1},
        reference_schedule=ref,
        costs=CostConfig(),
        capacity_cfg=CapacityConfig(),
        turnover=0.0,
        prefix="rhs3",
    )
    model.update()

    names = [v.VarName for v in model.getVars()]
    assert any("OR1" in n for n in names)
    assert not any("OR2" in n for n in names)
