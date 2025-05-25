import logging
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from integrated.sub_problem import solve_weekly_subproblem

class BendersThetaLearningCallback:
    """
    Lazy‐constraint callback for Benders decomposition of the integrated
    learning + optimization surgery‐scheduling model.

    On MIPSOL (when Gurobi finds a new integer feasible solution), this
    callback will:
      1. Read the current θ, Z, R, and η values from the master.
      2. For each scenario/week s, solve the LP subproblem to get Q_s and
         subgradient g_s.
      3. Add the optimality cut:
         η_s ≥ Q_s + ∑_{i,b} g_{i,b}·(Z_{i,b} - Z̄_{i,b})
    """

    def __init__(
        self,
        master_model: gp.Model,
        weekly_data: List[Tuple[Any, Any, Any]],
        Z_vars: Dict[Tuple[int,int,int], gp.Var],
        R_vars: Dict[Tuple[int,int],   gp.Var],
        theta_vars: Dict[int, gp.Var],
        eta_vars: Dict[int, gp.Var],
        df_pool_reference: Any,
        feature_names: List[str],
        params: Dict[str, Any],
    ):
        self.logger = logging.getLogger("BendersThetaCallback")
        self.model = master_model
        self.weekly_data = weekly_data
        self.Z_vars = Z_vars
        self.R_vars = R_vars
        self.theta_vars = theta_vars
        self.eta_vars = eta_vars
        self.df_pool = df_pool_reference
        self.feature_names = feature_names
        self.params = params

    def callback(self, model, where):
        # Only act when a new integer solution is found
        if where != GRB.Callback.MIPSOL:
            return

        # Pull current solution values
        theta_val = {j: model.cbGetSolution(var)
                     for j, var in self.theta_vars.items()}

        # For each scenario s = 0..|S|-1
        for s, (surgeries, blocks, actual_map) in enumerate(self.weekly_data):
            # Extract Z̄ and R̄ for scenario s
            Z_bar = { (i,b): model.cbGetSolution(self.Z_vars[(s,i,b)])
                      for (ss,i,b) in self.Z_vars if ss==s }
            R_bar = { i: model.cbGetSolution(self.R_vars[(s,i)])
                      for (ss,i) in self.R_vars if ss==s }

            # Solve subproblem to get Q_s and g_{i,b}
            Q_s, grad = solve_weekly_subproblem(
                surgeries_info         = surgeries,
                daily_blocks_info      = blocks,
                theta_values_by_index  = theta_val,
                Z_bar_assignments      = Z_bar,
                R_bar_rejections       = R_bar,
                df_pool_reference      = self.df_pool,
                ordered_feature_names  = self.feature_names,
                params_config          = self.params,
            )
            # Build left‐ and right‐hand sides of the cut
            eta_var = self.eta_vars[s]
            lhs = gp.LinExpr(eta_var)
            rhs = Q_s
            for (i,b), g_ib in grad.items():
                z_var = self.Z_vars[(s,i,b)]
                rhs += g_ib * (z_var - Z_bar[(i,b)])

            # Add the lazy optimality cut
            model.cbLazy(lhs >= rhs)
            self.logger.debug(f"Added Benders cut for scenario {s}: η_{s} ≥ {rhs}")

