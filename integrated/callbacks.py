import logging
from typing import Any, Dict, List, Tuple # Ensure all are imported

import gurobipy as gp
from gurobipy import GRB
import pandas as pd # Keep for type hinting df_pool_reference

from src.constants import DEFAULT_LOGGER_NAME
# solve_weekly_subproblem is imported by run_integrated and passed as fn

logger = logging.getLogger(DEFAULT_LOGGER_NAME)

gurobipy_internal_logger = logging.getLogger('gurobipy')
gurobipy_internal_logger.setLevel(logging.WARNING)

# Type alias for the weekly data tuple passed from run_integrated.py
# (list_of_surgeries_for_week, dict_of_daily_block_capacities, dict_of_actual_durations_map)
EnhancedWeeklyDataTypeForCallback = Tuple[List[Dict[str, Any]], Dict[int, int], Dict[Any, float]]


class BendersThetaLearningCallback:
    """
    Lazy-constraint callback for integrated Benders (GTSM-Benders):
      - Fires at each integer solution of the master (where==MIPSOL)
      - Extracts (theta, Z_bar, R_bar) from the master for each week
      - Calls the weekly subproblem (LP) to get Q_s (OT/IT cost) and subgradients g_it
      - Adds one Benders cut per week to the master:
          eta[s] >= Q_s + sum_{i,b} g_{i,b}(Z[s,i,b] - Z_bar[s,i,b])
    """

    def __init__(
        self,
        weekly_data_list: List[EnhancedWeeklyDataTypeForCallback], # Use the specific type
        Z_master_vars: Dict[Tuple[int, int, int], gp.Var], # (s, i_in_s, b_flat) -> Var
        R_master_vars: Dict[Tuple[int, int], gp.Var],     # (s, i_in_s) -> Var
        theta_master_vars: Dict[int, gp.Var],             # (j_feature_idx) -> Var
        eta_master_vars: Dict[int, gp.Var],               # (s_week_idx) -> Var
        solve_weekly_subproblem_fn, # Function from sub_problem.py
        df_pool_reference: pd.DataFrame,
        ordered_feature_names: List[str], # Semantic feature names
        params_config: Dict[str, Any],    # PARAMS
    ):
        self.weekly_data_list = weekly_data_list
        self.Z_master_vars = Z_master_vars
        self.R_master_vars = R_master_vars
        self.theta_master_vars = theta_master_vars
        self.eta_master_vars = eta_master_vars
        self._solve_weekly_lp_subproblem = solve_weekly_subproblem_fn # Store the function

        # Store these for passing to the subproblem
        self.df_pool_reference = df_pool_reference
        self.ordered_feature_names = ordered_feature_names
        self.params_config = params_config
        
        self.callback_iteration_count = 0
        self.total_cuts_added = 0
        logger.debug("BendersThetaLearningCallback (GTSM) initialized.")

    def __call__(self, model: gp.Model, where: int):
        if where != GRB.Callback.MIPSOL:
            return

        self.callback_iteration_count += 1
        logger.info(f"Benders GTSM Callback: MIPSOL Iteration #{self.callback_iteration_count}")

        # 1) Extract current global theta values (integer indexed) from master
        current_theta_values_by_index: Dict[int, float] = {
            j: model.cbGetSolution(self.theta_master_vars[j]) for j in self.theta_master_vars
        }
        # Can log this if very verbose debugging is needed, but it's a lot of data
        # logger.debug(f"  Callback: Current global theta_val: {current_theta_values_by_index}")

        cuts_added_this_master_iteration = 0

        # 2) Loop through each historical week s to solve its subproblem
        for s_week_idx, weekly_data_item in enumerate(self.weekly_data_list):
            surgeries_in_week, daily_blocks_in_week, actual_durations_map = weekly_data_item
            num_surgeries_in_this_week = len(surgeries_in_week)
            num_flat_blocks_this_week = sum(daily_blocks_in_week.values())

            # Extract Z_bar^(s) and R_bar^(s) for this week s from the master solution
            # Z_bar_assignments_s: (surg_idx_in_week, flat_block_idx) -> 0 or 1
            # R_bar_rejections_s: (surg_idx_in_week) -> 0 or 1
            
            Z_bar_s: Dict[Tuple[int, int], int] = {}
            for i_idx_in_week in range(num_surgeries_in_this_week):
                for b_flat_idx in range(num_flat_blocks_this_week):
                    # Master Z var key is (s_week_idx, i_idx_in_week, b_flat_idx)
                    master_z_var = self.Z_master_vars.get((s_week_idx, i_idx_in_week, b_flat_idx))
                    if master_z_var: # Should always exist if dicts are built correctly
                        Z_bar_s[(i_idx_in_week, b_flat_idx)] = int(round(model.cbGetSolution(master_z_var)))


            R_bar_s: Dict[int, int] = {}
            for i_idx_in_week in range(num_surgeries_in_this_week):
                master_r_var = self.R_master_vars.get((s_week_idx, i_idx_in_week))
                if master_r_var:
                    R_bar_s[i_idx_in_week] = int(round(model.cbGetSolution(master_r_var)))
            
            # logger.debug(
            #     f"  Callback Week {s_week_idx}: Extracted Z_bar_s (sum={sum(Z_bar_s.values())}), "
            #     f"R_bar_s (sum={sum(R_bar_s.values())})"
            # )

            # 3) Solve the LP subproblem for week s
            try:
                # Call the subproblem function (solve_weekly_subproblem)
                # It now takes all necessary arguments
                Q_s_cost, subgradients_g_it = self._solve_weekly_lp_subproblem(
                    surgeries_info=surgeries_in_week,
                    daily_blocks_info=daily_blocks_in_week,
                    theta_values_by_index=current_theta_values_by_index,
                    Z_bar_assignments=Z_bar_s,
                    R_bar_rejections=R_bar_s, # Passed but not used by current subproblem LP
                    df_pool_reference=self.df_pool_reference,
                    ordered_feature_names=self.ordered_feature_names,
                    params_config=self.params_config
                )
            except Exception as e:
                logger.error(f"Callback: Subproblem for week {s_week_idx} solver call failed: {e}", exc_info=True)
                continue # Skip cut for this week

            # 4) Check if cut is needed and add it
            current_eta_s_value = model.cbGetSolution(self.eta_master_vars[s_week_idx])
            
            # The RHS of the cut is Q_s + sum g_it * (Z_it - Z_bar_it)
            # For cut violation check, we need: current_eta_s < Q_s
            # because at Z_bar_it, the sum term is zero.
            # If Q_s itself is already lower than current_eta_s, then the full cut might not be violated
            # if the sum term is negative enough. Standard check: if current_eta < Q_s.
            # Or more precisely, current_eta_s < evaluated_rhs_of_cut_at_Z_bar_s (which is Q_s)

            if Q_s_cost == float("inf"):
                logger.warning(
                    f"Callback Week {s_week_idx}: Subproblem infeasible for current (theta, Z_bar_s, R_bar_s). "
                    "This implies an issue. Adding a feasibility cut (eta_s >= large_number)."
                )
                # A very large number, or handle with proper Farkas certificate if subproblem provides it.
                # For now, making eta very large effectively penalizes this master solution.
                model.cbLazy(self.eta_master_vars[s_week_idx] >= 1e9) # Or some M constant
                cuts_added_this_master_iteration += 1
                self.total_cuts_added +=1
                continue # Move to next week

            # Build the cut expression: eta_s >= Q_s + sum g_it (Z_it - Z_bar_it)
            cut_rhs_expr = gp.LinExpr(Q_s_cost) # Initialize with the constant Q_s
            
            # subgradients_g_it keys are (i_idx_in_week, b_flat_idx)
            for (i_idx_in_week, b_flat_idx), g_value in subgradients_g_it.items():
                master_z_var = self.Z_master_vars.get((s_week_idx, i_idx_in_week, b_flat_idx))
                if master_z_var: # Should always be true
                    # Z_bar_s key is (i_idx_in_week, b_flat_idx)
                    z_bar_value = Z_bar_s.get((i_idx_in_week, b_flat_idx), 0) # Default to 0 if somehow missing
                    cut_rhs_expr.addTerms(g_value, master_z_var)
                    cut_rhs_expr.addConstant(-g_value * z_bar_value)
            
            # Only add cut if violated: current_eta_s < evaluated_rhs_at_current_Z_master
            # For the violation check, we evaluate the RHS at the *current master solution Z_master_vars*,
            # which means Z_it - Z_bar_it terms are zero. So, check if current_eta_s < Q_s_cost.
            cut_violation_tolerance = self.params_config.get("benders_cut_tolerance", 1e-4)
            if current_eta_s_value < Q_s_cost - cut_violation_tolerance:
                model.cbLazy(self.eta_master_vars[s_week_idx] >= cut_rhs_expr)
                cuts_added_this_master_iteration += 1
                self.total_cuts_added += 1
                logger.debug(
                    f"Callback Week {s_week_idx}: Added Benders cut. "
                    f"eta_{s_week_idx} ({current_eta_s_value:.2f}) >= Q_s({Q_s_cost:.2f}) + sum(g*(Z-Z_bar))."
                )
        
        if cuts_added_this_master_iteration > 0:
            logger.info(
                f"Benders GTSM Callback: MIPSOL Iteration #{self.callback_iteration_count} - "
                f"Added {cuts_added_this_master_iteration} Benders cut(s). "
                f"Total cuts generated so far: {self.total_cuts_added}."
            )
        else:
            logger.info(
                 f"Benders GTSM Callback: MIPSOL Iteration #{self.callback_iteration_count} - "
                 f"No cuts violated for any week with current (theta, Z, R). Solution may be optimal."
            )