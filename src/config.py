"""
Configuration parameters for the surgery scheduling application.

This dictionary, PARAMS, stores settings that control various aspects of the
application, from data loading and processing to model training and solving.
These parameters can be adjusted to experiment with different scenarios or
optimize performance.
"""

PARAMS: dict[str, any] = {
    "debug_mode": False,  # If True, enables faster, less comprehensive runs.
    # -------------------------------------------------------------------------
    # File Paths & Temporal Split Settings
    # -------------------------------------------------------------------------
    "excel_file_path": "data/UHNOperating_RoomScheduling2011-2013.xlsx",
    "output_file": "outputs/results.json",  # For detailed per-horizon results
    "aggregated_output_file": "outputs/agg_results.json",  # For summary results
    "theta_path": "outputs/final_theta.json",  # For integrated model coefficients
    "warmup_weeks": 52,  # Duration of the warm-up period in weeks
    "planning_horizon_days": 7,  # Duration of each planning horizon in days
    "NUM_HORIZONS": 20,  # Number of planning horizons to simulate
    # -------------------------------------------------------------------------
    # Operating Room Calendar & Capacity Settings
    # -------------------------------------------------------------------------
    "or_day_start_hour": 8,  # Hour (0-23) when ORs open (e.g., 8 for 08:00)
    "or_day_end_hour": 16,  # Hour (0-23) when ORs close for planning (e.g., 16 for 16:00)
    "block_size_minutes": 480,  # Duration of one OR block in minutes (e.g., 480 for 8 hours)
    "blocks_per_day": 1,  # Number of blocks per OR per day (typically 1 for 8h blocks)
    "capacity_reduction_percentage": 0.6,  # Factor to reduce historical capacity (0.0 to 1.0)
    "min_blocks_after_reduction": 1,  # Min blocks per day after reduction
    "min_procedure_duration": 1,  # Minimum procedure duration in minutes
    "max_procedure_duration": 480,  # Maximum procedure duration in minutes
    # -------------------------------------------------------------------------
    # Cost Coefficients for Optimization
    # -------------------------------------------------------------------------
    "cost_rejection_per_case": 20.0,  # Cost incurred for rejecting a surgery case
    "cost_overtime_per_min": 15.0,  # Cost per minute of OR overtime
    "cost_idle_per_min": 10.0,  # Cost per minute of OR idle time
    # -------------------------------------------------------------------------
    # Sampling / SAA & Machine Learning Hyperparameters
    # -------------------------------------------------------------------------
    "run_saa": True,  # If True, runs the Sample Average Approximation model
    "saa_scenarios": 100,  # Number of scenarios for SAA
    "saa_random_seed": 42,  # Random seed for SAA scenario generation
    "knn_neighbors": 5,  # Default number of neighbors for KNN predictor
    "knn_k_options": [3, 5, 7, 9, 11],  # Grid of k values for KNN tuning
    "lasso_alphas": [0.1, 0.5, 1.0, 5.0],  # Grid of alpha values for LassoCV
    "lasso_alpha_asym": 0.5,  # Alpha for asymmetric Lasso (quantile regressor)
    # -------------------------------------------------------------------------
    # Thresholds for Recoding Rare Categorical Features
    # -------------------------------------------------------------------------
    "min_samples_procedure": 50,  # Min samples for a procedure category
    "min_samples_surgeon": 10,  # Min samples for a surgeon category
    "min_samples_service": 20,  # Min samples for a service category
    # -------------------------------------------------------------------------
    # Integrated Model (Predict-then-Optimize with Gurobi) Settings
    # -------------------------------------------------------------------------
    "epsilon_block_mae": 60,  # Epsilon for per-block MAE constraint (minutes)
    "feature_scaling": "std",  # "std" for standardization, "raw" for no scaling
    "prediction_error_weight": 50,  # Weight for prediction error in the objective
    # -------------------------------------------------------------------------
    # Integrated Joint Predict-and-Optimize Settings
    # -------------------------------------------------------------------------
    "integrated_alpha": 0.1,          # Trade-off weight α ∈ [0,1] between prediction loss and scheduling cost
    "integrated_lambda": 0.5,         # L1 penalty λ on regression coefficients θ_j
    "num_planning_weeks": 10,  # Number of weeks to plan in the integrated model
    # -------------------------------------------------------------------------
    # Gurobi Solver Parameters (can be overridden by debug_mode logic)
    # -------------------------------------------------------------------------
    "gurobi_timelimit": 1800,  # Default Gurobi time limit in seconds
    "gurobi_mipgap": 0.01,  # Default Gurobi MIP gap
    "gurobi_heuristics": 0.05,  # Default Gurobi heuristics aggressiveness
    "gurobi_output_flag": 0,  # 0 for silent, 1 for verbose
    "gurobi_threads": 0,  # 0 for Gurobi to decide (usually all cores)
    # -------------------------------------------------------------------------
    # Benders Decomposition Settings (for integrated theta learning)
    # -------------------------------------------------------------------------
    "benders_lambda_l1_theta": 10,  # L1 regularization for theta in Benders master
    "theta_bound": 200,
    "benders_parallel_subproblems": True,  # Whether to solve Benders subproblems in parallel
    "benders_max_workers": None,  # Max workers for parallel subproblems (None means ThreadPoolExecutor default)
    # Debug mode specific Gurobi settings (used if debug_mode is True)
    "gurobi_timelimit_debug": 10,
    "gurobi_mipgap_debug": 0.10,
    "num_horizons_debug": 1,
    "gurobi_subproblem_milp_settings": {
        "TimeLimit": 30,  
        "MIPGap": 0.05,  
        "OutputFlag": 0,  
        "Threads": 1,  
    },
}
