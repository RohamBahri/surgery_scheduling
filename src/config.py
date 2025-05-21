PARAMS = {
    "debug_mode": False,  # set False for full runs
    # -------------------------------------------------------------------------
    # File & temporal split
    # -------------------------------------------------------------------------
    "excel_file_path": "data/UHNOperating_RoomScheduling2011-2013.xlsx",
    "warmup_weeks": 52,
    "planning_horizon_days": 7,
    "NUM_HORIZONS": 5,
    # -------------------------------------------------------------------------
    # Operating‑room calendar
    # -------------------------------------------------------------------------
    "or_day_start_hour": 8,  # 08:00 → rooms open
    "or_day_end_hour": 16,  # 16:00 → planning stops
    "block_size_minutes": 480,  # one block = 8 h
    "blocks_per_day": 1,  # 1 blocks per day per OR
    "capacity_reduction_percentage": 0.6,  # 0.0 for no reduction, e.g., 0.1 for 10% reduction
    "min_blocks_after_reduction": 1,  # Minimum blocks a day can have after reduction
    # -------------------------------------------------------------------------
    # Cost coefficients
    # -------------------------------------------------------------------------
    "cost_rejection_per_case": 20.0,
    "cost_overtime_per_min": 15.0,
    "cost_idle_per_min": 10.0,
    # -------------------------------------------------------------------------
    # Sampling / SAA & KNN hyper‑parameters
    # -------------------------------------------------------------------------
    "saa_scenarios": 10,
    "knn_neighbors": 5,
    "knn_k_options": [3, 5, 7, 9, 11],
    "lasso_alphas": [0.1, 0.5, 1.0, 5.0],
    # -------------------------------------------------------------------------
    # Thresholds for recoding rare categories
    # -------------------------------------------------------------------------
    "min_samples_procedure": 50,
    "min_samples_surgeon": 10,
    "min_samples_service": 20,
    # -------------------------------------------------------------------------
    # Gurobi solver settings
    # -------------------------------------------------------------------------
    "gurobi_timelimit": 1200,  # seconds
    "gurobi_mipgap": 0.01,
    "gurobi_output_flag": 0,
    "gurobi_heuristics": 1.0,
    "gurobi_rins": 10,
    "gurobi_threads": 0,
}
