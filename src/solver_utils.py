"""
Core Gurobi model building and solving utilities for surgery scheduling.
Includes implementations for deterministic, predictive, clairvoyant, SAA,
and integrated optimization models with graduated overtime costs.

All models use consistent objective functions, constraints, and variable bounds.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pandas as pd

from src.config import AppConfig, CONFIG
from src.constants import (
    DataColumns,
    FeatureColumns,
    ScheduleColumns,
    DomainConstants,
    GurobiConstants,
    LoggingConstants,
)
from src.data_processing import add_time_features

logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


def set_gurobi_model_parameters(
    model: gp.Model,
    config: AppConfig = CONFIG,
    override_timelimit: Optional[float] = None,
    override_mipgap: Optional[float] = None,
    override_heuristics: Optional[float] = None,
    override_output_flag: Optional[int] = None,
    override_threads: Optional[int] = None,
) -> None:
    """Sets common Gurobi parameters for a model.

    Uses defaults from `config`, with specific overrides possible.
    `debug_mode` in `config` influences default time limit, MIP gap, and heuristics.

    Args:
        model: The Gurobi model to configure.
        config: Application configuration containing Gurobi settings.
        override_timelimit: Specific time limit in seconds.
        override_mipgap: Specific MIP gap.
        override_heuristics: Specific heuristics aggressiveness (0 to 1).
        override_output_flag: Gurobi OutputFlag (0=silent, 1=verbose).
        override_threads: Number of threads for Gurobi (0=auto).
    """
    is_debug_mode = config.debug_mode

    # Default Gurobi parameters from config
    default_timelimit = config.gurobi.timelimit
    default_mipgap = config.gurobi.mipgap
    default_heuristics = config.gurobi.heuristics
    default_output_flag = config.gurobi.output_flag
    default_threads = config.gurobi.threads
    default_presolve = config.gurobi.presolve
    default_mip_focus = config.gurobi.mip_focus

    # Apply debug mode adjustments if active
    if is_debug_mode:
        logger.debug(
            f"Gurobi debug mode active for model {model.ModelName}. Applying fast-solve settings."
        )
        effective_timelimit = config.gurobi.timelimit_debug
        effective_mipgap = config.gurobi.mipgap_debug
        effective_heuristics = 0.50
        effective_mip_focus = GurobiConstants.MIP_FOCUS_FEASIBILITY
    else:
        effective_timelimit = default_timelimit
        effective_mipgap = default_mipgap
        effective_heuristics = default_heuristics
        effective_mip_focus = default_mip_focus

    # Apply function-level overrides
    if override_timelimit is not None:
        effective_timelimit = override_timelimit
    if override_mipgap is not None:
        effective_mipgap = override_mipgap
    if override_heuristics is not None:
        effective_heuristics = override_heuristics

    effective_output_flag = (
        override_output_flag
        if override_output_flag is not None
        else default_output_flag
    )
    effective_threads = (
        override_threads if override_threads is not None else default_threads
    )

    # Set parameters on the Gurobi model
    model.setParam(GRB.Param.OutputFlag, 0)  # Always silent by default
    model.setParam(GRB.Param.TimeLimit, effective_timelimit)
    model.setParam(GRB.Param.MIPGap, effective_mipgap)
    model.setParam(GRB.Param.Heuristics, effective_heuristics)
    model.setParam(GRB.Param.OutputFlag, effective_output_flag)
    model.setParam(GRB.Param.Threads, effective_threads)
    model.setParam(GRB.Param.Presolve, default_presolve)
    if effective_mip_focus != 0 or is_debug_mode:
        model.setParam(GRB.Param.MIPFocus, effective_mip_focus)

    if model.ModelName and "PredictedTime" in model.ModelName:
        # Aggressive settings for prediction-based models
        model.setParam(GRB.Param.MIPFocus, 1)  # Focus on feasibility
        model.setParam(GRB.Param.Cuts, 2)  # Moderate cuts
        model.setParam(GRB.Param.Heuristics, 0.5)  # More heuristics
        model.setParam(GRB.Param.ImpliedCuts, 2)  # Aggressive implied cuts


def solve_saa_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    scenario_duration_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    Solve the Sample Average Approximation (SAA) via Benders decomposition
    with graduated overtime costs and consistent objective function.

    All operational costs (OT/idle) are treated as recourse costs.

    Parameters
    ----------
    surgeries_info : list of dict
        Metadata for each surgery.
    daily_block_counts : dict[int, int]
        Mapping from day index to number of OR blocks available that day.
    config : AppConfig
        Application configuration.
    scenario_duration_matrix : np.ndarray, shape (n_surgeries, n_scenarios)
        Matrix of durations (in minutes) for each surgery under each scenario.

    Returns
    -------
    result : dict
        Dictionary with keys: status, obj, model
    """
    # Input validation
    num_surgeries = len(surgeries_info)
    if (
        not isinstance(scenario_duration_matrix, np.ndarray)
        or scenario_duration_matrix.ndim != 2
    ):
        raise ValueError("scenario_duration_matrix must be a 2-D numpy array")
    if scenario_duration_matrix.shape[0] != num_surgeries:
        raise ValueError(
            f"scenario_duration_matrix row count ({scenario_duration_matrix.shape[0]})"
            f" must equal number of surgeries ({num_surgeries})"
        )
    n_scenarios = scenario_duration_matrix.shape[1]

    # Build list of blocks (day, block_index)
    blocks: List[Tuple[int, int]] = []
    for day, count in daily_block_counts.items():
        if count < 1:
            continue
        for block in range(count):
            blocks.append((day, block))
    if not blocks:
        raise ValueError("No operating room blocks provided in daily_block_counts.")

    # Extract parameters
    block_size = config.operating_room.block_size_minutes
    cost_ot_normal = config.costs.overtime_per_min
    cost_it = config.costs.idle_per_min
    reject_cost = config.costs.rejection_per_case

    # Initialize model
    model = gp.Model("SAA_Benders_GraduatedOT")
    set_gurobi_model_parameters(model, config)
    model.Params.LazyConstraints = GurobiConstants.LAZY_CONSTRAINTS_ENABLED

    # First-stage decision variables
    x = model.addVars(
        (
            (i, d, b)
            for i in range(num_surgeries)
            for (d, b) in blocks
            if i < len(surgeries_info)  # Ensure index is valid
        ),
        vtype=GRB.BINARY,
        name="x",
    )
    r = model.addVars(range(num_surgeries), vtype=GRB.BINARY, name="r")
    theta = model.addVar(lb=0.0, name="theta")

    # Auxiliary variables for structural constraints (not in objective)
    ot_aux = model.addVars(
        blocks, lb=0.0, ub=config.costs.max_overtime_minutes, name="ot_aux"
    )
    it_aux = model.addVars(blocks, lb=0.0, ub=block_size, name="it_aux")

    # Constraints: each surgery is assigned exactly once or rejected
    for i in range(num_surgeries):
        model.addConstr(
            quicksum(x[i, d, b] for (d, b) in blocks) + r[i] == 1,
            name=f"assign_reject_{i}",
        )

    # Structural constraints using two-inequality slack style
    model_durations: Dict[int, float] = {
        i: float(surg[DataColumns.BOOKED_MIN]) for i, surg in enumerate(surgeries_info)
    }

    for d, b in blocks:
        total_duration = quicksum(
            model_durations[i] * x[i, d, b] for i in range(num_surgeries)
        )
        total_ot = ot_aux[d, b]

        # Two-inequality style for better performance
        model.addConstr(
            total_duration - block_size <= total_ot,
            name=f"ot_calc_{d}_{b}",
        )
        model.addConstr(
            block_size - total_duration <= it_aux[d, b],
            name=f"it_calc_{d}_{b}",
        )

    # Objective: only rejection cost, all OT/idle in recourse (theta)
    rejection_cost_total = quicksum(
        reject_cost * surgeries_info[i][DataColumns.BOOKED_MIN] * r[i]
        for i in range(num_surgeries)
    )

    model.setObjective(rejection_cost_total + theta, GRB.MINIMIZE)

    def _benders_callback_graduated(m: gp.Model, where: int) -> None:
        """Gurobi callback to add Benders cuts with graduated overtime costs."""
        if where != GRB.Callback.MIPSOL:
            return

        # Retrieve current solution values for x
        x_val = m.cbGetSolution(x)

        # Compute expected recourse cost across scenarios with graduated overtime
        total_rec = 0.0
        for k in range(n_scenarios):
            rec_k = 0.0
            durations_k = scenario_duration_matrix[:, k]
            for d, b in blocks:
                assigned_time = sum(
                    durations_k[i] * x_val[(i, d, b)] for i in range(num_surgeries)
                )
                if assigned_time > block_size:
                    overtime = assigned_time - block_size
                    # Graduated overtime cost
                    capped_overtime = min(overtime, config.costs.max_overtime_minutes)
                    rec_k += cost_ot_normal * capped_overtime
                else:
                    rec_k += cost_it * (block_size - assigned_time)
            total_rec += rec_k
        avg_rec = total_rec / n_scenarios

        # Add Benders cut
        m.cbLazy(theta >= avg_rec)
        logger.debug(f"Benders cut added: theta >= {avg_rec:.2f}")

    # Optimize with lazy cuts
    logger.info("Starting Benders decomposition solve (SAA)")
    model.optimize(_benders_callback_graduated)

    status = model.Status
    obj_val = model.ObjVal if status == GRB.OPTIMAL else None
    logger.info(f"Benders solve complete: status={status}, obj={obj_val}")

    return {"status": status, "obj": obj_val, "model": model}


def _solve_single_stage_deterministic_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    duration_key_for_model: str,
    model_name_suffix: str,
    is_lp_relaxation: bool = False,
) -> Dict[str, Any]:
    """Core solver for single-stage deterministic assignment models with graduated overtime costs.

    This function builds and solves models like "Deterministic" (using booked times),
    "Predictive" (using ML predictions), or "Clairvoyant" (using actual durations).

    Args:
        surgeries_info: List of surgery data.
        daily_block_counts: Maps day index to number of blocks.
        config: Application configuration.
        duration_key_for_model: The key in surgery_info dicts to use for durations.
        model_name_suffix: Suffix for the Gurobi model name.
        is_lp_relaxation: If True, relaxes binary variables to continuous.

    Returns:
        Dictionary with "obj", "status", and "model".
    """

    num_surgeries = len(surgeries_info)
    planning_horizon_days = config.data.planning_horizon_days
    block_size_min = config.operating_room.block_size_minutes
    cost_rejection = config.costs.rejection_per_case
    cost_overtime_normal = config.costs.overtime_per_min
    cost_idle = config.costs.idle_per_min

    model_det = gp.Model(f"SingleStage_{model_name_suffix}")
    set_gurobi_model_parameters(model_det, config)

    var_type = GRB.CONTINUOUS if is_lp_relaxation else GRB.BINARY

    # --- Decision Variables ---
    x_vars = model_det.addVars(
        [
            (i, d, b)
            for i in range(num_surgeries)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        vtype=var_type,
        name=GurobiConstants.VAR_X_PREFIX,
    )
    r_vars = model_det.addVars(
        num_surgeries, vtype=var_type, name=GurobiConstants.VAR_R_PREFIX
    )

    # overtime variables
    ot_vars = model_det.addVars(
        [
            (d, b)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        lb=0.0,
        ub=config.costs.max_overtime_minutes,  # Capped overtime
        name="ot",
    )

    it_vars = model_det.addVars(
        [
            (d, b)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        lb=0.0,
        ub=block_size_min,
        name=GurobiConstants.VAR_IT_PREFIX,
    )

    # --- Constraints ---
    # 1. Assignment or rejection for each surgery
    for i in range(num_surgeries):
        model_det.addConstr(
            quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            )
            + r_vars[i]
            == 1,
            name=f"assign_or_reject_{i}",
        )

    # 2. Block capacity using two-inequality slack style
    for d_day in range(planning_horizon_days):
        for b_block in range(daily_block_counts.get(d_day, 0)):
            sum_durations_in_block = quicksum(
                surgeries_info[i][duration_key_for_model] * x_vars[i, d_day, b_block]
                for i in range(num_surgeries)
            )

            total_overtime = ot_vars[d_day, b_block]

            # Two-inequality style for better performance
            model_det.addConstr(
                sum_durations_in_block - block_size_min <= total_overtime,
                name=f"ot_calc_{d_day}_{b_block}",
            )

            model_det.addConstr(
                block_size_min - sum_durations_in_block <= it_vars[d_day, b_block],
                name=f"it_calc_{d_day}_{b_block}",
            )

    # --- Objective Function ---
    rejection_cost_total = cost_rejection * quicksum(
        surgeries_info[i][DataColumns.BOOKED_MIN] * r_vars[i]
        for i in range(num_surgeries)
    )
    operational_cost = cost_overtime_normal * ot_vars.sum() + cost_idle * it_vars.sum()
    model_det.setObjective(rejection_cost_total + operational_cost, GRB.MINIMIZE)

    relax_msg = " (LP relaxation)" if is_lp_relaxation else ""
    logger.info(
        f"Optimizing {model_det.ModelName}{relax_msg} for {num_surgeries} surgeries..."
    )
    model_det.optimize()

    obj_val = model_det.ObjVal if model_det.SolCount > 0 else None
    if obj_val is None:
        logger.warning(
            f"{model_det.ModelName}{relax_msg} optimization did not find a feasible solution."
        )
    else:
        logger.info(
            f"{model_det.ModelName}{relax_msg} optimized. Objective: {obj_val:.2f}, Status: {model_det.Status}"
        )

    return {"obj": obj_val, "status": model_det.Status, "model": model_det}


def solve_deterministic_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Solves the deterministic model using booked surgery durations."""
    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        config,
        duration_key_for_model=DataColumns.BOOKED_MIN,
        model_name_suffix="BookedTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )


def solve_predictive_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Solves the predictive model using ML-predicted surgery durations."""
    # Ensure 'predicted_dur_min' key exists or fall back
    for surg in surgeries_info:
        if "predicted_dur_min" not in surg:
            logger.warning(
                f"Surgery id {surg.get('id','N/A')} missing 'predicted_dur_min'. "
                f"Falling back to '{DataColumns.BOOKED_MIN}' for this surgery in predictive model."
            )
            surg["predicted_dur_min"] = round(surg[DataColumns.BOOKED_MIN])

    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        config,
        duration_key_for_model="predicted_dur_min",
        model_name_suffix="PredictedTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )


def solve_clairvoyant_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Solves the clairvoyant model using actual surgery durations."""
    if not (surgeries_info and ScheduleColumns.ACTUAL_DUR_MIN in surgeries_info[0]):
        logger.error(
            f"Clairvoyant model: Cannot find duration key '{ScheduleColumns.ACTUAL_DUR_MIN}' "
            "in surgery data. Ensure `select_surgeries` provides this key."
        )
        return {
            "obj": None,
            "status": "Error_Missing_Actual_Duration_Key",
            "model": None,
        }

    return _solve_single_stage_deterministic_model(
        surgeries_info,
        daily_block_counts,
        config,
        duration_key_for_model=ScheduleColumns.ACTUAL_DUR_MIN,
        model_name_suffix="ClairvoyantActualTime",
        is_lp_relaxation=kwargs.get("lp_relax", False),
    )


def validate_model_consistency(config: AppConfig) -> None:
    """
    Validate that all model parameters are consistent.
    Call this function before running any models to catch configuration issues.
    """
    logger.info("Validating model consistency...")

    # Check cost parameters exist
    required_cost_attrs = [
        "rejection_per_case",
        "overtime_per_min",
        "idle_per_min",
    ]
    for attr in required_cost_attrs:
        assert hasattr(config.costs, attr), f"Missing cost parameter: {attr}"
        value = getattr(config.costs, attr)
        assert value >= 0, f"Cost parameter {attr} must be non-negative, got {value}"

    # Check duration bounds
    min_dur = config.operating_room.min_procedure_duration
    block_size = config.operating_room.block_size_minutes
    max_overtime = DomainConstants.MAX_OVERTIME_MINUTES_PER_BLOCK
    max_dur = block_size + max_overtime

    assert min_dur > 0, f"Min duration must be positive, got {min_dur}"
    assert block_size > 0, f"Block size must be positive, got {block_size}"
    assert max_overtime >= 0, f"Max overtime must be non-negative, got {max_overtime}"
    assert (
        max_dur > min_dur
    ), f"Max duration ({max_dur}) must exceed min duration ({min_dur})"

    # Check overtime threshold
    ot_threshold = config.costs.overtime_threshold_minutes
    assert (
        0 < ot_threshold <= max_overtime
    ), f"Overtime threshold ({ot_threshold}) must be between 0 and {max_overtime}"

    # Check cost hierarchy makes sense
    normal_cost = config.costs.overtime_per_min

    logger.info("✓ Model consistency validated")
    logger.info(f"  Duration bounds: [{min_dur}, {max_dur}] minutes")
    logger.info(f"  Overtime threshold: {ot_threshold} minutes")
    logger.info(f"  Overtime costs: ${normal_cost}/min (normal)")
    logger.info(f"  Block size: {block_size} minutes")


def solve_knn_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    historical_data: pd.DataFrame = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Solves the KNN model by finding k nearest neighbors and using their actual durations.

    For each surgery, finds K most similar historical surgeries and uses their
    actual durations as scenarios for SAA optimization.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if historical_data is None or historical_data.empty:
        logger.warning(
            "No historical data for KNN. Falling back to deterministic model."
        )
        return solve_deterministic_model(
            surgeries_info, daily_block_counts, config, **kwargs
        )

    k = config.ml.knn_neighbors
    num_surgeries = len(surgeries_info)

    logger.info(
        f"KNN solver: k={k}, historical_data_size={len(historical_data)}, current_surgeries={num_surgeries}"
    )

    # Prepare historical data with time features
    hist_data = historical_data.copy()
    hist_data = add_time_features(hist_data)

    # Validate required columns
    required_features = FeatureColumns.ALL
    missing_features = [
        col for col in required_features if col not in hist_data.columns
    ]
    if missing_features:
        logger.error(f"KNN: Missing features in historical data: {missing_features}")
        return solve_deterministic_model(
            surgeries_info, daily_block_counts, config, **kwargs
        )

    try:
        # Prepare historical features and targets
        X_hist = hist_data[required_features].copy()
        for cat_col in FeatureColumns.CATEGORICAL:
            X_hist[cat_col] = (
                X_hist[cat_col].astype(str).fillna(DomainConstants.UNKNOWN_CATEGORY)
            )

        hist_durations = hist_data[DataColumns.PROCEDURE_DURATION_MIN].values

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), FeatureColumns.NUMERIC),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    FeatureColumns.CATEGORICAL,
                ),
            ],
            remainder="drop",
        )

        # Fit and transform historical data
        X_hist_processed = preprocessor.fit_transform(X_hist)
        logger.info(f"KNN: Historical features shape: {X_hist_processed.shape}")

        if len(X_hist_processed) < k:
            logger.warning(
                f"KNN: Not enough historical data ({len(X_hist_processed)}) for k={k}"
            )
            return solve_deterministic_model(
                surgeries_info, daily_block_counts, config, **kwargs
            )

        # Fit KNN on historical data
        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(X_hist_processed)

        # Prepare current surgery features
        current_surgery_features = []
        for surgery in surgeries_info:
            surgery_features = {
                DataColumns.BOOKED_MIN: surgery[DataColumns.BOOKED_MIN],
                DataColumns.MAIN_PROCEDURE_ID: surgery["proc_id"],
                DataColumns.WEEK_OF_YEAR: 26,  # Mid-year default
                DataColumns.MONTH: 6,  # Mid-year default
                DataColumns.YEAR: 2024,  # Current year default
                DataColumns.PATIENT_TYPE: DomainConstants.OTHER_CATEGORY,
                DataColumns.SURGEON_CODE: DomainConstants.OTHER_CATEGORY,
                DataColumns.CASE_SERVICE: DomainConstants.OTHER_CATEGORY,
            }
            current_surgery_features.append(surgery_features)

        # Convert to DataFrame and preprocess
        current_df = pd.DataFrame(current_surgery_features)
        for cat_col in FeatureColumns.CATEGORICAL:
            current_df[cat_col] = current_df[cat_col].astype(str)

        # Transform current surgery features
        X_current_processed = preprocessor.transform(current_df[required_features])
        logger.info(f"KNN: Current surgery features shape: {X_current_processed.shape}")

        # Create scenario matrix: (num_surgeries, k_scenarios)
        scenario_matrix = np.zeros((num_surgeries, k))
        all_neighbor_indices = []

        # For each surgery, find its k nearest neighbors
        for i in range(num_surgeries):
            try:
                # Find k nearest neighbors for this surgery
                distances, indices = knn.kneighbors(X_current_processed[i : i + 1])
                neighbor_indices = indices[0]
                all_neighbor_indices.extend(neighbor_indices)

                # Get actual durations from neighbors
                neighbor_durations = hist_durations[neighbor_indices]

                # Clip durations to reasonable bounds
                neighbor_durations = np.clip(
                    neighbor_durations,
                    DomainConstants.MIN_PROCEDURE_DURATION,
                    config.operating_room.block_size_minutes * 2,
                )

                # Store scenarios for this surgery
                scenario_matrix[i, :] = neighbor_durations

                # Debug info for first few surgeries
                if i < 3:
                    logger.debug(
                        f"KNN: Surgery {i}, neighbors: {neighbor_indices}, durations: {neighbor_durations}"
                    )

            except Exception as e:
                logger.warning(f"KNN: Error processing surgery {i}: {e}")
                # Fallback to booked duration for all scenarios
                scenario_matrix[i, :] = surgeries_info[i][DataColumns.BOOKED_MIN]

        # Validate scenario matrix
        if np.any(scenario_matrix <= 0):
            logger.warning("KNN: Found non-positive durations in scenario matrix")

        # Check diversity of neighbors
        unique_neighbors = len(set(all_neighbor_indices))
        neighbor_coverage = (
            unique_neighbors / len(hist_data) if len(hist_data) > 0 else 0
        )

        logger.info(
            f"KNN: Using {unique_neighbors} unique historical observations "
            f"out of {len(hist_data)} available (coverage: {neighbor_coverage:.1%})"
        )
        logger.info(f"KNN: Scenario matrix shape: {scenario_matrix.shape}")
        logger.info(
            f"KNN: Scenario matrix stats - min: {scenario_matrix.min():.1f}, "
            f"max: {scenario_matrix.max():.1f}, mean: {scenario_matrix.mean():.1f}"
        )

        # Use SAA solver with KNN scenarios
        result = solve_saa_model(
            surgeries_info, daily_block_counts, config, scenario_matrix
        )

        # Add debug info to result
        if result:
            result["knn_debug"] = {
                "k_used": k,
                "unique_neighbors": unique_neighbors,
                "neighbor_coverage": neighbor_coverage,
                "scenario_shape": scenario_matrix.shape,
                "scenario_mean": float(scenario_matrix.mean()),
                "scenario_std": float(scenario_matrix.std()),
                "scenario_min": float(scenario_matrix.min()),
                "scenario_max": float(scenario_matrix.max()),
            }

        logger.info("KNN: Successfully created scenario matrix and solved")
        return result

    except Exception as e:
        logger.error(f"KNN: Error in processing: {e}", exc_info=True)
        return solve_deterministic_model(
            surgeries_info, daily_block_counts, config, **kwargs
        )


def solve_utilization_maximization_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Solves the utilization maximization model using predicted surgery durations.

    Maximizes the total duration of scheduled surgeries rather than minimizing costs.
    Uses predicted durations in the objective function from XGBoost model.

    Args:
        surgeries_info: List of surgery data with predictions attached.
        daily_block_counts: Maps day index to number of blocks.
        config: Application configuration.
        **kwargs: Additional arguments (e.g., lp_relax).

    Returns:
        Dictionary with "obj", "status", and "model".
    """

    # Ensure 'predicted_dur_min' key exists or fall back to booked time
    for surg in surgeries_info:
        if "predicted_dur_min" not in surg:
            logger.warning(
                f"Surgery id {surg.get('id','N/A')} missing 'predicted_dur_min'. "
                f"Falling back to '{DataColumns.BOOKED_MIN}' for this surgery in utilization maximization model."
            )
            surg["predicted_dur_min"] = round(surg[DataColumns.BOOKED_MIN])

    num_surgeries = len(surgeries_info)
    planning_horizon_days = config.data.planning_horizon_days
    block_size_min = config.operating_room.block_size_minutes

    model_util = gp.Model("UtilizationMaximization")
    set_gurobi_model_parameters(model_util, config)

    var_type = GRB.CONTINUOUS if kwargs.get("lp_relax", False) else GRB.BINARY

    # --- Decision Variables ---
    # x_idb: 1 if surgery i is assigned to day d, block b
    x_vars = model_util.addVars(
        [
            (i, d, b)
            for i in range(num_surgeries)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        vtype=var_type,
        name=GurobiConstants.VAR_X_PREFIX,
    )

    # R_i: 1 if surgery i is rejected
    r_vars = model_util.addVars(
        num_surgeries, vtype=var_type, name=GurobiConstants.VAR_R_PREFIX
    )

    # --- Constraints ---
    # 1. Each surgery is either assigned to exactly one block or rejected
    for i in range(num_surgeries):
        model_util.addConstr(
            quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            )
            + r_vars[i]
            == 1,
            name=f"assign_or_reject_{i}",
        )

    # 2. Block capacity constraints
    for d_day in range(planning_horizon_days):
        for b_block in range(daily_block_counts.get(d_day, 0)):
            # Sum of predicted durations in block cannot exceed block capacity
            model_util.addConstr(
                quicksum(
                    surgeries_info[i]["predicted_dur_min"] * x_vars[i, d_day, b_block]
                    for i in range(num_surgeries)
                )
                <= block_size_min,
                name=f"capacity_{d_day}_{b_block}",
            )

    # --- Objective Function: Maximize total predicted duration of scheduled surgeries ---
    total_utilization = quicksum(
        surgeries_info[i]["predicted_dur_min"] * x_vars[i, d, b]
        for i in range(num_surgeries)
        for d in range(planning_horizon_days)
        for b in range(daily_block_counts.get(d, 0))
    )

    model_util.setObjective(total_utilization, GRB.MAXIMIZE)

    relax_msg = " (LP relaxation)" if kwargs.get("lp_relax", False) else ""
    logger.info(
        f"Optimizing {model_util.ModelName}{relax_msg} for {num_surgeries} surgeries "
        f"(maximizing predicted utilization)..."
    )
    model_util.optimize()

    obj_val = model_util.ObjVal if model_util.SolCount > 0 else None
    if obj_val is None:
        logger.warning(
            f"{model_util.ModelName}{relax_msg} optimization did not find a feasible solution."
        )
    else:
        logger.info(
            f"{model_util.ModelName}{relax_msg} optimized. Predicted Utilization: {obj_val:.2f} minutes, "
            f"Status: {model_util.Status}"
        )

    return {"obj": obj_val, "status": model_util.Status, "model": model_util}


def solve_balanced_utilization_model(
    surgeries_info: List[Dict[str, Any]],
    daily_block_counts: Dict[int, int],
    config: AppConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Solves the balanced utilization-throughput model using predicted surgery durations.

    Maximizes (total_duration) × (number_of_surgeries) to balance both utilization and throughput.
    Uses linearization techniques to handle the nonlinear objective.

    Args:
        surgeries_info: List of surgery data with predictions attached.
        daily_block_counts: Maps day index to number of blocks.
        config: Application configuration.
        **kwargs: Additional arguments (e.g., lp_relax).

    Returns:
        Dictionary with "obj", "status", and "model".
    """

    # Ensure 'predicted_dur_min' key exists or fall back to booked time
    for surg in surgeries_info:
        if "predicted_dur_min" not in surg:
            logger.warning(
                f"Surgery id {surg.get('id','N/A')} missing 'predicted_dur_min'. "
                f"Falling back to '{DataColumns.BOOKED_MIN}' for this surgery in balanced utilization model."
            )
            surg["predicted_dur_min"] = round(surg[DataColumns.BOOKED_MIN])

    num_surgeries = len(surgeries_info)
    planning_horizon_days = config.data.planning_horizon_days
    block_size_min = config.operating_room.block_size_minutes

    model_balanced = gp.Model("BalancedUtilizationThroughput")
    set_gurobi_model_parameters(model_balanced, config)

    var_type = GRB.CONTINUOUS if kwargs.get("lp_relax", False) else GRB.BINARY

    # --- Decision Variables ---
    # x_idb: 1 if surgery i is assigned to day d, block b
    x_vars = model_balanced.addVars(
        [
            (i, d, b)
            for i in range(num_surgeries)
            for d in range(planning_horizon_days)
            for b in range(daily_block_counts.get(d, 0))
        ],
        vtype=var_type,
        name=GurobiConstants.VAR_X_PREFIX,
    )

    # R_i: 1 if surgery i is rejected
    r_vars = model_balanced.addVars(
        num_surgeries, vtype=var_type, name=GurobiConstants.VAR_R_PREFIX
    )

    # s_i: 1 if surgery i is scheduled (auxiliary variable)
    s_vars = model_balanced.addVars(num_surgeries, vtype=var_type, name="s")

    # Link s_i to assignment variables
    for i in range(num_surgeries):
        model_balanced.addConstr(
            s_vars[i]
            == quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            ),
            name=f"schedule_indicator_{i}",
        )

    # Total number of scheduled surgeries
    total_surgeries = quicksum(s_vars[i] for i in range(num_surgeries))

    # Linearization variables: w_i represents the contribution of surgery i to the product
    # w_i = predicted_duration_i × total_surgeries × s_i
    w_vars = model_balanced.addVars(num_surgeries, lb=0.0, name="w")

    # Big-M for linearization (conservative upper bound)
    max_predicted_duration = max(surg["predicted_dur_min"] for surg in surgeries_info)
    big_M = max_predicted_duration * num_surgeries

    # Linearization constraints for w_i = predicted_duration_i × total_surgeries × s_i
    for i in range(num_surgeries):
        predicted_dur = surgeries_info[i]["predicted_dur_min"]

        # If surgery i is not scheduled (s_i = 0), then w_i = 0
        model_balanced.addConstr(
            w_vars[i] <= big_M * s_vars[i],
            name=f"w_upper_scheduled_{i}",
        )

        # If surgery i is scheduled (s_i = 1), then w_i = predicted_duration_i × total_surgeries
        model_balanced.addConstr(
            w_vars[i] <= predicted_dur * total_surgeries,
            name=f"w_upper_duration_{i}",
        )

        # If surgery i is scheduled, w_i >= predicted_duration_i × total_surgeries
        model_balanced.addConstr(
            w_vars[i] >= predicted_dur * total_surgeries - big_M * (1 - s_vars[i]),
            name=f"w_lower_{i}",
        )

    # --- Standard Constraints ---
    # 1. Each surgery is either assigned to exactly one block or rejected
    for i in range(num_surgeries):
        model_balanced.addConstr(
            quicksum(
                x_vars[i, d, b]
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
            )
            + r_vars[i]
            == 1,
            name=f"assign_or_reject_{i}",
        )

    # 2. Block capacity constraints
    for d_day in range(planning_horizon_days):
        for b_block in range(daily_block_counts.get(d_day, 0)):
            model_balanced.addConstr(
                quicksum(
                    surgeries_info[i]["predicted_dur_min"] * x_vars[i, d_day, b_block]
                    for i in range(num_surgeries)
                )
                <= block_size_min,
                name=f"capacity_{d_day}_{b_block}",
            )

    # --- Objective Function: Maximize sum of w_i (which represents total_duration × total_surgeries) ---
    model_balanced.setObjective(
        quicksum(w_vars[i] for i in range(num_surgeries)), GRB.MAXIMIZE
    )

    relax_msg = " (LP relaxation)" if kwargs.get("lp_relax", False) else ""
    logger.info(
        f"Optimizing {model_balanced.ModelName}{relax_msg} for {num_surgeries} surgeries "
        f"(maximizing utilization × throughput)..."
    )
    model_balanced.optimize()

    obj_val = model_balanced.ObjVal if model_balanced.SolCount > 0 else None
    if obj_val is None:
        logger.warning(
            f"{model_balanced.ModelName}{relax_msg} optimization did not find a feasible solution."
        )
    else:
        # Calculate the actual utilization and throughput for logging
        if model_balanced.SolCount > 0:
            total_util = sum(
                surgeries_info[i]["predicted_dur_min"] * x_vars[i, d, b].X
                for i in range(num_surgeries)
                for d in range(planning_horizon_days)
                for b in range(daily_block_counts.get(d, 0))
                if x_vars[i, d, b].X > 0.5
            )
            total_count = sum(
                s_vars[i].X for i in range(num_surgeries) if s_vars[i].X > 0.5
            )
            logger.info(
                f"{model_balanced.ModelName}{relax_msg} optimized. "
                f"Objective: {obj_val:.2f}, Utilization: {total_util:.1f} min, "
                f"Throughput: {total_count:.0f} surgeries, Product: {total_util * total_count:.1f}, "
                f"Status: {model_balanced.Status}"
            )
        else:
            logger.info(
                f"{model_balanced.ModelName}{relax_msg} optimized. Objective: {obj_val:.2f}, "
                f"Status: {model_balanced.Status}"
            )

    return {"obj": obj_val, "status": model_balanced.Status, "model": model_balanced}
