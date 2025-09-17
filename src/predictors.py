"""
This module provides functions for training various surgery duration prediction models
and for creating predictor callables from pre-trained coefficients (theta model).
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb


from src.config import AppConfig
from src.constants import (
    DataColumns,
    FeatureColumns,
    DomainConstants,
    LoggingConstants,
)
from src.data_processing import add_time_features

# Setup logger
logger = logging.getLogger(LoggingConstants.DEFAULT_LOGGER_NAME)


def _validate_dataframe_for_training(
    df_warm_up: pd.DataFrame,
    required_target_col: str = DataColumns.PROCEDURE_DURATION_MIN,
) -> bool:
    """Validates if the warm-up DataFrame is suitable for training."""
    if df_warm_up.empty:
        logger.warning("Warm-up data is empty. Skipping predictor training.")
        return False

    required_cols_for_training = FeatureColumns.ALL + [required_target_col]
    missing_columns = [
        col for col in required_cols_for_training if col not in df_warm_up.columns
    ]
    if missing_columns:
        logger.error(
            f"Missing required columns in warm-up data for training: {missing_columns}. "
            "Skipping predictor training."
        )
        return False
    return True


def _prepare_features_target(
    df_input_with_time_features: pd.DataFrame,
    target_col: str = DataColumns.PROCEDURE_DURATION_MIN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares X and y. Assumes df_input_with_time_features already has time features."""
    X = df_input_with_time_features[FeatureColumns.ALL].copy()
    for cat_col in FeatureColumns.CATEGORICAL:
        X[cat_col] = X[cat_col].astype(str).fillna(DomainConstants.UNKNOWN_CATEGORY)
    y = df_input_with_time_features[target_col].clip(
        lower=DomainConstants.MIN_PROCEDURE_DURATION
    )
    return X, y


def _create_sklearn_preprocessor() -> ColumnTransformer:
    """Creates a standard ColumnTransformer for numeric and categorical features."""
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
    return preprocessor


def train_lasso_predictor(
    df_warm_up: pd.DataFrame, config: AppConfig
) -> Optional[Pipeline]:
    """Trains a LASSO CV model to predict procedure duration.

    The model uses features defined in `FeatureColumns.ALL`. Categorical features
    are one-hot encoded, and numeric features are standardized.
    Cross-validation (TimeSeriesSplit) is used to select the best alpha.

    Args:
        df_warm_up: DataFrame for the warm-up period. Must contain all features
            in `FeatureColumns.ALL` and the target `DataColumns.PROCEDURE_DURATION_MIN`.
        config: Application configuration containing ML settings.

    Returns:
        A trained scikit-learn Pipeline if successful, otherwise None.
    """
    logger.info("Starting LASSO predictor training.")

    df_warm_up_processed = df_warm_up.copy()
    df_warm_up_processed = add_time_features(df_warm_up_processed)

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    X_train, y_train = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    alpha_grid = config.ml.lasso_alphas
    n_splits = 5  # Fixed for now, could be made configurable
    time_series_cv = TimeSeriesSplit(n_splits=n_splits)

    lasso_cv_model = LassoCV(
        alphas=alpha_grid,
        cv=time_series_cv,
        max_iter=10000,
        n_jobs=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", lasso_cv_model),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during LASSO pipeline fitting: {e}", exc_info=True)
        return None

    best_alpha = float(pipeline.named_steps["regressor"].alpha_)
    logger.info(
        f"LASSO predictor training complete. Best alpha: {best_alpha:.4g}, "
        f"CV folds: {n_splits}, Features used: {len(FeatureColumns.ALL)}."
    )
    return pipeline


def train_lasso_asym(df_warm_up: pd.DataFrame, config: AppConfig) -> Optional[Pipeline]:
    """Trains an L1-penalized quantile regressor (asymmetric LASSO).

    This model minimizes pinball loss, with the quantile determined by the
    relative costs of overtime and idle time.
    τ* = c_overtime / (c_overtime + c_idle).

    Args:
        df_warm_up: DataFrame for the warm-up period. Must contain features
            in `FeatureColumns.ALL` and the target `DataColumns.PROCEDURE_DURATION_MIN`.
        config: Application configuration containing cost and ML settings.

    Returns:
        A trained scikit-learn Pipeline if successful, otherwise None.
    """
    logger.info("Starting Asymmetric LASSO (Quantile Regressor) training.")

    df_warm_up_processed = df_warm_up.copy()
    df_warm_up_processed = add_time_features(df_warm_up_processed)

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    cost_overtime = config.costs.overtime_per_min
    cost_idle = config.costs.idle_per_min

    X_train, y_train = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    # Calculate the target quantile
    if (cost_overtime + cost_idle) == 0:
        logger.warning(
            "Sum of overtime and idle costs is zero. Defaulting quantile to 0.5."
        )
        target_quantile = 0.5
    else:
        target_quantile = cost_overtime / (cost_overtime + cost_idle)

    quantile_regressor = QuantileRegressor(
        quantile=target_quantile,
        alpha=config.ml.lasso_alpha_asym,
        fit_intercept=True,
        solver="highs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", quantile_regressor),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(
            f"Error during Asymmetric LASSO pipeline fitting: {e}", exc_info=True
        )
        return None

    try:
        num_transformed_features = len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
    except Exception:
        num_transformed_features = "N/A"

    logger.info(
        f"Asymmetric LASSO training complete. Quantile (τ): {target_quantile:.3f}, "
        f"Alpha: {quantile_regressor.alpha}, Observations: {len(X_train)}, "
        f"Transformed Features: {num_transformed_features}."
    )
    return pipeline


def train_xgboost_predictor(
    df_warm_up: pd.DataFrame, config: AppConfig
) -> Optional[Pipeline]:
    """Trains an XGBoost regressor for procedure duration prediction.

    The model uses gradient boosting with automatic handling of categorical features
    and missing values. Uses time series cross-validation for hyperparameter tuning.

    Args:
        df_warm_up: DataFrame for the warm-up period. Must contain all features
            in `FeatureColumns.ALL` and the target `DataColumns.PROCEDURE_DURATION_MIN`.
        config: Application configuration containing ML settings.

    Returns:
        A trained scikit-learn Pipeline with XGBoost if successful, otherwise None.
    """

    logger.info("Starting XGBoost predictor training.")

    df_warm_up_processed = df_warm_up.copy()
    df_warm_up_processed = add_time_features(df_warm_up_processed)

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    X_train, y_train = _prepare_features_target(df_warm_up_processed)

    # For XGBoost, we use a simpler preprocessor that handles categorical features differently
    # XGBoost can handle categorical features directly, but we'll still use some preprocessing
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

    # XGBoost regressor with parameters from config
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=config.ml.xgboost_n_estimators,
        max_depth=config.ml.xgboost_max_depth,
        learning_rate=config.ml.xgboost_learning_rate,
        random_state=config.ml.xgboost_random_state,
        n_jobs=1,
        verbosity=0,  # Suppress XGBoost warnings
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb_regressor),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during XGBoost pipeline fitting: {e}", exc_info=True)
        return None

    try:
        num_transformed_features = len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
    except Exception:
        num_transformed_features = "N/A"

    logger.info(
        f"XGBoost predictor training complete. "
        f"N_estimators: {config.ml.xgboost_n_estimators}, "
        f"Max_depth: {config.ml.xgboost_max_depth}, "
        f"Learning_rate: {config.ml.xgboost_learning_rate}, "
        f"Observations: {len(X_train)}, "
        f"Transformed Features: {num_transformed_features}."
    )
    return pipeline


def train_knn_predictor(
    df_warm_up: pd.DataFrame, config: AppConfig
) -> Optional[Pipeline]:
    """Trains a K-Nearest Neighbors (KNN) regressor for procedure duration.

    The number of neighbors (k) is chosen by minimizing a proxy cost function
    (based on overtime and idle time costs) on a validation set derived from
    a chronological split of the warm-up data.

    Args:
        df_warm_up: DataFrame for the warm-up period.
        config: Application configuration containing ML and cost settings.

    Returns:
        A trained scikit-learn Pipeline with the best k, if successful. None otherwise.
    """
    logger.info("Starting KNN predictor training.")

    df_warm_up_processed = df_warm_up.copy()
    df_warm_up_processed = add_time_features(df_warm_up_processed)

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    k_options_grid = config.ml.knn_k_options
    cost_overtime = config.costs.overtime_per_min
    cost_idle = config.costs.idle_per_min

    X_full, y_full = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    # Chronological 80/20 split for validation
    train_split_ratio = 0.8  # Could be made configurable
    split_index = int(len(X_full) * train_split_ratio)

    if split_index == 0 or split_index == len(X_full):
        logger.error(
            f"KNN training: Train/validation split resulted in an empty set "
            f"(split_index={split_index}, total_len={len(X_full)}). "
            "Check data size and split ratio. Aborting KNN training."
        )
        return None

    X_train_fold, X_val_fold = X_full.iloc[:split_index], X_full.iloc[split_index:]
    y_train_fold, y_val_fold = y_full.iloc[:split_index], y_full.iloc[split_index:]

    best_k_found: Optional[int] = None
    min_proxy_cost: float = np.inf

    logger.debug(
        f"KNN: Tuning k using {len(X_train_fold)} train samples and "
        f"{len(X_val_fold)} validation samples."
    )

    for k_val in k_options_grid:
        knn_regressor = KNeighborsRegressor(n_neighbors=k_val, weights="distance")
        current_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", knn_regressor)]
        )
        try:
            current_pipeline.fit(X_train_fold, y_train_fold)
        except Exception as e:
            logger.warning(
                f"KNN training with k={k_val} failed during fit: {e}", exc_info=True
            )
            continue

        predictions_val = current_pipeline.predict(X_val_fold)

        # Calculate proxy cost on validation set
        if DataColumns.BOOKED_MIN not in X_val_fold.columns:
            logger.error(
                f"'{DataColumns.BOOKED_MIN}' not found in validation features for KNN proxy cost. Skipping k={k_val}."
            )
            continue

        overtime_val = np.maximum(
            predictions_val - X_val_fold[DataColumns.BOOKED_MIN].values, 0.0
        )
        idle_time_val = np.maximum(
            X_val_fold[DataColumns.BOOKED_MIN].values - predictions_val, 0.0
        )
        current_proxy_cost = (
            cost_overtime * overtime_val + cost_idle * idle_time_val
        ).mean()
        logger.debug(
            f"KNN: k={k_val}, Proxy Cost (Validation)={current_proxy_cost:.2f}"
        )

        if current_proxy_cost < min_proxy_cost:
            min_proxy_cost = current_proxy_cost
            best_k_found = k_val

    if best_k_found is None:
        logger.warning(
            "KNN training: No suitable k found from the provided options. Skipping."
        )
        return None

    logger.info(
        f"KNN: Best k found: {best_k_found} with proxy cost {min_proxy_cost:.2f} on validation set. "
        "Re-training on full warm-up data."
    )

    # Re-train the best model on the full warm-up set
    final_knn_regressor = KNeighborsRegressor(
        n_neighbors=best_k_found, weights="distance"
    )
    final_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", final_knn_regressor)]
    )
    try:
        final_pipeline.fit(X_full, y_full)
    except Exception as e:
        logger.error(
            f"Error during final KNN pipeline fitting with k={best_k_found}: {e}",
            exc_info=True,
        )
        return None

    logger.info("KNN predictor training complete.")
    return final_pipeline


def tune_knn_k_for_optimization(
    df_warm_up: pd.DataFrame,
    config: AppConfig,
    k_candidates: Optional[List[int]] = None,
) -> int:
    """
    Tune K parameter specifically for KNN optimization (not prediction).

    This tunes K based on how well KNN scenarios predict actual outcomes
    using time series cross-validation.

    Args:
        df_warm_up: Warm-up period data
        config: Application configuration
        k_candidates: List of K values to test

    Returns:
        Optimal K value for KNN optimization
    """
    if k_candidates is None:
        k_candidates = [3, 5, 7, 9, 11, 15, 20, 25]

    if df_warm_up.empty:
        logger.warning(
            "Empty warm-up data for KNN optimization tuning, using default K=5"
        )
        return 5

    # Add time features
    df_processed = df_warm_up.copy()
    df_processed = add_time_features(df_processed)

    # Check required columns
    required_cols = FeatureColumns.ALL + [DataColumns.PROCEDURE_DURATION_MIN]
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        logger.error(f"Missing columns for KNN optimization tuning: {missing_cols}")
        return 5

    # Prepare features and target
    X = df_processed[FeatureColumns.ALL].copy()
    y = df_processed[DataColumns.PROCEDURE_DURATION_MIN].values

    # Handle categorical features
    for cat_col in FeatureColumns.CATEGORICAL:
        X[cat_col] = X[cat_col].astype(str).fillna(DomainConstants.UNKNOWN_CATEGORY)

    # Create preprocessing pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import mean_absolute_error

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

    # Use time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    k_scores = {}

    logger.info(f"Tuning KNN K for optimization with candidates: {k_candidates}")

    for k in k_candidates:
        if k >= len(X):
            continue

        fold_maes = []

        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < k:
                continue

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                # Preprocess features
                X_train_processed = preprocessor.fit_transform(X_train)
                X_val_processed = preprocessor.transform(X_val)

                # Fit KNN
                knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
                knn.fit(X_train_processed)

                # Make predictions using mean of K neighbors
                distances, indices = knn.kneighbors(X_val_processed)
                predictions = np.array(
                    [y_train[neighbor_indices].mean() for neighbor_indices in indices]
                )

                # Calculate MAE
                mae = mean_absolute_error(y_val, predictions)
                fold_maes.append(mae)

            except Exception as e:
                logger.warning(f"Error in KNN optimization tuning for k={k}: {e}")
                fold_maes.append(float("inf"))

        if fold_maes:
            k_scores[k] = np.mean(fold_maes)
            logger.debug(f"KNN optimization tuning: k={k}, CV MAE={k_scores[k]:.2f}")

    if not k_scores:
        logger.warning(
            "KNN optimization tuning failed for all K values, using default k=5"
        )
        return 5

    # Select best K
    best_k = min(k_scores.keys(), key=lambda k: k_scores[k])
    logger.info(
        f"KNN optimization tuning complete. Best k={best_k} with CV MAE={k_scores[best_k]:.2f}"
    )

    return best_k


def _precompute_feature_matrix_for_theta(
    df_data: pd.DataFrame, feature_order: List[str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Precomputes the feature matrix for the theta predictor using standardization.

    This function prepares a numerical matrix based on the provided `feature_order`,
    which includes numeric features (standardized) and one-hot encoded
    representations for categorical features.

    Args:
        df_data: DataFrame containing the raw data for features.
        feature_order: Ordered list of feature names. Numeric features
            should typically come first. Categorical features are expected
            in the format "column_name=value".

    Returns:
        A tuple containing:
        - feature_matrix (np.ndarray): The (n_samples x n_features) matrix.
        - column_to_index_map (Dict[str, int]): Mapping from feature name to
          its column index in the matrix.
    """
    df_augmented = df_data.copy()
    df_augmented = add_time_features(df_augmented)

    # Ensure DataColumns.BOOKED_MIN exists
    if DataColumns.BOOKED_MIN not in df_augmented:
        raise KeyError(
            f"'{DataColumns.BOOKED_MIN}' not found in DataFrame for theta predictor."
        )

    # Numeric part - always use standardization
    numeric_matrix_raw = df_augmented[FeatureColumns.THETA_NUMERIC_RAW].to_numpy(
        dtype=float
    )
    mean_vals = numeric_matrix_raw.mean(axis=0)
    std_devs = numeric_matrix_raw.std(axis=0, ddof=0)
    std_devs[std_devs == 0] = 1.0  # Avoid division by zero
    numeric_matrix_scaled = (numeric_matrix_raw - mean_vals) / std_devs

    # Identify categorical vs. constant columns from feature_order
    other_feature_keys = [
        key for key in feature_order if key not in FeatureColumns.THETA_NUMERIC_RAW
    ]
    categorical_keys_in_order = [key for key in other_feature_keys if "=" in key]
    constant_keys_in_order = [key for key in other_feature_keys if "=" not in key]

    # One-hot categorical part
    num_samples = len(df_augmented)
    categorical_matrix = np.zeros(
        (num_samples, len(categorical_keys_in_order)), dtype=float
    )
    for j_idx, combined_key in enumerate(categorical_keys_in_order):
        try:
            original_col_name, value_str = combined_key.split("=", maxsplit=1)
            if original_col_name not in df_augmented.columns:
                logger.warning(
                    f"Theta predictor: Original column '{original_col_name}' for one-hot "
                    f"feature '{combined_key}' not found in DataFrame. This feature will be all zeros."
                )
                continue
            categorical_matrix[:, j_idx] = (
                df_augmented[original_col_name].astype(str).values == value_str
            ).astype(float)
        except ValueError:
            logger.error(
                f"Theta predictor: Malformed categorical key '{combined_key}'. "
                "Expected format 'column_name=value'. This feature will be problematic."
            )

    # Constant part (e.g., for bias/intercept term)
    constant_matrix: Optional[np.ndarray] = None
    if constant_keys_in_order:
        constant_matrix = np.ones(
            (num_samples, len(constant_keys_in_order)), dtype=float
        )

    # Combine all parts: numeric | categorical | constant
    matrices_to_stack = [numeric_matrix_scaled]
    if categorical_keys_in_order:
        matrices_to_stack.append(categorical_matrix)
    if constant_keys_in_order and constant_matrix is not None:
        matrices_to_stack.append(constant_matrix)

    full_feature_matrix = np.hstack(matrices_to_stack)

    # Create mapping from feature name to its final column index
    final_column_to_index_map = {name: idx for idx, name in enumerate(feature_order)}

    return full_feature_matrix, final_column_to_index_map


def make_theta_predictor(
    theta_json_path: Union[str, Path],
    df_pool_for_scaling_and_lookup: pd.DataFrame,
) -> Optional[Callable[[Dict[str, Any]], float]]:
    """
    Fast θ-based predictor using standardized features.

    Args:
        theta_json_path: Path to the JSON file containing theta coefficients.
        df_pool_for_scaling_and_lookup: DataFrame used for scaling and lookup.

    Returns:
        A callable predictor function or None if creation fails.
    """
    logger.info(f"Creating theta predictor from: {theta_json_path}")

    # Load θ coefficients
    try:
        theta_coefficients: Dict[str, float] = json.loads(
            Path(theta_json_path).read_text()
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Cannot load theta JSON ({exc}).")
        return None

    # Harmonise legacy alias
    if (
        "booked_min" in theta_coefficients
        and DataColumns.BOOKED_MIN not in theta_coefficients
    ):
        theta_coefficients[DataColumns.BOOKED_MIN] = theta_coefficients.pop(
            "booked_min"
        )

    # Feature ordering & θ vector
    ordered_feature_names: List[str] = FeatureColumns.THETA_NUMERIC_RAW + [
        k for k in theta_coefficients if k not in FeatureColumns.THETA_NUMERIC_RAW
    ]
    theta_vector_np = np.asarray(
        [theta_coefficients.get(k, 0.0) for k in ordered_feature_names], dtype=float
    )

    # Ensure index == 'id'
    if "id" in df_pool_for_scaling_and_lookup.columns and (
        df_pool_for_scaling_and_lookup.index.name != "id"
    ):
        df_pool_for_scaling_and_lookup = df_pool_for_scaling_and_lookup.set_index(
            "id", drop=False
        )

    try:
        F_pool_matrix, _ = _precompute_feature_matrix_for_theta(
            df_pool_for_scaling_and_lookup, ordered_feature_names
        )
    except KeyError as e:
        logger.error(f"Pre-compute failed: {e}")
        return None

    base_predictions_for_pool = F_pool_matrix @ theta_vector_np

    id_to_row: Dict[Any, int] = {
        idx: pos for pos, idx in enumerate(df_pool_for_scaling_and_lookup.index)
    }

    # Mean / σ for booked-time column (needed for scaled delta)
    mu_b = df_pool_for_scaling_and_lookup[DataColumns.BOOKED_MIN].mean()
    sigma_b = df_pool_for_scaling_and_lookup[DataColumns.BOOKED_MIN].std(ddof=0)
    if sigma_b == 0.0:
        sigma_b = 1.0  # avoid div-by-zero

    # Position of booked_min coeff in θ vector
    booked_coeff_idx = FeatureColumns.THETA_NUMERIC_RAW.index(DataColumns.BOOKED_MIN)
    booked_coeff_val = theta_vector_np[booked_coeff_idx]

    # Predictor closure
    def _predict_single_surgery_theta(surg_dict: Dict[str, Any]) -> float:
        sid = surg_dict["id"]

        try:
            row_idx = id_to_row[sid]
            pred = float(base_predictions_for_pool[row_idx])
        except KeyError:
            logger.error(f"Surgery ID {sid} not in θ cache – returning 0.0.")
            return 0.0

        # Scaled booked-time override (optional)
        if DataColumns.BOOKED_MIN in surg_dict:
            orig_b = df_pool_for_scaling_and_lookup.at[sid, DataColumns.BOOKED_MIN]
            new_b = float(surg_dict[DataColumns.BOOKED_MIN])

            if new_b != orig_b:
                delta_scaled = (new_b - mu_b) / sigma_b - (orig_b - mu_b) / sigma_b
                pred += delta_scaled * booked_coeff_val

        max_reasonable_duration = DomainConstants.DEFAULT_BLOCK_SIZE_MINUTES * 2
        pred = max(DomainConstants.MIN_PROCEDURE_DURATION, round(pred, 1))
        pred = min(pred, max_reasonable_duration)  # Trim extreme values

        # ADDED: Round to nearest 5 minutes
        pred = round(pred / 5) * 5

        return float(pred)

    logger.info("Theta predictor (std-scaled) created successfully.")
    return _predict_single_surgery_theta
