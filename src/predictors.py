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

from src.constants import (
    ALL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    COL_ACTUAL_START,
    COL_BOOKED_MIN,
    COL_BOOKED_MIN,
    COL_MONTH,
    COL_PROCEDURE_DURATION_MIN,
    COL_WEEK_OF_YEAR,
    COL_YEAR,
    DEFAULT_LOGGER_NAME,
    MIN_PROCEDURE_DURATION,
    NUMERIC_FEATURE_COLS,
    SCALING_STD,
    UNKNOWN_CATEGORY,
)
from src.data_processing import add_time_features  # For type hinting and potential use

# Setup logger
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

# Helper list for numeric keys used in theta predictor, ensuring consistency
# These are the raw feature names before scaling for theta model
THETA_NUMERIC_KEYS_RAW: List[str] = [
    COL_BOOKED_MIN,
    COL_WEEK_OF_YEAR,
    COL_MONTH,
    COL_YEAR,
]


def _validate_dataframe_for_training(
    df_warm_up: pd.DataFrame, required_target_col: str = COL_PROCEDURE_DURATION_MIN
) -> bool:
    """Validates if the warm-up DataFrame is suitable for training."""
    if df_warm_up.empty:
        logger.warning("Warm-up data is empty. Skipping predictor training.")
        return False

    required_cols_for_training = ALL_FEATURE_COLS + [required_target_col]
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
    target_col: str = COL_PROCEDURE_DURATION_MIN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares X and y. Assumes df_input_with_time_features already has time features."""
    X = df_input_with_time_features[ALL_FEATURE_COLS].copy()
    for cat_col in CATEGORICAL_FEATURE_COLS:
        X[cat_col] = X[cat_col].astype(str).fillna(UNKNOWN_CATEGORY)
    y = df_input_with_time_features[target_col].clip(lower=MIN_PROCEDURE_DURATION)
    return X, y


def _create_sklearn_preprocessor() -> ColumnTransformer:
    """Creates a standard ColumnTransformer for numeric and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURE_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURE_COLS,
            ),
        ],
        remainder="drop",  # Ensure no other columns accidentally pass through
    )
    return preprocessor


def train_lasso_predictor(
    df_warm_up: pd.DataFrame, params: Dict[str, Any]
) -> Optional[Pipeline]:
    """Trains a LASSO CV model to predict procedure duration.

    The model uses features defined in `ALL_FEATURE_COLS`. Categorical features
    are one-hot encoded, and numeric features are standardized.
    Cross-validation (TimeSeriesSplit) is used to select the best alpha.

    Args:
        df_warm_up: DataFrame for the warm-up period. Must contain all features
            in `ALL_FEATURE_COLS` and the target `COL_PROCEDURE_DURATION_MIN`.
        params: Configuration dictionary. Expected keys:
            - 'lasso_alphas' (List[float]): Grid of alpha values for LassoCV.
            - (Optional) 'cv_n_splits' (int): Number of splits for TimeSeriesSplit, defaults to 5.


    Returns:
        A trained scikit-learn Pipeline if successful, otherwise None.
    """
    logger.info("Starting LASSO predictor training.")

    # Add time features to df_warm_up first, so validation is accurate
    df_warm_up_processed = df_warm_up.copy()  # Work on a copy
    df_warm_up_processed = add_time_features(
        df_warm_up_processed
    )  # Ensure time features

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    X_train, y_train = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    alpha_grid = params.get("lasso_alphas", [0.1, 0.5, 1.0, 5.0])
    n_splits = params.get("cv_n_splits", 5)  # Allow configuring n_splits
    time_series_cv = TimeSeriesSplit(n_splits=n_splits)

    lasso_cv_model = LassoCV(
        alphas=alpha_grid,
        cv=time_series_cv,
        max_iter=10000,  # Consider making this configurable
        n_jobs=-1,  # Use all available cores
        random_state=42,  # For reproducibility
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
        f"CV folds: {n_splits}, Features used: {len(ALL_FEATURE_COLS)}."
    )
    return pipeline


def train_lasso_asym(
    df_warm_up: pd.DataFrame, params: Dict[str, Any]
) -> Optional[Pipeline]:
    """Trains an L1-penalized quantile regressor (asymmetric LASSO).

    This model minimizes pinball loss, with the quantile determined by the
    relative costs of overtime and idle time.
    τ* = c_overtime / (c_overtime + c_idle).

    Args:
        df_warm_up: DataFrame for the warm-up period. Must contain features
            in `ALL_FEATURE_COLS` and the target `COL_PROCEDURE_DURATION_MIN`.
        params: Configuration dictionary. Must contain:
            - 'cost_overtime_per_min' (float)
            - 'cost_idle_per_min' (float)
            - (Optional) 'lasso_alpha_asym' (float): Regularization strength, default 0.5.

    Returns:
        A trained scikit-learn Pipeline if successful, otherwise None.

    Raises:
        KeyError: If required cost parameters are missing from `params`.
    """
    logger.info("Starting Asymmetric LASSO (Quantile Regressor) training.")

    # Add time features to df_warm_up first, so validation is accurate
    df_warm_up_processed = df_warm_up.copy()  # Work on a copy
    df_warm_up_processed = add_time_features(
        df_warm_up_processed
    )  # Ensure time features

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    try:
        cost_overtime = params["cost_overtime_per_min"]
        cost_idle = params["cost_idle_per_min"]
    except KeyError as e:
        logger.error(f"Missing cost parameter for asymmetric LASSO training: {e}")
        raise

    X_train, y_train = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    # Calculate the target quantile
    if (cost_overtime + cost_idle) == 0:  # Avoid division by zero
        logger.warning(
            "Sum of overtime and idle costs is zero. Defaulting quantile to 0.5."
        )
        target_quantile = 0.5
    else:
        target_quantile = cost_overtime / (cost_overtime + cost_idle)

    quantile_regressor = QuantileRegressor(
        quantile=target_quantile,
        alpha=params.get("lasso_alpha_asym", 0.5),  # L1 penalty
        fit_intercept=True,
        solver="highs",  # HiGHS solver supports L1 penalty and is generally efficient
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

    # Get number of features from preprocessor
    try:
        num_transformed_features = len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
    except Exception:  # Fallback if get_feature_names_out fails for some reason
        num_transformed_features = "N/A"

    logger.info(
        f"Asymmetric LASSO training complete. Quantile (τ): {target_quantile:.3f}, "
        f"Alpha: {quantile_regressor.alpha}, Observations: {len(X_train)}, "
        f"Transformed Features: {num_transformed_features}."
    )
    return pipeline


def train_knn_predictor(
    df_warm_up: pd.DataFrame, params: Dict[str, Any]
) -> Optional[Pipeline]:
    """Trains a K-Nearest Neighbors (KNN) regressor for procedure duration.

    The number of neighbors (k) is chosen by minimizing a proxy cost function
    (based on overtime and idle time costs) on a validation set derived from
    a chronological split of the warm-up data.

    Args:
        df_warm_up: DataFrame for the warm-up period.
        params: Configuration dictionary. Must contain:
            - 'knn_k_options' (List[int]): Grid of k values to test.
            - 'cost_overtime_per_min' (float)
            - 'cost_idle_per_min' (float)
            - (Optional) 'knn_train_split_ratio' (float): Ratio for train/validation split, default 0.8.

    Returns:
        A trained scikit-learn Pipeline with the best k, if successful. None otherwise.

    Raises:
        KeyError: If required cost or k_options parameters are missing.
    """
    logger.info("Starting KNN predictor training.")

    # Add time features to df_warm_up first, so validation is accurate
    df_warm_up_processed = df_warm_up.copy()  # Work on a copy
    df_warm_up_processed = add_time_features(
        df_warm_up_processed
    )  # Ensure time features

    if not _validate_dataframe_for_training(df_warm_up_processed):
        return None

    try:
        k_options_grid = params["knn_k_options"]
        cost_overtime = params["cost_overtime_per_min"]
        cost_idle = params["cost_idle_per_min"]
    except KeyError as e:
        logger.error(f"Missing parameter for KNN training: {e}")
        raise

    X_full, y_full = _prepare_features_target(df_warm_up_processed)
    preprocessor = _create_sklearn_preprocessor()

    # Chronological 80/20 split for validation (default, can be parameterized)
    train_split_ratio = params.get("knn_train_split_ratio", 0.8)
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
    best_pipeline_config: Optional[Pipeline] = None

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

        # Predictions on validation set
        # Note: X_val_fold is already prepared (time features, categorical handling)
        # from _prepare_features_target call on X_full.
        # The pipeline's preprocessor will transform X_val_fold.
        predictions_val = current_pipeline.predict(X_val_fold)

        # Calculate proxy cost on validation set
        # 'booked_time_minutes' needs to be present in X_val_fold for this calculation
        if COL_BOOKED_MIN not in X_val_fold.columns:
            logger.error(
                f"'{COL_BOOKED_MIN}' not found in validation features for KNN proxy cost. Skipping k={k_val}."
            )
            continue

        overtime_val = np.maximum(
            predictions_val - X_val_fold[COL_BOOKED_MIN].values, 0.0
        )
        idle_time_val = np.maximum(
            X_val_fold[COL_BOOKED_MIN].values - predictions_val, 0.0
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
            # Store the pipeline configuration that achieved this best cost
            # We'll re-initialize and fit on full data later
            # best_pipeline_config = current_pipeline # This would store the fitted pipeline on subset

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


def _precompute_feature_matrix_for_theta(
    df_data: pd.DataFrame, feature_order: List[str], scaling_method: str
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Precomputes the feature matrix for the theta predictor.

    This function prepares a numerical matrix based on the provided `feature_order`,
    which includes numeric features (potentially scaled) and one-hot encoded
    representations for categorical features. It also handles constant terms
    like 'bias' or 'intercept' if present in `feature_order`.

    Args:
        df_data: DataFrame containing the raw data for features.
        feature_order: Ordered list of feature names. Numeric features
            (from `THETA_NUMERIC_KEYS_RAW`) should typically come first.
            Categorical features are expected in the format "column_name=value".
            Other names are treated as constant terms (e.g., "bias").
        scaling_method: Scaling to apply to numeric features ("std" or "raw").

    Returns:
        A tuple containing:
        - feature_matrix (np.ndarray): The (n_samples x n_features) matrix.
        - column_to_index_map (Dict[str, int]): Mapping from feature name to
          its column index in the matrix.
    """
    df_augmented = df_data.copy()
    df_augmented = add_time_features(df_augmented)  # Ensures week, month, year

    # Ensure COL_BOOKED_MIN exists, alias if needed. Theta expects COL_BOOKED_MIN.
    if COL_BOOKED_MIN not in df_augmented and COL_BOOKED_MIN in df_augmented:
        df_augmented[COL_BOOKED_MIN] = df_augmented[COL_BOOKED_MIN]
    elif COL_BOOKED_MIN not in df_augmented:
        raise KeyError(
            f"'{COL_BOOKED_MIN}' or its source '{COL_BOOKED_MIN}' "
            "not found in DataFrame for theta predictor."
        )

    # Numeric part
    numeric_matrix_raw = df_augmented[THETA_NUMERIC_KEYS_RAW].to_numpy(dtype=float)

    if scaling_method.lower() == SCALING_STD:
        mean_vals = numeric_matrix_raw.mean(axis=0)
        std_devs = numeric_matrix_raw.std(axis=0, ddof=0)
        std_devs[std_devs == 0] = 1.0  # Avoid division by zero for constant features
        numeric_matrix_scaled = (numeric_matrix_raw - mean_vals) / std_devs
    else:  # "raw" or any other scaling
        numeric_matrix_scaled = numeric_matrix_raw

    # Identify categorical vs. constant columns from feature_order
    # (excluding the numeric ones already processed)
    other_feature_keys = [
        key for key in feature_order if key not in THETA_NUMERIC_KEYS_RAW
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
            # Ensure the original column exists in df_augmented
            if original_col_name not in df_augmented.columns:
                logger.warning(
                    f"Theta predictor: Original column '{original_col_name}' for one-hot "
                    f"feature '{combined_key}' not found in DataFrame. This feature will be all zeros."
                )
                continue  # Skip if base column missing, effectively making this feature 0
            categorical_matrix[:, j_idx] = (
                df_augmented[original_col_name].astype(str).values == value_str
            ).astype(float)
        except ValueError:
            logger.error(
                f"Theta predictor: Malformed categorical key '{combined_key}'. "
                "Expected format 'column_name=value'. This feature will be problematic."
            )
            # This column in categorical_matrix will remain zeros.

    # Constant part (e.g., for bias/intercept term)
    constant_matrix: Optional[np.ndarray] = None
    if constant_keys_in_order:
        constant_matrix = np.ones(
            (num_samples, len(constant_keys_in_order)), dtype=float
        )

    # Combine all parts: numeric | categorical | constant
    # The order must match `feature_order`
    matrices_to_stack = [numeric_matrix_scaled]
    if categorical_keys_in_order:  # only stack if there are categorical features
        matrices_to_stack.append(categorical_matrix)
    if (
        constant_keys_in_order and constant_matrix is not None
    ):  # only stack if there are constant features
        matrices_to_stack.append(constant_matrix)

    full_feature_matrix = np.hstack(matrices_to_stack)

    # Create mapping from feature name to its final column index
    # This ensures the theta_vector aligns correctly with the matrix
    final_column_to_index_map = {name: idx for idx, name in enumerate(feature_order)}

    return full_feature_matrix, final_column_to_index_map


def make_theta_predictor(
    theta_json_path: Union[str, Path],
    df_pool_for_scaling_and_lookup: pd.DataFrame,
    scaling_method: str = SCALING_STD,          # keep for API consistency
) -> Optional[Callable[[Dict[str, Any]], float]]:
    """
    Fast θ-based predictor (std-scaled features only).
    """
    logger.info(f"Creating theta predictor from: {theta_json_path}")

    # ------------------------------------------------------------------ #
    # 1. Load θ                                                           #
    # ------------------------------------------------------------------ #
    try:
        theta_coefficients: Dict[str, float] = json.loads(
            Path(theta_json_path).read_text()
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Cannot load theta JSON ({exc}).")
        return None

    # Harmonise legacy alias ------------------------------------------- #
    if "booked_min" in theta_coefficients and COL_BOOKED_MIN not in theta_coefficients:
        theta_coefficients[COL_BOOKED_MIN] = theta_coefficients.pop("booked_min")

    # ------------------------------------------------------------------ #
    # 2. Feature ordering & θ vector                                      #
    # ------------------------------------------------------------------ #
    ordered_feature_names: List[str] = THETA_NUMERIC_KEYS_RAW + [
        k for k in theta_coefficients if k not in THETA_NUMERIC_KEYS_RAW
    ]
    theta_vector_np = np.asarray(
        [theta_coefficients.get(k, 0.0) for k in ordered_feature_names], dtype=float
    )

    # ------------------------------------------------------------------ #
    # 3. Pre-compute feature matrix                                       #
    # ------------------------------------------------------------------ #
    if scaling_method.lower() != SCALING_STD:
        raise ValueError(
            "make_theta_predictor is now hard-wired for SCALING_STD only."
        )

    # Ensure index == 'id' --------------------------------------------- #
    if "id" in df_pool_for_scaling_and_lookup.columns and (
        df_pool_for_scaling_and_lookup.index.name != "id"
    ):
        df_pool_for_scaling_and_lookup = df_pool_for_scaling_and_lookup.set_index(
            "id", drop=False
        )

    try:
        F_pool_matrix, _ = _precompute_feature_matrix_for_theta(
            df_pool_for_scaling_and_lookup, ordered_feature_names, scaling_method
        )
    except KeyError as e:
        logger.error(f"Pre-compute failed: {e}")
        return None

    base_predictions_for_pool = F_pool_matrix @ theta_vector_np

    id_to_row: Dict[Any, int] = {
        idx: pos for pos, idx in enumerate(df_pool_for_scaling_and_lookup.index)
    }

    # Mean / σ for booked-time column (needed for scaled delta) -------- #
    mu_b = df_pool_for_scaling_and_lookup[COL_BOOKED_MIN].mean()
    sigma_b = df_pool_for_scaling_and_lookup[COL_BOOKED_MIN].std(ddof=0)
    if sigma_b == 0.0:
        sigma_b = 1.0  # avoid div-by-zero

    # Position of booked_min coeff in θ vector ------------------------- #
    booked_coeff_idx = THETA_NUMERIC_KEYS_RAW.index(COL_BOOKED_MIN)
    booked_coeff_val = theta_vector_np[booked_coeff_idx]

    # ------------------------------------------------------------------ #
    # 4. Predictor closure                                                #
    # ------------------------------------------------------------------ #
    def _predict_single_surgery_theta(surg_dict: Dict[str, Any]) -> float:
        sid = surg_dict["id"]

        try:
            row_idx = id_to_row[sid]
            pred = float(base_predictions_for_pool[row_idx])
        except KeyError:
            logger.error(f"Surgery ID {sid} not in θ cache – returning 0.0.")
            return 0.0

        # -------- scaled booked-time override (optional) -------------- #
        if COL_BOOKED_MIN in surg_dict:
            orig_b = df_pool_for_scaling_and_lookup.at[sid, COL_BOOKED_MIN]
            new_b = float(surg_dict[COL_BOOKED_MIN])

            if new_b != orig_b:
                delta_scaled = (new_b - mu_b) / sigma_b - (orig_b - mu_b) / sigma_b
                pred += delta_scaled * booked_coeff_val

        return max(MIN_PROCEDURE_DURATION, round(pred, 1))

    logger.info("Theta predictor (std-scaled) created successfully.")
    return _predict_single_surgery_theta
