import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_processing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    add_time_features,
)


# =============================================================================
# MODEL TRAINING (LASSO)
# =============================================================================
def train_lasso_predictor(df_warm: pd.DataFrame, params: dict):
    """
    Train a LASSO model to predict procedure_duration_min.
    Expects df_warm with:
      - booked_time_minutes
      - actual_start (pd.Timestamp)
      - procedure_duration_min
      - patient_type, main_procedure_id, surgeon_code, case_service
    Automatically creates week_of_year, month, year features.
    """
    df = df_warm.copy()
    if df.empty:
        print("[train_lasso_predictor] WARNING – warm-up data empty, skipping.")
        return None

    df = add_time_features(df)
    feature_cols = ALL_FEATURES
    num_cols = NUMERIC_FEATURES
    cat_cols = CATEGORICAL_FEATURES

    # check required columns
    missing = [
        c for c in feature_cols + ["procedure_duration_min"] if c not in df.columns
    ]
    if missing:
        print(f"[train_lasso_predictor] Missing columns {missing}; cannot train.")
        return None

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")
    y = df["procedure_duration_min"].clip(lower=1.0)

    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    alpha_grid = params.get("lasso_alphas", [0.1, 0.5, 1.0, 5.0])
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    lasso_cv = LassoCV(
        alphas=alpha_grid,
        cv=tscv,
        max_iter=10000,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(
        [
            ("preprocessor", preproc),
            ("regressor", lasso_cv),
        ]
    )

    try:
        pipe.fit(X, y)
    except Exception as e:
        print(f"[train_lasso_predictor] ERROR fitting pipeline: {e}")
        return None

    best_alpha = float(pipe.named_steps["regressor"].alpha_)
    print(
        f"[train_lasso_predictor] LASSO trained – best α = {best_alpha:.4g}, folds = {n_splits}"
    )
    return pipe


# -------------------------------------------------------------------------
#  Asymmetric‑loss LASSO (pin‑ball loss with ℓ₁ penalty)
# -------------------------------------------------------------------------
def train_lasso_asym(df_warm: pd.DataFrame, params: dict):
    """
    Train an ℓ₁‑penalised quantile regressor (a “LASSO‑like” model) that
    minimises the pin‑ball loss aligned with planning costs.

    Let *c_ot*   = cost of one minute of overtime,
        *c_idle* = cost of one minute of idle time.

    The optimal quantile to forecast is
        τ* = c_ot / (c_ot + c_idle).

    *Example* – If overtime costs twice as much as idle:
        c_ot = 2, c_idle = 1  ⇒  τ* = 2 / 3  ≈ 0.667
    so the regressor predicts the **66.7th percentile** of historical
    durations for each feature vector rather than the mean/median.

    Parameters
    ----------
    df_warm : pd.DataFrame
        Cleaned historical dataset (the “warm‑up” window).
    params : dict
        Must contain:
            - "cost_overtime_per_min"
            - "cost_idle_per_min"
        Optional:
            - "lasso_alpha_asym"  (regularisation strength, default 0.1)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Same interface as the other predictor objects: `.predict(X)`.
    """
    if df_warm.empty:
        raise ValueError("Warm‑up dataframe is empty – cannot train predictor.")

    # 1)  Build feature matrix -------------------------------------------------
    df = add_time_features(df_warm.copy())
    X = df[ALL_FEATURES].copy()
    y = df["procedure_duration_min"].clip(lower=1.0)  # guard against zeros

    # Ensure categorical features are strings (OneHotEncoder requirement)
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str).fillna("Unknown")

    # 2)  Compute target quantile ---------------------------------------------
    c_ot = params["cost_overtime_per_min"]
    c_idle = params["cost_idle_per_min"]
    tau = c_ot / (c_ot + c_idle)

    # 3)  Model pipeline -------------------------------------------------------
    qr = QuantileRegressor(
        quantile=tau,
        alpha=params.get("lasso_alpha_asym", 0.5),
        fit_intercept=True,
        solver="highs",  # HiGHS is fast and supports ℓ₁ penalty
    )

    pipe = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), NUMERIC_FEATURES),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            CATEGORICAL_FEATURES,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("qr", qr),
        ]
    )

    # 4)  Fit ------------------------------------------------------------------
    pipe.fit(X, y)

    print(
        f"[train_lasso_asym] τ={tau:.3f}, "
        f"alpha={qr.alpha}, "
        f"n_obs={len(df)}, "
        f"n_feat={pipe['pre'].transformers_[0][2].__len__() + pipe['pre'].transformers_[1][2].__len__()}"
    )

    return pipe


# =============================================================================
# MODEL TRAINING (KNN PREDICTOR FOR WEIGHT-BASED METHOD)
# =============================================================================
def train_knn_predictor(df_warm: pd.DataFrame, params: dict):
    """
    Train a KNN model (k chosen by proxy-cost minimization) to predict
    procedure_duration_min.  Same feature set & temporal derivation as LASSO.
    """
    df = df_warm.copy()
    if df.empty:
        print("[train_knn_predictor] WARNING – warm-up data empty, skipping.")
        return None

    df = add_time_features(df)
    feature_cols = ALL_FEATURES
    num_cols = NUMERIC_FEATURES
    cat_cols = CATEGORICAL_FEATURES

    missing = [
        c for c in feature_cols + ["procedure_duration_min"] if c not in df.columns
    ]
    if missing:
        print(f"[train_knn_predictor] Missing columns {missing}; cannot train.")
        return None

    # prepare data
    X_full = df[feature_cols].copy()
    y_full = df["procedure_duration_min"].clip(lower=1.0)
    for c in cat_cols:
        X_full[c] = X_full[c].astype(str).fillna("Unknown")

    # chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_full.iloc[:split_idx], y_full.iloc[split_idx:]

    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    k_grid = params.get("knn_k_options", [3, 5, 7, 9, 11])
    c_ot = params["cost_overtime_per_min"]
    c_idle = params["cost_idle_per_min"]

    best_k, best_cost, best_pipe = None, np.inf, None

    for k in k_grid:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        pipe = Pipeline([("preprocessor", preproc), ("regressor", knn)])
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"[train_knn_predictor] K={k} fit error: {e}")
            continue

        pred = pipe.predict(X_val)
        overtime = np.maximum(pred - X_val["booked_time_minutes"].values, 0.0)
        idle = np.maximum(X_val["booked_time_minutes"].values - pred, 0.0)
        cost = (c_ot * overtime + c_idle * idle).mean()

        if cost < best_cost:
            best_cost, best_k, best_pipe = cost, k, pipe

    if best_pipe is None:
        print("[train_knn_predictor] No valid k found; skipping.")
        return None

    # re‐train on full warm-up set
    final_knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
    final_pipe = Pipeline([("preprocessor", preproc), ("regressor", final_knn)])
    final_pipe.fit(X_full, y_full)

    print(
        f"[train_knn_predictor] KNN trained – best k = {best_k}, proxy cost = {best_cost:.2f}"
    )
    return final_pipe
