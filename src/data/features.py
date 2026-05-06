"""Authoritative feature pipeline.

This is the single source of truth for feature construction, encoding, and
transformation.  Every method that needs features must use this module.
There is no other place in the codebase that constructs feature matrices.

The pipeline:
  1. Selects numeric columns (booked time, calendar features).
  2. Selects categorical columns (patient type, procedure, surgeon, service).
  3. Standardises numeric features.
  4. One-hot encodes categorical features.

The fitted encoder is a scikit-learn ``ColumnTransformer`` wrapped in a
thin class that also handles the DataFrame ↔ CaseRecord conversion.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.types import CaseRecord, Col, Domain, WeeklyInstance

logger = logging.getLogger(__name__)

# ── Feature specification ────────────────────────────────────────────────────
# These lists define the exact columns that enter the model.  Changing them
# here changes them everywhere.

NUMERIC_FEATURES: List[str] = [
    Col.BOOKED_MINUTES,
    Col.WEEK_OF_YEAR,
    Col.MONTH,
    Col.YEAR,
]

CATEGORICAL_FEATURES: List[str] = [
    Col.PATIENT_TYPE,
    Col.PROCEDURE_ID,
    Col.SURGEON_CODE,
    Col.CASE_SERVICE,
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN: str = Col.PROCEDURE_DURATION


class FeatureEncoder:
    """Fit-once, transform-many feature encoder.

    Wraps a scikit-learn ``ColumnTransformer`` that standardises numeric
    features and one-hot encodes categorical features.  After fitting on
    the warm-up DataFrame, it can transform both DataFrames (for batch
    training) and lists of ``CaseRecord`` objects (for per-instance
    inference).
    """

    def __init__(self) -> None:
        self._transformer: Optional[ColumnTransformer] = None
        self._is_fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> "FeatureEncoder":
        """Fit the encoder on the warm-up (training) DataFrame.

        Parameters
        ----------
        df_train : DataFrame
            Must contain all columns listed in ``ALL_FEATURES``.
        """
        X = self._prepare_frame(df_train)
        self._transformer = self._build_transformer()
        self._transformer.fit(X)
        self._is_fitted = True
        n_features = self._transformer.get_feature_names_out().shape[0]
        logger.info("FeatureEncoder fitted: %d input cols → %d encoded features.",
                     len(ALL_FEATURES), n_features)
        return self

    # ── Transform ────────────────────────────────────────────────────────

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame to a feature matrix.

        Parameters
        ----------
        df : DataFrame
            Must contain all columns listed in ``ALL_FEATURES``.

        Returns
        -------
        ndarray, shape (n_samples, n_encoded_features)
        """
        self._check_fitted()
        X = self._prepare_frame(df)
        return self._transformer.transform(X)

    def transform_cases(self, cases: List[CaseRecord]) -> np.ndarray:
        """Transform a list of CaseRecord objects to a feature matrix.

        This is used at planning time when the method receives a
        ``WeeklyInstance`` instead of a raw DataFrame.

        Returns
        -------
        ndarray, shape (len(cases), n_encoded_features)
        """
        self._check_fitted()
        df = self._cases_to_frame(cases)
        return self._transformer.transform(df)

    def transform_instance(self, instance: WeeklyInstance) -> np.ndarray:
        """Convenience: transform all cases from a ``WeeklyInstance``."""
        return self.transform_cases(instance.cases)

    # ── Target extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_target(df: pd.DataFrame) -> np.ndarray:
        """Return realized room time as a 1-D target vector."""
        y = df[TARGET_COLUMN].values.astype(float)
        return y

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return len(self._transformer.get_feature_names_out())

    @property
    def feature_names(self) -> List[str]:
        self._check_fitted()
        return list(self._transformer.get_feature_names_out())

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_transformer() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False),
                 CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

    @staticmethod
    def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns and ensure categoricals are strings."""
        X = df[ALL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str).fillna(Domain.UNKNOWN)
        return X

    @staticmethod
    def _cases_to_frame(cases: List[CaseRecord]) -> pd.DataFrame:
        """Convert CaseRecord objects to a DataFrame with feature columns."""
        rows = []
        for c in cases:
            rows.append({
                Col.BOOKED_MINUTES:  c.booked_duration_min,
                Col.WEEK_OF_YEAR:    c.week_of_year,
                Col.MONTH:           c.month,
                Col.YEAR:            c.year,
                Col.PATIENT_TYPE:    str(c.patient_type),
                Col.PROCEDURE_ID:    str(c.procedure_id),
                Col.SURGEON_CODE:    str(c.surgeon_code),
                Col.CASE_SERVICE:    str(c.service),
            })
        return pd.DataFrame(rows)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("FeatureEncoder has not been fitted yet. "
                               "Call .fit(df_train) first.")
