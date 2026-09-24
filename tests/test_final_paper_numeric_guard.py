from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import final_paper_numeric_guard as guard


def test_known_eight_hour_phi_roundoff_is_accepted() -> None:
    # Exact values from the failed Stage-1 week-4 run on 2026-09-24.
    recomputed = 154909.6444510882
    solver_sum = 154909.64427304053
    err = abs(recomputed - solver_sum)
    assert err > guard.PHI_ACCOUNTING_ATOL
    assert err < guard.phi_accounting_tolerance(recomputed, solver_sum)
    guard.assert_phi_accounting_close(recomputed, solver_sum, context="regression")


def test_material_phi_accounting_error_still_fails() -> None:
    with pytest.raises(AssertionError, match="decomposed Phi mismatch"):
        guard.assert_phi_accounting_close(
            154909.6444510882,
            154909.6344510882,
            context="material",
        )


def test_phi_accounting_tolerance_is_scale_aware() -> None:
    assert guard.phi_accounting_tolerance(1.0, 1.0) == pytest.approx(1e-5)
    assert guard.phi_accounting_tolerance(200000.0, 200000.0) == pytest.approx(0.002)


def test_supported_crossfit_has_no_penalty_futurewarning() -> None:
    # Five week groups, each containing both classes, so every fold fits a real
    # LogisticRegression model rather than the constant-probability fallback.
    groups = np.repeat(np.arange(5), 4)
    labels = np.tile(np.array([0, 1, 0, 1], dtype=int), 5)
    x1 = np.linspace(-1.0, 1.0, len(labels))
    X = sparse.csr_matrix(np.column_stack([np.ones(len(labels)), x1]))
    arrays = SimpleNamespace(X=X, week_ids=groups)
    settings = SimpleNamespace(random_seed=42)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred, metrics = guard.warning_free_crossfit_pi(arrays, labels, settings)

    assert pred.shape == labels.shape
    assert np.all((pred > 0) & (pred < 1))
    assert np.isfinite(metrics["brier"])
    penalty_warnings = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning) and "penalty" in str(w.message).lower()
    ]
    assert not penalty_warnings
