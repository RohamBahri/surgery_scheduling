from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import final_paper_numeric_guard as guard
import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def test_known_eight_hour_phi_roundoff_is_accepted() -> None:
    # Exact values from the failed Stage-1 week-4 run on 2026-09-24.
    recomputed = 154909.6444510882
    solver_sum = 154909.64427304053
    err = abs(recomputed - solver_sum)
    assert err > guard.PHI_ACCOUNTING_WARN_ATOL
    assert err < guard.phi_accounting_tolerance(recomputed, solver_sum)
    guard.assert_phi_accounting_close(recomputed, solver_sum, context="regression")


def test_material_phi_accounting_error_still_fails() -> None:
    # A one-cost-unit discrepancy is far above the numerical allowance and is
    # large enough to represent a real accounting/model error.
    with pytest.raises(AssertionError, match="decomposed Phi mismatch"):
        guard.assert_phi_accounting_close(
            154909.6444510882,
            154908.6444510882,
            context="material",
        )


def test_phi_accounting_tolerance_is_scale_aware() -> None:
    assert guard.phi_accounting_tolerance(1.0, 1.0) == pytest.approx(1e-2)
    assert guard.phi_accounting_tolerance(200000.0, 200000.0) == pytest.approx(0.02)


def test_install_all_guards_covers_late_training_and_holdout_oracle_paths() -> None:
    guard.install_all_guards()
    # Stage 2's holdout oracle goes through the runtime worker; Stage 1's late
    # seed audit goes through the deterministic worker. Both must be guarded.
    assert runtime._solve_process_task is guard.training_process_task
    assert science.deterministic_eval_worker is guard.deterministic_eval_worker
    assert sensitivity._worker is guard.sensitivity_worker
    assert final.final_solve_week is guard.reviewed_final_solve_week
    assert base.solve_week is guard.reviewed_final_solve_week


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
