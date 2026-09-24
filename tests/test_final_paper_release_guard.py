from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import final_paper_finalization_fixes as hardening
import final_paper_release_guard as release
import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def _two_site_arrays() -> base.Arrays:
    X = sparse.csr_matrix([[1.0, 0.0], [1.0, 1.0]], dtype=float)
    booked = np.array([300.0, 300.0])
    actual = np.array([300.0, 300.0])
    return base.Arrays(
        X=X,
        booked=booked,
        actual=actual,
        error=actual - booked,
        week_ids=np.array([0, 0]),
        case_ids=np.array([1, 2]),
        week_slices={0: np.array([0, 1])},
    )


def test_release_install_covers_every_process_worker() -> None:
    release.install_reviewed_guards()
    assert runtime._solve_process_task is release.runtime_worker
    assert science.deterministic_eval_worker is release.deterministic_worker
    assert sensitivity._worker is release.sensitivity_worker
    assert hardening._solve_site_shift_grid is release.constrained_site_shift_grid


def test_site_shift_search_enforces_coefficient_box_before_selection() -> None:
    final.install_final_adapter()
    science.apply_scientific_fixes()
    a = _two_site_arrays()
    s = science.ScientificFinalSettings(
        data="dummy.xlsx",
        artifact_root="artifacts/test",
    )
    # This is the boundary case from the review: an unconstrained grid can pick
    # a large discarded-response shift with zero case loss even though the zero
    # policy is a valid coefficient-feasible optimum.
    w, hist = release.constrained_site_shift_grid(a, s, lam=0.5)
    assert np.all(np.isfinite(w))
    assert np.all(np.abs(w) <= s.coefficient_bound + 1e-9)
    assert hist[-1]["accepted"] is True
    assert hist[-1]["coefficient_bound"] == s.coefficient_bound


def test_zero_booked_regret_makes_gap_closed_undefined() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario": "zero_turnover",
                "method": "BOOKED",
                "avg_regret_lower": 0.0,
                "avg_regret_upper": 0.0,
            },
            {
                "scenario": "zero_turnover",
                "method": "VF",
                "avg_regret_lower": 0.0,
                "avg_regret_upper": 0.0,
            },
        ]
    )
    out = release._safe_gap_closed_columns(summary)
    assert out["gap_closed_lower_pct"].isna().all()
    assert out["gap_closed_upper_pct"].isna().all()
    assert set(out["gap_closed_lower_status"]) == {"UNDEFINED_ZERO_BOOKED_REGRET"}
    assert set(out["gap_closed_upper_status"]) == {"UNDEFINED_ZERO_BOOKED_REGRET"}


def test_positive_booked_regret_keeps_gap_closed_defined() -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario": "regular",
                "method": "BOOKED",
                "avg_regret_lower": 100.0,
                "avg_regret_upper": 120.0,
            },
            {
                "scenario": "regular",
                "method": "VF",
                "avg_regret_lower": 50.0,
                "avg_regret_upper": 60.0,
            },
        ]
    )
    out = release._safe_gap_closed_columns(summary)
    vf = out[out["method"] == "VF"].iloc[0]
    assert np.isfinite(float(vf["gap_closed_lower_pct"]))
    assert np.isfinite(float(vf["gap_closed_upper_pct"]))
    assert vf["gap_closed_lower_status"] == "DEFINED"
    assert vf["gap_closed_upper_status"] == "DEFINED"
