from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import experiment_protocol as protocol


def test_registry_is_frozen_and_has_training_evaluation_matrix() -> None:
    reg = protocol.load_registry()
    names = [x["name"] for x in reg["scenarios"]]
    assert names == [
        "primary",
        "lower_responsiveness",
        "narrower_tolerance",
        "broader_tolerance",
    ]
    assert all(x["train"] and x["evaluate"] for x in reg["scenarios"])
    assert reg["evaluation_protocol"]["work_limit_per_site"] == 1200.0
    assert reg["evaluation_protocol"]["mip_gap"] == 0.0005


def test_late_vf_roundoff_is_accepted_but_material_error_is_not() -> None:
    # Exact failure after almost 12 hours on week 57, vf_candidate_3.
    a = 21074.971248
    b = 21074.9452086
    assert abs(a - b) == pytest.approx(0.0260394, abs=1e-7)
    assert protocol.phi_accounting_tolerance(a, b) >= abs(a - b)
    protocol.assert_phi_accounting_close(a, b, context="late-vf-regression")

    # A real one-cost-unit accounting difference must still fail.
    with pytest.raises(AssertionError, match="decomposed Phi mismatch"):
        protocol.assert_phi_accounting_close(a, a - 1.0, context="material")


def test_numeric_tolerance_is_bounded_below_cost_quantum() -> None:
    assert protocol.phi_accounting_tolerance(1.0, 1.0) == pytest.approx(0.1)
    assert protocol.phi_accounting_tolerance(200000.0, 200000.0) == pytest.approx(0.4)
    assert protocol.phi_accounting_tolerance(1e9, 1e9) == pytest.approx(0.5)


def test_projected_benchmark_respects_display_cap() -> None:
    a = SimpleNamespace(
        booked=np.array([100.0, 20.0]),
        error=np.array([200.0, -200.0]),
    )
    s = SimpleNamespace(alpha=0.8, h=100.0, display_cap=30.0)
    planning, corr = protocol.implementable_oracle_planning(a, s)
    # h is 100 but the displayed recommendation can move by only 30 before alpha.
    assert corr[0] == pytest.approx(24.0)
    # The second case is also constrained by the 1-minute recommendation floor.
    assert corr[1] == pytest.approx(-15.2)
    assert planning[1] == pytest.approx(4.8)


def test_registered_parameters_reject_posthoc_unregistered_scenario() -> None:
    protocol.validate_registered_parameters("primary", 0.8, 30.0, purpose="train")
    with pytest.raises(RuntimeError):
        protocol.validate_registered_parameters("primary", 0.7, 30.0, purpose="train")
    with pytest.raises(RuntimeError):
        protocol.registered_scenario("new_after_results", purpose="evaluate")
