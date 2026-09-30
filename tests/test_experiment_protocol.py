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


def test_solver_phi_difference_is_diagnostic_not_a_live_kill_switch() -> None:
    for a, b in [
        (21074.971248, 21074.9452086),
        (17707.6214425, 17707.5074103),
        (1000.0, 900.0),
    ]:
        protocol.assert_phi_accounting_close(a, b, context="live-diagnostic")

    with pytest.raises(AssertionError, match="non-finite Phi accounting"):
        protocol.assert_phi_accounting_close(float("nan"), 1.0, context="nonfinite")


def test_numeric_tolerance_is_bounded_below_cost_quantum() -> None:
    assert protocol.phi_accounting_tolerance(1.0, 1.0) == pytest.approx(0.1)
    assert protocol.phi_accounting_tolerance(200000.0, 200000.0) == pytest.approx(0.4)
    assert protocol.phi_accounting_tolerance(1e9, 1e9) == pytest.approx(0.5)


def test_shared_plan_math_compatibility_ignores_provenance_only_hashes(monkeypatch) -> None:
    current = {
        "src/solvers/fixed_capacity.py": "a",
        "src/core/column.py": "b",
        "src/core/config.py": "c",
        "run_final_paper_experiment.py": "d",
        "final_paper_scientific_fixes.py": "e",
    }
    monkeypatch.setattr(protocol, "_model_source_identity", lambda: dict(current))
    legacy_manifest = {
        "source_git_head": "old-commit-is-informational",
        "model_source_sha256": {
            **current,
            "experiment_protocol.py": "old-protocol-hash",
            "final_paper_numeric_guard.py": "old-numeric-hash",
            "experiment_registry.json": "old-extra-hash",
        },
    }
    protocol._assert_shared_math_compatible(legacy_manifest)

    legacy_manifest["model_source_sha256"]["src/core/column.py"] = "changed-math"
    with pytest.raises(RuntimeError, match="mathematical planner/cost source changed"):
        protocol._assert_shared_math_compatible(legacy_manifest)


def test_projected_benchmark_respects_display_cap() -> None:
    a = SimpleNamespace(
        booked=np.array([100.0, 20.0]),
        error=np.array([200.0, -200.0]),
    )
    s = SimpleNamespace(alpha=0.8, h=100.0, display_cap=30.0)
    planning, corr = protocol.implementable_oracle_planning(a, s)
    assert corr[0] == pytest.approx(24.0)
    assert corr[1] == pytest.approx(-15.2)
    assert planning[1] == pytest.approx(4.8)


def test_registered_parameters_reject_posthoc_unregistered_scenario() -> None:
    protocol.validate_registered_parameters("primary", 0.8, 30.0, purpose="train")
    with pytest.raises(RuntimeError):
        protocol.validate_registered_parameters("primary", 0.7, 30.0, purpose="train")
    with pytest.raises(RuntimeError):
        protocol.registered_scenario("new_after_results", purpose="evaluate")
