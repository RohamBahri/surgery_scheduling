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



def test_training_math_compatibility_allows_provenance_only_commit_change(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "git_head", lambda: "current")
    identities = {
        "old": {"a.py": "same", "b.py": "same"},
        "current": {"a.py": "same", "b.py": "same"},
        "changed": {"a.py": "same", "b.py": "different"},
    }
    monkeypatch.setattr(
        protocol,
        "_training_math_source_identity_at_head",
        lambda head: dict(identities[head]),
    )
    monkeypatch.setattr(protocol, "TRAINING_MATH_SOURCE_FILES", ("a.py", "b.py"))

    assert protocol.assert_training_math_compatible("old") == identities["current"]
    with pytest.raises(RuntimeError, match="different training mathematics"):
        protocol.assert_training_math_compatible("changed")


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


def test_evaluation_amendment_defers_narrower_tolerance() -> None:
    amendment = protocol.load_amendment()
    assert amendment["status"] == "SCOPE_AND_EXECUTION_AMENDMENT"
    assert amendment["active_scenarios"] == [
        "primary",
        "lower_responsiveness",
        "broader_tolerance",
    ]
    assert amendment["deferred_scenarios"] == ["narrower_tolerance"]
    assert [x["name"] for x in protocol.active_scenarios(purpose="train")] == [
        "primary",
        "lower_responsiveness",
        "broader_tolerance",
    ]
    assert [x["name"] for x in protocol.active_scenarios(purpose="evaluate")] == [
        "primary",
        "lower_responsiveness",
        "broader_tolerance",
    ]


def test_evaluation_amendment_uses_efficient_matrix_engine() -> None:
    amendment = protocol.load_amendment()
    execution = amendment["evaluation_execution"]
    assert execution["engine"] == "holdout_matrix_engine_2026_10_01_v1"
    assert execution["policy_solver"]["total_work_limit_per_site"] == 1200.0
    assert execution["policy_solver"]["mip_gap"] == 0.0005
    assert execution["policy_solver"]["threads_per_site"] == 1
    assert execution["policy_solver"]["seed"] == 42
