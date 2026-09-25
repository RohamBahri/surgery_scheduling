from __future__ import annotations

import experiment_protocol as protocol
import final_paper_numeric_guard as numeric
import final_paper_release_guard as release
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import final_paper_shared_plans as shared
import run_final_paper as wrapper
import run_final_paper_evaluation as evaluation
import run_final_paper_experiment as final
import run_final_paper_training as training


def test_train_wrapper_installs_spawn_safe_protocol_guards(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_main():
        observed["runtime"] = runtime._solve_process_task
        observed["deterministic"] = science.deterministic_eval_worker
        observed["tolerance"] = numeric.phi_accounting_tolerance(21074.971248, 21074.9452086)

    monkeypatch.setattr(training, "main", fake_main)
    monkeypatch.setattr(wrapper.hardening, "stamp_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.numeric_guard, "stamp_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.release_guard, "stamp_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.shared, "stamp_training_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper.shared, "configure_training_scenario", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper.protocol, "stamp_training_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper, "_rewrite_frozen_next_step", lambda root: None)

    wrapper._train([
        "--artifact-root", str(tmp_path / "train"),
        "--shared-plans-root", str(tmp_path / "shared"),
        "--scenario-name", "primary",
        "--alpha", "0.8", "--h", "30",
    ])

    assert observed["runtime"] is protocol.training_process_task
    assert observed["deterministic"] is protocol.deterministic_eval_worker
    assert observed["tolerance"] >= 0.0260393


def test_evaluate_wrapper_requires_sealed_bundle_and_installs_protocol_guards(monkeypatch, tmp_path) -> None:
    observed = {}
    training_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    experiment_root = tmp_path / "experiment"

    def fake_main():
        observed["runtime"] = runtime._solve_process_task
        observed["deterministic"] = science.deterministic_eval_worker

    monkeypatch.setattr(evaluation, "main", fake_main)
    monkeypatch.setattr(wrapper.protocol, "verify_sealed_bundle", lambda *args, **kwargs: ({}, "primary"))
    monkeypatch.setattr(wrapper.protocol, "update_consumption", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper.hardening, "verify_training_finalization", lambda root: {})
    monkeypatch.setattr(wrapper.numeric_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.release_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.shared, "verify_shared_plan_provenance", lambda root: None)
    monkeypatch.setattr(wrapper.hardening, "write_benchmark_interpretation", lambda root: None)

    wrapper._evaluate([
        "--experiment-root", str(experiment_root),
        "--training-artifact-root", str(training_root),
        "--artifact-root", str(eval_root),
    ])

    assert observed["runtime"] is protocol.training_process_task
    assert observed["deterministic"] is protocol.deterministic_eval_worker


def test_protocol_numeric_guard_survives_stage_reinstallation() -> None:
    shared.install_spawn_safe_weekly_logging()
    protocol.install_numeric_policy()
    release.install_reviewed_guards()
    final.install_final_adapter()
    runtime.apply_runtime_fixes()
    science.apply_scientific_fixes()
    # Reinstalling the protocol after stage initialization is the supported order.
    shared.install_spawn_safe_weekly_logging()
    protocol.install_numeric_policy()
    release.install_reviewed_guards()

    assert runtime._solve_process_task is protocol.training_process_task
    assert science.deterministic_eval_worker is protocol.deterministic_eval_worker
    assert numeric.phi_accounting_tolerance(21074.971248, 21074.9452086) >= 0.0260393
    with __import__("pytest").raises(AssertionError):
        numeric.assert_phi_accounting_close(21074.0, 21073.0, context="material")
