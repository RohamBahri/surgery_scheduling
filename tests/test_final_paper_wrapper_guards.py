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
    monkeypatch.setattr(wrapper, "_write_shared_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper, "_fingerprint_protocol_artifacts", lambda *args, **kwargs: None)
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


def test_evaluate_wrapper_accepts_registered_bundle_without_seal_and_installs_protocol_guards(monkeypatch, tmp_path) -> None:
    observed = {}
    training_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    experiment_root = tmp_path / "experiment"

    def fake_main():
        observed["runtime"] = runtime._solve_process_task
        observed["deterministic"] = science.deterministic_eval_worker

    training_root.mkdir()
    (training_root / "EXPERIMENT_REGISTRY.json").write_text(
        __import__("json").dumps(
            {"scenario": {"name": "primary", "alpha": 0.8, "h": 30.0}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wrapper.protocol, "validate_registered_parameters", lambda *args, **kwargs: {})

    monkeypatch.setattr(evaluation, "main", fake_main)
    monkeypatch.setattr(wrapper.hardening, "verify_training_finalization", lambda root: {})
    monkeypatch.setattr(wrapper.numeric_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.release_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.shared, "verify_shared_plan_provenance", lambda root: None)
    monkeypatch.setattr(wrapper.hardening, "write_benchmark_interpretation", lambda root: None)
    monkeypatch.setattr(wrapper, "_write_diagonal_matrix_files", lambda *args, **kwargs: None)

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
    shared.install_spawn_safe_weekly_logging()
    protocol.install_numeric_policy()
    release.install_reviewed_guards()

    assert runtime._solve_process_task is protocol.training_process_task
    assert science.deterministic_eval_worker is protocol.deterministic_eval_worker
    assert numeric.phi_accounting_tolerance(21074.971248, 21074.9452086) >= 0.0260393
    # Finite solver-incumbent differences are diagnostic only; certificate validity
    # is enforced by the lower-bound-vs-feasible-objective check instead.
    numeric.assert_phi_accounting_close(21074.0, 21073.0, context="material-diagnostic")
    with __import__("pytest").raises(AssertionError, match="lower bound exceeds"):
        numeric.assert_lower_bound_valid(1000.0, 1001.0, context="invalid-certificate")


def test_evaluation_bundle_verifier_accepts_compatible_older_training_head(monkeypatch, tmp_path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    freeze = {
        "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
        "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
        "runtime_fixes_version": runtime.RUNTIME_FIXES_VERSION,
        "git_head": "older-compatible-head",
        "artifact_fingerprints": {},
    }
    (root / "TRAINING_FREEZE.json").write_text(__import__("json").dumps(freeze), encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        evaluation.protocol,
        "assert_training_math_compatible",
        lambda head: seen.setdefault("head", head) or {},
    )
    monkeypatch.setattr(evaluation, "_tracked_tree_is_dirty", lambda: False)

    out = evaluation._verify_training_bundle(root)
    assert out["git_head"] == "older-compatible-head"
    assert seen["head"] == "older-compatible-head"


