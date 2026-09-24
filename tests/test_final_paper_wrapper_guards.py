from __future__ import annotations

import final_paper_numeric_guard as numeric
import final_paper_release_guard as release
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper as wrapper
import run_final_paper_evaluation as evaluation
import run_final_paper_training as training


def test_train_wrapper_installs_runtime_and_deterministic_guards(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_main():
        observed["runtime"] = runtime._solve_process_task
        observed["deterministic"] = science.deterministic_eval_worker

    monkeypatch.setattr(training, "main", fake_main)
    monkeypatch.setattr(wrapper.hardening, "stamp_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.numeric_guard, "stamp_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.release_guard, "stamp_training_bundle", lambda root: None)

    wrapper._train(["--artifact-root", str(tmp_path)])

    assert observed["runtime"] is numeric.training_process_task
    assert observed["deterministic"] is numeric.deterministic_eval_worker


def test_evaluate_wrapper_installs_oracle_and_policy_guards(monkeypatch, tmp_path) -> None:
    observed = {}
    training_root = tmp_path / "train"
    eval_root = tmp_path / "eval"

    def fake_main():
        observed["runtime"] = runtime._solve_process_task
        observed["deterministic"] = science.deterministic_eval_worker

    monkeypatch.setattr(evaluation, "main", fake_main)
    monkeypatch.setattr(wrapper.hardening, "verify_training_finalization", lambda root: {})
    monkeypatch.setattr(wrapper.numeric_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.release_guard, "verify_training_bundle", lambda root: None)
    monkeypatch.setattr(wrapper.hardening, "write_benchmark_interpretation", lambda root: None)

    wrapper._evaluate(
        [
            "--training-artifact-root",
            str(training_root),
            "--artifact-root",
            str(eval_root),
        ]
    )

    # The holdout oracle uses the runtime worker; policy scheduling uses the
    # deterministic worker. Both must be guarded before evaluation.main starts.
    assert observed["runtime"] is numeric.training_process_task
    assert observed["deterministic"] is numeric.deterministic_eval_worker
