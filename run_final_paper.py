#!/usr/bin/env python3
"""Supported entry point for the paper experiment.

Commands
--------
shared-plans   solve/refine behavior-independent training Oracle and BOOKED plans
train          train one registered behavioral scenario
evaluate       evaluate one registered training bundle; evaluation may be repeated
sensitivities  required structural sensitivities
preflight      structural/solver preflight

Do not execute the implementation-stage scripts directly for the paper run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import experiment_protocol as protocol
import experiment_audit as audit  # side effect: installs resilience before shared worker capture
import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric_guard
import final_paper_release_guard as release_guard
import final_paper_shared_plans as shared


def _arg_value(argv: list[str], name: str) -> str | None:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _strip_value_args(argv: list[str], names: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        direct = next((name for name in names if token == name), None)
        inline = next((name for name in names if token.startswith(name + "=")), None)
        if inline is not None:
            i += 1; continue
        if direct is not None:
            if i + 1 >= len(argv):
                raise SystemExit(f"{direct} requires a value")
            i += 2; continue
        out.append(token); i += 1
    return out


def _strip_flags(argv: list[str], names: set[str]) -> list[str]:
    return [x for x in argv if x not in names]


def _dispatch_main(module, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "stage.py")] + argv
        module.main()
    finally:
        sys.argv = old


def _install_review_stack() -> None:
    shared.install_flexible_behavior_validation()
    shared.install_spawn_safe_weekly_logging()
    protocol.install_numeric_policy()
    protocol.install_behavioral_protocol()
    protocol.install_shared_plan_validation(shared)
    release_guard.install_reviewed_guards()
    shared.install_weekly_logging()


def _rewrite_frozen_next_step(root: Path) -> None:
    path = Path(root) / "TRAINING_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["next_step"] = (
        "Review the completed training diagnostics, then evaluate any registered scenario directly. "
        "Holdout evaluation is repeatable; no experiment seal is required."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_shared_provenance(training_root: Path, shared_root: Path) -> None:
    manifest = Path(shared_root).resolve() / "SHARED_PLANS_MANIFEST.json"
    if not manifest.exists():
        raise RuntimeError(f"Missing shared-plan manifest: {manifest}")
    payload = {
        "shared_plans_root": str(Path(shared_root).resolve()),
        "manifest_sha256": protocol.sha256_file(manifest),
        "registry_sha256": protocol.registry_hash(),
        "git_head": protocol.git_head(),
    }
    (Path(training_root) / "SHARED_PLAN_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fingerprint_protocol_artifacts(training_root: Path) -> None:
    root = Path(training_root)
    freeze_path = root / "TRAINING_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["experiment_protocol_version"] = protocol.PROTOCOL_VERSION
    freeze["experiment_registry_sha256"] = protocol.registry_hash()
    fps = dict(freeze.get("artifact_fingerprints", {}))
    for name in ("EXPERIMENT_REGISTRY.json", "SHARED_PLAN_PROVENANCE.json"):
        p = root / name
        if not p.exists():
            raise RuntimeError(f"Missing protocol artifact before freeze: {name}")
        fps[name] = protocol.sha256_file(p)
    freeze["artifact_fingerprints"] = fps
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_diagonal_matrix_files(eval_root: Path, scenario: str) -> None:
    import pandas as pd
    summary_path = Path(eval_root) / "FINAL_HOLDOUT_SUMMARY.csv"
    weekly_path = Path(eval_root) / "FINAL_HOLDOUT_WEEKLY.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df.insert(0, "response_scenario", scenario)
        df.insert(0, "training_scenario", scenario)
        df.to_csv(Path(eval_root) / "RESPONSE_MATRIX_DIAGONAL_SUMMARY.csv", index=False)
    if weekly_path.exists():
        df = pd.read_csv(weekly_path)
        df.insert(0, "response_scenario", scenario)
        df.insert(0, "training_scenario", scenario)
        df.to_csv(Path(eval_root) / "RESPONSE_MATRIX_DIAGONAL_WEEKLY.csv", index=False)


def _train(argv: list[str]) -> None:
    import run_final_paper_training as training

    root_arg = _arg_value(argv, "--artifact-root")
    if not root_arg:
        raise SystemExit("train requires --artifact-root")
    scenario_name = _arg_value(argv, "--scenario-name")
    if not scenario_name:
        raise SystemExit("train requires --scenario-name from experiment_registry.json")
    row = protocol.registered_scenario(scenario_name, purpose="train")
    alpha = float(_arg_value(argv, "--alpha") or row["alpha"])
    h = float(_arg_value(argv, "--h") or row["h"])
    protocol.validate_registered_parameters(scenario_name, alpha, h, purpose="train")
    shared_arg = _arg_value(argv, "--shared-plans-root")
    if not shared_arg:
        raise SystemExit("Registered training requires --shared-plans-root")
    shared_root = Path(shared_arg).resolve()
    stage_argv = _strip_value_args(argv, {"--alpha", "--h", "--scenario-name", "--shared-plans-root"})

    hardening.install_training_fixes(training)
    _install_review_stack()
    shared.configure_training_scenario(training, alpha=alpha, h=h, scenario_name=scenario_name, shared_root=shared_root)
    _install_review_stack()
    _dispatch_main(training, stage_argv)
    hardening.stamp_training_bundle(Path(root_arg))
    numeric_guard.stamp_training_bundle(Path(root_arg))
    release_guard.stamp_training_bundle(Path(root_arg))
    shared.stamp_training_bundle(Path(root_arg), scenario_name=scenario_name, alpha=alpha, h=h, shared_root=shared_root)
    _write_shared_provenance(Path(root_arg), shared_root)
    protocol.stamp_training_registry(Path(root_arg), scenario_name=scenario_name, alpha=alpha, h=h)
    _fingerprint_protocol_artifacts(Path(root_arg))
    _rewrite_frozen_next_step(Path(root_arg))
    print(json.dumps({
        "status": "TRAINING_COMPLETE_REVIEWED",
        "scenario": row,
        "registry_sha256": protocol.registry_hash(),
        "shared_plans_root": str(shared_root),
        "artifact_root": str(Path(root_arg).resolve()),
    }, indent=2))


def _evaluate(argv: list[str]) -> None:
    import run_final_paper_evaluation as evaluation

    train_arg = _arg_value(argv, "--training-artifact-root")
    eval_arg = _arg_value(argv, "--artifact-root")
    experiment_arg = _arg_value(argv, "--experiment-root")
    if not train_arg or not eval_arg or not experiment_arg:
        raise SystemExit("evaluate requires --experiment-root, --training-artifact-root and --artifact-root")
    training_root, eval_root, experiment_root = Path(train_arg), Path(eval_arg), Path(experiment_arg)
    stamp_path = training_root / "EXPERIMENT_REGISTRY.json"
    if not stamp_path.exists():
        raise RuntimeError(f"Training bundle has no experiment registry stamp: {stamp_path}")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    scenario = str(stamp["scenario"]["name"])
    protocol.validate_registered_parameters(
        scenario,
        float(stamp["scenario"]["alpha"]),
        float(stamp["scenario"]["h"]),
        purpose="evaluate",
    )

    hardening.verify_training_finalization(training_root)
    _install_review_stack()
    numeric_guard.verify_training_bundle(training_root)
    release_guard.verify_training_bundle(training_root)
    shared.verify_shared_plan_provenance(training_root)
    hardening.install_evaluation_fixes(evaluation)
    _install_review_stack()
    evaluation._run_response_sensitivities = protocol.registered_response_sensitivity_runner
    stage_argv = _strip_value_args(argv, {"--experiment-root"})
    stage_argv = _strip_flags(stage_argv, {"--resume"})
    if "--run-response-sensitivities" not in stage_argv:
        stage_argv.append("--run-response-sensitivities")
    _dispatch_main(evaluation, stage_argv)
    hardening.write_benchmark_interpretation(eval_root)
    _write_diagonal_matrix_files(eval_root, scenario)
    print(json.dumps({
        "status": "HOLDOUT_EVALUATION_COMPLETE_REVIEWED",
        "training_scenario": scenario,
        "registry_sha256": protocol.registry_hash(),
        "artifact_root": str(eval_root.resolve()),
    }, indent=2))


def _evaluate_matrix(argv: list[str]) -> None:
    import run_final_paper_matrix_evaluation as runner
    runner.main(argv)


def _sensitivities(argv: list[str]) -> None:
    import run_final_paper_required_sensitivities as runner
    train_arg = _arg_value(argv, "--training-artifact-root")
    root_arg = _arg_value(argv, "--artifact-root")
    if not train_arg or not root_arg:
        raise SystemExit("sensitivities requires --training-artifact-root and --artifact-root")
    hardening.verify_training_finalization(Path(train_arg))
    _install_review_stack()
    numeric_guard.verify_training_bundle(Path(train_arg))
    release_guard.verify_training_bundle(Path(train_arg))
    shared.verify_shared_plan_provenance(Path(train_arg))
    hardening.install_sensitivity_fixes(runner)
    _install_review_stack()
    # The sensitivity hardening hook installs the legacy robust worker first.
    # Re-select the reviewed numeric worker last so finite solver-vs-recomputed
    # Phi roundoff remains diagnostic, exactly as in the matrix evaluator.
    release_guard.install_reviewed_guards(sensitivity_runner_module=runner)
    _dispatch_main(runner, argv)
    print(json.dumps({"status": "REQUIRED_SENSITIVITIES_COMPLETE_REVIEWED", "artifact_root": str(Path(root_arg).resolve())}, indent=2))


def _preflight(argv: list[str]) -> None:
    import run_final_paper_preflight as preflight
    _install_review_stack()
    _dispatch_main(preflight, argv)


def _shared_plans(argv: list[str]) -> None:
    import run_final_paper_shared_plans as runner
    if not protocol.tracked_tree_clean():
        raise RuntimeError("Shared-plan generation/refinement requires a clean tracked tree")
    warm = _arg_value(argv, "--warm-start-root")
    protocol.configure_refinement_parent(None if warm is None else Path(warm))
    _install_review_stack()
    _dispatch_main(runner, argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__); return
    stage, argv = sys.argv[1].strip().lower(), sys.argv[2:]
    if stage in {"shared-plans", "shared_plans", "backbone"}:
        _shared_plans(argv)
    elif stage == "train":
        _train(argv)
    elif stage in {"evaluate", "evaluation"}:
        _evaluate(argv)
    elif stage == "matrix":
        _evaluate_matrix(argv)
    elif stage in {"sensitivities", "sensitivity"}:
        _sensitivities(argv)
    elif stage == "preflight":
        _preflight(argv)
    else:
        raise SystemExit("Unknown stage. Use shared-plans, train, evaluate, matrix, sensitivities, or preflight.")


if __name__ == "__main__":
    main()
