#!/usr/bin/env python3
"""Supported entry point for the paper experiment.

Commands
--------
shared-plans   solve/refine behavior-independent training Oracle and BOOKED plans
train          train one registered behavioral scenario
seal           seal all accepted registered training bundles before holdout access
evaluate       evaluate one sealed training bundle; use --resume only after failure
sensitivities  required structural sensitivities
preflight      structural/solver preflight

Do not execute the implementation-stage scripts directly for the paper run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import experiment_protocol as protocol
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


def _has_flag(argv: list[str], name: str) -> bool:
    return name in argv


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
        "Train every scenario in experiment_registry.json using the same immutable shared-plan artifact, "
        "review diagnostics, then seal all accepted bundles with `python run_final_paper.py seal ...` "
        "before any holdout evaluation."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    shared.configure_training_scenario(
        training, alpha=alpha, h=h, scenario_name=scenario_name, shared_root=shared_root
    )
    # Reinstall because configure_training_scenario changes runtime hooks.
    _install_review_stack()
    _dispatch_main(training, stage_argv)
    hardening.stamp_training_bundle(Path(root_arg))
    numeric_guard.stamp_training_bundle(Path(root_arg))
    release_guard.stamp_training_bundle(Path(root_arg))
    shared.stamp_training_bundle(
        Path(root_arg), scenario_name=scenario_name, alpha=alpha, h=h, shared_root=shared_root
    )
    protocol.stamp_training_registry(Path(root_arg), scenario_name=scenario_name, alpha=alpha, h=h)
    _rewrite_frozen_next_step(Path(root_arg))
    print(json.dumps({
        "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED_REVIEWED",
        "scenario": row,
        "registry_sha256": protocol.registry_hash(),
        "shared_plans_root": str(shared_root),
        "artifact_root": str(Path(root_arg).resolve()),
    }, indent=2))


def _seal(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="run_final_paper.py seal")
    p.add_argument("--experiment-root", required=True)
    p.add_argument("--shared-plans-root", required=True)
    p.add_argument("--training-root", action="append", required=True)
    args = p.parse_args(argv)
    seal = protocol.seal_experiment(
        Path(args.experiment_root), [Path(x) for x in args.training_root], Path(args.shared_plans_root)
    )
    print(json.dumps({
        "status": seal["status"],
        "experiment_root": str(Path(args.experiment_root).resolve()),
        "registry_sha256": seal["registry_sha256"],
        "accepted_scenarios": sorted(seal["accepted_training_bundles"]),
    }, indent=2))


def _evaluate(argv: list[str]) -> None:
    import run_final_paper_evaluation as evaluation

    train_arg = _arg_value(argv, "--training-artifact-root")
    eval_arg = _arg_value(argv, "--artifact-root")
    experiment_arg = _arg_value(argv, "--experiment-root")
    if not train_arg or not eval_arg or not experiment_arg:
        raise SystemExit("evaluate requires --experiment-root, --training-artifact-root and --artifact-root")
    training_root, eval_root, experiment_root = Path(train_arg), Path(eval_arg), Path(experiment_arg)
    resume = _has_flag(argv, "--resume")
    _, scenario = protocol.verify_sealed_bundle(experiment_root, training_root)
    if resume:
        protocol.prepare_exact_restart(experiment_root, training_root, eval_root)
    protocol.update_consumption(experiment_root, scenario, state="STARTING", evaluation_root=eval_root)

    hardening.verify_training_finalization(training_root)
    _install_review_stack()
    numeric_guard.verify_training_bundle(training_root)
    release_guard.verify_training_bundle(training_root)
    shared.verify_shared_plan_provenance(training_root)
    hardening.install_evaluation_fixes(evaluation)
    _install_review_stack()
    # Generate off-diagonal response conditions from the committed registry at
    # the same primary evaluation budget. The diagonal is the normal evaluation.
    evaluation._run_response_sensitivities = protocol.registered_response_sensitivity_runner
    stage_argv = _strip_value_args(argv, {"--experiment-root"})
    stage_argv = _strip_flags(stage_argv, {"--resume"})
    if "--run-response-sensitivities" not in stage_argv:
        stage_argv.append("--run-response-sensitivities")
    try:
        _dispatch_main(evaluation, stage_argv)
    except Exception:
        protocol.update_consumption(experiment_root, scenario, state="FAILED_RESTARTABLE", evaluation_root=eval_root)
        raise
    hardening.write_benchmark_interpretation(eval_root)
    protocol.update_consumption(experiment_root, scenario, state="COMPLETE", evaluation_root=eval_root)
    print(json.dumps({
        "status": "HOLDOUT_EVALUATION_COMPLETE_REVIEWED",
        "training_scenario": scenario,
        "registry_sha256": protocol.registry_hash(),
        "artifact_root": str(eval_root.resolve()),
    }, indent=2))


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
    elif stage == "seal":
        _seal(argv)
    elif stage in {"evaluate", "evaluation"}:
        _evaluate(argv)
    elif stage in {"sensitivities", "sensitivity"}:
        _sensitivities(argv)
    elif stage == "preflight":
        _preflight(argv)
    else:
        raise SystemExit("Unknown stage. Use shared-plans, train, seal, evaluate, sensitivities, or preflight.")


if __name__ == "__main__":
    main()
