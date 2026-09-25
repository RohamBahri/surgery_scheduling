#!/usr/bin/env python3
"""Single supported entry point for the final paper experiment.

Usage
-----
Solve behavior-independent training weekly plans once::

    python run_final_paper.py shared-plans --artifact-root artifacts/shared_plans [...]

Train one behavioral scenario using the shared weekly plans::

    python run_final_paper.py train \
      --shared-plans-root artifacts/shared_plans \
      --scenario-name primary --alpha 0.8 --h 30 \
      [arguments accepted by run_final_paper_training.py]

Primary one-shot holdout Stage 2::

    python run_final_paper.py evaluate [arguments accepted by run_final_paper_evaluation.py]

Required structural sensitivities::

    python run_final_paper.py sensitivities [arguments accepted by run_final_paper_required_sensitivities.py]

Existing structural/solver preflight::

    python run_final_paper.py preflight [arguments accepted by run_final_paper_preflight.py]

The wrapper installs final review hardening, all-path numerical accounting guards,
spawn-safe persistent Gurobi logging, shared behavior-independent training plans,
and arbitrary valid behavioral (alpha, h) scenarios. Do not execute the
stage/legacy experiment scripts directly for the final paper run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    """Remove wrapper-owned --name value / --name=value options."""

    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        direct = next((name for name in names if token == name), None)
        inline = next((name for name in names if token.startswith(name + "=")), None)
        if inline is not None:
            i += 1
            continue
        if direct is not None:
            if i + 1 >= len(argv):
                raise SystemExit(f"{direct} requires a value")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _dispatch_main(module, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "stage.py")] + argv
        module.main()
    finally:
        sys.argv = old


def _rewrite_frozen_next_step(root: Path) -> None:
    """Ensure the frozen bundle never instructs users to bypass the wrapper."""

    path = Path(root) / "TRAINING_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["next_step"] = (
        "Review all Stage-1 diagnostics. Then run Stage 2 exactly once with "
        "`python run_final_paper.py evaluate ...` from a clean git worktree at "
        "the frozen commit. Do not execute run_final_paper_evaluation.py directly."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _train(argv: list[str]) -> None:
    import run_final_paper_training as training

    root_arg = _arg_value(argv, "--artifact-root")
    if not root_arg:
        raise SystemExit("train requires --artifact-root")
    alpha = float(_arg_value(argv, "--alpha") or 0.8)
    h = float(_arg_value(argv, "--h") or 30.0)
    scenario_name = _arg_value(argv, "--scenario-name") or f"alpha_{alpha:g}_h_{h:g}"
    shared_arg = _arg_value(argv, "--shared-plans-root")
    shared_root = None if shared_arg is None else Path(shared_arg).resolve()
    stage_argv = _strip_value_args(
        argv,
        {"--alpha", "--h", "--scenario-name", "--shared-plans-root"},
    )

    hardening.install_training_fixes(training)
    shared.install_spawn_safe_weekly_logging()
    shared.configure_training_scenario(
        training,
        alpha=alpha,
        h=h,
        scenario_name=scenario_name,
        shared_root=shared_root,
    )
    # Install every guarded worker after shared support has replaced the worker
    # source functions; this survives the stage's internal initialization and
    # macOS ProcessPoolExecutor spawn.
    release_guard.install_reviewed_guards()
    shared.install_weekly_logging()

    _dispatch_main(training, stage_argv)
    hardening.stamp_training_bundle(Path(root_arg))
    numeric_guard.stamp_training_bundle(Path(root_arg))
    release_guard.stamp_training_bundle(Path(root_arg))
    shared.stamp_training_bundle(
        Path(root_arg),
        scenario_name=scenario_name,
        alpha=alpha,
        h=h,
        shared_root=shared_root,
    )
    _rewrite_frozen_next_step(Path(root_arg))
    print(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED_REVIEWED",
                "scenario": {"name": scenario_name, "alpha": alpha, "h": h},
                "shared_plans_root": None if shared_root is None else str(shared_root),
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
                "release_guard_version": release_guard.RELEASE_GUARD_VERSION,
                "shared_plans_version": shared.SHARED_PLANS_VERSION,
                "artifact_root": str(Path(root_arg).resolve()),
            },
            indent=2,
        )
    )


def _evaluate(argv: list[str]) -> None:
    import run_final_paper_evaluation as evaluation

    train_arg = _arg_value(argv, "--training-artifact-root")
    eval_arg = _arg_value(argv, "--artifact-root")
    if not train_arg or not eval_arg:
        raise SystemExit("evaluate requires --training-artifact-root and --artifact-root")
    hardening.verify_training_finalization(Path(train_arg))
    numeric_guard.verify_training_bundle(Path(train_arg))
    release_guard.verify_training_bundle(Path(train_arg))
    shared.verify_shared_plan_provenance(Path(train_arg))
    shared.install_flexible_behavior_validation()
    shared.install_spawn_safe_weekly_logging()
    hardening.install_evaluation_fixes(evaluation)
    # Stage 2 also calls the runtime worker for the holdout oracle, so install
    # all workers here rather than only the deterministic policy worker.
    release_guard.install_reviewed_guards(evaluation_module=evaluation)
    shared.install_weekly_logging()
    _dispatch_main(evaluation, argv)
    hardening.write_benchmark_interpretation(Path(eval_arg))
    print(
        json.dumps(
            {
                "status": "HOLDOUT_EVALUATION_COMPLETE_REVIEWED",
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
                "release_guard_version": release_guard.RELEASE_GUARD_VERSION,
                "shared_plans_version": shared.SHARED_PLANS_VERSION,
                "artifact_root": str(Path(eval_arg).resolve()),
            },
            indent=2,
        )
    )


def _sensitivities(argv: list[str]) -> None:
    import run_final_paper_required_sensitivities as runner

    train_arg = _arg_value(argv, "--training-artifact-root")
    root_arg = _arg_value(argv, "--artifact-root")
    if not train_arg or not root_arg:
        raise SystemExit("sensitivities requires --training-artifact-root and --artifact-root")
    hardening.verify_training_finalization(Path(train_arg))
    numeric_guard.verify_training_bundle(Path(train_arg))
    release_guard.verify_training_bundle(Path(train_arg))
    shared.verify_shared_plan_provenance(Path(train_arg))
    shared.install_flexible_behavior_validation()
    shared.install_spawn_safe_weekly_logging()
    hardening.install_sensitivity_fixes(runner)
    release_guard.install_reviewed_guards(sensitivity_runner_module=runner)
    shared.install_weekly_logging()
    _dispatch_main(runner, argv)
    print(
        json.dumps(
            {
                "status": "REQUIRED_SENSITIVITIES_COMPLETE_REVIEWED",
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
                "release_guard_version": release_guard.RELEASE_GUARD_VERSION,
                "shared_plans_version": shared.SHARED_PLANS_VERSION,
                "artifact_root": str(Path(root_arg).resolve()),
            },
            indent=2,
        )
    )


def _preflight(argv: list[str]) -> None:
    import run_final_paper_preflight as preflight

    shared.install_spawn_safe_weekly_logging()
    # The solver check exercises the same all-path numerical worker installation
    # used by the expensive training and evaluation stages.
    release_guard.install_reviewed_guards()
    shared.install_weekly_logging()
    _dispatch_main(preflight, argv)


def _shared_plans(argv: list[str]) -> None:
    import run_final_paper_shared_plans as runner

    _dispatch_main(runner, argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    stage = sys.argv[1].strip().lower()
    argv = sys.argv[2:]
    if stage in {"shared-plans", "shared_plans", "backbone"}:
        _shared_plans(argv)
    elif stage == "train":
        _train(argv)
    elif stage in {"evaluate", "evaluation"}:
        _evaluate(argv)
    elif stage in {"sensitivities", "sensitivity"}:
        _sensitivities(argv)
    elif stage == "preflight":
        _preflight(argv)
    else:
        raise SystemExit(
            f"Unknown stage {stage!r}. Use shared-plans, train, evaluate, sensitivities, or preflight."
        )


if __name__ == "__main__":
    main()
