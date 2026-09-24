#!/usr/bin/env python3
"""Single supported entry point for the final paper experiment.

Usage
-----
Training-only Stage 1::

    python run_final_paper.py train [arguments accepted by run_final_paper_training.py]

Primary one-shot holdout Stage 2::

    python run_final_paper.py evaluate [arguments accepted by run_final_paper_evaluation.py]

Required structural sensitivities::

    python run_final_paper.py sensitivities [arguments accepted by run_final_paper_required_sensitivities.py]

Existing structural/solver preflight::

    python run_final_paper.py preflight [arguments accepted by run_final_paper_preflight.py]

This wrapper installs the final review hardening and the scale-aware numerical
accounting guard before dispatching to the existing stage modules.  Do not
execute the legacy experiment scripts directly for the final paper run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric_guard


def _arg_value(argv: list[str], name: str) -> str | None:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _dispatch_main(module, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "stage.py")] + argv
        module.main()
    finally:
        sys.argv = old


def _train(argv: list[str]) -> None:
    import run_final_paper_training as training

    hardening.install_training_fixes(training)
    numeric_guard.install_training_guard()
    root_arg = _arg_value(argv, "--artifact-root")
    if not root_arg:
        raise SystemExit("train requires --artifact-root")
    _dispatch_main(training, argv)
    hardening.stamp_training_bundle(Path(root_arg))
    numeric_guard.stamp_training_bundle(Path(root_arg))
    print(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED_REVIEWED",
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
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
    hardening.install_evaluation_fixes(evaluation)
    numeric_guard.install_evaluation_guard()
    _dispatch_main(evaluation, argv)
    hardening.write_benchmark_interpretation(Path(eval_arg))
    print(
        json.dumps(
            {
                "status": "HOLDOUT_EVALUATION_COMPLETE_REVIEWED",
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
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
    hardening.install_sensitivity_fixes(runner)
    numeric_guard.install_sensitivity_guard()
    _dispatch_main(runner, argv)
    print(
        json.dumps(
            {
                "status": "REQUIRED_SENSITIVITIES_COMPLETE_REVIEWED",
                "finalization_fixes_version": hardening.FINALIZATION_FIXES_VERSION,
                "numeric_guard_version": numeric_guard.NUMERIC_GUARD_VERSION,
                "artifact_root": str(Path(root_arg).resolve()),
            },
            indent=2,
        )
    )


def _preflight(argv: list[str]) -> None:
    import run_final_paper_preflight as preflight

    # The solver check uses the same process worker as Stage 1, so install the
    # reviewed numeric guard here as well.
    numeric_guard.install_training_guard()
    _dispatch_main(preflight, argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    stage = sys.argv[1].strip().lower()
    argv = sys.argv[2:]
    if stage == "train":
        _train(argv)
    elif stage in {"evaluate", "evaluation"}:
        _evaluate(argv)
    elif stage in {"sensitivities", "sensitivity"}:
        _sensitivities(argv)
    elif stage == "preflight":
        _preflight(argv)
    else:
        raise SystemExit(f"Unknown stage {stage!r}. Use train, evaluate, sensitivities, or preflight.")


if __name__ == "__main__":
    main()
