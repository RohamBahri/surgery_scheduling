#!/usr/bin/env python3
"""Training-only gate for the final two-site paper experiment.

Run this first. It executes the complete 72-week training pipeline (including
RA, OS, VF and the optional saturation diagnostic), writes the frozen policies
and training artifacts, and exits immediately before any holdout optimization
or policy evaluation.

The accepted training bundle is fingerprinted at the holdout lock.  Do not run
holdout evaluation until that bundle has been reviewed and an evaluation-only
entry point has verified the same fingerprints.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import final_paper_runtime_fixes as fixes
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def _tracked_tree_is_dirty() -> bool:
    """Ignore untracked data/artifacts, but reject modified tracked source."""

    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        return unstaged != 0 or staged != 0
    except Exception:
        return False


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    args = base.parse_args()
    artifact_root = Path(args.artifact_root).resolve()

    # A final scientific run must not silently mix new outputs with stale files.
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise RuntimeError(
            f"Artifact root is not empty: {artifact_root}. Use a fresh directory for each final run."
        )
    if _tracked_tree_is_dirty():
        raise RuntimeError("Tracked repository files are modified. Commit or revert them before the final training run.")

    original = base.build_bundles

    def training_only_build_bundles(*args_, **kwargs_):
        offset = int(kwargs_.get("offset", 0))
        if offset >= 72:
            required = [
                artifact_root / "POLICIES.npz",
                artifact_root / "FEATURE_MANIFEST.json",
                artifact_root / "DATA_FREEZE.json",
            ]
            missing = [str(p.name) for p in required if not p.exists()]
            if missing:
                raise RuntimeError(f"Training reached the holdout lock without required artifacts: {missing}")

            fingerprints = {p.name: base.sha256_file(p) for p in required}
            payload = {
                "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                "script_version": final.SCRIPT_VERSION,
                "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
                "git_head": base.git_head(),
                "holdout_evaluation_run": False,
                "artifact_fingerprints": fingerprints,
                "next_step": (
                    "Review the complete training bundle. Holdout evaluation must load and verify these exact "
                    "policy/feature/data fingerprints; do not retrain accepted policies."
                ),
            }
            base.write_json(artifact_root / "TRAINING_FREEZE.json", payload)
            base.write_json(
                artifact_root / "RUN_STATUS.json",
                {
                    "status": "TRAINING_COMPLETE_HOLDOUT_LOCKED",
                    "holdout_consumed": False,
                    "script_version": final.SCRIPT_VERSION,
                    "runtime_fixes_version": fixes.RUNTIME_FIXES_VERSION,
                    "git_head": base.git_head(),
                    "artifact_fingerprints": fingerprints,
                },
            )
            base.LOG.info("[HOLDOUT LOCK] training complete; stopping before holdout optimization/evaluation")
            raise SystemExit(0)
        return original(*args_, **kwargs_)

    base.build_bundles = training_only_build_bundles
    base.main()


if __name__ == "__main__":
    main()
