#!/usr/bin/env python3
"""Run the two prespecified structural holdout sensitivities.

Run this only after the primary Stage-2 evaluation has completed, from a clean
Git worktree at the exact Stage-1 commit.  The sensitivity specification and
source code are frozen by that commit before the holdout is evaluated, so this
runner cannot be changed in response to the primary result without invalidating
the commit check.

It evaluates the already-frozen policies under:
  1. regular-template fixed capacity with 30-minute turnover;
  2. primary median-count capacity with zero turnover.

No model is retrained and no primary result is overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as fixes
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dirty() -> bool:
    try:
        return (
            subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
            or subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
        )
    except Exception:
        return True


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/UHNOperating_RoomScheduling2011-2013.xlsx")
    p.add_argument("--training-artifact-root", required=True)
    p.add_argument("--primary-evaluation-root", required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    final.install_final_adapter()
    fixes.apply_runtime_fixes()
    science.apply_scientific_fixes()
    args = _args()

    train_root = Path(args.training_artifact_root).resolve()
    primary_root = Path(args.primary_evaluation_root).resolve()
    root = Path(args.artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Sensitivity artifact root is not empty: {root}")

    freeze = _read_json(train_root / "TRAINING_FREEZE.json")
    if freeze.get("status") != "TRAINING_COMPLETE_HOLDOUT_LOCKED":
        raise RuntimeError("Training bundle is not frozen/accepted")
    if freeze.get("scientific_spec_version") != science.SCIENTIFIC_SPEC_VERSION:
        raise RuntimeError("Scientific specification differs from frozen Stage 1")
    if freeze.get("runtime_fixes_version") != fixes.RUNTIME_FIXES_VERSION:
        raise RuntimeError("Runtime-fix version differs from frozen Stage 1")
    if freeze.get("git_head") != base.git_head():
        raise RuntimeError(
            "Git HEAD differs from Stage 1. Run this from a worktree at the frozen commit."
        )
    if _dirty():
        raise RuntimeError("Tracked source is modified; sensitivity run requires a clean worktree")
    for name, expected in dict(freeze.get("artifact_fingerprints", {})).items():
        path = train_root / name
        if not path.exists() or base.sha256_file(path) != expected:
            raise RuntimeError(f"Frozen Stage-1 artifact changed: {name}")

    consumed = train_root / "HOLDOUT_EVALUATED.json"
    if not consumed.exists():
        raise RuntimeError("Run the primary Stage-2 evaluation before structural sensitivities")
    primary_decision = primary_root / "FINAL_DECISION.json"
    if not primary_decision.exists():
        raise RuntimeError("Primary evaluation root has no FINAL_DECISION.json")
    consumed_payload = _read_json(consumed)
    if consumed_payload.get("evaluation_final_decision_sha256") != base.sha256_file(primary_decision):
        raise RuntimeError("Primary evaluation does not match the Stage-1 consumption marker")

    started = train_root / "REQUIRED_SENSITIVITIES_STARTED.json"
    done = train_root / "REQUIRED_SENSITIVITIES_COMPLETE.json"
    if started.exists() or done.exists():
        raise RuntimeError("Required sensitivities have already been started for this bundle")

    settings = _read_json(train_root / "FROZEN_SETTINGS.json")
    settings["data"] = args.data
    settings["artifact_root"] = str(root)
    if args.cores is not None:
        settings["cores"] = int(args.cores)
    if args.verbose:
        settings["verbose"] = True
    s = science.ScientificFinalSettings(**settings)
    s.validate()

    data_path = Path(s.data).resolve()
    data_freeze = _read_json(train_root / "DATA_FREEZE.json")
    if base.sha256_file(data_path) != data_freeze["input_sha256"]:
        raise RuntimeError("Workbook differs from frozen Stage 1")

    root.mkdir(parents=True, exist_ok=True)
    base.setup_logging(root, s.verbose)
    base.write_json(
        started,
        {
            "status": "REQUIRED_SENSITIVITIES_STARTED",
            "git_head": base.git_head(),
            "primary_final_decision_sha256": base.sha256_file(primary_decision),
            "specification": {
                "capacity": science.REQUIRED_CAPACITY_SENSITIVITY,
                "turnover_minutes": science.REQUIRED_TURNOVER_SENSITIVITY_MINUTES,
                "methods": list(sensitivity.SENSITIVITY_METHODS),
            },
            "started_unix_time": time.time(),
        },
    )

    start = time.monotonic()
    try:
        cfg = base.build_config(s)
        df = base.load_data(cfg)
        scoped, _ = base.apply_experiment_scope(df, cfg)
        train_starts, hold_starts, _ = base.choose_split(scoped, s)
        ws = base.week_start_series(scoped)
        hold_start = min(hold_starts)
        hist = scoped[ws < hold_start].copy()
        context = base.build_candidate_pools(hist, cfg)
        elig = base.build_eligibility_maps(hist, cfg)
        hold_weeks = base.build_bundles(
            scoped,
            hold_starts,
            cfg=cfg,
            candidate_pools=context,
            eligibility_maps=elig,
            offset=s.train_weeks,
        )
        enc = science.ScientificFeatureEncoder.from_manifest(
            _read_json(train_root / "FEATURE_MANIFEST.json")
        )
        hold_a = base.build_arrays(hold_weeks, enc)
        if hold_a.p != science.EXPECTED_FEATURES:
            raise RuntimeError("Frozen feature dimension changed")

        policy_file = np.load(train_root / "POLICIES.npz")
        policies = {"BOOKED": None}
        for name in ("NAIVE", "RA", "RA_FULL", "SITE_SHIFT", "OS", "VF"):
            policies[name] = np.asarray(policy_file[name], float)

        _, summary = sensitivity.run_required_sensitivities(
            scoped=scoped,
            hold_starts=hold_starts,
            cfg=cfg,
            context=context,
            primary_weeks=hold_weeks,
            primary_arrays=hold_a,
            encoder=enc,
            policies=policies,
            s=s,
            root=root,
        )
        payload = {
            "status": "REQUIRED_SENSITIVITIES_COMPLETE",
            "git_head": base.git_head(),
            "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
            "scenarios": sorted(summary["scenario"].unique().tolist()),
            "methods": list(sensitivity.SENSITIVITY_METHODS),
            "elapsed_seconds": time.monotonic() - start,
            "summary_sha256": base.sha256_file(root / "REQUIRED_SENSITIVITY_SUMMARY.csv"),
        }
        base.write_json(root / "RUN_STATUS.json", payload)
        base.write_json(done, payload)
        base.LOG.info("[DONE] required structural sensitivities complete")
    except Exception as exc:
        base.write_json(
            root / "RUN_STATUS.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        base.LOG.exception("Required sensitivity run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
