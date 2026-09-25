"""Automatic comparability checks for registered training scenarios."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import experiment_protocol as protocol

AUDIT_VERSION = "experiment_comparability_2026_09_25_v1"


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _settings_signature(root: Path) -> str:
    settings = dict(_json(root / "FROZEN_SETTINGS.json"))
    # Expected scenario/output differences; everything else must agree.
    for key in ("alpha", "h", "artifact_root", "verbose", "data"):
        settings.pop(key, None)
    return _canonical_hash(settings)


def _last_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _max_csv_float(path: Path, column: str) -> float | None:
    if not path.exists():
        return None
    vals = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                pass
    return max(vals) if vals else None


def validate_training_comparability(training_roots: Iterable[Path]) -> dict[str, Any]:
    roots = [Path(x).resolve() for x in training_roots]
    rows: dict[str, Any] = {}
    invariant_sets: dict[str, set[str]] = {
        "data_freeze": set(),
        "feature_manifest": set(),
        "regularization": set(),
        "settings_except_scenario": set(),
        "shared_plan_provenance": set(),
    }
    for root in roots:
        stamp = _json(root / "EXPERIMENT_REGISTRY.json")
        name = str(stamp["scenario"]["name"])
        hashes = {
            "data_freeze": protocol.sha256_file(root / "DATA_FREEZE.json"),
            "feature_manifest": protocol.sha256_file(root / "FEATURE_MANIFEST.json"),
            "regularization": protocol.sha256_file(root / "REGULARIZATION.json"),
            "settings_except_scenario": _settings_signature(root),
            "shared_plan_provenance": protocol.sha256_file(root / "SHARED_PLAN_PROVENANCE.json"),
        }
        for key, value in hashes.items():
            invariant_sets[key].add(value)
        vf = _json(root / "VF_STATUS.json") if (root / "VF_STATUS.json").exists() else {}
        last = _last_csv_row(root / "VF_TRAJECTORY.csv")
        rows[name] = {
            "training_root": str(root),
            "alpha": stamp["scenario"]["alpha"],
            "h": stamp["scenario"]["h"],
            "vf_attempted_outer_iterations": vf.get("attempted_outer_iterations"),
            "vf_termination_reason": vf.get("termination_reason"),
            "vf_last_trajectory_row": last,
            "max_training_oracle_gap_native_psi": _max_csv_float(root / "ORACLE_TRAIN.csv", "gap_native_psi"),
            "invariant_hashes": hashes,
        }
    mismatches = {key: sorted(vals) for key, vals in invariant_sets.items() if len(vals) != 1}
    if mismatches:
        raise RuntimeError(f"Registered training bundles are not comparable on frozen invariants: {mismatches}")
    return {
        "audit_version": AUDIT_VERSION,
        "registry_sha256": protocol.registry_hash(),
        "status": "COMPARABLE_FROZEN_INPUTS",
        "important_interpretation": (
            "Different VF iteration counts or termination reasons do not invalidate equal-budget comparisons. "
            "They are reported as budget-limited algorithm outcomes and are not convergence certificates."
        ),
        "common_invariant_hashes": {k: next(iter(v)) for k, v in invariant_sets.items()},
        "scenarios": rows,
    }


def write_report(experiment_root: Path, training_roots: Iterable[Path]) -> Path:
    report = validate_training_comparability(training_roots)
    p = Path(experiment_root).resolve() / "COMPARABILITY_REPORT.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def verify_report(experiment_root: Path) -> dict[str, Any]:
    p = Path(experiment_root).resolve() / "COMPARABILITY_REPORT.json"
    if not p.exists():
        raise RuntimeError("Holdout evaluation requires the pre-holdout comparability report")
    report = _json(p)
    if report.get("status") != "COMPARABLE_FROZEN_INPUTS" or report.get("registry_sha256") != protocol.registry_hash():
        raise RuntimeError("Comparability report does not match the committed experiment registry")
    return report
