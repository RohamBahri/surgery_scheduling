"""Automatic comparability checks for registered training scenarios."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import experiment_protocol as protocol
import final_paper_resilience as resilience

# The supported wrapper imports this module before any expensive stage starts.
# Install the non-destructive late-run guards once, before Stage 1/2 initialize
# their runtime hooks.
resilience.install()

AUDIT_VERSION = "experiment_comparability_2026_10_01_v4"


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _settings_signature(root: Path) -> str:
    settings = dict(_json(root / "FROZEN_SETTINGS.json"))
    for key in ("alpha", "h", "artifact_root", "verbose", "data"):
        settings.pop(key, None)
    return _canonical_hash(settings)


def _json_signature_ignoring(root: Path, name: str, ignored: tuple[str, ...]) -> str:
    payload = dict(_json(root / name))
    for key in ignored:
        payload.pop(key, None)
    return _canonical_hash(payload)


def _data_freeze_signature(root: Path) -> str:
    # git_head is provenance only. The cohort/split/data hashes remain frozen.
    return _json_signature_ignoring(root, "DATA_FREEZE.json", ("git_head",))


def _shared_plan_provenance_signature(root: Path) -> str:
    # The same immutable shared-plan manifest may be consumed by compatible
    # later commits; only the consuming git_head changes.
    return _json_signature_ignoring(root, "SHARED_PLAN_PROVENANCE.json", ("git_head",))


def _regularization_signature(root: Path) -> str:
    """Compare the frozen calibration rule, not scenario-dependent fitted values.

    COMMON_COST and NAIVE lambdas are intentionally calibrated from losses that
    change with the registered behavioral scenario. Requiring byte-identical
    REGULARIZATION.json files would therefore reject scientifically comparable
    scenarios. We instead verify that every bundle uses the same eta/p rule and
    that RA/RA_FULL/OS/VF all share that bundle's COMMON_COST lambda.
    """
    reg = dict(_json(root / "REGULARIZATION.json"))
    required = ("COMMON_COST", "NAIVE", "RA", "RA_FULL", "OS", "VF")
    missing = [name for name in required if name not in reg]
    if missing:
        raise RuntimeError(f"REGULARIZATION.json is missing required entries {missing}: {root}")

    common = dict(reg["COMMON_COST"])
    naive = dict(reg["NAIVE"])
    eta = float(common["eta"])
    p = float(common["p"])
    common_loss = float(common["zero_policy_full_weight_case_loss"])
    common_lambda = float(common["lambda"])
    naive_loss = float(naive["zero_policy_lad_minutes"])
    naive_lambda = float(naive["lambda"])

    expected_common = eta * common_loss / p
    expected_naive = float(naive["eta"]) * naive_loss / float(naive["p"])
    scale_common = max(1.0, abs(common_lambda), abs(expected_common))
    scale_naive = max(1.0, abs(naive_lambda), abs(expected_naive))
    if abs(common_lambda - expected_common) > 1e-10 * scale_common:
        raise RuntimeError(f"COMMON_COST lambda does not follow eta*loss0/p in {root}")
    if abs(naive_lambda - expected_naive) > 1e-10 * scale_naive:
        raise RuntimeError(f"NAIVE lambda does not follow eta*LAD0/p in {root}")
    if abs(float(naive["eta"]) - eta) > 1e-12 or abs(float(naive["p"]) - p) > 1e-12:
        raise RuntimeError(f"NAIVE and COMMON_COST regularization eta/p differ in {root}")

    for name in ("RA", "RA_FULL", "OS", "VF"):
        row = dict(reg[name])
        if abs(float(row["lambda"]) - common_lambda) > 1e-10 * scale_common:
            raise RuntimeError(f"{name} does not use the common cost-scale lambda in {root}")
        if abs(float(row["eta"]) - eta) > 1e-12 or abs(float(row["p"]) - p) > 1e-12:
            raise RuntimeError(f"{name} regularization eta/p differs from COMMON_COST in {root}")

    return _canonical_hash(
        {
            "rule_version": "scenario_scaled_eta_loss0_over_p_v1",
            "eta": eta,
            "p": p,
            "common_methods": ["RA", "RA_FULL", "OS", "VF"],
            "common_rule": "lambda=eta*zero_policy_full_weight_case_loss/p",
            "naive_rule": "lambda=eta*zero_policy_lad_minutes/p",
        }
    )


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
        "scientific_spec": set(),
        "sensitivity_plan": set(),
        "regularization_rule": set(),
        "settings_except_scenario": set(),
        "shared_plan_provenance": set(),
    }
    for root in roots:
        stamp = _json(root / "EXPERIMENT_REGISTRY.json")
        name = str(stamp["scenario"]["name"])
        hashes = {
            "data_freeze": _data_freeze_signature(root),
            "feature_manifest": protocol.sha256_file(root / "FEATURE_MANIFEST.json"),
            "scientific_spec": protocol.sha256_file(root / "SCIENTIFIC_SPEC.json"),
            "sensitivity_plan": protocol.sha256_file(root / "SENSITIVITY_PLAN.json"),
            "regularization_rule": _regularization_signature(root),
            "settings_except_scenario": _settings_signature(root),
            "shared_plan_provenance": _shared_plan_provenance_signature(root),
        }
        for key, value in hashes.items():
            invariant_sets[key].add(value)
        vf = _json(root / "VF_STATUS.json") if (root / "VF_STATUS.json").exists() else {}
        rows[name] = {
            "training_root": str(root),
            "alpha": stamp["scenario"]["alpha"],
            "h": stamp["scenario"]["h"],
            "vf_attempted_outer_iterations": vf.get("attempted_outer_iterations"),
            "vf_termination_reason": vf.get("termination_reason"),
            "vf_last_trajectory_row": _last_csv_row(root / "VF_TRAJECTORY.csv"),
            "max_training_oracle_gap_native_psi": _max_csv_float(root / "ORACLE_TRAIN.csv", "gap_native_psi"),
            "invariant_hashes": hashes,
        }
    mismatches = {key: sorted(vals) for key, vals in invariant_sets.items() if len(vals) != 1}
    if mismatches:
        raise RuntimeError(f"Registered training bundles are not comparable on frozen invariants: {mismatches}")
    return {
        "audit_version": AUDIT_VERSION,
        "registry_sha256": protocol.registry_hash(),
        "amendment_sha256": protocol.amendment_hash(),
        "active_scenarios": [x["name"] for x in protocol.active_scenarios(purpose="train")],
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
    root = Path(experiment_root).resolve()
    p = root / "COMPARABILITY_REPORT.json"
    if not p.exists():
        raise RuntimeError("Evaluation requires a comparability report")
    report = _json(p)
    if report.get("status") != "COMPARABLE_FROZEN_INPUTS" or report.get("registry_sha256") != protocol.registry_hash():
        raise RuntimeError("Comparability report does not match the committed experiment registry")
    if report.get("amendment_sha256") != protocol.amendment_hash():
        raise RuntimeError("Comparability report does not match the current experiment amendment")
    # Deliberately no holdout-cache installation here. At this point the
    # training scenario is not yet known, and installing a response-dependent
    # projected benchmark under a sentinel scenario can poison later cache
    # provenance. Correctness takes priority over cross-row cache reuse.
    return report
