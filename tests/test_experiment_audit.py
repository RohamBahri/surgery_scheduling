from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiment_audit as audit


def _write_reg(root: Path, *, common_loss: float, naive_loss: float, eta: float = 0.01, p: float = 10.0) -> None:
    common_lambda = eta * common_loss / p
    naive_lambda = eta * naive_loss / p
    payload = {
        "COMMON_COST": {
            "zero_policy_full_weight_case_loss": common_loss,
            "lambda": common_lambda,
            "eta": eta,
            "p": p,
        },
        "NAIVE": {
            "zero_policy_lad_minutes": naive_loss,
            "lambda": naive_lambda,
            "eta": eta,
            "p": p,
        },
        "RA": {"lambda": common_lambda, "eta": eta, "p": p},
        "RA_FULL": {"lambda": common_lambda, "eta": eta, "p": p},
        "OS": {"lambda": common_lambda, "eta": eta, "p": p},
        "VF": {"lambda": common_lambda, "eta": eta, "p": p},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "REGULARIZATION.json").write_text(json.dumps(payload), encoding="utf-8")


def test_regularization_signature_accepts_scenario_scaled_values(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_reg(a, common_loss=100.0, naive_loss=50.0)
    _write_reg(b, common_loss=250.0, naive_loss=90.0)
    assert audit._regularization_signature(a) == audit._regularization_signature(b)


def test_regularization_signature_rejects_method_not_using_common_lambda(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    _write_reg(root, common_loss=100.0, naive_loss=50.0)
    payload = json.loads((root / "REGULARIZATION.json").read_text(encoding="utf-8"))
    payload["VF"]["lambda"] += 1.0
    (root / "REGULARIZATION.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="VF does not use the common cost-scale lambda"):
        audit._regularization_signature(root)


def test_regularization_signature_rejects_wrong_formula(tmp_path: Path) -> None:
    root = tmp_path / "bad_formula"
    _write_reg(root, common_loss=100.0, naive_loss=50.0)
    payload = json.loads((root / "REGULARIZATION.json").read_text(encoding="utf-8"))
    payload["COMMON_COST"]["lambda"] += 1.0
    for name in ("RA", "RA_FULL", "OS", "VF"):
        payload[name]["lambda"] = payload["COMMON_COST"]["lambda"]
    (root / "REGULARIZATION.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="COMMON_COST lambda does not follow"):
        audit._regularization_signature(root)


def test_data_and_shared_provenance_signatures_ignore_git_head_only(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()

    data = {"git_head": "old", "cohort_sha256": "abc", "train_cases": 20951}
    prov = {
        "git_head": "old",
        "manifest_sha256": "manifest",
        "registry_sha256": "registry",
        "shared_plans_root": "/same/shared/root",
    }
    (a / "DATA_FREEZE.json").write_text(json.dumps(data), encoding="utf-8")
    (a / "SHARED_PLAN_PROVENANCE.json").write_text(json.dumps(prov), encoding="utf-8")

    data["git_head"] = "new"
    prov["git_head"] = "new"
    (b / "DATA_FREEZE.json").write_text(json.dumps(data), encoding="utf-8")
    (b / "SHARED_PLAN_PROVENANCE.json").write_text(json.dumps(prov), encoding="utf-8")

    assert audit._data_freeze_signature(a) == audit._data_freeze_signature(b)
    assert audit._shared_plan_provenance_signature(a) == audit._shared_plan_provenance_signature(b)

    data["train_cases"] = 20950
    (b / "DATA_FREEZE.json").write_text(json.dumps(data), encoding="utf-8")
    assert audit._data_freeze_signature(a) != audit._data_freeze_signature(b)
