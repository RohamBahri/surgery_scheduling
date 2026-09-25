"""Protocol utilities for the registered multi-scenario paper experiment.

The registry is committed before holdout access.  Training bundles record its
hash.  After every registered training scenario is accepted, ``seal_experiment``
creates an immutable experiment-level manifest.  Evaluation may only consume a
bundle named in that seal; interrupted evaluations may resume only against the
same seal, bundle, and output directory.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

REGISTRY_FILE = Path(__file__).resolve().with_name("experiment_registry.json")
PROTOCOL_VERSION = "experiment_protocol_2026_09_25_v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path | None = None) -> dict[str, Any]:
    p = REGISTRY_FILE if path is None else Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    names = [str(x["name"]) for x in payload.get("scenarios", [])]
    if len(names) != len(set(names)) or not names:
        raise RuntimeError("Experiment registry scenario names must be nonempty and unique")
    return payload


def registry_hash(path: Path | None = None) -> str:
    return hashlib.sha256(_canonical_bytes(load_registry(path))).hexdigest()


def registered_scenario(name: str, *, purpose: str) -> dict[str, Any]:
    if purpose not in {"train", "evaluate"}:
        raise ValueError("purpose must be train or evaluate")
    for row in load_registry().get("scenarios", []):
        if str(row.get("name")) == str(name):
            if not bool(row.get(purpose, False)):
                raise RuntimeError(f"Scenario {name!r} is not registered for {purpose}")
            return dict(row)
    raise RuntimeError(f"Scenario {name!r} is not in the committed experiment registry")


def validate_registered_parameters(name: str, alpha: float, h: float, *, purpose: str) -> dict[str, Any]:
    row = registered_scenario(name, purpose=purpose)
    if abs(float(row["alpha"]) - float(alpha)) > 1e-12 or abs(float(row["h"]) - float(h)) > 1e-12:
        raise RuntimeError(
            f"Scenario {name!r} parameters differ from the registry: "
            f"registered=(alpha={row['alpha']}, h={row['h']}), requested=(alpha={alpha}, h={h})"
        )
    return row


def tracked_tree_clean() -> bool:
    try:
        a = subprocess.run(["git", "diff", "--quiet"], check=False).returncode
        b = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode
        return a == 0 and b == 0
    except Exception:
        return False


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _bundle_files(root: Path) -> dict[str, str]:
    names = (
        "TRAINING_FREEZE.json",
        "POLICIES.npz",
        "BEHAVIOR_SCENARIO.json",
        "FEATURE_MANIFEST.json",
        "REGULARIZATION.json",
        "VF_STATUS.json",
        "VF_TRAJECTORY.csv",
        "TRAIN_LIBRARY_SURFACES.csv",
        "SHARED_PLAN_PROVENANCE.json",
        "NUMERIC_GUARD.json",
        "RELEASE_GUARD.json",
    )
    out: dict[str, str] = {}
    for name in names:
        p = Path(root) / name
        if p.exists():
            out[name] = sha256_file(p)
    for required in ("TRAINING_FREEZE.json", "POLICIES.npz", "BEHAVIOR_SCENARIO.json", "FEATURE_MANIFEST.json"):
        if required not in out:
            raise RuntimeError(f"Training bundle is missing required artifact {required}")
    return out


def bundle_hash(root: Path) -> str:
    return hashlib.sha256(_canonical_bytes(_bundle_files(Path(root)))).hexdigest()


def stamp_training_registry(root: Path, *, scenario_name: str, alpha: float, h: float) -> dict[str, Any]:
    row = validate_registered_parameters(scenario_name, alpha, h, purpose="train")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry_hash(),
        "registry_version": load_registry().get("registry_version"),
        "scenario": row,
        "git_head": git_head(),
    }
    p = Path(root) / "EXPERIMENT_REGISTRY.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _shared_manifest_hash(shared_root: Path) -> str:
    p = Path(shared_root) / "SHARED_PLANS_MANIFEST.json"
    if not p.exists():
        raise RuntimeError(f"Missing shared-plan manifest: {p}")
    return sha256_file(p)


def seal_experiment(experiment_root: Path, training_roots: Iterable[Path], shared_root: Path) -> dict[str, Any]:
    if not tracked_tree_clean():
        raise RuntimeError("Experiment sealing requires a clean tracked tree")
    root = Path(experiment_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seal_path = root / "EXPERIMENT_SEAL.json"
    registry = load_registry()
    reg_hash = registry_hash()
    expected = [str(x["name"]) for x in registry["scenarios"] if bool(x.get("train", False))]
    accepted: dict[str, Any] = {}
    heads: set[str] = set()
    shared_hash = _shared_manifest_hash(Path(shared_root))
    for tr in training_roots:
        tr = Path(tr).resolve()
        stamp_path = tr / "EXPERIMENT_REGISTRY.json"
        if not stamp_path.exists():
            raise RuntimeError(f"Training bundle lacks registry stamp: {tr}")
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp.get("registry_sha256") != reg_hash:
            raise RuntimeError(f"Training bundle registry hash differs: {tr}")
        scenario = str(stamp["scenario"]["name"])
        if scenario in accepted:
            raise RuntimeError(f"Duplicate accepted training scenario: {scenario}")
        provenance = tr / "SHARED_PLAN_PROVENANCE.json"
        if not provenance.exists():
            raise RuntimeError(f"Training bundle has no shared-plan provenance: {tr}")
        prov = json.loads(provenance.read_text(encoding="utf-8"))
        if prov.get("manifest_sha256") != shared_hash:
            raise RuntimeError(f"Training bundle {scenario} did not use the sealed shared-plan artifact")
        files = _bundle_files(tr)
        freeze = json.loads((tr / "TRAINING_FREEZE.json").read_text(encoding="utf-8"))
        head = str(freeze.get("git_head"))
        heads.add(head)
        accepted[scenario] = {
            "training_root": str(tr),
            "bundle_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
            "files": files,
            "git_head": head,
        }
    if sorted(accepted) != sorted(expected):
        raise RuntimeError(f"Seal requires exactly the registered training scenarios: expected={expected}, got={sorted(accepted)}")
    if len(heads) != 1 or next(iter(heads)) != git_head():
        raise RuntimeError(f"All accepted bundles must use the current single clean commit; bundle heads={sorted(heads)}, current={git_head()}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SEALED_BEFORE_HOLDOUT",
        "sealed_unix_time": time.time(),
        "git_head": git_head(),
        "registry_sha256": reg_hash,
        "registry": registry,
        "shared_plans_root": str(Path(shared_root).resolve()),
        "shared_plans_manifest_sha256": shared_hash,
        "accepted_training_bundles": accepted,
    }
    if seal_path.exists():
        old = json.loads(seal_path.read_text(encoding="utf-8"))
        old_cmp = dict(old); old_cmp.pop("sealed_unix_time", None)
        new_cmp = dict(payload); new_cmp.pop("sealed_unix_time", None)
        if old_cmp != new_cmp:
            raise RuntimeError("Experiment is already sealed with different inputs; additions/substitutions are forbidden")
        return old
    seal_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "EXPERIMENT_CONSUMPTION.json").write_text(
        json.dumps({"status": "SEALED_NOT_STARTED", "registry_sha256": reg_hash, "bundles": {}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_sealed_bundle(experiment_root: Path, training_root: Path) -> tuple[dict[str, Any], str]:
    root = Path(experiment_root).resolve()
    seal_path = root / "EXPERIMENT_SEAL.json"
    if not seal_path.exists():
        raise RuntimeError("Holdout evaluation requires an experiment seal created before holdout access")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("registry_sha256") != registry_hash():
        raise RuntimeError("Committed experiment registry differs from the sealed registry")
    tr = Path(training_root).resolve()
    current_files = _bundle_files(tr)
    current_hash = hashlib.sha256(_canonical_bytes(current_files)).hexdigest()
    matches = [
        name for name, meta in seal.get("accepted_training_bundles", {}).items()
        if str(Path(meta["training_root"]).resolve()) == str(tr)
    ]
    if len(matches) != 1:
        raise RuntimeError("Training bundle is not one of the pre-holdout sealed bundles")
    name = matches[0]
    meta = seal["accepted_training_bundles"][name]
    if current_hash != meta.get("bundle_sha256") or current_files != meta.get("files"):
        raise RuntimeError(f"Sealed training bundle {name} has changed")
    if seal.get("git_head") != git_head() or not tracked_tree_clean():
        raise RuntimeError("Evaluation must use the exact clean commit recorded by the experiment seal")
    return seal, name


def update_consumption(experiment_root: Path, scenario: str, *, state: str, evaluation_root: Path) -> None:
    root = Path(experiment_root).resolve()
    p = root / "EXPERIMENT_CONSUMPTION.json"
    payload = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"bundles": {}}
    bundles = dict(payload.get("bundles", {}))
    old = bundles.get(scenario)
    eval_path = str(Path(evaluation_root).resolve())
    if old and old.get("evaluation_root") != eval_path:
        raise RuntimeError(f"Scenario {scenario} consumption is already tied to another evaluation directory")
    bundles[scenario] = {"state": state, "evaluation_root": eval_path, "updated_unix_time": time.time()}
    payload["bundles"] = bundles
    payload["status"] = "HOLDOUT_CONSUMPTION_IN_PROGRESS" if state != "COMPLETE" else payload.get("status", "HOLDOUT_CONSUMPTION_IN_PROGRESS")
    expected = set(json.loads((root / "EXPERIMENT_SEAL.json").read_text(encoding="utf-8"))["accepted_training_bundles"])
    if expected and all(bundles.get(x, {}).get("state") == "COMPLETE" for x in expected):
        payload["status"] = "HOLDOUT_CONSUMPTION_COMPLETE"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
