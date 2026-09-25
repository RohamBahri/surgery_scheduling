"""Protocol utilities for the registered multi-scenario paper experiment.

The registry is committed before holdout access. Training bundles record its
hash. After every registered training scenario is accepted, ``seal_experiment``
creates an immutable experiment-level manifest. Evaluation may only consume a
bundle named in that seal; interrupted evaluations may restart only against the
same seal, bundle, and output directory.

This module also installs the reviewed numerical-accounting policy and the
shared-plan provenance checks used by the multi-scenario pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REGISTRY_FILE = Path(__file__).resolve().with_name("experiment_registry.json")
PROTOCOL_VERSION = "experiment_protocol_2026_09_25_v2"
NUMERIC_POLICY_VERSION = "phi_accounting_2026_09_25_v5"
PHI_ACCOUNTING_ATOL = 0.1
PHI_ACCOUNTING_RTOL = 2e-6
PHI_ACCOUNTING_MAX_TOL = 0.5
PHI_ACCOUNTING_WARN_ATOL = 1e-5
_ACTIVE_REFINEMENT_PARENT: Path | None = None
_SHARED_PATCHED = False
_SHARED_ORIGINAL_SAVE = None
_SHARED_ORIGINAL_LOAD = None


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


def registered_scenarios(*, purpose: str) -> list[dict[str, Any]]:
    return [dict(x) for x in load_registry().get("scenarios", []) if bool(x.get(purpose, False))]


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


def phi_accounting_tolerance(a: float, b: float) -> float:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return min(PHI_ACCOUNTING_MAX_TOL, max(PHI_ACCOUNTING_ATOL, PHI_ACCOUNTING_RTOL * scale))


def assert_phi_accounting_close(recomputed: float, solver_sum: float, *, context: str) -> None:
    """Accept solver-feasibility roundoff while still rejecting material accounting errors.

    The cap at 0.5 cost units is far below the model's 10-per-minute idle cost
    quantum. The independently recomputed feasible schedule remains the reported
    incumbent, so accepting this audit difference cannot improve an upper bound.
    """
    import final_paper_numeric_guard as numeric
    import run_final_vf_experiment as base

    a, b = float(recomputed), float(solver_sum)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise AssertionError(f"{context}: non-finite Phi accounting values {a!r}, {b!r}")
    err = abs(a - b)
    tol = phi_accounting_tolerance(a, b)
    if err > tol:
        rel = err / max(1.0, abs(a), abs(b))
        raise AssertionError(
            f"{context}: decomposed Phi mismatch recomputed={a:.12g} solver_sum={b:.12g} "
            f"abs_error={err:.6g} relative_error={rel:.3g} tolerance={tol:.6g}"
        )
    if err > PHI_ACCOUNTING_WARN_ATOL:
        rel = err / max(1.0, abs(a), abs(b))
        base.LOG.warning(
            "[NUMERIC-AUDIT] %s | recomputed Phi and summed site solver Phi differ within "
            "the reviewed feasibility-roundoff allowance: abs=%.6g rel=%.3g tol=%.6g",
            context, err, rel, tol,
        )


def _set_numeric_policy() -> None:
    import final_paper_numeric_guard as numeric
    numeric.NUMERIC_GUARD_VERSION = NUMERIC_POLICY_VERSION
    numeric.PHI_ACCOUNTING_ATOL = PHI_ACCOUNTING_ATOL
    numeric.PHI_ACCOUNTING_RTOL = PHI_ACCOUNTING_RTOL
    numeric.PHI_ACCOUNTING_WARN_ATOL = PHI_ACCOUNTING_WARN_ATOL
    numeric.phi_accounting_tolerance = phi_accounting_tolerance
    numeric.assert_phi_accounting_close = assert_phi_accounting_close


def training_process_task(*args, **kwargs):
    _set_numeric_policy()
    import final_paper_shared_plans as shared
    return shared.training_process_task_logged(*args, **kwargs)


def deterministic_eval_worker(*args, **kwargs):
    _set_numeric_policy()
    import final_paper_shared_plans as shared
    return shared.deterministic_eval_worker_logged(*args, **kwargs)


def sensitivity_worker(*args, **kwargs):
    _set_numeric_policy()
    import final_paper_shared_plans as shared
    return shared.sensitivity_worker_logged(*args, **kwargs)


def install_numeric_policy() -> None:
    """Install the numerical policy in parent and macOS-spawned worker paths."""
    _set_numeric_policy()
    import final_paper_numeric_guard as numeric
    numeric.training_process_task = training_process_task
    numeric.deterministic_eval_worker = deterministic_eval_worker
    numeric.sensitivity_worker = sensitivity_worker


def implementable_oracle_planning(a, s):
    """Projected hindsight benchmark over the actually display-reachable interval."""
    import run_final_paper_experiment as final
    cap = min(float(s.h), float(s.display_cap))
    lower = -float(s.alpha) * np.minimum(
        cap, np.maximum(a.booked - final.MIN_RECOMMENDED_DURATION, 0.0)
    )
    upper = np.full(len(a.booked), float(s.alpha) * cap, dtype=float)
    corr = np.clip(a.error, lower, upper)
    planning = a.booked + corr
    if np.any(planning <= 0) or np.any(~np.isfinite(planning)):
        raise AssertionError("Projected hindsight benchmark produced invalid planning durations")
    return planning, corr


def install_behavioral_protocol() -> None:
    import final_paper_scientific_fixes as science
    science.implementable_oracle_planning = implementable_oracle_planning


def _duration_fingerprint(weeks: Sequence[Any], duration_by_week: Mapping[int, np.ndarray]) -> str:
    payload = []
    for week in sorted(weeks, key=lambda w: int(w.position)):
        d = np.asarray(duration_by_week[int(week.position)], dtype=np.float64)
        if len(d) != week.instance.num_cases:
            raise RuntimeError(f"Duration vector length changed for week {week.position}")
        payload.append({
            "week": int(week.position),
            "case_ids": [str(c.case_id) for c in week.instance.cases],
            "durations_hex": [float(x).hex() for x in d],
        })
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _model_source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    relevant = (
        "src/solvers/fixed_capacity.py",
        "src/core/column.py",
        "src/core/config.py",
        "run_final_paper_experiment.py",
        "final_paper_numeric_guard.py",
        "final_paper_scientific_fixes.py",
        "experiment_protocol.py",
        "experiment_registry.json",
    )
    out = {}
    for rel in relevant:
        p = root / rel
        if not p.exists():
            raise RuntimeError(f"Model source file missing: {rel}")
        out[rel] = sha256_file(p)
    return out


def _gurobi_version() -> str:
    try:
        import gurobipy as gp
        return ".".join(map(str, gp.gurobi.version()))
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def configure_refinement_parent(root: Path | None) -> None:
    global _ACTIVE_REFINEMENT_PARENT
    _ACTIVE_REFINEMENT_PARENT = None if root is None else Path(root).resolve()


def install_shared_plan_validation(shared_module=None) -> None:
    """Strengthen reusable-bound provenance without changing planner mathematics."""
    global _SHARED_PATCHED, _SHARED_ORIGINAL_SAVE, _SHARED_ORIGINAL_LOAD
    if _SHARED_PATCHED:
        return
    if shared_module is None:
        import final_paper_shared_plans as shared_module
    _SHARED_ORIGINAL_SAVE = shared_module.save_plan_set
    _SHARED_ORIGINAL_LOAD = shared_module.load_plan_set

    def save_plan_set(root, kind, plans, weeks, duration_by_week, s):
        _SHARED_ORIGINAL_SAVE(root, kind, plans, weeks, duration_by_week, s)
        manifest_path = Path(root) / "SHARED_PLANS_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = dict(manifest.get("plan_sets", {})).get(str(kind).lower(), {})
        meta["processed_duration_sha256"] = _duration_fingerprint(weeks, duration_by_week)
        meta["ordered_case_ids_in_duration_hash"] = True
        meta["solver"] = {
            "gurobi_version": _gurobi_version(),
            "seed": int(s.random_seed),
            "threads_per_site": 1,
            "time_limit_seconds": int(s.oracle_seconds if str(kind).lower() == "oracle" else s.train_planner_seconds),
            "target_native_psi_gap": float(s.oracle_gap if str(kind).lower() == "oracle" else s.train_planner_gap),
        }
        manifest.setdefault("plan_sets", {})[str(kind).lower()] = meta
        manifest["model_source_sha256"] = _model_source_identity()
        manifest["registry_sha256"] = registry_hash()
        manifest["source_git_head"] = git_head()
        manifest["refinement_parent_manifest_sha256"] = (
            None if _ACTIVE_REFINEMENT_PARENT is None
            else sha256_file(_ACTIVE_REFINEMENT_PARENT / "SHARED_PLANS_MANIFEST.json")
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def load_plan_set(root, kind, weeks, duration_by_week, s):
        manifest_path = Path(root) / "SHARED_PLANS_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_git_head") != git_head():
            raise RuntimeError("Shared bounds were produced by a different commit; use assignments only as warm starts, not cached bounds")
        if manifest.get("registry_sha256") != registry_hash():
            raise RuntimeError("Shared-plan registry hash differs from the committed experiment design")
        if manifest.get("model_source_sha256") != _model_source_identity():
            raise RuntimeError("Shared-plan mathematical model/cost-accounting source identity changed")
        meta = dict(manifest.get("plan_sets", {})).get(str(kind).lower())
        if not meta:
            raise RuntimeError(f"Shared plan artifact has no {kind!r} plan set")
        current_duration_hash = _duration_fingerprint(weeks, duration_by_week)
        if meta.get("processed_duration_sha256") != current_duration_hash:
            raise RuntimeError(
                f"Shared {kind} processed-duration fingerprint changed; cached lower bounds cannot be reused"
            )
        return _SHARED_ORIGINAL_LOAD(root, kind, weeks, duration_by_week, s)

    shared_module.save_plan_set = save_plan_set
    shared_module.load_plan_set = load_plan_set
    _SHARED_PATCHED = True


def _bundle_files(root: Path) -> dict[str, str]:
    names = (
        "TRAINING_FREEZE.json", "POLICIES.npz", "BEHAVIOR_SCENARIO.json",
        "FEATURE_MANIFEST.json", "REGULARIZATION.json", "VF_STATUS.json",
        "VF_TRAJECTORY.csv", "TRAIN_LIBRARY_SURFACES.csv",
        "SHARED_PLAN_PROVENANCE.json", "NUMERIC_GUARD.json", "RELEASE_GUARD.json",
        "EXPERIMENT_REGISTRY.json",
    )
    out: dict[str, str] = {}
    for name in names:
        p = Path(root) / name
        if p.exists():
            out[name] = sha256_file(p)
    for required in ("TRAINING_FREEZE.json", "POLICIES.npz", "BEHAVIOR_SCENARIO.json", "FEATURE_MANIFEST.json", "EXPERIMENT_REGISTRY.json"):
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
    shared_manifest = json.loads((Path(shared_root) / "SHARED_PLANS_MANIFEST.json").read_text(encoding="utf-8"))
    if shared_manifest.get("source_git_head") != git_head() or shared_manifest.get("registry_sha256") != reg_hash:
        raise RuntimeError("Shared plans were not generated under the current clean registered experiment commit")
    for tr in training_roots:
        tr = Path(tr).resolve()
        stamp = json.loads((tr / "EXPERIMENT_REGISTRY.json").read_text(encoding="utf-8"))
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
        head = str(freeze.get("git_head")); heads.add(head)
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
    seal = json.loads((root / "EXPERIMENT_SEAL.json").read_text(encoding="utf-8"))
    if seal.get("registry_sha256") != registry_hash():
        raise RuntimeError("Committed experiment registry differs from the sealed registry")
    tr = Path(training_root).resolve()
    current_files = _bundle_files(tr)
    current_hash = hashlib.sha256(_canonical_bytes(current_files)).hexdigest()
    matches = [name for name, meta in seal.get("accepted_training_bundles", {}).items() if str(Path(meta["training_root"]).resolve()) == str(tr)]
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
    payload["status"] = "HOLDOUT_CONSUMPTION_IN_PROGRESS"
    expected = set(json.loads((root / "EXPERIMENT_SEAL.json").read_text(encoding="utf-8"))["accepted_training_bundles"])
    if expected and all(bundles.get(x, {}).get("state") == "COMPLETE" for x in expected):
        payload["status"] = "HOLDOUT_CONSUMPTION_COMPLETE"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_exact_restart(experiment_root: Path, training_root: Path, evaluation_root: Path) -> str:
    """Permit a failed evaluation to restart only with the exact sealed inputs/path."""
    _, scenario = verify_sealed_bundle(experiment_root, training_root)
    tr = Path(training_root).resolve(); er = Path(evaluation_root).resolve()
    started = tr / "HOLDOUT_EVALUATION_STARTED.json"
    done = tr / "HOLDOUT_EVALUATED.json"
    if done.exists() or not started.exists():
        raise RuntimeError("Exact restart is only allowed after a started-but-incomplete evaluation")
    marker = json.loads(started.read_text(encoding="utf-8"))
    if str(Path(marker.get("evaluation_artifact_root", "")).resolve()) != str(er):
        raise RuntimeError("Restart output directory differs from the original frozen evaluation directory")
    status = er / "RUN_STATUS.json"
    if not status.exists() or not str(json.loads(status.read_text(encoding="utf-8")).get("status", "")).startswith("FAILED"):
        raise RuntimeError("Restart requires a recorded failed evaluation")
    if er.exists():
        shutil.rmtree(er)
    started.unlink()
    update_consumption(experiment_root, scenario, state="RESTARTING_EXACT_INPUTS", evaluation_root=er)
    return scenario


def registered_response_sensitivity_runner(weeks, a, policies, oracle_plans, s, root: Path) -> None:
    """Evaluate off-diagonal registered response conditions at the primary budget."""
    import copy
    import pandas as pd
    import final_paper_scientific_fixes as science
    import run_final_paper_experiment as final

    training_row = next(
        (x for x in registered_scenarios(purpose="train") if abs(float(x["alpha"]) - float(s.alpha)) <= 1e-12 and abs(float(x["h"]) - float(s.h)) <= 1e-12),
        None,
    )
    if training_row is None:
        raise RuntimeError("Frozen training response parameters are not registered")
    rows = []
    oracle_lb = {w: r.bound for w, r in oracle_plans.items()}
    learned = {k: v for k, v in policies.items() if k != "BOOKED"}
    for response in registered_scenarios(purpose="evaluate"):
        if response["name"] == training_row["name"]:
            continue
        ss = copy.copy(s); ss.alpha = float(response["alpha"]); ss.h = float(response["h"])
        scenario_policies = dict(learned)
        impl_planning, impl_corr = implementable_oracle_planning(a, ss)
        plan_maps: dict[str, Any] = {}
        for name, w in scenario_policies.items():
            dm, _ = __import__("run_final_paper_evaluation")._policy_meta(name, w, a, ss)
            plan_maps[name] = science.deterministic_eval_solve_batch(
                weeks, dm, ss,
                work_limit=ss.final_planner_work_limit,
                wall_seconds=ss.final_planner_seconds,
                gap=ss.final_planner_gap,
                label=f"matrix_{training_row['name']}__{response['name']}__{name}",
            )
        impl_dm = {wk: impl_planning[idx] for wk, idx in a.week_slices.items()}
        plan_maps["IMPLEMENTABLE_ORACLE"] = science.deterministic_eval_solve_batch(
            weeks, impl_dm, ss,
            work_limit=ss.final_planner_work_limit,
            wall_seconds=ss.final_planner_seconds,
            gap=ss.final_planner_gap,
            label=f"matrix_{training_row['name']}__{response['name']}__IMPLEMENTABLE_ORACLE",
        )
        for name, plans in plan_maps.items():
            for wk, idx in a.week_slices.items():
                plan = plans[wk]
                rc = float(plan.column.compute_cost(a.actual[idx], final.final_cost_cfg(ss), final.PRIMARY_TURNOVER))
                rows.append({
                    "training_scenario": training_row["name"],
                    "response_scenario": response["name"],
                    "alpha": response["alpha"], "h": response["h"],
                    "method": name, "week": wk, "realized_cost": rc,
                    "planning_gap_native_psi": float(plan.gap), "planning_status": str(plan.status),
                    "regret_upper": max(0.0, rc - float(oracle_lb[wk])),
                    "evaluation_work_limit": float(ss.final_planner_work_limit),
                    "evaluation_mip_gap": float(ss.final_planner_gap),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(root) / "RESPONSE_MATRIX_OFFDIAGONAL_WEEKLY.csv", index=False)
    if len(frame):
        frame.groupby(["training_scenario", "response_scenario", "alpha", "h", "method"], as_index=False).agg(
            avg_realized_cost=("realized_cost", "mean"),
            avg_regret_upper=("regret_upper", "mean"),
            max_planning_gap_native_psi=("planning_gap_native_psi", "max"),
        ).to_csv(Path(root) / "RESPONSE_MATRIX_OFFDIAGONAL_SUMMARY.csv", index=False)
