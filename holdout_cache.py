"""Experiment-level cache for holdout quantities shared by every training scenario.

The realized-duration Oracle, BOOKED schedule, and projected hindsight benchmark
for a given evaluation-response condition do not depend on which policy-training
scenario is being evaluated.  Once holdout consumption has legitimately started
under a sealed experiment, this module stores those solves once and reuses them
only when the exact seal, code, instances, durations, costs, and solver protocol
match.

Cached lower bounds are never accepted on a provenance mismatch.  Assignments
from an incompatible cache are not loaded by this module.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import experiment_protocol as protocol
import final_paper_numeric_guard as numeric
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.types import BlockId
from src.solvers.fixed_capacity import column_from_assignment

CACHE_VERSION = "holdout_cache_2026_09_25_v1"
_ORIGINAL_ORACLE = runtime.safe_solve_oracle_batch
_ORIGINAL_DETERMINISTIC = science.deterministic_eval_solve_batch
_ACTIVE_ROOT: Path | None = None
_ACTIVE_TRAINING_SCENARIO: str | None = None


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _block_payload(bid: BlockId) -> list[Any]:
    return [int(bid.day_index), str(bid.site), str(bid.room)]


def _block_from_payload(x) -> BlockId:
    return BlockId(int(x[0]), str(x[1]), str(x[2]))


def _week_and_duration_hash(weeks, duration_by_week) -> str:
    rows = []
    for w in sorted(weeks, key=lambda z: int(z.position)):
        d = np.asarray(duration_by_week[int(w.position)], dtype=np.float64)
        rows.append({
            "week": int(w.position),
            "start": str(w.start.date()),
            "case_ids": [str(c.case_id) for c in w.instance.cases],
            "durations_hex": [float(x).hex() for x in d],
            "blocks": [
                [int(b.id.day_index), str(b.id.site), str(b.id.room), float(b.capacity_minutes)]
                for b in w.instance.calendar.candidates
            ],
            "eligibility": [
                [[int(b.day_index), str(b.site), str(b.room)] for b in w.instance.case_eligible_blocks.get(i, [])]
                for i in range(w.instance.num_cases)
            ],
        })
    return _canonical_hash(rows)


def _seal_hash(experiment_root: Path) -> str:
    return protocol.sha256_file(Path(experiment_root) / "EXPERIMENT_SEAL.json")


def _cache_dir() -> Path:
    if _ACTIVE_ROOT is None:
        raise RuntimeError("Holdout cache is not configured")
    p = _ACTIVE_ROOT / "holdout_shared"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json.gz"


def _manifest_path(key: str) -> Path:
    return _cache_dir() / f"{key}.manifest.json"


def _assignment(plan) -> list[list[Any]]:
    out = []
    for i in range(plan.column.n_cases):
        bids = [bid for (j, bid), v in plan.column.z_assign.items() if int(j) == i and float(v) > 0.5]
        if len(bids) != 1:
            raise RuntimeError(f"Cached plan week {plan.week} case {i} is not assigned exactly once")
        out.append(_block_payload(bids[0]))
    return out


def _solver_protocol(kind: str, s, *, work_limit=None, wall_seconds=None, gap=None) -> dict[str, Any]:
    if kind == "realized_oracle":
        return {
            "kind": kind,
            "oracle_seconds": int(s.oracle_seconds),
            "oracle_gap": float(s.oracle_gap),
            "oracle_retry_seconds": int(s.oracle_retry_seconds),
            "oracle_numeric_tol": float(s.oracle_numeric_tol),
            "threads_per_site": 1,
            "seed": int(s.random_seed),
        }
    return {
        "kind": kind,
        "work_limit": float(work_limit),
        "wall_seconds": int(wall_seconds),
        "mip_gap": float(gap),
        "threads_per_site": 1,
        "seed": int(s.random_seed),
    }


def _base_manifest(key, weeks, duration_by_week, s, solver_protocol) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "key": key,
        "seal_sha256": _seal_hash(_ACTIVE_ROOT),
        "registry_sha256": protocol.registry_hash(),
        "git_head": protocol.git_head(),
        "model_source_sha256": protocol._model_source_identity(),
        "week_duration_structure_sha256": _week_and_duration_hash(weeks, duration_by_week),
        "costs": {
            "overtime": float(s.overtime),
            "idle": float(s.idle),
            "turnover": float(final.PRIMARY_TURNOVER),
        },
        "solver_protocol": solver_protocol,
    }


def _save(key, plans, weeks, duration_by_week, s, solver_protocol) -> None:
    rows = []
    lookup = {int(w.position): w for w in weeks}
    for wk in sorted(lookup):
        p = plans[wk]
        d = np.asarray(duration_by_week[wk], float)
        feasible = float(p.column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER))
        numeric.assert_phi_accounting_close(feasible, float(p.objective), context=f"holdout-cache {key} week {wk}")
        if float(p.bound) > feasible + numeric.phi_accounting_tolerance(feasible, p.bound):
            raise RuntimeError(f"Holdout cache {key} week {wk}: lower bound exceeds feasible cost")
        rows.append({
            "week": wk,
            "n_cases": p.column.n_cases,
            "assignment": _assignment(p),
            "objective": feasible,
            "bound": float(p.bound),
            "gap": float(p.gap),
            "status": str(p.status),
            "solve_seconds": float(p.solve_seconds),
            "exact": bool(p.exact),
            "tiebreak_used": bool(p.tiebreak_used),
        })
    data_path = _key_path(key)
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        json.dump({"plans": rows}, f, sort_keys=True)
    manifest = _base_manifest(key, weeks, duration_by_week, s, solver_protocol)
    manifest["plan_file"] = data_path.name
    manifest["plan_file_sha256"] = protocol.sha256_file(data_path)
    _manifest_path(key).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(key, weeks, duration_by_week, s, solver_protocol):
    mp = _manifest_path(key); dp = _key_path(key)
    if not mp.exists() or not dp.exists():
        return None
    stored = json.loads(mp.read_text(encoding="utf-8"))
    current = _base_manifest(key, weeks, duration_by_week, s, solver_protocol)
    for field, value in current.items():
        if stored.get(field) != value:
            raise RuntimeError(
                f"Holdout cache {key!r} provenance mismatch at {field}; cached bounds cannot be reused"
            )
    if stored.get("plan_file_sha256") != protocol.sha256_file(dp):
        raise RuntimeError(f"Holdout cache {key!r} plan file changed")
    with gzip.open(dp, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    entries = {int(x["week"]): x for x in payload["plans"]}
    lookup = {int(w.position): w for w in weeks}
    if set(entries) != set(lookup):
        raise RuntimeError(f"Holdout cache {key!r} week set changed")
    out = {}
    for wk, w in lookup.items():
        e = entries[wk]
        if int(e["n_cases"]) != w.instance.num_cases or len(e["assignment"]) != w.instance.num_cases:
            raise RuntimeError(f"Holdout cache {key!r} week {wk} case count changed")
        column = column_from_assignment(
            w.instance, {i: _block_from_payload(v) for i, v in enumerate(e["assignment"])},
            enforce_eligibility=True,
        )
        d = np.asarray(duration_by_week[wk], float)
        feasible = float(column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER))
        numeric.assert_phi_accounting_close(feasible, float(e["objective"]), context=f"holdout-cache load {key} week {wk}")
        bound = float(e["bound"])
        if bound > feasible + numeric.phi_accounting_tolerance(feasible, bound):
            raise RuntimeError(f"Holdout cache {key!r} week {wk}: cached lower bound invalid")
        out[wk] = base.PlanResult(
            week=wk, column=column, objective=feasible, bound=bound, gap=float(e["gap"]),
            status=f"CACHED[{e['status']}]", solve_seconds=float(e["solve_seconds"]),
            exact=bool(e["exact"]), tiebreak_used=bool(e.get("tiebreak_used", False)),
        )
    base.LOG.info("[HOLDOUT-CACHE] reused %s from %s", key, _cache_dir())
    return out


def _oracle_wrapper(weeks, duration_by_week, s, *, label):
    if str(label) != "oracle_holdout":
        return _ORIGINAL_ORACLE(weeks, duration_by_week, s, label=label)
    key = "realized_oracle"
    solver = _solver_protocol(key, s)
    cached = _load(key, weeks, duration_by_week, s, solver)
    if cached is not None:
        return cached
    result = _ORIGINAL_ORACLE(weeks, duration_by_week, s, label=label)
    _save(key, result, weeks, duration_by_week, s, solver)
    return result


def _deterministic_wrapper(weeks, duration_by_week, s, *, work_limit, wall_seconds, gap, label, seed=None):
    text = str(label)
    key = None
    if text == "final_holdout_BOOKED":
        key = "booked"
    elif text == "final_holdout_implementable_oracle" and _ACTIVE_TRAINING_SCENARIO:
        key = f"projected_{_ACTIVE_TRAINING_SCENARIO}"
    elif "__IMPLEMENTABLE_ORACLE" in text and text.startswith("matrix_"):
        # matrix_<training>__<response>__IMPLEMENTABLE_ORACLE
        pieces = text.split("__")
        if len(pieces) >= 3:
            key = f"projected_{pieces[1]}"
    if key is None:
        kwargs = dict(work_limit=work_limit, wall_seconds=wall_seconds, gap=gap, label=label)
        if seed is not None:
            kwargs["seed"] = seed
        return _ORIGINAL_DETERMINISTIC(weeks, duration_by_week, s, **kwargs)
    solver = _solver_protocol(key, s, work_limit=work_limit, wall_seconds=wall_seconds, gap=gap)
    cached = _load(key, weeks, duration_by_week, s, solver)
    if cached is not None:
        return cached
    kwargs = dict(work_limit=work_limit, wall_seconds=wall_seconds, gap=gap, label=label)
    if seed is not None:
        kwargs["seed"] = seed
    result = _ORIGINAL_DETERMINISTIC(weeks, duration_by_week, s, **kwargs)
    _save(key, result, weeks, duration_by_week, s, solver)
    return result


def install(experiment_root: Path, training_scenario: str) -> None:
    global _ACTIVE_ROOT, _ACTIVE_TRAINING_SCENARIO
    _ACTIVE_ROOT = Path(experiment_root).resolve()
    _ACTIVE_TRAINING_SCENARIO = str(training_scenario)
    # Source hooks survive evaluation.main -> apply_runtime_fixes().
    runtime.safe_solve_oracle_batch = _oracle_wrapper
    # This scientific function is called directly and is not replaced by apply_scientific_fixes().
    science.deterministic_eval_solve_batch = _deterministic_wrapper
