"""Shared weekly-plan backbone for multi-scenario final-paper experiments.

The expensive training realized-oracle and BOOKED schedules do not depend on the
behavioral response parameters (alpha, h).  This module makes those solves
portable, validates them against the exact data/weekly instances, and lets every
behavioral-scenario training run reuse them instead of solving the same 144
site-week MILPs again.

It also installs persistent per-site Gurobi logs for every reviewed weekly solve
(including macOS spawned workers) and permits arbitrary valid behavioral
scenarios while keeping the planning/data specification frozen.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric
import final_paper_required_sensitivities as sensitivity
import final_paper_runtime_fixes as runtime
import final_paper_scientific_fixes as science
import run_final_paper_experiment as final
import run_final_vf_experiment as base
from src.core.column import ScheduleColumn
from src.core.config import SolverConfig
from src.core.types import BlockId
from src.solvers.fixed_capacity import (
    column_from_assignment,
    schedule_metrics,
    solve_fixed_capacity_assignment,
)


SHARED_PLANS_VERSION = "final_paper_shared_plans_2026_09_24_v1"
_ACTIVE_LABEL = "weekly"
_ORIGINAL_TRAIN_WORKER = numeric.training_process_task
_ORIGINAL_DETERMINISTIC_WORKER = numeric.deterministic_eval_worker
_ORIGINAL_SENSITIVITY_WORKER = numeric.sensitivity_worker
_ORIGINAL_RUNTIME_ORACLE = runtime.safe_solve_oracle_batch
_ORIGINAL_RUNTIME_BATCH = runtime.safe_solve_batch


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text[:120] or "unnamed"


def _duration_hash(values: np.ndarray) -> str:
    a = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()[:12]


def weekly_log_file(
    s,
    *,
    week: int,
    site: str,
    label: str | None = None,
    durations: np.ndarray | None = None,
    suffix: str | None = None,
) -> str:
    """Return a unique persistent Gurobi log path under the run artifact root."""

    root = Path(s.artifact_root).resolve() / "gurobi_logs"
    family = _safe_name(label or _ACTIVE_LABEL)
    parts = [_safe_name(site)]
    if durations is not None:
        parts.append(_duration_hash(np.asarray(durations, dtype=float)))
    if suffix:
        parts.append(_safe_name(suffix))
    filename = "__".join(parts) + ".log"
    return str(root / family / f"week_{int(week):03d}" / filename)


def _logged_site_assignment(
    week: base.WeekBundle,
    durations: np.ndarray,
    s,
    site: str,
    *,
    time_limit: int,
    mip_gap: float,
    threads: int,
    warm: ScheduleColumn | None,
    objective_mode: str = final.PRIMARY_OBJECTIVE,
):
    view = final._site_view(week, site)
    local_durations = np.asarray(durations, dtype=float)[view.global_indices]
    cfg = SolverConfig(
        time_limit_seconds=max(1, int(time_limit)),
        mip_gap=max(0.0, float(mip_gap)),
        threads=max(1, int(threads)),
        verbose=bool(s.verbose),
        mip_gap_abs=1e-10,
        seed=int(s.random_seed),
        log_file=weekly_log_file(
            s,
            week=week.position,
            site=site,
            label=_ACTIVE_LABEL,
            durations=local_durations,
            suffix=f"t{int(time_limit)}_g{float(mip_gap):.6g}",
        ),
    )
    result = solve_fixed_capacity_assignment(
        view.instance,
        local_durations,
        final.final_cost_cfg(s),
        final.PRIMARY_TURNOVER,
        cfg,
        objective_mode=objective_mode,
        warm_start=final._localize_warm(warm, view),
        symmetry_breaking=True,
    )
    return view, local_durations, result


def _logged_fixed_site_once(
    week,
    durations,
    s,
    site: str,
    *,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    seed: int,
    turnover: float,
):
    view = final._site_view(week, site)
    local_d = np.asarray(durations, dtype=float)[view.global_indices]
    cfg = SolverConfig(
        time_limit_seconds=max(1, int(wall_seconds)),
        work_limit=float(work_limit),
        mip_gap=max(0.0, float(mip_gap)),
        threads=1,
        verbose=bool(s.verbose),
        mip_gap_abs=1e-10,
        seed=int(seed),
        log_file=weekly_log_file(
            s,
            week=week.position,
            site=site,
            label=_ACTIVE_LABEL,
            durations=local_d,
            suffix=(
                f"work{float(work_limit):.6g}_wall{int(wall_seconds)}_"
                f"gap{float(mip_gap):.6g}_tau{float(turnover):.6g}"
            ),
        ),
    )
    result = solve_fixed_capacity_assignment(
        view.instance,
        local_d,
        final.final_cost_cfg(s),
        float(turnover),
        cfg,
        objective_mode=final.PRIMARY_OBJECTIVE,
        symmetry_breaking=True,
    )
    if result.column is None or any(
        x is None for x in (result.phi_ub, result.phi_lb, result.psi_ub, result.psi_lb)
    ):
        raise RuntimeError(
            f"Week {week.position} site {site}: deterministic planner returned no incumbent/bound "
            f"({result.diagnostics.status})"
        )
    return view, result


def install_weekly_logging() -> None:
    """Install persistent logging in both time-limited and WorkLimit site solves."""

    final._solve_site_assignment = _logged_site_assignment
    hardening._fixed_site_once = _logged_fixed_site_once


def training_process_task_logged(*args, **kwargs):
    global _ACTIVE_LABEL
    _ACTIVE_LABEL = str(kwargs.get("label", "training"))
    install_weekly_logging()
    return _ORIGINAL_TRAIN_WORKER(*args, **kwargs)


def deterministic_eval_worker_logged(*args, **kwargs):
    global _ACTIVE_LABEL
    _ACTIVE_LABEL = str(kwargs.get("label", "deterministic"))
    install_weekly_logging()
    return _ORIGINAL_DETERMINISTIC_WORKER(*args, **kwargs)


def sensitivity_worker_logged(*args, **kwargs):
    global _ACTIVE_LABEL
    _ACTIVE_LABEL = str(kwargs.get("label", "sensitivity"))
    install_weekly_logging()
    return _ORIGINAL_SENSITIVITY_WORKER(*args, **kwargs)


def install_spawn_safe_weekly_logging() -> None:
    """Make the logged workers the functions selected by the reviewed guard.

    Assigning these module functions (rather than a parent-only monkeypatch) is
    intentional: ProcessPoolExecutor with macOS spawn imports this module in each
    child, so the child installs the same log-aware site solver before solving.
    """

    numeric.training_process_task = training_process_task_logged
    numeric.deterministic_eval_worker = deterministic_eval_worker_logged
    numeric.sensitivity_worker = sensitivity_worker_logged
    install_weekly_logging()


def flexible_final_validate(self) -> None:
    """Freeze the planning design while allowing user-declared response regimes."""

    if self.site.upper() != "TGH+TWH":
        raise ValueError("Final experiment is frozen to pooled TGH+TWH.")
    if self.train_weeks != 72 or self.holdout_weeks != 22:
        raise ValueError("Final split is frozen to 72 train / 22 holdout weeks.")
    if not (0.0 <= float(self.alpha) < 1.0):
        raise ValueError("Behavioral alpha must satisfy 0 <= alpha < 1.")
    if not math.isfinite(float(self.h)) or float(self.h) <= 0:
        raise ValueError("Behavioral h must be finite and positive.")
    if not math.isfinite(float(self.display_cap)) or float(self.display_cap) <= 0:
        raise ValueError("display_cap must be finite and positive.")
    if abs(float(self.turnover) - float(final.PRIMARY_TURNOVER)) > 1e-12:
        raise ValueError("Primary paper turnover is frozen to 30 minutes.")
    if abs(float(self.opening)) > 1e-12:
        raise ValueError("Fixed capacity has zero block activation cost.")
    if abs(float(self.oracle_gap) - float(final.PRIMARY_ORACLE_GAP)) > 1e-12:
        raise ValueError("Primary oracle target is frozen to a 0.1% native Psi gap.")
    if abs(float(self.oracle_numeric_tol) - float(final.PRIMARY_ORACLE_RETRY_TOL)) > 1e-12:
        raise ValueError("Primary oracle retry tolerance is frozen to 0.1%.")
    if int(self.cores) <= 0:
        raise ValueError("cores must be positive")
    if float(self.max_wall_minutes) <= float(self.final_reserve_minutes):
        raise ValueError("max wall must exceed final reserve")


def install_flexible_behavior_validation() -> None:
    # ScientificFinalSettings.validate calls super().validate(), so replacing the
    # FinalSettings method preserves its coefficient/work-limit checks while
    # removing only the old alpha=.8,h=30 freeze.
    final.FinalSettings.validate = flexible_final_validate


def _block_payload(bid: BlockId) -> list[Any]:
    return [int(bid.day_index), str(bid.site), str(bid.room)]


def _block_from_payload(value: Sequence[Any]) -> BlockId:
    if len(value) != 3:
        raise ValueError(f"Invalid block payload: {value!r}")
    return BlockId(int(value[0]), str(value[1]), str(value[2]))


def _week_signature(weeks: Sequence[base.WeekBundle]) -> str:
    payload = []
    for week in sorted(weeks, key=lambda w: int(w.position)):
        inst = week.instance
        payload.append(
            {
                "week": int(week.position),
                "start": str(week.start.date()),
                "case_ids": [str(c.case_id) for c in inst.cases],
                "blocks": [
                    [*_block_payload(b.id), float(b.capacity_minutes)]
                    for b in inst.calendar.candidates
                ],
                "eligibility": [
                    [_block_payload(b) for b in inst.case_eligible_blocks.get(i, [])]
                    for i in range(inst.num_cases)
                ],
            }
        )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _manifest_path(root: Path) -> Path:
    return Path(root) / "SHARED_PLANS_MANIFEST.json"


def _plan_path(root: Path, kind: str) -> Path:
    return Path(root) / f"{kind.upper()}_TRAIN_PLANS.json.gz"


def _summary_path(root: Path, kind: str) -> Path:
    return Path(root) / f"{kind.upper()}_TRAIN.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _plan_payload(plan: base.PlanResult) -> dict[str, Any]:
    assignment: list[list[Any]] = []
    for i in range(plan.column.n_cases):
        bids = [
            bid
            for (j, bid), value in plan.column.z_assign.items()
            if int(j) == i and float(value) > 0.5
        ]
        if len(bids) != 1:
            raise ValueError(f"Week {plan.week}: case {i} has {len(bids)} stored assignments")
        assignment.append(_block_payload(bids[0]))
    return {
        "week": int(plan.week),
        "n_cases": int(plan.column.n_cases),
        "assignment": assignment,
        "objective_phi": float(plan.objective),
        "bound_phi": float(plan.bound),
        "gap_native_psi": float(plan.gap),
        "status": str(plan.status),
        "solve_seconds": float(plan.solve_seconds),
        "exact": bool(plan.exact),
        "tiebreak_used": bool(plan.tiebreak_used),
    }


def _native_gap(week, durations: np.ndarray, s, phi_ub: float, phi_lb: float) -> float:
    d = np.asarray(durations, dtype=float)
    capacity = sum(float(b.capacity_minutes) for b in week.instance.calendar.candidates)
    k = float(s.idle) * (
        capacity - float(d.sum()) - float(final.PRIMARY_TURNOVER) * week.instance.num_cases
    )
    return float(base.rel_gap(float(phi_ub) - k, float(phi_lb) - k))


def _current_manifest_base(weeks: Sequence[base.WeekBundle], s) -> dict[str, Any]:
    data_path = Path(s.data).resolve()
    return {
        "version": SHARED_PLANS_VERSION,
        "scientific_spec_version": science.SCIENTIFIC_SPEC_VERSION,
        "source_git_head": base.git_head(),
        "data_path_name": data_path.name,
        "input_sha256": base.sha256_file(data_path),
        "week_signature": _week_signature(weeks),
        "train_weeks": len(weeks),
        "sites": list(final.PRIMARY_SITES),
        "roster": science.PRIMARY_ROSTER,
        "turnover_minutes": float(final.PRIMARY_TURNOVER),
        "overtime_per_minute": float(s.overtime),
        "idle_per_minute": float(s.idle),
        "objective_mode": final.PRIMARY_OBJECTIVE,
    }


def save_plan_set(
    root: Path,
    kind: str,
    plans: Mapping[int, base.PlanResult],
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s,
) -> None:
    """Persist one behavior-independent weekly plan set plus its certificates."""

    kind = str(kind).lower()
    if kind not in {"oracle", "booked"}:
        raise ValueError("kind must be 'oracle' or 'booked'")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = {int(w.position) for w in weeks}
    if set(map(int, plans)) != expected:
        raise RuntimeError(f"{kind} plan set does not cover all training weeks")

    week_lookup = {int(w.position): w for w in weeks}
    payloads = []
    rows = []
    for wk in sorted(expected):
        plan = plans[wk]
        d = np.asarray(duration_by_week[wk], dtype=float)
        recomputed = float(
            plan.column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
        )
        numeric.assert_phi_accounting_close(
            recomputed,
            float(plan.objective),
            context=f"shared {kind} save week {wk}",
        )
        if float(plan.bound) > recomputed + numeric.phi_accounting_tolerance(recomputed, plan.bound):
            raise RuntimeError(f"{kind} week {wk}: lower bound exceeds feasible incumbent")
        p = _plan_payload(plan)
        p["objective_phi"] = recomputed
        p["gap_native_psi"] = _native_gap(
            week_lookup[wk], d, s, recomputed, float(plan.bound)
        )
        payloads.append(p)
        rows.append(
            {
                "week": wk,
                "week_start": str(week_lookup[wk].start.date()),
                "objective_phi": p["objective_phi"],
                "bound_phi": p["bound_phi"],
                "gap_native_psi": p["gap_native_psi"],
                "status": p["status"],
                "exact": p["exact"],
                "solve_seconds": p["solve_seconds"],
            }
        )

    path = _plan_path(root, kind)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"kind": kind, "plans": payloads}, f, sort_keys=True)
    base.write_csv(_summary_path(root, kind), rows)

    manifest_path = _manifest_path(root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = _current_manifest_base(weeks, s)
        for key, value in current.items():
            if key in manifest and manifest[key] != value and key != "source_git_head":
                raise RuntimeError(f"Shared-plan manifest mismatch for {key}: {manifest[key]!r} != {value!r}")
        manifest.update(current)
    else:
        manifest = _current_manifest_base(weeks, s)
    kinds = dict(manifest.get("plan_sets", {}))
    kinds[kind] = {
        "file": path.name,
        "sha256": _sha256(path),
        "summary_file": _summary_path(root, kind).name,
        "summary_sha256": _sha256(_summary_path(root, kind)),
        "weeks": len(payloads),
        "max_native_psi_gap": max(float(p["gap_native_psi"]) for p in payloads),
        "exact_weeks": sum(bool(p["exact"]) for p in payloads),
    }
    manifest["plan_sets"] = kinds
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_plan_set(
    root: Path,
    kind: str,
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s,
) -> dict[int, base.PlanResult]:
    """Load and aggressively validate a reusable plan set against current weeks."""

    kind = str(kind).lower()
    root = Path(root).resolve()
    manifest_path = _manifest_path(root)
    if not manifest_path.exists():
        raise RuntimeError(f"Shared plan root has no {manifest_path.name}: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _current_manifest_base(weeks, s)
    for key in (
        "version",
        "scientific_spec_version",
        "input_sha256",
        "week_signature",
        "train_weeks",
        "sites",
        "roster",
        "turnover_minutes",
        "overtime_per_minute",
        "idle_per_minute",
        "objective_mode",
    ):
        if manifest.get(key) != current.get(key):
            raise RuntimeError(
                f"Shared plan artifact is incompatible at {key}: "
                f"stored={manifest.get(key)!r}, current={current.get(key)!r}"
            )
    meta = dict(manifest.get("plan_sets", {})).get(kind)
    if not meta:
        raise RuntimeError(f"Shared plan artifact has no {kind!r} plan set")
    path = root / str(meta["file"])
    if not path.exists() or _sha256(path) != str(meta["sha256"]):
        raise RuntimeError(f"Shared {kind} plan file is missing or fingerprint changed")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    entries = {int(p["week"]): p for p in payload.get("plans", [])}
    lookup = {int(w.position): w for w in weeks}
    if set(entries) != set(lookup):
        raise RuntimeError(f"Shared {kind} plan weeks do not match current training weeks")

    out: dict[int, base.PlanResult] = {}
    for wk, week in sorted(lookup.items()):
        p = entries[wk]
        if int(p["n_cases"]) != week.instance.num_cases or len(p["assignment"]) != week.instance.num_cases:
            raise RuntimeError(f"Shared {kind} week {wk} case count changed")
        assignment = {
            i: _block_from_payload(value)
            for i, value in enumerate(p["assignment"])
        }
        column = column_from_assignment(week.instance, assignment, enforce_eligibility=True)
        d = np.asarray(duration_by_week[wk], dtype=float)
        recomputed = float(
            column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
        )
        numeric.assert_phi_accounting_close(
            recomputed,
            float(p["objective_phi"]),
            context=f"shared {kind} load week {wk}",
        )
        bound = float(p["bound_phi"])
        tol = numeric.phi_accounting_tolerance(recomputed, bound)
        if bound > recomputed + tol:
            raise RuntimeError(f"Shared {kind} week {wk}: lower bound exceeds current feasible cost")
        gap = _native_gap(week, d, s, recomputed, bound)
        out[wk] = base.PlanResult(
            week=wk,
            column=column,
            objective=recomputed,
            bound=bound,
            gap=gap,
            status=str(p["status"]),
            solve_seconds=float(p["solve_seconds"]),
            exact=bool(p["exact"] and abs(recomputed - bound) <= 1e-6),
            tiebreak_used=bool(p.get("tiebreak_used", False)),
        )
    return out


def merge_refinement(
    old: Mapping[int, base.PlanResult],
    new: Mapping[int, base.PlanResult],
    weeks: Sequence[base.WeekBundle],
    duration_by_week: Mapping[int, np.ndarray],
    s,
) -> dict[int, base.PlanResult]:
    """Keep the best feasible incumbent and strongest valid lower bound per week."""

    lookup = {int(w.position): w for w in weeks}
    if set(old) != set(new) or set(old) != set(lookup):
        raise RuntimeError("Refinement plan sets must cover the identical weeks")
    out: dict[int, base.PlanResult] = {}
    for wk, week in lookup.items():
        a, b = old[wk], new[wk]
        chosen = b if float(b.objective) <= float(a.objective) + 1e-9 else a
        ub = min(float(a.objective), float(b.objective))
        lb = max(float(a.bound), float(b.bound))
        d = np.asarray(duration_by_week[wk], dtype=float)
        feasible = float(
            chosen.column.compute_cost(d, final.final_cost_cfg(s), final.PRIMARY_TURNOVER)
        )
        numeric.assert_phi_accounting_close(
            feasible, ub, context=f"shared refinement week {wk}"
        )
        ub = feasible
        if lb > ub + numeric.phi_accounting_tolerance(ub, lb):
            raise RuntimeError(f"Refinement week {wk}: merged lower bound exceeds incumbent")
        out[wk] = base.PlanResult(
            week=wk,
            column=chosen.column,
            objective=ub,
            bound=lb,
            gap=_native_gap(week, d, s, ub, lb),
            status=f"REFINED[{a.status}|{b.status}]",
            solve_seconds=float(a.solve_seconds + b.solve_seconds),
            exact=bool(a.exact or b.exact),
            tiebreak_used=bool(chosen.tiebreak_used),
        )
    return out


def install_shared_plan_cache(shared_root: Path) -> None:
    """Make Stage 1 reuse shared oracle/BOOKED plans and fail if they are absent."""

    root = Path(shared_root).resolve()

    def cached_oracle(weeks, duration_by_week, s, *, label):
        if str(label) == "oracle_train":
            base.LOG.info("[SHARED] loading training realized oracle from %s", root)
            return load_plan_set(root, "oracle", weeks, duration_by_week, s)
        return _ORIGINAL_RUNTIME_ORACLE(weeks, duration_by_week, s, label=label)

    def cached_batch(
        weeks,
        duration_by_week,
        s,
        *,
        seconds,
        gap,
        label,
        warm_by_week=None,
        deterministic_tiebreak=False,
        tiebreak_seconds=None,
    ):
        if str(label) == "booked_train":
            base.LOG.info("[SHARED] loading training BOOKED schedules from %s", root)
            return load_plan_set(root, "booked", weeks, duration_by_week, s)
        return _ORIGINAL_RUNTIME_BATCH(
            weeks,
            duration_by_week,
            s,
            seconds=seconds,
            gap=gap,
            label=label,
            warm_by_week=warm_by_week,
            deterministic_tiebreak=deterministic_tiebreak,
            tiebreak_seconds=tiebreak_seconds,
        )

    # apply_runtime_fixes() runs inside the stage main after the wrapper. It
    # reads these module attributes, so patching the source hooks survives that
    # initialization sequence.
    runtime.safe_solve_oracle_batch = cached_oracle
    runtime.safe_solve_batch = cached_batch


def configure_training_scenario(
    training_module,
    *,
    alpha: float,
    h: float,
    scenario_name: str,
    shared_root: Path | None,
) -> None:
    """Inject an arbitrary valid response regime into the reviewed Stage 1."""

    install_flexible_behavior_validation()
    original = training_module._settings_from_args

    def settings_from_args(args, artifact_root):
        s = original(args, artifact_root)
        s.alpha = float(alpha)
        s.h = float(h)
        return s

    training_module._settings_from_args = settings_from_args
    training_module.FINAL_BEHAVIOR_SCENARIO = {
        "name": str(scenario_name),
        "alpha": float(alpha),
        "h": float(h),
        "shared_plans_root": None if shared_root is None else str(Path(shared_root).resolve()),
    }
    if shared_root is not None:
        install_shared_plan_cache(Path(shared_root))


def stamp_training_bundle(
    root: Path,
    *,
    scenario_name: str,
    alpha: float,
    h: float,
    shared_root: Path | None,
) -> None:
    """Freeze scenario identity and shared-plan provenance into Stage-1 artifacts."""

    root = Path(root).resolve()
    scenario_path = root / "BEHAVIOR_SCENARIO.json"
    shared_manifest_sha = None
    shared_manifest_path = None
    if shared_root is not None:
        shared_manifest_path = _manifest_path(Path(shared_root).resolve())
        if not shared_manifest_path.exists():
            raise RuntimeError("Shared-plan manifest disappeared before Stage-1 freeze")
        shared_manifest_sha = _sha256(shared_manifest_path)
    payload = {
        "version": SHARED_PLANS_VERSION,
        "name": str(scenario_name),
        "alpha": float(alpha),
        "h": float(h),
        "shared_plans_root": None if shared_root is None else str(Path(shared_root).resolve()),
        "shared_plans_manifest_sha256": shared_manifest_sha,
        "interpretation": (
            "Behavior-specific policy training; realized-oracle and BOOKED weekly plans are "
            "behavior-independent and may be reused from the validated shared backbone."
        ),
    }
    scenario_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError("Cannot stamp behavior scenario: TRAINING_FREEZE.json missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["shared_plans_version"] = SHARED_PLANS_VERSION
    freeze["behavior_scenario"] = {"name": str(scenario_name), "alpha": float(alpha), "h": float(h)}
    if shared_manifest_sha is not None:
        freeze["shared_plans_manifest_sha256"] = shared_manifest_sha
    fps = dict(freeze.get("artifact_fingerprints", {}))
    fps[scenario_path.name] = _sha256(scenario_path)
    freeze["artifact_fingerprints"] = fps
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")


def verify_shared_plan_provenance(training_root: Path) -> None:
    """Verify the scenario stamp when present; older primary bundles remain readable."""

    root = Path(training_root).resolve()
    freeze_path = root / "TRAINING_FREEZE.json"
    if not freeze_path.exists():
        return
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if "shared_plans_version" not in freeze:
        return
    if freeze.get("shared_plans_version") != SHARED_PLANS_VERSION:
        raise RuntimeError("Training bundle uses a different shared-plan format version")
    scenario = root / "BEHAVIOR_SCENARIO.json"
    expected = dict(freeze.get("artifact_fingerprints", {})).get(scenario.name)
    if not scenario.exists() or not expected or _sha256(scenario) != expected:
        raise RuntimeError("Frozen BEHAVIOR_SCENARIO.json is missing or changed")
