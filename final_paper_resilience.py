"""Non-destructive resilience for expensive final-paper runs.

The core scientific validity checks remain hard failures.  This layer only
prevents *diagnostic* or emergency-fail-safe logic from discarding an otherwise
valid many-hour run:

* the late training tie/seed audit is diagnostic and therefore records a failed
  diagnostic row instead of aborting Stage 1;
* the optional saturation stress is diagnostic and therefore records its
  exception in SATURATION_STATUS rather than aborting after policies are frozen;
* if deterministic WorkLimit scheduling hits the emergency wall clock twice but
  the second solve still returned a valid incumbent and bound, keep that valid
  result with TIME_LIMIT status instead of throwing it away.

Missing incumbents/bounds, invalid schedules, non-finite accounting, data-freeze
changes, pinned-library corruption, unattempted VF search, and material Phi
accounting mismatches remain hard failures.
"""
from __future__ import annotations

import math
import traceback

import final_paper_finalization_fixes as hardening
import final_paper_runtime_fixes as runtime
import run_final_paper_training as training
import run_final_vf_experiment as base

RESILIENCE_VERSION = "final_paper_resilience_2026_09_25_v1"
_INSTALLED = False
_ORIGINAL_TIE_AUDIT = training._training_tie_seed_audit
_ORIGINAL_SATURATION = runtime.safe_saturation_test


def safe_training_tie_seed_audit(train_weeks, train_a, s):
    """Never discard trained policies because this post-training diagnostic failed."""
    try:
        return _ORIGINAL_TIE_AUDIT(train_weeks, train_a, s)
    except Exception as exc:
        base.LOG.warning(
            "[DIAGNOSTIC-WARN] tie/seed audit failed after core policy training; "
            "continuing and recording the diagnostic failure: %s",
            exc,
            exc_info=True,
        )
        return [
            {
                "week": -1,
                "week_start": "",
                "seed": -1,
                "planning_objective_phi": float("nan"),
                "planning_bound_phi": float("nan"),
                "planning_gap_native_psi": float("nan"),
                "status": "DIAGNOSTIC_FAILED_CONTINUED",
                "realized_cost": float("nan"),
                "error": repr(exc),
            }
        ]


def safe_saturation_test(*args, **kwargs):
    """Treat saturation as an optional diagnostic, never as a Stage-1 validity gate."""
    try:
        return _ORIGINAL_SATURATION(*args, **kwargs)
    except Exception as exc:
        base.LOG.warning(
            "[DIAGNOSTIC-WARN] optional saturation stress failed after core policy "
            "training; continuing and recording the failure: %s",
            exc,
            exc_info=True,
        )
        return [], {
            "skipped": False,
            "status": "DIAGNOSTIC_FAILED_CONTINUED",
            "reason": repr(exc),
            "traceback": traceback.format_exc(),
            "policies_already_frozen_before_saturation": True,
            "partial_library_enrichment_may_have_occurred": True,
        }


def resilient_deterministic_site_solve(
    week,
    durations,
    s,
    site: str,
    *,
    work_limit: float,
    wall_seconds: int,
    mip_gap: float,
    seed: int,
):
    """Keep a valid second TIME_LIMIT incumbent instead of aborting the run.

    ``_fixed_site_once`` already refuses results without a complete incumbent
    and bound.  Therefore a second TIME_LIMIT reaching this function is still a
    legitimate bounded solve; it is simply not the intended deterministic
    WorkLimit stopping event.  We preserve it, mark TIME_LIMIT through the
    returned diagnostics, and let downstream reporting expose the status/gap.
    """
    initial_wall = max(
        int(wall_seconds),
        int(math.ceil(hardening.EMERGENCY_WALL_TO_WORK_MULTIPLIER * float(work_limit))),
    )
    view, result = hardening._fixed_site_once(
        week,
        durations,
        s,
        site,
        work_limit=work_limit,
        wall_seconds=initial_wall,
        mip_gap=mip_gap,
        seed=seed,
        turnover=__import__("run_final_paper_experiment").PRIMARY_TURNOVER,
    )
    if result.diagnostics.status != "TIME_LIMIT":
        return view, result

    retry_wall = max(
        initial_wall + 1,
        int(math.ceil(initial_wall * hardening.EMERGENCY_RETRY_MULTIPLIER)),
    )
    base.LOG.warning(
        "[DET-PLANNER] week=%s site=%s emergency wall cap %ss fired; retrying "
        "with %ss while preserving WorkLimit=%.3g",
        week.position,
        site,
        initial_wall,
        retry_wall,
        work_limit,
    )
    view, result = hardening._fixed_site_once(
        week,
        durations,
        s,
        site,
        work_limit=work_limit,
        wall_seconds=retry_wall,
        mip_gap=mip_gap,
        seed=seed,
        turnover=__import__("run_final_paper_experiment").PRIMARY_TURNOVER,
    )
    if result.diagnostics.status == "TIME_LIMIT":
        base.LOG.warning(
            "[DET-PLANNER-WARN] week=%s site=%s emergency wall cap fired twice, "
            "but the second solve returned a valid incumbent and bound; keeping it "
            "with TIME_LIMIT status instead of aborting the expensive run.",
            week.position,
            site,
        )
    return view, result


def install() -> None:
    """Install idempotently before the supported wrapper starts a stage."""
    global _INSTALLED
    if _INSTALLED:
        return
    training._training_tie_seed_audit = safe_training_tie_seed_audit
    runtime.safe_saturation_test = safe_saturation_test
    hardening.robust_deterministic_site_solve = resilient_deterministic_site_solve
    _INSTALLED = True
