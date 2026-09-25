"""Non-destructive resilience for expensive final-paper runs.

Core scientific validity checks remain hard failures. This layer prevents
*diagnostic*, candidate-selection, or emergency-fail-safe logic from discarding
an otherwise valid many-hour run:

* the late training tie/seed audit is diagnostic and records failure instead of
  aborting Stage 1;
* optional saturation records failure instead of aborting after policies exist;
* an anomalous post-refresh VF candidate is rejected and the previous certified
  policy is retained instead of terminating the entire search;
* if a deterministic WorkLimit solve hits the emergency wall clock twice but the
  second solve returned a valid incumbent and bound, that bounded result is kept
  with TIME_LIMIT status instead of being thrown away. This applies to primary
  evaluation and structural sensitivities.

Missing incumbents/bounds, invalid schedules, non-finite accounting, data-freeze
changes, pinned-library corruption, an entirely unattempted VF search, and
material Phi-accounting mismatches remain hard failures because accepting those
would make the final scientific artifact untrustworthy.
"""
from __future__ import annotations

import math
import time
import traceback

import numpy as np

import final_paper_finalization_fixes as hardening
import final_paper_numeric_guard as numeric
import final_paper_runtime_fixes as runtime
import run_final_paper_experiment as final
import run_final_paper_training as training
import run_final_vf_experiment as base
from src.solvers.fixed_capacity import schedule_metrics

RESILIENCE_VERSION = "final_paper_resilience_2026_09_25_v2"
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


def resilient_train_vf(
    start_w,
    a,
    weeks,
    lib,
    oracle_lb,
    s,
    lam,
    root,
    *,
    search_deadline=None,
):
    """VF outer loop that rejects a bad refreshed candidate instead of crashing.

    Library enrichment is monotone: adding schedules cannot invalidate an old
    library certificate. If floating-point/implementation noise ever reports a
    refreshed candidate above the old anchor, we keep the old policy and old
    certified value, retain any newly discovered schedules, record the rejected
    step, and continue. This is safer than terminating a many-hour run and does
    not manufacture a descent claim.
    """
    cur = base.project_policy(start_w, a, s)
    traj = []
    stagnant = 0
    current_plans = base.plan_policy_training(
        cur,
        a,
        weeks,
        lib,
        s,
        seconds=s.train_planner_seconds,
        gap=s.train_planner_gap,
        label="vf_anchor_0",
    )
    lib.add_plans(current_plans, "VF_ANCHOR_0")

    for outer in range(1, s.vf_outer + 1):
        if search_deadline is not None and base.remaining(search_deadline) < 120:
            base.LOG.warning(
                "[VF] stopping before outer %d to preserve final-evaluation reserve", outer
            )
            break

        before_pool = lib.size()
        anchor = base.library_metrics(cur, a, lib, oracle_lb, s, lam)
        _, _, cur_planning = base.correction_and_planning(cur, a, s)
        eps_parts = []
        for wk, idx in a.week_slices.items():
            lib_obj = lib.best(wk, cur_planning[idx], s)[1]
            eps_parts.append(max(0.0, lib_obj - current_plans[wk].bound))
        eps = float(np.mean(eps_parts))

        fixed = base.selected_library_columns(cur, a, lib, s)
        spec = base.FixedSpec(
            f"VF_OUTER_{outer}",
            a,
            np.full(len(a.error), s.overtime),
            np.full(len(a.error), s.idle),
            fixed,
            lam,
            s,
        )
        cand, hist = base.run_pdca(
            spec, cur, s, max_iterations=s.pdca_vf_inner_iterations
        )
        candm = base.library_metrics(cand, a, lib, oracle_lb, s, lam)
        accepted = bool(
            candm["certificate"]
            <= anchor["certificate"] + 1e-7 * max(1.0, abs(anchor["certificate"]))
        )
        D = max(0.0, anchor["certificate"] - candm["certificate"])
        maxdw = float(np.max(np.abs(cand - cur)))
        post_refresh_rejected = False

        if accepted:
            candidate_plans = base.plan_policy_training(
                cand,
                a,
                weeks,
                lib,
                s,
                seconds=s.train_planner_seconds,
                gap=s.train_planner_gap,
                label=f"vf_candidate_{outer}",
            )
            cadded = lib.add_plans(candidate_plans, f"VF_CANDIDATE_{outer}")
            candidate_changes = 0
            for wk in a.week_slices:
                sig0 = base.canonical_schedule_signature(
                    current_plans[wk].column, lib.week_lookup[wk].instance
                )
                sig1 = base.canonical_schedule_signature(
                    candidate_plans[wk].column, lib.week_lookup[wk].instance
                )
                candidate_changes += int(sig0 != sig1)

            refreshed = base.library_metrics(cand, a, lib, oracle_lb, s, lam)
            allowed = anchor["certificate"] + 1e-6 * max(
                1.0, abs(anchor["certificate"])
            )
            if refreshed["certificate"] <= allowed:
                finalm = refreshed
                cplans = candidate_plans
                changes = candidate_changes
            else:
                # A candidate-selection/checking anomaly is recoverable: the old
                # policy and its already-valid certificate remain available, and
                # the newly found schedules can only enlarge the library.
                post_refresh_rejected = True
                accepted = False
                base.LOG.warning(
                    "[VF-WARN] outer=%d refreshed candidate certificate %.12g exceeds "
                    "anchor %.12g; rejecting the candidate, retaining the previous "
                    "policy, and continuing with the enriched library.",
                    outer,
                    refreshed["certificate"],
                    anchor["certificate"],
                )
                cand = cur.copy()
                cplans = current_plans
                changes = 0
                maxdw = 0.0
                # The pre-refresh anchor remains a valid certificate because its
                # schedules are still present in the enlarged library.
                finalm = dict(anchor)
        else:
            cand = cur.copy()
            candm = dict(anchor)
            cplans = current_plans
            cadded = 0
            changes = 0
            maxdw = 0.0
            finalm = dict(anchor)

        rel = (
            anchor["certificate"] - finalm["certificate"]
        ) / max(1.0, abs(anchor["certificate"]))
        row = {
            "outer": outer,
            "pool_before": before_pool,
            "candidate_added": cadded,
            "pool_after": lib.size(),
            "certificate_anchor": anchor["certificate"],
            "certificate_candidate_before_refresh": candm["certificate"],
            "certificate_final": finalm["certificate"],
            "relative_total_tightening": rel,
            "policy_decrease_D": D,
            "planner_uncertainty_epsilon": eps,
            "true_vf_descent_certified": bool(
                accepted and (not post_refresh_rejected) and D > eps + 1e-8
            ),
            "max_anchor_planner_gap": max(r.gap for r in current_plans.values()),
            "max_candidate_planner_gap": max(r.gap for r in cplans.values()),
            "schedule_changes": changes,
            "accepted": accepted,
            "post_refresh_guard_rejected": post_refresh_rejected,
            "pdca_iterations": len(hist),
            "max_abs_dw": maxdw,
        }
        traj.append(row)
        base.LOG.info(
            "[VF] outer=%d C %.3f -> %.3f | add=%d | D=%.2f eps=%.2f | "
            "changes=%d | accepted=%s | post_refresh_rejected=%s",
            outer,
            anchor["certificate"],
            finalm["certificate"],
            cadded,
            D,
            eps,
            changes,
            accepted,
            post_refresh_rejected,
        )
        cur = cand
        current_plans = cplans
        if rel < s.vf_stop_relative_certificate and changes <= s.vf_stop_schedule_changes:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= s.vf_stop_patience:
            break

    base.write_csv(root / "VF_TRAJECTORY.csv", traj)
    return cur, traj


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
    """Keep a valid second TIME_LIMIT incumbent instead of aborting the run."""
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
        turnover=final.PRIMARY_TURNOVER,
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
        turnover=final.PRIMARY_TURNOVER,
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


def resilient_sensitivity_worker(
    week,
    durations,
    s,
    *,
    turnover,
    work_limit,
    wall_seconds,
    mip_gap,
    label,
):
    """Sensitivity analogue of the non-destructive deterministic wall-cap rule."""
    d = np.asarray(durations, float)
    t0 = time.perf_counter()
    parts = []
    solver_phi_ub = phi_lb = solver_psi_ub = psi_lb = 0.0
    statuses = []
    all_optimal = True
    initial_wall = max(
        int(wall_seconds),
        int(math.ceil(hardening.EMERGENCY_WALL_TO_WORK_MULTIPLIER * float(work_limit))),
    )
    for site in final.PRIMARY_SITES:
        view, result = hardening._fixed_site_once(
            week,
            d,
            s,
            site,
            work_limit=work_limit,
            wall_seconds=initial_wall,
            mip_gap=mip_gap,
            seed=int(s.random_seed),
            turnover=float(turnover),
        )
        if result.diagnostics.status == "TIME_LIMIT":
            retry_wall = max(
                initial_wall + 1,
                int(math.ceil(initial_wall * hardening.EMERGENCY_RETRY_MULTIPLIER)),
            )
            view, result = hardening._fixed_site_once(
                week,
                d,
                s,
                site,
                work_limit=work_limit,
                wall_seconds=retry_wall,
                mip_gap=mip_gap,
                seed=int(s.random_seed),
                turnover=float(turnover),
            )
            if result.diagnostics.status == "TIME_LIMIT":
                base.LOG.warning(
                    "[SENSITIVITY-WARN] %s week=%s site=%s wall cap fired twice; "
                    "keeping the valid incumbent/bound with TIME_LIMIT status.",
                    label,
                    week.position,
                    site,
                )
        # _fixed_site_once already hard-fails if incumbent/bounds are absent.
        parts.append((view, result.column))
        solver_phi_ub += float(result.phi_ub)
        phi_lb += float(result.phi_lb)
        solver_psi_ub += float(result.psi_ub)
        psi_lb += float(result.psi_lb)
        statuses.append(f"{site}:{result.diagnostics.status}")
        all_optimal = all_optimal and bool(result.diagnostics.proven_optimal)

    column = final._merge_site_columns(week.instance, parts)
    recomputed_phi = float(
        schedule_metrics(column, d, final.final_cost_cfg(s), float(turnover))["phi"]
    )
    numeric.assert_phi_accounting_close(
        recomputed_phi,
        solver_phi_ub,
        context=f"Sensitivity week {week.position} {label}",
    )
    return base.PlanResult(
        week=week.position,
        column=column,
        objective=recomputed_phi,
        bound=float(phi_lb),
        gap=numeric._accounted_native_gap(
            week,
            d,
            s,
            turnover=float(turnover),
            recomputed_phi_ub=recomputed_phi,
            psi_lb=psi_lb,
        ),
        status="OPTIMAL" if all_optimal else "|".join(statuses),
        solve_seconds=time.perf_counter() - t0,
        exact=bool(all_optimal and abs(solver_psi_ub - psi_lb) <= 1e-6),
        tiebreak_used=False,
    )


def install() -> None:
    """Install idempotently before the supported wrapper starts a stage."""
    global _INSTALLED
    if _INSTALLED:
        return
    training._training_tie_seed_audit = safe_training_tie_seed_audit
    runtime.safe_saturation_test = safe_saturation_test
    hardening._ORIGINAL_TRAIN_VF = resilient_train_vf
    hardening.robust_deterministic_site_solve = resilient_deterministic_site_solve
    # final_paper_shared_plans imports later and captures this source worker;
    # protocol wrappers can then add logging without restoring the aborting rule.
    numeric.sensitivity_worker = resilient_sensitivity_worker
    _INSTALLED = True
