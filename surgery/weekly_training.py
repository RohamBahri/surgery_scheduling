"""Full-cohort response-aware CASE and schedule-aware VF learning.

CASE: minimize post-response asymmetric case error + L1.
VF: alternating exact weekly planning oracles with differentiable,
fixed-schedule majorization of the *valid* planning-regret upper bound.
Every accepted VF update decreases the unsmoothed library upper bound;
the freshly solved schedules certify the active predicted value function.
The hindsight oracle term is independent of w, so omitted during fitting.

No surrogate here is claimed to optimize GAP or SPO policies.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from .planner import display_and_duration
from .weekly_planner import solve_week, evaluate

C_OVER, C_UNDER, TURNOVER, CAPACITY = 1.75, 1.0, 30., 480.


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checksum_array(array):
    a = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def response_with_derivative(raw, booked, alpha, h):
    """Match display_and_duration exactly, including clipping and attenuation."""
    raw, booked = np.asarray(raw, float), np.asarray(booked, float)
    lo = np.maximum(-180., 1. - booked)
    clipped = np.clip(raw, lo, 180.)
    _, duration = display_and_duration(raw, booked, alpha, h)
    magnitude = np.abs(clipped)
    if h is None:
        dr = np.ones_like(raw)
    else:
        q = 1. - alpha
        dr = np.where(magnitude < h, alpha,
                      np.where((q > 0) & (q * magnitude < h), -q, 0.))
        # Choose the middle derivative at kinks; exact values handle acceptance.
        dr = np.where(np.isclose(magnitude, h, atol=1.e-10), .5 * (alpha - q), dr)
        if q > 0:
            dr = np.where(np.isclose(q * magnitude, h, atol=1.e-10), -.5 * q, dr)
    dr *= ((raw > lo) & (raw < 180.))
    return duration, dr


def asymmetric_error(error):
    e = np.asarray(error, float)
    return C_OVER * np.maximum(e, 0.) + C_UNDER * np.maximum(-e, 0.)


def smoothed_error(error, eps):
    e = np.asarray(error, float)
    v = eps * (C_OVER * np.logaddexp(0., e / eps)
               + C_UNDER * np.logaddexp(0., -e / eps))
    # derivative wrt predicted duration; e = actual - predicted
    d = -C_OVER * expit(e / eps) + C_UNDER * expit(-e / eps)
    return v, d


def training_lambda(X, seed=20261013, draws=160):
    """Calibrate penalty on TRAIN only, normalized by training cases."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws // 16):
        noise = rng.choice((-1., 1.), (16, len(X)))
        values.extend(np.max(np.abs(noise @ X[:, 1:]), axis=1))
    return float((C_OVER + C_UNDER) * 1.1 * np.quantile(values, .9) / len(X))


def fixed_room_matrix(weeks, plans, index, n):
    """Build fixed physical-room/day loads, including UNUSED staffed blocks."""
    rr, cc, count = [], [], []
    row = 0
    for week in weeks:
        key = week.group + "|" + week.monday
        plan = plans.get(key)
        if not plan or not plan.get("complete"):
            raise ValueError(f"Missing certified training schedule for {key}")
        chosen = {int(cid): (int(day), str(room)) for cid, day, room in plan["assignment"]}
        if set(chosen) != set(map(int, week.case_ids)):
            raise ValueError(f"Plan must assign every case once: {key}")
        slots = {slot: j for j, slot in enumerate(week.slots)}
        counts = np.zeros(week.k, dtype=int)
        for cid, slot in chosen.items():
            j = slots[slot]
            rr.append(row + j)
            cc.append(index[cid])
            counts[j] += 1
        count.extend(counts.tolist())
        row += week.k
    mat = csr_matrix((np.ones(len(cc)), (rr, cc)), shape=(row, n))
    counts = np.array(count)
    offsets = TURNOVER * np.maximum(counts - 1, 0)
    if (np.diff(mat.indptr) != counts).any():
        raise AssertionError("Room matrix counts inconsistent")
    return mat, offsets


def fixed_exact(w, X, booked, actual, alpha, h, lam, mat=None, offsets=None):
    duration, _ = response_with_derivative(X @ w, booked, alpha, h)
    loss = float(asymmetric_error(actual - duration).sum())
    if mat is not None:
        load = mat @ duration + offsets
        loss += float((C_OVER * np.maximum(load - CAPACITY, 0.)
                       + C_UNDER * np.maximum(CAPACITY - load, 0.)).sum())
    return loss / len(booked) + lam * np.abs(w[1:]).sum()


def fixed_smooth(w, X, booked, actual, alpha, h, lam, mat, offsets,
                 eps=2., l1_eps=.02):
    duration, dr = response_with_derivative(X @ w, booked, alpha, h)
    losses, gradient_duration = smoothed_error(actual - duration, eps)
    value = float(losses.sum())
    if mat is not None:
        load = mat @ duration + offsets
        extra, gd = smoothed_error(load - CAPACITY, eps)
        # over/under of room load (positive excess is overtime):
        # smoothed_error(e) derivative wrt e = -d (d is wrt prediction).
        value += float(extra.sum())
        gradient_duration = gradient_duration - mat.T @ gd
    per_case = len(booked)
    coef = np.asarray(w[1:], float)
    smooth_l1 = np.sqrt(coef * coef + l1_eps * l1_eps)
    value = value / per_case + lam * float(smooth_l1.sum())
    grad = X.T @ (gradient_duration * dr) / per_case
    grad[1:] += lam * coef / smooth_l1
    return value, np.asarray(grad, float)


def fit_fixed(X, booked, actual, alpha, h, lam, initial, mat=None,
              offsets=None, iterations=220):
    """Smooth optimization; accept only genuine unsmoothed objective gains."""
    start = np.asarray(initial, float)
    if start.shape != (X.shape[1],) or not np.isfinite(start).all():
        raise ValueError("Invalid initial policy")
    before = fixed_exact(start, X, booked, actual, alpha, h, lam, mat, offsets)
    x = start.copy()
    passes = []
    for eps in (5., 1.5):
        result = minimize(fixed_smooth, x,
                          args=(X, booked, actual, alpha, h, lam, mat, offsets, eps),
                          jac=True, method="L-BFGS-B",
                          bounds=[(-150., 150.)] * len(start),
                          options={"maxiter": int(iterations),
                                   "ftol": 1e-10, "maxls": 35})
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError("Policy optimizer returned nonfinite values")
        candidate = np.asarray(result.x, float)
        exact = fixed_exact(candidate, X, booked, actual, alpha, h, lam, mat, offsets)
        passes.append({"smoothing": eps, "iterations": int(result.nit),
                       "smooth_objective": float(result.fun),
                       "exact_objective": float(exact),
                       "optimizer_success": bool(result.success),
                       "message": str(result.message)})
        if exact < before - 1.e-7:
            before, x = exact, candidate
    return {"w": x.tolist(), "objective": float(before),
            "passes": passes, "accepted": bool(np.max(np.abs(x-start)) > 1e-9)}


def fit_case(X, booked, actual, alpha, h, lam, lasso, iterations):
    starts = [np.zeros(X.shape[1]), np.asarray(lasso, float)]
    candidates = [fit_fixed(X, booked, actual, alpha, h, lam,
                            start, iterations=iterations) for start in starts]
    best = min(candidates, key=lambda p: p["objective"])
    return {**best, "method": "CASE", "objective_definition":
            "asymmetric post-response error per case + lambda*L1",
            "multistart_objectives": [a["objective"] for a in candidates]}


def planning_task(task):
    """Solve one group-week as independent physical-room components."""
    week, duration, fingerprint, seconds, threads = task
    assignments, complete, cost, reason = [], True, 0., ""
    for component in week.components():
        lookup = dict(zip(map(int, week.case_ids), duration))
        d = np.array([lookup[int(cid)] for cid in component.case_ids], float)
        res = solve_week(component, d, seconds=seconds, threads=threads, tie=False)
        if not res.complete or res.assignment is None:
            complete = False
            reason = str(res.status)
            break
        actual_cost = evaluate(component, res.assignment, d)["cost"]
        if abs(actual_cost - res.value) > .1:
            raise AssertionError("Recomputed certified primary plan disagrees")
        cost += actual_cost
        assignments.extend(
            (int(cid), int(component.slots[int(j)][0]),
             str(component.slots[int(j)][1]))
            for cid, j in zip(component.case_ids, res.assignment)
        )
    return {"key": week.group + "|" + week.monday, "sha256": fingerprint,
            "complete": bool(complete), "assignment": assignments if complete else [],
            "predicted_cost": float(cost) if complete else None, "status": reason}


def solve_training_plans(root, weeks, X, booked, index, w, outer,
                         alpha, h, seconds, threads, workers):
    path = Path(root) / f"vf_plans_outer_{outer:02d}.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    _, reviewed = display_and_duration(X @ np.asarray(w, float), booked, alpha, h)
    pending = []
    for week in weeks:
        key = week.group + "|" + week.monday
        locations = np.asarray([index[int(cid)] for cid in week.case_ids], int)
        values = np.asarray(reviewed[locations], float)
        fingerprint = checksum_array(values)
        if (key in state and state[key].get("complete")
                and state[key].get("sha256") == fingerprint):
            continue
        pending.append((week, values, fingerprint, seconds, threads))
    print(f"VF outer {outer}: {len(pending)} group-weeks need certified plans", flush=True)

    def remember(item):
        state[item["key"]] = item
        save_json(path, state)

    if workers == 1:
        for task in pending:
            remember(planning_task(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(planning_task, task) for task in pending]
            for future in as_completed(futures):
                remember(future.result())
    incomplete = [w.group + "|" + w.monday for w in weeks if not
                  state.get(w.group + "|" + w.monday, {}).get("complete")]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} uncertified training plans (e.g., {incomplete[:3]}); "
            "rerun same command with higher --seconds")
    return state


def fit_vf(root, weeks, X, booked, actual, index, alpha, h, lam,
           start, settings, max_outer, iterations, seconds, threads, workers):
    """Monotone majorization of the certified weekly VF upper bound.

    For each outer iteration, exact predicted scheduling returns a minimizer.
    The fixed-assignment upper bound touches the VF objective at the current
    iterate. Accepting a strict improvement in that upper bound cannot
    increase the unsmoothed *true* schedule-aware VF objective.
    """
    root = Path(root)
    path = root / "vf_progress.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "settings": settings, "w": list(start), "iterations": []}
    if state.get("settings") != settings:
        raise ValueError("VF training settings changed; use a new run directory")
    if state.get("complete"):
        return state
    stopping = "outer_iteration_cap"
    for outer in range(len(state["iterations"]), max_outer):
        current = np.asarray(state["w"], float)
        plans = solve_training_plans(root, weeks, X, booked, index, current, outer,
                                     alpha, h, seconds, threads, workers)
        S, offsets = fixed_room_matrix(weeks, plans, index, len(booked))
        before = fixed_exact(current, X, booked, actual, alpha, h, lam, S, offsets)
        updated = fit_fixed(X, booked, actual, alpha, h, lam, current,
                            S, offsets, iterations=iterations)
        candidate = np.asarray(updated["w"], float)
        after = fixed_exact(candidate, X, booked, actual, alpha, h, lam, S, offsets)
        accepted = after + 1.e-7 < before
        if accepted:
            state["w"] = candidate.tolist()
        state["iterations"].append({
            "outer": outer + 1, "touching_vf_bound_minus_oracle": float(before),
            "candidate_fixed_bound_minus_oracle": float(after),
            "accepted": bool(accepted), "relative_improvement":
            max(0., (before - after) / max(1., abs(before))),
            "certified_group_weeks": len(weeks), "fit_passes": updated["passes"]
        })
        save_json(path, state)
        print(f"VF outer {outer + 1}: before={before:.5f} "
              f"after={after:.5f} accepted={accepted}", flush=True)
        if not accepted or (before-after) / max(1., abs(before)) < 1e-4:
            stopping = "no_accepted_gain" if not accepted else "relative_improvement"
            break
    state["complete"] = True
    state["stopping"] = stopping
    state["method"] = "VF"
    state["objective_definition"] = (
        "sum asymmetric post-response error + min predicted OR cost, "
        "per case + lambda*L1; realized oracle term omitted as constant")
    save_json(path, state)
    return state
