"""Behavioral advice comparison: one frozen CASE layer, one exact planner, two policies.

No realized evaluation durations are consulted during recommendation. All schedules
used in scoring are certified by the unchanged hash-tie planner.
"""
import hashlib
import os
import time
import numpy as np

from surgery.anchored_weekly_solver import solve_week
from surgery.planner import display_and_duration, reachable
from surgery.weekly_planner import evaluate, loads
from surgery.asdbs import mean_cost

ALPHA, H, LAM = .8, 30., .001


def reviewed(booked, raw):
    lo, hi = reachable(booked, ALPHA, H)
    return np.clip(np.asarray(booked, float) + raw, lo, hi)


def display(booked, d):
    """Inverse response: every reviewed target is attained with |display| <= 30."""
    lo, hi = reachable(booked, ALPHA, H)
    d = np.asarray(d, float)
    if np.any(d < lo - 1e-8) or np.any(d > hi + 1e-8):
        raise ValueError("Unreachable reviewed duration")
    return np.clip((d - booked) / ALPHA, np.maximum(-H, 1. - booked), H)


def certified(c, d, seed, seconds):
    threads = int(os.environ.get("SURGERY_SOLVER_THREADS", "1"))
    answer = solve_week(c, d, seconds=max(.01, seconds), threads=threads,
                        tie=True, hash_seed=int(seed))
    if not answer.complete or answer.assignment is None:
        raise RuntimeError(f"Uncertified {c.group}/{c.monday} seed={seed}: {answer.status}")
    return tuple(map(int, answer.assignment))


def case_features(c, baseline_z, baseline_d, row_X, names):
    """Five planning fields, fixed from CASE before decision-aware optimization."""
    s = row_X[:, names.index("hist_surgeon_mean_abs_error")]
    q = row_X[:, names.index("hist_procedure_mean_abs_error")]
    zs = baseline_z.values() if isinstance(baseline_z, dict) else [baseline_z]
    p = np.mean([np.clip((loads(c, z, baseline_d)[np.asarray(z, int)]
                          - 480.) / 120., -2., 2.) for z in zs], axis=0)
    return np.column_stack((s, q, p, s*p, q*p))


def residual_history(items, data, frame, residuals=None):
    """Residual donors from fitting weeks only; target week excluded during fitting."""
    _, d0 = display_and_duration(data["X"] @ data["case_w"],
                                 data["booked"], ALPHA, H)
    e = (np.log(np.maximum(1., data["actual"]) / d0) if residuals is None
         else np.asarray(residuals, float))
    if e.shape != d0.shape or not np.isfinite(e).all():
        raise ValueError("Invalid residual vector")
    by_id = dict(zip(frame.case_id.astype(int), frame.surgeon.astype(str)))
    pools = {}
    for _, c, rows in items:
        for k, i in enumerate(rows):
            s = by_id[int(c.case_ids[k])]
            pools.setdefault(s, []).append((c.monday, np.log(d0[i]), e[i]))
    return {s: np.asarray(v, dtype=object) for s, v in pools.items()}


def scenarios(c, d0, pools, seed, count=50):
    """Casewise 30 nearest same-surgeon historical log residual donors."""
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(f"{seed}:{c.group}:{c.monday}:{min(c.case_ids)}".encode()).digest()[:8], "little"))
    pool_all = np.concatenate(list(pools.values()))
    result = np.empty((count, c.n))
    for i, surgeon_index in enumerate(c.surgeon):
        s = str(c.surgeon_ids[int(surgeon_index)])
        pool = pools.get(s, pool_all)
        pool = pool[np.asarray(pool[:, 0] != c.monday)]
        if len(pool) == 0:
            pool = pool_all[np.asarray(pool_all[:, 0] != c.monday)]
        if len(pool) == 0:
            raise ValueError("No historical residual donors")
        closest = np.argsort(abs(pool[:, 1].astype(float) - np.log(d0[i])))[:30]
        residual = rng.choice(pool[closest, 2].astype(float), size=count)
        result[:, i] = np.maximum(1., d0[i]*np.exp(residual))
    return result


def cost(c, z, bank):
    return mean_cost(c, z, bank)


def score(c, durations, banks, seeds, seconds, cached=None):
    plans = (cached if cached is not None
             else {r: certified(c, durations, r, seconds) for r in seeds})
    return plans, {b: float(np.mean([cost(c, plans[r], draws) for r in seeds]))
                   for b, draws in banks.items()}


def objective(c, d, d0, plans, scenarios_A, seeds):
    return (np.mean([cost(c, plans[r], scenarios_A) for r in seeds])
            + LAM/2 * np.sum((d - d0)**2))


def transition(c, d, current, target):
    """Minimum-length local step making the target preferred; verify after stepping."""
    from surgery.weekly_planner import _objective_tolerance
    jc = evaluate(c, current, d)["cost"]
    jt = evaluate(c, target, d)["cost"]
    load_current = loads(c, current, d)
    load_target = loads(c, target, d)
    g = 2.75*((load_current[np.asarray(current)] > 480.).astype(float)
              - (load_target[np.asarray(target)] > 480.).astype(float))
    lo, hi = reachable(c.booked, ALPHA, H)
    if np.dot(g, g) > 0:
        step = max(0., jt - jc + 2*_objective_tolerance(jc)) * g / np.dot(g, g)
        return np.clip(d + step, lo, hi)
    # Cross the strict overtime threshold rather than land exactly on it.
    # Non-improving proposals may act as intermediate probes.
    for z, load in ((current, load_current), (target, load_target)):
        for room in np.argsort(abs(load - 480.)):
            ids = np.flatnonzero(np.asarray(z) == room)
            if not len(ids):
                continue
            destination = 481. if load[room] <= 480. else 479.
            shifted = np.clip(d[ids] + (destination - load[room]) / len(ids),
                              lo[ids], hi[ids])
            proposal = d.copy()
            proposal[ids] = shifted
            if np.linalg.norm(proposal - d) > 1e-8:
                return proposal
    return d


def prescription(c, d0, banks, *, seeds=(0,), seconds=120., budget=900., passes=3,
                 reference=False):
    """Coordinate search up to 20 cases; transition probes above 20 cases.

    reference=True reproduces the original coordinate-only restricted arm.
    """
    deadline = time.monotonic() + budget
    baseline, bc = score(c, d0, banks, seeds, max(seconds, 600.))
    if all(len(c.allowed[int(s)]) == 1 for s in c.surgeon):
        return d0, baseline, False, "FIXED", bc
    d = d0.copy()
    plans = baseline
    best = objective(c, d, d0, plans, banks["A"], seeds)
    library = set(baseline.values())
    cache = {tuple(np.round(d0, 6)): baseline}
    lo, hi = reachable(c.booked, ALPHA, H)

    def consider(candidate):
        nonlocal d, plans, best
        if np.allclose(candidate, d):
            return False
        available = min(seconds, deadline-time.monotonic())
        if available < .1:
            return False
        fingerprint = tuple(np.round(candidate, 6))
        if fingerprint not in cache:
            try:
                cache[fingerprint] = {r: certified(c, candidate, r, available) for r in seeds}
            except RuntimeError:
                return False
        found = cache[fingerprint]
        library.update(found.values())
        value = objective(c, candidate, d0, found, banks["A"], seeds)
        if value < best - 1e-8:
            d, plans, best = candidate, found, value
            return True
        return False

    # For large components avoid n*9*3 expensive exact solves.
    if c.n <= 20:
        for _ in range(passes):
            changed = False
            for i in range(c.n):
                for v in np.linspace(lo[i], hi[i], 9):
                    if time.monotonic() >= deadline:
                        break
                    candidate = d.copy()
                    candidate[i] = v
                    changed |= consider(candidate)
                if time.monotonic() >= deadline:
                    break
            if not changed or time.monotonic() >= deadline:
                break
    else:
        for boundary in (lo, hi):
            consider(boundary.copy())
        # Build useful alternative schedules with small case perturbations.
        for i in np.argsort(-(hi-lo))[:min(c.n, 20)]:
            for v in (lo[i], hi[i]):
                trial = d.copy()
                trial[i] = v
                consider(trial)

    if c.n > 20 and deadline-time.monotonic() > 1:
        from surgery.asdbs import saa
        aspiration = saa(c, banks["A"], min(seconds, deadline-time.monotonic()))
        if aspiration is not None:
            library.add(aspiration)

    # Up to three certified boundary probes in a chain. Only a bank-A
    # improvement commits a proposal; non-improving probes remain temporary.
    if c.n > 20 and not reference:
        for target in sorted(library, key=lambda z: cost(c, z, banks["A"]))[:8]:
            probe, probe_plans = d.copy(), plans
            for _ in range(3):
                if time.monotonic() >= deadline:
                    break
                trial = transition(c, probe, probe_plans[seeds[0]], target)
                if np.allclose(trial, probe):
                    break
                accepted = consider(trial)
                if accepted:
                    probe, probe_plans = d.copy(), plans
                else:
                    found = cache.get(tuple(np.round(trial, 6)))
                    if found is None:
                        break
                    probe, probe_plans = trial, found

    candidate_b = objective(c, d, d0, plans, banks["B"], seeds)
    baseline_b = objective(c, d0, d0, baseline, banks["B"], seeds)
    if candidate_b >= baseline_b - 1e-8:
        return d0, baseline, False, "REJECTED_B", bc
    return d, plans, True, "ACCEPTED_B", bc


def component_job(job):
    """One Gurobi thread per process; evaluation durations are used only at the end."""
    ck, c, rows, data, H, pools, theta, beta, seeds, seconds, budget = job
    booked = c.booked
    d0 = reviewed(booked, data["X"][rows] @ data["case_w"])
    banks = {a: scenarios(c, d0, pools, 20261020 + j, 200 if a == "C" else 50)
             for j, a in enumerate(("A", "B", "C"))}
    base = {r: certified(c, d0, r, max(seconds, 600.)) for r in seeds}
    outputs = {"CASE": (d0, base)}
    statuses = {"CASE": "REFERENCE"}
    booked_z = {r: certified(c, booked, r, max(seconds, 600.)) for r in seeds}
    outputs["BOOKED"] = (booked, booked_z)
    statuses["BOOKED"] = "REFERENCE"
    risk = reviewed(booked, np.quantile(banks["A"], 1.75/2.75, axis=0)-booked)
    try:
        outputs["RISK"] = (risk, {r: certified(c, risk, r, seconds) for r in seeds})
        statuses["RISK"] = "CERTIFIED"
    except RuntimeError:
        outputs["RISK"] = (d0, base)
        statuses["RISK"] = "UNCERTIFIED_FALLBACK"
    for name, coeff in (("STAT", beta), ("SHARED", beta+theta)):
        d = reviewed(booked, d0 - booked + H[rows] @ coeff)
        try:
            outputs[name] = (d, {r: certified(c, d, r, seconds) for r in seeds})
            statuses[name] = "CERTIFIED"
        except RuntimeError:
            outputs[name] = (d0, base)
            statuses[name] = "UNCERTIFIED_FALLBACK"
    for name, fitted_seeds in (("PER_WEEK_REF", (0,)),
                               ("PER_WEEK", (0,)), ("PER_WEEK_ROBUST", seeds)):
        if name == "PER_WEEK_REF" and c.n > 20:
            outputs[name] = (d0, base)
            statuses[name] = "CASE_ABOVE_20"
            continue
        if name == "PER_WEEK" and c.n <= 20:
            outputs[name] = outputs["PER_WEEK_REF"]
            statuses[name] = statuses["PER_WEEK_REF"]
            continue
        try:
            d, z, accepted, status, _ = prescription(
                c, d0, banks, seeds=fitted_seeds, seconds=seconds,
                budget=budget if c.n <= 20 else 2*budget,
                reference=(name == "PER_WEEK_REF"))
            outputs[name] = (d, {r: (z[r] if r in z else certified(c, d, r, max(seconds, 600.)))
                                 for r in seeds})
            statuses[name] = status
        except RuntimeError as exc:
            outputs[name] = (d0, base)
            statuses[name] = "UNCERTIFIED_FALLBACK: " + str(exc)
    report = {}
    for name, (d, z) in outputs.items():
        metrics = {str(r): evaluate(c, z[r], c.actual) for r in seeds}
        report[name] = dict(
            status=statuses[name],
            realized={r: v["cost"] for r, v in metrics.items()},
            overtime={r: v["overtime"] for r, v in metrics.items()},
            idle={r: v["idle"] for r, v in metrics.items()},
            over720={r: v["room_days_over_720"] for r, v in metrics.items()},
            model={bank: np.mean([cost(c, z[r], draws) for r in seeds])
                   for bank, draws in banks.items()},
            changed=int(np.sum(abs(d-d0) > 1e-7)),
            mean_abs_adjustment=float(np.mean(abs(d-d0))),
            assignments={str(r): list(z[r]) for r in seeds},
            displayed=display(booked, d).tolist(),
        )
    return ck, dict(site=c.group, monday=c.monday, cases=c.n, arms=report)

# Worker state is installed ONCE per process rather than pickling matrices and
# residual pools thousands of times for independent Gurobi component solves.
_STATE = {}


def init_worker(state):
    global _STATE
    _STATE = state
    os.environ["SURGERY_SOLVER_THREADS"] = str(state.get("threads", 1))
    from threadpoolctl import threadpool_limits
    _STATE["_thread_limit"] = threadpool_limits(1)


def baseline_job(item):
    ck, c, rows = item
    state = _STATE
    data, names = state["data"], state["names"]
    d0 = reviewed(c.booked, data["X"][rows] @ data["case_w"])
    try:
        z = {r: certified(c, d0, r, max(state["seconds"], 600.)) for r in (0, 1, 2)}
    except RuntimeError as exc:
        raise RuntimeError(f"CASE feature solve failed for {ck}: {exc}") from exc
    return ck, rows, case_features(c, z, d0, data["X"][rows], names)


def shared_job(item):
    """Score candidates on the same training components and scenario banks."""
    ck, c, rows = item
    state = _STATE
    data, H, pools = state["data"], state["H"], state["pools"]
    d0 = reviewed(c.booked, data["X"][rows] @ data["case_w"])
    draws = {name: scenarios(c, d0, pools, 20261020+97*state["round_id"]+j)
             for j, name in enumerate(("A", "B"))}
    out, cache = [], {}
    for coeff in state["candidates"]:
        d = reviewed(c.booked, d0-c.booked + H[rows] @ coeff)
        fingerprint = tuple(np.round(d, 6))
        try:
            if fingerprint not in cache:
                cap = max(state["seconds"], 600.) if len(out) == 0 else state["seconds"]
                cache[fingerprint] = [certified(c, d, r, cap)
                                       for r in state["seeds"]]
            z = cache[fingerprint]
        except RuntimeError:
            # An uncertified proposed policy cannot be ranked as an improvement.
            out.append(None)
            continue
        penalty = LAM/2 * np.sum((d - d0)**2)
        out.append([float(np.mean([cost(c, a, draws[bank])
                                   for a in z])) + penalty
                    for bank in ("A", "B")])
    return ck, c.monday, c.n, out


def evaluated_job(item):
    ck, c, rows = item
    s = _STATE
    return component_job((ck, c, rows, s["data"], s["H"], s["pools"],
                          s["theta"], s["beta"], s["seeds"],
                          s["seconds"], s["budget"]))


def sensitivity_job(item):
    """Replan frozen displays under alternative response maps or an observed cutoff.

    Uses no evaluation actual durations before the certified deployment.
    """
    import json
    from pathlib import Path
    from surgery.planner import response

    ck, c, _ = item
    state = _STATE
    original = json.loads((Path(state["components_dir"])/(ck+".json")).read_text())
    seeds = state["seeds"]
    outputs = {}
    proposed = {name: np.asarray(rec["displayed"], float)
                for name, rec in original["arms"].items()}
    known = np.asarray([
        state["decision_dates"].get(int(cid)) is not None
        and state["decision_dates"][int(cid)] < c.monday
        for cid in c.case_ids], bool)
    unknown = sum(state["decision_dates"].get(int(cid)) is None for cid in c.case_ids)
    configurations = {
        f"alpha_{a:g}_h_{h:g}": (a, h, None)
        for a, h in state["response_sweep"]
    }
    configurations["cutoff_week_start"] = (.8, 30., known)
    for label, (alpha, h, mask) in configurations.items():
        result = {}
        case_display = proposed["CASE"]
        case_u = case_display if mask is None else np.where(mask, case_display, case_display)
        case_d = c.booked + response(case_u, alpha, h)
        if alpha == .8 and h == 30. and mask is None:
            case_z = {seed: tuple(original["arms"]["CASE"]["assignments"][str(seed)])
                      for seed in seeds}
        else:
            case_z = {seed: certified(c, case_d, seed, max(600., state["seconds"]))
                      for seed in seeds}
        for name, u in proposed.items():
            mixed_u = u if mask is None else np.where(mask, u, case_display)
            d = c.booked + response(mixed_u, alpha, h)
            if np.array_equal(mixed_u, case_u):
                z, status = case_z, "CASE_REUSE"
            elif alpha == .8 and h == 30. and mask is None:
                z = {seed: tuple(original["arms"][name]["assignments"][str(seed)])
                     for seed in seeds}
                status = "FROZEN_REUSE"
            else:
                try:
                    z = {seed: certified(c, d, seed, state["seconds"]) for seed in seeds}
                    status = "CERTIFIED"
                except RuntimeError as exc:
                    z, status = case_z, "UNCERTIFIED_FALLBACK: " + str(exc)
            # An uncertified optional arm must be scored as its *deployed*
            # CASE fallback, not with the unverified recommendation.
            final_d = case_d if status.startswith("UNCERTIFIED") else d
            result[name] = dict(
                status=status, changed=int(np.sum(abs(final_d-case_d) > 1e-7)),
                realized={str(seed): evaluate(c, z[seed], c.actual)["cost"]
                          for seed in seeds})
        outputs[label] = dict(site=c.group, monday=c.monday, cases=c.n,
                              arms=result, eligible_at_cutoff=int(known.sum()),
                              unknown_decision_date=int(unknown))
    return ck, outputs
