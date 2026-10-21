"""Behavioral advice comparison: one frozen CASE layer, one exact planner, two policies.

No realized evaluation durations are consulted during recommendation. All schedules
used in scoring are certified by the unchanged hash-tie planner.
"""
import hashlib
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
    answer = solve_week(c, d, seconds=max(.01, seconds), threads=1,
                        tie=True, hash_seed=int(seed))
    if not answer.complete or answer.assignment is None:
        raise RuntimeError("Uncertified " + str(answer.status))
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
    # Locally flat: move to a 480-minute regime boundary, then reconsider.
    for z, load in ((current, load_current), (target, load_target)):
        for room in np.argsort(abs(load-480.)):
            ids = np.flatnonzero(np.asarray(z) == room)
            if not len(ids):
                continue
            step = (480. - load[room]) / len(ids)
            shifted = np.clip(d[ids] + step, lo[ids], hi[ids])
            if np.linalg.norm(shifted - d[ids]) > 1e-8:
                proposal = d.copy()
                proposal[ids] = shifted
                return proposal
    return d


def prescription(c, d0, banks, *, seeds=(0,), seconds=120., budget=900., passes=3):
    """Nine-point coordinate and schedule-transition search, bank-B confirmation."""
    deadline = time.monotonic() + budget
    baseline, bc = score(c, d0, banks, seeds, seconds)
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

    # Aim at good schedules, not only single-coordinate duration changes.
    for target in sorted(library, key=lambda z: cost(c, z, banks["A"]))[:8]:
        for _ in range(3):
            if time.monotonic() >= deadline:
                break
            current = plans[seeds[0]]
            trial = transition(c, d, current, target)
            if np.allclose(trial, d):
                break
            if not consider(trial):
                break

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
    banks = {a: scenarios(c, d0, pools, 20261020 + j, 50)
             for j, a in enumerate(("A", "B", "C"))}
    base = {r: certified(c, d0, r, seconds) for r in seeds}
    outputs = {"CASE": (d0, base)}
    statuses = {"CASE": "REFERENCE"}
    booked_z = {r: certified(c, booked, r, seconds) for r in seeds}
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
    for name, fitted_seeds in (("PER_WEEK", (0,)), ("PER_WEEK_ROBUST", seeds)):
        try:
            d, z, accepted, status, _ = prescription(
                c, d0, banks, seeds=fitted_seeds, seconds=seconds, budget=budget)
            outputs[name] = (d, {r: (z[r] if r in z else certified(c, d, r, seconds))
                                 for r in seeds})
            statuses[name] = status
        except RuntimeError:
            outputs[name] = (d0, base)
            accepted, status = False, "UNCERTIFIED_FALLBACK"
            statuses[name] = status
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
    from threadpoolctl import threadpool_limits
    _STATE["_thread_limit"] = threadpool_limits(1)


def baseline_job(item):
    ck, c, rows = item
    state = _STATE
    data, names = state["data"], state["names"]
    d0 = reviewed(c.booked, data["X"][rows] @ data["case_w"])
    z = {r: certified(c, d0, r, state["seconds"]) for r in (0, 1, 2)}
    return ck, rows, case_features(c, z, d0, data["X"][rows], names)


def shared_job(item):
    """Score candidates on the same training components and scenario banks."""
    ck, c, rows = item
    state = _STATE
    data, H, pools = state["data"], state["H"], state["pools"]
    d0 = reviewed(c.booked, data["X"][rows] @ data["case_w"])
    draws = {name: scenarios(c, d0, pools, 20261020+j)
             for j, name in enumerate(("A", "B"))}
    out, cache = [], {}
    for coeff in state["candidates"]:
        d = reviewed(c.booked, d0-c.booked + H[rows] @ coeff)
        fingerprint = tuple(np.round(d, 6))
        try:
            if fingerprint not in cache:
                cache[fingerprint] = [certified(c, d, r, state["seconds"])
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
