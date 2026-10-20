"""Scenario-based behavioral decision-aware search. No evaluation outcomes are used here."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from surgery.anchored_core import key
from surgery.anchored_training import verify_policy
from surgery.anchored_weekly_solver import solve_week, hash_score
from surgery.planner import display_and_duration, reachable
from surgery.weekly_planner import evaluate, _objective_tolerance


LAMBDA = 0.001
K = 50
ROUNDS = 10


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def components(weeks, index):
    return [(key(c), c, np.asarray([index[int(cid)] for cid in c.case_ids], int))
            for week in weeks for c in week.components()]


def mean_cost(c, assignment, scenarios):
    """Vectorized exact idle/overtime cost, including idle in unused roster slots."""
    z = np.asarray(assignment, int)
    counts = np.bincount(z, minlength=c.k)
    loads = np.zeros((len(scenarios), c.k))
    for j in range(c.k):
        if counts[j]:
            loads[:, j] = scenarios[:, z == j].sum(axis=1) + 30 * (counts[j] - 1)
    return float((np.maximum(480 - loads, 0) +
                  1.75 * np.maximum(loads - 480, 0)).mean(axis=0).sum())


def scenario_bank(items, data, case_w, alpha, h, seed):
    """Nonchronological offline surgeon-week block bootstrap; target Monday excluded.

    CASE may be fitted in-sample; no target week's realized residual is a donor
    for that week's scenarios. One donor week is chosen per surgeon/scenario.
    """
    _, predicted = display_and_duration(data["X"] @ case_w, data["booked"], alpha, h)
    errors = np.log(data["actual"] / predicted)
    history = {}
    for _, c, rows in items:
        for i, s in enumerate(c.surgeon):
            history.setdefault(str(c.surgeon_ids[int(s)]), {}).setdefault(c.monday, []).append(
                float(errors[rows[i]]))
    all_history = {}
    for surgeon, periods in history.items():
        for monday, values in periods.items():
            all_history.setdefault(monday, []).extend(values)
    bank = {}
    for ck, c, rows in items:
        rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(f"{seed}:{ck}".encode()).digest()[:8], "little"))
        scenarios = np.empty((K, c.n))
        for s, surgeon in enumerate(c.surgeon_ids):
            positions = np.flatnonzero(c.surgeon == s)
            pool = [(week, vals) for week, vals in history.get(str(surgeon), {}).items()
                    if week != c.monday]
            if not pool:
                pool = [(week, vals) for week, vals in all_history.items()
                        if week != c.monday]
            if not pool:
                raise ValueError("No other calendar week to draw duration residuals")
            for k in range(K):
                vals = pool[int(rng.integers(len(pool)))][1]
                e = rng.choice(vals, size=len(positions), replace=True)
                scenarios[k, positions] = np.clip(predicted[rows[positions]] * np.exp(e), 1., 1440.)
        bank[ck] = scenarios
    return bank


def plan(c, duration, seconds):
    result = solve_week(c, duration, seconds=seconds, threads=1, tie=True, hash_seed=0)
    if not result.complete or result.assignment is None:
        raise RuntimeError(f"Uncertified deployed plan {key(c)}: {result.status}")
    return tuple(map(int, result.assignment))


def saa(c, scenarios, seconds):
    """SAA is only a feasible library aspiration, never the deployed planner."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.anchored_weekly_solver import _model
    from surgery.weekly_planner import _cost, _assignment
    m, x, used, byroom = _model(c, seconds, 1)
    try:
        objective = gp.quicksum(
            _cost(m, x, used, byroom, scenarios[k], exact=False, name=f"saa{k}")
            for k in range(len(scenarios))) / len(scenarios)
        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()
        return tuple(map(int, _assignment(c, x))) if m.Status == GRB.OPTIMAL else None
    finally:
        m.dispose()


def reachable_search(c, initial, scenarios, seconds, passes=2):
    """Constructive free-case improvement on A; no claim of optimal inducibility."""
    lo, hi = reachable(c.booked, 0.8, 30.)
    d = np.clip(np.asarray(initial, float), lo, hi)
    z = plan(c, d, seconds)
    best = mean_cost(c, z, scenarios)
    found = {z}
    for _ in range(passes):
        changed = False
        for i in range(c.n):
            choice = None
            for value in (float(lo[i]), float(hi[i])):
                if abs(d[i] - value) < 1e-8:
                    continue
                trial = d.copy()
                trial[i] = value
                zz = plan(c, trial, seconds)
                found.add(zz)
                value_cost = mean_cost(c, zz, scenarios)
                if value_cost < best - 1e-8:
                    best = value_cost
                    choice = (trial, zz)
            if choice is not None:
                d, z = choice
                changed = True
        if not changed:
            break
    return d, found


def initialize(items, data, wcase, bank, seconds, checkpoint, fingerprint,
               *, saa_limit=256, reach_limit=120):
    """Resumable library construction; no training component is dropped."""
    checkpoint = Path(checkpoint)
    init = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(
        fingerprint=fingerprint, library={}, desired={},
        saa_attempted=0, saa_optimal=0, reachable_attempted=0,
        extreme_attempted=0, extreme_certified=0)
    if init["fingerprint"] != fingerprint:
        raise ValueError("Library input/code changed; use a new directory")
    _, post = display_and_duration(data["X"] @ wcase, data["booked"], .8, 30.)
    flex = [(ck, c, rows) for ck, c, rows in items
            if any(len(c.allowed[int(s)]) > 1 for s in c.surgeon)]
    rng = np.random.default_rng(11219)
    eligible = [v for v in flex if 2 <= v[1].n <= 20]
    rng.shuffle(eligible)
    target = {ck for ck, _, _ in eligible[:reach_limit]}
    aspir = {ck for ck, _, _ in eligible[:saa_limit]}
    for j, (ck, c, rows) in enumerate(items):
        if ck in init["library"]:
            continue
        d = post[rows]
        assignments = {plan(c, d, seconds)}
        # Give every flexible component at least two additional attempts to
        # generate a genuine alternative, not only sampled SAA components.
        if any(len(c.allowed[int(s)]) > 1 for s in c.surgeon):
            low, high = reachable(c.booked, .8, 30.)
            for extreme in (low, high):
                init["extreme_attempted"] += 1
                try:
                    assignments.add(plan(c, extreme, min(seconds, 90.)))
                    init["extreme_certified"] += 1
                except RuntimeError:
                    # Optional aspiration only. No case disappears from the
                    # later full-planner, all-component training objective.
                    pass
        if ck in aspir:
            zz = saa(c, bank[ck], min(seconds, 180.))
            init["saa_attempted"] += 1
            if zz is not None:
                assignments.add(zz)
                init["saa_optimal"] += 1
        if ck in target:
            free, found = reachable_search(c, d, bank[ck], min(seconds, 120.))
            init["desired"][ck] = (free - d).tolist()
            init["reachable_attempted"] += 1
            assignments.update(found)
        init["library"][ck] = [list(v) for v in sorted(assignments)]
        if (j + 1) % 40 == 0:
            write(checkpoint, init)
            print(f"ASDBS library {j+1}/{len(items)}", flush=True)
    write(checkpoint, init)
    desired = np.zeros(len(post))
    for ck, _, rows in items:
        if ck in init["desired"]:
            desired[rows] = init["desired"][ck]
    return init["library"], desired, {
        name:init[name] for name in ("saa_attempted","saa_optimal",
                                  "reachable_attempted","extreme_attempted",
                                  "extreme_certified")}



def selected(c, assignments, duration):
    """Hash-faithful choice inside a finite schedule library (proposal only)."""
    costs = np.asarray([evaluate(c, a, duration)["cost"] for a in assignments])
    optimum = float(costs.min())
    tol = _objective_tolerance(optimum)
    eligible = np.flatnonzero(costs <= optimum + tol + 1e-9)
    return int(min(eligible, key=lambda j: (
        hash_score(c, assignments[j], 0),
        hash_score(c, assignments[j], 1),
        tuple(assignments[j]))))


def library_loss(items, data, bank, library, w, wcase):
    _, d = display_and_duration(data["X"] @ w, data["booked"], .8, 30.)
    total = 0.
    for ck, c, rows in items:
        choices = library[ck]
        j = 0 if len(choices) == 1 else selected(c, choices, d[rows])
        total += data["_costs"][ck][j]
    return total / len(d) + LAMBDA * float(np.mean((data["X"] @ (w - wcase))**2)) / 2


def directions(X, names, desired, round_id):
    """Full coordinates are cycled in ten blocks, not discarded."""
    p = X.shape[1]
    result = []
    for j in range(p):
        if j % ROUNDS == round_id:
            v = np.zeros(p)
            v[j] = 1.
            result.append((f"feature:{names[j]}", v))
    for token in ("spread", "abs_error", "surgeon_week", "site:", "interaction"):
        cols = [j for j, name in enumerate(names) if token in name]
        if cols:
            v = np.zeros(p)
            v[cols] = 1.
            result.append((f"block:{token}", v))
    if np.any(desired):
        v = np.linalg.lstsq(X, desired, rcond=1e-8)[0]
        result.append(("projected_reachable_changes", v))
    normalized = []
    for label, v in result:
        scale = float(np.linalg.norm(X @ v) / np.sqrt(len(X)))
        if scale > 1e-10:
            normalized.append((label, v / scale))
    return normalized


def boundary_candidates(items, data, library, w, v, *, max_points=16):
    """Heuristic detection of library decision boundaries; not exhaustive."""
    X = data["X"]
    score = X @ w
    move = X @ v
    candidate = {float(k) for k in (-32, -24, -16, -8, -4, 0, 4, 8, 16, 24, 32)}
    valuable = sorted((entry for entry in items if len(library[entry[0]]) > 1),
                      key=lambda row: (-row[1].n, row[0]))[:80]
    knots = np.linspace(-32., 32., 17)
    proposals = []
    for ck, c, rows in valuable:
        choices = library[ck]
        if len(choices) < 2:
            continue
        predicted = []
        for t in knots:
            lower, upper = reachable(c.booked, .8, 30.)
            d = c.booked + np.clip(score[rows] + t * move[rows],
                                   lower - c.booked, upper - c.booked)
            predicted.append([evaluate(c, z, d)["cost"] for z in choices])
        predicted = np.asarray(predicted)
        for a in range(len(choices)):
            for b in range(a + 1, len(choices)):
                gaps = predicted[:, a] - predicted[:, b]
                for shift in (0., .05, -.05):
                    y = gaps - shift
                    for j in np.flatnonzero(y[:-1] * y[1:] < 0):
                        eta = knots[j] - y[j] * (knots[j+1] - knots[j]) / (y[j+1] - y[j])
                        advantage = abs(data["_costs"][ck][a] -
                                        data["_costs"][ck][b])
                        proposals.append((advantage, float(eta)))
    proposals.sort(reverse=True)
    for _, eta in proposals[:max_points]:
        for offset in (-.01, .01):
            if -32 <= eta + offset <= 32:
                candidate.add(eta + offset)
    return sorted(candidate)


def verify(output, name, weeks, index, data, w, seconds, workers):
    return verify_policy(output, name, weeks, index, data["X"], data["booked"],
                         w, .8, 30., seconds=seconds, threads=1,
                         workers=workers)


def bank_difference(items, bank, incumbent, candidate, reg_difference, N):
    """One vector entry per independent *joint scenario replication*."""
    diff = np.zeros(K)
    for ck, c, _ in items:
        zold = np.asarray(incumbent[ck]["assignment"], int)
        znew = np.asarray(candidate[ck]["assignment"], int)
        if np.array_equal(zold, znew):
            continue
        a = bank[ck]
        for k in range(K):
            diff[k] += (evaluate(c, zold, a[k])["cost"] -
                        evaluate(c, znew, a[k])["cost"]) / N
    return diff - reg_difference


def train(folder, weeks, index, data, names, wcase, *, seconds=600., workers=12):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items = components(weeks, index)
    N = len(data["booked"])
    wcase = np.asarray(wcase, float)
    if data["X"].shape[1] != len(wcase) or len(items) == 0:
        raise ValueError("Feature or cohort mismatch")
    fingerprint = hashlib.sha256(
        data["X"].tobytes() + data["actual"].tobytes() +
        data["booked"].tobytes() + data["case_id"].tobytes() +
        wcase.tobytes() + Path(__file__).read_bytes()).hexdigest()
    path = folder / "asdbs_progress.json"
    state = json.loads(path.read_text()) if path.exists() else None
    if state is not None and state["fingerprint"] != fingerprint:
        raise ValueError("ASDBS input/code changed; use a new artifact directory")
    bankA = scenario_bank(items, data, wcase, .8, 30., 4101)
    if state is None:
        library, desired, initialization = initialize(
            items, data, wcase, bankA, seconds,
            folder / "asdbs_library_initialization.json", fingerprint)
        state = dict(fingerprint=fingerprint, policy=wcase.tolist(),
                     case_policy=wcase.tolist(), desired=desired.tolist(),
                     library=library, initialization=initialization,
                     rounds=[], complete=False)
        write(path, state)
    if state["complete"]:
        return state
    costs = {ck:[mean_cost(c, z, bankA[ck]) for z in state["library"][ck]]
             for ck, c, _ in items}
    data = dict(data, _bank=bankA, _costs=costs)
    for r in range(len(state["rounds"]), ROUNDS):
        w = np.asarray(state["policy"], float)
        before = library_loss(items, data, bankA, state["library"], w, wcase)
        best = (before, w, "none")
        for name, v in directions(data["X"], names, np.asarray(state["desired"]), r):
            for eta in boundary_candidates(items, data, state["library"], w, v):
                if eta == 0:
                    continue
                proposal = w + eta * v
                value = library_loss(items, data, bankA, state["library"], proposal, wcase)
                if value < best[0] - 1e-9:
                    best = (value, proposal, name)
        if best[2] == "none":
            state["rounds"].append(dict(round=r+1, accepted=False, reason="no_library_improvement"))
            write(path, state)
            continue
        proposal = best[1]
        # Full planner, not library selection, determines acceptance.
        old = verify(folder / "training_deploy", "incumbent", weeks, index,
                     data, w, seconds, workers)
        new = verify(folder / "training_deploy", "candidate", weeks, index,
                     data, proposal, seconds, workers)
        reg = LAMBDA / 2 * float(np.mean(
            (data["X"] @ (proposal-wcase))**2 - (data["X"] @ (w-wcase))**2))
        bankB = scenario_bank(items, data, wcase, .8, 30., 83000+r)
        delta = bank_difference(items, bankB, old, new, reg, N)
        improvement = float(delta.mean())
        se = float(delta.std(ddof=1) / math.sqrt(K))
        accepted = bool(improvement > 1.645 * se and improvement > 0)
        for ck, c, _ in items:
            for row in (old[ck], new[ck]):
                z = list(map(int, row["assignment"]))
                if z not in state["library"][ck]:
                    state["library"][ck].append(z)
                    data["_costs"][ck].append(mean_cost(c, z, bankA[ck]))
        if accepted:
            state["policy"] = proposal.tolist()
        state["rounds"].append(dict(round=r+1, accepted=accepted,
                                    direction=best[2], library_before=before,
                                    library_candidate=best[0],
                                    bank_b_gain_per_case=improvement,
                                    bank_b_se=se))
        write(path, state)
        print(f"ASDBS round {r+1}: gain={improvement:.5f} se={se:.5f} "
              f"accepted={accepted}", flush=True)
    state["complete"] = True
    write(path, state)
    return state
