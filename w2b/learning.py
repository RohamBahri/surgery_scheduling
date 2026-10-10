"""Three small, honest W2b schedule-library learners.

VF:        min_L J(z,d) + casewise error - oracle.
Gap-VF:    gamma min_L J(z,d) + max_L[J(z,a)-gamma J(z,d)] - oracle.
SPO-style: gamma J(z^a,d) + max_L[J(z,a)-gamma J(z,d)] - oracle.

The gap/SPO restricted-library objectives are proxies, NOT globally valid
certificates. Full loss-augmented separation is attempted after each update.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from w2b.planner import implement, schedule_cost, cached_solve


def theta(error):
    return 1.75 * np.maximum(error, 0) + np.maximum(-error, 0)


def fit_case_error(weeks, *, alpha, h, max_evals=700, bound=100.):
    """A common smoothed response-aware predictive initialization (not a VF fit)."""
    X = np.vstack([week.X for week in weeks])
    b = np.concatenate([week.booked for week in weeks])
    a = np.concatenate([week.actual for week in weeks])
    def f(w):
        e = a - implement(b, X, w, alpha, h)
        return float(np.mean(1.375 * np.hypot(e, 3) + 0.375 * e)
                     + .01 * np.sum(np.abs(w[1:])))
    out = minimize(f, np.zeros(X.shape[1]), method="Powell",
                   bounds=[(-bound, bound)] * X.shape[1],
                   options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-4})
    return np.asarray(out.x, float), {"loss": f(out.x), "evaluations": int(out.nfev)}


@dataclass
class Library:
    week: object
    move_penalty: float
    assignments: list = field(default_factory=list)

    def add(self, assignment):
        if assignment is None:
            return False
        signature = tuple(assignment)
        if signature not in {tuple(x) for x in self.assignments}:
            self.assignments.append(list(assignment))
            return True
        return False

    def costs(self, duration):
        return np.array([schedule_cost(self.week, z, duration, self.move_penalty)
                         for z in self.assignments], float)


def library_objective(method, weeks, libs, w, *, alpha, h, gamma,
                      oracle_value, oracle_assignment, lam):
    total = 0.
    for week, lib, va, za in zip(weeks, libs, oracle_value, oracle_assignment):
        d = implement(week.booked, week.X, w, alpha, h)
        planned = lib.costs(d)
        if not len(planned):
            raise RuntimeError("Empty schedule library")
        if method == "vf":
            loss = (planned.min()
                    + float(theta(week.actual - d).sum()) - va)
        else:
            realized = lib.costs(week.actual)
            adversarial_proxy = float(np.max(realized - gamma * planned))
            if method == "gap":
                loss = gamma * planned.min() + adversarial_proxy - va
            elif method == "spo":
                loss = (gamma * schedule_cost(week, za, d, lib.move_penalty)
                        + adversarial_proxy - va)
            else:
                raise ValueError(method)
        total += loss
    return float(total / len(weeks) + lam * np.sum(np.abs(w[1:])))


def full_policy_check(weeks, libs, weights, *, alpha, h, gamma, method,
                      oracle_value, cache, seconds, adversary_seconds,
                      move_penalty, threads):
    """Enrich library; report exact realized cost and lower/upper surrogate brackets.

    For gap / SPO, an incomplete adversarial solve only bounds the maximum.
    An incomplete ordinary planner makes the policy result uncertified.
    """
    check = {"realized": 0., "regret": 0., "bound_low": 0.,
             "bound_high": 0., "missing": 0, "added": 0,
             "planner_incomplete": 0, "adversary_incomplete": 0,
             "planner_statuses": [], "adversary_statuses": []}
    for week, lib, va in zip(weeks, libs, oracle_value):
        d = implement(week.booked, week.X, weights, alpha, h)
        plan = cached_solve(cache, week, d, seconds=seconds, threads=threads,
                            move_penalty=move_penalty, tie=True)
        check["planner_statuses"].append(plan.status)
        if not plan.optimal:
            check["planner_incomplete"] += 1
        if plan.assignment is None:
            check["missing"] += 1
            continue
        check["added"] += int(lib.add(plan.assignment))
        cost = schedule_cost(week, plan.assignment, week.actual, move_penalty)
        check["realized"] += cost
        check["regret"] += cost - va
        if method == "vf":
            # V(d) in [planner lower bound, observed incumbent]. Exact V is
            # required for a certified value, not for valid upper bound.
            envelope = float(theta(week.actual - d).sum())
            check["bound_low"] += (plan.lower if plan.lower is not None else 0.) + envelope - va
            check["bound_high"] += plan.cost + envelope - va
            continue

        adv = cached_solve(cache, week, d, adversary=True, gamma=gamma,
                           seconds=adversary_seconds, threads=threads,
                           move_penalty=move_penalty, tie=False)
        check["adversary_statuses"].append(adv.status)
        if not adv.optimal:
            check["adversary_incomplete"] += 1
        if adv.assignment is not None:
            check["added"] += int(lib.add(adv.assignment))
        if adv.lower is None or adv.upper is None or plan.lower is None:
            check["missing"] += 1
            continue
        if method == "gap":
            check["bound_low"] += gamma * plan.lower + adv.lower - va
            check["bound_high"] += gamma * plan.cost + adv.upper - va
        else:
            ref = next(z for z in lib.assignments
                       if tuple(z) == tuple(lib._oracle_assignment))
            jref = schedule_cost(week, ref, d, move_penalty)
            check["bound_low"] += gamma * jref + adv.lower - va
            check["bound_high"] += gamma * jref + adv.upper - va

    for k in ("realized", "regret", "bound_low", "bound_high"):
        check[k] /= max(1, len(weeks))
    check["certified"] = (check["missing"] == 0
                          and check["planner_incomplete"] == 0
                          and check["adversary_incomplete"] == 0)
    return check


def train_method(method, weeks, *, start, alpha, h, gamma, lam, outer,
                 inner_evals, seconds, adversary_seconds, threads,
                 cache, move_penalty, initial_libraries=None):
    if method not in ("vf", "gap", "spo"):
        raise ValueError(method)
    libs = [Library(week, move_penalty) for week in weeks]
    va, za = [], []
    for week, lib in zip(weeks, libs):
        booked = cached_solve(cache, week, week.booked, seconds=seconds,
                              threads=threads, move_penalty=move_penalty, tie=True)
        actual = cached_solve(cache, week, week.actual, seconds=seconds,
                              threads=threads, move_penalty=move_penalty, tie=True)
        if not booked.optimal or not actual.optimal:
            raise RuntimeError(f"Seed solve incomplete: {week.group} {week.monday}, "
                               f"booked={booked.status} actual={actual.status}. "
                               "Increase --seconds or reduce --max-cases.")
        lib.add(booked.assignment)
        lib.add(actual.assignment)
        # Store hindsight schedule reference for SPO; it is NOT assumed to
        # be reachable by any response-limited common policy.
        lib._oracle_assignment = list(actual.assignment)
        va.append(actual.cost)
        za.append(actual.assignment)
    w = np.asarray(start, float).copy()
    log = []
    check = full_policy_check(weeks, libs, w, alpha=alpha, h=h, gamma=gamma,
                              method=method, oracle_value=va, cache=cache,
                              seconds=seconds, adversary_seconds=adversary_seconds,
                              move_penalty=move_penalty, threads=threads)
    log.append({"outer": 0, "library_proxy": library_objective(
        method, weeks, libs, w, alpha=alpha, h=h, gamma=gamma,
        oracle_value=va, oracle_assignment=za, lam=lam),
        "weights": w.tolist(), "evaluation": check,
        "library_size": sum(len(lib.assignments) for lib in libs)})
    for iteration in range(1, outer + 1):
        previous = library_objective(method, weeks, libs, w, alpha=alpha, h=h,
                                     gamma=gamma, oracle_value=va,
                                     oracle_assignment=za, lam=lam)
        result = minimize(
            lambda v: library_objective(method, weeks, libs, v,
                alpha=alpha, h=h, gamma=gamma, oracle_value=va,
                oracle_assignment=za, lam=lam),
            w, method="Powell", bounds=[(-100., 100.)] * len(w),
            options={"maxfev": inner_evals, "ftol": 1e-4, "xtol": 1e-3})
        candidate = np.asarray(result.x)
        # Always verify full-policy consequences before accepting a step.
        candidate_check = full_policy_check(
            weeks, libs, candidate, alpha=alpha, h=h, gamma=gamma,
            method=method, oracle_value=va, cache=cache, seconds=seconds,
            adversary_seconds=adversary_seconds, move_penalty=move_penalty,
            threads=threads)
        new_old = library_objective(method, weeks, libs, w, alpha=alpha, h=h,
                                    gamma=gamma, oracle_value=va,
                                    oracle_assignment=za, lam=lam)
        new_candidate = library_objective(method, weeks, libs, candidate,
                                          alpha=alpha, h=h, gamma=gamma,
                                          oracle_value=va, oracle_assignment=za, lam=lam)
        accepted = new_candidate <= new_old + 1e-8
        if accepted:
            w = candidate
            check = candidate_check
        log.append({"outer": iteration, "accepted": accepted,
                    "initial_library_proxy": previous,
                    "library_proxy": new_candidate if accepted else new_old,
                    "weight_change": float(np.max(np.abs(candidate - log[-1]["weights"]))),
                    "weights": w.tolist(), "evaluation": check,
                    "candidate_evaluation": candidate_check,
                    "library_size": sum(len(lib.assignments) for lib in libs),
                    "inner_evals": int(result.nfev)})
    return {"method": method, "weights": w.tolist(),
            "oracle_mean": float(np.mean(va)),
            "iterations": log, "status": "heuristic_library_MM",
            "notice": "Restricted-library proxy is not a certified full regret bound."}
