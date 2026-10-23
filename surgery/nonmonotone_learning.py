"""Non-monotone surgeon-response CASE and value-function MM / pDCA.

This is a new research trainer, not a change to the frozen 2026 experiments.
Policy controls DISPLAYED advice. Each case clips its own display; shared
coefficients are never restricted to the intersection of reachable intervals.

For fixed schedules, the CASE and VF losses admit G(w)-H(w) decompositions
with convex G,H. A cutting-plane convex QP supplies pDCA updates; an outer
exact-planner step supplies a touching VF majorizer. No global optimum of the
nonconvex learning problem is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse

C_OVER = 1.75
C_UNDER = 1.0
KAPPA = C_OVER + C_UNDER
CAPACITY = 480.0
TURNOVER = 30.0


def response(u, alpha, h):
    """The original non-monotone adoption, attenuation and rejection law."""
    u = np.asarray(u, dtype=float)
    if not (0 < alpha <= 1 and h > 0):
        raise ValueError("Require 0 < alpha <= 1 and h > 0")
    return np.sign(u) * np.minimum(
        alpha * np.abs(u), np.maximum(h - (1 - alpha) * np.abs(u), 0.0)
    )


@dataclass(frozen=True)
class Scenario:
    alpha: float
    h: float
    weight: float = 1.0

    def __post_init__(self):
        if not (0 < self.alpha <= 1 and self.h > 0 and self.weight > 0):
            raise ValueError("Invalid scenario")


class ResponseDC:
    """Exact DC decomposition of response(clip(raw, lower_i, cap)).

    The continuous piecewise-linear composite is constant to the left of
    lower_i. At every breakpoint, split the change of slope into its positive
    and negative parts. Then response = P-N, with both P and N convex.
    This handles display clipping, short bookings, and all rejection regions.
    """

    def __init__(self, booked, alpha, h, cap, canonical_nominal=None):
        self.booked = np.asarray(booked, dtype=float)
        if self.booked.ndim != 1 or np.any(self.booked < 1):
            raise ValueError("Positive one-dimensional booked durations required")
        if not (0 < alpha <= 1 and h > 0 and cap > 0):
            raise ValueError("Invalid response or cap")
        self.alpha, self.h, self.cap = float(alpha), float(h), float(cap)
        self.canonical_nominal = canonical_nominal
        if canonical_nominal is not None:
            a0, h0 = canonical_nominal
            if not (0 < a0 <= 1 and h0 > 0):
                raise ValueError("Invalid nominal response for minimum-magnitude display")
        self.lo = np.maximum(-self.cap, 1.0 - self.booked)
        self.hi = np.full_like(self.lo, self.cap)

        def response_knots(a, threshold):
            q = 1-a
            values = [-threshold, threshold]
            if q > 0:
                values += [-threshold/q, threshold/q]
            return values

        knots_for_cases, jumps_for_cases, offsets = [], [], []
        for lo, hi in zip(self.lo, self.hi):
            lo, hi = float(lo), float(hi)
            def f(x):
                displayed = np.clip(x, lo, hi)
                if canonical_nominal is not None:
                    a0, h0 = canonical_nominal
                    displayed = response(displayed, a0, h0) / a0
                return float(response(displayed, alpha, h))
            values = [lo, hi]
            if canonical_nominal is None:
                values.extend(response_knots(alpha, h))
            else:
                a0, h0 = canonical_nominal
                boundaries = [lo]+sorted(x for x in response_knots(a0,h0)
                                          if lo < x < hi)+[hi]
                values.extend(boundaries)
                targets = [a0*x for x in response_knots(alpha,h)]
                for x0,x1 in zip(boundaries[:-1],boundaries[1:]):
                    y0=float(response(x0,a0,h0))
                    y1=float(response(x1,a0,h0))
                    if abs(y1-y0) < 1e-12:
                        continue
                    for target in targets:
                        t=(target-y0)/(y1-y0)
                        if 0 < t < 1:
                            values.append(x0+t*(x1-x0))
            values=sorted(x for x in values if lo <= x <= hi)
            unique=[]
            for x in values:
                if not unique or x-unique[-1] > 1e-9:
                    unique.append(x)
            if len(unique)<2:
                raise ValueError("Empty displayed-advice domain")
            slopes=[0.0]
            slopes += [(f(x1)-f(x0))/(x1-x0)
                       for x0,x1 in zip(unique[:-1],unique[1:])]
            slopes += [0.0]
            jumps=[slopes[j+1]-slopes[j] for j in range(len(unique))]
            knots_for_cases.append(unique)
            jumps_for_cases.append(jumps)
            offsets.append(f(lo))
        width=max(map(len,knots_for_cases))
        self.knots=np.full((len(self.lo),width),np.inf)
        self.positive=np.zeros_like(self.knots)
        self.negative=np.zeros_like(self.knots)
        for i,(knots,jumps) in enumerate(zip(knots_for_cases,jumps_for_cases)):
            for j,(k,delta) in enumerate(zip(knots,jumps)):
                self.knots[i,j]=k
                self.positive[i,j]=max(delta,0.0)
                self.negative[i,j]=max(-delta,0.0)
        self.offset=np.asarray(offsets)
        self._check_shapes()

    def _check_shapes(self):
        assert self.positive.shape == self.negative.shape == self.knots.shape
        assert self.positive.shape[0] == len(self.booked)

    def parts(self, raw):
        raw = np.asarray(raw, float)
        if raw.shape != self.booked.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid raw advice")
        distance = np.maximum(raw[:, None] - self.knots, 0.0)
        positive = self.offset + (distance * self.positive).sum(axis=1)
        negative = (distance * self.negative).sum(axis=1)
        active = (raw[:, None] > self.knots).astype(float)
        active += 0.5 * (raw[:, None] == self.knots)
        dp = (active * self.positive).sum(axis=1)
        dn = (active * self.negative).sum(axis=1)
        return positive, negative, dp, dn

    def display(self, raw):
        u = np.clip(raw, self.lo, self.hi)
        if self.canonical_nominal is not None:
            a0, h0 = self.canonical_nominal
            u = response(u, a0, h0) / a0
        return u

    def reviewed(self, raw):
        return self.booked + response(self.display(raw), self.alpha, self.h)

    def verify(self, raw, tol=1e-7):
        p, n, _, _ = self.parts(raw)
        if not np.allclose(p - n, self.reviewed(raw) - self.booked,
                           rtol=0, atol=tol):
            raise AssertionError("DC response != exact clipped non-monotone response")


def occupied_room_matrix(items, plans, n_cases):
    """Sparse incidence for occupied physical room-days only.

    items are the verified components (key, Week, global row indices);
    plans maps each key to a length-Week.n certified assignment.
    """
    rr, cc, counts = [], [], []
    for key, week, rows in items:
        rows = np.asarray(rows, int)
        if key not in plans:
            raise ValueError("Missing certified plan: " + str(key))
        z = np.asarray(plans[key], int)
        if z.shape != (week.n,):
            raise ValueError("Assignment length mismatch")
        # Also validates eligibility and observed surgeon-day retention.
        from surgery.weekly_planner import loads
        loads(week, z, week.booked)
        for j in sorted(set(z.tolist())):
            selected = rows[z == j]
            rr.extend([len(counts)] * len(selected))
            cc.extend(selected.tolist())
            counts.append(len(selected))
    if len(cc) != n_cases or len(set(cc)) != n_cases:
        raise AssertionError("Room incidence must partition all training cases")
    matrix = sparse.csr_matrix((np.ones(len(cc)), (rr, cc)),
                               shape=(len(counts), n_cases))
    return matrix, np.asarray(counts, int)


class DCObjective:
    """Case error or fixed-plan VF, over one common displayed-advice policy.

    raw = base + X @ w. The reference base is ZERO for CASE and the frozen
    fitted CASE display for VF. The same raw advice enters every scenario.
    For VF, actual_bank is training-only duration scenario draws (draws,cases).
    The exact fixed-plan reduced VF objective is G-H.
    """

    def __init__(self, booked, X, base, scenarios, cap, *, mode,
                 actual=None, actual_bank=None, room_matrices=None,
                 ridge=0.001, l1=0.0, regularize_from=None, canonical_nominal=None):
        self.booked = np.asarray(booked, float)
        self.X = np.asarray(X, float)
        self.base = np.asarray(base, float)
        n, p = self.X.shape
        if self.booked.shape != (n,) or self.base.shape != (n,):
            raise ValueError("Shape mismatch: booked/base")
        if mode not in ("CASE", "VF"):
            raise ValueError("Unknown objective")
        self.mode = mode
        self.scenarios = tuple(scenarios)
        if not self.scenarios:
            raise ValueError("At least one scenario required")
        weights = np.array([s.weight for s in self.scenarios], float)
        self.weights = weights / weights.sum()
        self.canonical_nominal = canonical_nominal
        self.maps = [ResponseDC(self.booked, s.alpha, s.h, cap, canonical_nominal)
                     for s in self.scenarios]
        self.cap = float(cap)
        self.ridge = float(ridge)
        self.l1 = float(l1)
        if self.ridge < 0 or self.l1 < 0:
            raise ValueError("Penalties must be nonnegative")
        self.reference = np.zeros(p) if regularize_from is None else np.asarray(regularize_from, float)
        if self.reference.shape != (p,):
            raise ValueError("Reference coefficient shape mismatch")
        self.actual = None if actual is None else np.asarray(actual, float)
        self.bank = None if actual_bank is None else np.asarray(actual_bank, float)
        if mode == "CASE" and (self.actual is None or self.actual.shape != (n,)):
            raise ValueError("CASE requires actual training durations")
        if mode == "VF" and (self.bank is None or self.bank.ndim != 2 or
                              self.bank.shape[1] != n or self.bank.shape[0] < 1):
            raise ValueError("VF requires a frozen training scenario bank")
        self.rooms = room_matrices
        if mode == "VF":
            if self.rooms is None or len(self.rooms) != len(self.scenarios):
                raise ValueError("VF needs a fixed schedule for each response scenario")
            for S in self.rooms:
                if S.shape[1] != n:
                    raise ValueError("Invalid room incidence")
        self.n, self.p = n, p

    def replace_rooms(self, matrices):
        if self.mode != "VF":
            raise ValueError("Only VF has room schedules")
        return DCObjective(self.booked, self.X, self.base, self.scenarios,
                           self.cap, mode="VF", actual_bank=self.bank,
                           room_matrices=matrices, ridge=self.ridge,
                           l1=self.l1, regularize_from=self.reference,
                           canonical_nominal=self.canonical_nominal)

    def parts(self, w):
        """Return G,H,subgradient G,subgradient H; both G,H are convex."""
        w = np.asarray(w, float)
        if w.shape != (self.p,) or not np.isfinite(w).all():
            raise ValueError("Bad coefficient vector")
        raw = self.base + self.X @ w
        G = H = 0.0
        dg = np.zeros(self.p)
        dh = np.zeros(self.p)
        for k, (weight, response_map) in enumerate(zip(self.weights, self.maps)):
            P, N, Pp, Np = response_map.parts(raw)
            if self.mode == "CASE":
                e = self.actual - self.booked
                choose_p = np.where(P > e + N, 1., np.where(P < e + N, 0., .5))
                value = KAPPA * np.maximum(e + N, P).sum() - C_UNDER * e.sum()
                H_local = C_OVER * P.sum() + C_UNDER * N.sum()
                coeff_p = KAPPA * choose_p
                coeff_n = KAPPA * (1. - choose_p)
                dh_raw = C_OVER * Pp + C_UNDER * Np
            else:
                e = self.bank - self.booked[None, :]
                choose_p = np.where(P[None, :] > e + N[None, :], 1.,
                                    np.where(P[None, :] < e + N[None, :], 0., .5))
                case_term = np.maximum(e + N[None, :], P[None, :]).mean(axis=0)
                coeff_p = KAPPA * choose_p.mean(axis=0)
                coeff_n = KAPPA * (1. - choose_p.mean(axis=0))
                S = self.rooms[k]
                counts = np.diff(S.indptr)
                load_offset = (S @ self.booked + TURNOVER * (counts - 1) - CAPACITY)
                pp, nn = S @ P, S @ N
                high = np.where(load_offset + pp > nn, 1.,
                         np.where(load_offset + pp < nn, 0., .5))
                room_cost = TURNOVER * S.shape[0] + KAPPA * np.maximum(load_offset + pp, nn).sum()
                value = KAPPA * case_term.sum() + room_cost
                H_local = KAPPA * (P.sum() + N.sum())
                coeff_p += KAPPA * np.asarray(S.T @ high).ravel()
                coeff_n += KAPPA * np.asarray(S.T @ (1. - high)).ravel()
                dh_raw = KAPPA * (Pp + Np)
            G += weight * value / self.n
            H += weight * H_local / self.n
            dg += weight * (self.X.T @ (coeff_p * Pp + coeff_n * Np)) / self.n
            dh += weight * (self.X.T @ dh_raw) / self.n
        diff = w - self.reference
        G += 0.5 * self.ridge * np.dot(diff, diff)
        dg += self.ridge * diff
        if self.l1:
            G += self.l1 * np.abs(diff).sum()
            dg += self.l1 * np.sign(diff)
        return float(G), float(H), dg, dh

    def value(self, w):
        g, h, _, _ = self.parts(w)
        return float(g - h)

    def scenario_values(self, w):
        """Exact decomposed loss under each response scenario (without penalty)."""
        vals = []
        for k, s in enumerate(self.scenarios):
            obj = DCObjective(self.booked, self.X, self.base, [s], self.cap,
                 mode=self.mode, actual=self.actual, actual_bank=self.bank,
                 room_matrices=None if self.mode == "CASE" else [self.rooms[k]],
                 ridge=0., l1=0., canonical_nominal=self.canonical_nominal)
            vals.append(obj.value(w))
        return vals


def _solve_qp_cuts(objective, center, h_gradient, lower, upper, prox,
                   *, max_cuts=60, tolerance=1e-5, backend="gurobi"):
    """Solve convex pDCA master using supporting cuts for G, not all case hinges.

    A Gurobi QP solves each finite-cut master globally. The remaining
    G(candidate)-master_epigraph gap bounds optimization error of the
    convex majorizer. 'scipy' is for smoke tests, not solver certification.
    """
    center = np.asarray(center, float)
    cuts = []
    point = center.copy()
    G0, _, grad0, _ = objective.parts(point)
    cuts.append((point.copy(), G0, grad0.copy()))
    certified = backend == "gurobi"
    trace = []
    if backend not in ("gurobi", "scipy"):
        raise ValueError("Unknown bundle backend")
    if backend == "gurobi":
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("Production pDCA requires Gurobi; use --backend scipy for a non-certified smoke test") from exc
        model = gp.Model("convex_dc_bundle")
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        variables = model.addMVar(len(center), lb=lower, ub=upper, name="w")
        epi = model.addVar(lb=-GRB.INFINITY, name="G_epi")
        model.setObjective(epi - h_gradient @ variables +
            prox / 2.0 * ((variables - center) @ (variables - center)), GRB.MINIMIZE)
        def add_cut(v, g, grad):
            model.addConstr(epi >= g + grad @ (variables - v))
    else:
        from scipy.optimize import minimize
        def add_cut(v, g, grad):
            return None
    try:
        for j in range(max_cuts):
            v, g, grad = cuts[-1]
            add_cut(v, g, grad)
            if backend == "gurobi":
                variables.Start = point
                model.optimize()
                if model.Status != GRB.OPTIMAL:
                    return point, dict(status="MASTER_UNCERTIFIED", trace=trace, certified=False)
                point = np.asarray(variables.X, float)
                supporting = float(epi.X)
            else:
                size = len(center)
                start = np.r_[point, max(g0 + np.dot(gi, point-vi) for vi,g0,gi in cuts)]
                def fun(x):
                    w, t = x[:-1], x[-1]
                    return t - h_gradient @ w + prox / 2 * np.square(w-center).sum()
                def jac(x):
                    return np.r_[-h_gradient + prox * (x[:-1]-center), 1.]
                constraints = [{"type":"ineq",
                    "fun":lambda x,vi=vi,g0=g0,gi=gi: x[-1]-g0-gi@(x[:-1]-vi),
                    "jac":lambda x,vi=vi,g0=g0,gi=gi: np.r_[-gi,1.]}
                    for vi,g0,gi in cuts]
                ans = minimize(fun, start, jac=jac, bounds=list(zip(lower,upper))+[(None,None)],
                               constraints=constraints, method="SLSQP",
                               options={"maxiter":300,"ftol":1e-10})
                if not ans.success or not np.isfinite(ans.fun):
                    return point, dict(status="SCIPY_MASTER_FAILED", message=ans.message,
                                       trace=trace, certified=False)
                point = np.asarray(ans.x[:-1], float)
                supporting = float(ans.x[-1])
            G, _, grad, _ = objective.parts(point)
            gap = max(0., G - supporting)
            trace.append(dict(iteration=j+1, gap=float(gap)))
            if gap <= tolerance * max(1., abs(G)):
                return point, dict(status="INNER_CONVERGED", trace=trace,
                                   certified=certified, gap=float(gap))
            cuts.append((point.copy(), G, grad.copy()))
        return point, dict(status="INNER_CUT_CAP", trace=trace, certified=False,
                           gap=trace[-1]["gap"])
    finally:
        if backend == "gurobi":
            model.dispose()


def pdca(objective, start, *, radius=64., outer=25, prox=1e-3,
         max_cuts=60, tolerance=1e-5, backend="gurobi"):
    """pDCA with certified exact-objective descent checks.

    Bounds are on coefficients, not on all casewise implemented durations.
    Stopping at an iteration cap or a non-certified master is explicit.
    """
    x = np.asarray(start, float).copy()
    radius = float(radius)
    if radius <= 0 or outer < 1 or prox < 0:
        raise ValueError("Invalid learning budget")
    lower = np.full(len(x), -radius)
    upper = np.full(len(x), radius)
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("Start outside coefficient box; do not silently project")
    history = []
    reason = "ITERATION_CAP"
    for k in range(outer):
        g, h, _, hg = objective.parts(x)
        old = g-h
        trial, inner = _solve_qp_cuts(objective,x,hg,lower,upper,prox,
                       max_cuts=max_cuts,tolerance=tolerance,backend=backend)
        value = objective.value(trial)
        improved = bool(np.isfinite(value) and value < old-1e-8*max(1.,abs(old)))
        history.append(dict(iteration=k+1,before=float(old),after=float(value),
                            accepted=improved,inner_status=inner["status"],
                            inner_certified=inner["certified"],
                            inner_gap=inner.get("gap"),
                            cuts=len(inner["trace"])))
        if improved:
            x = trial
        else:
            reason = "NO_VERIFIED_DESCENT"
            break
        if inner["status"] != "INNER_CONVERGED":
            reason = inner["status"]
            break
    return dict(weights=x.tolist(),objective=objective.value(x),
                status=reason,iterations=history,backend=backend)


def mm_vf(items, booked, H, case_display, scenarios, actual_bank, cap,
          *, root, iterations=8, radius=32., ridge=.001,
          pdca_steps=20, max_cuts=60, seconds_per_plan=120.,
          backend="gurobi", solve_plans=None, canonical_nominal=None):
    """Outer exact-plan MM, inner full-response pDCA; resumable per iteration.

    solve_plans(items, durations, scenario_index, iteration) must return
    mapping component key -> certified assignment. The runner supplies the
    exact weekly Gurobi implementation and checkpoints every plan.
    """
    from pathlib import Path
    import json
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    state_path = root/"vf_mm_state.json"
    if state_path.exists():
        state=json.loads(state_path.read_text())
        theta=np.asarray(state["theta"],float)
    else:
        theta=np.zeros(H.shape[1])
        state={"theta":theta.tolist(),"history":[],"status":"RUNNING"}
    if solve_plans is None:
        raise ValueError("Exact scenario-specific planner callback required")
    actual_bank=np.asarray(actual_bank,float)
    for k in range(len(state["history"]),iterations):
        d_by_scenario=[]
        rooms=[]
        for j,scenario in enumerate(scenarios):
            mapping=ResponseDC(booked,scenario.alpha,scenario.h,cap,canonical_nominal)
            reviewed=mapping.reviewed(case_display+H@theta)
            d_by_scenario.append(reviewed)
            plan=solve_plans(items,reviewed,j,k)
            S,_=occupied_room_matrix(items,plan,len(booked))
            rooms.append(S)
        objective=DCObjective(booked,H,case_display,scenarios,cap,mode="VF",
                              actual_bank=actual_bank,room_matrices=rooms,ridge=ridge,
                              canonical_nominal=canonical_nominal)
        old=objective.value(theta)
        fitted=pdca(objective,theta,radius=radius,outer=pdca_steps,
                    max_cuts=max_cuts,backend=backend)
        proposal=np.asarray(fitted["weights"],float)
        upper=objective.value(proposal)
        # The true VF score is <= fixed-plan majorizer at the candidate,
        # and equals the fixed-plan majorizer at the incumbent.
        accepted=bool(upper < old-1e-8*max(1.,abs(old)))
        state["history"].append(dict(outer=k+1,before=old,majorizer_after=upper,
                         accepted=accepted,plans=len(items)*len(scenarios),
                         pdca_status=fitted["status"],pdca_iterations=fitted["iterations"]))
        if accepted:
            theta=proposal
            state["theta"]=theta.tolist()
        if not accepted:
            state["status"]="NO_VERIFIED_MM_GAIN"
        tmp=state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,sort_keys=True,indent=2,allow_nan=False))
        tmp.replace(state_path)
        if not accepted:
            break
    else:
        state["status"]="OUTER_ITERATION_CAP"
        state_path.write_text(json.dumps(state,sort_keys=True,indent=2,allow_nan=False))
    return state
