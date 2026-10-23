"""CASE pDCA and fixed-roster VF schedule-library MM.

Nominal behavioral assumptions are fixed: alpha=.8, h=30. CASE learns a
shared RAW advice score; VF adds five shared congestion-aware coefficients.
Only deployed advice is canonicalized to its minimum-magnitude preimage.
Optimization retains the full adoption / attenuation / rejection response.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse

C_OVER = 1.75
C_UNDER = 1.0
KAPPA = C_OVER + C_UNDER
CAPACITY = 480.
TURNOVER = 30.


def response(u, alpha=.8, h=30.):
    """Non-monotone response, not the linear inverse-response approximation."""
    u=np.asarray(u,float)
    q=1.-alpha
    return np.sign(u)*np.minimum(alpha*np.abs(u),
                np.maximum(h-q*np.abs(u),0.))


@dataclass(frozen=True)
class Scenario:
    alpha: float = .8
    h: float = 30.

    def __post_init__(self):
        if not (0 < self.alpha < 1 and self.h > 0):
            raise ValueError("Response requires 0<alpha<1 and h>0")


class ResponseDC:
    """Exact DC compiler for max{F(clip(u,-cap,cap)), floor-booked}.

    The floor belongs to the IMPLEMENTED duration, not the displayed advice.
    Thus very negative advice can attenuate/reject instead of being clipped
    into the acceptance branch for short booked cases.

    The composite is piecewise linear in the raw displayed-advice score.
    Split each slope jump into positive and negative hinge coefficients:
    f(u)=P(u)-N(u), with convex P,N. No binary response variables.
    """

    def __init__(self, booked, alpha=.8, h=30., cap=180., floor=1.):
        self.booked=np.asarray(booked,float)
        if self.booked.ndim != 1 or np.any(self.booked < floor):
            raise ValueError("Booked durations must exceed the positive-duration floor")
        self.alpha,self.h,self.cap,self.floor=map(float,(alpha,h,cap,floor))
        if not (0<alpha<1 and h>0 and cap>0 and floor>0):
            raise ValueError("Invalid nominal response")
        self.lo=np.full_like(self.booked,-cap)
        self.hi=np.full_like(self.booked,cap)
        self.offset=np.empty_like(self.booked)
        all_knots=[];all_jumps=[]
        q=1-alpha
        for i,b in enumerate(self.booked):
            cutoff=floor-float(b)
            def actual(x):
                return max(float(response(np.clip(x,-cap,cap),alpha,h)),cutoff)
            base=[-cap,cap,-h,h,-h/q,h/q]
            base=sorted(set(float(t) for t in base if -cap <= t <= cap))
            roots=list(base)
            for x0,x1 in zip(base[:-1],base[1:]):
                y0=float(response(x0,alpha,h));y1=float(response(x1,alpha,h))
                if abs(y1-y0)<1.e-12:
                    continue
                frac=(cutoff-y0)/(y1-y0)
                if 1e-12<frac<1-1e-12:
                    roots.append(x0+frac*(x1-x0))
            roots=sorted(roots)
            unique=[]
            for x in roots:
                if not unique or x-unique[-1]>1e-9:
                    unique.append(x)
            if len(unique)<2:
                raise ValueError("Invalid raw score domain")
            slopes=[0.]+[(actual(v)-actual(u))/(v-u)
                for u,v in zip(unique[:-1],unique[1:])]+[0.]
            jumps=[slopes[j+1]-slopes[j] for j in range(len(unique))]
            all_knots.append(unique);all_jumps.append(jumps)
            self.offset[i]=actual(-cap)
        width=max(map(len,all_knots))
        self.knots=np.full((len(self.booked),width),np.inf)
        self.positive=np.zeros_like(self.knots)
        self.negative=np.zeros_like(self.knots)
        for i,(knots,jumps) in enumerate(zip(all_knots,all_jumps)):
            for j,(k,delta) in enumerate(zip(knots,jumps)):
                self.knots[i,j]=k
                self.positive[i,j]=max(delta,0.)
                self.negative[i,j]=max(-delta,0.)

    def parts(self, raw):
        raw=np.asarray(raw,float)
        if raw.shape!=self.booked.shape:
            raise ValueError("Advice scores do not match case array")
        hinge=np.maximum(raw[:,None]-self.knots,0.)
        P=self.offset+(hinge*self.positive).sum(axis=1)
        N=(hinge*self.negative).sum(axis=1)
        active=(raw[:,None]>self.knots).astype(float)
        active+=.5*(raw[:,None]==self.knots)
        return P,N,(active*self.positive).sum(axis=1),(
            active*self.negative).sum(axis=1)

    def correction(self, raw):
        u=np.clip(np.asarray(raw,float),-self.cap,self.cap)
        return np.maximum(response(u,self.alpha,self.h),self.floor-self.booked)

    def display(self, raw):
        """Smallest signed display inducing nominal implemented correction."""
        return self.correction(raw)/self.alpha

    def reviewed(self, raw):
        return self.booked+self.correction(raw)

    def verify(self, raw):
        P,N,_,_=self.parts(raw)
        correction=self.correction(raw)
        if not np.allclose(P-N,correction,rtol=0,atol=1e-8):
            raise AssertionError("Response DC decomposition is not exact")
        displayed=self.display(raw)
        if np.any(np.abs(displayed)>np.minimum(self.cap,self.h)+1e-8):
            raise AssertionError("Canonical display exceeds increasing response branch")
        adopted=np.maximum(response(displayed,self.alpha,self.h),
                           self.floor-self.booked)
        if not np.allclose(adopted,correction,rtol=0,atol=1e-8):
            raise AssertionError("Minimum-magnitude display changes adopted booking")


def room_matrix(items, plans, n_cases):
    """All roster blocks, INCLUDING UNUSED 480-minute blocks."""
    rr=[];cc=[];counts=[]
    for key,c,rows in items:
        z=np.asarray(plans[key],int)
        rows=np.asarray(rows,int)
        from surgery.weekly_planner import loads
        loads(c,z,c.booked) # inspect eligibility and required day retention
        if len(rows)!=c.n:
            raise AssertionError("Component rows incomplete")
        for j in range(c.k):
            ix=rows[z==j]
            rr.extend([len(counts)]*len(ix))
            cc.extend(ix.tolist())
            counts.append(len(ix))
    if len(cc)!=n_cases or len(set(cc))!=n_cases:
        raise AssertionError("Every case must appear exactly once")
    S=sparse.csr_matrix((np.ones(len(cc)),(rr,cc)),
                        shape=(len(counts),n_cases))
    return S,np.asarray(counts,int)


class DCObjective:
    """Full asymmetric cost/envelope exactly as manuscript Section 4.

    CASE: sum theta(actual - (booked+F(u))).
    VF: sum J(fixed_schedules,booked+F(u))
           + sum theta(actual-(booked+F(u))).
    Both divide by NUMBER OF TRAINING COMPONENTS, not number of cases.
    The L1 coefficient penalty is convex; intercept is unpenalized for CASE.
    """

    def __init__(self, booked, X, base, response_map, *,
                 mode, actual, components, rooms=None, lam=0.,
                 penalize=None):
        self.booked=np.asarray(booked,float)
        self.X=np.asarray(X,float)
        self.base=np.asarray(base,float)
        self.actual=np.asarray(actual,float)
        self.map=response_map
        self.mode=mode
        if mode not in ("CASE","VF"):
            raise ValueError("mode must be CASE or VF")
        self.n,self.p=self.X.shape
        if self.booked.shape!=(self.n,) or self.base.shape!=(self.n,) or self.actual.shape!=(self.n,):
            raise ValueError("CASE/VF array shape mismatch")
        self.components=int(components)
        self.rooms=rooms
        if self.components<=0:
            raise ValueError("Need positive number of components")
        if mode=="VF" and (rooms is None or rooms.shape[1]!=self.n):
            raise ValueError("VF requires fixed physical-room assignments")
        self.lam=float(lam)
        self.penalize=np.ones(self.p) if penalize is None else np.asarray(penalize,float)
        if self.penalize.shape!=(self.p,):
            raise ValueError("Penalty mask length mismatch")

    def replace_rooms(self,S):
        return DCObjective(self.booked,self.X,self.base,self.map,mode=self.mode,
                           actual=self.actual,components=self.components,rooms=S,
                           lam=self.lam,penalize=self.penalize)

    def parts(self,w):
        w=np.asarray(w,float)
        u=self.base+self.X@w
        P,N,Pp,Np=self.map.parts(u)
        e=self.actual-self.booked
        A=np.maximum(e+N,P)
        choose_a=np.where(P>e+N,1.,np.where(P<e+N,0.,.5))
        G=KAPPA*A.sum()-C_UNDER*e.sum()
        if self.mode=="CASE":
            H=C_OVER*P.sum()+C_UNDER*N.sum()
            dp=KAPPA*choose_a*Pp
            dn=KAPPA*(1-choose_a)*Np
            hraw=C_OVER*Pp+C_UNDER*Np
        else:
            S=self.rooms
            counts=np.diff(S.indptr)
            kappa=S@self.booked+TURNOVER*np.maximum(counts-1,0)-CAPACITY
            pp=S@P
            nn=S@N
            O=np.maximum(kappa+pp,nn)
            I=np.maximum(nn-kappa,pp)
            G+=C_OVER*O.sum()+C_UNDER*I.sum()
            H=KAPPA*(P.sum()+N.sum())
            choose_o=np.where(kappa+pp>nn,1.,np.where(kappa+pp<nn,0.,.5))
            choose_i=np.where(nn-kappa>pp,1.,np.where(nn-kappa<pp,0.,.5))
            grad_P=KAPPA*choose_a+C_OVER*np.asarray(S.T@choose_o).ravel()+C_UNDER*np.asarray(S.T@(1-choose_i)).ravel()
            grad_N=KAPPA*(1-choose_a)+C_OVER*np.asarray(S.T@(1-choose_o)).ravel()+C_UNDER*np.asarray(S.T@choose_i).ravel()
            dp=grad_P*Pp
            dn=grad_N*Np
            hraw=KAPPA*(Pp+Np)
        G/=self.components
        H/=self.components
        dg=self.X.T@(dp+dn)/self.components
        dh=self.X.T@hraw/self.components
        G+=self.lam*np.dot(self.penalize,np.abs(w))
        dg+=self.lam*self.penalize*np.sign(w)
        return float(G),float(H),np.asarray(dg),np.asarray(dh)

    def value(self,w):
        G,H,_,_=self.parts(w)
        return float(G-H)


def library_value(items,libraries,d,actual,lam,w):
    """True finite-library envelope, with full unused-roster idle cost."""
    from surgery.weekly_planner import evaluate
    total=0.
    for key,c,rows in items:
        if not libraries.get(key):
            raise ValueError("Every component needs a feasible schedule")
        segment=d[rows]
        best=min(evaluate(c,z,segment)["cost"] for z in libraries[key])
        e=c.actual-segment
        total+=best+(C_OVER*np.maximum(e,0.)+
                     C_UNDER*np.maximum(-e,0.)).sum()
    return float(total/len(items)+lam*np.abs(w).sum())


def choose_library_schedules(items,libraries,d):
    """Touch the current LIBRARY surrogate, not a guessed global optimum."""
    from surgery.weekly_planner import evaluate
    selected={}
    for key,c,rows in items:
        costs=[evaluate(c,z,d[rows])["cost"] for z in libraries[key]]
        pos=min(range(len(costs)),key=lambda j:(costs[j],tuple(libraries[key][j])))
        selected[key]=libraries[key][pos]
    return selected


def _save(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


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






def mm_vf(items, booked, H, case_raw, response_map, actual, *,
          root, lam, initial_libraries, iterations=30, radius=64.,
          pdca_steps=50, max_cuts=80, backend="gurobi",
          solve_plans=None):
    """Finite-library majorization followed by full-response pDCA.

    Every planner incumbent is FEASIBLE; certification is not required to
    enlarge a library. A time-limited dual bound is recorded for epsilon.
    Every accepted update decreases the EXACT finite-library surrogate.
    Library enlargement can only tighten that surrogate.
    """
    if solve_plans is None:
        raise ValueError("Planner callback required")
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    status=root/"vf_library_progress.json"
    if status.exists():
        state=json.loads(status.read_text())
        theta=np.asarray(state["theta"],float)
        libraries=state["libraries"]
    else:
        theta=np.zeros(H.shape[1])
        libraries={key:[list(map(int,z)) for z in initial_libraries[key]]
                   for key,_,_ in items}
        state=dict(theta=theta.tolist(),history=[],libraries=libraries,
                   status="RUNNING")
    if not all(libraries.get(key) for key,_,_ in items):
        raise ValueError("Every component requires a feasible initial assignment")
    dcase=np.asarray(booked,float)+response_map.correction(case_raw)
    case_certificate=library_value(items,libraries,dcase,actual,lam,theta)
    state.setdefault("case_initial_certificate",case_certificate)
    for outer in range(len(state["history"]),iterations):
        d=np.asarray(booked,float)+response_map.correction(case_raw+H@theta)
        candidates=solve_plans(items,d,outer)
        gaps=[]
        for key,c,rows in items:
            candidate=candidates.get(key)
            if candidate is not None:
                z=candidate.get("assignment") if isinstance(candidate,dict) else candidate
                if z is not None:
                    z=list(map(int,z))
                    from surgery.weekly_planner import loads
                    loads(c,z,d[rows])
                    if z not in libraries[key]:
                        libraries[key].append(z)
            from surgery.weekly_planner import evaluate
            value=min(evaluate(c,z,d[rows])["cost"] for z in libraries[key])
            bound=candidate.get("bound") if isinstance(candidate,dict) else None
            gaps.append(max(0.,value-max(0.,float(bound))) if bound is not None else value)
        selected=choose_library_schedules(items,libraries,d)
        S,_=room_matrix(items,selected,len(booked))
        objective=DCObjective(booked,H,case_raw,response_map,mode="VF",
                              actual=actual,components=len(items),rooms=S,lam=lam)
        before=library_value(items,libraries,d,actual,lam,theta)
        if abs(before-objective.value(theta))>1e-5*max(1.,abs(before)):
            raise AssertionError("MM touching identity failed: objective normalization/rooms")
        fitted=pdca(objective,theta,radius=radius,outer=pdca_steps,
                    max_cuts=max_cuts,backend=backend)
        trial=np.asarray(fitted["weights"],float)
        new_d=booked+response_map.correction(case_raw+H@trial)
        after=library_value(items,libraries,new_d,actual,lam,trial)
        accepted=bool(after<before-1e-8*max(1.,abs(before)))
        improvement=max(0.,before-after) if accepted else 0.
        state["history"].append(dict(outer=outer+1,
                      before=float(before),candidate=float(after),
                      delta=float(improvement),
                      epsilon=float(np.mean(gaps)),
                      accepted=accepted,
                      added_schedules=sum(len(z) for z in libraries.values()),
                      pdca_status=fitted["status"],
                      inner=fitted["iterations"]))
        if accepted:
            theta=trial
            state["theta"]=theta.tolist()
        state["libraries"]=libraries
        state["status"]="RUNNING"
        _save(status,state)
        if not accepted:
            state["status"]="NO_CONFIRMED_IMPROVEMENT"
            _save(status,state)
            break
        if improvement/max(1.,abs(before))<1e-6:
            state["status"]="RELATIVE_IMPROVEMENT"
            _save(status,state)
            break
    else:
        state["status"]="OUTER_ITERATION_CAP"
        _save(status,state)
    return state
