"""CASE pDCA and fixed-roster VF schedule-library MM.

Nominal behavioral assumptions are fixed: alpha=.8, h=30. CASE learns a
shared RAW advice score; VF adds five shared congestion-aware coefficients.
Only deployed advice is canonicalized to its minimum-magnitude preimage.
Optimization retains the full adoption / attenuation / rejection response.
"""
from __future__ import annotations
import json
import time
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
        if self.booked.ndim != 1 or np.any(self.booked <= 0):
            raise ValueError("Booked durations must be positive")
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


def planned_cost_fast(week, assignment, durations):
    """Vectorized primary cost for a previously validated assignment.

    No surgeon-day diagnostics or eligibility loops on each re-score:
    loads() validates once when a schedule enters the library.
    Turnover and unused roster block idle are both included.
    """
    z=np.asarray(assignment,dtype=int)
    d=np.asarray(durations,dtype=float)
    counts=np.bincount(z,minlength=week.k)
    room_load=(np.bincount(z,weights=d+TURNOVER,minlength=week.k)
               -TURNOVER*(counts>0))
    return float((C_OVER*np.maximum(room_load-CAPACITY,0.)
                  +C_UNDER*np.maximum(CAPACITY-room_load,0.)).sum())


def library_value(items,libraries,d,actual,lam,w):
    """True finite-library envelope, with full unused-roster idle cost."""
    total=0.
    for key,c,rows in items:
        if not libraries.get(key):
            raise ValueError("Every component needs a feasible schedule")
        segment=d[rows]
        best=min(planned_cost_fast(c,z,segment) for z in libraries[key])
        e=c.actual-segment
        total+=best+(C_OVER*np.maximum(e,0.)+
                     C_UNDER*np.maximum(-e,0.)).sum()
    return float(total/len(items)+lam*np.abs(w).sum())


def choose_library_schedules(items,libraries,d):
    """Touch the current LIBRARY surrogate, not a guessed global optimum."""
    selected={}
    for key,c,rows in items:
        costs=[planned_cost_fast(c,z,d[rows]) for z in libraries[key]]
        pos=min(range(len(costs)),key=lambda j:(costs[j],tuple(libraries[key][j])))
        selected[key]=libraries[key][pos]
    return selected


def _save(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


class CaseConvexQP:
    """One reusable exact convex epigraph QP for the CASE pDCA steps.

    At each step minimize G(w)-gradH(current)@w + prox/2*||w-current||^2.
    Per-case hinge auxiliaries model P and N, and A>=max(e+N,P).
    All hinge coefficients are nonnegative, so these inequalities provide
    the EXACT convex G, not a cutting-plane approximation. Reuse the same
    constraints and warm start throughout CASE learning.
    """

    def __init__(self, objective, lower, upper):
        import gurobipy as gp
        from gurobipy import GRB
        if objective.mode != "CASE":
            raise ValueError("Exact CASE QP is intended for CASE only")
        self.gp=gp
        self.GRB=GRB
        self.objective=objective
        m=gp.Model("case_pdca_exact_epigraph")
        m.Params.OutputFlag=0
        self.model=m
        n,p=objective.n,objective.p
        self.w=m.addMVar(p,lb=lower,ub=upper,name="weights")
        raw=m.addMVar(n,lb=-GRB.INFINITY,name="raw_case_score")
        m.addConstr(raw == sparse.csr_matrix(objective.X) @ self.w + objective.base,
                    name="raw_from_features")
        P=0. * raw + objective.map.offset
        N=0. * raw
        # Sparse per-hinge embedding avoids artificial constraints on absent
        # case-specific floor breakpoints and uses no response binaries.
        for j in range(objective.map.knots.shape[1]):
            wp=objective.map.positive[:,j]
            wn=objective.map.negative[:,j]
            active=np.flatnonzero((wp+wn)>0.)
            if not len(active):
                continue
            hinge=m.addMVar(len(active),lb=0.,name=f"hinge_{j}")
            m.addConstr(hinge >= raw[active]-objective.map.knots[active,j],
                        name=f"hinge_lower_{j}")
            rows=active
            columns=np.arange(len(active))
            shape=(n,len(active))
            if np.any(wp[active]):
                plus=sparse.csr_matrix((wp[active],(rows,columns)),shape=shape)
                P=P+plus@hinge
            if np.any(wn[active]):
                minus=sparse.csr_matrix((wn[active],(rows,columns)),shape=shape)
                N=N+minus@hinge
        A=m.addMVar(n,lb=-GRB.INFINITY,name="case_envelope")
        e=objective.actual-objective.booked
        m.addConstr(A >= P,name="case_P")
        m.addConstr(A >= e+N,name="case_e_plus_N")
        absw=m.addMVar(p,lb=0.,name="abs_coeff")
        m.addConstr(absw>=self.w)
        m.addConstr(absw>=-self.w)
        self.base_G=(KAPPA/objective.components)*A.sum() \
            -C_UNDER*float(e.sum())/objective.components \
            +objective.lam*(objective.penalize@absw)

    def solve(self, center, gradient, prox):
        center=np.asarray(center,float)
        m=self.model
        w=self.w
        m.setObjective(self.base_G-gradient@w
                 +(.5*prox)*((w-center)@(w-center)),
                 self.GRB.MINIMIZE)
        w.Start=center
        m.optimize()
        if not m.SolCount:
            return center,dict(status="CASE_QP_NO_SOLUTION",
                               certified=False,trace=[],solver_status=int(m.Status))
        candidate=np.asarray(w.X,float)
        ok=m.Status==self.GRB.OPTIMAL
        return candidate,dict(status="CASE_QP_OPTIMAL" if ok else "CASE_QP_FEASIBLE",
                              certified=bool(ok),trace=[],
                              solver_status=int(m.Status),
                              objective=float(m.ObjVal))

    def close(self):
        self.model.dispose()


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
         max_cuts=60, tolerance=1e-5, backend="gurobi",
         progress_label=None, progress_path=None):
    """Proximal DCA with exact CASE QP and verified objective descent.

    CASE/Gurobi: reuse an exact convex epigraph QP, not an approximate
    high-dimensional cutting-plane master. VF: use the small-dimensional
    convex cutting-plane QP. An accepted step is NEVER discarded simply
    because the inner cut budget was exhausted.
    """
    x=np.asarray(start,float).copy()
    radius=float(radius)
    if radius<=0 or outer<1 or prox<0:
        raise ValueError("Invalid pDCA settings")
    lower=np.full(len(x),-radius)
    upper=np.full(len(x),radius)
    if np.any(x<lower) or np.any(x>upper):
        raise ValueError("Initial coefficients outside their bounds")
    history=[]
    started=time.perf_counter()
    reason="ITERATION_CAP"
    if progress_label:
        print(f"{progress_label} pDCA: setting up optimizer "
              f"(maximum {outer} steps)",flush=True)
    solver=(CaseConvexQP(objective,lower,upper)
            if objective.mode=="CASE" and backend=="gurobi" else None)
    if progress_label:
        print(f"{progress_label} pDCA: optimizer ready after "
              f"{time.perf_counter()-started:.1f}s",flush=True)
    try:
        for k in range(outer):
            g,h,_,hg=objective.parts(x)
            old=g-h
            if solver is not None:
                trial,inner=solver.solve(x,hg,prox)
            else:
                trial,inner=_solve_qp_cuts(
                    objective,x,hg,lower,upper,prox,
                    max_cuts=max_cuts,tolerance=tolerance,backend=backend)
            value=objective.value(trial)
            improved=bool(np.isfinite(value)
                     and value<old-1e-8*max(1.,abs(old)))
            history.append(dict(
                iteration=k+1,before=float(old),after=float(value),
                accepted=improved,
                step_l2=float(np.linalg.norm(trial-x)),
                proximal_parameter=float(prox),
                inner_status=inner["status"],
                inner_certified=inner["certified"],
                inner_gap=inner.get("gap"),
                cuts=len(inner["trace"])))
            row=history[-1]
            row["elapsed_seconds"]=round(time.perf_counter()-started,3)
            if progress_label:
                print(f"{progress_label} pDCA {k+1}/{outer}: "
                      f"objective {old:.6f} -> {value:.6f}; "
                      f"{'accepted' if improved else 'not accepted'}; "
                      f"{inner['status']}; "
                      f"elapsed {row['elapsed_seconds']:.1f}s",flush=True)
            if progress_path is not None:
                path=Path(progress_path)
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps(dict(**row,
                        weights=(trial if improved else x).tolist()))+"\n")
            if not improved:
                reason="NO_VERIFIED_DESCENT"
                break
            x=np.asarray(trial,float)
            # This is an objective-based stop, not a solver cut-cap gate.
            if old-value < 1e-6*max(1.,abs(old)):
                reason="RELATIVE_IMPROVEMENT"
                break
    finally:
        if solver is not None:
            solver.close()
    return dict(weights=x.tolist(),objective=objective.value(x),
                status=reason,iterations=history,backend=backend)



def vf_directional_refine(items, libraries, booked, H, case_raw,
                          response_map, actual, lam, reference, radius,
                          *, steps=(16., 8., 4., 2., 1.)):
    """Bounded multidirectional refinement of the SAME finite-library VF loss.

    Pairs and coordinate directions are searched independently of pDCA.
    This is the VF_DIRECTIONS optimizer, distinct from VF_PDCA.
    No approximate planner outputs decide acceptance: actual library
    objective values are compared with the same full training data.
    """
    x=np.asarray(reference,float)
    n=len(x)
    basis=np.eye(n)
    directions=[sgn*basis[j] for j in range(n) for sgn in (-1.,1.)]
    # Include non-axis motion to cross the nonmonotone response plateaus.
    for j in range(1,n):
        for sign in (-1.,1.):
            vec=basis[2 if n>2 else 0]+sign*basis[j]
            norm=np.max(np.abs(vec))
            if norm:
                directions.append(vec/norm)
                directions.append(-vec/norm)
    # Also search radial steps along the current multidirectional iterate.
    if np.linalg.norm(x)>1e-8:
        v=x/max(1.,np.max(np.abs(x)))
        directions.extend([v,-v])
    d0=booked+response_map.correction(case_raw+H@x)
    best=library_value(items,libraries,d0,actual,lam,x)
    chosen=x.copy()
    seen={tuple(np.round(x,9))}
    tried=0
    for step in steps:
        for v in directions:
            candidate=np.clip(x+step*v,-radius,radius)
            key=tuple(np.round(candidate,9))
            if key in seen:
                continue
            seen.add(key)
            d=booked+response_map.correction(case_raw+H@candidate)
            score=library_value(items,libraries,d,actual,lam,candidate)
            tried+=1
            if score<best-1e-8*max(1.,abs(best)):
                best=score
                chosen=candidate.copy()
    return chosen,dict(objective=float(best),evaluated=tried,
                       improved=bool(np.linalg.norm(chosen-x)>1e-10))


def mm_vf(items, booked, H, case_raw, response_map, actual, *,
          root, lam, initial_libraries, iterations=30, radius=64.,
          pdca_steps=50, max_cuts=80, backend="gurobi",
          solve_plans=None, optimizer="pdca"):
    """Finite-library majorization followed by full-response pDCA.

    Every planner incumbent is FEASIBLE; certification is not required to
    enlarge a library. A time-limited dual bound is recorded for epsilon.
    Every accepted update decreases the EXACT finite-library surrogate.
    Library enlargement can only tighten that surrogate.
    """
    if optimizer not in ("pdca","directions"):
        raise ValueError("VF optimizer must be 'pdca' or 'directions'")
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
                   optimizer=optimizer,status="RUNNING")
    if state.get('optimizer',optimizer)!=optimizer:
        raise ValueError('VF checkpoint optimizer mismatch')
    if state.get("status") in ("NO_CONFIRMED_IMPROVEMENT",
                               "RELATIVE_IMPROVEMENT",
                               "OUTER_ITERATION_CAP"):
        print(f"VF_{optimizer.upper()}: reusing completed checkpoint "
              f"({len(state['history'])} iterations)",flush=True)
        return state
    if not all(libraries.get(key) for key,_,_ in items):
        raise ValueError("Every component requires a feasible initial assignment")
    run_started=time.perf_counter()
    for outer in range(len(state["history"]),iterations):
        outer_started=time.perf_counter()
        print(f"VF_{optimizer.upper()} MM {outer+1}/{iterations}: obtaining planner schedules "
              f"for {len(items)} components",flush=True)
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
            value=min(planned_cost_fast(c,z,d[rows]) for z in libraries[key])
            bound=candidate.get("bound") if isinstance(candidate,dict) else None
            gaps.append(max(0.,value-max(0.,float(bound))) if bound is not None else value)
        selected=choose_library_schedules(items,libraries,d)
        S,_=room_matrix(items,selected,len(booked))
        objective=DCObjective(booked,H,case_raw,response_map,mode="VF",
                              actual=actual,components=len(items),rooms=S,lam=lam)
        before=library_value(items,libraries,d,actual,lam,theta)
        if abs(before-objective.value(theta))>1e-5*max(1.,abs(before)):
            raise AssertionError("MM touching identity failed: objective normalization/rooms")
        if optimizer=="pdca":
            fitted=pdca(objective,theta,radius=radius,outer=pdca_steps,
                        max_cuts=max_cuts,tolerance=1e-7,backend=backend)
            trial=np.asarray(fitted["weights"],float)
            search=dict(pdca_status=fitted["status"],
                        inner=fitted["iterations"])
        else:
            trial,dlog=vf_directional_refine(
                items,libraries,booked,H,case_raw,response_map,actual,
                lam,theta,radius)
            search=dict(directions_checked=dlog["evaluated"],
                        direction_improved=dlog["improved"])
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
                      search=search))
        if accepted:
            theta=trial
            state["theta"]=theta.tolist()
        row=state["history"][-1]
        row["elapsed_seconds"]=round(time.perf_counter()-outer_started,3)
        print(f"VF_{optimizer.upper()} MM {outer+1}/{iterations}: "
              f"objective {before:.6f} -> {after:.6f}; "
              f"delta={improvement:.6f}, epsilon={row['epsilon']:.6f}, "
              f"accepted={accepted}; schedules={row['added_schedules']}; "
              f"outer elapsed={row['elapsed_seconds']:.1f}s; "
              f"total elapsed={time.perf_counter()-run_started:.1f}s",flush=True)
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
