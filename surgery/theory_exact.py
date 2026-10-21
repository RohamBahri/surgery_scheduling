"""Certified finite-sample VF and EG oracles for the UHN weekly planner.

The objective is the REDUCED score 30 * open_rooms + 2.75 * overtime.
All reported MIP bounds have the correct direction; a timeout is never optimal.
Gurobi is imported inside solver functions so the pure-math tests run without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time

import numpy as np

CAP = 480.0
TURN = 30.0
KAPPA = 2.75


@dataclass
class Certificate:
    value: float | None
    lower: float | None
    upper: float | None
    complete: bool
    status: str
    seconds: float
    assignment: tuple[int, ...] | None = None
    reviewed: list[float] | None = None

    def to_json(self):
        return self.__dict__.copy()


def reduced(c, z, durations):
    """Exactly the schedule-dependent part of the weekly planner's cost."""
    from surgery.weekly_planner import loads
    z = np.asarray(z, dtype=int)
    count = np.bincount(z, minlength=c.k)
    return float(TURN * np.count_nonzero(count) + KAPPA *
                 np.maximum(loads(c, z, durations) - CAP, 0.).sum())


def expected_reduced(c, z, bank):
    bank = np.asarray(bank, float)
    if bank.ndim != 2 or bank.shape[1] != c.n or not len(bank):
        raise ValueError("Expected (scenarios, cases) array")
    from surgery.asdbs import mean_cost
    # J = 480K - sum A - 30n + phi; cancel the scenario-specific constant.
    return float(mean_cost(c, z, bank) - CAP*c.k + bank.mean(axis=0).sum() + TURN*c.n)


def stop_loss(d, bank):
    d, bank = np.asarray(d, float), np.asarray(bank, float)
    if bank.ndim != 2 or bank.shape[1] != len(d):
        raise ValueError("Incompatible durations and marginal bank")
    return float(np.maximum(bank-d[None, :], 0).mean(axis=0).sum())


def vf_objective(c, z, d, bank, penalty_center=None, lam=0.):
    answer = reduced(c, z, d) + KAPPA*stop_loss(d, bank)
    if penalty_center is not None:
        answer += lam/2*np.square(np.asarray(d)-np.asarray(penalty_center)).sum()
    return float(answer)


def coefficient_box_images(d0, H, coef_lo, coef_hi, case_lo, case_hi):
    """Safe per-case duration intervals for a coefficient rectangle."""
    H = np.asarray(H, float)
    lo, hi = np.asarray(coef_lo, float), np.asarray(coef_hi, float)
    center = np.asarray(d0, float)
    rawlo = center + np.maximum(H, 0)@lo + np.minimum(H, 0)@hi
    rawhi = center + np.maximum(H, 0)@hi + np.minimum(H, 0)@lo
    lower = np.clip(rawlo, case_lo, case_hi)
    upper = np.clip(rawhi, case_lo, case_hi)
    return lower, upper


def _setup(c, seconds, threads, name):
    from surgery.anchored_weekly_solver import _model
    m, x, used, byroom = _model(c, seconds, threads)
    m.ModelName = name
    return m, x, used, byroom


def _safe_certificate(m, started, objective_name, c=None, x=None, d=None):
    from gurobipy import GRB
    status = str(m.Status)
    have = m.SolCount > 0
    lo = float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED) else None
    up = float(m.ObjVal) if have else None
    complete = m.Status == GRB.OPTIMAL
    z = None
    if have and c is not None:
        from surgery.weekly_planner import _assignment
        z = tuple(map(int, _assignment(c, x)))
    if lo is not None and up is not None and lo > up + 1e-5:
        raise AssertionError(f"Invalid {objective_name} MIP dual bound: {lo} > {up}")
    return Certificate(up, lo, up, complete, status, time.monotonic()-started,
                       z, None if d is None else list(np.asarray(d, float)))


def _product_loads(m, c, x, used, byroom, durations, bounds):
    """Linear exact products of assignment binaries with variable duration."""
    import gurobipy as gp
    from gurobipy import GRB
    lower, upper = map(np.asarray, bounds)
    products = {}
    for j, cases in enumerate(byroom):
        for i in cases:
            lo, hi = float(lower[i]), float(upper[i])
            p = m.addVar(lb=0., ub=hi, name=f"dx_{i}_{j}")
            m.addConstr(p >= lo*x[i,j]); m.addConstr(p <= hi*x[i,j])
            m.addConstr(p >= durations[i] - hi*(1-x[i,j]))
            m.addConstr(p <= durations[i] - lo*(1-x[i,j]))
            products[i,j] = p
    loads = []
    for j, cases in enumerate(byroom):
        loads.append(gp.quicksum(products[i,j] + TURN*x[i,j] for i in cases)
                     - TURN*used[j])
    return loads


def _stoploss(m, durations, bank, prefix):
    """Exact finite-support marginal stop loss with convex tangent epigraphs.

    For samples a_1...a_M, f(d)=M^-1 sum (a_j-d)^+.  Its slopes are
    -#{a_j>d}/M. Every distinct sample is a kink, so tangent cuts at
    all kinks plus the left box endpoint give the EXACT epigraph.
    """
    import gurobipy as gp
    bank = np.asarray(bank, float)
    outputs=[]
    for i, d in enumerate(durations):
        samples=np.sort(bank[:,i]); distinct=np.unique(samples)
        f=m.addVar(lb=0., name=f"{prefix}_stoploss_{i}")
        for a in distinct:
            v=float(np.maximum(samples-a, 0.).mean())
            slope=-float((samples>a).mean())
            m.addConstr(f >= v+slope*(d-float(a)))
        # Below the smallest sample slope -1. Above the largest, f >= 0.
        m.addConstr(f >= float(samples.mean())-d)
        outputs.append(f)
    return gp.quicksum(outputs)


def _duration_variables(m, c, base, H, coeff, bounds, prefix):
    """Exact clipping with disjunctive indicators; no unproved PWL relaxation."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.planner import reachable
    lower, upper = reachable(c.booked, .8, 30.)
    d=[]
    for i in range(c.n):
        lo, hi=float(lower[i]),float(upper[i]); initial=float(base[i])
        v=m.addVar(lb=lo, ub=hi, name=f"{prefix}_duration_{i}")
        expr=initial+gp.quicksum(float(H[i,j])*coeff[j] for j in range(H.shape[1]))
        # Indicators implement q = clip(expr,lo,hi) exactly at boundaries.
        indicators=m.addVars(3, vtype=GRB.BINARY, name=f"{prefix}_regime_{i}")
        m.addConstr(indicators.sum()==1)
        m.addGenConstrIndicator(indicators[0],True,expr<=lo)
        m.addGenConstrIndicator(indicators[0],True,v==lo)
        m.addGenConstrIndicator(indicators[1],True,expr>=lo)
        m.addGenConstrIndicator(indicators[1],True,expr<=hi)
        m.addGenConstrIndicator(indicators[1],True,v==expr)
        m.addGenConstrIndicator(indicators[2],True,expr>=hi)
        m.addGenConstrIndicator(indicators[2],True,v==hi)
        d.append(v)
    return d,lower,upper


def _vf_component(m, c, base, d0, H, coeff, bank, lam, *, index):
    """One full VF component, no dropped schedules or scenario samples."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.anchored_weekly_solver import _model
    # _model creates another model; this routine builds every component on the
    # COMMON shared-policy model so only the five coefficients couple weeks.
    arcs=[(i,j) for i in range(c.n) for j in c.allowed[int(c.surgeon[i])]]
    x=m.addVars(arcs,vtype=GRB.BINARY,name=f"x_{index}")
    for i in range(c.n):
        m.addConstr(gp.quicksum(x[i,j] for j in c.allowed[int(c.surgeon[i])])==1)
    for s,days in enumerate(c.required_days):
        cases=np.flatnonzero(c.surgeon==s)
        for weekday in days:
            m.addConstr(gp.quicksum(x[int(i),j] for i in cases
                for j in c.allowed[s] if c.slots[j][0]==weekday)>=1)
    byroom=[[i for i,j in arcs if j==k] for k in range(c.k)]
    used=m.addVars(c.k,vtype=GRB.BINARY,name=f"used_{index}")
    for j,cases in enumerate(byroom):
        total=gp.quicksum(x[i,j] for i in cases)
        m.addConstr(total>=used[j]);m.addConstr(total<=len(cases)*used[j])
    d,lo,hi=_duration_variables(m,c,base,H,coeff,None,f"t{index}")
    load=_product_loads(m,c,x,used,byroom,d,(lo,hi))
    ot=[]
    for j in range(c.k):
        v=m.addVar(lb=0., name=f"t{index}_ot{j}")
        m.addConstr(v>=load[j]-CAP)
        ot.append(v)
    sl=_stoploss(m,d,bank,f"t{index}")
    penalty=gp.quicksum((d[i]-float(d0[i]))*(d[i]-float(d0[i])) for i in range(c.n))
    obj=TURN*used.sum()+KAPPA*gp.quicksum(ot)+KAPPA*sl+lam/2*penalty
    return obj, (c,x,d)


def vf_shared_miqp(items, data, H, beta, pools, *, seconds, threads=8,
                    coefficient_radius=32., lam=.001, bank_seed=20261020):
    """Single globally bounded MIP/MIQP on ALL training components.

    This has 5 coefficient variables and no schedule-library restriction.
    At time limit ObjBound remains a valid global lower bound of this model;
    a feasible objective gives an upper bound. The published planner's hash
    cannot enter VF because VF minimizes its *primary value* Q(d).
    """
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.advice_comparison import reviewed, scenarios
    start=time.monotonic()
    if not items: raise ValueError("Empty training set")
    model=gp.Model("vf_shared_all_training_components")
    model.Params.OutputFlag=1
    model.Params.Threads=int(threads)
    model.Params.TimeLimit=float(seconds)
    model.Params.MIPGap=0
    model.Params.MIPGapAbs=0
    model.Params.NonConvex=2
    # base/coefficients are exactly the same five-feature representation as STAT.
    coef=model.addVars(5,lb=-float(coefficient_radius),ub=float(coefficient_radius),name="theta")
    objective=gp.QuadExpr(); used_cases=0
    details=[]
    for t,(ck,c,rows) in enumerate(items):
        d0=reviewed(c.booked,data["X"][rows]@data["case_w"])
        bank=scenarios(c,d0,pools,bank_seed,count=50)
        off=d0+H[rows]@np.asarray(beta,float)
        local, vars_=_vf_component(model,c,off,d0,H[rows],coef,bank,lam,index=t)
        objective+=local
        details.append((ck,vars_,bank,d0,H[rows]))
        used_cases+=c.n
    model.setObjective(objective/used_cases,GRB.MINIMIZE)
    remaining=seconds-(time.monotonic()-start)
    if remaining<=0:
        model.dispose()
        return Certificate(None,None,None,False,'MODEL_BUILD_TIME_LIMIT',
                           time.monotonic()-start)
    model.Params.TimeLimit=remaining
    model.optimize()
    rec=_safe_certificate(model,start,"shared VF")
    if model.SolCount:
        from surgery.weekly_planner import _assignment
        from surgery.planner import reachable
        theta=np.asarray([coef[j].X for j in range(5)],float)
        rec.reviewed=theta.tolist()
        reconstructed=0.
        for ck, (component, x, d),bank,d0,h_rows in details:
            durations=np.asarray([v.X for v in d],float)
            low,high=reachable(component.booked,.8,30.)
            independently=np.clip(d0+h_rows@(np.asarray(beta,float)+theta),low,high)
            if not np.allclose(durations,independently,atol=1e-4,rtol=0):
                raise AssertionError(f'Shared clipping reconstruction failed: {ck}')
            assign=_assignment(component,x)
            reconstructed+=vf_objective(component,assign,durations,bank,d0,lam)
        reconstructed/=used_cases
        if abs(reconstructed-rec.upper)>max(.01,1e-5*abs(reconstructed)):
            raise AssertionError(f'Shared VF primal mismatch {reconstructed} != {rec.upper}')
    model.dispose()
    return rec


def vf_per_week_milp(c, bank, *, seconds, threads=1, d0=None, lam=.001):
    """Globally bounded FULL per-instance VF (not a coordinate grid search)."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.planner import reachable
    start=time.monotonic()
    m,x,used,byroom=_setup(c,seconds,threads,"vf_per_week")
    lo,hi=reachable(c.booked,.8,30.)
    d=[m.addVar(lb=float(lo[i]),ub=float(hi[i]),name=f"d{i}") for i in range(c.n)]
    load=_product_loads(m,c,x,used,byroom,d,(lo,hi))
    ot=[]
    for j in range(c.k):
        v=m.addVar(lb=0.,name=f"ot{j}");m.addConstr(v>=load[j]-CAP);ot.append(v)
    stop=_stoploss(m,d,bank,"pw")
    penalty=0
    if d0 is not None:
        penalty=lam/2*gp.quicksum((d[i]-float(d0[i]))**2 for i in range(c.n))
    m.setObjective(TURN*used.sum()+KAPPA*gp.quicksum(ot)+KAPPA*stop+penalty,GRB.MINIMIZE)
    m.optimize()
    rec=_safe_certificate(m,start,"per-week VF",c,x)
    if m.SolCount:
        rec.reviewed=[float(v.X) for v in d]
        from surgery.weekly_planner import _assignment
        rec.assignment=tuple(map(int,_assignment(c,x)))
        actual=vf_objective(c,rec.assignment,rec.reviewed,bank,d0,lam)
        if abs(actual-rec.upper)>max(.001,1e-5*abs(actual)):
            raise AssertionError("VF objective mismatch")
    m.dispose()
    return rec


def vf_fixed_plan_lp(c, z, bank, *, seconds, d0=None, lam=0.):
    """Globally optimal quantile fill for ONE given schedule (convex QP)."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.planner import reachable
    from surgery.weekly_planner import loads
    start=time.monotonic()
    m=gp.Model("quantile_fill_fixed_plan")
    m.Params.OutputFlag=0;m.Params.TimeLimit=float(seconds)
    lo,hi=reachable(c.booked,.8,30.)
    d=[m.addVar(lb=float(lo[i]),ub=float(hi[i]),name=f"d{i}") for i in range(c.n)]
    z=np.asarray(z,int);used=set(map(int,z));ot=[]
    for j in used:
        indices=np.flatnonzero(z==j)
        overtime=m.addVar(lb=0.,name=f"ot{j}")
        m.addConstr(overtime>=gp.quicksum(d[i] for i in indices)+TURN*(len(indices)-1)-CAP)
        ot.append(overtime)
    sl=_stoploss(m,d,bank,"fixed")
    penalty=0. if d0 is None else lam/2*gp.quicksum((d[i]-float(d0[i]))**2 for i in range(c.n))
    m.setObjective(TURN*len(used)+KAPPA*gp.quicksum(ot)+KAPPA*sl+penalty,GRB.MINIMIZE)
    m.optimize()
    rec=_safe_certificate(m,start,"quantile fill")
    if m.SolCount:rec.reviewed=[float(x.X) for x in d]
    m.dispose()
    return rec


def expected_gap_mip(c,d,bank,*,seconds,threads=1,gamma=1.):
    """Exact MILP adversary for expected GAP with the TRUE value-dependent tolerance.

    Uses all marginal/joint bank scenarios. Each sample-room overtime hinge is
    represented exactly (including in MAX), never as a free epigraph.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.anchored_weekly_solver import solve_week
    from surgery.weekly_planner import _objective_tolerance
    start=time.monotonic()
    ans=solve_week(c,d,seconds=seconds,threads=threads,tie=False)
    if not ans.complete or ans.value is None:
        return Certificate(None,None,None,False,"PRIMARY_UNCERTIFIED",time.monotonic()-start)
    q=ans.value - CAP*c.k + float(np.sum(d))+TURN*c.n
    eps=_objective_tolerance(ans.value)
    m,x,used,byroom=_setup(c,seconds,threads,"expected_gap")
    zero=m.addVar(lb=0,ub=0,name="zero")
    def exact_overtime(a,prefix):
        ot=[]
        for j,cases in enumerate(byroom):
            expr=gp.quicksum((float(a[i])+TURN)*x[i,j] for i in cases)-TURN*used[j]-CAP
            excess=m.addVar(lb=-CAP, name=f"{prefix}_excess_{j}")
            m.addConstr(excess==expr)
            v=m.addVar(lb=0,name=f"{prefix}_ot_{j}")
            m.addGenConstrMax(v,[excess,zero])
            ot.append(v)
        return TURN*used.sum()+KAPPA*gp.quicksum(ot)
    pred=exact_overtime(np.asarray(d,float),"pred")
    g=gp.LinExpr()
    for s,a in enumerate(np.asarray(bank,float)):
        g+=exact_overtime(a,f"actual{s}")/len(bank)
    excess=m.addVar(lb=0,name="outside_tie_penalty")
    m.addConstr(excess>=pred-q-eps)
    m.setObjective(g-float(gamma)*excess,GRB.MAXIMIZE)
    m.optimize()
    # Maximization has the reverse dual direction.
    status=str(m.Status); has=m.SolCount>0
    feasible=float(m.ObjVal) if has else None
    bound=float(m.ObjBound) if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT,GRB.SUBOPTIMAL) else None
    if feasible is not None and bound is not None and feasible>bound+1e-4:
        raise AssertionError("Expected GAP upper dual bound below feasible adversary")
    rec=Certificate(feasible,feasible,bound,m.Status==GRB.OPTIMAL,status,
                    time.monotonic()-start)
    if has:
        from surgery.weekly_planner import _assignment
        rec.assignment=tuple(map(int,_assignment(c,x)))
    m.dispose()
    return rec


def saa_mip(c, bank, *, seconds, threads=1):
    """Unrestricted stochastic schedule: deterministic LB for direct advice.

    Used as a RELAXATION: an SAA lower bound is always a valid lower bound on
    every reviewed-policy objective for this fixed scenario bank. It does not
    assert that the SAA assignment can be induced by advice.
    """
    import gurobipy as gp
    from gurobipy import GRB
    start=time.monotonic()
    m,x,used,byroom=_setup(c,seconds,threads,"stochastic_planner")
    obj=TURN*used.sum()
    for s,a in enumerate(np.asarray(bank,float)):
        for j,cases in enumerate(byroom):
            excess=gp.quicksum((float(a[i])+TURN)*x[i,j] for i in cases)-TURN*used[j]-CAP
            ot=m.addVar(lb=0,name=f"s{s}_j{j}_overtime")
            m.addConstr(ot>=excess)
            obj+=(KAPPA/len(bank))*ot
    m.setObjective(obj,GRB.MINIMIZE)
    m.optimize()
    cert=_safe_certificate(m,start,"SAA",c,x)
    if m.SolCount:
        truth=expected_reduced(c,cert.assignment,bank)
        if abs(truth-cert.value)>max(1e-3,1e-6*abs(truth)):
            raise AssertionError(f"SAA cost reconstruction: {truth} != {cert.value}")
    m.dispose()
    return cert


def second_best_primary(c, d, selected, *, seconds, threads=1):
    """Certified LOWER bound on the primary cost of any other schedule."""
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.weekly_planner import _cost, _assignment
    start=time.monotonic()
    m,x,used,byroom=_setup(c,seconds,threads,"second_best_planner")
    m.addConstr(gp.quicksum(x[i,int(selected[i])] for i in range(c.n)) <= c.n-1)
    obj=_cost(m,x,used,byroom,d,exact=False,name="primary")
    m.setObjective(obj,GRB.MINIMIZE)
    m.optimize()
    if m.Status==GRB.INFEASIBLE:
        ans=Certificate(float('inf'),float('inf'),float('inf'),True,'ONE_FEASIBLE_SCHEDULE',time.monotonic()-start)
    else:
        ans=_safe_certificate(m,start,"second-best",c,x)
    m.dispose()
    return ans


def regret_bounds(c, bank, d, d0, *, seconds, seeds=(0,1,2), lam=.001):
    """Certified deployed upper bound and unconditional SAA lower bound.

    The full-case planner must be certified. Unfinished SAA still contributes
    its dual lower bound (or zero, if the MILP generated no finite bound).
    """
    from surgery.anchored_weekly_solver import solve_week
    from surgery.weekly_planner import evaluate
    z=[];start=time.monotonic()
    for seed in seeds:
        p=solve_week(c,d,seconds=seconds,threads=1,tie=True,hash_seed=seed)
        if not p.complete or p.assignment is None:
            return dict(complete=False,status=f"PLANNER_{p.status}",lower=None,upper=None,seconds=time.monotonic()-start)
        z.append(tuple(map(int,p.assignment)))
    penalty=lam/2*np.sum((np.asarray(d)-np.asarray(d0))**2)
    upper=float(np.mean([expected_reduced(c,a,bank) for a in z])+penalty)
    lower=saa_mip(c,bank,seconds=seconds)
    lval=max(0.,float(lower.lower or 0.))
    if lval>upper+1e-4:
        raise AssertionError("Unrestricted SAA lower bound exceeds deployed feasible value")
    return dict(complete=True, status='CERTIFIED_DEPLOYMENT', lower=lval,
                upper=upper, assignments=[list(a) for a in z],
                saa=lower.to_json(), seconds=time.monotonic()-start)


def sharp_margin_bound(c, bank, d_center, boxlo, boxhi, *, seconds, seeds=(0,1,2),deadline=None):
    """Certified fixed deployed schedules throughout a full duration box.

    This is intentionally CONSERVATIVE: it never treats a hash tie as stable
    without a strict primary gap certificate.
    """
    from surgery.anchored_weekly_solver import solve_week
    from surgery.weekly_planner import _objective_tolerance
    eps=KAPPA*np.maximum(np.abs(boxlo-d_center),np.abs(boxhi-d_center)).sum()
    def limit():
        return seconds if deadline is None else min(seconds,max(0.,deadline-time.monotonic()))
    plans=[]
    for seed in seeds:
        if limit()<=0:return None
        ans=solve_week(c,d_center,seconds=limit(),threads=1,tie=True,hash_seed=seed)
        if not ans.complete or ans.assignment is None:return None
        plans.append(tuple(map(int,ans.assignment)))
    if len(set(plans))!=1:
        return None
    if limit()<=0:return None
    ans=solve_week(c,d_center,seconds=limit(),threads=1,tie=False)
    if not ans.complete or ans.assignment is None:return None
    selected=plans[0]
    if tuple(map(int,ans.assignment))!=selected:return None
    if limit()<=0:return None
    runner=second_best_primary(c,d_center,selected,seconds=limit())
    if runner.lower is None:return None
    if runner.lower-ans.value > .1+eps+1e-7:
        return selected
    return None


def exact_box_policy(c, bank, d0, *, seconds=900., solve_seconds=120.,
                     nodes=100, seeds=(0,1,2), lam=.001, restricted=None,
                     saa_reference=None):
    """Global-bound optimization over ALL casewise reviewed durations.

    `restricted` optionally limits target schedules (ANCHORED restricted step).
    The master lower bound is scenario SAA or a certified box-invariant schedule.
    No component or dimension is silently omitted; returns dual/primal gap on
    timeout. With a dense set of hash ties it may fail to close even small gaps.
    """
    import heapq
    from surgery.planner import reachable
    from surgery.anchored_weekly_solver import solve_week
    lo,hi=map(lambda x: np.asarray(x,float),reachable(c.booked,.8,30.))
    base=np.clip(np.asarray(d0,float),lo,hi)
    clock=time.monotonic()
    deadline=clock+seconds
    def limit():
        return min(solve_seconds,max(0.,deadline-time.monotonic()))
    saa=(saa_reference if saa_reference is not None else
         saa_mip(c,bank,seconds=max(.001,limit())))
    global_lb=max(0.,float(saa.lower or 0.))
    if restricted == 'case+saa':
        from surgery.anchored_weekly_solver import solve_week
        incumbent=[]
        for seed in seeds:
            if limit()<=0:
                return dict(lower=global_lb,upper=None,complete=False,
                            status='TIME_LIMIT',nodes=0,seconds=time.monotonic()-clock,
                            reviewed=None,assignments=None,saa=saa.to_json())
            result=solve_week(c,base,seconds=limit(),threads=1,tie=True,hash_seed=seed)
            if not result.complete:
                return dict(lower=global_lb,upper=None,complete=False,
                            status='UNCERTIFIED_CASE',nodes=0,seconds=time.monotonic()-clock,
                            reviewed=None,assignments=None,saa=saa.to_json())
            incumbent.append(tuple(map(int,result.assignment)))
        target=tuple(map(int,saa.assignment)) if saa.assignment is not None else None
        # Per-hash admissible targets; exact planner still decides which is deployed.
        target_set=tuple({incumbent[j]} | ({target} if target is not None else set())
                         for j in range(len(seeds)))
    elif restricted is None:
        target_set=None
    else:
        common=set(tuple(map(int,z)) for z in restricted)
        target_set=tuple(common for _ in seeds)
    best=np.inf;bestd=None;bestz=None
    num=0;uid=0;queue=[(global_lb,uid,lo,hi)]
    retired_lb=float('inf')
    def penalty_min(a,b):
        return lam/2*float(np.square(np.clip(base,a,b)-base).sum())
    def examine(d):
        nonlocal best,bestd,bestz
        candidates=[]
        for seed in seeds:
            if limit()<=0:return None
            r=solve_week(c,d,seconds=limit(),threads=1,tie=True,hash_seed=seed)
            if not r.complete:return None
            candidates.append(tuple(map(int,r.assignment)))
        if target_set is not None and any(z not in target_set[j]
                                          for j,z in enumerate(candidates)):
            return candidates
        obj=float(np.mean([expected_reduced(c,z,bank) for z in candidates])
                  +lam/2*np.square(d-base).sum())
        if obj<global_lb-1e-5:
            raise AssertionError('SAA lower bound exceeds a certified deployed feasible policy')
        if obj<best:
            best,bestd,bestz=obj,d.copy(),candidates
        return candidates
    examine(base)
    while queue and num<nodes and time.monotonic()-clock<seconds:
        lb,_,a,b=heapq.heappop(queue)
        if lb>=best-1e-7:
            retired_lb=min(retired_lb,lb)
            continue
        num+=1
        mid=(a+b)/2
        outcome=examine(mid)
        if outcome is None:
            # Cannot certify deployment here; node remains unresolved.
            uid+=1;heapq.heappush(queue,(lb,uid,a,b))
            break
        stable=sharp_margin_bound(c,bank,mid,a,b,seconds=solve_seconds,
                                  seeds=seeds,deadline=deadline)
        if stable is not None:
            if target_set is not None and any(stable not in group for group in target_set):
                continue  # Certified infeasible for the restricted policy class.
            # Every point in this box deploys exactly the stable assignment.
            d_star=np.clip(base,a,b)
            exact=expected_reduced(c,stable,bank)+lam/2*np.square(d_star-base).sum()
            if exact<best:
                best,bestd,bestz=exact,d_star,[stable]*len(seeds)
            retired_lb=min(retired_lb,exact)
            continue
        widths=b-a
        if widths.max()<1e-6:
            uid+=1;heapq.heappush(queue,(lb,uid,a,b))
            break
        index=int(np.argmax(widths))
        for side in (0,1):
            na,nb=a.copy(),b.copy()
            if side==0:nb[index]=mid[index]
            else:na[index]=mid[index]
            # Valid but sometimes weak lower bound; no false proof from a library.
            clb=global_lb+penalty_min(na,nb)
            if clb<best-1e-7:
                uid+=1;heapq.heappush(queue,(clb,uid,na,nb))
            else:
                retired_lb=min(retired_lb,clb)
    final_lb=min([x[0] for x in queue]+[retired_lb,best])
    gap=best-final_lb if isfinite(best) else float('inf')
    status=('OPTIMAL_CERTIFIED' if isfinite(best) and gap<=1e-9
            else 'GAP_CERTIFIED' if isfinite(best) and gap<=1e-7
            else 'TIME_OR_NODE_LIMIT')
    return dict(lower=float(final_lb),upper=None if not isfinite(best) else float(best),
                complete=status in ('OPTIMAL_CERTIFIED','GAP_CERTIFIED'),status=status,nodes=num,
                tolerance_fathomed_lower=None if not isfinite(retired_lb) else float(retired_lb),
                seconds=time.monotonic()-clock,
                reviewed=None if bestd is None else bestd.tolist(),
                assignments=None if bestz is None else [list(z) for z in bestz],
                saa=saa.to_json())