"""Global lower/upper bound search for direct and restricted-target SHARED.

All components contribute to every node.  This is deliberately a bound-based
algorithm, not a claim that a finite schedule library is globally complete.
A cutoff returns a numerical gap (which can be large), never a false optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
import json
import time
from pathlib import Path

import numpy as np

from surgery.theory_exact import (expected_reduced, saa_mip,
    coefficient_box_images, sharp_margin_bound, KAPPA)
from surgery.advice_comparison import reviewed, scenarios
from surgery.planner import reachable
from surgery.anchored_weekly_solver import solve_week


@dataclass
class Problem:
    key: str
    week: object
    d0: np.ndarray
    H: np.ndarray
    bank: np.ndarray
    lower: float
    targets: tuple[set, ...] | None = None


def _save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,allow_nan=False));p.replace(path)


def build_problems(items,data,H,beta,pools,output,*,seconds=120.,
                   seeds=(0,1,2),target_restricted=False):
    """Compute full-component SAA dual lower bounds; cache every certified result."""
    folder=Path(output)/'saa_bounds';folder.mkdir(parents=True,exist_ok=True)
    problems=[]
    for j,(key,c,rows) in enumerate(items):
        d0=reviewed(c.booked,data['X'][rows]@data['case_w'])
        bank=scenarios(c,d0,pools,20261020,count=50)
        p=folder/(key+'.json')
        if p.exists(): rec=json.loads(p.read_text())
        else:
            cert=saa_mip(c,bank,seconds=seconds)
            rec=cert.to_json()
            _save(p,rec)
        lb=max(0.,float(rec['lower'] if rec['lower'] is not None else 0.))
        targets=None
        if target_restricted:
            starting=np.clip(d0+H[rows]@beta,*reachable(c.booked,.8,30.))
            own=[]
            for seed in seeds:
                sol=solve_week(c,starting,seconds=seconds,tie=True,hash_seed=seed)
                if not sol.complete:raise RuntimeError('Uncertified target baseline '+key)
                own.append(tuple(map(int,sol.assignment)))
            candidate=tuple(map(int,rec['assignment'])) if rec.get('assignment') else None
            targets=tuple(set([z]+([candidate] if candidate is not None else []))
                          for z in own)
            # Restricted schedule minimum is a valid stronger lower bound.
            lb=float(np.mean([min(expected_reduced(c,z,bank) for z in group)
                              for group in targets]))
        problems.append(Problem(key,c,d0,H[rows],bank,lb,targets))
    return problems


def _evaluate(problems,theta,beta,seconds,lam=.001,seeds=(0,1,2),deadline=None):
    total=0.; n=0
    for p in problems:
        c=p.week
        lo,hi=reachable(c.booked,.8,30.)
        d=np.clip(p.d0+p.H@(beta+theta),lo,hi)
        zs=[]
        for seed in seeds:
            budget=seconds if deadline is None else min(seconds,max(0.,deadline-time.monotonic()))
            if budget<=0:return None
            sol=solve_week(c,d,seconds=budget,tie=True,hash_seed=seed)
            if not sol.complete:return None
            zs.append(tuple(map(int,sol.assignment)))
        if p.targets is not None and any(z not in p.targets[i]
                                          for i,z in enumerate(zs)):
            return None
        total+=float(np.mean([expected_reduced(c,z,p.bank) for z in zs])
                     +lam/2*np.square(d-p.d0).sum())
        n+=c.n
    return total/n


def _node_lower(problems,boxlo,boxhi,beta,seconds,lam,seeds,
                *,screen_stability=False,deadline=None):
    total=0.; n=0;stable_count=0
    center=(boxlo+boxhi)/2
    for p in problems:
        c=p.week
        lo,hi=reachable(c.booked,.8,30.)
        dmin,dmax=coefficient_box_images(p.d0+p.H@beta,p.H,boxlo,boxhi,lo,hi)
        closest=np.clip(p.d0,dmin,dmax)
        penalty=lam/2*np.square(closest-p.d0).sum()
        component_lb=p.lower
        if screen_stability:
            if all(len(c.allowed[int(s)])==1 for s in c.surgeon):
                z=tuple(map(int,c.original))
            else:
                d=np.clip(p.d0+p.H@(beta+center),lo,hi)
                z=sharp_margin_bound(c,p.bank,d,dmin,dmax,seconds=seconds,
                                     seeds=seeds,deadline=deadline)
            if z is not None:
                if p.targets is not None and any(z not in T for T in p.targets):
                    return float('inf'),stable_count
                component_lb=expected_reduced(c,z,p.bank)
                stable_count+=1
        total+=component_lb+penalty
        n+=c.n
    return total/n,stable_count


def global_shared_search(problems,beta,*,out,seconds=7200.,
                         solve_seconds=120.,max_nodes=40,radius=32.,
                         start_theta=None,lam=.001,seeds=(0,1,2),
                         gap_tolerance=1e-3):
    """Certified continuous coefficient-space branch-and-bound.

    Lower bound: independent SAA MIP + exact boxwise regularizer lower bound;
    optionally strengthen with a certified stable planner.  Upper bound:
    every component deployed using the full two-hash exact planner.  No
    global-optimality claim is made unless the reported absolute gap closes.
    """
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+seconds;theta=np.zeros(len(beta))
    if start_theta is not None:theta=np.clip(np.asarray(start_theta,float),-radius,radius)
    midpoint=_evaluate(problems,theta,np.asarray(beta),solve_seconds,lam,seeds,deadline=deadline)
    baseline=_evaluate(problems,np.zeros_like(theta),np.asarray(beta),solve_seconds,lam,seeds,deadline=deadline)
    if baseline is None:
        raise RuntimeError('Cannot certify baseline for global search')
    if midpoint is None or midpoint>=baseline:
        theta=np.zeros_like(theta);best=baseline
    else:best=midpoint
    best_theta=theta.copy()
    lower=np.full(len(beta),-float(radius));upper=np.full(len(beta),float(radius))
    global_lb,_=_node_lower(problems,lower,upper,np.asarray(beta),solve_seconds,
                            lam,seeds,screen_stability=False,deadline=deadline)
    queue=[];serial=0;heappush(queue,(global_lb,serial,lower,upper))
    progress=[];processed=0
    retired_lb=float('inf')  # Bounds of tolerance-fathomed regions remain valid.
    while queue and processed<max_nodes and time.monotonic()-start<seconds:
        lb,_,a,b=heappop(queue)
        if lb>=best-gap_tolerance:
            retired_lb=min(retired_lb,lb)
            continue
        processed+=1
        # The expensive stability oracle is only used on promising tight boxes.
        width=float(np.max(b-a))
        use_stability=width<max(1.,radius/2)
        tighter,stable=_node_lower(problems,a,b,np.asarray(beta),solve_seconds,
                            lam,seeds,screen_stability=use_stability,deadline=deadline)
        if tighter>=best-gap_tolerance:
            retired_lb=min(retired_lb,tighter)
            continue
        center=(a+b)/2
        val=_evaluate(problems,center,np.asarray(beta),solve_seconds,lam,seeds,deadline=deadline)
        if val is not None and val<best:
            best=val;best_theta=center.copy()
        impact=np.zeros(len(beta))
        for p in problems:impact+=np.abs(p.H).sum(axis=0)
        split=int(np.argmax((b-a)*impact))
        if b[split]-a[split]<=1e-7:
            serial+=1;heappush(queue,(tighter,serial,a,b));break
        for side in (0,1):
            aa,bb=a.copy(),b.copy()
            if side==0:bb[split]=center[split]
            else:aa[split]=center[split]
            sub,_=_node_lower(problems,aa,bb,np.asarray(beta),solve_seconds,
                              lam,seeds,screen_stability=False)
            if sub<best-gap_tolerance:
                serial+=1;heappush(queue,(sub,serial,aa,bb))
            else:
                retired_lb=min(retired_lb,sub)
        current_lb=min([q[0] for q in queue]+[retired_lb,best])
        progress.append(dict(node=processed,lower=current_lb,upper=best,
                             stable_on_node=stable,remaining=len(queue),
                             elapsed=time.monotonic()-start))
        _save(output/'progress.json',dict(history=progress,theta=best_theta.tolist()))
    final_lb=min([q[0] for q in queue]+[retired_lb,best])
    # Tolerance-fathomed regions are NOT exact: keep their dual bounds.
    result=dict(theta=best_theta.tolist(),lower=float(final_lb),upper=float(best),
                gap=float(max(0.,best-final_lb)),nodes=processed,
                seconds=time.monotonic()-start,
                status=('OPTIMAL_CERTIFIED' if best-final_lb<=1e-8
                         else 'GAP_CERTIFIED' if best-final_lb<=gap_tolerance+1e-10
                         else 'BOUNDED_UNRESOLVED'),
                 tolerance_fathomed_lower=None if not np.isfinite(retired_lb) else float(retired_lb),
                allowed_schedules='incumbent + SAA target per seed' if problems[0].targets is not None else 'all feasible',
                guarantee='finite-sample scenario objective only; all components included')
    _save(output/'result.json',result)
    return result