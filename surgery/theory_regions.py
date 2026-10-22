"""Global lower/upper bound search for direct and restricted-target SHARED.

All components contribute to every node.  This is deliberately a bound-based
algorithm, not a claim that a finite schedule library is globally complete.
A cutoff returns a numerical gap (which can be large), never a false optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from heapq import heappush, heappop, heapify
import hashlib
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


def _saa_task(job):
    key,week,bank,seconds=job
    return key,saa_mip(week,bank,seconds=seconds).to_json()


def _target_task(job):
    key,week,starting,seeds,seconds=job
    selected=[]
    for seed in seeds:
        sol=solve_week(week,starting,seconds=seconds,threads=1,tie=True,hash_seed=seed)
        if not sol.complete or sol.assignment is None:
            raise RuntimeError(f'Uncertified target baseline {key} seed={seed}: {sol.status}')
        selected.append(list(map(int,sol.assignment)))
    return key,selected


def _parallel_preprocess(work, worker, folder, seconds, cores, deadline):
    """Checkpoint finished solver tasks and stop launching new jobs at deadline."""
    from concurrent.futures import wait,FIRST_COMPLETED
    submitted=0;completed=0;failed=0
    if not work:
        return dict(submitted=0,completed=0,failed=0)
    with ProcessPoolExecutor(max_workers=min(cores,len(work)),
                             mp_context=get_context('spawn')) as pool:
        source=iter(work);active={}
        def launch():
            nonlocal submitted
            if time.monotonic()>=deadline:return False
            try:item=next(source)
            except StopIteration:return False
            future=pool.submit(worker,item)
            active[future]=item
            submitted+=1
            return True
        for _ in range(min(2*cores,len(work))):
            if not launch():break
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                item=active.pop(future)
                ck=item[0]
                try:
                    name,result=future.result()
                    if name!=ck:raise AssertionError('Preprocessing key mismatch')
                    _save(folder/(ck+'.json'),result)
                    completed+=1
                except (RuntimeError,ValueError) as exc:
                    # A failed target is not silently declared inducible.
                    _save(folder/(ck+'.json'),dict(
                           status='UNRESOLVED',error=str(exc)))
                    failed+=1
                if completed%100==0 and completed:
                    print('Preprocessing completed:',completed,'/',len(work),flush=True)
                launch()
    return dict(submitted=submitted,completed=completed,failed=failed)


def build_problems(items,data,H,beta,pools,output,*,seconds=90.,
                   seeds=(0,1,2),target_restricted=False,cores=1,
                   stage_seconds=7200.):
    """Certified SAA dual preprocessing under a finite shared wall budget.

    Unfinished SAA tasks contribute zero, always a VALID (weak) lower bound
    because the reduced cost is nonnegative. No missing target schedule is
    misrepresented as a feasible inductive alternative.
    """
    from surgery.planner import reachable
    start=time.monotonic();deadline=start+stage_seconds
    root=Path(output);folder=root/'saa_bounds'
    folder.mkdir(parents=True,exist_ok=True)
    prepared=[];missing=[]
    base=data['X']@data['case_w']
    for ck,c,rows in items:
        d0=reviewed(c.booked,base[rows])
        bank=scenarios(c,d0,pools,20261020,count=50)
        file=folder/(ck+'.json')
        prepared.append((ck,c,rows,d0,bank,file))
        if not file.exists():missing.append((ck,c,bank,seconds))
    stats=_parallel_preprocess(missing,_saa_task,folder,seconds,cores,deadline)
    target_stats=dict(submitted=0,completed=0,failed=0)
    target_folder=root/'target_plans'
    if target_restricted:
        target_folder.mkdir(parents=True,exist_ok=True)
        missing_targets=[]
        for ck,c,rows,d0,bank,file in prepared:
            dest=target_folder/(ck+'.json')
            if dest.exists():continue
            start_d=np.clip(d0+H[rows]@beta,*reachable(c.booked,.8,30.))
            missing_targets.append((ck,c,start_d,seeds,seconds))
        target_stats=_parallel_preprocess(missing_targets,_target_task,
                                          target_folder,seconds,cores,deadline)
    problems=[];unresolved_saa=0;unresolved_targets=0
    for ck,c,rows,d0,bank,file in prepared:
        rec=json.loads(file.read_text()) if file.exists() else dict(
             status='PREPROCESS_WALL_LIMIT',lower=None,assignment=None)
        if rec.get('lower') is None:unresolved_saa+=1
        lb=max(0.,float(rec['lower'] or 0.))
        targets=None
        if target_restricted:
            targetfile=target_folder/(ck+'.json')
            own=json.loads(targetfile.read_text()) if targetfile.exists() else dict(
                status='PREPROCESS_WALL_LIMIT')
            if not isinstance(own,list) or len(own)!=len(seeds):
                unresolved_targets+=1
                continue
            candidate=(tuple(map(int,rec['assignment'])) if rec.get('assignment') else None)
            targets=tuple(set([tuple(map(int,z))]+([candidate] if candidate else []))
                          for z in own)
            lb=float(np.mean([min(expected_reduced(c,z,bank) for z in group)
                              for group in targets]))
        problems.append(Problem(ck,c,d0,H[rows],bank,lb,targets))
    _save(root/('target_preprocessing.json' if target_restricted else 'saa_preprocessing.json'),
          dict(components=len(items),ready_components=len(problems),
               unresolved_saa=unresolved_saa,unresolved_targets=unresolved_targets,
               saa_tasks=stats,target_tasks=target_stats,
               elapsed_seconds=time.monotonic()-start,wall_budget=stage_seconds,
               threads_per_solve=1,parallel_workers=min(cores,len(items))))
    if target_restricted and unresolved_targets:
        raise RuntimeError(f'{unresolved_targets} restricted target sets were uncertified; '
                           'do not claim an ANCHORED_TARGET optimum')
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
    final_result=output/'result.json'
    if final_result.exists():
        # A completed time-budgeted search is a frozen fitted method.
        # Continuing it silently would invalidate cached evaluation records.
        return json.loads(final_result.read_text())
    start=time.monotonic();deadline=start+seconds
    checkpoint=output/'search_state.json'
    signature=hashlib.sha256(json.dumps(dict(
        components=[(p.key,p.week.n,p.targets is not None) for p in problems],
        beta=np.asarray(beta).tolist(),radius=radius,lam=lam,
        seeds=list(seeds),gap_tolerance=gap_tolerance),
        sort_keys=True).encode()).hexdigest()

    def save_state():
        # The popped node is written out only after being fully processed.
        # A forced interruption during work replays the previous checkpoint.
        _save(checkpoint,dict(signature=signature,incumbent=best,
              best_theta=best_theta.tolist(),processed=processed,
              serial=serial,retired_lb=None if np.isinf(retired_lb) else retired_lb,
              queue=[dict(lower=q[0],serial=q[1],
                          a=q[2].tolist(),b=q[3].tolist()) for q in queue],
              history=progress))

    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get('signature')!=signature:
            raise RuntimeError('Global-search checkpoint incompatible with model')
        best=float(saved['incumbent'])
        best_theta=np.asarray(saved['best_theta'],float)
        processed=int(saved['processed']);serial=int(saved['serial'])
        retired_lb=float('inf') if saved['retired_lb'] is None else float(saved['retired_lb'])
        progress=saved['history']
        queue=[(float(q['lower']),int(q['serial']),
                np.asarray(q['a'],float),np.asarray(q['b'],float))
               for q in saved['queue']]
        heapify(queue)
    else:
        theta=np.zeros(len(beta))
        if start_theta is not None:
            theta=np.clip(np.asarray(start_theta,float),-radius,radius)
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
        if global_lb>baseline+1e-5:
            raise AssertionError('SAA global lower bound exceeds certified CASE objective')
        queue=[];serial=0;heappush(queue,(global_lb,serial,lower,upper))
        progress=[];processed=0;retired_lb=float('inf')
        save_state()
    session_processed=0
    while queue and session_processed<max_nodes and time.monotonic()-start<seconds:
        lb,_,a,b=heappop(queue)
        if lb>=best-gap_tolerance:
            retired_lb=min(retired_lb,lb)
            continue
        processed+=1;session_processed+=1
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
        save_state()
        _save(output/'progress.json',dict(history=progress,theta=best_theta.tolist()))
    save_state()
    final_lb=min([q[0] for q in queue]+[retired_lb,best])
    # Tolerance-fathomed regions are NOT exact: keep their dual bounds.
    result=dict(theta=best_theta.tolist(),lower=float(final_lb),upper=float(best),
                gap=float(max(0.,best-final_lb)),nodes=processed,
                session_nodes=session_processed,resumable_checkpoint=str(checkpoint),
                seconds=time.monotonic()-start,
                status=('OPTIMAL_CERTIFIED' if best-final_lb<=1e-8
                         else 'GAP_CERTIFIED' if best-final_lb<=gap_tolerance+1e-10
                         else 'BOUNDED_UNRESOLVED'),
                 tolerance_fathomed_lower=None if not np.isfinite(retired_lb) else float(retired_lb),
                allowed_schedules='incumbent + SAA target per seed' if problems[0].targets is not None else 'all feasible',
                guarantee='finite-sample scenario objective only; all components included')
    _save(output/'result.json',result)
    return result