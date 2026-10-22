"""Shared schedule libraries and matched bounded search for SHARED, VF, EG-LIB, ANCHORED-T.

This is a finite-library experimental approximation, not an unrestricted
EG or inducibility certificate. A proposed update is accepted ONLY after
complete, certified Bank-A AND Bank-B comparisons on all training cases.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json
import time
import numpy as np

from surgery.anchored_core import key
from surgery.theory_exact import expected_reduced
from surgery.theory_cache import connect,signature

_LIBRARY_STATE=None


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    temp.replace(path)


def _library_init(state):
    global _LIBRARY_STATE
    _LIBRARY_STATE=state
    from threadpoolctl import threadpool_limits
    _LIBRARY_STATE['_limits']=threadpool_limits(1)


def _library_one(task):
    from surgery.theory_exact import saa_mip
    from surgery.weekly_planner import evaluate
    from surgery.anchored_weekly_solver import solve_week
    from pilot import run_theory_full as full
    ck,c,rows,seconds=task
    state=_LIBRARY_STATE
    tic=time.monotonic()
    d0=full.reviewed(c.booked,state['base'][rows])
    A=full.scenarios(c,d0,state['pools'],20261020,count=50)
    B=full.scenarios(c,d0,state['pools'],20261021,count=50)
    solutions=[tuple(map(int,c.original))]
    statuses={'ORIGINAL':'FEASIBLE'}
    try:
        result=solve_week(c,d0,seconds=min(seconds,35.),threads=1,
                          tie=True,hash_seed=0)
        if result.complete and result.assignment is not None:
            solutions.append(tuple(map(int,result.assignment)))
            statuses['CASE']='CERTIFIED'
        else:statuses['CASE']='UNRESOLVED'
    except Exception as exc:
        statuses['CASE']='UNRESOLVED:'+type(exc).__name__
    try:
        left=max(.01,seconds-(time.monotonic()-tic))
        result=solve_week(c,d0,seconds=min(6.,left),threads=1,tie=False)
        if result.complete and result.assignment is not None:
            solutions.append(tuple(map(int,result.assignment)))
            statuses['PRIMARY']='CERTIFIED'
        else:statuses['PRIMARY']='UNRESOLVED'
    except Exception as exc:
        statuses['PRIMARY']='UNRESOLVED:'+type(exc).__name__
    try:
        left=max(.01,seconds-(time.monotonic()-tic))
        ans=saa_mip(c,A,seconds=min(12.,left),threads=1)
        if ans.assignment is not None:
            solutions.append(tuple(map(int,ans.assignment)))
            statuses['SAA']='OPTIMAL' if ans.complete else 'FEASIBLE_UNFINISHED'
        else:statuses['SAA']='NO_INCUMBENT'
    except Exception as exc:
        statuses['SAA']='UNRESOLVED:'+type(exc).__name__
    unique=list(dict.fromkeys(solutions))
    gA=[float(expected_reduced(c,z,A)) for z in unique]
    gB=[float(expected_reduced(c,z,B)) for z in unique]
    return ck,dict(schedules=[list(z) for z in unique],g_A=gA,g_B=gB,
                   statuses=statuses,seconds=time.monotonic()-tic,
                   n=c.n,scenario_A_seed=20261020,scenario_B_seed=20261021)


def build_library(ctx,out,*,cores=15,seconds=1800.,task_seconds=25.):
    from pilot import run_theory_full as full
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pending=[(ck,c,rows,min(task_seconds,seconds))
             for ck,c,rows in sorted(ctx['train'],key=lambda x:-x[1].n)
             if not (out/(ck+'.json')).exists()]
    deadline=time.monotonic()+seconds
    if pending:
        state=dict(base=ctx['fit']['X']@ctx['fit']['case_w'],
                   pools=ctx['pools'])
        with ProcessPoolExecutor(max_workers=min(cores,len(pending)),
            mp_context=get_context('spawn'),initializer=_library_init,
            initargs=(state,)) as pool:
            source=iter(pending);active={}
            def launch():
                if time.monotonic()>=deadline:return False
                try:job=next(source)
                except StopIteration:return False
                active[pool.submit(_library_one,job)]=job[0]
                return True
            for _ in range(min(cores,len(pending))):launch()
            while active:
                complete,_=wait(active,return_when=FIRST_COMPLETED)
                for fut in complete:
                    ck=active.pop(fut)
                    try:
                        found,rec=fut.result()
                        if found!=ck:raise AssertionError('Library key mismatch')
                        _save(out/(ck+'.json'),rec)
                    except Exception as exc:
                        _save(out/(ck+'.error.json'),
                              dict(status='LIBRARY_ERROR',message=repr(exc)))
                    launch()
    return load_library(ctx,out)


def load_library(ctx,out):
    library={}
    for ck,c,rows in ctx['train']:
        file=Path(out)/(ck+'.json')
        if file.exists():
            rec=json.loads(file.read_text())
            if rec.get('n')!=c.n or not rec.get('schedules'):
                raise AssertionError('Invalid saved schedule library')
            library[ck]=rec
    return library


def enrich_library(ctx,library,coeff,out):
    """Add certified current policy schedules using the exact deployment cache.

    No additional planner solves: only assignments previously certified by
    the current training scorer can enter the libraries.
    """
    from pilot import run_theory_full as full
    base=ctx['fit']['X']@ctx['fit']['case_w']
    conn=connect(str(ctx['root']/'exact_deployments.sqlite3'))
    added=0
    try:
        for ck,c,rows in ctx['train']:
            if ck not in library:continue
            d0=full.reviewed(c.booked,base[rows])
            d=full.reviewed_policy(c,rows,ctx['H'],d0,coeff)
            sig=signature(c,d,(0,))
            row=conn.execute('SELECT payload FROM deployments WHERE signature=?',(sig,)).fetchone()
            if row is None:continue
            decoded=json.loads(row[0])
            z=list(map(int,decoded['assignments']['0']))
            rec=library[ck]
            if z in rec['schedules']:continue
            A=full.scenarios(c,d0,ctx['pools'],20261020,count=50)
            B=full.scenarios(c,d0,ctx['pools'],20261021,count=50)
            rec['schedules'].append(z)
            rec['g_A'].append(float(expected_reduced(c,z,A)))
            rec['g_B'].append(float(expected_reduced(c,z,B)))
            _save(Path(out)/(ck+'.json'),rec)
            added+=1
    finally:conn.close()
    return added


def _sample(items,n=150):
    """Fixed, mixed-size training screen; all cases remain in full confirmation."""
    if len(items)<=n:return list(items)
    rng=np.random.default_rng(20261022)
    ordered=sorted(items,key=lambda x:-x[1].n)
    head=ordered[:min(15,n//5)]
    selected={x[0] for x in head}
    remaining=[x for x in items if x[0] not in selected]
    indices=rng.choice(len(remaining),size=n-len(head),replace=False)
    return head+[remaining[int(i)] for i in indices]


def train_objective(ctx,method,libraries,out,*,cores=15,seconds=2700.,
                    solve_seconds=45.,task_seconds=150.,rounds=6):
    """Same STAT start, steps and screening for all four method objectives.

    No partial acceptance or cross-method chaining. Bank-A AND Bank-B
    must each cover ALL training components for a certified update.
    """
    from pilot import run_theory_full as full
    method=str(method)
    if method not in ('SHARED','VF','EG_LIB','ANCHORED_T'):
        raise ValueError('Unknown training objective')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+seconds
    path=out/'progress.json'
    beta=np.asarray(ctx['beta'],float)
    if path.exists():
        state=json.loads(path.read_text())
        theta=np.asarray(state['theta'],float)
        history=state['history']
    else:
        theta=np.zeros(5);history=[]
    if theta.shape!=(5,):raise AssertionError('Invalid five-feature checkpoint')
    state=dict(base=ctx['fit']['X']@ctx['fit']['case_w'],
               H=ctx['H'],pools=ctx['pools'],method=method,
               libraries=libraries,seconds=solve_seconds,
               fixed_banks=True,retry_seconds=min(task_seconds,180.),
               cache_path=str(ctx['root']/'exact_deployments.sqlite3'))
    connect(state['cache_path']).close()
    screen=_sample(ctx['train'])
    missing_initial=len(ctx['train'])-len(libraries)
    stopped=None
    for round_id in range(len(history),rounds):
        if deadline-time.monotonic()<10:
            stopped='STAGE_DEADLINE';break
        step=16./2**round_id
        starts=[beta+theta]
        for j in range(5):
            for direction in (-1.,1.):
                trial=np.clip(theta+direction*step*np.eye(5)[j],-16.,16.)
                starts.append(beta+trial)
        rounddir=out/f'round_{round_id:02d}'
        with ProcessPoolExecutor(max_workers=cores,mp_context=get_context('spawn'),
                initializer=full._score_initializer,initargs=(state,)) as pool:
            preview,eligible=full.parallel_score(screen,ctx['fit'],ctx['H'],
                [w.tolist() for w in starts],method,ctx['pools'],
                solve_seconds,cores,round_id,rounddir/'screen',
                pool=pool,task_seconds=task_seconds,deadline=deadline)
            indices=[0]+sorted((j for j in range(1,len(starts)) if eligible[j]),
                               key=lambda j:preview[j])[:3]
            if len(indices)==1:
                history.append(dict(round=round_id,status='NO_CERTIFIED_SCREEN_DIRECTION',
                                    step=step,screen_certified=int(eligible.sum())))
                _save(path,dict(theta=theta.tolist(),history=history,method=method))
                continue
            candidates=[starts[i].tolist() for i in indices]
            A,goodA=full.parallel_score(ctx['train'],ctx['fit'],ctx['H'],
                    candidates,method,ctx['pools'],solve_seconds,cores,round_id,
                    rounddir/'all_A',pool=pool,task_seconds=task_seconds,
                    deadline=deadline)
            eligible_full=[j for j in range(1,len(indices)) if goodA[0] and goodA[j]
                           and A[j]<A[0]-1e-7]
            decision=dict(round=round_id,step=step,
                          full_A_completed=int(goodA.sum()),
                          training_components=len(ctx['train']),
                          training_cases=len(ctx['fit']['booked']),
                          original_objective_A=float(A[0]) if goodA[0] else None,
                          winning_objective_A=None,
                          own_objective_B_change=None,
                          accepted=False)
            if eligible_full:
                chosen=min(eligible_full,key=lambda j:A[j])
                B,goodB=full.parallel_score(ctx['train'],ctx['fit'],ctx['H'],
                   [candidates[0],candidates[chosen]],method,ctx['pools'],
                   solve_seconds,cores,-1,rounddir/'all_B',
                   pool=pool,task_seconds=task_seconds,deadline=deadline)
                decision['winning_objective_A']=float(A[chosen])
                decision['bank_B_certified']=bool(goodB.all())
                if goodB.all():
                    decision['own_objective_B_change']=float(B[1]-B[0])
                    if B[1]<B[0]-1e-7:
                        theta=np.asarray(candidates[chosen])-beta
                        decision['accepted']=True
                        decision['candidate_index']=indices[chosen]
            else:decision['status']='NO_FULL_CERTIFIED_IMPROVEMENT'
            history.append(decision)
            _save(path,dict(theta=theta.tolist(),history=history,method=method))
        if decision['accepted']:
            enrich_library(ctx,libraries,beta+theta,
                           ctx['root']/'training_library')
    status=('ACCEPTED_FULL_A_B' if any(v.get('accepted') for v in history)
            else 'NO_CERTIFIED_IMPROVEMENT')
    if stopped:status+='_'+stopped
    answer=dict(status=status,method=method,coeff=(beta+theta).tolist(),
                theta=theta.tolist(),rounds_completed=len(history),
                rounds_requested=rounds,history=history,
                library_coverage=len(libraries)/len(ctx['train']),
                seconds=time.monotonic()-start,
                optimizer='common_coordinate_16_8_4_2_1_0.5',
                caveat='No partial acceptance; unfinished evaluation is not a loss')
    _save(out/'trained.json',answer)
    return answer


def probe_landscape(ctx,libraries,out,*,cores=15,seconds=1800.,solve_seconds=25.):
    """Diagnostic scan of all four training objectives around shared STAT."""
    from pilot import run_theory_full as full
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    subset=_sample(ctx['train'])
    starts=[np.asarray(ctx['beta'],float)]
    labels=['STAT']
    for j in range(5):
        for distance in (2.,4.,8.,16.):
            for sign in (-1.,1.):
                x=np.asarray(ctx['beta'],float).copy()
                x[j]+=sign*distance
                labels.append(f'axis{j}_{sign*distance:g}')
                starts.append(x)
    objectives=('SHARED','VF','EG_LIB','ANCHORED_T')
    t0=time.monotonic()
    results={}
    for method in objectives:
        remaining=seconds-(time.monotonic()-t0)
        if remaining<5:break
        state=dict(base=ctx['fit']['X']@ctx['fit']['case_w'],
                  H=ctx['H'],pools=ctx['pools'],method=method,
                  libraries=libraries,seconds=solve_seconds,
                  fixed_banks=True,retry_seconds=0.,
                  cache_path=str(ctx['root']/'exact_deployments.sqlite3'))
        with ProcessPoolExecutor(max_workers=cores,mp_context=get_context('spawn'),
              initializer=full._score_initializer,initargs=(state,)) as pool:
            values,valid=full.parallel_score(subset,ctx['fit'],ctx['H'],
                     [x.tolist() for x in starts],method,ctx['pools'],
                     solve_seconds,cores,0,out/method,
                     pool=pool,task_seconds=min(90.,remaining),
                     deadline=time.monotonic()+remaining)
            results[method]=dict(labels=labels,
                scores=[float(v) if ok else None for v,ok in zip(values,valid)],
                exact_components=len(subset),completed_candidates=int(valid.sum()))
            _save(out/'landscape.json',results)
    return results
