#!/usr/bin/env python3
"""Single-split UHN study: fit on all 24,506 pre-2013 cases, evaluate 8,774 later cases.

Run from the surgery_scheduling repository root after copying this file to
pilot/ and theory_exact.py to surgery/.  Requires a licensed Gurobi install.

Every eligible site-week component is evaluated; unresolved optimizations are
reported, not replaced by cheaper objectives or hidden from denominators.
Evaluation actual durations are used ONLY in the final outcome calculation.
"""
from __future__ import annotations

import os
for _env in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_env]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['SURGERY_RESEARCH_LINK']='inverse' 
from pilot.run_advice_comparison import (prepare, case_baseline, fit_features,
    test_features, out_of_fold_residuals, shared_control, fit_shared,
    final_data, report)
from surgery.asdbs import components, write
from surgery.advice_comparison import (reviewed, scenarios, residual_history, display,
    certified, objective, prescription, LAM)
from surgery.anchored_core import key
from surgery.theory_exact import (vf_shared_miqp, vf_per_week_milp, vf_fixed_plan_lp,
    expected_gap_mip, exact_box_policy, expected_reduced, reduced,
    vf_objective, saa_mip)
from surgery.weekly_planner import evaluate, loads
from surgery.theory_regions import build_problems, global_shared_search
from surgery.theory_bilevel import value_function_exact
from surgery.planner import response

SEEDS=tuple(int(v) for v in os.environ.get('SURGERY_HASH_SEEDS','0,1,2').split(','))
SWEEP=((.8,30.),(.5,30.),(.8,15.),(.5,15.))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,sort_keys=True,allow_nan=False,indent=2))
    tmp.replace(path)


def secure_manifest(root,args):
    p=Path(__file__).resolve().parents[1]
    from pilot.run_advice_comparison import SOURCE_DEPENDENCIES
    files=sorted(set(SOURCE_DEPENDENCIES) | {'pilot/run_theory_full.py',
          'surgery/theory_exact.py','surgery/theory_regions.py','surgery/theory_bilevel.py',
          'surgery/theory_reporting.py','surgery/theory_cache.py',
          'surgery/theory_stat.py'})
    config=dict(workbook=digest(args.workbook),source={f:digest(p/f) for f in files},
                folds=args.fold,final_acknowledged=args.acknowledge_final,seconds=args.seconds,pw_seconds=args.pw_seconds,
                shared_seconds=args.shared_seconds,rounds=args.rounds,search_seconds=args.search_seconds,
                task_seconds=args.task_seconds,round_seconds=args.round_seconds,
                global_nodes=args.global_nodes,global_seconds=args.global_seconds,
                eval_arm_seconds=args.eval_arm_seconds,
                preprocess_seconds=args.preprocess_seconds,
                setup_seconds=args.setup_seconds,
                coefficient_radius=args.radius,cores=args.cores,methods=args.methods,
                pw_nodes=args.pw_nodes,stat_seconds=args.stat_seconds,
                eval_arms=args.eval_arms,
                scenario_seed=20261020,scenarios=dict(A=50,B=50,C=200),
                response_grid=SWEEP,hash_seeds=SEEDS,
                note='Only full verified deployments count as completed; no silent fallback')
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(config)):
        raise RuntimeError('Changed model specification: choose a NEW output directory')
    atomic_json(path,config)
    return config


def make_context(args,fold,cohort):
    root=args.out/'single_split_2013'
    root.mkdir(parents=True,exist_ok=True)
    setup_start=time.monotonic()
    def setup_left():
        remaining=args.setup_seconds-(time.monotonic()-setup_start)
        if remaining<=.1:raise RuntimeError('SETUP_WALL_LIMIT: checkpoints retained')
        return remaining
    if fold=='2013':
        fit,held,tw,ew,tidx,eidx,names=final_data(args.workbook)
    else:
        fit,held,tw,ew,tidx,eidx,names,_=prepare(args.workbook,int(fold))
    if fold=='2013':
        if len(fit['booked'])!=24506 or len(held['booked'])!=8774:
            raise AssertionError('Unexpected training/test cohort for 2013')
    elif len(fit['booked'])+len(held['booked'])!=24506:
        raise AssertionError('The full 24,506-case development cohort is required')
    setup_left()
    case_baseline(root,fit,args.cores)
    held['case_w']=fit['case_w']
    titems,eitems=components(tw,tidx),components(ew,eidx)
    def assert_partition(items,n,label):
        indices=np.concatenate([np.asarray(rows,int) for _,_,rows in items])
        if len(indices)!=n or not np.array_equal(np.sort(indices),np.arange(n)):
            raise AssertionError(f'{label} cases were dropped or duplicated')
    assert_partition(titems,len(fit['booked']),'Fit')
    assert_partition(eitems,len(held['booked']),'Evaluation')
    H,mu,sd=fit_features(root,titems,fit,names,args.seconds,args.cores,
                         stage_seconds=setup_left())
    H_eval=test_features(eitems,held,names,mu,sd,args.seconds,args.cores,
                         folder=root,stage_seconds=setup_left())
    fframe=cohort[cohort.case_id.isin(fit['case_id'])]
    setup_left()
    errors=out_of_fold_residuals(root,cohort,tw,fit,args.cores)
    pools=residual_history(titems,fit,fframe,errors)
    setup_left()
    beta=shared_control(fit,H)
    atomic_json(root/'stat.json',dict(beta=beta.tolist()))
    return dict(root=root,train=titems,eval=eitems,fit=fit,held=held,
                H=H,H_eval=H_eval,pools=pools,beta=beta)


def init_bank(c,d0,pools,seed=20261020):
    return dict(A=scenarios(c,d0,pools,seed+0,count=50),
                B=scenarios(c,d0,pools,seed+1,count=50),
                C=scenarios(c,d0,pools,seed+2,count=200))


def reviewed_policy(c,rows,H,base,coeff):
    from surgery.planner import reachable
    return reviewed(c.booked,base-c.booked+H[rows]@coeff)


def deploy(c,d,seconds,bank=None,deadline=None):
    from surgery.theory_cache import cached_deploy
    state=_SCORE_STATE if _SCORE_STATE is not None else {}
    def exact_solve(component,durations,seed):
        remaining=seconds if deadline is None else min(seconds,deadline-time.monotonic())
        if remaining<=.1:
            raise RuntimeError('DEPLOY_WALL_LIMIT')
        return certified(component,durations,seed,remaining)
    return cached_deploy(c,d,SEEDS,exact_solve,
           conn=state.get("planner_conn"),stats=state.get("planner_stats"))


def scores(c,d,d0,bank,method,seconds,seed=0,deadline=None):
    """Score with an exact deployed planner and a shared candidate deadline."""
    def limit():
        remaining=seconds if deadline is None else min(seconds,deadline-time.monotonic())
        if remaining<=.1:raise RuntimeError('CANDIDATE_WALL_LIMIT')
        return remaining
    if method=='VF':
        from surgery.anchored_weekly_solver import solve_week
        ans=solve_week(c,d,seconds=limit(),tie=False,threads=1)
        if not ans.complete or ans.assignment is None:
            raise RuntimeError('VF primary planner incomplete: '+ans.status)
        return vf_objective(c,ans.assignment,d,bank,d0,LAM)
    if method=='EG':
        ans=expected_gap_mip(c,d,bank,seconds=limit(),gamma=1.)
        if not ans.complete:
            raise RuntimeError('EG adversary incomplete: '+ans.status)
        return float(ans.value)+LAM/2*np.square(d-d0).sum()
    if method in ('EG_LIB','ANCHORED_T'):
        library=(_SCORE_STATE or {}).get('libraries',{}).get(key(c))
        if library is None or not library.get('schedules'):
            raise RuntimeError('SCHEDULE_LIBRARY_UNRESOLVED')
        from surgery.theory_exact import reduced as phi
        schedules=library['schedules']
        # Bank A/B g(z) depends only on the fixed realized-duration bank,
        # not the proposed coefficients. Reuse precomputed library values.
        cached_key=('g_A' if seed==20261020 else
                    'g_B' if seed==20261021 else None)
        if cached_key and len(library.get(cached_key,[]))==len(schedules):
            g=library[cached_key]
        else:
            g=[expected_reduced(c,z,bank) for z in schedules]
        f=[phi(c,z,d) for z in schedules]
        if method=='EG_LIB':
            from surgery.anchored_weekly_solver import solve_week
            from surgery.weekly_planner import _objective_tolerance
            ans=solve_week(c,d,seconds=limit(),threads=1,tie=False)
            if not ans.complete or ans.value is None:
                raise RuntimeError('EG_LIBRARY_PRIMARY_UNCERTIFIED: '+ans.status)
            Q=float(ans.value-480*c.k+np.sum(d)+30*c.n)
            eps=_objective_tolerance(ans.value)
            # B_1 is a gap loss. Its hinge includes Q; do not add Q again.
            if ans.assignment is None:
                raise RuntimeError('EG_LIBRARY_NO_PRIMARY_ASSIGNMENT')
            primary=tuple(map(int,ans.assignment))
            # Include the primary-optimal schedule at this candidate d,
            # even if it does not occur in the frozen schedule library.
            selected_g=expected_reduced(c,primary,bank)
            selected_f=phi(c,primary,d)
            losses=[gz-max(fz-Q-eps,0.) for gz,fz in zip(g,f)]
            losses.append(selected_g-max(selected_f-Q-eps,0.))
            value=max(losses)
        else:
            target=int(np.argmin(g))
            value=f[target]+max(gz-fz for gz,fz in zip(g,f))
        return float(value+LAM/2*np.square(d-d0).sum())
    if method in ('SHARED','SHARED_BASE','SHARED_PLUS','SHARED_LEGACY','OBS','ROBUST'):
        if method=='ROBUST':
            u=display(c.booked,d)
            cost=[]
            for alpha,h in SWEEP:
                altered=np.maximum(1.,c.booked+response(u,alpha,h))
                z=deploy(c,altered,limit(),deadline=deadline)
                cost.append(np.mean([expected_reduced(c,zi,bank) for zi in z.values()]))
            return float(np.mean(cost)+LAM/2*np.square(d-d0).sum())
        z=deploy(c,d,limit(),deadline=deadline)
        if method=='OBS':
            # Realized outcomes are allowed in FITTING ONLY, not evaluation.
            bank=np.asarray(c.actual,float)[None,:]
        return float(np.mean([expected_reduced(c,zi,bank) for zi in z.values()])
                     +LAM/2*np.square(d-d0).sum())
    raise ValueError(method)


_SCORE_STATE = None


def _score_initializer(state):
    global _SCORE_STATE
    _SCORE_STATE=state
    os.environ['SURGERY_SOLVER_THREADS']='1'
    from threadpoolctl import threadpool_limits
    from surgery.theory_cache import connect
    _SCORE_STATE["_thread_limit"]=threadpool_limits(limits=1)
    _SCORE_STATE["planner_conn"]=(
        connect(state["cache_path"]) if state.get("cache_path") else None)
    _SCORE_STATE["planner_stats"]=dict(hits=0,solves=0)


def _train_component(task):
    """Bounded coefficient shard. Unfinished scores are missing, not approximate."""
    if len(task) in (7,8):
        if len(task)==8:
            ck,c,rows,coefficients,round_id,indices,task_seconds,parent_deadline=task
        else:
            ck,c,rows,coefficients,round_id,indices,task_seconds=task
            parent_deadline=None
        state=_SCORE_STATE
        base,H,method,pools,seconds=(state["base"],state["H"],
                                      state["method"],state["pools"],state["seconds"])
        legacy=False
    else:
        # Direct compatibility with deterministic unit tests.
        ck,c,rows,base,H,_,coefficients,method,pools,seconds,round_id=task
        indices=list(range(len(coefficients)))
        task_seconds=float('inf')
        parent_deadline=None
        legacy=True
    started=time.monotonic()
    deadline=min(started+task_seconds,
                 float('inf') if parent_deadline is None else parent_deadline)
    d0=reviewed(c.booked,base[rows])
    # Give components with >=25 cases a 120 s primary/planner allowance.
    # They are already split into one candidate per scoring shard.
    seconds=max(float(seconds),120.) if c.n>=25 and not legacy else float(seconds)
    seed=(20261020 if round_id>=0 else
          20261021 if (_SCORE_STATE or {}).get('fixed_banks') else
          20261021+1009*(-round_id-1))
    bank=scenarios(c,d0,pools,seed,count=50)
    results=[]
    for j in indices:
        stats=(_SCORE_STATE or {}).get('planner_stats',{})
        before=(stats.get('hits',0),stats.get('solves',0))
        remaining=deadline-time.monotonic()
        if remaining<=.1:
            results.append((int(j),dict(value=None,complete=False,
                                         status='TASK_WALL_LIMIT')))
            continue
        d=reviewed_policy(c,rows,H,d0,np.asarray(coefficients[j]))
        try:
            value=scores(c,d,d0,bank,method,seconds,seed=seed,deadline=deadline)
            record=dict(value=float(value),complete=True,
                    planner_cache_hits=stats.get('hits',0)-before[0],
                    planner_solver_calls=stats.get('solves',0)-before[1])
        except RuntimeError as exc:
            record=dict(value=None,complete=False,status=str(exc))
            extended=(_SCORE_STATE or {}).get('retry_seconds',0.)
            if extended and deadline-time.monotonic()>5.:
                try:
                    value=scores(c,d,d0,bank,method,
                         min(extended,deadline-time.monotonic()),seed=seed,
                         deadline=deadline)
                    record=dict(value=float(value),complete=True,
                                retried_after=str(exc))
                except RuntimeError as second:
                    record['retry_status']=str(second)
        results.append((int(j),record))
    if legacy:
        return ck,c.n,[record for _,record in results]
    return ck,c.n,results


def parallel_score(items,data,H,coefficients,method,pools,seconds,cores,round_id,
                   folder,*,pool=None,task_seconds=600.,deadline=None):
    """Full-component candidate evaluation, sharded and checkpointed exactly.

    A candidate is certified only when EVERY component has a certified score.
    No incomplete or timed-out component enters the comparative objective.
    """
    folder.mkdir(parents=True,exist_ok=True)
    ids=[key(c) for _,c,_ in items]
    gathered={ck:{} for ck in ids}
    base=data['X']@data['case_w']
    identity=hashlib.sha256(json.dumps(dict(method=method,coefficients=coefficients,
                     bank_round=round_id),sort_keys=True).encode()).hexdigest()
    tasks=[]
    for ck,c,rows in items:
        # One large component's 51 candidates are now spread across workers.
        # ROBUST makes four response-model by three hash deployments for
        # EACH candidate. Keep one candidate per task to avoid serializing
        # twelve solves for several candidates behind the same hard case.
        shard=(1 if method=='ROBUST' or c.n>=25
               else 3 if c.n>=12 else 6)
        for start in range(0,len(coefficients),shard):
            indices=list(range(start,min(start+shard,len(coefficients))))
            path=folder/(ck+f'_shard_{start:04d}.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('identity')!=identity or record.get('indices')!=indices:
                    raise RuntimeError(f'Stale candidate shard for {ck}, {start}')
                values=record['results']
                if len(values)==len(indices) and all(v.get('complete') for v in values):
                    gathered[ck].update({j:v for j,v in zip(indices,values)})
                    continue
                # An unfinished task must be eligible for a fresh retry.
                # Its incomplete record is not a frozen certified score.
            allowance=max(float(task_seconds),240.) if c.n>=25 else task_seconds
            tasks.append((ck,c,rows,coefficients,round_id,indices,allowance,path))
    def collect(executor):
        # Bounded in-flight work limits memory, and avoids launching the entire
        # backlog after a method or round has used its time allocation.
        from concurrent.futures import wait, FIRST_COMPLETED
        queue=iter(tasks);active={}
        def submit_next():
            if deadline is not None and time.monotonic()>=deadline:return False
            try:work=next(queue)
            except StopIteration:return False
            ck,c,rows,coeff,ri,ind,limit,path=work
            future=executor.submit(_train_component,(ck,c,rows,coeff,ri,ind,limit,deadline))
            active[future]=(ck,ind,path)
            return True
        # Do not queue a second wave with a fresh subtask allowance.
        for _ in range(min(cores,len(tasks))):
            if not submit_next():break
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                ck,indices,path=active.pop(future)
                got,_,results=future.result()
                if got!=ck or [j for j,_ in results]!=indices:
                    raise AssertionError('Scored candidate shard mismatch')
                values=[v for _,v in results]
                record=dict(identity=identity,indices=indices,results=values)
                atomic_json(path,record)
                gathered[ck].update({j:v for j,v in results})
                submit_next()
    if tasks:
        if pool is not None:
            collect(pool)
        else:
            state=dict(base=base,H=H,method=method,pools=pools,seconds=seconds)
            with ProcessPoolExecutor(max_workers=min(cores,len(tasks)),
                   mp_context=get_context('spawn'),initializer=_score_initializer,
                   initargs=(state,)) as executor:
                collect(executor)
    valid=np.ones(len(coefficients),bool)
    score=np.zeros(len(coefficients))
    unresolved=0
    unresolved_components=set()
    for ck in ids:
        for j in range(len(coefficients)):
            x=gathered[ck].get(j)
            if x is None or not x['complete']:
                valid[j]=False
                unresolved+=1
                unresolved_components.add(ck)
            else:
                score[j]+=x['value']
    atomic_json(folder/'batch_status.json',dict(method=method,round=round_id,
                components=len(items),candidates=len(coefficients),
                unfinished_scores=unresolved,
                unresolved_components=len(unresolved_components),
                unresolved_cases=sum(c.n for ck,c,_ in items if ck in unresolved_components),
                complete_candidates=int(valid.sum()),
                remaining_shards=len(tasks)))
    return score,valid


def _selection_paired_candidate(folder,items,indices,min_case_coverage=.98):
    """Heuristic paired *training-only* decision from finished exact shards.

    A missing score is NOT replaced with zero or a fake bound. The update
    is explicitly uncertified over the missing components.
    """
    sums=[0.]*len(indices)
    coverage=[0]*len(indices)
    total=sum(c.n for _,c,_ in items)
    for ck,c,_ in items:
        records={}
        for file in folder.glob(ck+'_shard_*.json'):
            entry=json.loads(file.read_text())
            for j,result in zip(entry['indices'],entry['results']):
                records[j]=result
        incumbent=records.get(0)
        if incumbent is None or not incumbent.get('complete'):continue
        for j in range(1,len(indices)):
            candidate=records.get(j)
            if candidate is not None and candidate.get('complete'):
                sums[j]+=float(candidate['value']-incumbent['value'])
                coverage[j]+=c.n
    viable=[j for j in range(1,len(indices))
            if coverage[j]>=min_case_coverage*total and
            sums[j] < -.10*coverage[j]]
    if not viable:return None
    j=min(viable,key=lambda j:sums[j]/coverage[j])
    return dict(j=j,paired_cases=coverage[j],total_cases=total,
                paired_case_fraction=coverage[j]/total,
                observed_training_difference=sums[j],
                warning='Paired exact subset ONLY; not a full-cohort guarantee')


def train_search(args,ctx,method,*,warm_start=None):
    """Exact candidate scoring + reproducible full-training directional search.

    This is an APPROXIMATION of the original direct/EG outer problem. Neither
    method is called globally optimal or locally stationary by this runner.
    """
    root=ctx['root']/method.lower();root.mkdir(exist_ok=True)
    result=root/'trained.json'
    if result.exists():return np.asarray(json.loads(result.read_text())['coeff'],float)
    H=ctx['H'];beta=ctx['beta'];items=ctx['train'];fit=ctx['fit']
    from numpy.linalg import norm
    import itertools
    directions=np.eye(H.shape[1]).tolist()
    if not getattr(args,'selection_fast',False):
        for j,k in itertools.combinations(range(H.shape[1]),2):
            x=np.eye(H.shape[1]);directions.extend(((x[j]+x[k])/np.sqrt(2),
                                                   (x[j]-x[k])/np.sqrt(2)))
    progress=root/'progress.json'
    if progress.exists():
        saved=json.loads(progress.read_text())
        theta=np.asarray(saved['theta'],float)
        history=saved['history']
        if len(theta)!=H.shape[1] or len(history)>args.rounds:
            raise RuntimeError('Invalid saved shared search state')
    else:
        theta=np.zeros(H.shape[1]);history=[]
    flexible=[x for x in items if any(len(x[1].allowed[int(s)])>1 for s in x[1].surgeon)]
    candidates=flexible or items
    rg=np.random.default_rng(20261020)
    sampled=rg.choice(len(candidates),size=min(32 if getattr(args,'selection_fast',False) else 150,len(candidates)),replace=False)
    shortlist_items=[candidates[int(j)] for j in sampled]
    search_start=time.monotonic()
    method_deadline=search_start+args.search_seconds
    warm_start_status='NOT_REQUESTED' if warm_start is None else 'PENDING'
    exhausted_reason=None
    score_state=dict(base=fit['X']@fit['case_w'],H=H,method=method,
                     pools=ctx['pools'],seconds=args.seconds,
                     retry_seconds=(min(360.,args.task_seconds)
                         if getattr(args,'selection_fast',False) else 0.),
                     cache_path=str(ctx['root']/'exact_deployments.sqlite3'))
    # Initialize SQLite once before worker startup; all workers then share
    # the same WAL-enabled table safely under its database locks.
    from surgery.theory_cache import connect
    init_cache=connect(score_state['cache_path'])
    init_cache.close()
    with ProcessPoolExecutor(max_workers=args.cores,
            mp_context=get_context('spawn'),initializer=_score_initializer,
            initargs=(score_state,)) as score_pool:
        if not history and warm_start is not None:
            # The VF solution is only a warm start. It must beat STAT on
            # THIS METHOD'S actual objective before becoming the incumbent.
            candidate=np.clip(np.asarray(warm_start,float)-beta,-args.radius,args.radius)
            if candidate.shape!=theta.shape:raise AssertionError('Warm-start dimension mismatch')
            v=[beta.tolist(),(beta+candidate).tolist()]
            costs,valid=parallel_score(items,fit,H,v,method,ctx['pools'],
                        args.seconds,args.cores,0,root/'warm_start_A',
                        pool=score_pool,task_seconds=args.task_seconds,
                        deadline=min(method_deadline,time.monotonic()+args.round_seconds))
            if valid.all() and costs[1]<costs[0]-1e-8:
                if method=='OBS':
                    theta=candidate.copy()
                    warm_start_status='ACCEPTED_OBS_TRAIN_COST_NO_INDEPENDENT_BANK'
                else:
                    fresh,confirmed=parallel_score(items,fit,H,v,method,ctx['pools'],
                        args.seconds,args.cores,-101,root/'warm_start_B',
                        pool=score_pool,task_seconds=args.task_seconds,
                        deadline=min(method_deadline,time.monotonic()+args.round_seconds))
                    if confirmed.all() and fresh[1]<fresh[0]-1e-8:
                        theta=candidate.copy()
                        warm_start_status='ACCEPTED_A_AND_B'
                    else:warm_start_status='REJECTED_BY_B_OR_TIMEOUT'
            else:warm_start_status='REJECTED_BY_A_OR_TIMEOUT'
        for round_id in range(len(history),args.rounds):
            if time.monotonic()>=method_deadline:
                exhausted_reason='METHOD_WALL_LIMIT'
                break
            round_deadline=min(method_deadline,time.monotonic()+args.round_seconds)
            step=args.radius/2**round_id
            proposed=[beta+theta]
            for v in directions:
                for sign in (-1.,1.):
                    proposed.append(beta+np.clip(theta+sign*step*np.asarray(v),
                                                  -args.radius,args.radius))
            # Include coordinate-wise extreme advice coefficients. The old
            # 8/2**round schedule could move only 15.984 units in ten rounds,
            # even though the declared coefficient domain is [-32,32].
            for j in range(len(theta)) if not getattr(args,'selection_fast',False) else []:
                for sign in (-1.,1.):
                    boundary=theta.copy();boundary[j]=sign*args.radius
                    proposed.append(beta+boundary)
            fititems=[(ck,c,rows) for ck,c,rows in shortlist_items]
            preview,valid=parallel_score(fititems,fit,H,[p.tolist() for p in proposed],
                                 method,ctx['pools'],args.seconds,args.cores,round_id,
                                 root/f'round_{round_id:02d}'/'screen',pool=score_pool,
                                 task_seconds=args.task_seconds,deadline=round_deadline)
            order=sorted((i for i in range(1,len(proposed)) if valid[i]),key=lambda i:preview[i])[:(2 if getattr(args,'selection_fast',False) else 5)]
            indices=[0]+order
            allscore,valid=parallel_score(items,fit,H,[proposed[i].tolist() for i in indices],
                               method,ctx['pools'],args.seconds,args.cores,round_id,
                               root/f'round_{round_id:02d}'/'all',pool=score_pool,
                               task_seconds=args.task_seconds,deadline=round_deadline)
            if not valid[0]:
                if getattr(args,'selection_fast',False):
                    partial=_selection_paired_candidate(
                          root/f'round_{round_id:02d}'/'all',items,indices)
                    if partial is not None:
                        theta=proposed[indices[partial['j']]]-beta
                        history.append(dict(round=round_id,accepted=True,
                            accepted_partial_training_only=True,
                            paired_subset=partial,bank_B=None,
                            complete_candidates=int(valid.sum())))
                        atomic_json(root/'progress.json',
                                    dict(history=history,theta=theta.tolist()))
                        continue
                exhausted_reason='INCOMPLETE_FULL_BANK_A_SCORE'
                break
            alternatives=[j for j in range(1,len(indices)) if valid[j]
                          and allscore[j]<allscore[0]-1e-8]
            winner=min(alternatives,key=lambda j:allscore[j],default=None)
            accepted=False; confirmation=None
            if winner is not None:
                if method=='OBS':
                    # Actual training durations are the SAME in A and B;
                    # changing the RNG seed creates no independent evidence.
                    accepted=True
                    confirmation=dict(status='NOT_APPLICABLE_SAME_OUTCOMES',
                                      description='OBS optimized on all historical fit outcomes')
                else:
                    c0=proposed[0].tolist();c1=proposed[indices[winner]].tolist()
                    bscore,bvalid=parallel_score(items,fit,H,[c0,c1],method,ctx['pools'],
                                 args.seconds,args.cores,-round_id-1,
                                 root/f'round_{round_id:02d}'/'confirm_B',pool=score_pool,
                                 task_seconds=args.task_seconds,deadline=round_deadline)
                    confirmation=dict(incumbent=float(bscore[0]) if bvalid[0] else None,
                                      candidate=float(bscore[1]) if bvalid[1] else None,
                                      completed=bool(bvalid.all()))
                    if not bvalid[0]:
                        exhausted_reason='INCOMPLETE_BANK_B_CONFIRMATION'
                        break
                    accepted=bool(bvalid[1] and bscore[1]<bscore[0]-1e-8)
                if accepted:theta=proposed[indices[winner]]-beta
            history.append(dict(round=round_id,accepted=accepted,
                                winner=None if not accepted else int(indices[winner]),
                                baseline=float(allscore[0]),
                                chosen=float(allscore[winner]) if winner is not None else None,
                                bank_B=confirmation,
                                complete_candidates=int(valid.sum())))
            atomic_json(root/'progress.json',dict(history=history,theta=theta.tolist()))
    coeff=beta+theta
    atomic_json(result,dict(coeff=coeff.tolist(),theta=theta.tolist(),history=history,
                 completed_rounds=len(history),requested_rounds=args.rounds,
                 status='COMPLETE_ROUNDS' if len(history)==args.rounds else
                        (exhausted_reason or 'BUDGET_EXHAUSTED'),
                 wall_seconds_this_session=time.monotonic()-search_start,
                 warm_start=warm_start_status,
                 guarantee=('Training-only partially paired updates are heuristic '
                     'and NOT full-cohort guarantees.'
                     if any(x.get('accepted_partial_training_only') for x in history)
                     else 'OBS: historical fitting outcomes (not independent Bank B)'
                     if method=='OBS' else
                     'Full-bank comparisons when certified; independent Bank B '
                     'does not establish population significance. Not globally optimal.')))
    return coeff


def train_vf(args,ctx):
    p=ctx['root']/'vf_global.json'
    if p.exists():return json.loads(p.read_text())
    cert=vf_shared_miqp(ctx['train'],ctx['fit'],ctx['H'],ctx['beta'],ctx['pools'],
           seconds=args.shared_seconds,threads=args.cores,
           coefficient_radius=args.radius,
           restart_dir=ctx['root']/'vf_restart')
    payload=cert.to_json()
    payload['meaning']='global bound on penalized scenario VF within coefficient box'
    atomic_json(p,payload)
    return payload


_EVAL_STATE = None


def _eval_initializer(state):
    """One Gurobi thread per independent evaluation worker."""
    global _EVAL_STATE
    _EVAL_STATE=state
    os.environ['SURGERY_SOLVER_THREADS']='1'
    from threadpoolctl import threadpool_limits
    from surgery.theory_cache import connect
    state['_thread_limit']=threadpool_limits(limits=1)
    state['planner_conn']=connect(state['cache_path'])
    state['planner_stats']=dict(hits=0,solves=0)
    global _SCORE_STATE
    _SCORE_STATE=state


def _eval_arm_task(task):
    """Independent budgeted (component, arm) task; no arm can starve another."""
    ck,c,rows,arm,budget=task
    state=_EVAL_STATE
    tic=time.monotonic();deadline=tic+float(budget)
    from gurobipy import GurobiError
    from surgery.anchored_weekly_solver import solve_week
    base=state['base'];H=state['H'];pools=state['pools']
    models=state['models']
    seconds=(max(state['seconds'],state.get('large_eval_solver_seconds',0.))
             if c.n>=25 else state['seconds'])
    pw_seconds=state['pw_seconds']
    d0=reviewed(c.booked,base[rows])
    banks=init_bank(c,d0,pools)
    report=dict(status='UNRESOLVED',arm=arm,budget_seconds=float(budget),
                solver_threads=1)
    def left(request):
        amount=min(float(request),deadline-time.monotonic())
        if amount<=.1:raise RuntimeError('ARM_WALL_LIMIT')
        return amount
    def deployment(duration,preverified=None):
        t0=time.monotonic()
        dur=np.asarray(duration,float)
        z=None
        if preverified is not None and set(map(int,preverified))==set(SEEDS):
            # Only caller-provided exact deployments for the SAME duration
            # are reused, e.g. from the certified PER_WEEK search itself.
            z={int(k):tuple(map(int,v)) for k,v in preverified.items()}
        if z is None:
            z=deploy(c,dur,left(seconds),deadline=deadline)
        outcome=dict(status='COMPLETE',duration=dur.tolist(),
               assignments={str(sd):list(zi) for sd,zi in z.items()},
               realized={str(sd):float(evaluate(c,zi,c.actual)['cost'])
                         for sd,zi in z.items()},
               model={name:float(np.mean([expected_reduced(c,zi,draws)
                                         for zi in z.values()]))
                      for name,draws in banks.items()})
        comonotone=np.sort(np.asarray(banks['C'],float),axis=0)
        outcome['model']['C_comonotone']=float(np.mean(
              [expected_reduced(c,zi,comonotone) for zi in z.values()]))
        outcome['over720']={str(sd):int(evaluate(c,zi,c.actual)['room_days_over_720'])
                            for sd,zi in z.items()}
        outcome['max_room_load']={str(sd):float(evaluate(c,zi,c.actual)['max_room_load'])
                                  for sd,zi in z.items()}
        regimes={};imbalances={}
        for sd,zi in z.items():
            load=loads(c,zi,dur)
            occupied=np.bincount(np.asarray(zi,int),minlength=c.k)>0
            overflow=float(np.maximum(load[occupied]-480.,0.).sum())
            idle=float(np.maximum(480.-load[occupied],0.).sum())
            imbalances[str(sd)]=min(overflow,idle)
            regimes[str(sd)]=('LIGHT_UNIFORM' if overflow<1e-7 else
                              'HEAVY_UNIFORM' if idle<1e-7 else 'MIXED')
        outcome['regime']=regimes
        outcome['planned_imbalance']=imbalances
        outcome['hash_assignment_disagreement']=int(
              len({tuple(v) for v in z.values()})>1)
        vals=list(outcome['realized'].values())
        outcome['hash_cost_disagreement']=int(max(vals)-min(vals)>1e-5)
        outcome['changed']=int(np.sum(np.abs(dur-d0)>1e-7))
        outcome['verification_seconds']=time.monotonic()-t0
        return outcome
    try:
        if arm=='HINDSIGHT':
            result=solve_week(c,c.actual,seconds=left(pw_seconds),
                              threads=1,tie=False)
            report=dict(complete=bool(result.complete),
                cost=float(result.value) if result.complete else None,
                lower=result.bound,status=result.status,
                budget_seconds=budget,solver_threads=1)
        elif arm=='SAA':
            result=saa_mip(c,banks['A'],seconds=left(pw_seconds),threads=1)
            report=dict(status=result.status,certified=bool(result.complete),
                lower=result.lower,upper=result.upper,
                seconds=result.seconds,solver_threads=1,budget_seconds=budget,
                assignment=None if result.assignment is None else list(result.assignment))
            if result.assignment is not None:
                z=tuple(map(int,result.assignment))
                report['model']={name:float(expected_reduced(c,z,draws))
                                 for name,draws in banks.items()}
                report['model']['C_comonotone']=float(
                      expected_reduced(c,z,np.sort(banks['C'],axis=0)))
                report['realized']=float(evaluate(c,z,c.actual)['cost'])
        else:
            extra={}
            verified=None
            if arm=='CASE':d=d0
            elif arm=='BOOKED':d=np.asarray(c.booked,float)
            elif arm=='RISK':
                d=reviewed(c.booked,np.quantile(banks['A'],1.75/2.75,axis=0)-c.booked)
            elif arm in models:
                coeff=models[arm]
                if coeff is None:
                    raise RuntimeError('TRAINING_UNRESOLVED')
                d=reviewed_policy(c,rows,H,d0,coeff)
            elif arm in ('PER_WEEK','PER_WEEK_ROBUST'):
                seeds=(0,) if arm=='PER_WEEK' else SEEDS
                d,verified,accepted,status,_=prescription(c,d0,banks,seeds=seeds,
                       seconds=left(seconds),budget=left(pw_seconds))
                extra=dict(search_status=status,search_accepted=bool(accepted),
                           operational_fallback=not bool(accepted))
            elif arm=='QFILL':
                baseline=deploy(c,d0,left(seconds),deadline=deadline)
                cert=vf_fixed_plan_lp(c,baseline[0],banks['A'],
                       seconds=left(pw_seconds),d0=d0)
                extra=dict(vf_cert=cert.to_json())
                if cert.reviewed is None:raise RuntimeError('QFILL_NO_FEASIBLE_ADVICE')
                d=np.asarray(cert.reviewed,float)
            elif arm=='VF_PW':
                cert=vf_per_week_milp(c,banks['A'],seconds=left(pw_seconds),d0=d0,
                                      threads=1)
                extra=dict(vf_cert=cert.to_json())
                if cert.reviewed is None:raise RuntimeError('VF_PW_NO_FEASIBLE_ADVICE')
                d=np.asarray(cert.reviewed,float)
            elif arm in ('PW_VFCG','PW_BOUNDED','PW_ANCHORED'):
                if arm=='PW_VFCG':
                    ans=value_function_exact(c,banks['A'],d0,
                        seconds=left(pw_seconds),separation_seconds=min(seconds,left(pw_seconds)),
                        seeds=SEEDS)
                else:
                    ans=exact_box_policy(c,banks['A'],d0,seconds=left(pw_seconds),
                       solve_seconds=min(seconds,left(pw_seconds)),
                       nodes=state['pw_nodes'],
                       restricted='case+saa' if arm=='PW_ANCHORED' else None)
                extra=dict(global_certificate=ans)
                if ans['reviewed'] is None:
                    raise RuntimeError('NO_FEASIBLE_INCUMBENT')
                d=np.asarray(ans['reviewed'],float)
                # Exact VFCG output records the SAME hash seeds/durations.
                if arm=='PW_VFCG' and ans.get('assignments') is not None:
                    verified={int(k):tuple(v) for k,v in ans['assignments'].items()}
            else:
                raise ValueError('Unknown evaluation method '+arm)
            report=deployment(d,preverified=verified)
            report.update(extra)
    except (RuntimeError,GurobiError) as exc:
        report.update(status='UNRESOLVED',error=str(exc))
    except Exception as exc:
        # Unexpected code/mathematics errors must not kill other tasks; the
        # final audit is invalid until every INTERNAL_ERROR is investigated.
        import traceback
        report.update(status='INTERNAL_ERROR',error=repr(exc),
                      traceback=traceback.format_exc())
    report.setdefault('budget_seconds',float(budget))
    report.setdefault('solver_threads',1)
    report['method_wall_seconds']=time.monotonic()-tic
    return ck,arm,report


def validate_evaluation_arms(chosen):
    """Keep the paper launcher's evaluation-arm contract aligned with the runner."""
    admissible=set(('BOOKED','CASE','RISK','STAT','STAT_PLUS','SHARED',
                    'VF','VF_SHARED','EG','OBS','ROBUST','SHARED_GLOBAL',
                    'ANCHORED_TARGET','SHARED_BASE','SHARED_PLUS','VF_PLUS',
                    'GAP_PLUS','ANCHORED_PLUS','EG_LIB','ANCHORED_T','PER_WEEK','PER_WEEK_ROBUST','QFILL',
                    'VF_PW','PW_VFCG','PW_BOUNDED','PW_ANCHORED','SAA','HINDSIGHT'))
    if len(chosen)!=len(set(chosen)) or not set(chosen)<=admissible:
        raise ValueError('Duplicate or unsupported evaluation method')
    if 'CASE' not in chosen:
        raise ValueError('CASE must be included for the predeclared fallback')


def evaluate_all(args,ctx,models):
    """One method per component per task, with independent wall limits."""
    evaluation_start=time.monotonic()
    root=ctx['root']/'full_evaluation'
    root.mkdir(exist_ok=True)
    shard_root=root/'per_arm';shard_root.mkdir(exist_ok=True)
    chosen=list(args.eval_arms)
    validate_evaluation_arms(chosen)
    state=dict(base=ctx['held']['X']@ctx['held']['case_w'],
        H=ctx['H_eval'],pools=ctx['pools'],models=models,
        seconds=args.seconds,pw_seconds=args.pw_seconds,pw_nodes=args.pw_nodes,
        cache_path=str(ctx['root']/'exact_deployments.sqlite3'),
        large_eval_solver_seconds=getattr(args,'large_eval_solver_seconds',0.))
    from surgery.theory_cache import connect
    connection=connect(state['cache_path']);connection.close()
    jobs=[];done={}
    for ck,c,rows in sorted(ctx['eval'],key=lambda item:-item[1].n):
        done[ck]=dict(site=c.group,monday=c.monday,cases=c.n,methods={},
                      hindsight={},saa_reference={},mechanism_diagnostics={})
        for arm in chosen:
            path=shard_root/(ck+'__'+arm+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if arm=='HINDSIGHT':done[ck]['hindsight']=record
                elif arm=='SAA':done[ck]['saa_reference']=record
                else:done[ck]['methods'][arm]=record
                continue
            jobs.append((ck,c,rows,arm,args.eval_arm_seconds))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.cores,len(jobs)),
                mp_context=get_context('spawn'),initializer=_eval_initializer,
                initargs=(state,)) as pool:
            from concurrent.futures import wait,FIRST_COMPLETED
            backlog=iter(jobs);active={};completed=0
            def submit_next():
                try:job=next(backlog)
                except StopIteration:return False
                future=pool.submit(_eval_arm_task,job)
                active[future]=(job[0],job[3])
                return True
            for _ in range(min(args.cores,len(jobs))):
                if not submit_next():break
            while active:
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    ck,arm=active.pop(future)
                    try:
                        returned_ck,returned_arm,record=future.result()
                        if returned_ck!=ck or returned_arm!=arm:
                            raise AssertionError('Evaluation identity mismatch')
                    except Exception as exc:
                        import traceback
                        record=dict(status='INTERNAL_ERROR',error=repr(exc),
                                    traceback=traceback.format_exc())
                    atomic_json(shard_root/(ck+'__'+arm+'.json'),record)
                    if arm=='HINDSIGHT':done[ck]['hindsight']=record
                    elif arm=='SAA':done[ck]['saa_reference']=record
                    else:done[ck]['methods'][arm]=record
                    completed+=1
                    if completed%100==0:
                        print(f'Evaluation arms saved {completed}/{len(jobs)}',flush=True)
                    submit_next()
    for ck,_,_ in ctx['eval']:
        record=done[ck]
        record['elapsed']=sum(float(x.get('method_wall_seconds',0.))
                     for x in record['methods'].values())
        atomic_json(root/(ck+'.json'),record)
    rows=[done[ck] for ck,_,_ in ctx['eval']]
    arms=sorted({m for row in rows for m in row['methods']})
    output=dict(fold=ctx['root'].name,n_cases=sum(r['cases'] for r in rows),
                components=len(rows),n_mondays=len(set(r['monday'] for r in rows)),arms={})
    # A feasible starting policy is not evidence of successful training.
    # These records prevent a zero-round EG/SHARED fit from appearing to be
    # an equally optimized candidate in the final paper comparison.
    training_paths={'VF_SHARED':ctx['root']/'vf_global.json',
                    'STAT_PLUS':ctx['root']/'stat_plus.json',
                    'STAT':ctx['root']/'stat.json'}
    for name in ('SHARED','OBS','EG','ROBUST'):
        training_paths[name]=ctx['root']/name.lower()/'trained.json'
    output['training_status']={}
    for name,path in training_paths.items():
        if path.exists():
            saved=json.loads(path.read_text())
            output['training_status'][name]=dict(
                status=saved.get('status',
                       'SURROGATE_SOLVER_RESULT' if name=='VF_SHARED'
                       else 'FITTED_CONTROL'),
                completed_rounds=saved.get('completed_rounds'),
                requested_rounds=saved.get('requested_rounds'),
                feasible=bool(models.get(name) is not None),
                solver_bound=saved.get('lower'),
                solver_incumbent=saved.get('upper'))
        else:
            output['training_status'][name]=dict(
                status='NOT_TRAINED_OR_UNRESOLVED',
                feasible=bool(models.get(name) is not None))

    for arm in arms:
        records=[r for r in rows if r['methods'].get(arm,{}).get('status')=='COMPLETE']
        # Never compare methods on different selected cohorts without disclosure.
        output['arms'][arm]=dict(completed_components=len(records),
                completed_cases=sum(r['cases'] for r in records),
                unresolved=len(rows)-len(records),
                uniform_light=sum(all(v=='LIGHT_UNIFORM' for v in r['methods'][arm]['regime'].values()) for r in records),
                uniform_heavy=sum(all(v=='HEAVY_UNIFORM' for v in r['methods'][arm]['regime'].values()) for r in records),
                hash_assignment_disagreement=sum(r['methods'][arm]['hash_assignment_disagreement'] for r in records),
                hash_realized_cost_disagreement=sum(r['methods'][arm]['hash_cost_disagreement'] for r in records),
                mean_realized_cost_per_case=None if len(records)!=len(rows) else
                sum(np.mean(list(r['methods'][arm]['realized'].values())) for r in records)/sum(r['cases'] for r in records),
                model_C_per_case=None if len(records)!=len(rows) else
                sum(r['methods'][arm]['model']['C'] for r in records)/sum(r['cases'] for r in records),
                 deployment_evaluation_wall_hours=sum(
                     float(r['methods'][arm].get('verification_seconds',0.))
                     for r in rows)/3600.,
                 method_total_wall_hours=sum(
                     float(r['methods'].get(arm,{}).get('method_wall_seconds',0.))
                     for r in rows)/3600.,
                 timeout_components=sum('LIMIT' in str(r['methods'].get(arm,{}).get(
                     'error','')) or 'LIMIT' in str(r['methods'].get(arm,{}).get('status',''))
                     for r in rows),
                 internal_error_components=sum(r['methods'].get(arm,{}).get('status')
                    =='INTERNAL_ERROR' for r in rows))
    # Full-case operational fallback is a DISTINCT policy, not an
    # exact-result substitution: missing method deployments use CASE.
    output['fallback_CASE']={}
    case_ok=all(r['methods'].get('CASE',{}).get('status')=='COMPLETE' for r in rows)
    for arm in arms:
        unavailable=[r for r in rows if r['methods'].get(arm,{}).get('status')!='COMPLETE']
        report=dict(fallback_components=len(unavailable),
                    fallback_cases=sum(r['cases'] for r in unavailable))
        if case_ok:
            def cost_record(row):
                own=row['methods'].get(arm,{})
                if own.get('status')=='COMPLETE':return own
                return row['methods']['CASE']
            report['all_case_mean_realized_per_case']=sum(
                 np.mean(list(cost_record(r)['realized'].values()))
                 for r in rows)/sum(r['cases'] for r in rows)
            report['all_case_model_C_per_case']=sum(
                 cost_record(r)['model']['C'] for r in rows)/sum(r['cases'] for r in rows)
        output['fallback_CASE'][arm]=report
    output['audit_failed']=any(
       rec.get('status')=='INTERNAL_ERROR'
       for row in rows for rec in list(row['methods'].values())+
         [row.get('saa_reference',{}),row.get('hindsight',{})])
    output['evaluation_elapsed_wall_hours']=(time.monotonic()-evaluation_start)/3600.
    output['worker_hours_all_attempts']=sum(
        float(rec.get('method_wall_seconds',0.))
        for row in rows for rec in list(row['methods'].values())+
             [row.get('saa_reference',{}),row.get('hindsight',{})])/3600.
    output['audit_status']='INVALID_INTERNAL_ERRORS' if output['audit_failed'] else (
          'COMPLETE' if all(not r['unresolved'] for r in output['arms'].values())
          else 'PARTIAL_WITH_EXPLICIT_UNRESOLVED')
    # Model-dependence stress preserves each case's 200-sample marginal.
    output['dependence_stress']={}
    n_all=sum(r['cases'] for r in rows)
    for arm in arms:
        valid=[r for r in rows if r['methods'].get(arm,{}).get('status')=='COMPLETE'
               and 'C_comonotone' in r['methods'][arm].get('model',{})]
        result=dict(completed_components=len(valid),
                    unresolved=len(rows)-len(valid))
        if len(valid)==len(rows):
            base=sum(r['methods'][arm]['model']['C'] for r in valid)/n_all
            stressed=sum(r['methods'][arm]['model']['C_comonotone']
                         for r in valid)/n_all
            result.update(model_C_per_case=base,
                          comonotone_C_per_case=stressed,
                          dependence_increase=stressed-base)
        output['dependence_stress'][arm]=result
    # Separate sensitivity effects from the main deployed hash objective;
    # incomplete mechanisms do not delete primary realized-cost observations.
    output['mechanism_summary']={}
    for arm in ('CASE','STAT','SHARED','OBS','EG','ROBUST','VF_SHARED'):
        matched=[r for r in rows if arm in r.get('mechanism_diagnostics',{})]
        summaries=dict(attempted_components=len(matched),
                       missing_components=len(rows)-len(matched))
        balance=[(r, r['mechanism_diagnostics'][arm].get('balanced_tie',{}))
                 for r in matched]
        completed=[(r,m) for r,m in balance if m.get('status')=='COMPLETE']
        summaries['balanced_tie_completed']=len(completed)
        if len(completed)==len(rows):
            summaries['balanced_tie_realized_per_case']=sum(
                m['realized'] for _,m in completed)/n_all
            summaries['balanced_tie_model_C_per_case']=sum(
                m['model_C'] for _,m in completed)/n_all
        sensitivity={}
        for alpha,h in SWEEP[1:]:
            key=f'{alpha:g}_{h:g}'
            selected=[(r,r['mechanism_diagnostics'][arm].get(
                        'response_sweep',{}).get(key,{})) for r in matched]
            good=[(r,m) for r,m in selected
                  if m.get('status') in ('COMPLETE','REUSED_NOMINAL')]
            response_rec=dict(completed_components=len(good),
                              unresolved=len(rows)-len(good))
            if len(good)==len(rows):
                response_rec.update(cost_realized_per_case=sum(
                    m['cost_realized'] for _,m in good)/n_all,
                    cost_model_C_per_case=sum(
                    m['cost_model_C'] for _,m in good)/n_all)
            sensitivity[key]=response_rec
        summaries['response_sweep']=sensitivity
        output['mechanism_summary'][arm]=summaries

    hindsight_all=all(r.get('hindsight',{}).get('complete') for r in rows)
    if hindsight_all:
        output['hindsight_cost_per_case']=sum(float(r['hindsight']['cost']) for r in rows)/sum(r['cases'] for r in rows)
        for name,arm in output['arms'].items():
            if not arm['unresolved']:
                arm['realized_regret_per_case']=arm['mean_realized_cost_per_case']-output['hindsight_cost_per_case']
    else:
        output['hindsight_unresolved']=sum(not r.get('hindsight',{}).get('complete') for r in rows)
    # SAA is a NONDEPLOYABLE oracle, not another advice method. Distinguish
    # primal and dual bounds; incomplete SAA solves remain explicit.
    references=[r.get('saa_reference',{}) for r in rows]
    saa=dict(certified_components=sum(bool(x.get('certified')) for x in references),
             primal_components=sum(x.get('upper') is not None for x in references),
             dual_components=sum(x.get('lower') is not None for x in references),
             total_components=len(rows),
             hours=sum(float(x.get('seconds',0)) for x in references)/3600.,
             interpretation='Unrestricted stochastic schedule; NOT implementable through advice')
    n_cases=sum(r['cases'] for r in rows)
    if all(x.get('lower') is not None for x in references):
        saa['model_A_lower_per_case']=sum(float(x['lower']) for x in references)/n_cases
    if all(x.get('upper') is not None for x in references):
        saa['model_A_upper_per_case']=sum(float(x['upper']) for x in references)/n_cases
    if all(x.get('assignment') is not None for x in references):
        saa['realized_cost_per_case']=sum(float(x['realized']) for x in references)/n_cases
        saa['model_A_feasible_per_case']=sum(float(x['model']['A']) for x in references)/n_cases
        saa['model_C_feasible_per_case']=sum(float(x['model']['C']) for x in references)/n_cases
    output['saa_reference']=saa
    # All comparisons below use the same Bank-A reduced scheduling score.
    # A feasible per-week policy is an upper bound on the best inducible
    # objective, not its global optimum. The universal SAA dual bound gives
    # a conservative upper bound on the unregularized inducibility gap.
    output['inducibility_bounds']={}
    if 'model_A_lower_per_case' in saa:
        L=saa['model_A_lower_per_case']
        for name in ('PER_WEEK_ROBUST','PW_BOUNDED','PW_ANCHORED','PW_VFCG'):
            records=[r['methods'].get(name,{}) for r in rows]
            if len(records)!=len(rows) or any(
                    rec.get('status')!='COMPLETE' for rec in records):
                continue
            U=sum(float(rec['model']['A']) for rec in records)/n_cases
            if U<L-1e-4:
                raise AssertionError(f'{name} below unconditional SAA lower bound')
            output['inducibility_bounds'][name]=dict(
                scenario_A_deployed_cost_per_case=U,
                inducibility_gap_lower=0.,
                inducibility_gap_upper=max(0.,U-L),
                statement='Valid conservative interval [0, U_policy - LB_SAA]; '
                          'NOT a point estimate of the optimal inducibility gap. '
                          'Use bounds only for their declared policy class and seed objective; '
                          'PW_VFCG now uses the same three-seed average.')
    if 'model_A_upper_per_case' in saa and 'model_A_lower_per_case' in saa:
        output['saa_reference']['scenario_A_optimality_gap_per_case']=(
            saa['model_A_upper_per_case']-saa['model_A_lower_per_case'])

    # Full-cohort scalability/structural diagnostics, without selective case
    # deletion. Any incomplete arm has no headline cost estimate.
    def size_bucket(n):
        return ('1' if n==1 else '2-5' if n<=5 else '6-10' if n<=10
                else '11-20' if n<=20 else '21-34' if n<=34 else '35+')
    output['strata']={'site':{},'size':{}}
    for label,partition in [('site',lambda r:r['site']),('size',lambda r:size_bucket(r['cases']))]:
        for v in sorted({partition(r) for r in rows}):
            group=[r for r in rows if partition(r)==v]
            n_cases=sum(r['cases'] for r in group)
            summary=dict(cases=n_cases,components=len(group),arms={})
            for arm in arms:
                accepted=[r for r in group if r['methods'].get(arm,{}).get('status')=='COMPLETE']
                summary['arms'][arm]=dict(completed=len(accepted),unresolved=len(group)-len(accepted),
                    cost_per_case=None if len(accepted)!=len(group) else
                    sum(np.mean(list(r['methods'][arm]['realized'].values())) for r in accepted)/n_cases)
            output['strata'][label][v]=summary
    output['optimization_certificates']={}
    for arm in ('QFILL','VF_PW','PW_ANCHORED','PW_BOUNDED','PW_VFCG'):
        info=[r['methods'].get(arm,{}) for r in rows]
        cert=[x.get('global_certificate') or x.get('vf_cert') for x in info]
        cert=[x for x in cert if x is not None]
        gaps=[float(x['upper'])-float(x['lower']) for x in cert
              if x.get('upper') is not None and x.get('lower') is not None]
        output['optimization_certificates'][arm]=dict(
            attempted=len(cert),globally_certified=sum(bool(x.get('complete')) for x in cert),
            unresolved=len(rows)-sum(bool(x.get('complete')) for x in cert),
            gaps_count=len(gaps),gap_median=None if not gaps else float(np.median(gaps)),
            gap_p90=None if not gaps else float(np.quantile(gaps,.9)),
            solver_hours=sum(float(x.get('seconds',0)) for x in cert)/3600.)
    output['component_runtime_hours']=sum(r.get('elapsed',0.) for r in rows)/3600.
    output['calibration_warning']='Scenario and 2013 outcomes are not used interchangeably; reported certificates apply to bank A only'
    # Calendar-Monday paired bootstrap. Hash seeds are repeated algorithmic
    # deployments, never additional independent observations.
    from collections import defaultdict
    rng=np.random.default_rng(20261020)
    mondays=sorted({r['monday'] for r in rows})
    by_week=defaultdict(lambda:defaultdict(float));den=defaultdict(int)
    for r in rows:
        den[r['monday']]+=r['cases']
        for name,rec in r['methods'].items():
            if rec.get('status')=='COMPLETE':
                by_week[r['monday']][name]+=float(np.mean(list(rec['realized'].values())))
    if all(r['methods']['CASE'].get('status')=='COMPLETE' for r in rows):
        boot=rng.integers(len(mondays),size=(20000,len(mondays)))
        total=np.asarray([den[m] for m in mondays],float)
        output['paired_vs_CASE']={}
        rawp=[]
        for arm in arms:
            if arm=='CASE' or output['arms'][arm]['unresolved']:
                continue
            diff=np.asarray([by_week[m][arm]-by_week[m]['CASE'] for m in mondays])
            est=float(diff.sum()/total.sum())
            samples=diff[boot].sum(axis=1)/total[boot].sum(axis=1)
            p=float((1+np.count_nonzero(abs(samples-est)>=abs(est)))/(len(samples)+1))
            output['paired_vs_CASE'][arm]=dict(difference=est,
                ci95=np.quantile(samples,[.025,.975]).tolist(),p_uncorrected=p)
            rawp.append((p,arm))
        for rank,(p,arm) in enumerate(sorted(rawp)):
            adjusted=max(p2*(len(rawp)-r) for r,(p2,_) in enumerate(sorted(rawp)[:rank+1]))
            output['paired_vs_CASE'][arm]['p_holm']=float(min(1.,adjusted))
        # Directly compare with the matched five-feature statistical policy
        # and with SHARED, using identical Monday resamples and complete cases.
        for reference in ('STAT_PLUS','SHARED'):
            if output['arms'].get(reference,{}).get('unresolved',1):
                continue
            contrasts={}
            for arm in arms:
                if arm==reference or output['arms'][arm]['unresolved']:
                    continue
                diff=np.asarray([by_week[m][arm]-by_week[m][reference]
                                 for m in mondays],float)
                estimate=float(diff.sum()/total.sum())
                draws=diff[boot].sum(axis=1)/total[boot].sum(axis=1)
                p=float((1+np.count_nonzero(
                    np.abs(draws-estimate)>=abs(estimate)))/(len(draws)+1))
                contrasts[arm]=dict(difference=estimate,
                    ci95=np.quantile(draws,[.025,.975]).tolist(),
                    p_exploratory_uncorrected=p)
            output['paired_vs_'+reference]=contrasts
        output['inference_note']=(
            'Exploratory week-clustered bootstrap; same Monday resamples and '
            'same hash rules for every arm. Holm applies to CASE contrasts only; '
            'STAT_PLUS and SHARED contrasts are unadjusted exploratory comparisons.')
    else:output['inference_note']='CASE not fully certified; no inferential contrasts'
    atomic_json(ctx['root']/'theory_summary.json',output)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workbook',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cores',type=int,default=15)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--pw-seconds',type=float,default=480)
    ap.add_argument('--pw-nodes',type=int,default=100)
    ap.add_argument('--shared-seconds',type=float,default=14400)
    ap.add_argument('--stat-seconds',type=float,default=1800.)
    ap.add_argument('--radius',type=float,default=32.)
    ap.add_argument('--rounds',type=int,default=10)
    ap.add_argument('--search-seconds',type=float,default=14400.,
                    help='Soft per-method directional training budget; stops between rounds')
    ap.add_argument('--task-seconds',type=float,default=1500.,
                    help='Maximum wall seconds for one component/candidate shard')
    ap.add_argument('--round-seconds',type=float,default=7200.,
                    help='Maximum wall seconds for a directional learning round')
    ap.add_argument('--fold',choices=['2013'],default='2013',
                    help='Frozen design: all 24,506 pre-2013 cases train; 8,774 2013 cases test')
    ap.add_argument('--acknowledge-final',action='store_true',
                    help='Explicitly acknowledge that the 2013 test is being opened once')
    ap.add_argument('--methods',default='STAT_PLUS,VF,SHARED,EG,OBS,ROBUST')
    ap.add_argument('--global-nodes',type=int,default=40)
    ap.add_argument('--global-seconds',type=float,default=14400.)
    ap.add_argument('--eval-arm-seconds','--eval-component-seconds',
                    dest='eval_arm_seconds',type=float,default=1500.,
                    help='Independent wall budget for each test (component, method) task')
    ap.add_argument('--eval-arms',default=(
        'BOOKED,CASE,RISK,STAT,STAT_PLUS,SHARED,VF_SHARED,EG,OBS,ROBUST,'
        'PER_WEEK_ROBUST,QFILL,VF_PW,SAA,HINDSIGHT'),
        help='Full-test methods; optional expensive solvers are opt-in')
    ap.add_argument('--setup-seconds',type=float,default=14400.,
                    help='Cooperative overall training/test feature-preparation wall limit')
    ap.add_argument('--preprocess-seconds',type=float,default=14400.,
                    help='Whole wall budget for training SAA preprocessing stage')
    args=ap.parse_args()
    args.methods=[s.strip().upper() for s in args.methods.split(',')]
    args.eval_arms=[s.strip().upper() for s in args.eval_arms.split(',')]
    if not set(args.methods)<=set(['STAT_PLUS','VF','SHARED','SHARED_LEGACY','OBS','EG','ROBUST','GLOBAL','ANCHORED']):
        ap.error('Unknown method')
    if not 1<=args.cores<=15 or min(args.seconds,args.shared_seconds,args.pw_seconds,args.radius,args.search_seconds,args.task_seconds,args.round_seconds,
                                         args.eval_arm_seconds,args.preprocess_seconds,args.setup_seconds,
                                         args.stat_seconds)<=0:
        ap.error('Invalid time, coefficient radius or cores')
    if not args.acknowledge_final:
        ap.error('Opening the 2013 test requires --acknowledge-final')
    args.out.mkdir(parents=True,exist_ok=True)
    secure_manifest(args.out,args)
    from run_weekly_rich import read_cohort
    raw,cohort=read_cohort(args.workbook)
    summaries=[]
    for fold in ('2013',):
        try:
            ctx=make_context(args,fold,cohort)
        except RuntimeError as exc:
            if not any(term in str(exc) for term in ('FEATURE_SETUP_INCOMPLETE','SETUP_WALL_LIMIT')):
                raise
            root=args.out/'single_split_2013'
            atomic_json(root/'setup_status.json',dict(status='DEPENDENCY_UNRESOLVED',
                        error=str(exc),training_cases=24506,test_cases=8774))
            atomic_json(args.out/'summary.json',dict(status='INCOMPLETE_SETUP',
                        error=str(exc),no_test_outcomes_used=True))
            print('Setup incomplete; durable progress retained:',exc,flush=True)
            return
        models={'STAT':ctx['beta']}
        failures={}
        if 'STAT_PLUS' in args.methods:
            from surgery.theory_stat import fit_stat_plus
            try:
                models['STAT_PLUS']=fit_stat_plus(ctx['fit'],ctx['H'],ctx['beta'],
                    radius=args.radius,seconds=args.stat_seconds,
                    out=ctx['root']/'stat_plus.json')
            except RuntimeError as exc:
                failures['STAT_PLUS']=str(exc);models['STAT_PLUS']=None
        if 'VF' in args.methods:
            try:
                vf=train_vf(args,ctx)
                models['VF_SHARED']=(None if vf.get('reviewed') is None else
                                    ctx['beta']+np.asarray(vf['reviewed'],float))
            except RuntimeError as exc:
                failures['VF']=str(exc);models['VF_SHARED']=None
        if 'SHARED' in args.methods:
            try:models['SHARED']=train_search(
                    args,ctx,'SHARED',warm_start=models.get('VF_SHARED'))
            except RuntimeError as exc:
                failures['SHARED']=str(exc);models['SHARED']=None
        if 'SHARED_LEGACY' in args.methods:
            try:
                theta=fit_shared(ctx['root'],ctx['train'],ctx['fit'],ctx['H'],ctx['pools'],
                                 ctx['beta'],args.seconds,args.cores,args.rounds,SEEDS)
                models['SHARED_LEGACY']=ctx['beta']+theta
            except RuntimeError as exc:
                failures['SHARED_LEGACY']=str(exc);models['SHARED_LEGACY']=None
        # The global direct and target-restricted optimizers use the SAME full
        # fixed Bank A objective and return explicit primal/dual gaps.
        if 'GLOBAL' in args.methods or 'ANCHORED' in args.methods:
            bounds_root=ctx['root']/'global_method'
            try:
                allproblems=build_problems(ctx['train'],ctx['fit'],ctx['H'],
                       ctx['beta'],ctx['pools'],bounds_root,
                       seconds=args.seconds,target_restricted=False,cores=args.cores,
                       stage_seconds=args.preprocess_seconds)
            except RuntimeError as exc:
                allproblems=None
                failures['GLOBAL_SAA_BUILD']=str(exc)
            if 'GLOBAL' in args.methods:
                try:
                    if allproblems is None:raise RuntimeError('SAA bounds unavailable')
                    prior=(models.get('SHARED') if models.get('SHARED') is not None
                           else ctx['beta'])-ctx['beta']
                    cert=global_shared_search(allproblems,ctx['beta'],
                         out=bounds_root/'direct',seconds=args.global_seconds,
                         solve_seconds=args.seconds,max_nodes=args.global_nodes,
                         radius=args.radius,start_theta=prior,cores=args.cores)
                    models['SHARED_GLOBAL']=ctx['beta']+np.asarray(cert['theta'])
                except RuntimeError as exc:
                    failures['SHARED_GLOBAL']=str(exc);models['SHARED_GLOBAL']=None
            if 'ANCHORED' in args.methods:
                try:
                    targets=build_problems(ctx['train'],ctx['fit'],ctx['H'],
                         ctx['beta'],ctx['pools'],bounds_root,
                         seconds=args.seconds,target_restricted=True,cores=args.cores,
                         stage_seconds=args.preprocess_seconds)
                    cert=global_shared_search(targets,ctx['beta'],
                         out=bounds_root/'target',seconds=args.global_seconds,
                         solve_seconds=args.seconds,max_nodes=args.global_nodes,
                         radius=args.radius,cores=args.cores)
                    models['ANCHORED_TARGET']=ctx['beta']+np.asarray(cert['theta'])
                except RuntimeError as exc:
                    failures['ANCHORED_TARGET']=str(exc);models['ANCHORED_TARGET']=None
        for method in args.methods:
            if method in ('SHARED','SHARED_LEGACY','GLOBAL','ANCHORED','VF','STAT_PLUS'):continue
            try:
                models[method]=train_search(args,ctx,method,
                                      warm_start=models.get('VF_SHARED'))
            except RuntimeError as exc:
                failures[method]=str(exc)
                models[method]=None
        atomic_json(ctx['root']/'training_failures.json',failures)
        summary=evaluate_all(args,ctx,models)
        summaries.append(summary)
        print(json.dumps(summary,indent=2),flush=True)
    atomic_json(args.out/'summary.json',dict(
        design='single chronological split',training_cases=24506,test_cases=8774,
        training_period='pre-2013',test_period='2013',
        outcome_evaluations=summaries,
        warning='Test outcomes must not be used to tune subsequent choices'))

if __name__=='__main__':main()