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

SEEDS=(0,1,2)
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
          'surgery/theory_reporting.py','surgery/theory_cache.py'})
    config=dict(workbook=digest(args.workbook),source={f:digest(p/f) for f in files},
                folds=args.fold,final_acknowledged=args.acknowledge_final,seconds=args.seconds,pw_seconds=args.pw_seconds,
                shared_seconds=args.shared_seconds,rounds=args.rounds,search_seconds=args.search_seconds,
                global_nodes=args.global_nodes,global_seconds=args.global_seconds,
                coefficient_radius=args.radius,cores=args.cores,methods=args.methods,
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
    if fold=='2013':
        fit,held,tw,ew,tidx,eidx,names=final_data(args.workbook)
    else:
        fit,held,tw,ew,tidx,eidx,names,_=prepare(args.workbook,int(fold))
    if fold=='2013':
        if len(fit['booked'])!=24506 or len(held['booked'])!=8774:
            raise AssertionError('Unexpected training/test cohort for 2013')
    elif len(fit['booked'])+len(held['booked'])!=24506:
        raise AssertionError('The full 24,506-case development cohort is required')
    case_baseline(root,fit,args.cores)
    held['case_w']=fit['case_w']
    titems,eitems=components(tw,tidx),components(ew,eidx)
    def assert_partition(items,n,label):
        indices=np.concatenate([np.asarray(rows,int) for _,_,rows in items])
        if len(indices)!=n or not np.array_equal(np.sort(indices),np.arange(n)):
            raise AssertionError(f'{label} cases were dropped or duplicated')
    assert_partition(titems,len(fit['booked']),'Fit')
    assert_partition(eitems,len(held['booked']),'Evaluation')
    H,mu,sd=fit_features(root,titems,fit,names,args.seconds,args.cores)
    H_eval=test_features(eitems,held,names,mu,sd,args.seconds,args.cores,folder=root)
    fframe=cohort[cohort.case_id.isin(fit['case_id'])]
    errors=out_of_fold_residuals(root,cohort,tw,fit,args.cores)
    pools=residual_history(titems,fit,fframe,errors)
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
    if method in ('SHARED','SHARED_LEGACY','OBS','ROBUST'):
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
    from threadpoolctl import threadpool_limits
    from surgery.theory_cache import connect
    _SCORE_STATE["_thread_limit"]=threadpool_limits(limits=1)
    _SCORE_STATE["planner_conn"]=(
        connect(state["cache_path"]) if state.get("cache_path") else None)
    _SCORE_STATE["planner_stats"]=dict(hits=0,solves=0)


def _train_component(task):
    """Bounded coefficient shard. Unfinished scores are missing, not approximate."""
    if len(task)==7:
        ck,c,rows,coefficients,round_id,indices,task_seconds=task
        state=_SCORE_STATE
        base,H,method,pools,seconds=(state["base"],state["H"],
                                      state["method"],state["pools"],state["seconds"])
        legacy=False
    else:
        # Direct compatibility with deterministic unit tests.
        ck,c,rows,base,H,_,coefficients,method,pools,seconds,round_id=task
        indices=list(range(len(coefficients)))
        task_seconds=float('inf')
        legacy=True
    started=time.monotonic();deadline=started+task_seconds
    d0=reviewed(c.booked,base[rows])
    bank=scenarios(c,d0,pools,(20261020 if round_id>=0
                        else 20261021+1009*(-round_id-1)),count=50)
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
            value=scores(c,d,d0,bank,method,seconds,deadline=deadline)
            record=dict(value=float(value),complete=True,
                    planner_cache_hits=stats.get('hits',0)-before[0],
                    planner_solver_calls=stats.get('solves',0)-before[1])
        except RuntimeError as exc:
            record=dict(value=None,complete=False,status=str(exc))
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
        shard=1 if c.n>=25 else 3 if c.n>=12 else 6
        for start in range(0,len(coefficients),shard):
            indices=list(range(start,min(start+shard,len(coefficients))))
            path=folder/(ck+f'_shard_{start:04d}.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('identity')!=identity or record.get('indices')!=indices:
                    raise RuntimeError(f'Stale candidate shard for {ck}, {start}')
                values=record['results']
                if len(values)==len(indices):
                    gathered[ck].update({j:v for j,v in zip(indices,values)})
                    continue
            tasks.append((ck,c,rows,coefficients,round_id,indices,task_seconds,path))
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
            future=executor.submit(_train_component,(ck,c,rows,coeff,ri,ind,limit))
            active[future]=(ck,ind,path)
            return True
        for _ in range(min(2*cores,len(tasks))):
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
    for ck in ids:
        for j in range(len(coefficients)):
            x=gathered[ck].get(j)
            if x is None or not x['complete']:
                valid[j]=False
                unresolved+=1
            else:
                score[j]+=x['value']
    atomic_json(folder/'batch_status.json',dict(method=method,round=round_id,
                components=len(items),candidates=len(coefficients),
                unfinished_scores=unresolved,complete_candidates=int(valid.sum()),
                remaining_shards=len(tasks)))
    return score,valid


def train_search(args,ctx,method):
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
    sampled=rg.choice(len(candidates),size=min(150,len(candidates)),replace=False)
    shortlist_items=[candidates[int(j)] for j in sampled]
    search_start=time.monotonic()
    score_state=dict(base=fit['X']@fit['case_w'],H=H,method=method,
                     pools=ctx['pools'],seconds=args.seconds,
                     cache_path=str(ctx['root']/'exact_deployments.sqlite3'))
    # Initialize SQLite once before worker startup; all workers then share
    # the same WAL-enabled table safely under its database locks.
    from surgery.theory_cache import connect
    init_cache=connect(score_state['cache_path'])
    init_cache.close()
    with ProcessPoolExecutor(max_workers=args.cores,
            mp_context=get_context('spawn'),initializer=_score_initializer,
            initargs=(score_state,)) as score_pool:
        for round_id in range(len(history),args.rounds):
            if time.monotonic()-search_start>=args.search_seconds:
                break
            step=8./2**round_id
            proposed=[beta+theta]
            for v in directions:
                for sign in (-1.,1.):
                    proposed.append(beta+theta+sign*step*np.asarray(v))
            fititems=[(ck,c,rows) for ck,c,rows in shortlist_items]
            preview,valid=parallel_score(fititems,fit,H,[p.tolist() for p in proposed],
                                 method,ctx['pools'],args.seconds,args.cores,round_id,
                                 root/f'round_{round_id:02d}'/'screen',pool=score_pool)
            order=sorted((i for i in range(1,len(proposed)) if valid[i]),key=lambda i:preview[i])[:5]
            indices=[0]+order
            allscore,valid=parallel_score(items,fit,H,[proposed[i].tolist() for i in indices],
                               method,ctx['pools'],args.seconds,args.cores,round_id,
                               root/f'round_{round_id:02d}'/'all',pool=score_pool)
            if not valid[0]:
                raise RuntimeError('Incumbent score uncertified on full Bank A; increase --seconds')
            alternatives=[j for j in range(1,len(indices)) if valid[j]
                          and allscore[j]<allscore[0]-1e-8]
            winner=min(alternatives,key=lambda j:allscore[j],default=None)
            accepted=False; confirmation=None
            if winner is not None:
                # Fresh independent Bank B for each round, never used in proposing
                # this candidate. This is a safeguard, not a sequential p-value.
                c0=proposed[0].tolist();c1=proposed[indices[winner]].tolist()
                bscore,bvalid=parallel_score(items,fit,H,[c0,c1],method,ctx['pools'],
                             args.seconds,args.cores,-round_id-1,
                             root/f'round_{round_id:02d}'/'confirm_B',pool=score_pool)
                confirmation=dict(incumbent=float(bscore[0]) if bvalid[0] else None,
                                  candidate=float(bscore[1]) if bvalid[1] else None,
                                  completed=bool(bvalid.all()))
                if not bvalid[0]:
                    raise RuntimeError('Incumbent score uncertified on Bank B')
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
                 status='COMPLETE_ROUNDS' if len(history)==args.rounds else 'BUDGET_EXHAUSTED',
                 wall_seconds_this_session=time.monotonic()-search_start,
                 guarantee='full-Bank-A candidate search with independent per-round Bank-B confirmation; outer search not globally optimal or population-certified'))
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


def _eval_component(task):
    ck,c,rows,base,H,models,pools,seconds,pw_seconds,pw_nodes=task
    d0=reviewed(c.booked,base[rows]);banks=init_bank(c,d0,pools)
    cases={};start=time.monotonic()
    def add(name,d,extra=None):
        evaluation_start=time.monotonic()
        record=dict(status='COMPLETE',duration=list(np.asarray(d,float)))
        if extra:record.update(extra)
        try:
            selected=deploy(c,d,seconds)
            record['assignments']={str(s):list(z) for s,z in selected.items()}
            record['realized']={str(s):evaluate(c,z,c.actual)['cost'] for s,z in selected.items()}
            record['model']={bank:float(np.mean([expected_reduced(c,z,draws) for z in selected.values()]))
                             for bank,draws in banks.items()}
            record['over720']={str(s):evaluate(c,z,c.actual)['room_days_over_720'] for s,z in selected.items()}
            record['max_room_load']={str(s):evaluate(c,z,c.actual)['max_room_load']
                                     for s,z in selected.items()}
            regime={}; imbalance={}
            for seed,z in selected.items():
                l=loads(c,z,d)
                occupied=np.bincount(np.asarray(z,int),minlength=c.k)>0
                excess=l[occupied]-480.
                overtime=float(np.maximum(excess,0).sum())
                idle=float(np.maximum(-excess,0).sum())
                imbalance[str(seed)]=min(overtime,idle)
                regime[str(seed)]=('LIGHT_UNIFORM' if overtime<1e-7 else
                                  'HEAVY_UNIFORM' if idle<1e-7 else 'MIXED')
            record['regime']=regime
            record['planned_imbalance']=imbalance
            record['hash_assignment_disagreement']=(len({tuple(z) for z in selected.values()})>1)
            record['hash_cost_disagreement']=(max(record['realized'].values())-
                                              min(record['realized'].values())>1e-5)
            record['changed']=int(np.sum(abs(np.asarray(d)-d0)>1e-7))
        except RuntimeError as exc:record.update(status='UNRESOLVED',error=str(exc))
        record['evaluation_seconds']=time.monotonic()-evaluation_start
        cases[name]=record
    add('CASE',d0)
    add('BOOKED',c.booked)
    risk=reviewed(c.booked,np.quantile(banks['A'],1.75/2.75,axis=0)-c.booked)
    add('RISK',risk)
    # Unrestricted feasible-advice benchmarks on EVERY component; same
    # reference algorithms and exact final planner as the existing study.
    try:
        d_per,_,accepted,status,_=prescription(c,d0,banks,seeds=(0,),
                 seconds=seconds,budget=pw_seconds if c.n<=20 else 2*pw_seconds)
        add('PER_WEEK',d_per,dict(search_status=status,search_accepted=bool(accepted)))
    except RuntimeError as exc:
        cases['PER_WEEK']=dict(status='UNRESOLVED',error=str(exc))
    try:
        d_per,_,accepted,status,_=prescription(c,d0,banks,seeds=SEEDS,
                 seconds=seconds,budget=pw_seconds if c.n<=20 else 2*pw_seconds)
        add('PER_WEEK_ROBUST',d_per,dict(search_status=status,search_accepted=bool(accepted)))
    except RuntimeError as exc:
        cases['PER_WEEK_ROBUST']=dict(status='UNRESOLVED',error=str(exc))
    for name,coeff in models.items():
        if coeff is None:
            cases[name]=dict(status='TRAINING_UNRESOLVED')
            continue
        d=reviewed_policy(c,rows,H,d0,coeff)
        add(name,d)
    # Unrestricted conditional stochastic benchmark. This assignment is
    # feasible but NOT deployable advice; never run it through the hash planner.
    # Bank A chooses it; B/C and realized durations evaluate it independently.
    saa=saa_mip(c,banks['A'],seconds=pw_seconds)
    reference=dict(status=saa.status,certified=bool(saa.complete),
                   lower=saa.lower,upper=saa.upper,seconds=saa.seconds,
                   assignment=None if saa.assignment is None else list(saa.assignment))
    if saa.assignment is not None:
        z=tuple(map(int,saa.assignment))
        reference.update(
            model={name:expected_reduced(c,z,draws) for name,draws in banks.items()},
            realized=float(evaluate(c,z,c.actual)['cost']))
        if saa.lower is not None and saa.lower>reference['model']['A']+1e-4:
            raise AssertionError('Unrestricted SAA lower bound exceeds its primal')
    # Independently trained case-advice methods may be weaker than the
    # unrestricted SAA, but they can never beat its certified global lower.
    for name,record in cases.items():
        if record.get('status')=='COMPLETE' and saa.lower is not None:
            if record['model']['A']<saa.lower-1e-4:
                raise AssertionError(f'{name} beats the unrestricted SAA dual bound')

    # One separate exact-QP (fixed CASE assignment), one global VF per instance,
    # one unrestricted SAA bound, one all-dimensional direct advice B&B.
    if cases['CASE']['status']=='COMPLETE':
        z=cases['CASE']['assignments']['0']
        qfill=vf_fixed_plan_lp(c,z,banks['A'],seconds=pw_seconds,d0=d0)
        if qfill.reviewed is not None:
            add('QFILL',qfill.reviewed,dict(vf_cert=qfill.to_json()))
        else:cases['QFILL']=dict(status='UNRESOLVED',vf_cert=qfill.to_json())
    vf=vf_per_week_milp(c,banks['A'],seconds=pw_seconds,d0=d0)
    if vf.reviewed is not None:
        add('VF_PW',vf.reviewed,dict(vf_cert=vf.to_json()))
    else:cases['VF_PW']=dict(status='UNRESOLVED',vf_cert=vf.to_json())
    pw=exact_box_policy(c,banks['A'],d0,seconds=pw_seconds,
                        solve_seconds=seconds,nodes=pw_nodes,saa_reference=saa)
    if pw['reviewed'] is not None:
        add('PW_BOUNDED',pw['reviewed'],dict(global_certificate=pw))
    else:cases['PW_BOUNDED']=dict(status='UNRESOLVED',global_certificate=pw)
    anchor=exact_box_policy(c,banks['A'],d0,seconds=pw_seconds,
                        solve_seconds=seconds,nodes=pw_nodes,restricted='case+saa',
                        saa_reference=saa)
    if anchor['reviewed'] is not None:
        add('PW_ANCHORED',anchor['reviewed'],dict(global_certificate=anchor))
    else:cases['PW_ANCHORED']=dict(status='UNRESOLVED',global_certificate=anchor)
    vfcg=value_function_exact(c,banks['A'],d0,seconds=pw_seconds,
                              separation_seconds=seconds,seed=0,saa_reference=saa)
    if vfcg['reviewed'] is not None:
        add('PW_VFCG',vfcg['reviewed'],dict(global_certificate=vfcg,
             optimization_hash_seed=0))
    else:cases['PW_VFCG']=dict(status='UNRESOLVED',global_certificate=vfcg)
    # Hindsight is an evaluation-only constant: no method uses these
    # realized outcomes to select a schedule, advice, or fitted coefficient.
    from surgery.anchored_weekly_solver import solve_week
    hindsight=solve_week(c,c.actual,seconds=max(seconds,600.),threads=1,tie=False)
    hindsight_record=dict(complete=hindsight.complete,
           cost=hindsight.value if hindsight.complete else None,
           lower=hindsight.bound,
           status=hindsight.status)
    # The exact PW algorithm never substitutes an unrestricted SAA assignment
    # as an inducible schedule. This gap is unconditional for the given bank.
    return ck,dict(site=c.group,monday=c.monday,cases=c.n,
                    methods=cases,hindsight=hindsight_record,saa_reference=reference,
                    elapsed=time.monotonic()-start)


def evaluate_all(args,ctx,models):
    root=ctx['root']/'full_evaluation';root.mkdir(exist_ok=True)
    base=ctx['held']['X']@ctx['held']['case_w']
    done={};todo=[]
    for ck,c,rows in ctx['eval']:
        p=root/(ck+'.json')
        if p.exists():done[ck]=json.loads(p.read_text());continue
        todo.append((ck,c,rows,base,ctx['H_eval'],models,ctx['pools'],
                     args.seconds,args.pw_seconds,args.pw_nodes))
    if todo:
        with ProcessPoolExecutor(max_workers=min(args.cores,len(todo)),
                                 mp_context=get_context('spawn')) as pool:
            futures=[pool.submit(_eval_component,job) for job in todo]
            for idx,future in enumerate(as_completed(futures)):
                ck,record=future.result()
                atomic_json(root/(ck+'.json'),record)
                done[ck]=record
                if (idx+1)%50==0:print('Saved',idx+1,'/',len(todo),'new complete components',flush=True)
    rows=[done[ck] for ck,_,_ in ctx['eval']]
    arms=sorted({m for row in rows for m in row['methods']})
    output=dict(fold=ctx['root'].name,n_cases=sum(r['cases'] for r in rows),
                components=len(rows),n_mondays=len(set(r['monday'] for r in rows)),arms={})
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
                 evaluation_wall_hours=sum(float(r['methods'][arm].get('evaluation_seconds',0.))
                                            for r in records)/3600.)
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
                          'Seed-specific optimization certificate not interchangeable '
                          'with the three-seed deployment objective.')
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
        output['inference_note']='Exploratory all-arm Holm; cluster calendar Mondays, hashes averaged within cluster'
    else:output['inference_note']='CASE not fully certified; no inferential contrasts'
    atomic_json(ctx['root']/'theory_summary.json',output)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workbook',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cores',type=int,default=15)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--pw-seconds',type=float,default=900)
    ap.add_argument('--pw-nodes',type=int,default=100)
    ap.add_argument('--shared-seconds',type=float,default=21600)
    ap.add_argument('--radius',type=float,default=32.)
    ap.add_argument('--rounds',type=int,default=10)
    ap.add_argument('--search-seconds',type=float,default=86400.,
                    help='Soft per-method directional training budget; stops between rounds')
    ap.add_argument('--fold',choices=['2013'],default='2013',
                    help='Frozen design: all 24,506 pre-2013 cases train; 8,774 2013 cases test')
    ap.add_argument('--acknowledge-final',action='store_true',
                    help='Explicitly acknowledge that the 2013 test is being opened once')
    ap.add_argument('--methods',default='VF,SHARED,OBS,EG,ROBUST,GLOBAL,ANCHORED')
    ap.add_argument('--global-nodes',type=int,default=40)
    ap.add_argument('--global-seconds',type=float,default=14400.)
    args=ap.parse_args()
    args.methods=[s.strip().upper() for s in args.methods.split(',')]
    if not set(args.methods)<=set(['VF','SHARED','SHARED_LEGACY','OBS','EG','ROBUST','GLOBAL','ANCHORED']):
        ap.error('Unknown method')
    if not 1<=args.cores<=15 or min(args.seconds,args.shared_seconds,args.pw_seconds,args.radius,args.search_seconds)<=0:
        ap.error('Invalid time, coefficient radius or cores')
    if not args.acknowledge_final:
        ap.error('Opening the 2013 test requires --acknowledge-final')
    args.out.mkdir(parents=True,exist_ok=True)
    secure_manifest(args.out,args)
    from run_weekly_rich import read_cohort
    raw,cohort=read_cohort(args.workbook)
    summaries=[]
    for fold in ('2013',):
        ctx=make_context(args,fold,cohort)
        models={'STAT':ctx['beta']}
        failures={}
        if 'SHARED' in args.methods:
            try:models['SHARED']=train_search(args,ctx,'SHARED')
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
                       seconds=args.seconds,target_restricted=False,cores=args.cores)
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
                         radius=args.radius,start_theta=prior)
                    models['SHARED_GLOBAL']=ctx['beta']+np.asarray(cert['theta'])
                except RuntimeError as exc:
                    failures['SHARED_GLOBAL']=str(exc);models['SHARED_GLOBAL']=None
            if 'ANCHORED' in args.methods:
                try:
                    targets=build_problems(ctx['train'],ctx['fit'],ctx['H'],
                         ctx['beta'],ctx['pools'],bounds_root,
                         seconds=args.seconds,target_restricted=True,cores=args.cores)
                    cert=global_shared_search(targets,ctx['beta'],
                         out=bounds_root/'target',seconds=args.global_seconds,
                         solve_seconds=args.seconds,max_nodes=args.global_nodes,
                         radius=args.radius)
                    models['ANCHORED_TARGET']=ctx['beta']+np.asarray(cert['theta'])
                except RuntimeError as exc:
                    failures['ANCHORED_TARGET']=str(exc);models['ANCHORED_TARGET']=None
        for method in args.methods:
            if method in ('SHARED','SHARED_LEGACY','GLOBAL','ANCHORED'):continue
            try:
                if method=='VF':
                    rec=train_vf(args,ctx)
                    models['VF_SHARED']=(None if rec['reviewed'] is None else
                                         ctx['beta']+np.asarray(rec['reviewed']))
                else:models[method]=train_search(args,ctx,method)
            except RuntimeError as exc:
                failures[method]=str(exc)
                models['VF_SHARED' if method=='VF' else method]=None
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