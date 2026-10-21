#!/usr/bin/env python3
"""Full UHN development study of direct, VF, EG, and prescription methods.

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
          'surgery/theory_reporting.py'})
    config=dict(workbook=digest(args.workbook),source={f:digest(p/f) for f in files},
                folds=args.fold,final_acknowledged=args.acknowledge_final,seconds=args.seconds,pw_seconds=args.pw_seconds,
                shared_seconds=args.shared_seconds,rounds=args.rounds,global_nodes=args.global_nodes,global_seconds=args.global_seconds,
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
    root=args.out/f'fold_{fold}'
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


def deploy(c,d,seconds,bank=None):
    z={}
    for seed in SEEDS:
        z[seed]=certified(c,d,seed,seconds)
    return z


def scores(c,d,d0,bank,method,seconds,seed=0):
    """Return fitted scenario objective. Every solver is complete or rejected."""
    if method=='VF':
        from surgery.anchored_weekly_solver import solve_week
        ans=solve_week(c,d,seconds=seconds,tie=False,threads=1)
        if not ans.complete or ans.assignment is None:
            raise RuntimeError('VF primary planner incomplete: '+ans.status)
        return vf_objective(c,ans.assignment,d,bank,d0,LAM)
    if method=='EG':
        ans=expected_gap_mip(c,d,bank,seconds=seconds,gamma=1.)
        if not ans.complete:
            raise RuntimeError('EG adversary incomplete: '+ans.status)
        return float(ans.value)+LAM/2*np.square(d-d0).sum()
    if method in ('SHARED','SHARED_LEGACY','OBS','ROBUST'):
        if method=='ROBUST':
            u=display(c.booked,d)
            cost=[]
            for alpha,h in SWEEP:
                altered=np.maximum(1.,c.booked+response(u,alpha,h))
                z=deploy(c,altered,seconds)
                cost.append(np.mean([expected_reduced(c,zi,bank) for zi in z.values()]))
            return float(np.mean(cost)+LAM/2*np.square(d-d0).sum())
        z=deploy(c,d,seconds)
        if method=='OBS':
            # Realized outcomes are allowed in FITTING ONLY, not evaluation.
            bank=np.asarray(c.actual,float)[None,:]
        return float(np.mean([expected_reduced(c,zi,bank) for zi in z.values()])
                     +LAM/2*np.square(d-d0).sum())
    raise ValueError(method)


def _train_component(task):
    ck,c,rows,base,H,beta,coefficients,method,pools,seconds,round_id=task
    d0=reviewed(c.booked,base[rows]); bank=scenarios(c,d0,pools,
                     (20261020 if round_id>=0 else 20261021+1009*(-round_id-1)),count=50)
    result=[]
    for coeff in coefficients:
        d=reviewed_policy(c,rows,H,d0,np.asarray(coeff))
        try:result.append(dict(value=scores(c,d,d0,bank,method,seconds),complete=True))
        except RuntimeError as exc:result.append(dict(value=None,complete=False,status=str(exc)))
    return ck,c.n,result


def parallel_score(items,data,H,coefficients,method,pools,seconds,cores,round_id,folder):
    """Resume all certified component/candidate evaluations individually."""
    folder.mkdir(parents=True,exist_ok=True)
    ids=[key(c) for _,c,_ in items]
    tasks=[];gathered={}
    base=data['X']@data['case_w']
    identity=hashlib.sha256(json.dumps(dict(method=method,coefficients=coefficients,
                        bank_round=round_id),sort_keys=True).encode()).hexdigest()
    for ck,c,rows in items:
        filename=folder/(ck+'.json')
        if filename.exists():
            record=json.loads(filename.read_text())
            if record.get('identity') != identity:
                raise RuntimeError(f'Stale candidate-score checkpoint for {ck}; use a fresh artifact directory')
            if len(record['results'])==len(coefficients):
                gathered[ck]=record
                continue
        tasks.append((ck,c,rows,base,H,None,coefficients,method,pools,seconds,round_id))
    # Per-worker thread counts are fixed; a Gurobi solve never uses all cores.
    import multiprocessing
    if tasks:
        with ProcessPoolExecutor(max_workers=min(cores,len(tasks)),
                                 mp_context=get_context('spawn')) as exe:
            futures=[exe.submit(_train_component,t) for t in tasks]
            for future in as_completed(futures):
                ck,n,res=future.result()
                record=dict(cases=n,results=res,identity=identity)
                atomic_json(folder/(ck+'.json'),record)
                gathered[ck]=record
    assert len(gathered)==len(items)
    valid=np.ones(len(coefficients),bool)
    score=np.zeros(len(coefficients))
    for ck in ids:
        for k,x in enumerate(gathered[ck]['results']):
            if not x['complete']:valid[k]=False
            else:score[k]+=x['value']
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
    theta=np.zeros(5);history=[]
    flexible=[x for x in items if any(len(x[1].allowed[int(s)])>1 for s in x[1].surgeon)]
    candidates=flexible or items
    rg=np.random.default_rng(20261020)
    sampled=rg.choice(len(candidates),size=min(150,len(candidates)),replace=False)
    shortlist_items=[candidates[int(j)] for j in sampled]
    for round_id in range(args.rounds):
        step=8./2**round_id
        proposed=[beta+theta]
        for v in directions:
            for sign in (-1.,1.):
                proposed.append(beta+theta+sign*step*np.asarray(v))
        fititems=[(ck,c,rows) for ck,c,rows in shortlist_items]
        preview,valid=parallel_score(fititems,fit,H,[p.tolist() for p in proposed],
                             method,ctx['pools'],args.seconds,args.cores,round_id,
                             root/f'round_{round_id:02d}'/'screen')
        order=sorted((i for i in range(1,len(proposed)) if valid[i]),key=lambda i:preview[i])[:5]
        indices=[0]+order
        allscore,valid=parallel_score(items,fit,H,[proposed[i].tolist() for i in indices],
                           method,ctx['pools'],args.seconds,args.cores,round_id,
                           root/f'round_{round_id:02d}'/'all')
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
                         root/f'round_{round_id:02d}'/'confirm_B')
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
                 guarantee='full-Bank-A candidate search with independent per-round Bank-B confirmation; outer search not globally optimal or population-certified'))
    return coeff


def train_vf(args,ctx):
    p=ctx['root']/'vf_global.json'
    if p.exists():return json.loads(p.read_text())
    cert=vf_shared_miqp(ctx['train'],ctx['fit'],ctx['H'],ctx['beta'],ctx['pools'],
           seconds=args.shared_seconds,threads=args.cores,
           coefficient_radius=args.radius)
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
                        solve_seconds=seconds,nodes=pw_nodes)
    if pw['reviewed'] is not None:
        add('PW_BOUNDED',pw['reviewed'],dict(global_certificate=pw))
    else:cases['PW_BOUNDED']=dict(status='UNRESOLVED',global_certificate=pw)
    anchor=exact_box_policy(c,banks['A'],d0,seconds=pw_seconds,
                        solve_seconds=seconds,nodes=pw_nodes,restricted='case+saa')
    if anchor['reviewed'] is not None:
        add('PW_ANCHORED',anchor['reviewed'],dict(global_certificate=anchor))
    else:cases['PW_ANCHORED']=dict(status='UNRESOLVED',global_certificate=anchor)
    vfcg=value_function_exact(c,banks['A'],d0,seconds=pw_seconds,
                              separation_seconds=seconds,seed=0)
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
                    methods=cases,hindsight=hindsight_record,elapsed=time.monotonic()-start)


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
    ap.add_argument('--fold',choices=['0','1','both','2013'],default='both')
    ap.add_argument('--acknowledge-final',action='store_true',
                    help='Use once only after freezing methods; no result-based retuning')
    ap.add_argument('--methods',default='VF,SHARED,OBS,EG,ROBUST,GLOBAL,ANCHORED')
    ap.add_argument('--global-nodes',type=int,default=40)
    ap.add_argument('--global-seconds',type=float,default=14400.)
    args=ap.parse_args()
    args.methods=[s.strip().upper() for s in args.methods.split(',')]
    if not set(args.methods)<=set(['VF','SHARED','SHARED_LEGACY','OBS','EG','ROBUST','GLOBAL','ANCHORED']):
        ap.error('Unknown method')
    if not 1<=args.cores<=15 or min(args.seconds,args.shared_seconds,args.pw_seconds,args.radius)<=0:
        ap.error('Invalid time, coefficient radius or cores')
    if args.fold=='2013' and not args.acknowledge_final:
        ap.error('2013 holdout requires explicit --acknowledge-final after all choices are frozen')
    args.out.mkdir(parents=True,exist_ok=True)
    secure_manifest(args.out,args)
    from run_weekly_rich import read_cohort
    raw,cohort=read_cohort(args.workbook)
    summaries=[]
    for fold in ([0,1] if args.fold=='both' else ([args.fold] if args.fold=='2013' else [int(args.fold)])):
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
                       seconds=args.seconds,target_restricted=False)
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
                         seconds=args.seconds,target_restricted=True)
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
    atomic_json(args.out/'summary.json',dict(folds=summaries))
    if args.fold=='both':
        from surgery.theory_reporting import pooled_development
        pooled=pooled_development(args.out)
        print('Pooled development comparison:',pooled['n_cases'],'cases,',
              pooled['n_mondays'],'calendar Mondays',flush=True)

if __name__=='__main__':main()