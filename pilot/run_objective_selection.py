#!/usr/bin/env python3
"""One bounded UHN objective-selection experiment, not a general leaderboard."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['SURGERY_HASH_SEEDS']='0'
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'

# The older full runner sets SURGERY_RESEARCH_LINK on import. Restore the
# caller's value so importing this module cannot alter unrelated legacy code.
_previous_link=os.environ.get('SURGERY_RESEARCH_LINK')
from pilot import run_theory_full as full
if _previous_link is None:os.environ.pop('SURGERY_RESEARCH_LINK',None)
else:os.environ['SURGERY_RESEARCH_LINK']=_previous_link
from surgery.objective_selection import build_library,load_library,train_objective,probe_landscape

# 10h16m stage limits, plus unused reserve until the 11h30m supervisor ceiling.
STAGES=(('setup',7200),('library',1800),('scan',1800),
        ('shared',4500),('vf',2700),('eg',2700),('anchored',1800),
        ('certificates',3600),('freeze',60),('evaluate',9000),('report',600))
METHODS=('SHARED','VF','EG_LIB','ANCHORED_T')
ARMS=('BOOKED','CASE','STAT','SHARED','VF','EG_LIB','ANCHORED_T')
ROOT_NAME='objective_selection_2013'


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
    tmp.replace(path)


def root(args):
    return args.out/ROOT_NAME


def load_context(args):
    with (root(args)/'context.pkl').open('rb') as f:
        return pickle.load(f)


def policy(args,name):
    path=root(args)/'policies'/(name+'.json')
    if not path.exists():return None
    rec=json.loads(path.read_text())
    coeff=rec.get('coeff')
    return None if coeff is None else np.asarray(coeff,float)


def publish(args,name,record):
    if record.get('coeff') is not None:
        coef=np.asarray(record['coeff'],float)
        if coef.shape!=(5,) or not np.isfinite(coef).all():
            raise AssertionError('Invalid shared five-feature coefficients')
    save(root(args)/'policies'/(name+'.json'),record)


def _recover(args,stage_name):
    name={'shared':'SHARED','vf':'VF','eg':'EG_LIB','anchored':'ANCHORED_T'}.get(stage_name)
    if not name or (root(args)/'policies'/(name+'.json')).exists():return
    file=root(args)/'training'/name/'progress.json'
    if not file.exists():return
    doc=json.loads(file.read_text())
    theta=np.asarray(doc['theta'],float)
    if theta.shape!=(5,):return
    ctx=load_context(args)
    publish(args,name,dict(method=name,coeff=(ctx['beta']+theta).tolist(),
       status='INTERRUPTED_WITH_CHECKPOINT',
       accepted_rounds=sum(bool(x.get('accepted')) for x in doc['history']),
       rounds_completed=len(doc['history']),provenance='incomplete_own_objective_search'))


def train(args,method,minutes):
    ctx=load_context(args)
    libraries=load_library(ctx,root(args)/'library')
    output=train_objective(ctx,method,libraries,root(args)/'training'/method,
         cores=args.cores,seconds=minutes*60.,rounds=6,
         solve_seconds=args.train_solve_seconds,task_seconds=150.)
    publish(args,method,output)


def _score_pair(ctx,method,coeff,out,deadline,cores):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from surgery.theory_cache import connect
    libraries=load_library(ctx,out.parents[2]/'library')
    state=dict(base=ctx['fit']['X']@ctx['fit']['case_w'],H=ctx['H'],
         method=method,pools=ctx['pools'],libraries=libraries,
         seconds=45.,fixed_banks=True,retry_seconds=75.,
         cache_path=str(ctx['root']/'exact_deployments.sqlite3'))
    connect(state['cache_path']).close()
    with ProcessPoolExecutor(max_workers=cores,mp_context=get_context('spawn'),
         initializer=full._score_initializer,initargs=(state,)) as pool:
        estimates={}
        for name,seed in (('A',0),('B',-1)):
            if deadline-time.monotonic()<2:break
            a,valid=full.parallel_score(ctx['train'],ctx['fit'],ctx['H'],
                  [ctx['beta'].tolist(),np.asarray(coeff,float).tolist()],
                  method,ctx['pools'],45.,cores,seed,
                  out/name,pool=pool,task_seconds=90.,deadline=deadline)
            estimates[name]=dict(complete=bool(valid.all()),
                 incumbent=float(a[0]) if valid[0] else None,
                 learned=float(a[1]) if valid[1] else None,
                 difference=float(a[1]-a[0]) if valid.all() else None)
        return estimates


def _eg_certificate_one(task):
    """Exact or bounded EG value on a fixed real training component."""
    from surgery.theory_exact import expected_gap_mip
    from surgery.anchored_weekly_solver import solve_week
    from surgery.planner import reachable
    c,d,d0,bank,label,seconds=task
    t0=time.monotonic()
    try:
        primary=solve_week(c,d,seconds=min(seconds/3,18.),
                           threads=1,tie=False)
        if not primary.complete:
            return dict(status='PRIMARY_UNCERTIFIED',label=label,site=c.group,
                        monday=c.monday,n=c.n)
        q=float(primary.value-480*c.k+np.sum(d)+30*c.n)
        remaining=max(.1,seconds-(time.monotonic()-t0))
        ans=expected_gap_mip(c,d,bank,seconds=remaining,threads=1)
        penalty=.001/2*np.square(d-d0).sum()
        return dict(status=ans.status,label=label,site=c.group,
            monday=c.monday,n=c.n,globally_closed=bool(ans.complete),
            lower=None if ans.lower is None else q+ans.lower+penalty,
            upper=None if ans.upper is None else q+ans.upper+penalty,
            lower_is_adversary_primal=ans.lower is not None,
            upper_is_adversary_dual=ans.upper is not None,
            seconds=time.monotonic()-t0)
    except Exception as exc:
        return dict(status='UNRESOLVED',error=repr(exc),label=label,
                    site=c.group,monday=c.monday,n=c.n)


def eg_final_certificates(ctx,coeff,output,*,cores=15,seconds=420.):
    """Ground exact EG intervals on a mixed-size TRAINING subset only."""
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    from surgery.objective_selection import _sample
    start=time.monotonic()
    base=ctx['fit']['X']@ctx['fit']['case_w']
    subset=_sample(ctx['train'],n=12)
    tasks=[]
    for ck,c,rows in subset:
        d0=full.reviewed(c.booked,base[rows])
        bank=full.scenarios(c,d0,ctx['pools'],20261020,count=50)
        for label,w in (('STAT',ctx['beta']),('EG_LIB',coeff)):
            d=full.reviewed_policy(c,rows,ctx['H'],d0,w)
            tasks.append((c,d,d0,bank,label,65.))
    result=[]
    with ProcessPoolExecutor(max_workers=min(cores,len(tasks)),
         mp_context=get_context('spawn')) as pool:
        pending=[pool.submit(_eg_certificate_one,t) for t in tasks]
        for future in as_completed(pending):
            result.append(future.result())
            save(output,dict(entries=result,total_targets=len(tasks),
                 certified=sum(r.get('globally_closed',False) for r in result),
                 elapsed=time.monotonic()-start,
                 interpretation='sampled original EG, not EG_LIB training objective'))
    return result


def certify(args):
    """Training-only 2x2: optimize own objective vs deployed objective."""
    ctx=load_context(args)
    end=time.monotonic()+3500.
    out=root(args)/'certificates';out.mkdir(parents=True,exist_ok=True)
    if policy(args,'EG_LIB') is not None and not (out/'eg_exact_sample.json').exists():
        eg_final_certificates(ctx,policy(args,'EG_LIB'),out/'eg_exact_sample.json',
                              cores=args.cores,seconds=420.)
    for method in METHODS:
        destination=out/(method+'.json')
        if destination.exists():continue
        coeff=policy(args,method)
        if coeff is None:
            save(destination,dict(status='NO_TRAINED_POLICY'));continue
        remaining=end-time.monotonic()
        if remaining<10:break
        budget=min(680.,remaining)
        own=_score_pair(ctx,method,coeff,out/method/'own',
                        time.monotonic()+budget/2,args.cores)
        remaining=end-time.monotonic()
        direct=_score_pair(ctx,'SHARED',coeff,out/method/'direct',
                     time.monotonic()+min(budget/2,max(1.,remaining)),args.cores) if remaining>5 else {}
        ownB=own.get('B',{})
        directB=direct.get('B',{})
        save(destination,dict(method=method,own=own,deployed=direct,
             own_objective_improved=(None if not ownB.get('complete') else
                                    ownB['difference']<0),
             deployed_objective_improved=(None if not directB.get('complete')
                                         else directB['difference']<0),
             status='COMPARED_FULL_TRAINING_BANKS' if
                      ownB.get('complete') and directB.get('complete') else
                      'PARTIAL_UNCERTIFIED'))
    # Global VF MIQP certificate is an OPTIONAL diagnostic, not VF training.
    remaining=end-time.monotonic()
    if remaining>300 and policy(args,'VF') is not None:
        try:
            from surgery.theory_exact import vf_shared_miqp
            cert=vf_shared_miqp(ctx['train'],ctx['fit'],ctx['H'],ctx['beta'],
                   ctx['pools'],seconds=min(remaining-60,1800.),
                   threads=args.cores,coefficient_radius=16.,
                   restart_dir=ctx['root']/'vf_optional_certificate',
                   start_coeff=policy(args,'VF'))
            save(out/'vf_global_bound.json',cert.to_json())
        except Exception as exc:
            save(out/'vf_global_bound.json',dict(status='CERTIFICATE_ERROR',
                                                  error=repr(exc)))


def stage(args):
    name=args.stage
    if name=='setup':
        from run_weekly_rich import read_cohort
        _,cohort=read_cohort(args.workbook)
        from types import SimpleNamespace
        setup_args=SimpleNamespace(out=args.out,workbook=args.workbook,
                   cores=args.cores,seconds=args.setup_solve_seconds,
                   setup_seconds=7000.)
        ctx=full.make_context(setup_args,'2013',cohort)
        folder=root(args);folder.mkdir(parents=True,exist_ok=True)
        p=folder/'context.pkl';tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as handle:pickle.dump(ctx,handle,pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        save(folder/'setup.json',dict(status='COMPLETE',
             training_cases=len(ctx['fit']['booked']),
             evaluation_cases=len(ctx['held']['booked']),
             training_components=len(ctx['train']),
             evaluation_components=len(ctx['eval'])))
    elif name=='library':
        ctx=load_context(args)
        library=build_library(ctx,root(args)/'library',cores=args.cores,
                              seconds=1700.,task_seconds=24.)
        save(root(args)/'library_status.json',dict(
              complete_components=len(library),total_components=len(ctx['train'])))
    elif name=='scan':
        ctx=load_context(args);libraries=load_library(ctx,root(args)/'library')
        probe_landscape(ctx,libraries,root(args)/'landscape',
                        cores=args.cores,seconds=1700.,solve_seconds=30.)
    elif name in ('shared','vf','eg','anchored'):
        train(args,dict(shared='SHARED',vf='VF',eg='EG_LIB',
                        anchored='ANCHORED_T')[name],
              dict(shared=73,vf=43,eg=43,anchored=28)[name])
    elif name=='certificates':certify(args)
    elif name=='freeze':
        existing=root(args)/'frozen.json'
        if existing.exists():return
        policies={}
        for method in METHODS:
            file=root(args)/'policies'/(method+'.json')
            policies[method]=(json.loads(file.read_text()) if file.exists()
                    else dict(status='NOT_TRAINED',coeff=None))
        save(existing,dict(train_n=24506,test_n=8774,
                           hash_seeds=[0],policy_class='five_feature',
                           policy_records=policies,arms=list(ARMS),
                           statement='Policies are frozen before 2013 evaluation'))
    elif name=='evaluate':
        ctx=load_context(args)
        frozen=json.loads((root(args)/'frozen.json').read_text())
        models={'STAT':np.asarray(ctx['beta'],float)}
        for method,record in frozen['policy_records'].items():
            models[method]=None if record.get('coeff') is None else np.asarray(record['coeff'],float)
        from types import SimpleNamespace
        settings=SimpleNamespace(eval_arms=list(ARMS),
                   eval_arm_seconds=100.,seconds=args.eval_solve_seconds,
                   pw_seconds=1.,pw_nodes=1,cores=args.cores)
        full.evaluate_all(settings,ctx,models)
    elif name=='report':summarize(args)
    else:raise ValueError(name)


def summarize(args):
    from surgery.weekly_planner import evaluate
    folder=root(args);context_path=folder/'context.pkl'
    if not context_path.exists():
        save(folder/'selection_summary.json',dict(status='SETUP_INCOMPLETE'));return
    ctx=load_context(args)
    shards=ctx['root']/'full_evaluation'/'per_arm'
    records=[]
    for ck,c,_ in ctx['eval']:
        outputs={}
        for arm in ARMS:
            file=shards/(ck+'__'+arm+'.json')
            outputs[arm]=json.loads(file.read_text()) if file.exists() else dict(status='NOT_EVALUATED')
        records.append((c,outputs))
    N=sum(c.n for c,_ in records)
    coverage={}
    for arm in ARMS:
        ok=[(c,row[arm]) for c,row in records if row[arm].get('status')=='COMPLETE']
        count=sum(c.n for c,_ in ok)
        coverage[arm]=dict(certified_cases=count,exact_fraction=count/N,
             certified_mean=(sum(np.mean(list(v['realized'].values())) for _,v in ok)/count
                       if count else None),
             unresolved_cases=N-count,
             attempted_components=sum(row[arm]['status']!='NOT_EVALUATED' for _,row in records))
    fallback={};weekdata={}
    for c,row in records:
        m=weekdata.setdefault(c.monday,dict(cases=0,values={a:0. for a in ARMS}))
        m['cases']+=c.n
        try:recorded=float(evaluate(c,c.original,c.actual)['cost'])
        except Exception:recorded=None
        for arm in ARMS:
            source=row[arm]
            if source.get('status')!='COMPLETE':source=row['CASE']
            if source.get('status')=='COMPLETE':
                value=float(np.mean(list(source['realized'].values())))
                kind='CERTIFIED' if source is row[arm] else 'CASE_FALLBACK'
            else:
                value=recorded
                kind='RECORDED_RETROSPECTIVE_FALLBACK' if recorded is not None else 'MISSING'
            fallback.setdefault(arm,dict(cost=0.,covered_cases=0,other_fallback_cases=0))
            if value is not None:
                fallback[arm]['cost']+=value
                fallback[arm]['covered_cases']+=c.n
                m['values'][arm]+=value
            if kind!='CERTIFIED':fallback[arm]['other_fallback_cases']+=c.n
    for arm,rec in fallback.items():
        rec['mean_realized_per_case']=(rec['cost']/N if rec['covered_cases']==N else None)
        del rec['cost']
    total_valid=all(rec['covered_cases']==N for rec in fallback.values())
    contrasts={}
    if total_valid:
        weeks=sorted(weekdata)
        den=np.array([weekdata[m]['cases'] for m in weeks],float)
        bootstrap=np.random.default_rng(20261022).integers(
                     len(weeks),size=(10000,len(weeks)))
        denom=den[bootstrap].sum(axis=1)
        for baseline in ('STAT','CASE','SHARED'):
            contrasts[baseline]={}
            for arm in ARMS:
                if arm==baseline:continue
                diff=np.array([weekdata[w]['values'][arm]-
                               weekdata[w]['values'][baseline] for w in weeks])
                est=float(diff.sum()/N)
                samples=diff[bootstrap].sum(axis=1)/denom
                lo,hi=np.quantile(samples,[.025/6,.975+(.025*(1-1/6))])
                centered=samples-est
                p=float((1+np.count_nonzero(abs(centered)>=abs(est)))/
                        (len(samples)+1))
                contrasts[baseline][arm]=dict(difference=est,
                         simultaneous_ci_approx=[float(lo),float(hi)],p_uncorrected=p)
    summaries={}
    frozen=folder/'frozen.json'
    if frozen.exists():summaries=json.loads(frozen.read_text())['policy_records']
    diagnostics={}
    for method in METHODS:
        file=folder/'certificates'/(method+'.json')
        diagnostics[method]=(json.loads(file.read_text()) if file.exists() else
                              dict(status='CERTIFICATES_NOT_FINISHED'))
    output=dict(status='FALLBACK_COMPLETE' if total_valid else 'PARTIAL_TEST',
          cases=N,calendar_weeks=len(weekdata),method_coverage=coverage,
          fallback_replay=fallback,contrasts=contrasts,
          training=summaries,training_objective_vs_deployed=diagnostics,
          landscape=(json.loads((folder/'landscape'/'landscape.json').read_text())
                     if (folder/'landscape'/'landscape.json').exists() else {}),
          caveats=['EG_LIB is a finite-schedule lower approximation to EG',
                   'ANCHORED_T uses library targets and is not exact inducibility',
                   'Axis scans do not certify global or unrestricted local minima',
                   'Recorded original assignment is retrospective, not a prospective fallback',
                   'Previously exposed 2013 cohort and modeled surgeon response'])
    save(folder/'selection_summary.json',output)


def supervisor(args):
    folder=root(args);folder.mkdir(parents=True,exist_ok=True)
    repository=Path(__file__).resolve().parents[1]
    deps=('pilot/run_objective_selection.py','surgery/objective_selection.py',
          'pilot/run_theory_full.py','surgery/theory_exact.py',
          'surgery/weekly_planner.py','surgery/anchored_weekly_solver.py',
          'surgery/advice_comparison.py','pilot/run_advice_comparison.py')
    contract=dict(workbook=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
          source={p:hashlib.sha256((repository/p).read_bytes()).hexdigest() for p in deps},
          methods=list(METHODS),arms=list(ARMS),stage_limits=list(STAGES),
          solve_seconds=dict(setup=args.setup_solve_seconds,
                 train=args.train_solve_seconds,eval=args.eval_solve_seconds),
          cores=args.cores,hash_seeds=[0])
    manifest=folder/'run_contract.json'
    if manifest.exists() and json.loads(manifest.read_text())!=contract:
        raise RuntimeError('Changed scientific specification: select a NEW --out directory')
    save(manifest,contract)
    start=time.monotonic();all_status={}
    for name,seconds in STAGES:
        left=11.5*3600-(time.monotonic()-start)
        if left<20:
            all_status[name]=dict(status='GLOBAL_LIMIT');break
        if name not in ('setup','report') and not (folder/'context.pkl').exists():
            all_status[name]=dict(status='DEPENDENCY_SETUP_INCOMPLETE');continue
        if name=='evaluate' and not (folder/'frozen.json').exists():
            all_status[name]=dict(status='NO_FROZEN_POLICIES');continue
        cmd=[sys.executable,str(Path(__file__).resolve()),'--stage',name,
             '--workbook',str(args.workbook),'--out',str(args.out),
             '--cores',str(args.cores),
             '--setup-solve-seconds',str(args.setup_solve_seconds),
             '--train-solve-seconds',str(args.train_solve_seconds),
             '--eval-solve-seconds',str(args.eval_solve_seconds)]
        tic=time.monotonic()
        logfile=folder/(name+'.log')
        with logfile.open('a') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True)
            try:
                proc.wait(timeout=min(seconds,left-10))
                status='COMPLETE' if proc.returncode==0 else 'FAILED'
            except subprocess.TimeoutExpired:
                status='WALL_LIMIT'
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
        if name in ('shared','vf','eg','anchored') and status!='COMPLETE':
            method=dict(shared='SHARED',vf='VF',eg='EG_LIB',
                        anchored='ANCHORED_T')[name]
            path=folder/'training'/method/'progress.json'
            if path.exists() and not (folder/'policies'/(method+'.json')).exists():
                try:
                    ctx=load_context(args)
                    saved=json.loads(path.read_text())
                    coef=ctx['beta']+np.asarray(saved['theta'],float)
                    publish(args,method,dict(method=method,coeff=coef.tolist(),
                        status='INTERRUPTED_WITH_CHECKPOINT',
                        accepted_rounds=sum(bool(h.get('accepted'))
                                            for h in saved['history']),
                        rounds_completed=len(saved['history'])))
                except Exception:pass
        all_status[name]=dict(status=status,seconds=time.monotonic()-tic,
                     exit_code=proc.returncode,log=str(logfile))
        save(folder/'run_status.json',all_status)
        print(name,status,flush=True)
    if not (folder/'selection_summary.json').exists():
        try:summarize(args)
        except Exception as exc:
            save(folder/'summary_error.json',dict(error=repr(exc)))


def main():
    # Only an actual experiment process requests the inverse-response link.
    os.environ['SURGERY_RESEARCH_LINK']='inverse'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=[n for n,_ in STAGES])
    parser.add_argument('--workbook',type=Path,default=Path(
           'data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    parser.add_argument('--out',type=Path,default=Path(
           'artifacts/objective_selection_12h'))
    parser.add_argument('--cores',type=int,default=15)
    parser.add_argument('--setup-solve-seconds',type=float,default=240.)
    parser.add_argument('--train-solve-seconds',type=float,default=45.)
    parser.add_argument('--eval-solve-seconds',type=float,default=40.)
    args=parser.parse_args()
    if not args.workbook.exists():parser.error('Workbook not found')
    if args.cores<1 or args.cores>15:parser.error('Expected 1-15 cores')
    if args.stage:stage(args)
    else:supervisor(args)

if __name__=='__main__':main()
