#!/usr/bin/env python3
"""One bounded SHARED/VF/GAP/ANCHORED method-selection campaign.

Use --mode run for the stage supervisor, --mode stage only to retry an
individual unfinished stage before the 2013 policies are frozen.
Only the report phase reads test outcomes for method comparison.
"""
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
os.environ['SURGERY_HASH_SEEDS']='0'  # exact same primary selector in every stage
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

from pilot import run_theory_full as full
from surgery.theory_stat import fit_stat_plus
from surgery.theory_selection import anchored_proposal,gap_proposal

# Upper limits in seconds; total = 676 minutes, reserve 14 minutes
# within the user's 11.5-hour ceiling.
STAGES=(
    ('setup',7200),('stat',900),('shared_base',2700),('vf',4500),
    ('shared_plus',4500),('gap',2700),('anchored',4500),
    ('freeze',60),('evaluate',12600),('report',900),
)
ARMS=('BOOKED','CASE','STAT_PLUS','SHARED_BASE','SHARED_PLUS',
      'VF_PLUS','GAP_PLUS','ANCHORED_PLUS')
LEARNERS=('STAT_PLUS','SHARED_BASE','VF_PLUS','SHARED_PLUS',
          'GAP_PLUS','ANCHORED_PLUS')


def atomic(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False))
    tmp.replace(path)


def policies_dir(args):
    p=args.out/'selection'/'policies';p.mkdir(parents=True,exist_ok=True)
    return p


def record_policy(args,name,coeff,status,**other):
    values=(None if coeff is None else np.asarray(coeff,float).tolist())
    if values is not None and (len(values)!=5 or not np.isfinite(values).all()):
        raise AssertionError('A learned five-feature policy must have finite coefficients')
    atomic(policies_dir(args)/(name+'.json'),dict(
        method=name,coeff=values,status=status,**other))


def get_policy(args,name):
    p=policies_dir(args)/(name+'.json')
    if not p.exists():return None
    obj=json.loads(p.read_text())
    return None if obj['coeff'] is None else np.asarray(obj['coeff'],float)


def context(args):
    path=args.out/'selection'/'context.pkl'
    if not path.exists():raise RuntimeError('SETUP_NOT_FINISHED')
    with open(path,'rb') as handle:return pickle.load(handle)


def _train_args(args,minutes,rounds=3):
    # train_search reads this light, explicit settings object.
    from types import SimpleNamespace
    return SimpleNamespace(
        cores=args.cores,seconds=args.train_solver_seconds,
        task_seconds=min(480.,minutes*60.),
        round_seconds=minutes*60.,search_seconds=minutes*60.,
        rounds=rounds,radius=32.,selection_fast=True,
        shared_seconds=minutes*60.)


def stage(args):
    stage=args.stage
    if stage=='setup':
        from run_weekly_rich import read_cohort
        _,cohort=read_cohort(args.workbook)
        from types import SimpleNamespace
        settings=SimpleNamespace(out=args.out,workbook=args.workbook,
                 cores=args.cores,seconds=args.setup_solver_seconds,
                 setup_seconds=7100.)
        ctx=full.make_context(settings,'2013',cohort)
        path=args.out/'selection'/'context.pkl'
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with open(tmp,'wb') as handle:pickle.dump(ctx,handle,pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        atomic(args.out/'selection'/'setup.json',dict(
            status='COMPLETE',training_cases=len(ctx['fit']['booked']),
            test_cases=len(ctx['held']['booked']),
            training_components=len(ctx['train']),test_components=len(ctx['eval'])))
        return
    if stage=='report':
        summarize(args)
        return
    if stage=='freeze':
        folder=policies_dir(args)
        existing=args.out/'selection'/'frozen.json'
        if existing.exists():return
        payload={name:(json.loads((folder/(name+'.json')).read_text())
                       if (folder/(name+'.json')).exists() else
                       dict(method=name,status='NOT_TRAINED',coeff=None))
                 for name in LEARNERS}
        assert all(v['coeff'] is None or len(v['coeff'])==5 for v in payload.values())
        # Freeze outputs independently from future reporting and diagnostic config.
        atomic(existing,dict(methods=payload,arms=list(ARMS),
                   hash_seeds=list(full.SEEDS),training_cases=24506,
                   test_cases=8774,contract='One chronological split; no test-based tuning'))
        return
    ctx=context(args)
    if stage=='stat':
        coeff=fit_stat_plus(ctx['fit'],ctx['H'],ctx['beta'],seconds=840.,
                            out=ctx['root']/'stat_plus.json')
        record_policy(args,'STAT_PLUS',coeff,'TRAINED',
                      source='same-five-feature-asymmetric-duration-loss')
    elif stage=='shared_base':
        cfg=_train_args(args,43,rounds=2)
        coeff=full.train_search(cfg,ctx,'SHARED_BASE')
        rec=json.loads((ctx['root']/'shared_base'/'trained.json').read_text())
        trained=bool(any(x.get('accepted') for x in rec['history']))
        record_policy(args,'SHARED_BASE',coeff,
             'PAIRED_TRAINING_HEURISTIC' if any(
               x.get('accepted_partial_training_only') for x in rec['history']) else
             'TRAINED_WITH_ACCEPTED_UPDATES' if trained else 'NO_ACCEPTED_UPDATE',
             source='bounded-original-directional-shared-search',
             training=rec)
    elif stage=='vf':
        cfg=_train_args(args,72)
        cert=full.train_vf(cfg,ctx)
        theta=cert.get('reviewed')
        coeff=None if theta is None else ctx['beta']+np.asarray(theta,float)
        record_policy(args,'VF_PLUS',coeff,
             'FEASIBLE_GAP_OPEN' if coeff is not None and not cert.get('complete')
             else 'CERTIFIED' if coeff is not None else 'NO_FEASIBLE_VF',
             source='single-level-global-VF-MIQP',
             lower=cert.get('lower'),upper=cert.get('upper'))
    elif stage=='shared_plus':
        baseline=get_policy(args,'SHARED_BASE')
        if baseline is None:baseline=np.asarray(ctx['beta'],float)
        path=ctx['root']/'shared_plus'/'progress.json'
        if not path.exists():
            full.atomic_json(path,dict(history=[],theta=(baseline-ctx['beta']).tolist()))
        cfg=_train_args(args,73,rounds=4)
        coeff=full.train_search(cfg,ctx,'SHARED_PLUS',
                              warm_start=get_policy(args,'VF_PLUS'))
        rec=json.loads((ctx['root']/'shared_plus'/'trained.json').read_text())
        improved=bool(any(x.get('accepted') for x in rec['history']) or
                      str(rec.get('warm_start','')).startswith('ACCEPTED'))
        record_policy(args,'SHARED_PLUS',coeff,
            'PAIRED_TRAINING_HEURISTIC' if any(
              x.get('accepted_partial_training_only') for x in rec['history']) else
            'TRAINED_WITH_ACCEPTED_UPDATES' if improved else 'NO_ACCEPTED_UPDATE',
            source='VF-warm-started-direct-shared-with-smaller-search',
            training=rec,initial_policy='SHARED_BASE')
    elif stage=='gap':
        baseline=get_policy(args,'SHARED_PLUS')
        if baseline is None:baseline=get_policy(args,'SHARED_BASE')
        if baseline is None:
            record_policy(args,'GAP_PLUS',None,'DEPENDENCY_UNRESOLVED')
            return
        result,details=gap_proposal(ctx,baseline,get_policy(args,'VF_PLUS'),
                args.out/'selection'/'gap',seconds=2550.,cores=args.cores)
        record_policy(args,'GAP_PLUS',result,
            details['status'],source='EG-interval-guided-direct-advice-candidate',
            provenance=details)
    elif stage=='anchored':
        baseline=get_policy(args,'SHARED_PLUS')
        if baseline is None:baseline=get_policy(args,'SHARED_BASE')
        if baseline is None:
            record_policy(args,'ANCHORED_PLUS',None,'DEPENDENCY_UNRESOLVED')
            return
        result,details=anchored_proposal(ctx,baseline,
                 args.out/'selection'/'anchored',seconds=4200.,cores=args.cores)
        record_policy(args,'ANCHORED_PLUS',result,
            details['status'],source='anchored-schedule-tangent-proposal-verified-directly',
            provenance=details)
    elif stage=='evaluate':
        frozen=args.out/'selection'/'frozen.json'
        if not frozen.exists():raise RuntimeError('POLICIES_NOT_FROZEN')
        names=json.loads(frozen.read_text())['methods']
        models={name:(None if obj.get('coeff') is None else np.asarray(obj['coeff'],float))
                for name,obj in names.items()}
        from types import SimpleNamespace
        cfg=SimpleNamespace(eval_arms=list(ARMS),eval_arm_seconds=180.,
               seconds=args.eval_solver_seconds,pw_seconds=1.,pw_nodes=1,
               cores=args.cores)
        full.evaluate_all(cfg,ctx,models)
    else:raise ValueError('Unknown stage '+str(stage))


def _week_ci(diff,den,rng,draws=5000,alpha=.01):
    N=len(diff)
    boots=rng.integers(0,N,size=(draws,N))
    estimates=diff[boots].sum(axis=1)/den[boots].sum(axis=1)
    estimate=float(diff.sum()/den.sum())
    lo,hi=np.quantile(estimates,[alpha/2,1-alpha/2])
    centered=estimates-estimate
    p=(1+np.count_nonzero(np.abs(centered)>=abs(estimate)))/(draws+1)
    return dict(effect=estimate,ci_familywise=[float(lo),float(hi)],p_unadjusted=float(p))


def summarize(args):
    path=args.out/'selection'/'context.pkl'
    if not path.exists():
        atomic(args.out/'selection'/'selection_summary.json',
               dict(status='SETUP_INCOMPLETE',meaningful_comparison=False))
        return
    ctx=context(args)
    directory=ctx['root']/'full_evaluation'/'per_arm'
    allrows=[]
    for ck,c,_ in ctx['eval']:
        records={}
        for arm in ARMS:
            file=directory/(ck+'__'+arm+'.json')
            records[arm]=(json.loads(file.read_text()) if file.exists() else
                          dict(status='NOT_EVALUATED'))
        allrows.append((c,records))
    N=sum(c.n for c,_ in allrows)
    totals={}
    for arm in ARMS:
        valid=[(c,rs[arm]) for c,rs in allrows
               if rs[arm].get('status')=='COMPLETE']
        certified_cases=sum(c.n for c,_ in valid)
        totals[arm]=dict(certified_cases=certified_cases,
             certified_only_cost_per_case=(sum(float(np.mean(list(
                  rec['realized'].values()))) for _,rec in valid)/certified_cases
                  if certified_cases else None),
             exact_coverage=certified_cases/N,
             certified_components=len(valid),
             attempted_components=sum(rs[arm]['status']!='NOT_EVALUATED'
                                      for _,rs in allrows),
             total_components=len(allrows),
             solver_hours=sum(float(rs[arm].get('method_wall_seconds',0))
                              for _,rs in allrows)/3600.)
    from surgery.weekly_planner import evaluate as evaluate_realized
    originals={}
    original_invalid=[]
    for c,rs in allrows:
        if rs['CASE'].get('status')!='COMPLETE':
            try:
                originals[(c.group,c.monday,int(min(c.case_ids)))]=float(
                    evaluate_realized(c,c.original,c.actual)['cost'])
            except (ValueError,AssertionError) as exc:
                original_invalid.append(dict(site=c.group,monday=c.monday,error=str(exc)))
    def fallback_cost(c,rs,arm):
        own=rs[arm]
        if own.get('status')=='COMPLETE':
            return float(np.mean(list(own['realized'].values()))),'METHOD'
        if rs['CASE'].get('status')=='COMPLETE':
            return float(np.mean(list(rs['CASE']['realized'].values()))),'CASE'
        recorded=originals.get((c.group,c.monday,int(min(c.case_ids))))
        return recorded,'RECORDED_ORIGINAL' if recorded is not None else 'UNAVAILABLE'
    common=[(c,rs) for c,rs in allrows
            if all(rs[a].get('status')=='COMPLETE' for a in ARMS)]
    common_n=sum(c.n for c,_ in common)
    common_exact=dict(cases=common_n,components=len(common),
          mean_cost_per_case={
              a:(sum(float(np.mean(list(rs[a]['realized'].values())))
                     for _,rs in common)/common_n if common_n else None)
              for a in ARMS})
    missing_case=sum(c.n for c,rs in allrows
                     if rs['CASE'].get('status')!='COMPLETE')
    fallback={}
    for arm in ARMS:
        paired=[];original_cases=0
        for c,rs in allrows:
            cost,source=fallback_cost(c,rs,arm)
            if cost is not None:paired.append((c,cost))
            if source=='RECORDED_ORIGINAL':original_cases+=c.n
        covered=sum(c.n for c,_ in paired)
        fallback[arm]=dict(covered_cases=covered,
            fallback_cases=sum(c.n for c,rs in allrows if
                               rs[arm].get('status')!='COMPLETE'),
            original_assignment_fallback_cases=original_cases,
            mean_cost_per_case=(sum(cost for _,cost in paired)/N if covered==N
                                else None))
    rng=np.random.default_rng(20261022)
    byweek={}
    valid_test=not original_invalid and all(x['covered_cases']==N
                                            for x in fallback.values())
    for c,rs in allrows:
        monday=byweek.setdefault(c.monday,dict(cases=0,cost={k:0. for k in ARMS}))
        monday['cases']+=c.n
        for arm in ARMS:
            cost,_=fallback_cost(c,rs,arm)
            if cost is not None:monday['cost'][arm]+=cost
    comparisons={};recommended=None;selection_reason=None
    frozen=args.out/'selection'/'frozen.json'
    policies=(json.loads(frozen.read_text())['methods'] if frozen.exists() else {})
    if valid_test:
        weeks=sorted(byweek);den=np.array([byweek[m]['cases'] for m in weeks],float)
        for arm in ARMS:
            if arm=='SHARED_PLUS':continue
            difference=np.array([byweek[m]['cost'][arm]-
                                 byweek[m]['cost']['SHARED_PLUS'] for m in weeks])
            comparisons[arm]=_week_ci(difference,den,rng,alpha=.05/7.)
        # Do NOT select by numeric rank alone. A challenger must improve by
        # at least 0.5 per case with simultaneous intervals excluding zero.
        admissible_statuses={
            'VF_PLUS':{'FEASIBLE_GAP_OPEN','CERTIFIED'},
            'GAP_PLUS':{'ACCEPTED','PAIRED_TRAINING_HEURISTIC'},
            'ANCHORED_PLUS':{'ACCEPTED','PAIRED_TRAINING_HEURISTIC'},
            'SHARED_BASE':{'TRAINED_WITH_ACCEPTED_UPDATES','PAIRED_TRAINING_HEURISTIC'},
        }
        base_status=policies.get('SHARED_BASE',{}).get('status')
        plus_status=policies.get('SHARED_PLUS',{}).get('status')
        allowed_shared={'TRAINED_WITH_ACCEPTED_UPDATES','PAIRED_TRAINING_HEURISTIC'}
        if totals['SHARED_PLUS']['exact_coverage']<.98 or (
              policies.get('SHARED_PLUS',{}).get('coeff') is None):
            selection_reason='SHARED_PLUS_NOT_SUFFICIENTLY_EVALUATED'
        elif plus_status in allowed_shared:
            reference='SHARED_PLUS'
        elif base_status in allowed_shared and (
                totals['SHARED_BASE']['exact_coverage']>=.98):
            reference='SHARED_BASE'
        else:
            selection_reason='NO_MEANINGFUL_DIRECT_SHARED_TRAINING'
        if selection_reason is None:
            eligible=[a for a in admissible_statuses
                    if a!=reference and totals[a]['exact_coverage']>=.98
                    and policies.get(a,{}).get('status') in admissible_statuses[a]
                    and comparisons[a]['ci_familywise'][1]<-.5]
            recommended=(min(eligible,key=lambda k:comparisons[k]['effect'])
                         if eligible else reference)
            selection_reason=('SUPERIOR_CHALLENGER' if eligible
                              else 'NO_CREDIBLE_CHALLENGER_IMPROVEMENT')
    else:
        selection_reason='FALLBACK_COVERAGE_INCOMPLETE'
    status=('INSUFFICIENT_FALLBACK_COVERAGE' if not valid_test else
            'COMPLETE_COMPARISON_WITH_FALLBACK' if all(
              fallback[k]['mean_cost_per_case'] is not None for k in ARMS)
            else 'PARTIAL')
    output=dict(status=status,cases=N,calendar_weeks=len(byweek),
         case_missing=missing_case,original_fallback_errors=original_invalid,
         arms=totals,common_certified=common_exact,
         case_fallback=fallback,contrasts_vs_SHARED_PLUS=comparisons,
         provisional_main_method=recommended,selection_reason=selection_reason,
         training=policies,
         important_note=('Multiple-comparison-adjusted week-bootstrap intervals; '
          'select only challengers with >=98% exact case coverage and more '
          'than 0.5 cost units/case improvement. Fallback is CASE, then '
          'recorded original assignment; this is a RETROSPECTIVE fallback '
          'replay, not an implementable prospective policy or exact '
          'deployment. Historical 2013 has been '
          'previously exposed. An incomplete learner is NOT evidence against '
          'its mathematical objective.'))
    atomic(args.out/'selection'/'selection_summary.json',output)


def supervise(args):
    args.out.mkdir(parents=True,exist_ok=True)
    root=args.out/'selection';root.mkdir(exist_ok=True)
    marker=root/'run_contract.json'
    source=Path(__file__).resolve()
    repo=source.parents[1]
    provenance=[
        'pilot/run_selection_12h.py','pilot/run_theory_full.py',
        'pilot/run_advice_comparison.py','surgery/theory_selection.py',
        'surgery/theory_stat.py','surgery/theory_exact.py',
        'surgery/anchored_core.py','surgery/anchored_weekly_solver.py',
        'surgery/weekly_planner.py','surgery/advice_comparison.py',
        'surgery/theory_cache.py',
    ]
    sources={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in provenance}
    payload=dict(workbook_sha256=hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                 source_sha256=sources,hash_seed=0,method_arms=list(ARMS),
                 budgets=list(STAGES),cores=args.cores,
                 training_solver_seconds=args.train_solver_seconds,
                 evaluation_solver_seconds=args.eval_solver_seconds,
                 setup_solver_seconds=args.setup_solver_seconds)
    if marker.exists() and json.loads(marker.read_text())!=json.loads(json.dumps(payload)):
        raise RuntimeError('Run specification changed. Use a new --out directory')
    atomic(marker,payload)
    elapsed_start=time.monotonic()
    wall_ceiling=11.5*3600.
    events={}
    for name,limit in STAGES:
        elapsed=time.monotonic()-elapsed_start
        remaining=wall_ceiling-elapsed
        if remaining<=20:
            events[name]=dict(status='GLOBAL_WALL_LIMIT');break
        allowed=min(float(limit),remaining-20.)
        if name=='evaluate' and not (root/'frozen.json').exists():
            events[name]=dict(status='NO_FROZEN_POLICIES')
            continue
        if name not in ('setup','report','freeze') and not (root/'context.pkl').exists():
            events[name]=dict(status='SETUP_NOT_AVAILABLE')
            continue
        command=[sys.executable,str(source),'--mode','stage','--stage',name,
                 '--workbook',str(args.workbook),'--out',str(args.out),
                 '--cores',str(args.cores),
                 '--train-solver-seconds',str(args.train_solver_seconds),
                 '--eval-solver-seconds',str(args.eval_solver_seconds),
                 '--setup-solver-seconds',str(args.setup_solver_seconds)]
        log=root/(name+'.log')
        t=time.monotonic()
        with open(log,'a') as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                                     start_new_session=True,env=os.environ.copy())
            try:
                process.wait(timeout=allowed)
                status='COMPLETE' if process.returncode==0 else 'FAILED'
            except subprocess.TimeoutExpired:
                status='TIME_LIMIT'
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=4.)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
        events[name]=dict(status=status,seconds=time.monotonic()-t,
                          exit_code=process.returncode,log=str(log))
        atomic(root/'run_status.json',dict(events=events,wall_seconds=time.monotonic()-elapsed_start))
        print(name,status,round(time.monotonic()-elapsed_start),flush=True)
    # Always attempt a partial final report without touching training or test
    # outcomes for method selection until the predeclared report phase.
    if not (root/'selection_summary.json').exists() and (root/'context.pkl').exists():
        try:summarize(args)
        except Exception as exc:atomic(root/'report_error.json',dict(error=repr(exc)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('run','stage'),default='run')
    ap.add_argument('--stage',choices=[name for name,_ in STAGES])
    ap.add_argument('--workbook',type=Path,default=Path(
                    'data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    ap.add_argument('--out',type=Path,default=Path('artifacts/paper_selection_12h'))
    ap.add_argument('--cores',type=int,default=15)
    ap.add_argument('--train-solver-seconds',type=float,default=120.)
    ap.add_argument('--eval-solver-seconds',type=float,default=90.)
    ap.add_argument('--setup-solver-seconds',type=float,default=240.)
    args=ap.parse_args()
    if not args.workbook.exists():ap.error('Workbook missing')
    if not 1<=args.cores<=15:ap.error('Supported cores: 1..15')
    if args.mode=='run':supervise(args)
    else:
        if args.stage is None:ap.error('--stage is required in stage mode')
        stage(args)


if __name__=='__main__':
    main()
