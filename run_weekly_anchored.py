#!/usr/bin/env python3
"""End-to-end anchored regret learner; frozen 74-feature weekly UHN cohort.

Stages: references -> train -> certify -> evaluate.

Use a fresh --output directory per (site, fixed hyperparameter) experiment.
Only the `evaluate` stage scores the 2012 development-validation cases.
No 2013 outcomes are read. No validation-based hyperparameter selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from run_weekly_rich import load_matrices,source_fingerprint
from run_weekly_train import load_weeks,check_week_cases
from surgery.anchored_weekly_solver import evaluate_week
from surgery.planner import display_and_duration
from surgery.weekly_training import save_json
from surgery.anchored_training import build_reference_library, learn, verify_policy
from surgery.anchored_core import key
from surgery.anchored_master import prepare_components
from surgery.anchored_oracles import certified_bound

SOURCE_FILES=('run_weekly_anchored.py','surgery/anchored_core.py',
              'surgery/anchored_oracles.py','surgery/anchored_master.py',
              'surgery/anchored_training.py',
              'surgery/anchored_weekly_solver.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(args,meta,source,policies):
    root=Path(__file__).resolve().parent
    code=hashlib.sha256()
    for f in SOURCE_FILES:
        code.update(f.encode());code.update((root/f).read_bytes())
    return {'trainer_source_sha256':code.hexdigest(),
            'feature_source_sha256':source_fingerprint(),
            'workbook_sha256':sha(args.workbook),
            'matrix_sha256':meta['matrix_sha256'],
            'feature_names':meta['feature_names'],
            'baseline_models_sha256':source,
            'site':args.site,'gamma':args.gamma,
            'lambda_l1':args.lambda_l1,'prox':args.prox,
            'coef_prox':args.coef_prox,'ridge':args.ridge,
            'reference_penalty':args.reference_penalty,
            'line_fractions':args.line_fractions,'trust_growth':args.trust_growth,
            'terminal_separation':args.terminal_separation,
            'max_terminal_restarts':args.max_terminal_restarts,
            'initial_method':args.initial_method,'tight_surrogate':args.tight_surrogate,
            'pDCA_iterations':args.iterations,
            'closure_rounds':args.closure_rounds,'max_outer':args.max_outer,
             'exact_sep_batch':args.exact_sep_batch,
             'learner_variant':'schedule_specific_exact_tangent_adaptive_v3',
             'planner_tie_rule':'integer_hash_unique_v1',
             'hash_seed':args.hash_seed,
            'policy':f'booked + exact response(clip(X@w)), {len(meta["feature_names"])} shared features',
            'target':'verified empirical scheduling regret',
            'period':'2011-12 development training; no 2013 outcomes'}


def initialize(args):
    meta,frames=load_matrices(args.artifacts)
    origin=json.loads((args.artifacts/'learning_settings.json').read_text())
    if (origin['workbook_sha256']!=sha(args.workbook)
            or origin['matrix_sha256']!=meta['matrix_sha256']
            or origin['feature_code_sha256']!=source_fingerprint()
            or not np.isclose(args.alpha,origin['alpha'])
            or not np.isclose(args.h,origin['h'])):
        raise ValueError('Workbook, X or behavioral response differs from frozen CASE experiment')
    if (len(meta['feature_names']) != frames['train']['X'].shape[1]
            or meta['feature_names'][0]!='intercept'):
        raise ValueError('Policy features must exactly match frozen manifest')
    if len(frames['train']['case_id'])!=18705 or len(frames['validation']['case_id'])!=5779:
        raise ValueError('Unexpected train/validation cohort')
    filenames={'CASE':args.artifacts/'model_case.json',
               'LASSO':args.artifacts/'lasso_policy.json',
               'VF':args.artifacts/'vf_progress.json'}
    if args.gap_model is not None:
        filenames['VF_GAP']=args.gap_model
    if args.initial_policy is not None:
        filenames['SEED']=args.initial_policy
    raw={k:json.loads(path.read_text()) for k,path in filenames.items()}
    if raw['CASE'].get('settings')!=origin or raw['VF'].get('settings')!=origin:
        raise ValueError('CASE/VF coefficients not from frozen training settings')
    if not raw['VF'].get('complete'):
        raise ValueError('VF baseline incomplete')
    if raw['LASSO'].get('features')!=meta['feature_names']:
        raise ValueError('LASSO feature mismatch')
    if 'SEED' in raw and (raw['SEED'].get('feature_dimensions')!=len(meta['feature_names'])
                          or raw['SEED'].get('complete') is not True):
        raise ValueError('Warm-start policy must be completed, 74-feature model')
    if 'VF_GAP' in raw and (not raw['VF_GAP'].get('complete')
                          or raw['VF_GAP']['settings']['feature_source_sha256']!=source_fingerprint()
                          or raw['VF_GAP']['settings']['matrix_sha256']!=meta['matrix_sha256']):
        raise ValueError('Gap-VF experiment is incomplete or incompatible')
    policies={k:np.asarray(d['weights' if k=='LASSO' else 'w'],float)
              for k,d in raw.items()}
    for k,w in policies.items():
        if w.shape!=(len(meta['feature_names']),) or not np.isfinite(w).all():
            raise ValueError(f'Incomplete feature vector for {k}')
    ident=settings(args,meta,{k:sha(path) for k,path in filenames.items()},policies)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'anchored_settings.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=ident:
            raise ValueError('Anchored experiment code/data/models/settings changed. '
                             'Use a NEW output directory; previous checkpoints '
                             'cannot be migrated into this experiment.')
    else:
        save_json(path,ident)
    weeks=load_weeks(args.workbook,meta,'train')
    if args.site!='all':weeks=[w for w in weeks if w.group==args.site]
    # Restore row positions in frozen train matrix; NEVER refit, reorder or discard columns.
    id_in_X={int(cid):i for i,cid in enumerate(frames['train']['case_id'])}
    ids=[int(i) for w in weeks for i in w.case_ids]
    if len(set(ids))!=len(ids) or not set(ids).issubset(id_in_X):
        raise ValueError('Duplicated or unrecognized weekly case')
    rows=np.array([id_in_X[cid] for cid in ids],int)
    data={name:array[rows].copy() for name,array in frames['train'].items()}
    index={int(cid):i for i,cid in enumerate(data['case_id'])}
    if sum(w.n for w in weeks)!=len(data['case_id']):
        raise AssertionError('Case count mismatch')
    if args.site=='all':
        full_X=data['X'].astype(float)
        if (full_X.shape!=(18705,len(meta['feature_names'])) or not np.allclose(full_X[:,0],1.)
                or np.any(np.std(full_X[:,1:],axis=0)<1e-10)):
            raise ValueError('Full 74-feature design is incomplete or contains dropped/constant features')
    if args.site=='all' and (len(weeks)!=180 or len(rows)!=18705):
        raise ValueError('Full training must use all 180 weeks and 18705 cases')
    if args.site!='all' and len(weeks)!=60:
        raise ValueError('A full-site pilot must use all 60 training weeks')
    if 'SEED' in raw and (raw['SEED'].get('training_cases')!=len(data['case_id'])
                          or raw['SEED'].get('training_site_weeks')!=len(weeks)):
        raise ValueError('Warm-start model cohort or site differs from experiment')
    return meta,frames,weeks,index,data,policies,ident


def train(args):
    meta,frames,weeks,index,data,policies,ident=initialize(args)
    refs_path=args.output/'anchored_references.json'
    if not refs_path.exists():
        raise ValueError('Run --stage references before training')
    refs=json.loads(refs_path.read_text())
    if len(refs)!=sum(len(w.components()) for w in weeks) or any(
            not v.get('complete') for v in refs.values()):
        raise ValueError('References are incomplete')
    state=learn(args.output,weeks,index,data,refs,policies.get('SEED',policies[args.initial_method]),args.alpha,args.h,
                gamma=args.gamma,lam=args.lambda_l1,
                prox=args.prox,coef_prox=args.coef_prox,outer=args.max_outer,
                iterations=args.iterations,closure_rounds=args.closure_rounds,
                seconds=args.seconds,sep_seconds=args.sep_seconds,
                master_seconds=args.master_seconds,threads=args.threads,
                workers=args.workers,master_threads=args.master_threads,
                improvement=args.min_improvement,
                 sep_batch=args.exact_sep_batch,ridge=args.ridge,
                 reference=args.reference_penalty,
                 reference_w=policies['LASSO'] if args.reference_penalty else None,
                 line_fractions=tuple(float(q) for q in args.line_fractions.split(',')),
                 terminal_separation=args.terminal_separation,
                 max_terminal_restarts=args.max_terminal_restarts,
                 trust_growth=args.trust_growth,tight_surrogate=args.tight_surrogate)
    print('Anchored result:',state['stopping'],'accepted moves=',len(state['accepted']),flush=True)


def _certificate_week(task):
    blocks,duration,gamma,seconds,threads,report_gap,verified,tight_surrogate=task
    from surgery.anchored_oracles import certified_bound
    result={}
    for b in blocks:
        c=b['component'];d=duration[b['rows']]
        result[b['key']]=certified_bound(c,b['anchor'],d,b['oracle_cost'],
                None,gamma,seconds=seconds,threads=threads,exact=True,
                report_gap=report_gap,deployed_record=verified[b['key']],
                tight_surrogate=tight_surrogate)
    return result


def certify(args):
    _,_,weeks,index,data,_,_=initialize(args)
    refs=json.loads((args.output/'anchored_references.json').read_text())
    model=json.loads((args.output/'anchored_progress.json').read_text())
    if not model.get('complete'):raise ValueError('Training is not complete')
    w=np.asarray(model['w'],float)
    X=data['X'].astype(float)
    booked=data['booked']
    _,duration=display_and_duration(X@w,booked,args.alpha,args.h)
    # Certification must use exactly the same incumbent-aware reference rule as
    # training, including ties. The duration-keyed deployed cache normally makes
    # this free after the completed training run.
    incumbent=verify_policy(args.output,'certification_incumbent',weeks,index,
                            X,booked,w,args.alpha,args.h,
                            seconds=args.seconds,threads=args.threads,
                            workers=args.workers)
    tight_surrogate=bool(getattr(args,'tight_surrogate',False))
    blocks=prepare_components(weeks,index,refs,model['library'],
                              current=incumbent,current_duration=duration,
                              tight_surrogate=tight_surrogate)
    path=args.output/'anchored_bound_certificates.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    groups={}
    for b in blocks:
        if (report.get(b['key'],{}).get('certified') and report[b['key']].get('separation_exact')
                and (not args.report_gap or 'vf_gap_interval' in report[b['key']])):
            continue
        c=b['component']
        groups.setdefault((c.group,c.monday),[]).append(b)
    tasks=[(v,duration,args.gamma,args.seconds,args.threads,args.report_gap,incumbent,tight_surrogate)
           for v in groups.values()]
    def persist(rows):
        report.update(rows);save_json(path,report)
    print(f'Exact anchored bound: {len(tasks)} pending site-weeks',flush=True)
    if args.workers==1:
        for task in tasks:persist(_certificate_week(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_certificate_week,t) for t in tasks]
            for job in as_completed(future):persist(job.result())
    unresolved=[k for k,v in report.items() if not v.get('certified')]
    all_keys={b['key'] for b in blocks}
    missing=all_keys-set(report)
    if unresolved or missing:
        raise RuntimeError(f'Uncertified anchored bound: {len(unresolved)} unresolved, {len(missing)} missing')
    n=len(data['case_id'])
    summary={'regret_per_case':sum(v['regret'] for v in report.values())/n,
             'anchor_upper_per_case':sum(v['tie_safe_upper'] for v in report.values())/n,
             'all_exact_separation':all(v.get('separation_exact') for v in report.values()),
             'interpretation':'Exact tangible deployed regret, conservative anchored bound with tie correction'}
    if args.report_gap:
        intervals=[report[b['key']].get('vf_gap_interval') for b in blocks]
        if all(x is not None for x in intervals):
            summary['vf_gap_interval_per_case']=[
                sum(v[j] for v in intervals)/n for j in (0,1)]
            summary['vf_gap_all_exact']=all(report[b['key']].get('vf_gap_exact') for b in blocks)
        else:
            summary['vf_gap_interval_per_case']=None
            summary['vf_gap_all_exact']=False
    save_json(args.output/'anchored_bound_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def _eval_week(task):
    week,local,seconds,threads=task
    return evaluate_week((week,local,seconds,threads))


def evaluate(args):
    if args.site!='all':
        raise ValueError('Development validation reserved for completed full-cohort training')
    meta,frames,_,_,data,policies,ident=initialize(args)
    result=json.loads((args.output/'anchored_progress.json').read_text())
    if not result.get('complete') or result.get('training_cases')!=18705:
        raise ValueError('Require completed full-cohort model')
    policies['ANCHORED']=np.asarray(result['w'],float)
    valid=frames['validation'];X=valid['X'].astype(float);booked=valid['booked']
    predictions={'BOOKED':booked.copy()}
    for k,w in policies.items():
        _,d=display_and_duration(X@w,booked,args.alpha,args.h)
        predictions[k]=d
    weeks=load_weeks(args.workbook,meta,'validation')
    idx=check_week_cases(weeks,valid)
    if len(weeks)!=54:raise ValueError('Validation protocol requires exactly 54 group-weeks')
    path=args.output/'anchored_validation.csv';fileident=args.output/'anchored_validation_settings.json'
    ident_val={'train':ident,'model_sha256':sha(args.output/'anchored_progress.json'),
               'compared_policies':sorted(predictions)}
    if fileident.exists():
        if json.loads(fileident.read_text())!=ident_val:
            raise ValueError('Validation identity mismatch; do not overwrite')
    else:save_json(fileident,ident_val)
    records={}
    if path.exists():
        for record in pd.read_csv(path).to_dict('records'):
            records[(record['group'],record['monday'],record['method'])]=record
    tasks=[]
    for week in weeks:
        keys=[(week.group,week.monday,name) for name in predictions]
        if all(k in records and records[k]['complete']==True for k in keys):continue
        for k in keys:records.pop(k,None)
        rows=np.array([idx[int(cid)] for cid in week.case_ids],int)
        tasks.append((week,{name:pred[rows] for name,pred in predictions.items()},
                      args.seconds,args.threads))
    def persist(vals):
        for x in vals:
            records[(x['group'],x['monday'],x['method'])]=x
        temp=path.with_suffix('.tmp')
        pd.DataFrame(list(records.values())).to_csv(temp,index=False);temp.replace(path)
    if args.workers==1:
        for t in tasks:persist(_eval_week(t))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_eval_week,t) for t in tasks]
            for job in as_completed(future):persist(job.result())
    df=pd.DataFrame(records.values())
    if (len(df)!=54*len(predictions) or not df.complete.eq(True).all()):
        raise RuntimeError('Incomplete evaluation; retry with greater --seconds')
    summary={'methods':{},'by_group':{}}
    for name,g in df.groupby('method'):
        summary['methods'][name]={'cases':int(g.cases.sum()),
            'regret_per_case':float(g.regret.sum()/g.cases.sum()),
            'realized_cost':float(g.realized_cost.sum()),
            'mae':float(g.abs_error.sum()/g.cases.sum()),
            'room_days_over_720':int(g.room_days_over_720.sum()),
            'room_days_over_960':int(g.room_days_over_960.sum()),
            'room_days_over_1440':int(g.room_days_over_1440.sum()),
            'overtime_minutes':float(g.overtime.sum()),
            'moved_cases':int(g.moved_cases.sum())}
    for (group,name),g in df.groupby(['group','method']):
        summary['by_group'].setdefault(group,{})[name]={
            'regret_per_case':float(g.regret.sum()/g.cases.sum())}
    pivot=df.pivot(index=['monday','group'],columns='method',values='realized_cost')
    calendar=pivot.groupby(level='monday').sum()
    if len(calendar)!=18:raise AssertionError('Unpaired validation calendar weeks')
    rng=np.random.default_rng(20261013)
    draw=rng.integers(0,len(calendar),(20000,len(calendar)))
    summary['calendar_week_paired_bootstrap']={}
    for base in ('BOOKED','CASE','LASSO','VF','VF_GAP'):
        if base not in calendar:continue
        d=(calendar['ANCHORED']-calendar[base]).to_numpy(float)
        den=calendar[base].to_numpy(float)
        pct=100*d[draw].sum(axis=1)/den[draw].sum(axis=1)
        summary['calendar_week_paired_bootstrap']['ANCHORED_minus_'+base]={
            'change':float(d.sum()),'percent':float(100*d.sum()/den.sum()),
            'ci_95_percent':[float(v) for v in np.quantile(pct,[.025,.975])]}
    summary['statement']='Development validation only; no 2013 test. All comparisons paired over same 54 site-weeks.'
    save_json(args.output/'anchored_validation_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)



def sensitivity(args):
    """Prespecified hash seeds 1--4; keep seed-0 training and validation frozen.

    Separate resumable outputs; changing the selector here does not change
    weights, fitted baselines, the training checkpoint, or the 2013 holdout.
    """
    if args.site!='all' or args.hash_seed!=0:
        raise ValueError('Sensitivity uses the frozen all-site hash-seed-0 model')
    meta,frames,_,_,_,policies,ident=initialize(args)
    model_path=args.output/'anchored_progress.json'
    model=json.loads(model_path.read_text())
    if not model.get('complete') or model.get('training_cases')!=18705:
        raise ValueError('Sensitivity requires a completed full-cohort model')
    policies['ANCHORED']=np.asarray(model['w'],float)
    split=args.sensitivity_split
    frame=frames[split]
    X=frame['X'].astype(float)
    booked=frame['booked']
    methods=(['ANCHORED'] if split=='train'
             else ['BOOKED','CASE','LASSO','VF','VF_GAP','ANCHORED'])
    predictions={}
    for method in methods:
        if method=='BOOKED':
            predictions[method]=booked.copy()
        else:
            _,pred=display_and_duration(X@policies[method],booked,args.alpha,args.h)
            predictions[method]=pred
    weeks=load_weeks(args.workbook,meta,split)
    expected_weeks=180 if split=='train' else 54
    if len(weeks)!=expected_weeks:
        raise ValueError('Sensitivity split lost site-weeks')
    idx=check_week_cases(weeks,frame)
    identity={'base_settings':ident,'model_sha256':sha(model_path),
              'split':split,'hash_seeds':[1,2,3,4],'methods':methods}
    marker=args.output/f'hash_sensitivity_{split}_identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=identity:
            raise ValueError('Sensitivity identity mismatch: use a fresh run')
    else:
        save_json(marker,identity)
    report={'status':'FROZEN-POLICY HASH SENSITIVITY',
            'split':split,'seed0_deployment_is_the_primary_result':True,
            'model_sha256':identity['model_sha256'],'methods':methods,'seeds':{}}
    for seed in (1,2,3,4):
        os.environ['SURGERY_ANCHORED_HASH_SEED']=str(seed)
        path=args.output/f'hash_sensitivity_{split}_seed_{seed}.csv'
        records={}
        if path.exists():
            for record in pd.read_csv(path).to_dict('records'):
                records[(record['group'],record['monday'],record['method'])]=record
        tasks=[]
        for week in weeks:
            keys=[(week.group,week.monday,method) for method in methods]
            if all(records.get(k,{}).get('complete')==True for k in keys):
                continue
            for k in keys:records.pop(k,None)
            rows=np.array([idx[int(cid)] for cid in week.case_ids],int)
            tasks.append((week,{name:pred[rows] for name,pred in predictions.items()},
                          args.seconds,args.threads))
        def persist(rows):
            for row in rows:
                records[(row['group'],row['monday'],row['method'])]=row
            temp=path.with_suffix('.tmp')
            pd.DataFrame(list(records.values())).to_csv(temp,index=False)
            temp.replace(path)
        if args.workers==1:
            for task in tasks:persist(_eval_week(task))
        elif tasks:
            with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                                     mp_context=mp.get_context('spawn')) as pool:
                future=[pool.submit(_eval_week,t) for t in tasks]
                for job in as_completed(future):persist(job.result())
        df=pd.DataFrame(records.values())
        if len(df)!=expected_weeks*len(methods) or not df.complete.eq(True).all():
            raise RuntimeError(f'Seed {seed} incomplete on {split}; retry to resume')
        report['seeds'][str(seed)]={
            name:{'regret_per_case':float(g.regret.sum()/g.cases.sum()),
                  'realized_cost':float(g.realized_cost.sum()),
                  'room_days_over_960':int(g.room_days_over_960.sum()),
                  'room_days_over_1440':int(g.room_days_over_1440.sum())}
            for name,g in df.groupby('method')}
        save_json(args.output/f'hash_sensitivity_{split}_summary.json',report)
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workbook',type=Path)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--gap-model',type=Path)
    p.add_argument('--initial-method',choices=('CASE','LASSO','VF','VF_GAP'),default='CASE')
    p.add_argument('--initial-policy',type=Path,
                   help='Optional completed feature-compatible policy, e.g. v1 progress JSON; seed experiment, never claim its initial gain')
    p.add_argument('--exact-sep-batch',type=int,default=96,
                   help='Exact tangent separation components per round; 0 checks every active component')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('references','train','certify','evaluate','sensitivity'),required=True)
    p.add_argument('--sensitivity-split',choices=('train','validation'),
                   default='validation',
                   help='Frozen-weight multi-hash analysis, never model selection')
    p.add_argument('--hash-seed',type=int,default=0,
                   help='Fixed deployment hash seed, recorded in run identity')
    p.add_argument('--report-gap',action='store_true',
                   help='Also solve primary GAP planner/adversary in --stage certify')
    p.add_argument('--site',choices=('all','TGH','TWH-main','TWH-day'),default='all')
    p.add_argument('--alpha',type=float,default=.8)
    p.add_argument('--h',type=float,default=30.)
    p.add_argument('--gamma',type=float,default=2.)
    p.add_argument('--lambda-l1',type=float,default=0.)
    p.add_argument('--prox',type=float,default=.15)
    p.add_argument('--coef-prox',type=float,default=1e-5)
    p.add_argument('--ridge',type=float,default=0.,help='Coefficient L2 penalty')
    p.add_argument('--reference-penalty',type=float,default=0.,help='Score distance to frozen LASSO')
    p.add_argument('--line-fractions',default='1,0.5,0.25,0.125')
    p.add_argument('--trust-growth',type=float,default=1.)
    p.add_argument('--terminal-separation',action='store_true')
    p.add_argument('--tight-surrogate',action='store_true',help='Valid but narrower non-dominating competitor set')
    p.add_argument('--max-terminal-restarts',type=int,default=1)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--closure-rounds',type=int,default=2)
    p.add_argument('--max-outer',type=int,default=5)
    p.add_argument('--min-improvement',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=15)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--sep-seconds',type=float,default=90.)
    p.add_argument('--master-seconds',type=float,default=300.)
    p.add_argument('--master-threads',type=int,default=4,
                   help='Gurobi threads for the sequential continuous master QP')
    args=p.parse_args()
    if args.hash_seed < 0:
        p.error('hash seed must be nonnegative')
    try:
        fractions=tuple(float(q) for q in args.line_fractions.split(','))
        if not fractions or any(not np.isfinite(q) or q<=0 for q in fractions):
            raise ValueError()
    except ValueError:
        p.error('line-fractions must be comma-separated finite positive values')
    os.environ['SURGERY_ANCHORED_HASH_SEED'] = str(args.hash_seed)
    if (args.workers<1 or args.threads<1 or args.workers*args.threads>15
            or not 1<=args.master_threads<=15
            or args.gamma<1 or not 0<args.alpha<=1 or args.h<=0
            or args.lambda_l1<0 or args.ridge<0 or args.reference_penalty<0
            or not np.isfinite(args.trust_growth) or args.trust_growth<1.
            or args.max_terminal_restarts<0
            or args.prox<=0 or args.coef_prox<=0
            or args.max_outer<1 or args.iterations<1 or args.closure_rounds<1
             or args.exact_sep_batch<0
            or min(args.seconds,args.sep_seconds,args.master_seconds)<=0):
        p.error('Invalid finite solver budgets, scenario, or regularization settings')
    if args.stage=='references':
        _,_,weeks,index,data,policies,_=initialize(args)
        build_reference_library(args.output,weeks,index,data,policies,args.alpha,args.h,
                                args.seconds,args.threads,args.workers)
    elif args.stage=='train':train(args)
    elif args.stage=='certify':certify(args)
    elif args.stage=='sensitivity':sensitivity(args)
    else:evaluate(args)

if __name__=='__main__':main()
