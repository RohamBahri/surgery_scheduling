#!/usr/bin/env python3
"""Two-fold, training-period-only decision-model pilot.

Odd/even calendar Mondays are genuine outer evaluation folds: the rich
historical encoder, LASSO start, CASE penalty path, VF, VF-GAP, ANCHORED and
ANCHORED reference library are all built using fitting-fold outcomes ONLY.
Retrospective scheduling rosters are held fixed as an explicit assumption.
2012 development-validation and 2013 test outcomes are never scored here.
"""
import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pilot import run_pilot as core


def save(path, value):
    core.write_json(path, value)


def sha(path):
    return core.digest(path)


def paths(args):
    return args.root / (args.link+'_'+args.context) / ('fold_'+str(args.fold))


def train_folds(args):
    """Refit rich encodings on the fitting dates, never use frozen train X."""
    from run_weekly_rich import read_cohort, windows
    from run_weekly_train import load_weeks, check_week_cases
    from surgery.rich_features import RichCorrectionEncoder
    manifest=json.loads((args.artifacts/'manifest.json').read_text())
    if sha(args.workbook)!=manifest['workbook_sha256']:
        raise RuntimeError('Workbook has changed since frozen feature run')
    with np.load(args.artifacts/'train.npz') as z:
        recorded=z['case_id'].copy()
    if sha(args.artifacts/'train.npz')!=manifest['matrix_sha256']['train']:
        raise RuntimeError('Frozen train matrix SHA mismatch')
    raw,cohort=read_cohort(args.workbook)
    frame,_,_,_=windows(cohort,manifest['start_week_index'],
                         manifest['train_weeks'],manifest['validation_weeks'])
    if len(frame)!=len(recorded) or not np.array_equal(frame.case_id.to_numpy(int),recorded):
        raise RuntimeError('Fold case roster does not match frozen training cohort')
    weeks=load_weeks(args.workbook,manifest,'train')
    monday=sorted({w.monday for w in weeks})
    if len(weeks)!=180 or len(monday)!=60:
        raise RuntimeError('Unexpected 180 site-weeks / 60 calendar weeks')
    held={day for j,day in enumerate(monday) if j%2==args.fold}
    fitweeks=[w for w in weeks if w.monday not in held]
    evalweeks=[w for w in weeks if w.monday in held]
    if len(fitweeks)!=90 or len(evalweeks)!=90:
        raise RuntimeError('Unbalanced calendar folds')
    fitkeys={int(cid) for w in fitweeks for cid in w.case_ids}
    evalkeys={int(cid) for w in evalweeks for cid in w.case_ids}
    if fitkeys & evalkeys or len(fitkeys)+len(evalkeys)!=len(frame):
        raise RuntimeError('Disjoint case split failed')
    fit=frame.loc[frame.case_id.isin(fitkeys)].reset_index(drop=True)
    evaluation=frame.loc[frame.case_id.isin(evalkeys)].reset_index(drop=True)
    encoder=RichCorrectionEncoder(history_mode='leave_date_out')
    X_fit=encoder.fit_transform(fit)
    X_eval=encoder.transform(evaluation)
    names=list(encoder.names)
    fit_data=dict(X=X_fit,case_id=fit.case_id.to_numpy(int),
                  booked=fit.booked.to_numpy(float),actual=fit.actual.to_numpy(float))
    eval_data=dict(X=X_eval,case_id=evaluation.case_id.to_numpy(int),
                   booked=evaluation.booked.to_numpy(float),
                   actual=evaluation.actual.to_numpy(float))
    fit_index=check_week_cases(fitweeks,fit_data)
    eval_index=check_week_cases(evalweeks,eval_data)
    if args.context!='none':
        X_fit,extras,mu,sd=core.add_context(X_fit,fitweeks,fit_index,
                     variant=args.context)
        X_eval,_,_,_=core.add_context(X_eval,evalweeks,eval_index,
                     variant=args.context,scaler=(mu,sd))
        names+=list(extras)
        fit_data['X']=X_fit
        eval_data['X']=X_eval
    else:
        mu,sd=[],[]
    if not np.isfinite(X_fit).all() or not np.isfinite(X_eval).all():
        raise RuntimeError('Nonfinite fold predictor')
    return fit_data,eval_data,fitweeks,evalweeks,fit_index,eval_index,dict(
        fit_mondays=[x for x in monday if x not in held],
        evaluation_mondays=[x for x in monday if x in held],
        n_fit=len(fit),n_eval=len(evaluation),feature_names=names,
        context_mu=mu,context_sd=sd,
        fit_matrix_sha256=hashlib.sha256(np.ascontiguousarray(X_fit).tobytes()).hexdigest(),
        eval_matrix_sha256=hashlib.sha256(np.ascontiguousarray(X_eval).tobytes()).hexdigest(),
        note='Fold encoder trained on fitting-fold labels only; evaluation transform uses only fitting history')


def fold_lasso(args, data, root):
    """Training-fold BIC LASSO initialization. Never load full-fit weights."""
    from sklearn.linear_model import Lasso
    X=data['X'].astype(float)
    target=np.asarray(data['actual']-data['booked'],float)
    settings=[0.01,0.03,0.1,0.3,1.,3.,10.]
    rows=[]
    for lam in settings:
        m=Lasso(alpha=lam,fit_intercept=True,max_iter=20000,tol=1e-5)
        m.fit(X[:,1:],target)
        w=np.r_[m.intercept_,m.coef_]
        mse=float(np.mean((target-X@w)**2))
        df=1+int((np.abs(w[1:])>1e-6).sum())
        score=len(target)*np.log(max(1e-10,mse))+df*np.log(len(target))
        rows.append((score,lam,w))
    _,lam,w=min(rows,key=lambda item:(item[0],item[1]))
    path=root/'fold_lasso.json'
    obj=dict(weights=w.tolist(),selected_alpha=lam,
             selection='Gaussian BIC; fitted strictly on the fitting fold')
    if path.exists() and json.loads(path.read_text())!=obj:
        raise RuntimeError('Fitted fold LASSO changed; preserve run identity')
    if not path.exists():
        save(path,obj)


def identity(args, foldinfo):
    return dict(fold=args.fold,context=args.context,link=args.link,
                workbook_sha256=sha(args.workbook),
                manifest_sha256=sha(args.artifacts/'manifest.json'),
                runner_sha256=sha(__file__),pilot_sha256=sha('pilot/run_pilot.py'),
                alpha=args.alpha,h=args.h,gamma=args.gamma,
                max_outer=args.outer,decision_inner=args.inner,
                case_inner=args.case_inner,lambda_grid=list(args.lambda_grid),
                ridge_grid=list(args.ridges),hash_seeds=list(args.seeds),
                **foldinfo)


def setup(args):
    os.environ['SURGERY_RESEARCH_LINK']=args.link
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'
    folder=paths(args)
    folder.mkdir(parents=True,exist_ok=True)
    data,evaluation,weeks,evalweeks,index,evalindex,foldinfo=train_folds(args)
    stamp=identity(args,foldinfo)
    mark=folder/'fold_identity.json'
    if mark.exists() and json.loads(mark.read_text())!=stamp:
        raise RuntimeError('Fold/model settings changed; use a fresh ROOT')
    if not mark.exists():
        save(mark,stamp)
    return folder,data,evaluation,weeks,evalweeks,index,evalindex,stamp


def fitted_case(args,folder,data,stamp):
    args.out=folder
    args.fold_safe=True
    args.case_inner=args.case_inner
    args.lambda_grid=tuple(args.lambda_grid)
    fold_lasso(args,data,folder)
    core.case(args,data,stamp)


def fit_decision(args,folder,data,weeks,index,stamp):
    if not (folder/'case_policy.json').exists():
        raise RuntimeError('Run stage case first')
    args.out=folder
    if args.method=='vf':
        core.vf(args,data,weeks,index,stamp)
        return
    if args.method=='gap':
        core.gap(args,data,weeks,index,stamp)
        return
    if args.method!='anchored':
        raise ValueError(args.method)
    for ridge in args.ridges:
        # Distinct settings and checkpoint per ridge; no validation-based selection.
        name='ridge_'+format(ridge,'.6g').replace('.','p')
        work=folder/'sensitivity'/name
        work.mkdir(parents=True,exist_ok=True)
        target=work/'case_policy.json'
        source=folder/'case_policy.json'
        if target.exists() and sha(target)!=sha(source):
            raise RuntimeError('Anchor copy mismatch')
        if not target.exists():
            shutil.copyfile(source,target)
        # Rebuild references using ONLY this fold, never from the old full run.
        common=folder/'train_refs'/'anchored_references.json'
        if not common.exists():
            from surgery.anchored_training import build_reference_library
            case=json.loads(source.read_text())
            build_reference_library(common.parent,weeks,index,data,
                                    {'CASE_BIC':np.asarray(case['w'],float)},
                                    args.alpha,args.h,args.seconds,1,args.workers)
        args.out=work
        args.reuse_refs=common
        args.ridge=ridge
        core.anchored(args,data,weeks,index,stamp)
    args.out=folder


def oracle_task(task):
    """Evaluation outcomes are accessed ONLY after the policy fitting stage."""
    week,seconds=task
    from surgery.anchored_core import key
    from surgery.anchored_oracles import deployed
    result={}
    for c in week.components():
        r=deployed(c,c.actual,seconds=seconds,threads=1,hindsight=True)
        result[key(c)]=dict(complete=r['complete'],
                            realized_cost=r.get('realized_cost'),
                            status=r.get('status'))
    return result


def hindsight(args,evalweeks):
    from surgery.anchored_core import key
    # Shared between methods/contexts; immutable evaluator cache.
    path=args.root/'outer_oracles'/('fold_'+str(args.fold)+'.json')
    if path.exists():
        records=json.loads(path.read_text())
    else:
        records={}
    if args.workers==1:
        for week in evalweeks:
            needed=[key(c) for c in week.components()]
            if all(records.get(k,{}).get('complete') for k in needed):
                continue
            records.update(oracle_task((week,args.seconds)))
            save(path,records)
    else:
        jobs=[(week,args.seconds) for week in evalweeks
              if not all(records.get(key(c),{}).get('complete') for c in week.components())]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as executor:
            futures=[executor.submit(oracle_task,item) for item in jobs]
            for f in as_completed(futures):
                records.update(f.result())
                save(path,records)
    needed={key(c) for w in evalweeks for c in w.components()}
    if set(records)!=needed or not all(records[k].get('complete') for k in needed):
        raise RuntimeError('Evaluation hindsight oracle incomplete; rerun with larger --seconds')
    return records


def score(args,folder,evaluation,evalweeks,evalindex):
    """Out-of-fold plan evaluation; never feeds the scores back to fitting."""
    from surgery.anchored_training import verify_policy
    from surgery.anchored_core import key
    hindsight_rows=hindsight(args,evalweeks)
    p=evaluation['X'].shape[1]
    policies={'BOOKED':np.zeros(p)}
    case=folder/'case_policy.json'
    if not case.exists():
        raise RuntimeError('Fit CASE first')
    policies['CASE_BIC']=json.loads(case.read_text())['w']
    for name,path in (('VF',folder/'vf'/'vf_progress.json'),
                      ('VF_GAP',folder/'gap'/'gap_progress.json')):
        if path.exists():
            obj=json.loads(path.read_text())
            if obj.get('complete'):
                policies[name]=obj['w']
    for ridge in args.ridges:
        label='ridge_'+format(ridge,'.6g').replace('.','p')
        path=folder/'sensitivity'/label/'anchored'/'anchored_progress.json'
        if path.exists():
            obj=json.loads(path.read_text())
            if obj.get('complete'):
                policies['ANCHORED_'+label]=obj['w']
    outpath=folder/'heldout_scores.json'
    report=json.loads(outpath.read_text()) if outpath.exists() else dict(
        split='held-out training-period calendar weeks',
        n_cases=len(evaluation['case_id']),seed_results={},per_calendar_week=[])
    for seed in args.seeds:
        os.environ['SURGERY_ANCHORED_HASH_SEED']=str(seed)
        report['seed_results'].setdefault(str(seed),{})
        for name,w in policies.items():
            if name in report['seed_results'][str(seed)]:
                continue
            w=np.asarray(w,float)
            if len(w)!=p:
                raise RuntimeError('Fold model dimension mismatch: '+name)
            cache=folder/'heldout_verification'/('seed_'+str(seed))
            dep=verify_policy(cache,name,evalweeks,evalindex,
                              evaluation['X'],evaluation['booked'],w,
                              args.alpha,args.h,args.seconds,1,args.workers)
            records=[];total=0.;oracle_total=0.
            for week in evalweeks:
                comps=week.components()
                keys=[key(c) for c in comps]
                if not all(dep[k].get('complete') and hindsight_rows[k]['complete'] for k in keys):
                    raise RuntimeError('Held-out deployed plan incomplete')
                realized=sum(float(dep[k]['realized_cost']) for k in keys)
                oracle=sum(float(hindsight_rows[k]['realized_cost']) for k in keys)
                cases=sum(c.n for c in comps)
                total+=realized;oracle_total+=oracle
                records.append(dict(seed=seed,method=name,calendar_week=week.monday,
                                    site=week.group,cases=cases,regret=realized-oracle,
                                    realized_cost=realized,hindsight_cost=oracle))
            report['per_calendar_week']=[v for v in report['per_calendar_week']
                    if not (v['seed']==seed and v['method']==name)]+records
            report['seed_results'][str(seed)][name]=dict(
                complete=True,regret_per_case=(total-oracle_total)/len(evaluation['case_id']),
                realized_cost=total,hindsight_cost=oracle_total)
            save(outpath,report)
            print('Fold',args.fold,'seed',seed,name,'regret/case',
                  round((total-oracle_total)/len(evaluation['case_id']),4),flush=True)
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',required=True,choices=('prepare','case','decision','score'))
    ap.add_argument('--fold',type=int,required=True,choices=(0,1))
    ap.add_argument('--context',required=True,choices=('none','cohort','roster'))
    ap.add_argument('--link',required=True,choices=('original','inverse'))
    ap.add_argument('--method',choices=('vf','gap','anchored'),default='vf')
    ap.add_argument('--root',type=Path,default=Path('artifacts/decision_model_pilot_v2'))
    ap.add_argument('--artifacts',type=Path,default=Path('artifacts/weekly_rich_leakfix_v1'))
    ap.add_argument('--workbook',type=Path,default=Path('data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    ap.add_argument('--alpha',type=float,default=.8)
    ap.add_argument('--h',type=float,default=30.)
    ap.add_argument('--gamma',type=float,default=2.)
    ap.add_argument('--case-inner',type=int,default=500)
    ap.add_argument('--inner',type=int,default=35)
    ap.add_argument('--outer',type=int,default=2)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--seconds',type=float,default=300.)
    ap.add_argument('--master-seconds',type=float,default=120.)
    ap.add_argument('--sep-seconds',type=float,default=60.)
    ap.add_argument('--sep-batch',type=int,default=128)
    ap.add_argument('--closure-rounds',type=int,default=1)
    ap.add_argument('--lambda-grid',default='0.0005,0.001,0.002,0.004,0.008,0.016,0.032')
    ap.add_argument('--ridge-grid',default='0,0.0001,0.001')
    ap.add_argument('--seeds',default='0,1,2')
    args=ap.parse_args()
    args.lambda_grid=tuple(float(z) for z in args.lambda_grid.split(','))
    args.ridges=tuple(float(z) for z in args.ridge_grid.split(','))
    args.seeds=tuple(int(z) for z in args.seeds.split(','))
    args.reuse_refs=None
    args.fold_safe=True
    args.ridge=0.
    if not 1<=args.workers<=15 or args.case_inner<1 or args.outer<1 or args.inner<1:
        ap.error('Invalid pilot budgets')
    folder,data,evaluation,weeks,evalweeks,index,evalindex,stamp=setup(args)
    print('FOLD',args.fold,args.link,args.context,'FIT',len(data['case_id']),
          'EVAL',len(evaluation['case_id']),len(stamp['feature_names']),'features',
          flush=True)
    if args.stage=='case':
        fitted_case(args,folder,data,stamp)
    elif args.stage=='decision':
        fit_decision(args,folder,data,weeks,index,stamp)
    elif args.stage=='score':
        score(args,folder,evaluation,evalweeks,evalindex)


if __name__=='__main__':
    main()
