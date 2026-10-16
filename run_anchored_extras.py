#!/usr/bin/env python3
"""Optional research diagnostics; never used to tune on development labels.

boosted-fit: predeclared HistGradientBoosting residual predictor. The training
scores are group-out-of-fold, the validation scores are from a final predictor
fitted on ALL designated training cases. This is NOT an anchored shared linear
policy and is not used to generate any ANCHORED model updates.
boosted-evaluate: exact deployed weekly planning on 2012 development cases.
scenario-audit: evaluate frozen policies under predeclared behavior scenarios;
DOES NOT learn a single behavior-robust policy.
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

from run_weekly_rich import load_matrices, source_fingerprint
from run_weekly_train import load_weeks, check_week_cases
from surgery.weekly_training import save_json
from surgery.planner import display_and_duration
from surgery.anchored_weekly_solver import evaluate_week

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _frozen_identity(artifacts, output):
    meta, frames=load_matrices(artifacts)
    model_file=output/'anchored_progress.json'
    settings_file=output/'anchored_settings.json'
    if not model_file.exists() or not settings_file.exists():
        raise ValueError('Complete ANCHORED training and settings required')
    model=json.loads(model_file.read_text())
    settings=json.loads(settings_file.read_text())
    if not model.get('complete') or model.get('feature_dimensions')!=len(meta['feature_names']):
        raise ValueError('ANCHORED policy is not completed with matching dimensions')
    if (settings['matrix_sha256']!=meta['matrix_sha256']
            or settings['feature_source_sha256']!=source_fingerprint()):
        raise ValueError('ANCHORED artifact and feature identity mismatch')
    return meta,frames,model,settings

def _raw_preoperative(frame):
    """No outcome-derived history; all columns are known preoperatively."""
    cat=('group','service','procedure','surgeon','patient_type')
    x=frame[list(cat)].fillna('MISSING').astype(str).reset_index(drop=True)
    booked=np.asarray(frame['booked'],float)
    x['booked_minutes']=booked
    x['log_booked_minutes']=np.log1p(booked)
    day=pd.to_datetime(frame['date']).reset_index(drop=True)
    x['day_of_week']=day.dt.weekday.astype(float)
    x['month']=day.dt.month.astype(float)
    for name in ('dtt','consult'):
        if name in frame:
            lead=(day-pd.to_datetime(frame[name]).reset_index(drop=True)).dt.days.astype(float)
            x[name+'_lead_days']=np.where((lead>=0)&(lead<=3650),lead,np.nan)
        else:
            x[name+'_lead_days']=np.nan
    return x


def _boosted_estimator():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    cat=('group','service','procedure','surgeon','patient_type')
    num=('booked_minutes','log_booked_minutes','day_of_week','month',
         'dtt_lead_days','consult_lead_days')
    enc=ColumnTransformer([
        ('categorical',OneHotEncoder(handle_unknown='ignore',
                                    min_frequency=20,max_categories=40,
                                    sparse_output=False),list(cat)),
        ('numerical',SimpleImputer(strategy='median',keep_empty_features=True),
                     list(num))
    ],sparse_threshold=0.)
    model=HistGradientBoostingRegressor(
        loss='absolute_error',learning_rate=.05,max_iter=300,
        max_leaf_nodes=15,max_depth=3,l2_regularization=10.,
        min_samples_leaf=35,random_state=20261015)
    return Pipeline([('preoperative_only',enc),('predictor',model)])


def fit_boosted(args):
    from run_weekly_rich import read_cohort, windows
    meta,frames=load_matrices(args.artifacts)
    train=frames['train'];valid=frames['validation']
    if args.workbook is None:
        raise ValueError('Raw preoperative boosted baseline needs the workbook')
    _,cohort=read_cohort(args.workbook)
    train_frame,valid_frame,_,_=windows(cohort,meta['start_week_index'],
                                        meta['train_weeks'],meta['validation_weeks'])
    if (not np.array_equal(train_frame.case_id.to_numpy(int),train['case_id'])
            or not np.array_equal(valid_frame.case_id.to_numpy(int),valid['case_id'])):
        raise ValueError('Raw preoperative case ordering differs from frozen cohort')
    X=_raw_preoperative(train_frame)
    Xv=_raw_preoperative(valid_frame)
    y=np.asarray(train['actual']-train['booked'],float)
    # Grouped out-of-fold prediction with raw features only; no chronological
    # restrictions and no target-encoded values in any predictor fold.
    dates=np.unique(train['monday'])
    fold_by_date={date:i%5 for i,date in enumerate(sorted(dates))}
    fold=np.asarray([fold_by_date[d] for d in train['monday']],int)
    pred=np.full(len(X),np.nan,float)
    for k in range(5):
        model=_boosted_estimator()
        model.fit(X.loc[fold!=k],y[fold!=k])
        pred[fold==k]=model.predict(X.loc[fold==k])
    if not np.isfinite(pred).all():
        raise AssertionError('Missing out-of-fold predictions')
    final=_boosted_estimator()
    final.fit(X,y)
    test_pred=final.predict(Xv)
    target=args.output/'boosted_predictions.npz'
    cfg=args.output/'boosted_identity.json'
    identity=dict(feature_hash=meta['matrix_sha256'],feature_source=source_fingerprint(),
                  workbook_sha256=digest(args.workbook),
                  method='HistGradientBoosting_raw_preoperative_absolute_residual_v2',
                  train_label_use='other calendar-week folds only',
                  fold_seed=20261015,folds=5,
                  predictors=list(X.columns),
                  validation_training='all designated training cases; no validation labels')
    if target.exists() or cfg.exists():
        if not target.exists() or not cfg.exists() or json.loads(cfg.read_text())!=identity:
            raise ValueError('Boosted model identity differs; use a new OUT directory')
        with np.load(target) as old:
            if not (np.allclose(old['train_score'],pred,atol=1e-5)
                    and np.allclose(old['validation_score'],test_pred,atol=1e-5)):
                raise ValueError('Boosted predictions differ from previous result')
    else:
        np.savez_compressed(target,train_score=pred,validation_score=test_pred,
                            train_case_id=train['case_id'],
                            validation_case_id=valid['case_id'])
        save_json(cfg,identity)
    print(json.dumps({'file':str(target),'train_oof_mae_raw':float(np.mean(np.abs(y-pred))),
                      'raw_preoperative_features':list(X.columns),'n_train':len(X)},indent=2))


def _score_weeks(weeks,case_index,preds,seconds,threads,workers,path):
    records={}
    if path.exists():
        for row in pd.read_csv(path).to_dict('records'):
            records[(row['group'],str(row['monday']),row['method'])]=row
    tasks=[]
    for week in weeks:
        names=sorted(preds)
        keys=[(week.group,week.monday,name) for name in names]
        if all(k in records and records[k].get('complete')==True for k in keys):
            continue
        for k in keys:records.pop(k,None)
        rows=np.asarray([case_index[int(cid)] for cid in week.case_ids],int)
        tasks.append((week,{name:d[rows] for name,d in preds.items()},seconds,threads))
    def persist(rows):
        for v in rows:records[(v['group'],v['monday'],v['method'])]=v
        temp=path.with_suffix('.tmp')
        pd.DataFrame(records.values()).to_csv(temp,index=False)
        temp.replace(path)
    if workers==1:
        for task in tasks:
            persist(evaluate_week(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate_week,t) for t in tasks]
            for job in as_completed(futures):
                persist(job.result())
    return pd.DataFrame(records.values())

def _describe(df,expected):
    if len(df)!=expected or not df.complete.eq(True).all():
        raise RuntimeError('Incomplete exact planning evaluation')
    return {method:{
        'cases':int(d.cases.sum()),'regret_per_case':float(d.regret.sum()/d.cases.sum()),
        'realized_cost':float(d.realized_cost.sum()),
        'prediction_mae':float(d.abs_error.sum()/d.cases.sum()),
        'overtime':float(d.overtime.sum()),
        'room_days_over_960':int(d.room_days_over_960.sum())}
        for method,d in df.groupby('method')}

def _check_output_identity(path,identity):
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise ValueError('Do not mix experiments in the same output directory')
    else:
        save_json(path,identity)

def evaluate_boosted(args):
    # Independent forecast baseline: no ANCHORED training is required.
    meta,frames=load_matrices(args.artifacts)
    path=args.output/'boosted_predictions.npz'
    identpath=args.output/'boosted_identity.json'
    if not path.exists() or not identpath.exists():
        raise ValueError('Run boosted-fit first')
    boosted_identity=json.loads(identpath.read_text())
    if (boosted_identity.get('feature_hash')!=meta['matrix_sha256']
            or boosted_identity.get('workbook_sha256')!=digest(args.workbook)
            or boosted_identity.get('feature_source')!=source_fingerprint()):
        raise ValueError('Boosted score identity mismatch')
    valid=frames['validation'];b=valid['booked']
    with np.load(path) as x:
        if not np.array_equal(x['validation_case_id'],valid['case_id']):
            raise ValueError('Boosted predictions case order mismatch')
        score=x['validation_score']
    _,pred=display_and_duration(score,b,args.alpha,args.h)
    weeks=load_weeks(args.workbook,meta,'validation')
    idx=check_week_cases(weeks,valid)
    _check_output_identity(args.output/'boosted_evaluation_identity.json',
          dict(prediction_sha=digest(path),workbook_sha=digest(args.workbook),
               alpha=args.alpha,h=args.h,feature=meta['matrix_sha256']))
    df=_score_weeks(weeks,idx,{'BOOSTED':pred},args.seconds,1,args.workers,
                   args.output/'boosted_validation.csv')
    result=_describe(df,54)
    save_json(args.output/'boosted_validation_summary.json',result)
    print(json.dumps(result,indent=2))

def audit_scenarios(args):
    meta,frames,model,settings=_frozen_identity(args.artifacts,args.output)
    valid=frames['validation']
    weeks=load_weeks(args.workbook,meta,'validation')
    idx=check_week_cases(weeks,valid)
    root=args.artifacts
    methods={'ANCHORED':np.asarray(model['w'],float)}
    for name,path,key in (
        ('CASE',root/'model_case.json','w'),
        ('LASSO',root/'lasso_policy.json','weights'),
        ('VF',root/'vf_progress.json','w')):
        doc=json.loads(path.read_text())
        methods[name]=np.asarray(doc[key],float)
    for name,w in methods.items():
        if w.shape!=(len(meta['feature_names']),):
            raise ValueError('Frozen policy dimension mismatch: '+name)
    grid=[]
    for item in args.scenarios.split(','):
        alpha,h=map(float,item.split(':'))
        if not (0<alpha<=1 and h>0 and np.isfinite(h)):
            raise ValueError('Invalid behavioral scenario')
        grid.append((alpha,h))
    identity=dict(anchored=digest(args.output/'anchored_progress.json'),
                  features=meta['matrix_sha256'],grid=grid,
                  use='frozen-weight scenario sensitivity, not robust retraining')
    _check_output_identity(args.output/'scenario_audit_identity.json',identity)
    final={}
    for alpha,h in grid:
        preds={}
        for name,w in methods.items():
            _,preds[name]=display_and_duration(valid['X'].astype(float)@w,valid['booked'],alpha,h)
        label=f'a{alpha:g}_h{h:g}'.replace('.','p')
        path=args.output/f'scenario_{label}_validation.csv'
        df=_score_weeks(weeks,idx,preds,args.seconds,1,args.workers,path)
        final[label]=_describe(df,54*len(preds))
        save_json(args.output/'scenario_audit_summary.json',final)
    print(json.dumps(final,indent=2))


def audit_flexible_grid(args):
    """Exact finite-grid deployments for a FEW training-only small components.

    Exhaustive only over the chosen discrete grid of reachable effective
    durations, not globally optimal over the continuous behavioral box.
    """
    from itertools import product
    from surgery.planner import reachable
    from surgery.anchored_oracles import deployed
    from surgery.weekly_planner import evaluate
    from surgery.anchored_core import key as component_key
    meta,frames,model,settings=_frozen_identity(args.artifacts,args.output)
    refs_path=args.output/'anchored_references.json'
    if not refs_path.exists():
        raise ValueError('Certified training references required')
    refs=json.loads(refs_path.read_text())
    origin=json.loads((args.artifacts/'learning_settings.json').read_text())
    alpha=float(origin['alpha']);h=float(origin['h'])
    os.environ['SURGERY_ANCHORED_HASH_SEED']=str(settings['hash_seed'])
    train=frames['train']
    weeks=load_weeks(args.workbook,meta,'train')
    indices={int(cid):i for i,cid in enumerate(train['case_id'])}
    X=train['X'].astype(float)
    w=np.asarray(model['w'],float)
    _,predicted=display_and_duration(X@w,train['booked'],alpha,h)
    choices=[]
    for week in weeks:
        for c in week.components():
            if c.n>args.max_flex_cases or c.n<1:
                continue
            if not any(len(c.allowed[int(s)])>1 for s in c.surgeon):
                continue
            k=component_key(c)
            if k not in refs or not refs[k].get('complete'):
                continue
            nominal_gap=evaluate(c,c.original,c.actual)['cost']-refs[k]['oracle_cost']
            choices.append((nominal_gap,k,c))
    choices=sorted(choices,key=lambda x:(-x[0],x[1]))[:args.max_flex_components]
    identity=dict(feature_hash=meta['matrix_sha256'],
                  anchored_sha=digest(args.output/'anchored_progress.json'),
                  scenario=[alpha,h],max_cases=args.max_flex_cases,
                  max_components=args.max_flex_components,
                  grid_levels=args.grid_levels,hash_seed=settings['hash_seed'],
                  claim='complete enumeration of finite reachable grid only')
    idpath=args.output/'flexible_grid_identity.json'
    _check_output_identity(idpath,identity)
    path=args.output/'flexible_grid_audit.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    for _,k,c in choices:
        if report.get(k,{}).get('finished'):
            continue
        row=np.asarray([indices[int(cid)] for cid in c.case_ids],int)
        current=deployed(c,predicted[row],seconds=args.seconds,threads=1)
        low,high=reachable(c.booked,alpha,h)
        candidate_levels=[
            sorted(set(map(float,np.linspace(low[i],high[i],args.grid_levels))))
            for i in range(c.n)]
        best=float('inf');best_d=None
        completed=0;unresolved=0
        total=int(np.prod([len(v) for v in candidate_levels]))
        for candidate in product(*candidate_levels):
            d=np.asarray(candidate,float)
            r=deployed(c,d,seconds=args.seconds,threads=1)
            if not r['complete']:
                unresolved+=1
                continue
            completed+=1
            if r['realized_cost']<best:
                best=float(r['realized_cost']);best_d=d.tolist()
        record=dict(finished=True,cases=c.n,grid_size=total,
                    exact_grid_enumeration=unresolved==0,
                    certified_deployments=completed,unresolved_deployments=unresolved,
                    current_deployment_complete=bool(current['complete']),
                    current_realized_cost=current.get('realized_cost'),
                    best_grid_realized_cost=None if best_d is None else best,
                    best_grid_effective_durations=best_d,
                    oracle_cost=refs[k]['oracle_cost'])
        report[k]=record
        save_json(path,report)
        print(f'Grid audit {k}: {completed}/{total} certified',flush=True)
    print(json.dumps(dict(components=len(report),
                          fully_enumerated=sum(v['exact_grid_enumeration'] for v in report.values()),
                          path=str(path)),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('boosted-fit','boosted-evaluate','scenario-audit','flexible-audit'),required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workbook',type=Path)
    p.add_argument('--workers',type=int,default=15)
    p.add_argument('--seconds',type=float,default=600.)
    p.add_argument('--scenarios',default='.5:30,.8:30,1:30,.8:60')
    p.add_argument('--alpha',type=float,default=.8)
    p.add_argument('--h',type=float,default=30.)
    p.add_argument('--max-flex-components',type=int,default=5)
    p.add_argument('--max-flex-cases',type=int,default=3)
    p.add_argument('--grid-levels',type=int,default=3)
    args=p.parse_args()
    if (args.workers<1 or args.seconds<=0 or not 0<args.alpha<=1 or args.h<=0
            or args.max_flex_components<1
            or not 1<=args.max_flex_cases<=5 or not 2<=args.grid_levels<=5):
        p.error('Invalid worker count or wall time')
    if args.stage!='boosted-fit' and args.workbook is None:
        p.error('workbook is required for planning evaluation')
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['OPENBLAS_NUM_THREADS']='1'
    if args.stage=='boosted-fit':
        fit_boosted(args)
    elif args.stage=='boosted-evaluate':
        evaluate_boosted(args)
    elif args.stage=='flexible-audit':
        audit_flexible_grid(args)
    else:
        audit_scenarios(args)
if __name__=='__main__':
    main()
