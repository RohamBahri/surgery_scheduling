#!/usr/bin/env python3
"""Offline feature and recommendation diagnostics for ANCHORED research v3.

This script is a scientific AUDIT, not an alternative feature-estimation scheme.
It reports differences between training leave-date-out effects and deployment
full-offline-history effects on the SAME designated training cases.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_weekly_rich import load_matrices, read_cohort, windows
from surgery.rich_features import RichCorrectionEncoder, REMOVED_SERVICE_FEATURES
from surgery.planner import display_and_duration
from surgery.weekly_training import save_json

def response_regions(X,w,booked,alpha=.8,h=30.):
    score=np.asarray(X,float)@np.asarray(w,float)
    displayed,reviewed=display_and_duration(score,booked,alpha,h)
    abs_display=np.abs(displayed)
    return dict(
        full_adoption_fraction=float(np.mean(abs_display<=h+1e-8)),
        attenuation_fraction=float(np.mean((abs_display>h+1e-8)&
            (abs_display<h/(1-alpha)-1e-8))) if alpha<1 else 0.,
        rejection_fraction=float(np.mean(abs_display>=h/(1-alpha)-1e-8)) if alpha<1 else 0.,
        mean_absolute_effective_correction=float(np.mean(np.abs(reviewed-booked))),
        raw_score_max=float(np.max(np.abs(score))),
        coefficients_at_box=int(np.count_nonzero(np.abs(w)>=149.999)),
        score_rms=float(np.sqrt(np.mean(score**2))),
    )

def audit_features(encoder,training,fit_X):
    """Inspect empirical LODO-versus-deployment feature discrepancies."""
    full_X=encoder.transform(training)
    if full_X.shape!=fit_X.shape:
        raise ValueError('Training/transform feature schema inconsistent')
    date_error=training.assign(error=training.actual-training.booked)
    means=date_error.groupby(['service','date'],sort=False).error.transform('mean').to_numpy(float)
    summary={}
    for j,name in enumerate(encoder.names):
        if j==0:continue
        train=np.asarray(fit_X[:,j],float)
        full=np.asarray(full_X[:,j],float)
        diff=train-full
        with np.errstate(invalid='ignore'):
            corr=np.corrcoef(diff,means)[0,1] if np.std(diff)>1e-9 and np.std(means)>1e-9 else None
        summary[name]=dict(
            lodo_vs_full_rms=float(np.sqrt(np.mean(diff**2))),
            lodo_vs_full_max=float(np.max(np.abs(diff))),
            empirical_correlation_with_date_error=None if corr is None or not np.isfinite(corr) else float(corr),
            fit_sd=float(np.std(train)),full_sd=float(np.std(full)))
    return full_X,summary

def run(args):
    meta,frames=load_matrices(args.artifacts)
    raw,cohort=read_cohort(args.workbook)
    tr,_,_,_=windows(cohort,meta['start_week_index'],
                      meta['train_weeks'],meta['validation_weeks'])
    enc=RichCorrectionEncoder(history_mode=meta.get('history_mode','leave_date_out'))
    fit_X=enc.fit_transform(tr)
    if (enc.names!=meta['feature_names']
            or not np.array_equal(tr.case_id.to_numpy(int),frames['train']['case_id'])
            or not np.allclose(fit_X,frames['train']['X'],atol=5e-5)):
        raise ValueError('Rebuilt training encoder does not reproduce frozen matrices')
    full_X,shifts=audit_features(enc,tr,fit_X)
    names=enc.names
    if REMOVED_SERVICE_FEATURES.intersection(names):
        raise AssertionError('Service artifact columns still present')
    result=dict(
        feature_spec='service_feature_artifact_repair_v1',
        history_mode=enc.history_mode,
        n_train=len(fit_X),p=len(names),
        removed_features=sorted(REMOVED_SERVICE_FEATURES),
        historical_columns={k:v for k,v in shifts.items()
                            if k.startswith(('hist_','proportional_booked_x_','procedure_','surgeon_','interaction_booked_x_'))},
        policy_shifts={},training_acceptance=[])
    sources={'LASSO':args.artifacts/'lasso_policy.json',
             'CASE':args.artifacts/'model_case.json',
             'VF':args.artifacts/'vf_progress.json',
             'ANCHORED':args.output/'anchored_progress.json'}
    for name,path in sources.items():
        if not path.exists():continue
        obj=json.loads(path.read_text())
        w=np.asarray(obj.get('weights' if name=='LASSO' else 'w',[]),float)
        if w.shape!=(len(names),):continue
        discrepancy=(fit_X-full_X)@w
        a=frames['train']
        result['policy_shifts'][name]=dict(
            training_fit_vs_deploy_like_rms=float(np.sqrt(np.mean(discrepancy**2))),
            training_fit_vs_deploy_like_mae=float(np.mean(np.abs(discrepancy))),
            train=response_regions(fit_X,w,a['booked']),
            validation=response_regions(frames['validation']['X'],w,
                                        frames['validation']['booked']))
        if name=='ANCHORED':
            result['training_acceptance']=obj.get('accepted',[])
            result['stopping']=obj.get('stopping')
            result['separation_closed']=obj.get('separation_closed')
            result['complete']=obj.get('complete')
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'research_feature_audit.json',result)
    print(json.dumps(dict(n_train=result['n_train'],features=result['p'],
         removed=result['removed_features'],policies=result['policy_shifts'],
         largest_history_shifts=sorted(
             [{'feature':k,**v} for k,v in result['historical_columns'].items()],
             key=lambda r:-r['lodo_vs_full_rms'])[:15]),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workbook',type=Path)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('audit',),required=True)
    args=p.parse_args()
    run(args)

if __name__=='__main__':
    main()
