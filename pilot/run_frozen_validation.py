#!/usr/bin/env python3
"""Full training, hash freeze, then one 2012 development-validation evaluation.

The original 2013 test outcomes are never fitted/scored. Training never opens
validation.npz; validation evaluation requires frozen full-cohort policy hashes.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pilot import run_pilot as core
from pilot.run_fold_pilot import fold_lasso, hindsight
from run_weekly_train import load_weeks, check_week_cases

RIDGES = (0.001, 0.003)
LAMBDA_GRID = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032)
SEEDS = (0, 1, 2)
# Exact SHA256 of the initial runner (commit f17b0d4), before the metadata-only
# CASE selection-key fix. This migration is permitted ONLY before any freeze.
PRE_FIX_RUNNER_SHA256 = '7802b23fcf792c5abe82c705d5af80155480079f7b0569be54bddd6b2373d6be'
# Same original script including a trailing newline (the initial local file).
# Both SHA-256 values were verified against the first committed runner bytes.
PRE_FIX_RUNNER_SHA256_TRAILING_NEWLINE = 'a6eb369839a608552760e23fe044a44e9a5640f5f2aa846f6a9bf5579252d20f'
SOURCES = ('pilot/run_pilot.py','pilot/run_fold_pilot.py','run_weekly_rich.py',
           'run_weekly_train.py','surgery/data.py','surgery/rich_features.py',
           'surgery/weekly_data.py','surgery/planner.py','surgery/weekly_training.py',
           'surgery/weekly_planner.py','surgery/anchored_core.py',
           'surgery/anchored_master.py','surgery/anchored_training.py',
           'surgery/anchored_oracles.py','surgery/anchored_weekly_solver.py')


def sha(path):
    return core.digest(path)


def save(path, obj):
    core.write_json(path, obj)


def label(ridge):
    return 'ridge_' + format(ridge, '.6g').replace('.', 'p')


def initial_identity(args):
    mpath = args.artifacts/'manifest.json'
    tp = args.artifacts/'train.npz'
    if not mpath.is_file() or not tp.is_file() or not args.workbook.is_file():
        raise RuntimeError('Missing workbook or frozen rich-feature train/manifest artifacts')
    meta = json.loads(mpath.read_text())
    if sha(args.workbook) != meta['workbook_sha256'] or sha(tp) != meta['matrix_sha256']['train']:
        raise RuntimeError('Workbook or frozen train matrix hash changed')
    if (meta.get('train_cases'),meta.get('validation_cases')) != (18705,5779):
        raise RuntimeError('Unexpected 18705/5779 cohort sizes')
    if len(meta['training_keys']) != 180 or len(meta['validation_keys']) != 54:
        raise RuntimeError('Expected 180 training and 54 development-validation site-weeks')
    if set(map(tuple,meta['training_keys'])) & set(map(tuple,meta['validation_keys'])):
        raise RuntimeError('Training and development-validation keys overlap')
    return dict(experiment='frozen_inverse_cohort_development_validation',
                workbook_sha256=sha(args.workbook),manifest_sha256=sha(mpath),
                train_sha256=sha(tp),runner_sha256=sha(__file__),
                source_sha256={p:sha(p) for p in SOURCES},
                response_alpha=.8,response_h=30.,context=('none','cohort'),
                outer=2,inner=35,case_inner=500,ridges=list(RIDGES),
                lambda_grid=list(LAMBDA_GRID),seeds=list(SEEDS),
                workers=args.workers,seconds=300,sep_seconds=60,
                master_seconds=120,sep_batch=128,closure_rounds=1,
                selection='ridge 0.001 PRIMARY, 0.003 sensitivity; chosen before validation')


def assert_identity(args):
    identity = initial_identity(args)
    args.root.mkdir(parents=True,exist_ok=True)
    path = args.root/'full_training_settings.json'
    if path.exists():
        recorded=json.loads(path.read_text())
        if recorded!=identity:
            # One strictly scoped checkpoint migration: original runner crashed
            # after finishing the CASE BIC path but BEFORE saving any policy.
            # The scientific inputs, other code, and every method setting must
            # be byte-for-byte unchanged; only this runner's hash may differ.
            old_comparable={k:v for k,v in recorded.items()
                            if k!='runner_sha256'}
            new_comparable={k:v for k,v in identity.items()
                            if k!='runner_sha256'}
            empty_policies=all(not (args.root/context/'case_policy.json').exists()
                               for context in ('none','cohort'))
            if (recorded.get('runner_sha256') in (PRE_FIX_RUNNER_SHA256,
                    PRE_FIX_RUNNER_SHA256_TRAILING_NEWLINE)
                    and old_comparable==new_comparable and empty_policies
                    and not (args.root/'frozen_policies.json').exists()
                    and not (args.root/'validation'/'validation_scores.json').exists()):
                save(args.root/'metadata_fix_migration.json',dict(
                    reason='CASE policy serialization duplicate selection key',
                    previous_runner_sha256=recorded['runner_sha256'],
                    updated_runner_sha256=identity['runner_sha256'],
                    scientific_settings_unchanged=True))
                save(path,identity)
            else:
                changed = sorted(k for k in set(recorded) | set(identity)
                                 if k != 'runner_sha256'
                                 and recorded.get(k) != identity.get(k))
                raise RuntimeError(
                    'Full-training identity mismatch; refusing to overwrite the '
                    'existing checkpoint. Changed scientific fields: '
                    + repr(changed) + '; saved runner SHA256: '
                    + str(recorded.get('runner_sha256'))
                    + '; CASE policy files present: '
                    + str(not empty_policies) + '; frozen: '
                    + str((args.root/'frozen_policies.json').exists())
                    + '; validation scores present: '
                    + str((args.root/'validation'/'validation_scores.json').exists())
                    + '. If only the runner SHA differs, share this message; '
                    'do not delete training files or change the output root.')
    else:
        save(path,identity)
    return identity


def read_split(args,part):
    # NEVER request validation in --stage case or --stage anchored or --stage freeze.
    if part not in ('train','validation'):
        raise ValueError('Unexpected split')
    manifest=json.loads((args.artifacts/'manifest.json').read_text())
    file=args.artifacts/(part+'.npz')
    if sha(file)!=manifest['matrix_sha256'][part]:
        raise RuntimeError('Frozen '+part+' matrix SHA mismatch')
    with np.load(file) as npz:
        data={k:npz[k].copy() for k in npz.files}
    if len(data['case_id'])!=(18705 if part=='train' else 5779):
        raise RuntimeError('Unexpected '+part+' case count')
    if len(manifest['feature_names'])!=data['X'].shape[1]:
        raise RuntimeError('Mismatched rich feature count')
    weeks=load_weeks(args.workbook,manifest,part)
    index=check_week_cases(weeks,data)
    if len(weeks)!=(180 if part=='train' else 54):
        raise RuntimeError('Unexpected site-week count')
    if len(set(w.monday for w in weeks))!=(60 if part=='train' else 18):
        raise RuntimeError('Unexpected calendar-week count')
    data['X']=data['X'].astype(float)
    if not np.allclose(data['X'][:,0],1.):
        raise RuntimeError('Expected intercept column')
    return manifest,data,weeks,index


def fitting(args,context,ident):
    manifest,data,weeks,index=read_split(args,'train')
    if context=='cohort':
        x,names,mean,sd=core.add_context(data['X'],weeks,index,variant='cohort')
        data['X']=x
    elif context=='none':
        names,mean,sd=(),(),()
    else:
        raise ValueError(context)
    d=args.root/context
    d.mkdir(parents=True,exist_ok=True)
    feature=dict(context=context,feature_names=list(manifest['feature_names'])+list(names),
                 extra_mean=list(mean),extra_sd=list(sd),
                 matrix_hash=hashlib.sha256(np.ascontiguousarray(data['X']).tobytes()).hexdigest())
    path=d/'training_features.json'
    if path.exists() and json.loads(path.read_text())!=feature:
        raise RuntimeError('Full-training predictors changed')
    if not path.exists():save(path,feature)
    return data,weeks,index,dict(ident,context=context,feature_sha256=sha(path))


def case_settings_for_core(settings):
    # The experiment-level selection rule (primary ANCHORED ridge) is NOT the
    # CASE penalty-selection method. core.case() already writes a distinct
    # 'selection' field, so never pass that experiment metadata name through.
    metadata=dict(settings)
    metadata['experiment_selection']=metadata.pop('selection')
    return metadata


def recover_complete_case_path(path, settings, feature_count):
    """Recover exact finished CASE candidates after metadata-only crash.

    Partial or inconsistent paths are discarded by the caller; no partially
    fitted coefficient vector may be mistaken for a selected policy.
    """
    if not path.is_file():
        return None
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or len(rows)!=len(LAMBDA_GRID):
        return None
    if sorted(row.get('lambda_l1') for row in rows)!=sorted(LAMBDA_GRID):
        return None
    for row in rows:
        try:
            coeff=np.asarray(row['coefficients'],float)
            numbers=[row['bic_proxy'],row['asymmetric_loss_per_case']]
            if (coeff.shape!=(feature_count,) or not np.isfinite(coeff).all()
                    or not all(np.isfinite(v) for v in numbers)
                    or row.get('bic_formula')!=
                    '2*n*log(mean asymmetric loss)+df*log(n)'):
                raise RuntimeError('CASE BIC path contains invalid completed candidate')
        except (KeyError,TypeError,ValueError) as exc:
            raise RuntimeError('CASE BIC path malformed; cannot recover') from exc
    best=min(rows,key=lambda x:(x['bic_proxy'],x['lambda_l1']))
    result=case_settings_for_core(settings)
    result.update(method='CASE_BIC_PILOT',w=best['coefficients'],
                  lambda_l1=best['lambda_l1'],
                  training_loss=best['asymmetric_loss_per_case'],
                  bic_proxy=best['bic_proxy'],
                  selection='training-only asymmetric quasi-BIC; approximate effective df')
    return result


def fit_case(args,ident):
    args.fold_safe=True
    args.case_inner=500
    args.lambda_grid=LAMBDA_GRID
    for context in ('none','cohort'):
        data,weeks,index,settings=fitting(args,context,ident)
        args.out=args.root/context
        fold_lasso(args,data,args.out)  # Full-training-only LASSO start.
        p=args.out/'case_policy.json'
        expected=case_settings_for_core(settings)
        if p.exists():
            doc=json.loads(p.read_text())
            if any(doc.get(k)!=v for k,v in expected.items()):
                raise RuntimeError('Existing CASE does not match immutable settings')
        else:
            recovered=recover_complete_case_path(
                args.out/'case_bic_path.json',settings,data['X'].shape[1])
            if recovered is not None:
                save(p,recovered)
                print('Recovered fully fitted CASE from BIC checkpoint:',context,flush=True)
            else:
                core.case(args,data,expected)
        print('Completed CASE on ALL 60 training weeks:',context,flush=True)


def fit_anchored(args,ident):
    from surgery.anchored_training import build_reference_library
    from surgery.anchored_core import key
    data,weeks,index,settings=fitting(args,'cohort',ident)
    home=args.root/'cohort'
    anchor=home/'case_policy.json'
    if not anchor.is_file():raise RuntimeError('First run --stage case')
    refs=home/'train_refs'/'anchored_references.json'
    refs.parent.mkdir(parents=True,exist_ok=True)
    base=json.loads(anchor.read_text())
    build_reference_library(refs.parent,weeks,index,data,
              {'CASE_BIC':np.asarray(base['w'],float)},.8,30.,300.,1,args.workers)
    state=json.loads(refs.read_text())
    required={key(c) for w in weeks for c in w.components()}
    if set(state)!=required or not all(state[k]['complete'] for k in required):
        raise RuntimeError('ANCHORED full-training hindsight/reference library incomplete')
    for ridge in RIDGES:
        d=home/'sensitivity'/label(ridge)
        d.mkdir(parents=True,exist_ok=True)
        copy=d/'case_policy.json'
        if copy.exists() and sha(copy)!=sha(anchor):
            raise RuntimeError('ANCHORED anchor changed')
        if not copy.exists():shutil.copyfile(anchor,copy)
        args.out=d
        args.ridge=ridge
        args.reuse_refs=refs
        args.inner=35
        args.outer=2
        args.gamma=2.
        args.seconds=300.
        args.sep_seconds=60.
        args.master_seconds=120.
        args.sep_batch=128
        args.closure_rounds=1
        core.anchored(args,data,weeks,index,
                      dict(settings,method='ANCHORED',ridge=ridge,
                           anchor_sha256=sha(anchor),references_sha256=sha(refs)))
        result=d/'anchored'/'anchored_progress.json'
        if not result.exists() or not json.loads(result.read_text()).get('complete'):
            raise RuntimeError('ANCHORED training incomplete at ridge '+str(ridge))
        print('Completed full-training ANCHORED at ridge',ridge,flush=True)


def policy_files(args):
    return {'CASE_NO_CONTEXT':args.root/'none'/'case_policy.json',
            'CASE_COHORT':args.root/'cohort'/'case_policy.json',
            'ANCHORED_COHORT_RIDGE_0p001':args.root/'cohort'/'sensitivity'/label(.001)/'anchored'/'anchored_progress.json',
            'ANCHORED_COHORT_RIDGE_0p003':args.root/'cohort'/'sensitivity'/label(.003)/'anchored'/'anchored_progress.json'}


def freeze_payload(args,ident):
    policies=policy_files(args)
    supporting=[args.root/'none'/'training_features.json',
                args.root/'cohort'/'training_features.json',
                args.root/'none'/'case_bic_path.json',
                args.root/'cohort'/'case_bic_path.json']
    for path in [*policies.values(),*supporting]:
        if not path.is_file():
            raise RuntimeError('Full-training output missing: '+str(path))
    features={context:json.loads((args.root/context/'training_features.json').read_text())
              for context in ('none','cohort')}
    for name,path in policies.items():
        doc=json.loads(path.read_text())
        p=len(features['cohort' if 'COHORT' in name else 'none']['feature_names'])
        w=np.asarray(doc['w'],float)
        if len(w)!=p or not np.isfinite(w).all():
            raise RuntimeError('Policy coefficients wrong dimension/NaN: '+name)
        if name.startswith('ANCHORED') and not doc.get('complete'):
            raise RuntimeError('Decision-aware fitting incomplete: '+name)
    return dict(training_identity=ident,
                policy_hashes={name:sha(path) for name,path in policies.items()},
                supporting_hashes={str(path.relative_to(args.root)):sha(path)
                                   for path in supporting},
                primary='ANCHORED_COHORT_RIDGE_0p001',
                comparator='CASE_COHORT',
                secondary='CASE_NO_CONTEXT',
                sensitivity='ANCHORED_COHORT_RIDGE_0p003',
                no_validation_tuning=True)


def freeze(args,ident):
    path=args.root/'frozen_policies.json'
    payload=freeze_payload(args,ident)
    if path.exists():
        if json.loads(path.read_text())!=payload:
            raise RuntimeError('FROZEN TRAINING POLICY HAS CHANGED; do not re-freeze')
    else:
        if (args.root/'validation'/'validation_scores.json').exists():
            raise RuntimeError('Validation outcomes exist without prior frozen policies')
        save(path,payload)
    print('Frozen full-training policies:',path,flush=True)


def validate_freeze(args,ident):
    path=args.root/'frozen_policies.json'
    if not path.is_file():raise RuntimeError('Run --stage freeze before evaluation')
    if json.loads(path.read_text())!=freeze_payload(args,ident):
        raise RuntimeError('Policy, scientific code, or training input changed after freeze')
    return sha(path)


def read_evaluation(args,context):
    manifest,data,weeks,index=read_split(args,'validation')
    features=json.loads((args.root/context/'training_features.json').read_text())
    if context=='cohort':
        x,names,mu,sd=core.add_context(data['X'],weeks,index,
                 variant='cohort',scaler=(features['extra_mean'],features['extra_sd']))
        data['X']=x
        if list(manifest['feature_names'])+list(names)!=features['feature_names']:
            raise RuntimeError('Cohort context features changed')
    elif list(manifest['feature_names'])!=features['feature_names']:
        raise RuntimeError('No-context feature names changed')
    return data,weeks,index


def score(args,ident):
    from surgery.anchored_training import verify_policy
    from surgery.anchored_core import key
    fhash=validate_freeze(args,ident)  # BEFORE reading validation outcomes
    os.environ['SURGERY_RESEARCH_LINK']='inverse'
    no,nw,ni=read_evaluation(args,'none')
    co,weeks,index=read_evaluation(args,'cohort')
    if {(w.monday,w.group) for w in nw}!={(w.monday,w.group) for w in weeks}:
        raise RuntimeError('Validation rosters differ between feature specifications')
    vec={name:np.asarray(json.loads(path.read_text())['w'],float)
         for name,path in policy_files(args).items()}
    vec['BOOKED']=np.zeros(no['X'].shape[1])
    data_for={name:(no,nw,ni) if name in ('BOOKED','CASE_NO_CONTEXT')
              else (co,weeks,index) for name in vec}
    ev=args.root/'validation'
    ev.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        (ev/('seed_'+str(seed))).mkdir(parents=True,exist_ok=True)
    args.fold='validation'
    args.seconds=300.
    refs=hindsight(args,weeks)
    outfile=ev/'validation_scores.json'
    report=json.loads(outfile.read_text()) if outfile.exists() else dict(
        protocol='frozen_2012_18_calendar_week_development_validation',
        frozen_policy_sha256=fhash,case_count=5779,seed_results={},
        per_calendar_week=[],no_hyperparameter_selection=True)
    if report['frozen_policy_sha256']!=fhash:
        raise RuntimeError('Scores were generated from a different frozen policy')
    for seed in SEEDS:
        os.environ['SURGERY_ANCHORED_HASH_SEED']=str(seed)
        report['seed_results'].setdefault(str(seed),{})
        for name,weights in vec.items():
            if name in report['seed_results'][str(seed)]:continue
            data,wks,idx=data_for[name]
            verified=verify_policy(ev/('seed_'+str(seed)),name,wks,idx,
                data['X'],data['booked'],weights,.8,30.,300.,1,args.workers)
            c=0;delta=0;realized=0;hindsight_cost=0;rows=[]
            for wk in wks:
                components=wk.components()
                keys=[key(comp) for comp in components]
                if any(not verified[k]['complete'] or not refs[k]['complete'] for k in keys):
                    raise RuntimeError('Incomplete certified validation component')
                nc=sum(comp.n for comp in components)
                rc=sum(float(verified[k]['realized_cost']) for k in keys)
                hc=sum(float(refs[k]['realized_cost']) for k in keys)
                c+=nc;delta+=rc-hc;realized+=rc;hindsight_cost+=hc
                rows.append(dict(method=name,seed=seed,calendar_week=wk.monday,
                                 site=wk.group,cases=nc,regret=rc-hc,
                                 realized_cost=rc,hindsight_cost=hc))
            if c!=5779:raise RuntimeError('Validation scored an incomplete case cohort')
            report['per_calendar_week']=[r for r in report['per_calendar_week']
                 if not(r['method']==name and r['seed']==seed)]+rows
            report['seed_results'][str(seed)][name]=dict(complete=True,
                cases=c,regret_per_case=delta/c,realized_cost=realized,
                hindsight_cost=hindsight_cost)
            save(outfile,report)
            print('Validation seed',seed,name,round(delta/c,4),flush=True)
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'
    summarize(args)


def summarize(args):
    path=args.root/'validation'/'validation_scores.json'
    if not path.is_file():raise RuntimeError('Run --stage evaluate first')
    r=json.loads(path.read_text())
    rows=r['per_calendar_week']
    names=('BOOKED','CASE_NO_CONTEXT','CASE_COHORT',
           'ANCHORED_COHORT_RIDGE_0p001','ANCHORED_COHORT_RIDGE_0p003')
    idx={(a['method'],a['seed'],a['calendar_week'],a['site']):a for a in rows}
    mondays=sorted({a['calendar_week'] for a in rows})
    if len(mondays)!=18 or len(idx)!=18*3*len(SEEDS)*len(names):
        raise RuntimeError('Validation results incomplete; cannot summarize')
    totalcases=5779
    means={}
    for name in names:
        sub=[a for a in rows if a['method']==name]
        means[name]=sum(a['regret'] for a in sub)/(totalcases*len(SEEDS))
    rng=np.random.default_rng(20261019)
    reference='CASE_COHORT'
    paired={}
    for name in names:
        if name==reference:continue
        weekly=[]
        for monday in mondays:
            items=[]
            for seed in SEEDS:
                matched=[idx[(name,seed,monday,site)] for site in ('TGH','TWH-main','TWH-day')]
                baselines=[idx[(reference,seed,monday,site)] for site in ('TGH','TWH-main','TWH-day')]
                if [a['cases'] for a in matched]!=[b['cases'] for b in baselines]:
                    raise RuntimeError('Paired validation cases do not match')
                items.append((sum(a['regret']-b['regret'] for a,b in zip(matched,baselines)),
                              sum(a['cases'] for a in matched)))
            weekly.append((np.mean([v[0] for v in items]),np.mean([v[1] for v in items])))
        mat=np.array(weekly,float)
        draws=rng.integers(0,18,size=(10000,18))
        totals=mat[draws].sum(axis=1)
        observed=mat[:,0].sum()/mat[:,1].sum()
        paired[name]=dict(difference_per_case=float(observed),
           ci95=list(map(float,np.quantile(totals[:,0]/totals[:,1],[.025,.975]))))
    output=dict(methods=means,paired_versus_case_cohort=paired,
         calendar_weeks=18,cases=5779,
         primary='ANCHORED_COHORT_RIDGE_0p001',
         interval='Unadjusted 95% paired calendar-week cluster bootstrap',
         caveat='Independent development-validation, not final 2013 test. Sensitivity is not a second model-selection chance.')
    save(args.root/'validation'/'validation_summary.json',output)
    print(json.dumps(output,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('case','anchored','freeze','evaluate','summary'),required=True)
    p.add_argument('--root',type=Path,default=Path('artifacts/frozen_context_validation_v1'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts/weekly_rich_leakfix_v1'))
    p.add_argument('--workbook',type=Path,default=Path('data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    p.add_argument('--workers',type=int,default=12)
    args=p.parse_args()
    if not 1<=args.workers<=15:p.error('Require 1 to 15 solver workers')
    # Fixed preregistered method settings; not exposed as validation-tunable flags.
    args.alpha=.8
    args.h=30.
    args.gamma=2.
    args.case_inner=500
    args.inner=35
    args.outer=2
    args.seconds=300.
    args.sep_seconds=60.
    args.master_seconds=120.
    args.sep_batch=128
    args.closure_rounds=1
    args.lambda_grid=LAMBDA_GRID
    args.seeds=SEEDS
    args.ridge=.001
    args.reuse_refs=None
    args.fold_safe=True
    os.environ['SURGERY_RESEARCH_LINK']='inverse'
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'
    ident=assert_identity(args)
    if args.stage in ('case','anchored') and (args.root/'frozen_policies.json').exists():
        raise RuntimeError('Frozen experiment cannot be fitted/tuned again')
    if args.stage=='case':fit_case(args,ident)
    elif args.stage=='anchored':fit_anchored(args,ident)
    elif args.stage=='freeze':freeze(args,ident)
    elif args.stage=='evaluate':score(args,ident)
    else:
        validate_freeze(args,ident)
        summarize(args)


if __name__=='__main__':main()