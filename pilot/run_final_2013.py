#!/usr/bin/env python3
"""One-time 2013 final evaluation after full pre-2013 fitting and training-only audit.

No development/validation data is held out. Test outcomes are not used for
policy fitting, feature estimation or audit. The workbook cohort builder sees
source records to establish cleaning rules; only stage=test uses 2013 cases
for decisions, costs or metrics, after the model freeze.

Planning rosters and eligibility are reconstructed from performed events;
prospective availability is an explicit limitation, not proven by this audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pilot import run_pilot as core
from pilot.run_fold_pilot import fold_lasso, hindsight
from run_weekly_rich import read_cohort
from surgery.anchored_core import key
from surgery.anchored_training import build_reference_library, verify_policy, learn
from surgery.planner import display_and_duration
from surgery.weekly_data import build_weeks, performed_events
from surgery.weekly_planner import loads as room_loads
from surgery.rich_features import RichCorrectionEncoder

ALPHA, H, GAMMA = .8, 30., 2.
LAMBDAS = (.0005,.001,.002,.004,.008,.016,.032)
RIDGES = (.001,.003)
SEEDS = (0,1,2)
# Fixed before examining the 2013 final-test outcomes.
BUDGET = dict(workers=12, seconds=600, case_inner=650,
              vf_outer=4, vf_inner=180, gap_outer=3, gap_inner=120,
              anchored_outer=4, anchored_inner=20, anchored_closure=2,
              anchored_sep_batch=192, anchored_sep_seconds=180,
              anchored_master_seconds=300, anchored_master_threads=4,
              evaluation_seconds=900,
              evaluation_retry_seconds=1800, reference_retry_seconds=1200)
METHODS = ('BOOKED','CASE_NONE','CASE_COHORT','LASSO_COHORT','ANCHORED_0p001',
           'ANCHORED_0p003','VF_COHORT','GAP_COHORT')
SOURCE_FILES = ('pilot/run_final_2013.py','pilot/audit_final_2013.py',
                'pilot/run_pilot.py',
                'pilot/run_fold_pilot.py','run_weekly_rich.py',
                'surgery/data.py','surgery/rich_features.py',
                'surgery/weekly_data.py','surgery/planner.py',
                'surgery/weekly_training.py','surgery/gap_learning.py',
                'surgery/weekly_planner.py','surgery/anchored_core.py',
                'surgery/anchored_master.py','surgery/anchored_training.py',
                'surgery/anchored_oracles.py','surgery/anchored_weekly_solver.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def dump_pickle(path,obj):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp,path)


def read_pickle(path):
    # Files are generated locally under the fingerprinted experiment root;
    # never load pickles from unknown parties.
    with Path(path).open('rb') as f:return pickle.load(f)


def label(ridge):return 'ridge_'+format(ridge,'.6g').replace('.','p')


def method_paths(root):
    return {
        'CASE_NONE':root/'none'/'case_policy.json',
        'CASE_COHORT':root/'cohort'/'case_policy.json',
        'LASSO_COHORT':root/'cohort'/'fold_lasso.json',
        'ANCHORED_0p001':root/'cohort'/'sensitivity'/label(.001)/'anchored'/'anchored_progress.json',
        'ANCHORED_0p003':root/'cohort'/'sensitivity'/label(.003)/'anchored'/'anchored_progress.json',
        'VF_COHORT':root/'cohort'/'vf'/'vf_progress.json',
        'GAP_COHORT':root/'cohort'/'gap'/'gap_progress.json',
    }


def identity(args):
    if not args.workbook.is_file():raise RuntimeError('Missing workbook: '+str(args.workbook))
    return dict(experiment='whole_2011_2012_train_single_2013_final_test',
                workbook_sha256=sha(args.workbook),
                source_sha256={p:sha(p) for p in SOURCE_FILES},
                budget=BUDGET,alpha=ALPHA,h=H,gamma=GAMMA,
                lambdas=list(LAMBDAS),ridges=list(RIDGES),seeds=list(SEEDS),
                methods=list(METHODS),
                time_limits='fixed per independent MIP; incomplete solves must be reported, never scored',
                training_cutoff='2013-01-01',
                feature_rule='train leave-date-out, test transform with all pre-2013 training only',
                current_protocol='no development-validation; test once after freeze')


def settings(args):
    state=identity(args)
    args.root.mkdir(parents=True,exist_ok=True)
    path=args.root/'experiment_settings.json'
    if path.is_file():
        if json.loads(path.read_text())!=state:
            raise RuntimeError('Source, workbook or final-experiment settings changed. '
                               'Keep the old artifacts and use a NEW root, not a checkpoint overwrite.')
    else:
        if (args.root/'FINAL_FREEZE.json').exists() or (args.root/'test').exists():
            raise RuntimeError('Cannot create settings around existing frozen/test outputs')
        save(path,state)
    return state


def split_rows(cohort,part):
    frame=cohort.loc[cohort.split.eq('train' if part=='train' else 'test')].copy()
    if frame.empty or frame.case_id.duplicated().any():
        raise RuntimeError('Empty or duplicate case IDs in '+part)
    if part=='train' and not (frame.date < '2013-01-01').all():
        raise RuntimeError('2013 case included in training')
    if part=='test' and not (frame.date >= '2013-01-01').all():
        raise RuntimeError('Pre-2013 case included in final test')
    return frame.reset_index(drop=True)


def checked_weeks(cohort,events,part,frame):
    # PREP uses only events from before 2013; test's performed events are
    # considered only after freeze to reconstruct retrospective eligibility.
    ws=build_weeks(cohort,events,split=part,prior_dates=1)
    actual_ids={int(cid) for w in ws for cid in w.case_ids}
    target_ids=set(map(int,frame.case_id))
    if len(actual_ids)!=sum(w.n for w in ws) or actual_ids!=target_ids:
        raise RuntimeError('Week rosters and '+part+' case IDs disagree')
    return ws,{int(cid):i for i,cid in enumerate(frame.case_id)}


def prepare(args):
    settings(args)
    manifest=args.root/'training_manifest.json'
    if manifest.exists():
        verify_prepared(args)
        print('Full pre-2013 training artifacts already verified',flush=True)
        return
    raw,cohort=read_cohort(args.workbook)
    tr=split_rows(cohort,'train')
    # Training must contain every retained pre-2013 case, including Dec 31.
    if len(tr)!=24506:
        raise RuntimeError(f'Expected 24,506 pre-2013 training cases, found {len(tr)}; audit source cohort')
    historical_events=performed_events(raw)
    historical_events=historical_events.loc[historical_events.date<'2013-01-01'].copy()
    w,idx=checked_weeks(tr,historical_events,'train',tr)
    encoder=RichCorrectionEncoder(history_mode='leave_date_out')
    X=encoder.fit_transform(tr)
    if len({x.monday for x in w})!=79:
        raise RuntimeError('Expected all 79 pre-2013 calendar Mondays')
    if not np.isfinite(X).all() or not np.allclose(X[:,0],1.):
        raise RuntimeError('Invalid training feature matrix')
    cols={
        'X':np.asarray(X,dtype=np.float64),
        'case_id':tr.case_id.to_numpy(np.int64),
        'booked':tr.booked.to_numpy(float),
        'actual':tr.actual.to_numpy(float),
    }
    tmp=args.root/'training_data.tmp.npz'
    np.savez_compressed(tmp,**cols)
    os.replace(tmp,args.root/'training_data.npz')
    dump_pickle(args.root/'feature_encoder.pkl',encoder)
    dump_pickle(args.root/'training_weeks.pkl',w)
    days=sorted({z.monday for z in w})
    save(manifest,dict(cases=len(tr),calendar_weeks=len(days),group_weeks=len(w),
        components=sum(len(x.components()) for x in w),
        min_monday=days[0],max_monday=days[-1],
        first_test_case_id_excluded=True,
        training_npz_sha256=sha(args.root/'training_data.npz'),
        train_weeks_sha256=sha(args.root/'training_weeks.pkl'),
        encoder_sha256=sha(args.root/'feature_encoder.pkl'),
        feature_names=list(encoder.names),
        note='2012 formerly held-out validation weeks are now training; no validation policy selection'))
    print(f'PREPARE: {len(tr)} training cases, {len(days)} Mondays, '
          f'{len(w)} site-weeks, {sum(len(x.components()) for x in w)} components',flush=True)


def verify_prepared(args):
    doc=json.loads((args.root/'training_manifest.json').read_text())
    for name,field in [('training_data.npz','training_npz_sha256'),
                       ('training_weeks.pkl','train_weeks_sha256'),
                       ('feature_encoder.pkl','encoder_sha256')]:
        if sha(args.root/name)!=doc[field]:
            raise RuntimeError('Training artifact was changed: '+name)
    if doc['cases']!=24506 or doc['calendar_weeks']!=79:
        raise RuntimeError('Incomplete pre-2013 cohort')
    return doc


def training(args,context):
    manifest=verify_prepared(args)
    with np.load(args.root/'training_data.npz') as z:
        data={k:z[k].copy() for k in z.files}
    weeks=read_pickle(args.root/'training_weeks.pkl')
    index={int(cid):i for i,cid in enumerate(data['case_id'])}
    if len(weeks)!=manifest['group_weeks'] or sum(w.n for w in weeks)!=len(data['case_id']):
        raise RuntimeError('Cached training-week roster changed')
    if context=='cohort':
        d=args.root/'cohort'
        extra=d/'context_scaler.json'
        if extra.exists():
            old=json.loads(extra.read_text())
            X,names,means,sd=core.add_context(data['X'],weeks,index,variant='cohort',
                               scaler=(old['means'],old['sd']))
            if list(names)!=old['names']:
                raise RuntimeError('Cohort feature list mismatch')
        else:
            X,names,means,sd=core.add_context(data['X'],weeks,index,variant='cohort')
            save(extra,dict(names=list(names),means=list(means),sd=list(sd),
                            training_manifest_sha256=sha(args.root/'training_manifest.json')))
        data['X']=X
    return data,weeks,index


def fixed_args(args, out):
    args.out=out
    args.fold_safe=True
    args.alpha=ALPHA;args.h=H;args.gamma=GAMMA
    args.lambda_grid=LAMBDAS
    args.case_inner=BUDGET['case_inner'];args.inner=BUDGET['anchored_inner']
    args.outer=BUDGET['anchored_outer']
    args.seconds=BUDGET['seconds'];args.sep_seconds=BUDGET['anchored_sep_seconds']
    args.master_seconds=BUDGET['anchored_master_seconds']
    args.sep_batch=BUDGET['anchored_sep_batch']
    args.closure_rounds=BUDGET['anchored_closure']
    args.workers=BUDGET['workers'];args.master_threads=BUDGET['anchored_master_threads']
    args.reuse_refs=None
    return args


def train_case(args):
    settings(args)
    if (args.root/'FINAL_FREEZE.json').exists():raise RuntimeError('Policies already frozen')
    for context in ('none','cohort'):
        data,_,_=training(args,context)
        root=args.root/context;root.mkdir(parents=True,exist_ok=True)
        a=fixed_args(args,root)
        fold_lasso(a,data,root)
        path=root/'case_policy.json'
        if path.exists():
            doc=json.loads(path.read_text())
            if len(doc.get('w',[]))!=data['X'].shape[1]:
                raise RuntimeError('CASE policy feature mismatch')
        else:
            core.case(a,data,dict(policy_role=context,experiment='full_pre2013_train',
                       experiment_selection='ANCHORED ridge 0.001 primary; 0.003 sensitivity',
                       frozen_input_sha256=sha(args.root/'training_manifest.json')))
        print('Complete CASE:',context,flush=True)


def training_settings(args,method,context='cohort'):
    return dict(method=method,link='inverse',context=context,
                train_sha256=sha(args.root/'training_data.npz'),
                cohort_sha256=sha(args.root/'cohort'/'case_policy.json'),
                experiment_sha256=sha(args.root/'experiment_settings.json'))


def train_anchored(args):
    settings(args)
    if (args.root/'FINAL_FREEZE.json').exists():raise RuntimeError('Policies already frozen')
    data,weeks,index=training(args,'cohort')
    root=args.root/'cohort'
    case=root/'case_policy.json'
    if not case.is_file():raise RuntimeError('Run case first')
    a=fixed_args(args,root)
    refroot=root/'train_refs';refroot.mkdir(parents=True,exist_ok=True)
    refs=refroot/'anchored_references.json'
    reference_policies={'CASE_COHORT':np.asarray(json.loads(case.read_text())['w'],float)}
    try:
        build_reference_library(refroot,weeks,index,data,reference_policies,
             ALPHA,H,float(BUDGET['seconds']),1,BUDGET['workers'])
    except RuntimeError as error:
        if 'lack an EXACT' not in str(error):raise
        print('Retry only unfinished reference components with larger solver limit',flush=True)
        build_reference_library(refroot,weeks,index,data,reference_policies,
             ALPHA,H,float(BUDGET['reference_retry_seconds']),1,BUDGET['workers'])
    for ridge in RIDGES:
        target=root/'sensitivity'/label(ridge)
        target.mkdir(parents=True,exist_ok=True)
        dest=target/'case_policy.json'
        if dest.exists() and sha(dest)!=sha(case):raise RuntimeError('Anchor case file changed')
        if not dest.exists():shutil.copyfile(case,dest)
        fixed_args(args,target)
        output=target/'anchored';output.mkdir(parents=True,exist_ok=True)
        own_ref=output/'anchored_references.json'
        if own_ref.exists() and sha(own_ref)!=sha(refs):
            # The train method may improve its own feasible schedule library;
            # in that situation the existing output must be resumed as-is.
            if not (output/'anchored_progress.json').exists():
                raise RuntimeError('Unexpected mismatched ANCHORED references')
        elif not own_ref.exists():shutil.copyfile(refs,own_ref)
        checkpoint=output/'training_config.json'
        conf=dict(training_settings(args,'anchored_'+str(ridge)),ridge=ridge,
                  ref_root_sha256=sha(refs),iterations=BUDGET['anchored_inner'],
                  outer=BUDGET['anchored_outer'],sep_batch=BUDGET['anchored_sep_batch'])
        if checkpoint.exists():
            if json.loads(checkpoint.read_text())!=conf:
                raise RuntimeError('ANCHORED settings changed during fitting')
        else:save(checkpoint,conf)
        # Call learn directly. The legacy pilot wrapper silently clamps
        # iterations=min(12,args.inner), which defeats our declared budget.
        current_refs=json.loads(own_ref.read_text())
        initial=np.asarray(json.loads(case.read_text())['w'],float)
        result=learn(output,weeks,index,data,current_refs,initial,ALPHA,H,
            gamma=GAMMA,lam=0.,ridge=ridge,reference=0.,reference_w=initial,
            prox=.15,coef_prox=1e-5,outer=BUDGET['anchored_outer'],
            iterations=BUDGET['anchored_inner'],
            closure_rounds=BUDGET['anchored_closure'],
            seconds=BUDGET['seconds'],sep_seconds=BUDGET['anchored_sep_seconds'],
            master_seconds=BUDGET['anchored_master_seconds'],
            threads=1,workers=BUDGET['workers'],
            master_threads=BUDGET['anchored_master_threads'],
            sep_batch=BUDGET['anchored_sep_batch'],line_fractions=(1.,.5,.25,.125))
        p=output/'anchored_progress.json'
        if not p.is_file() or not json.loads(p.read_text()).get('complete'):
            raise RuntimeError('Unfinished ANCHORED training: '+str(ridge))
        print('Complete ANCHORED ridge',ridge,flush=True)


def train_vf(args):
    from surgery.weekly_training import fit_vf
    settings(args)
    if (args.root/'FINAL_FREEZE.json').exists():raise RuntimeError('Policies already frozen')
    data,weeks,index=training(args,'cohort')
    root=args.root/'cohort'
    case=json.loads((root/'case_policy.json').read_text())
    lam=float(case['lambda_l1'])
    output=root/'vf'
    settings_vf=training_settings(args,'vf');settings_vf['lambda_l1']=lam
    result=fit_vf(output,weeks,data['X'],data['booked'],data['actual'],index,
                  ALPHA,H,lam,case['w'],settings_vf,
                  BUDGET['vf_outer'],BUDGET['vf_inner'],BUDGET['seconds'],1,BUDGET['workers'])
    if not result.get('complete'):raise RuntimeError('VF training is incomplete')


def train_gap(args):
    from surgery.gap_learning import train_gap as fit_gap
    settings(args)
    if (args.root/'FINAL_FREEZE.json').exists():raise RuntimeError('Policies already frozen')
    data,weeks,index=training(args,'cohort')
    root=args.root/'cohort';case=json.loads((root/'case_policy.json').read_text())
    lam=float(case['lambda_l1'])
    record=fit_gap(root/'gap',weeks,index,data['X'],data['booked'],lam,case['w'],
                   dict(training_settings(args,'gap'),lambda_l1=lam),
                   ALPHA,H,GAMMA,BUDGET['gap_outer'],BUDGET['gap_inner'],6.,1.5,
                   BUDGET['seconds'],1,BUDGET['workers'],
                   refine_seconds=2*BUDGET['seconds'],refinement_budget=2)
    if not record.get('complete'):
        raise RuntimeError('GAP unresolved; inspect gap_progress.json and certificate intervals. '
                           'Do not freeze an incomplete method.')


def complete_policies(args):
    ret=method_paths(args.root)
    features=verify_prepared(args)['feature_names']
    for name,p in ret.items():
        if not p.is_file():raise RuntimeError('Missing final fitted policy: '+str(p))
        doc=json.loads(p.read_text())
        n=len(features)+(2 if name not in ('CASE_NONE',) else 0)
        w=np.asarray(doc.get('w',doc.get('weights')),dtype=float)
        if w.shape!=(n,) or not np.isfinite(w).all():
            raise RuntimeError('Invalid fitted coefficient dimension: '+name)
        if name.startswith(('ANCHORED','VF_','GAP_')) and not doc.get('complete'):
            raise RuntimeError('Decision optimizer unfinished: '+name)
        if name.startswith('ANCHORED'):
            for row in doc.get('outer_history',[]):
                for trial in row.get('trials',[]):
                    if trial.get('reason')=='unresolved_deployed_solver':
                        raise RuntimeError('Unresolved training deployment; stop before test')
        if name=='GAP_COHORT' and doc.get('paused'):
            raise RuntimeError('VF-GAP comparison uncertified')
    return ret


def frozen_payload(args):
    m=verify_prepared(args)
    files=complete_policies(args)
    audit=args.root/'train_audit'/'audit_summary.json'
    if not audit.is_file():raise RuntimeError('Run training-only operational audit first')
    doc=json.loads(audit.read_text())
    if not doc.get('complete') or doc.get('unresolved_certifications',1)>0:
        raise RuntimeError('Training operational audit incomplete')
    files.update({'TRAINING_MANIFEST':args.root/'training_manifest.json',
                  'ENCODER':args.root/'feature_encoder.pkl',
                  'TRAINING_DATA':args.root/'training_data.npz',
                  'TRAINING_ROSTERS':args.root/'training_weeks.pkl',
                  'COHORT_SCALER':args.root/'cohort'/'context_scaler.json',
                  'AUDIT':audit})
    return dict(settings_sha256=sha(args.root/'experiment_settings.json'),
                sources=identity(args)['source_sha256'],
                policy_sha256={name:sha(path) for name,path in files.items()},
                primary='ANCHORED_0p001',comparator='CASE_COHORT',
                secondary=['CASE_NONE','ANCHORED_0p003','VF_COHORT','GAP_COHORT'],
                test_unopened=True,
                acknowledged_operational_risks=doc['operational_risk_flags'])


def freeze(args):
    payload=frozen_payload(args)
    file=args.root/'FINAL_FREEZE.json'
    if file.is_file():
        if json.loads(file.read_text())!=payload:raise RuntimeError('Frozen policies changed')
    else:
        if (args.root/'test').exists():raise RuntimeError('Test artifacts exist without freeze')
        report=payload['acknowledged_operational_risks']
        if report and not args.acknowledge_audit:
            raise RuntimeError('Operational issues need explicit review. Read train_audit/audit_summary.json '
                               'and rerun freeze with --acknowledge-audit to record acceptance '
                               '(this does not claim the issues are resolved).')
        save(file,payload)
    print('FROZEN 2013 final policies:',file,flush=True)


def validate_freeze(args):
    file=args.root/'FINAL_FREEZE.json'
    if not file.is_file():raise RuntimeError('Freeze every method BEFORE test scoring')
    if json.loads(file.read_text())!=frozen_payload(args):
        raise RuntimeError('Scientific code, audit, or fitted policies changed after freeze')
    return sha(file)


def group_rows_by_calendar(weeks,records,oracle,seed,method):
    results=[]
    for w in weeks:
        keys=[key(c) for c in w.components()]
        if any(not records[z].get('complete') or not oracle[z].get('complete') for z in keys):
            raise RuntimeError('Uncertified component in final evaluation')
        results.append(dict(calendar_week=w.monday,site=w.group,seed=seed,
            method=method,cases=int(w.n),
            realized_cost=sum(records[z]['realized_cost'] for z in keys),
            regret=sum(records[z]['realized_cost']-oracle[z]['realized_cost'] for z in keys),
            hindsight_cost=sum(oracle[z]['realized_cost'] for z in keys),
            room_days_over_720=sum(records[z]['room_days_over_720'] for z in keys),
            room_days_over_960=sum(records[z].get('room_days_over_960',0) for z in keys),
            max_realized_room_load=max(records[z]['max_realized_room_load'] for z in keys)))
    return results


def summarize(args):
    path=args.root/'test'/'test_scores.json'
    if not path.is_file():raise RuntimeError('Test scoring not finished')
    doc=json.loads(path.read_text())
    rows=doc['per_site_week'];index={(x['method'],x['seed'],x['calendar_week'],x['site']):x for x in rows}
    timekeys={(x['calendar_week'],x['site']) for x in rows}
    allweeks=sorted({k[0] for k in timekeys})
    if len(index)!=len(timekeys)*len(SEEDS)*len(METHODS):
        raise RuntimeError('Incomplete test cases/methods/seeds; cannot summarize')
    ncases=doc['test_cases']
    mean={name:sum(r['regret'] for r in rows if r['method']==name)/(ncases*len(SEEDS))
          for name in METHODS}
    comps={}
    # Resample calendar Mondays (NOT site-weeks or solver hash seeds).
    rng=np.random.default_rng(20261019)
    for name in METHODS:
        if name=='CASE_COHORT':continue
        weeks=[]
        for monday in allweeks:
            components=sorted(site for (w,site) in timekeys if w==monday)
            delta=0.;n=0
            for seed in SEEDS:
                for site in components:
                    obs=index[name,seed,monday,site]
                    ref=index['CASE_COHORT',seed,monday,site]
                    if obs['cases']!=ref['cases']:raise AssertionError('Paired cases changed')
                    delta+=obs['regret']-ref['regret'];n+=ref['cases']
            weeks.append((delta/len(SEEDS),n/len(SEEDS)))
        table=np.asarray(weeks,float)
        resamples=rng.integers(0,len(allweeks),size=(10000,len(allweeks)))
        totals=table[resamples].sum(axis=1)
        observed=float(table[:,0].sum()/table[:,1].sum())
        comps[name]=dict(difference_per_case=observed,
                         ci95=list(map(float,np.quantile(totals[:,0]/totals[:,1],[.025,.975]))))
    # Compare primary also to the capped inverse-link LASSO cohort benchmark.
    primary_lasso=[]
    for monday in allweeks:
        delta=0.;n=0
        for site in sorted(s for w,s in timekeys if w==monday):
            for seed in SEEDS:
                a=index['ANCHORED_0p001',seed,monday,site]
                b=index['LASSO_COHORT',seed,monday,site]
                if a['cases']!=b['cases']:raise AssertionError('LASSO paired cohort mismatch')
                delta+=a['regret']-b['regret'];n+=a['cases']
        primary_lasso.append((delta/len(SEEDS),n/len(SEEDS)))
    table=np.asarray(primary_lasso,float)
    sample=rng.integers(0,len(allweeks),size=(10000,len(allweeks)))
    weighted=table[sample].sum(axis=1)
    lasso_comp=dict(difference_per_case=float(table[:,0].sum()/table[:,1].sum()),
                    ci95=list(map(float,np.quantile(weighted[:,0]/weighted[:,1],[.025,.975]))))
    save(args.root/'test'/'test_summary.json',dict(
        final_2013_test=True,months='2013-01 through 2013-06',
        cases=ncases,calendar_weeks=len(allweeks),
        means=mean,paired_versus_cohort_case=comps,
        primary_vs_lasso_cohort=lasso_comp,
        primary='ANCHORED_0p001',unadjusted_primary_ci=True,
        seeds='three deterministic tie-breaking variants; NOT independent observations',
        interpretation='One-time final 2013 holdout, never selected a method on test outcomes'))


def score_test(args):
    frozen_sha=validate_freeze(args)
    os.environ['SURGERY_RESEARCH_LINK']='inverse'
    raw,cohort=read_cohort(args.workbook)
    test=split_rows(cohort,'test')
    if len(test)!=8774:
        raise RuntimeError(f'Expected 8,774 2013 cases; got {len(test)}; STOP before scoring')
    encoder=read_pickle(args.root/'feature_encoder.pkl')
    base=np.asarray(encoder.transform(test),float)
    weeks,index=checked_weeks(cohort,performed_events(raw),'test',test)
    manifest=verify_prepared(args)
    if base.shape[1]!=len(manifest['feature_names']):
        raise RuntimeError('Training/test feature dimensions differ')
    traincontext=json.loads((args.root/'cohort'/'context_scaler.json').read_text())
    cohortX,names,_,_=core.add_context(base,weeks,index,variant='cohort',
                    scaler=(traincontext['means'],traincontext['sd']))
    if list(names)!=traincontext['names']:raise RuntimeError('Changed final context feature names')
    tr={'X':base,'booked':test.booked.to_numpy(float),'actual':test.actual.to_numpy(float)}
    co={'X':cohortX,'booked':tr['booked'],'actual':tr['actual']}
    vec={name:np.asarray((lambda d:d.get('w',d.get('weights')))(json.loads(path.read_text())),float)
         for name,path in method_paths(args.root).items()}
    vec['BOOKED']=np.zeros(base.shape[1])
    out=args.root/'test';out.mkdir(parents=True,exist_ok=True)
    # The sole reading/scoring of 2013 outcomes starts after the frozen hash check.
    args.fold='2013_final';args.seconds=BUDGET['evaluation_seconds'];args.workers=BUDGET['workers']
    try:
        oracle=hindsight(args,weeks)
    except RuntimeError as error:
        if 'hindsight oracle incomplete' not in str(error) and 'oracle incomplete' not in str(error):
            raise
        print('Retrying unresolved hindsight components with longer time limit',flush=True)
        args.seconds=BUDGET['evaluation_retry_seconds']
        oracle=hindsight(args,weeks)
    file=out/'test_scores.json'
    report=json.loads(file.read_text()) if file.exists() else dict(
        frozen_sha256=frozen_sha,test_cases=len(test),
        test_mondays=len({w.monday for w in weeks}),
        test_site_weeks=len(weeks),test_components=sum(len(w.components()) for w in weeks),
        per_site_week=[],finished={},methods=list(METHODS))
    if report['frozen_sha256']!=frozen_sha or report['test_cases']!=len(test):
        raise RuntimeError('Cannot resume test results from another frozen experiment')
    for seed in SEEDS:
        os.environ['SURGERY_ANCHORED_HASH_SEED']=str(seed)
        for name in METHODS:
            if name in report['finished'].get(str(seed),[]):continue
            obj=tr if name in ('BOOKED','CASE_NONE') else co
            try:
                results=verify_policy(out/('seed_'+str(seed)),name,weeks,index,
                    obj['X'],obj['booked'],vec[name],ALPHA,H,
                    BUDGET['evaluation_seconds'],1,BUDGET['workers'])
            except RuntimeError as error:
                if 'exact deployed plans unresolved' not in str(error):raise
                print('Retrying only incomplete deployment components:',name,seed,flush=True)
                results=verify_policy(out/('seed_'+str(seed)),name,weeks,index,
                    obj['X'],obj['booked'],vec[name],ALPHA,H,
                    BUDGET['evaluation_retry_seconds'],1,BUDGET['workers'])
            scored=group_rows_by_calendar(weeks,results,oracle,seed,name)
            if sum(x['cases'] for x in scored)!=len(test):
                raise RuntimeError('Final scoring omitted test cases')
            report['per_site_week']=[r for r in report['per_site_week']
                if not(r['method']==name and r['seed']==seed)]+scored
            report['finished'].setdefault(str(seed),[]).append(name)
            save(file,report)
            print('2013 FINAL:',seed,name,'regret/case',
                  round(sum(x['regret'] for x in scored)/len(test),5),flush=True)
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'
    summarize(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,
                   choices=('prepare','case','anchored','vf','gap','audit','freeze','test','summary'))
    p.add_argument('--root',type=Path,default=Path('artifacts/final_2013_audited_v1'))
    p.add_argument('--workbook',type=Path,default=Path('data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    p.add_argument('--acknowledge-audit',action='store_true',
                   help='Explicitly record operational-limitations review before final test')
    args=p.parse_args()
    if args.stage=='prepare':prepare(args);return
    settings(args)
    if args.stage=='case':train_case(args)
    elif args.stage=='anchored':train_anchored(args)
    elif args.stage=='vf':train_vf(args)
    elif args.stage=='gap':train_gap(args)
    elif args.stage=='audit':
        from pilot.audit_final_2013 import audit_training
        audit_training(args)
    elif args.stage=='freeze':freeze(args)
    elif args.stage=='test':score_test(args)
    else:validate_freeze(args);summarize(args)

if __name__=='__main__':main()