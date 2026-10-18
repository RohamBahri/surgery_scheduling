#!/usr/bin/env python3
"""Training-only pilot: CASE, VF, VF-GAP, and ANCHORED; two response links and
case-only versus reconstructed weekly scheduling-context predictors.

Never loads validation/test matrices. Existing training cohort and planner
are unchanged. The reconstructed rosters are retrospective proxies; do not
claim these scheduling-context features were prospectively observable.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_training(args):
    from run_weekly_train import load_weeks, check_week_cases
    root = args.artifacts
    meta = json.loads((root / 'manifest.json').read_text())
    if digest(root / 'train.npz') != meta['matrix_sha256']['train']:
        raise RuntimeError('Frozen training matrix SHA256 mismatch')
    if digest(args.workbook) != meta['workbook_sha256']:
        raise RuntimeError('Frozen workbook SHA256 mismatch')
    with np.load(root / 'train.npz') as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if len(data['case_id']) != 18705 or len(meta['training_keys']) != 180:
        raise RuntimeError('Unexpected training cohort')
    if len(meta['feature_names']) != data['X'].shape[1]:
        raise RuntimeError('Features and manifest disagree')
    weeks = load_weeks(args.workbook, meta, 'train')
    index = check_week_cases(weeks, data)
    return meta, data, weeks, index


def add_context(base, weeks, index):
    """Four booked-workload and eligibility features, never realized durations.

    WARNING: Week.slots and Week.allowed are derived from performed schedules
    and are NOT independently verified as known at recommendation time.
    """
    X = np.asarray(base, float)
    extra = np.full((len(X), 4), np.nan)
    names = ('site_week_booked_capacity_ratio',
             'surgeon_week_booked_capacity_ratio',
             'surgeon_week_case_count',
             'surgeon_eligible_room_days')
    for week in weeks:
        load = float(np.sum(week.booked)) / (480.0 * week.k)
        for s in range(len(week.surgeon_ids)):
            loc = np.flatnonzero(week.surgeon == s)
            booked = float(np.sum(week.booked[loc]))
            allowed = len(week.allowed[s])
            for j in loc:
                i = index[int(week.case_ids[j])]
                extra[i] = (load, booked / (480. * max(allowed, 1)),
                            float(len(loc)), float(allowed))
    if not np.isfinite(extra).all() or (np.std(extra, axis=0) < 1e-8).any():
        raise RuntimeError('Missing or constant weekly-context features')
    mean, sd = extra.mean(axis=0), extra.std(axis=0)
    return np.column_stack((X, (extra-mean)/sd)), names, mean.tolist(), sd.tolist()


def setup(args):
    os.environ['SURGERY_RESEARCH_LINK'] = args.link
    os.environ['SURGERY_ANCHORED_HASH_SEED'] = '0'
    if args.link == 'inverse':
        from surgery.planner import display_and_duration
        from surgery.weekly_training import response_with_derivative
        from surgery.anchored_core import hinge_response
        raw = np.array([-100., 0., 20., 100.])
        booked = np.full(len(raw), 200.)
        _, d = display_and_duration(raw, booked, args.alpha, args.h)
        lo = -args.alpha*min(args.h, 180.)
        hi = -lo
        expected = np.clip(raw, lo, hi)
        if not np.allclose(d-booked, expected):
            raise RuntimeError('Inverse link does not match effective correction')
        if not np.allclose(
                hinge_response(200., args.alpha, args.h).correction(raw),
                expected):
            raise RuntimeError('ANCHORED DC does not match the link')
        _, derivative = response_with_derivative(np.array([0.]),
                                                 np.array([200.]),
                                                 args.alpha, args.h)
        if not np.allclose(derivative, [1.]):
            raise RuntimeError('Inverse-link derivative mismatch')
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    code_files = ('surgery/planner.py', 'surgery/weekly_training.py',
                  'surgery/anchored_core.py', 'surgery/anchored_master.py',
                  'surgery/anchored_training.py', 'surgery/gap_learning.py',
                  'surgery/weekly_planner.py', 'surgery/anchored_weekly_solver.py')
    settings = dict(pilot_sha256=digest(__file__),
                    code_sha256={p:digest(p) for p in code_files},
                    original_train_sha256=digest(args.artifacts/'train.npz'),
                    source_manifest_sha256=digest(args.artifacts/'manifest.json'),
                    workbook_sha256=digest(args.workbook),
                    alpha=args.alpha, h=args.h, link=args.link,
                    context=args.context, inner_iterations=args.inner,
                    max_outer=args.outer, workers=args.workers,
                    seconds=args.seconds, sep_seconds=args.sep_seconds,
                    sep_batch=args.sep_batch,
                    closure_rounds=args.closure_rounds, ridge=args.ridge,
                    gamma=args.gamma, lambda_grid=list(args.lambda_grid),
                    hash_seeds=list(args.seeds),
                    roster_status='retrospectively reconstructed, not demonstrated ex ante')
    path = root/'pilot_settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise RuntimeError('Pilot settings changed; choose a fresh output directory')
    if not path.exists():
        write_json(path, settings)
    meta, data, weeks, index = load_training(args)
    if args.context == 'weekly':
        X, names, mean, sd = add_context(data['X'], weeks, index)
    else:
        X, names, mean, sd = data['X'].astype(float), (), (), ()
    if not np.allclose(X[:,0], 1.):
        raise RuntimeError('First feature must be the intercept')
    data['X'] = X
    features = dict(names=list(meta['feature_names'])+list(names),
                    extra_means=list(mean), extra_std=list(sd),
                    X_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                    caveat='Context uses reconstructed performed rosters, not proven prospective')
    fp = root/'pilot_features.json'
    if fp.exists() and json.loads(fp.read_text()) != features:
        raise RuntimeError('Feature matrix changed')
    if not fp.exists():
        write_json(fp, features)
    return data, weeks, index, settings


def initial_weights(args, p):
    obj = json.loads((args.artifacts/'lasso_policy.json').read_text())
    w = np.asarray(obj['weights'], float)
    if len(w)>p:
        raise RuntimeError('Baseline coefficient dimension mismatch')
    result = np.zeros(p)
    result[:len(w)] = w * (args.alpha if args.link=='inverse' else 1.)
    return result


def case(args, data, settings):
    from surgery.weekly_training import fit_fixed, asymmetric_error, response_with_derivative
    X, a, b = data['X'], data['actual'], data['booked']
    n,p = X.shape
    target = args.out/'case_policy.json'
    if target.exists():
        return json.loads(target.read_text())
    starts = [np.zeros(p), initial_weights(args,p)]
    rows = []
    for lam in sorted(args.lambda_grid, reverse=True):
        candidates = [fit_fixed(X,b,a,args.alpha,args.h,lam,start,
                                iterations=args.inner) for start in starts]
        chosen = min(candidates,key=lambda x:x['objective'])
        w = np.asarray(chosen['w'],float)
        post,_ = response_with_derivative(X@w,b,args.alpha,args.h)
        loss = float(asymmetric_error(a-post).mean())
        effect = np.abs(w[1:])*X[:,1:].std(axis=0)
        df = 1+int(np.count_nonzero(effect>0.05))
        # Quasi-BIC proxy on asymmetric loss, not a claim of exact likelihood df.
        bic = float(n*np.log(max(loss,1e-9))+df*np.log(n))
        row = dict(lambda_l1=lam, bic_proxy=bic, asymmetric_loss_per_case=loss,
                   active_effects_over_005min=df-1, coefficients=w.tolist(),
                   optimization=chosen['passes'])
        rows.append(row)
        starts = [w,initial_weights(args,p)]
        write_json(args.out/'case_bic_path.json', rows)
        print('CASE lambda',lam,'BIC proxy',round(bic,2),'active',df-1,flush=True)
    best=min(rows,key=lambda x:(x['bic_proxy'],x['lambda_l1']))
    result=dict(method='CASE_BIC_PILOT',w=best['coefficients'],
                lambda_l1=best['lambda_l1'],
                training_loss=best['asymmetric_loss_per_case'],
                bic_proxy=best['bic_proxy'],
                selection='training-only asymmetric quasi-BIC; approximate effective df',
                **settings)
    write_json(target,result)
    return result


def load_case(args):
    path=args.out/'case_policy.json'
    if not path.exists():
        raise RuntimeError('Run --stage case first')
    return json.loads(path.read_text())


def vf(args, data, weeks, index, settings):
    from surgery.weekly_training import fit_vf
    init=load_case(args)
    root=args.out/'vf';root.mkdir(exist_ok=True)
    desc=dict(pilot_settings=settings,anchor_sha256=digest(args.out/'case_policy.json'),
              lambda_l1=init['lambda_l1'],method='VF')
    result=fit_vf(root,weeks,data['X'],data['booked'],data['actual'],index,
                  args.alpha,args.h,init['lambda_l1'],init['w'],desc,
                  args.outer,args.inner,args.seconds,1,args.workers)
    print('VF:',result.get('stopping'),flush=True)


def gap(args, data, weeks, index, settings):
    from surgery.gap_learning import train_gap
    init=load_case(args)
    root=args.out/'gap';root.mkdir(exist_ok=True)
    desc=dict(pilot_settings=settings,anchor_sha256=digest(args.out/'case_policy.json'),
              lambda_l1=init['lambda_l1'],method='VF_GAP')
    result=train_gap(root,weeks,index,data['X'],data['booked'],
                     init['lambda_l1'],init['w'],desc,
                     args.alpha,args.h,args.gamma,args.outer,args.inner,6.,1.5,
                     args.seconds,1,args.workers,
                     refine_seconds=args.seconds*2,refinement_budget=0)
    print('VF-GAP:',result.get('stopping') or result.get('paused'),flush=True)


def anchored(args, data, weeks, index, settings):
    from surgery.anchored_training import build_reference_library,learn
    from surgery.anchored_core import key
    from surgery.weekly_planner import evaluate
    init=load_case(args)
    root=args.out/'anchored';root.mkdir(exist_ok=True)
    refs_file=root/'anchored_references.json'
    if not refs_file.exists() and args.reuse_refs is not None:
        shutil.copyfile(args.reuse_refs,refs_file)
        print('Reused training-only feasible schedule library',flush=True)
    if not refs_file.exists():
        policy_weights={'CASE':np.asarray(init['w'],float)}
        for name,path in [('VF',args.out/'vf'/'vf_progress.json'),
                          ('VF_GAP',args.out/'gap'/'gap_progress.json')]:
            if path.exists():
                policy_weights[name]=np.asarray(json.loads(path.read_text())['w'],float)
        refs=build_reference_library(root,weeks,index,data,policy_weights,
                                     args.alpha,args.h,args.seconds,1,args.workers)
    else:
        refs=json.loads(refs_file.read_text())
    expected={key(c) for week in weeks for c in week.components()}
    if set(refs)!=expected or not all(refs[k]['complete'] for k in expected):
        raise RuntimeError('Incomplete/mismatched reference library')
    if args.reuse_refs is not None:
        for week in weeks:
            for comp in week.components():
                rec=refs[key(comp)]
                if rec['cases']!=comp.n:
                    raise RuntimeError('Reference component mismatched')
                for anchor in rec['references']:
                    val=evaluate(comp,anchor['assignment'],comp.actual)['cost']
                    if abs(val-anchor['realized_cost'])>0.001:
                        raise RuntimeError('Reference realized cost mismatch')
    result=learn(root,weeks,index,data,refs,init['w'],args.alpha,args.h,
                 gamma=args.gamma,lam=0.,ridge=args.ridge,reference=0.,
                 reference_w=np.asarray(init['w'],float),
                 prox=.15,coef_prox=1e-5,outer=args.outer,
                 iterations=min(12,args.inner),closure_rounds=args.closure_rounds,
                 seconds=args.seconds,sep_seconds=args.sep_seconds,
                 master_seconds=args.master_seconds,threads=1,workers=args.workers,
                 master_threads=4,sep_batch=args.sep_batch,
                 line_fractions=(1.,.5,.25))
    print('ANCHORED:',result.get('stopping'),
          'accepted',len(result.get('accepted',[])),flush=True)


def compare(args,data,weeks,index):
    from surgery.anchored_training import verify_policy
    from surgery.anchored_core import key
    root=args.out/'comparison';root.mkdir(exist_ok=True)
    n=len(data['case_id']);p=data['X'].shape[1]
    policies={'BOOKED':np.zeros(p)}
    path=args.out/'case_policy.json'
    if path.exists():
        policies['CASE_BIC']=json.loads(path.read_text())['w']
    for name,path in [('VF',args.out/'vf'/'vf_progress.json'),
                      ('VF_GAP',args.out/'gap'/'gap_progress.json'),
                      ('ANCHORED',args.out/'anchored'/'anchored_progress.json')]:
        if path.exists() and json.loads(path.read_text()).get('complete',True):
            policies[name]=json.loads(path.read_text())['w']
    ref_path=(args.reuse_refs if args.reuse_refs and args.reuse_refs.exists()
              else args.out/'anchored'/'anchored_references.json')
    if not ref_path.exists():
        raise RuntimeError('Need --reuse-refs or ANCHORED reference library')
    refs=json.loads(ref_path.read_text())
    expected={key(c) for week in weeks for c in week.components()}
    if set(refs)!=expected or not all(refs[k].get('complete') for k in expected):
        raise RuntimeError('Incomplete hindsight references')
    hindsight=sum(float(refs[k]['oracle_cost']) for k in expected)
    result=dict(pilot='training_only',methods=list(policies),seeds={},
                hindsight_cost=hindsight,cases=n,
                warning='Different hash seeds are NOT independent datasets')
    for seed in args.seeds:
        os.environ['SURGERY_ANCHORED_HASH_SEED']=str(seed)
        directory=root/f'seed_{seed}';directory.mkdir(exist_ok=True)
        result['seeds'][str(seed)]={}
        for name,w in policies.items():
            try:
                if len(w)!=p:
                    raise RuntimeError('Feature dimension mismatch')
                dep=verify_policy(directory,name,weeks,index,data['X'],data['booked'],
                                  w,args.alpha,args.h,args.seconds,1,args.workers)
                if not all(v['complete'] for v in dep.values()):
                    raise RuntimeError('Uncertified deployment')
                total=sum(float(v['realized_cost']) for v in dep.values())
                result['seeds'][str(seed)][name]=dict(
                    complete=True,regret_per_case=(total-hindsight)/n,
                    realized_cost=total,certified_components=len(dep))
            except RuntimeError as exc:
                result['seeds'][str(seed)][name]=dict(complete=False,error=str(exc))
            write_json(args.out/'pilot_training_comparison.json',result)
            print('seed',seed,name,result['seeds'][str(seed)][name],flush=True)
    os.environ['SURGERY_ANCHORED_HASH_SEED']='0'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('prepare','case','vf','gap','anchored','compare'))
    p.add_argument('--workbook',type=Path,default=Path('data/UHNOperating_RoomScheduling2011-2013.xlsx'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts/weekly_rich_leakfix_v1'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-refs',type=Path)
    p.add_argument('--context',choices=('none','weekly'),required=True)
    p.add_argument('--link',choices=('original','inverse'),required=True)
    p.add_argument('--alpha',type=float,default=.8)
    p.add_argument('--h',type=float,default=30.)
    p.add_argument('--gamma',type=float,default=2.)
    p.add_argument('--inner',type=int,default=35)
    p.add_argument('--outer',type=int,default=2)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--master-seconds',type=float,default=120.)
    p.add_argument('--sep-seconds',type=float,default=60.)
    p.add_argument('--sep-batch',type=int,default=128)
    p.add_argument('--closure-rounds',type=int,default=1)
    p.add_argument('--ridge',type=float,default=.001)
    p.add_argument('--lambda-grid',default='0.0005,0.001,0.002,0.004,0.008,0.016,0.032')
    p.add_argument('--seeds',default='0,1,2')
    args=p.parse_args()
    args.lambda_grid=tuple(float(x) for x in args.lambda_grid.split(','))
    args.seeds=tuple(int(x) for x in args.seeds.split(','))
    if not (0<args.alpha<=1 and args.h>0 and args.outer>0 and args.inner>0
            and 1<=args.workers<=15 and args.seconds>0 and args.ridge>=0
            and all(v>0 for v in args.lambda_grid)):
        p.error('Invalid hyperparameters')
    data,weeks,index,settings=setup(args)
    print('Frozen TRAIN cohort',len(data['case_id']),len(weeks),
          data['X'].shape[1],'features',args.link,args.context,flush=True)
    if args.stage=='case':
        case(args,data,settings)
    elif args.stage=='vf':
        vf(args,data,weeks,index,settings)
    elif args.stage=='gap':
        gap(args,data,weeks,index,settings)
    elif args.stage=='anchored':
        anchored(args,data,weeks,index,settings)
    elif args.stage=='compare':
        compare(args,data,weeks,index)


if __name__=='__main__':
    main()
