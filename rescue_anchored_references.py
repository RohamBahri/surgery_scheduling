#!/usr/bin/env python3
"""Certify only unfinished weekly anchored references with larger solver budgets.

This auxiliary recovery tool intentionally is NOT included in SOURCE_FILES of
run_weekly_anchored.py. It calls the identical scientific planner and reference
construction as the original code, preserving the existing training fingerprint.
All completed references remain unchanged. Individual successful sub-solves are
checkpointed so a hard tie-break cannot repeatedly waste earlier solves.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_weekly_anchored import initialize
from surgery.anchored_core import key
from surgery.anchored_oracles import deployed
from surgery.planner import display_and_duration, reachable
from surgery.weekly_planner import evaluate, conservation_certificate, CAPACITY, TURNOVER, OVERTIME
from surgery.weekly_training import save_json


def cached_reference_for_component(c, local, candidates, saved, seconds, threads,
                                   *, solve=None, on_update=None):
    """Return certified reference row, or None and the failing solve status.

    saved is the durable cache for this single component; only COMPLETE
    solves are reusable. Reconstruct results using precisely the standard
    _week_task schema and insertion order. Never use an incomplete tie.
    """
    if on_update is None:
        on_update=lambda: None
    if solve is None:
        solve=deployed
    if 'hindsight' not in saved:
        result=solve(c,c.actual,seconds=seconds,threads=threads,hindsight=True)
        if not result.get('complete'):
            saved['last_failure']='hindsight: '+str(result.get('status'))
            on_update()
            return None,saved['last_failure']
        if result.get('assignment') is None:
            raise AssertionError('Complete hindsight result has no assignment')
        saved['hindsight']={'realized_cost':float(result['realized_cost']),
                            'assignment':list(map(int,result['assignment']))}
        saved.pop('last_failure',None)
        on_update()
    hindsight=saved['hindsight']
    reference_cache=saved.setdefault('references',{})
    for name,d in candidates.items():
        if name in reference_cache:
            continue
        result=solve(c,np.asarray(d,float),seconds=seconds,threads=threads)
        if not result.get('complete'):
            saved['last_failure']=name+': '+str(result.get('status'))
            on_update()
            return None,saved['last_failure']
        z=list(map(int,result['assignment']))
        ref={'source':name,
             'type':('box-aspiration' if name in ('BOX_CLIP','BOX_PAD') else 'shared-policy'),
             'beta':np.asarray(d,float).tolist(),
             'assignment':z,
             'realized_cost':float(result['realized_cost']),
             'predicted_cost':float(result['predicted_cost'])}
        if abs(evaluate(c,z,c.actual)['cost']-ref['realized_cost'])>1e-5:
            raise AssertionError('Reference objective inconsistent with assignment')
        reference_cache[name]=ref
        saved.pop('last_failure',None)
        on_update()
    return {'complete':True,'cases':c.n,
            'oracle_cost':float(hindsight['realized_cost']),
            'references':[reference_cache[name] for name in candidates]},None


def rescue(args):
    output=Path(args.output)
    settings_path=output/'anchored_settings.json'
    refs_path=output/'anchored_references.json'
    if not settings_path.exists() or not refs_path.exists():
        raise ValueError('Run the normal references stage first; no checkpoint found')
    original=json.loads(settings_path.read_text())
    base=json.loads((args.artifacts/'learning_settings.json').read_text())
    if 'SEED' in original.get('baseline_models_sha256',{}):
        raise ValueError('A warm-start SEED baseline requires a matching --initial-policy')
    identity=SimpleNamespace(
        workbook=args.workbook,artifacts=args.artifacts,gap_model=args.gap_model,
        initial_policy=None,output=output,site=original['site'],
        alpha=float(base['alpha']),h=float(base['h']),
        gamma=float(original['gamma']),lambda_l1=float(original['lambda_l1']),
        prox=float(original['prox']),coef_prox=float(original['coef_prox']),
        iterations=int(original['pDCA_iterations']),
        closure_rounds=int(original['closure_rounds']),
        max_outer=int(original['max_outer']),
        exact_sep_batch=int(original['exact_sep_batch']),
        resume_oracle_fix=False)
    _,_,weeks,index,data,policies,ident=initialize(identity)
    if ident!=original:
        raise ValueError('Dataset or scientific code differs from reference checkpoint')
    references=json.loads(refs_path.read_text())
    all_components={key(c):c for week in weeks for c in week.components()}
    pending=[name for name in sorted(all_components)
             if not references.get(name,{}).get('complete')]
    if args.component is not None:
        if args.component not in all_components:
            raise ValueError('Unknown component: '+args.component)
        pending=[k for k in pending if k==args.component]
    if not pending:
        print('No unfinished reference components remain.',flush=True)
        return True
    X=data['X'].astype(float)
    booked=data['booked'];actual=data['actual']
    lower,upper=reachable(booked,identity.alpha,identity.h)
    predictions={'BOOKED':booked.copy(),
                 'BOX_CLIP':np.clip(actual,lower,upper),
                 'BOX_PAD':upper}
    for name,w in policies.items():
        _,pred=display_and_duration(X@np.asarray(w,float),booked,
                                    identity.alpha,identity.h)
        predictions[name]=pred
    cache_path=output/'anchored_reference_rescue_cache.json'
    settings_digest=hashlib.sha256(settings_path.read_bytes()).hexdigest()
    cache=(json.loads(cache_path.read_text()) if cache_path.exists()
           else {'settings_sha256':settings_digest,'components':{}})
    if cache.get('settings_sha256')!=settings_digest:
        raise ValueError('Reference rescue cache belongs to another experiment')
    results=cache['components']
    failed=[]
    print(f'Pending: {len(pending)} reference components, '
          f'{args.seconds:.0f}s per solve, {args.threads} solver threads.',flush=True)
    for name in pending:
        c=all_components[name]
        rows=np.asarray([index[int(i)] for i in c.case_ids],dtype=int)
        methods={label:pred[rows] for label,pred in predictions.items()}
        prior=references.get(name,{})
        print(f'Component {name}: cases={c.n} rooms={c.k} '
              f'surgeons={len(c.surgeon_ids)}; '
              f'previous failure={prior.get("failure","unknown")}',flush=True)
        if getattr(args,'diagnose_only',False):
            mandated=sum(any(
                c.slots[j][0] in required
                and j in c.allowed[t]
                and sum(c.slots[x][0]==c.slots[j][0] for x in c.allowed[t])==1
                for t,required in enumerate(c.required_days))
                for j in range(c.k))
            print(f'  mandatory physical room-days: {mandated}/{c.k}',flush=True)
            for label,d in methods.items():
                historical=float(evaluate(c,c.original,d)['cost'])
                analytic=conservation_certificate(c,d)
                suffix=(f" analytic global optimum={analytic['cost']:.4f}"
                        if analytic is not None else ' conservation certificate not sufficient')
                print(f'  {label}: original planned cost={historical:.4f};{suffix}',flush=True)
            continue
        saved=results.setdefault(name,{})
        def persist():
            save_json(cache_path,cache)
        record,failure=cached_reference_for_component(
            c,rows,methods,saved,args.seconds,args.threads,on_update=persist)
        if failure:
            print(f'UNRESOLVED {name}: {failure}',flush=True)
            failed.append((name,failure))
        else:
            references[name]=record
            save_json(refs_path,references)
            print(f'CERTIFIED {name}: {len(record["references"])} reference methods',flush=True)
    if getattr(args,'diagnose_only',False):
        print('Read-only diagnostics complete; no solver calls or checkpoint writes.',flush=True)
        return True
    remaining=[k for k in all_components if not references.get(k,{}).get('complete')]
    if remaining:
        print(f'REMAINING {len(remaining)} components: '
              +', '.join(f'{k} ({references.get(k,{}).get("failure","pending")})'
                          for k in remaining[:5]),flush=True)
        return False
    print(f'All {len(all_components)} components are now certified; '
          'normal train stage may proceed.',flush=True)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('workbook',type=Path)
    p.add_argument('--artifacts',required=True,type=Path)
    p.add_argument('--gap-model',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--component',help='Optional single component key; defaults to all incomplete')
    p.add_argument('--diagnose-only',action='store_true',
                   help='Read-only structural/cost diagnostic; do not call Gurobi or modify checkpoints')
    p.add_argument('--seconds',default=1800.,type=float)
    p.add_argument('--threads',default=4,type=int)
    a=p.parse_args()
    if a.seconds<=0 or not 1<=a.threads<=15:
        p.error('Positive time limit and 1..15 threads required')
    if not rescue(a):
        raise SystemExit('Reference rescue incomplete; cached successful solves. '
                         'Inspect status and retry with a larger --seconds budget.')


if __name__=='__main__':
    main()
