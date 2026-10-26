#!/usr/bin/env python3
"""Evaluate intrinsic-domain parametric VF calibration (B .8/30 by default).

Runs in an existing surgery_scheduling repository checkout. Requires the
previous response_saturation_grid (unchanged) and licensed Gurobi for larger
components. Reuses original training policies, feature scaler and cached
planner solves. Checkpoints every component into cpro_<scenario>/components.

WARNING: EXACT_ENUMERATION is genuinely rationally exact for small instances.
NUMERIC_PARTITION means Gurobi-tolerance-dependent verification, NOT proof
under exact MILP oracles. UNRESOLVED is never reported as certified.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction as Q
from multiprocessing import get_context
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from surgery.parametric_ray import RationalComponent,saturation_endpoint
from surgery.parametric_ray_mip import discover_component,NumericOracle
from surgery.response_grid import TRAIN_SCENARIOS, mapping,verify_same_scenario,implemented_from_display
from surgery.weekly_planner import evaluate as schedule_evaluate
from pilot import run_response_scenario_grid as base
from pilot import extend_response_grid_evaluation as oldC


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    def normalize(v):
        if isinstance(v,Q):return str(v)
        if isinstance(v,dict):return {str(k):normalize(val) for k,val in v.items()}
        if isinstance(v,(list,tuple)):return [normalize(x) for x in v]
        if isinstance(v,np.integer):return int(v)
        if isinstance(v,np.floating):return float(v)
        return v
    tmp.write_text(json.dumps(normalize(value),sort_keys=True,indent=2))
    tmp.replace(path)


def worker(job):
    """No shared Gurobi model. Each component is independently checkpointable."""
    (key,week,raw,projection,alpha,h,upper,initial,
       seconds,solve_seconds,mu,max_rounds,max_cells,max_schedules,
       exact_limit,boundary_seconds,verify_boundaries)=job
    c=RationalComponent.from_week(week,raw,projection,alpha,h,Q(upper))
    started=time.monotonic()
    try:
        answer=discover_component(
            c,week,initial,seconds=seconds,
            separation_seconds=solve_seconds,mu=mu,
            max_rounds=max_rounds,max_cells=max_cells,
            max_schedules=max_schedules,exhaustive_limit=exact_limit,
            boundary_seconds=boundary_seconds,
            verify_boundaries=verify_boundaries)
    except Exception as ex:
        answer=dict(status='UNRESOLVED',reason='EXCEPTION',
                    error=f'{type(ex).__name__}: {ex}')
    if answer.get('status')=='UNRESOLVED' and initial:
        # A sound relaxed-hindsight *mathematical* formulation; float dual
        # bounds are labeled numerical, and 0 is always a rigorous fallback.
        try:
            oracle=NumericOracle(c,week,seconds=min(solve_seconds,7.),mu=mu)
            answer['full_ray_relaxation']=oracle.interval_lower_bound(
                Q(0),c.upper,initial)
        except Exception as ex:
            answer['full_ray_relaxation']=dict(
                unconditional_lower_bound=0.,error=str(ex))
    answer['component']=str(key)
    answer['n_cases']=week.n
    answer['n_slots']=week.k
    answer['elapsed_seconds']=round(time.monotonic()-started,3)
    return str(key),answer


def to_atoms(report,upper):
    """Normalize both exact and numerically checked partition structures."""
    if report['status']=='EXACT_ENUMERATION':
        atoms=report['atoms']
        points={Q(a['s']):Q(a['value']) for a in atoms if a['type']=='point'}
        opens={(Q(a['left']),Q(a['right'])):Q(a['value'])
               for a in atoms if a['type']=='open'}
    elif report['status']=='NUMERIC_PARTITION':
        points={Q(a['s']):Q(a['real']) for a in report['boundaries']}
        opens={(Q(a['a']),Q(a['b'])):Q(a['real']) for a in report['cells']}
    else:raise ValueError('Component is not completely represented')
    cuts=sorted(points)
    if cuts[0]!=0 or cuts[-1]!=upper or len(opens)!=len(cuts)-1:
        raise ValueError('Incomplete continuous ray coverage')
    for a,b in zip(cuts,cuts[1:]):
        if (a,b) not in opens:raise ValueError('Missing open interval')
    return points,opens


def merge_global(component_reports,upper):
    """Exact sweep over combined rational interval and boundary atoms.

    This computes the optimum of the *reported* step functions. It is a
    globally certified optimum only if every component is EXACT_ENUMERATION.
    """
    components=list(component_reports.values())
    data=[to_atoms(r,upper) for r in components]
    changes={}
    point_zero=Q(0)
    open_zero=Q(0)
    for pts,opens in data:
        cuts=sorted(pts)
        point_zero+=pts[Q(0)]
        if len(cuts)>1:open_zero+=opens[(cuts[0],cuts[1])]
        for j,s in enumerate(cuts[1:],start=1):
            left=opens[(cuts[j-1],cuts[j])]
            if j+1<len(cuts):right=opens[(cuts[j],cuts[j+1])]
            else:right=None
            row=changes.setdefault(s,dict(point_delta=Q(0),open_delta=Q(0)))
            row['point_delta']+=pts[s]-left
            if right is not None:
                row['open_delta']+=right-left
    best=(point_zero,Q(0),'boundary')
    current=open_zero
    prev=Q(0)
    for s in sorted(changes):
        if s>prev and current<best[0]:
            best=(current,(prev+s)/2,'stable_open')
        event=changes[s]
        value=current+event['point_delta']
        if value<best[0]:best=(value,s,'boundary')
        current+=event['open_delta']
        prev=s
    exact=all(r['status']=='EXACT_ENUMERATION' for r in components)
    return dict(
        best_total_cost=best[0],best_s=best[1],
        best_kind=best[2],exact_arithmetic_certificate=exact,
        interpretation='EXACT_CLOSED_RAY' if exact else
                      'NUMERIC_PARTITION_CANDIDATE_NOT_EXACT',
        component_count=len(components),boundary_event_count=len(changes),
        upper=upper)


def prepare(ctx,root,args,scenario):
    """Do not retrain a policy or use evaluation data for ray selection."""
    policy_set=json.loads((root/'selected_policies.json').read_text())
    policy=policy_set['policies'][scenario.name]
    if policy['scenario']!=vars(scenario):
        raise ValueError('Original training scenario does not match')
    cache=base.ContentCache(root,args.cores,args.plan_seconds)
    data=ctx['fit']
    b=np.asarray(data['booked'],float)
    raw=np.asarray(data['X'],float)@np.asarray(policy['case_coeff'],float)
    model=mapping(b,scenario)
    verify_same_scenario(raw,b,scenario)
    params=json.loads((root/'training'/scenario.name/'vf_features.json').read_text())
    H,_,_=base.five_features(ctx,ctx['train'],raw,model,cache,
        'train_'+scenario.name,
        scaler=(np.asarray(params['mean'],float),np.asarray(params['std'],float)))
    p=np.asarray(H,float)@np.asarray(policy['vf_theta'],float)
    upper=saturation_endpoint(raw,p,scenario.h)
    if float(upper)!=float('inf') and np.isfinite(float(upper)):
        print(f'Intrinsic plateau S={float(upper):.10g}; '
              f'no arbitrary coefficient cap',flush=True)
    else:
        raise OverflowError('Intrinsic S is not finite as a float: '
                            'exact domain retained, numerical solver cannot use it')
    return raw,p,upper,model,cache,policy


def verify_existing(root,args):
    # Pin to the original scientific code/parameters and fitted policies.
    oldC.validate_original_run(args,root)
    if not (root/'selected_policies.json').exists():
        raise FileNotFoundError('Original CASE/VF training policies missing')


def run(args):
    scenario=next((s for s in TRAIN_SCENARIOS if s.name==args.train_scenario),None)
    if scenario is None:raise ValueError('Unknown B training scenario')
    root=args.out
    verify_existing(root,args)
    out=root/('cpro_'+scenario.name)
    out.mkdir(parents=True,exist_ok=True)
    contract=dict(source_sha256=digest(__file__),
         rational_source_sha256=digest(Path(__file__).resolve().parents[1]/'surgery'/'parametric_ray.py'),
         mip_source_sha256=digest(Path(__file__).resolve().parents[1]/'surgery'/'parametric_ray_mip.py'),
         context_sha256=digest(args.context),
         original_contract_sha256=digest(root/'run_contract.json'),
         original_policies_sha256=digest(root/'selected_policies.json'),
         scenario=vars(scenario),intrinsic_domain=True,
         numerical_mu=args.mu,per_component_seconds=args.component_seconds,
         separation_seconds=args.separation_seconds,boundary_seconds=args.boundary_seconds,
         max_rounds=args.max_rounds,max_cells=args.max_cells,
         max_schedules=args.max_schedules,exact_limit=args.exact_limit,
         verify_boundaries=not args.skip_boundary_verification)
    file=out/'run_contract.json'
    if file.exists() and json.loads(file.read_text())!=contract:
        raise ValueError('Changed numerical/scientific specification; use a distinct output root')
    if not file.exists():atomic_json(file,contract)
    with args.context.open('rb') as handle:ctx=pickle.load(handle)
    print(f'Train components={len(ctx["train"])}',flush=True)
    raw,p,upper,model,cache,policy=prepare(ctx,root,args,scenario)
    domain=dict(S=upper,S_float=float(upper),vf_theta=policy['vf_theta'],
        case_raw_sha256=hashlib.sha256(np.ascontiguousarray(
            raw,dtype='<f8').tobytes()).hexdigest(),
        projection_sha256=hashlib.sha256(np.ascontiguousarray(
            p,dtype='<f8').tobytes()).hexdigest(),
        cases=len(raw),components=len(ctx['train']))
    domain_file=out/'intrinsic_domain.json'
    if domain_file.exists() and json.loads(domain_file.read_text())!=json.loads(
          json.dumps({k:str(v) if isinstance(v,Q) else v
                      for k,v in domain.items()},sort_keys=True)):
        raise ValueError('Intrinsic domain or realized float features changed')
    if not domain_file.exists():atomic_json(domain_file,domain)
    if args.dry_run:
        print('Dry run complete; no component MIPs solved',flush=True)
        return
    items=ctx['train'] if not args.pilot_components else ctx['train'][:args.pilot_components]
    used_rows=[int(i) for _,_,rows in items for i in rows]
    ids={key for key,_,_ in items}
    print(f'Processing {len(items)} components; original training scenarios unchanged',flush=True)
    if len(items)!=len(ids):raise AssertionError('Duplicate component names')
    booked=cache.solve(items,np.asarray(ctx['fit']['booked'],float),
                       'train_booked',tie=True)
    case=cache.solve(items,model.reviewed(raw),
                     'train_'+scenario.name+'_case_draft',tie=True)
    vf=cache.solve(items,model.reviewed(raw+p),
                   'cpro_'+scenario.name+'_plain_vf',tie=True)
    folder=out/'components'
    folder.mkdir(parents=True,exist_ok=True)
    import hashlib
    def path_for(key):
        return folder/(hashlib.sha256(str(key).encode()).hexdigest()+'.json')
    results={}
    tasks=[]
    for key,week,rows in items:
        path=path_for(key)
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('component')!=str(key):raise ValueError('Stale component cache')
            if saved.get('status')!='UNRESOLVED':
                results[str(key)]=saved
                continue
            print(f'{key}: retrying unresolved component checkpoint',flush=True)
        seeds=[]
        for bank in (booked,case,vf):
            if bank[key].get('assignment') is not None:
                seeds.append(tuple(bank[key]['assignment']))
        tasks.append((str(key),week,raw[rows],p[rows],scenario.alpha,
                 scenario.h,str(upper),seeds,
                 args.component_seconds,args.separation_seconds,args.mu,
                 args.max_rounds,args.max_cells,args.max_schedules,
                 args.exact_limit,args.boundary_seconds,
                 not args.skip_boundary_verification))
    print(f'{len(results)} cached components; {len(tasks)} new',flush=True)
    start=time.monotonic()
    if args.cores==1:
        outputs=(worker(t) for t in tasks)
        for i,(key,res) in enumerate(outputs,1):
            atomic_json(path_for(key),res)
            results[key]=json.loads(path_for(key).read_text())
            if i%25==0 or i==len(tasks):
                print(f'{i}/{len(tasks)} {res["status"]}',flush=True)
            if time.monotonic()-start>args.wall_minutes*60:
                print('Global wall budget exhausted: checkpointed partial run',flush=True)
                break
    elif tasks:
        # Submit bounded batches; never leave thousands of queued processes
        # running after the global wall budget has been exceeded.
        with ProcessPoolExecutor(max_workers=args.cores,
                 mp_context=get_context('spawn')) as pool:
            for off in range(0,len(tasks),max(args.cores,1)):
                if time.monotonic()-start>args.wall_minutes*60:break
                futures=[pool.submit(worker,task)
                         for task in tasks[off:off+args.cores]]
                for f in as_completed(futures):
                    key,res=f.result()
                    atomic_json(path_for(key),res)
                    results[key]=json.loads(path_for(key).read_text())
                print(f'Checkpointed {len(results)}/{len(items)} components',flush=True)
    counts={}
    for res in results.values():
        counts[res['status']]=counts.get(res['status'],0)+1
    # Globally valid but potentially very weak lower bound: exact minima for
    # exhaustively enumerated independent components plus the universal 0
    # (nonnegative OR cost) for every other component. The optional Gurobi
    # interval dual is informative, but NEVER substituted without verification.
    unconditional_lb=sum((Q(r['value']) if r['status']=='EXACT_ENUMERATION'
         else Q(0)) for r in results.values())
    report=dict(scenario=scenario.name,S=upper,counts=counts,
                total_components=len(items),completed_components=len(results),
                elapsed_seconds=time.monotonic()-start,
                full_training=(len(items)==len(ctx['train'])),
                unconditional_global_lower_bound=unconditional_lb,
                numerical_standard='Floating Gurobi results are NOT rational global certificates')
    if all(bank[key].get('complete') for bank in (case,vf)
           for key,week,rows in items):
        direct={arm:sum(schedule_evaluate(week,bank[key]['assignment'],
                       week.actual)['cost'] for key,week,rows in items)
                for arm,bank in [('CASE',case),('VF',vf)]}
        report['training_endpoints_numerically_certified']=direct
        report['endpoint_best_cost']=min(direct.values())
    complete=bool(len(results)==len(items) and all(
          r['status'] in ('EXACT_ENUMERATION','NUMERIC_PARTITION')
          for r in results.values()))
    if complete:
        candidate=merge_global(results,upper)
        report['partition_minimizer']=candidate
        if len(items)==len(ctx['train']):
            step=float(candidate['best_s'])
            durations=model.reviewed(raw+step*p)
            plans=cache.solve(ctx['train'],durations,
                     f'cpro_{scenario.name}_candidate',tie=True)
            if all(rec['complete'] for rec in plans.values()):
                actual_total=sum(schedule_evaluate(week,
                    plans[key]['assignment'],week.actual)['cost']
                    for key,week,rows in ctx['train'])
                report['certified_float_deployment_cost']=actual_total
                report['candidate_cost_agreement']=abs(
                    actual_total-float(candidate['best_total_cost']))<1e-5
                report['deployment_all_certified']=True
                report['can_evaluate_2013']=report['candidate_cost_agreement']
                if not report['candidate_cost_agreement']:
                    report['status']='UNRESOLVED_GLOBAL_NUMERIC_MISMATCH'
                else:
                    report['status']=candidate['interpretation']
            else:
                report['status']='UNRESOLVED_DEPLOYMENT_AT_SELECTED_STEP'
        else:
            report['status']='PILOT_ONLY_NOT_FULL_TRAINING'
    else:
        report['status']='UNRESOLVED_COMPONENTS_OR_WALL_BUDGET'
    atomic_json(out/'summary.json',report)
    print('COMPLETE:',report['status'],counts,flush=True)
    if args.evaluate and report.get('can_evaluate_2013',False):
        evaluate(ctx,root,out,scenario,policy,cache,float(report['partition_minimizer']['best_s']))
    elif args.evaluate:
        print('Cannot perform defensible evaluation: missing full training '
              'deployment certificate or candidate inconsistency',flush=True)


def evaluate(ctx,root,out,train,policy,cache,step):
    """Replay ONE frozen B recommendation through all original B/C scenarios."""
    import csv
    held=ctx['held']
    b=np.asarray(held['booked'],float)
    raw=np.asarray(held['X'],float)@np.asarray(policy['case_coeff'],float)
    m=mapping(b,train)
    scaler=json.loads((root/'training'/train.name/'vf_features.json').read_text())
    H,_,_=base.five_features(ctx,ctx['eval'],raw,m,cache,'eval_'+train.name,
        scaler=(np.asarray(scaler['mean'],float),np.asarray(scaler['std'],float)))
    # Explicit parentheses preserve the VF ray score exactly.
    display=verify_same_scenario(raw+step*(np.asarray(H)@
                  np.asarray(policy['vf_theta'])),b,train)
    name='CPRO__'+train.name
    path=out/'frozen_display.csv'
    if not path.exists():
        with path.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['case_id',name])
            for id_,u in zip(held['case_id'],display):
                writer.writerow([int(id_),float(u)])
    records=[]
    for scenario in oldC.ALL_SCENARIOS:
        revised=implemented_from_display(b,display,scenario)
        plans=cache.solve(ctx['eval'],revised,
            f'cpro_{train.name}_eval_{scenario.name}',tie=True)
        for key,week,rows in ctx['eval']:
            rec=plans[key]
            if rec.get('assignment') is None:continue
            metrics=schedule_evaluate(week,rec['assignment'],week.actual)
            records.append(dict(scenario=scenario.name,component=str(key),
                week=week.monday,cases=week.n,site=week.group,
                cost=float(metrics['cost']),complete=bool(rec['complete']),
                mae_sum=float(np.abs(week.actual-revised[rows]).sum())))
    atomic_json(out/'evaluation_component_records.json',records)
    summarize_evaluation(ctx,root,out,train,records)
    print('CPRO exploratory B/C replay saved',flush=True)



def summarize_evaluation(ctx,root,out,train,new_records):
    """Paired original CASE/VF and old ray on one C/B certified cohort."""
    import csv
    new={(r['scenario'],r['component']):r for r in new_records}
    methods=['BOOKED','NAIVE','CASE__'+train.name,'VF__'+train.name]
    if (root/'vf_ray_extension'/'evaluation_cells').exists():
        methods.append('VF_RAY__'+train.name)
    sources={}
    for sc in oldC.ALL_SCENARIOS:
        for method in methods:
            if method.startswith('VF_RAY__'):
                file=root/'vf_ray_extension'/'evaluation_cells'/sc.name/(method+'.json')
            elif sc.link=='saturating':
                file=root/'evaluation_cells'/sc.name/(method+'.json')
            else:
                file=root/'response_extension'/'evaluation_cells'/sc.name/(method+'.json')
            if not file.exists():
                raise FileNotFoundError('Comparison cell missing: '+str(file))
            entries=oldC.check_cell(file,sc,method,ctx['eval'],
                  extension=(sc.link=='nonmonotone' or method.startswith('VF_RAY__')))
            sources[sc.name,method]={str(r['component']):r for r in entries}
    keys={str(key) for key,week,rows in ctx['eval']}
    n={str(key):week.n for key,week,rows in ctx['eval']}
    week={str(key):week.monday for key,week,rows in ctx['eval']}
    common={key for key in keys if all(
        (sc.name,key) in new and new[sc.name,key]['complete']
        and all(sources[sc.name,m][key]['complete'] for m in methods)
        for sc in oldC.ALL_SCENARIOS)}
    if not common:
        raise RuntimeError('No globally common certified evaluation components')
    cases=sum(n[key] for key in common)
    import numpy as np
    all_weeks=sorted({week[key] for key in common})
    week_cases=np.array([sum(n[key] for key in common if week[key]==w)
                         for w in all_weeks],float)
    rng=np.random.default_rng(20261026)
    draws=rng.integers(len(all_weeks),size=(10000,len(all_weeks)))
    base_n=week_cases[draws].sum(axis=1)
    rows=[]
    result={}
    for sc in oldC.ALL_SCENARIOS:
        cpro=sum(new[sc.name,key]['cost'] for key in common)/cases
        cost={m:sum(sources[sc.name,m][key]['cost'] for key in common)/cases
              for m in methods}
        contrasts={}
        for method in methods:
            delta=np.array([sum(new[sc.name,key]['cost']-
                         sources[sc.name,method][key]['cost']
                         for key in common if week[key]==w)
                         for w in all_weeks])
            boot=delta[draws].sum(axis=1)/base_n
            contrasts['CPRO_minus_'+method]=dict(
                difference_per_case=float(delta.sum()/cases),
                bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
        result[sc.name]=dict(link=sc.link,CPRO_cost_per_case=cpro,
                             baselines_cost_per_case=cost,contrasts=contrasts)
        rows.append([sc.name,sc.link,cases,cpro,*[cost[m] for m in methods]])
    with (out/'evaluation_cost_grid_global_matched.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['scenario','link','globally_matched_cases','CPRO',*methods])
        writer.writerows(rows)
    atomic_json(out/'evaluation_summary.json',dict(
        status='EXPLORATORY_REUSED_2013',
        original_B_training_scenario=train.name,
        common_certified_components=len(common),
        common_certified_cases=cases,methods=['CPRO',*methods],
        scenarios=result,
        caution='If CPRO was NUMERIC_PARTITION rather than EXACT_ENUMERATION, '
               'this evaluation is not an exact global-ray certificate.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--context',type=Path,default=Path(
        'artifacts/objective_selection_12h/objective_selection_2013/context.pkl'))
    ap.add_argument('--out',type=Path,default=Path('artifacts/response_saturation_grid'))
    ap.add_argument('--train-scenario',default='SAT_A80_H30')
    ap.add_argument('--cores',type=int,default=15)
    ap.add_argument('--plan-seconds',type=float,default=180.)
    ap.add_argument('--case-steps',type=int,default=200)
    ap.add_argument('--vf-outers',type=int,default=30)
    ap.add_argument('--vf-pdca-steps',type=int,default=50)
    ap.add_argument('--cuts',type=int,default=160)
    ap.add_argument('--component-seconds',type=float,default=90.)
    ap.add_argument('--separation-seconds',type=float,default=7.)
    ap.add_argument('--boundary-seconds',type=float,default=30.)
    ap.add_argument('--wall-minutes',type=float,default=120.)
    ap.add_argument('--mu',type=float,default=1e-6)
    ap.add_argument('--max-rounds',type=int,default=25)
    ap.add_argument('--max-cells',type=int,default=500)
    ap.add_argument('--max-schedules',type=int,default=120)
    ap.add_argument('--exact-limit',type=int,default=64)
    ap.add_argument('--pilot-components',type=int,default=0,
                    help='First N training components only, for resource audits')
    ap.add_argument('--skip-boundary-verification',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--evaluate',action='store_true')
    args=ap.parse_args()
    if (not 1<=args.cores<=15 or args.wall_minutes<=0 or args.component_seconds<=0
        or args.separation_seconds<=0 or args.mu<0):
        ap.error('Invalid wall-time or numerical settings')
    run(args)

if __name__=='__main__':main()
