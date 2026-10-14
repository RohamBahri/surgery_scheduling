"""Resumable anchored surrogate training on the frozen 2011-12 rich-feature cohort.

All tuning, anchor selection and optimization use training data only. Each
accepted update is verified with the complete deterministic deployed planner.
Incomplete oracle certificates block acceptance; unresolved separation is
recorded, never described as a proof of surrogate optimality.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .anchored_core import key
from .anchored_master import pdca, prepare_components
from .anchored_oracles import deployed, separate
from .planner import display_and_duration, reachable
from .weekly_planner import loads
from .weekly_training import save_json, asymmetric_error


def _week_task(task):
    week,methods,seconds,threads=task
    full={name:dict(zip(map(int,week.case_ids),map(float,v)))
          for name,v in methods.items()}
    rows={}
    for c in week.components():
        k=key(c)
        # An incomplete component is stored and retried individually.
        actual=deployed(c,c.actual,seconds=seconds,threads=threads,hindsight=True)
        if not actual['complete']:
            rows[k]={'complete':False,'failure':f'hindsight: {actual["status"]}'}
            continue
        references=[]
        incomplete=None
        for label,source in methods.items():
            d=np.array([full[label][int(cid)] for cid in c.case_ids],float)
            chosen=deployed(c,d,seconds=seconds,threads=threads)
            if not chosen['complete']:
                incomplete=f'{label}: {chosen["status"]}'
                break
            references.append({'source':label,'type':(
                'shared-policy' if label not in ('BOX_CLIP','BOX_PAD') else 'box-aspiration'),
                'beta':d.tolist(),'assignment':chosen['assignment'],
                'realized_cost':chosen['realized_cost'],
                'predicted_cost':chosen['predicted_cost']})
        if incomplete:
            rows[k]={'complete':False,'failure':incomplete}
        else:
            rows[k]={'complete':True,'cases':c.n,'oracle_cost':actual['realized_cost'],
                     'references':references}
    return rows


def build_reference_library(output,weeks,index,data,policy_weights,alpha,h,
                            seconds=300.,threads=1,workers=15):
    """Certify each reference by the TRUE tie-broken planner, including probes."""
    output=Path(output); path=output/'anchored_references.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    X=data['X'].astype(float);booked=data['booked'];actual=data['actual']
    lower,upper=reachable(booked,alpha,h)
    predictions={'BOOKED':booked.copy(),
                 'BOX_CLIP':np.clip(actual,lower,upper),
                 'BOX_PAD':upper}
    for name,w in policy_weights.items():
        _,d=display_and_duration(X@np.asarray(w,float),booked,alpha,h)
        predictions[name]=d
    tasks=[]
    for week in weeks:
        components=week.components()
        if all(state.get(key(c),{}).get('complete') for c in components):
            continue
        # Run by week for moderately sized process payloads; fully complete
        # components are cached even if another component in that week times out.
        row=np.asarray([index[int(cid)] for cid in week.case_ids],int)
        tasks.append((week,{name:pred[row] for name,pred in predictions.items()},
                      seconds,threads))
    print(f'Anchored references: {len(tasks)} site-weeks to resolve',flush=True)
    def persist(result):
        for k,v in result.items():
            if not state.get(k,{}).get('complete'):
                state[k]=v
        save_json(path,state)
    if workers==1:
        for task in tasks:
            persist(_week_task(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_week_task,t) for t in tasks]
            for result in as_completed(future):
                persist(result.result())
    missing=[key(c) for week in weeks for c in week.components()
             if not state.get(key(c),{}).get('complete')]
    if missing:
        raise RuntimeError(f'{len(missing)} reference components incomplete; rerun with larger --seconds; e.g. {missing[:3]}')
    return state


def locked(component,alpha,h):
    """Sufficient analytic certificate: original uses the absolute minimum
    possible room count and remains below 480 throughout the reachable box.
    Tie stages then select the zero-movement original plan for all policies.
    """
    original=component.original
    rooms=set(map(int,original))
    required_weekdays={int(day) for days in component.required_days for day in days}
    if len(rooms)!=len(required_weekdays):
        return False
    _,upper=reachable(component.booked,alpha,h)
    return bool(np.all(loads(component,original,upper)<=480.+1e-8))


def _verify_week(task):
    week,local,seconds,threads=task
    lookup=dict(zip(map(int,week.case_ids),map(float,local)))
    result={}
    for c in week.components():
        d=np.array([lookup[int(i)] for i in c.case_ids],float)
        r=deployed(c,d,seconds=seconds,threads=threads)
        result[key(c)]=r
    return result


def verify_policy(output,name,weeks,index,X,booked,w,alpha,h,
                  seconds=300.,threads=1,workers=15):
    """Full deterministic planner, append-safe cache tied to exact predictions."""
    import hashlib
    output=Path(output)
    _,duration=display_and_duration(X@np.asarray(w,float),booked,alpha,h)
    sha=hashlib.sha256(np.ascontiguousarray(duration,dtype=np.float64).tobytes()).hexdigest()
    path=output/f'anchored_verify_{name}.json'
    doc=json.loads(path.read_text()) if path.exists() else {'sha256':sha,'components':{}}
    if doc.get('sha256')!=sha:
        raise ValueError(f'Policy fingerprint collision: {path}; use a new evaluation name')
    state=doc['components'];tasks=[]
    for week in weeks:
        if all(state.get(key(c),{}).get('complete') for c in week.components()):
            continue
        rows=np.asarray([index[int(cid)] for cid in week.case_ids],int)
        tasks.append((week,duration[rows],seconds,threads))
    print(f'{name}: {len(tasks)} site-weeks need full deterministic verification',flush=True)
    def persist(result):
        for k,v in result.items():
            if not state.get(k,{}).get('complete'):
                state[k]=v
        save_json(path,doc)
    if workers==1:
        for t in tasks: persist(_verify_week(t))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_verify_week,t) for t in tasks]
            for x in as_completed(future):persist(x.result())
    pending=[key(c) for week in weeks for c in week.components()
             if not state.get(key(c),{}).get('complete')]
    if pending:
        raise RuntimeError(f'{len(pending)} deployed tie-breaks unresolved (e.g. {pending[:3]}); rerun with higher --seconds')
    return state


def _sep_week(task):
    entries,duration,gamma,seconds,threads=task
    results={}
    for b in entries:
        c=b['component'];a=b['anchor'];d=duration[b['rows']]
        r=separate(c,d,a['beta'],gamma,slopes=a['slopes'],exact=False,
                   seconds=seconds,threads=threads)
        results[b['key']]=r
    return results


def enrich_cuts(output,tag,blocks,duration,gamma,library,
                seconds=90.,threads=1,workers=15):
    """Fast underestimator separation for competitor DISCOVERY only.

    This is not certified closure of the schedule-specific tangent surrogate;
    exact-tangent separation is required for that certificate.
    """
    output=Path(output)
    path=output/f'anchored_separation_{tag}.json'
    digest=hashlib.sha256(np.ascontiguousarray(duration,dtype=np.float64).tobytes()).hexdigest()
    doc=json.loads(path.read_text()) if path.exists() else {'sha256':digest,'components':{}}
    if doc.get('sha256')!=digest:
        raise ValueError(f'Separation input fingerprint mismatch: {path}')
    checkpoint=doc['components']
    tasks=[]
    # Group by site-week for manageable job scheduling.
    groups={}
    for b in blocks:
        if checkpoint.get(b['key'],{}).get('complete'):
            continue
        c=b['component']
        groups.setdefault((c.group,c.monday),[]).append(b)
    for entries in groups.values():
        tasks.append((entries,duration,gamma,seconds,threads))
    def persist(answer):
        checkpoint.update(answer)
        save_json(path,doc)
    print(f'{tag}: {len(tasks)} group-weeks require fast competitor discovery',flush=True)
    if workers==1:
        for t in tasks:persist(_sep_week(t))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_sep_week,t) for t in tasks]
            for x in as_completed(future):persist(x.result())
    added=0
    for k,r in checkpoint.items():
        z=r.get('assignment')
        if z is None:continue
        item=library.setdefault(k,[])
        if z not in item:
            item.append(z);added+=1
    return {'added':added,'resolved':sum(bool(v.get('complete')) for v in checkpoint.values()),
            'unresolved':sum(not bool(v.get('complete')) for v in checkpoint.values())}


def learn(output,weeks,index,data,refs,initial,alpha,h,
          *,gamma=2.,lam=0.,prox=.15,coef_prox=1e-5,
          outer=5,iterations=10,closure_rounds=2,seconds=300.,
          sep_seconds=90.,master_seconds=300.,threads=1,workers=15,
          improvement=1e-4):
    """Full-cohort training with finite-budget outer search.

    Claims: fixed-library majorization and actual empirical-cost improvement
    for accepted steps. Does NOT claim optimality of original bilevel problem.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    X=data['X'].astype(float);booked=data['booked'];actual=data['actual']
    n=len(booked)
    if n!=sum(w.n for w in weeks) or X.shape[0]!=n or len(initial)!=X.shape[1]:
        raise ValueError('Missing cases or columns; full features required')
    if len(set(map(int,data['case_id'])))!=n:
        raise ValueError('Duplicate case identifiers')
    state_path=output/'anchored_progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'w':list(map(float,initial)),'accepted':[],'outer':0,
        'library':{},'complete':False}
    if state.get('complete'):
        return state
    active=[c for week in weeks for c in week.components() if not locked(c,alpha,h)]
    excluded={key(c) for week in weeks for c in week.components() if locked(c,alpha,h)}
    for k in excluded:
        if not refs[k]['complete']:
            raise ValueError('Locked component lacks reference certification')
    print(f'Active {len(active)} components; {len(excluded)} analytically locked',flush=True)
    while state['outer']<outer:
        k=state['outer'];w=np.asarray(state['w'],float)
        incumbent=verify_policy(output,f'current_{k}',weeks,index,X,booked,w,alpha,h,
                                seconds,threads,workers)
        old_total=sum(v['realized_cost'] for v in incumbent.values())
        old_regret=(old_total-sum(refs[j]['oracle_cost'] for j in incumbent))/n
        if old_regret < -.02:
            raise AssertionError('Deployed training cost below certified hindsight optimum')
        blocks=prepare_components(weeks,index,refs,state['library'])
        blocks=[b for b in blocks if b['key'] not in excluded]
        if not blocks:raise RuntimeError('All components analytically locked: no learnable schedule changes')
        proposals=[]
        candidate=w.copy()
        inner=[]
        for cut_round in range(closure_rounds):
            blocks=prepare_components(weeks,index,refs,state['library'])
            blocks=[b for b in blocks if b['key'] not in excluded]
            outcome=pdca(blocks,X,booked,candidate,alpha,h,gamma,lam,
                         iterations,prox,coef_prox,master_seconds,threads)
            candidate=np.asarray(outcome['w'],float)
            _,dur=display_and_duration(X@candidate,booked,alpha,h)
            candidate_id=hashlib.sha256(np.ascontiguousarray(dur,dtype=np.float64).tobytes()).hexdigest()[:12]
            close=enrich_cuts(output,f'{k}_{cut_round}_{candidate_id}',blocks,dur,gamma,
                              state['library'],sep_seconds,threads,workers)
            inner.append({'pdca':outcome['inner'],'fast_cut_discovery':close,
                          'restricted_objective':outcome['restricted_objective']})
            save_json(state_path,state)
            if close['added']==0 and close['unresolved']==0:
                break
        trials=[];accepted=False
        previous_library_size=sum(len(v) for v in state['library'].values())
        improved_references=0
        for j,fraction in enumerate((1.,.5,.25,.125)):
            trial=w+fraction*(candidate-w)
            trial_id=hashlib.sha256(np.ascontiguousarray(trial,dtype=np.float64).tobytes()).hexdigest()[:12]
            plan=verify_policy(output,f'trial_{k}_{j}_{trial_id}',weeks,index,X,booked,trial,
                               alpha,h,seconds,threads,workers)
            total=sum(v['realized_cost'] for v in plan.values())
            regret=(total-sum(refs[key]['oracle_cost'] for key in plan))/n
            if regret < -.02:
                raise AssertionError('Candidate plan beat certified hindsight optimum')
            _,pred=display_and_duration(X@trial,booked,alpha,h)
            feature_effect=float(np.sqrt(np.mean((X@(trial-w))**2)))
            score={'fraction':fraction,'regret_per_case':regret,'cost_per_case':total/n,
                   'realized_overtime_minutes':float(sum(v['realized_overtime_minutes'] for v in plan.values())),
                   'room_days_over_720':int(sum(v['room_days_over_720'] for v in plan.values())),
                   'max_realized_room_load':float(max(v['max_realized_room_load'] for v in plan.values())),
                   'moved_cases':int(sum(v['moved_cases'] for v in plan.values())),
                   'prediction_mae':float(np.mean(abs(actual-pred))),
                   'prediction_asymmetric_loss':float(np.mean(asymmetric_error(actual-pred))),
                   'non_intercept_coefficients_changed':int(np.sum(abs(trial[1:]-w[1:])>1e-7)),
                   'rms_recommendation_change':feature_effect,
                   'accepted':bool(regret<old_regret-improvement)}
            trials.append(score)
            # A deployed schedule and its durations provide a genuine shared-policy
            # witness, even if the policy is rejected in aggregate. Refresh the
            # reference pool without using validation or hindsight-derived advice.
            new_refs=0
            for b in blocks:
                component=b['component'];ck=b['key']
                chosen=plan[ck]
                zz=chosen['assignment'];item=state['library'].setdefault(ck,[])
                if zz not in item:item.append(zz)
                existing=refs[ck]['references']
                if (chosen['realized_cost'] < min(r['realized_cost'] for r in existing)-1e-7):
                    existing.append({'source':f'TRAIN_CANDIDATE_{k}_{j}',
                         'type':'shared-policy','beta':pred[b['rows']].tolist(),
                         'assignment':zz,'realized_cost':chosen['realized_cost'],
                         'predicted_cost':chosen['predicted_cost']})
                    new_refs+=1
            improved_references+=new_refs
            if new_refs:
                save_json(output/'anchored_references.json',refs)
            if score['accepted']:
                state['w']=trial.tolist();accepted=True
                state['accepted'].append({'outer':k,'before':old_regret,'after':regret,
                                          'fraction':fraction})
                break
        state.setdefault('history',[]).append({'outer':k+1,'before':old_regret,
                            'inner':inner,'candidates':trials,'accepted':accepted})
        state['outer']=k+1
        save_json(state_path,state)
        if not accepted:
            gained=(sum(len(v) for v in state['library'].values())
                    >previous_library_size or improved_references>0)
            if not gained:
                state['stopping']='no_verified_improvement_at_tested_steps'
                break
            print('No accepted policy; new competitor/reference information '
                  'permits another outer attempt from the unchanged incumbent',flush=True)
    else:
        state['stopping']='outer_budget'
    state['complete']=True
    state['method']='ANCHORED_REGRET_DC'
    state['feature_dimensions']=int(X.shape[1])
    state['training_cases']=int(n)
    state['training_site_weeks']=len(weeks)
    state['model_claim']='monotone verified empirical regret, not global bilevel optimality'
    save_json(state_path,state)
    return state
