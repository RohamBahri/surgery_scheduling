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

from .anchored_core import (key, anchored_cut, anchored_score,
                            normalize_tangent, tangent_key, reference_fingerprint)
from .anchored_master import pdca, prepare_components
from .anchored_oracles import deployed, separate
from .planner import display_and_duration, reachable
from .weekly_planner import evaluate
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
                # Reference schedules are feasible *aspirations*, not claimed
                # deployed baselines. A time-limited optional MILP may supply
                # its feasible incumbent; otherwise the original assignment
                # is feasible by construction. Hindsight above MUST be exact.
                # An optional reference needs only a feasible aspiration.
                # Even an unfinished PRIMARY/HASH/UNIQUENESS solve may supply
                # an incumbent; the original assignment is otherwise feasible.
                z=chosen.get('assignment')
                if z is None:
                    z=list(map(int,c.original))
                objective_certified=False
                kind='uncertified-feasible-aspiration'
            else:
                z=chosen['assignment']
                objective_certified=True
                kind=('box-aspiration' if label in ('BOX_CLIP','BOX_PAD')
                      else 'shared-policy')
            # Independent feasibility and cost check for every proposed
            # reference, including an incomplete solver incumbent.
            pred=evaluate(c,z,d)
            real=evaluate(c,z,c.actual)
            references.append({'source':label,'type':kind,
                'beta':d.tolist(),'assignment':list(map(int,z)),
                'realized_cost':real['cost'],'predicted_cost':pred['cost'],
                'reference_optimality_certified':objective_certified,
                'solver_status':chosen.get('status')})
        if incomplete:
            rows[k]={'complete':False,'failure':incomplete}
        else:
            rows[k]={'complete':True,'cases':c.n,'oracle_cost':actual['realized_cost'],
                     'references':references}
    return rows


def build_reference_library(output,weeks,index,data,policy_weights,alpha,h,
                            seconds=300.,threads=1,workers=15):
    """Exact hindsight plus feasible, explicitly labeled schedule aspirations."""
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
        raise RuntimeError(f'{len(missing)} reference components lack an EXACT hindsight or a valid feasible proposal; e.g. {missing[:3]}')
    return state


def unique_schedule(component):
    """Prove that surgeon-day retention leaves exactly one feasible assignment.

    Cases of a surgeon share the same eligible slots. With multiple cases and
    multiple allowed slots, at least one reassignment is possible unless the
    surgeon has a single slot. For a single case the required day can force a
    unique slot even if other days are on the eligible roster.
    """
    for s,allowed in enumerate(component.allowed):
        if len(allowed)==1:
            continue
        count=int(np.count_nonzero(component.surgeon==s))
        days=component.required_days[s]
        if count!=1 or len(days)!=1:
            return False
        if sum(int(component.slots[j][0]==days[0]) for j in allowed)!=1:
            return False
    return True


def locked(component,alpha,h):
    """Only a unique feasible schedule is policy-independent under hash ties.

    The old sufficient-load test relied on preferring performed room-days;
    that preference is no longer part of deployment.
    """
    return unique_schedule(component)


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
                  seconds=300.,threads=1,workers=15,*,allow_incomplete=False):
    """Full deterministic planner, append-safe cache tied to exact predictions."""
    import hashlib
    output=Path(output)
    _,duration=display_and_duration(X@np.asarray(w,float),booked,alpha,h)
    sha=hashlib.sha256(np.ascontiguousarray(duration,dtype=np.float64).tobytes()).hexdigest()
    path=output/f'anchored_verify_{sha}.json'
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
    if pending and not allow_incomplete:
        details={k:state.get(k,{}).get('status') for k in pending[:5]}
        raise RuntimeError(f'{len(pending)} exact deployed plans unresolved '
                           f'(examples {details}); retry with more --seconds')
    return state


def _sep_week(task):
    """Solve exact schedule-specific tangent maximization on chosen components."""
    if len(task)==5:
        entries,duration,gamma,seconds,threads=task
        tight_surrogate=False
    else:
        entries,duration,gamma,seconds,threads,tight_surrogate=task
    results={}
    for b in entries:
        c=b['component'];anchor=b['anchor']
        r=separate(c,duration[b['rows']],anchor['beta'],gamma,exact=True,
                   seconds=seconds,threads=threads,
                   min_realized_cost=(evaluate(c,anchor['assignment'],c.actual)['cost']
                                      if tight_surrogate else None))
        results[b['key']]=r
    return results


def _separation_priority(block, current_duration, proposal_duration, gamma):
    """Prioritize apparent reductions in the restricted anchored loss."""
    c=block['component'];rows=block['rows']
    anchor=block['anchor'];cuts=block['cuts'];oracle=block['oracle_cost']
    before=anchored_score(c,anchor,cuts,current_duration[rows],oracle,gamma,exact=True)
    after=anchored_score(c,anchor,cuts,proposal_duration[rows],oracle,gamma,exact=True)
    # Components with large apparent improvement are most vulnerable to
    # missing competitor cuts. Also break ties by local duration movement.
    shift=float(np.linalg.norm(proposal_duration[rows]-current_duration[rows]))
    return (max(0.,before-after),shift)


def enrich_cuts(output,tag,blocks,duration,gamma,library,
                seconds=90.,threads=1,workers=15,*,prior_duration=None,
                batch=96,round_id=0,tight_surrogate=False):
    """Adaptive exact-tangent cut generation for the *training* loss.

    An exact Gurobi maximization is evaluated on a priority batch, not all
    components after every pDCA step. The returned primal tangent is added
    even on time-limit; a finite dual bound is used to certify closure of
    inspected components. Untested and unresolved components are counted.
    No global separation-closure claim is made unless every component is
    independently certified.
    """
    output=Path(output)
    duration=np.asarray(duration,float)
    prior=np.asarray(prior_duration if prior_duration is not None else duration,float)
    if batch<0:
        raise ValueError('Negative exact-separation batch size')
    components={b['key']:b for b in blocks}
    ordered=sorted(blocks,key=lambda b:(-_separation_priority(
        b,prior,duration,gamma)[0],-_separation_priority(
        b,prior,duration,gamma)[1],b['key']))
    if batch==0 or batch>=len(ordered):
        selected=ordered
    else:
        focus=max(1,(batch*2)//3)
        chosen=ordered[:focus]
        remaining=ordered[focus:]
        available=batch-len(chosen)
        offset=(round_id*available)%max(1,len(remaining))
        for k in range(min(available,len(remaining))):
            chosen.append(remaining[(offset+k)%len(remaining)])
        selected=chosen
    selected_keys={b['key'] for b in selected}
    path=output/f'anchored_exact_separation_{tag}.json'
    digest=hashlib.sha256()
    digest.update(np.ascontiguousarray(duration,dtype=np.float64).tobytes())
    digest.update(b'TIGHT' if tight_surrogate else b'ORIGINAL')
    for b in sorted(selected,key=lambda b:b['key']):
        digest.update(b['key'].encode('utf8'))
        digest.update(reference_fingerprint(b['anchor']['beta']).encode('ascii'))
    fingerprint=digest.hexdigest()
    doc=json.loads(path.read_text()) if path.exists() else {
        'sha256':fingerprint,'components':{},'exact_tangent':True}
    if doc.get('sha256')!=fingerprint:
        raise ValueError(f'Separation point/reference fingerprint mismatch: {path}')
    checkpoint=doc['components'];groups={}
    for b in selected:
        if checkpoint.get(b['key'],{}).get('complete'):
            continue
        c=b['component']
        groups.setdefault((c.group,c.monday),[]).append(b)
    tasks=[(items,duration,gamma,seconds,threads,tight_surrogate) for items in groups.values()]
    def persist(answer):
        checkpoint.update(answer)
        save_json(path,doc)
    print(f'{tag}: exact tangent separation for {len(selected)} of '
          f'{len(blocks)} components; {len(tasks)} site-weeks to solve',flush=True)
    if workers==1:
        for task in tasks:persist(_sep_week(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=mp.get_context('spawn')) as pool:
            future=[pool.submit(_sep_week,t) for t in tasks]
            for x in as_completed(future):persist(x.result())
    added=0;closed=0;unresolved=0;max_gap=0.
    for b in selected:
        k=b['key'];c=b['component'];r=checkpoint.get(k,{})
        known=list(b['cuts'])
        item=library.setdefault(k,[])
        if r.get('assignment') is not None and r.get('slopes') is not None:
            cut=normalize_tangent(c,dict(
                assignment=r['assignment'],slopes=r['slopes'],
                anchor_sha256=reference_fingerprint(b['anchor']['beta'])
            ),b['anchor']['beta'])
            signatures={tangent_key(normalize_tangent(c,x,b['anchor']['beta']))
                        for x in item}
            if tangent_key(cut) not in signatures:
                item.append(cut);added+=1
            known.append(cut)
        upper=r.get('upper')
        if upper is None or not np.isfinite(upper):
            unresolved+=1
            continue
        d=duration[b['rows']]
        restricted=max(anchored_cut(c,b['anchor'],z,d,gamma,exact=True)
                       for z in known)
        gap=max(0.,float(upper)-restricted)
        max_gap=max(max_gap,gap)
        if gap<=1e-4:
            closed+=1
        else:
            unresolved+=1
    return {'added':added,'closed':closed,'unresolved':unresolved,
            'tested':len(selected),'untested':len(blocks)-len(selected),
            'complete_oracles':sum(bool(checkpoint.get(k,{}).get('complete'))
                                   for k in selected_keys),
            'max_dual_cut_gap_tested':max_gap,
            'all_closed':bool(closed==len(blocks) and not unresolved)}




def learn(output,weeks,index,data,refs,initial,alpha,h,
          *,gamma=2.,lam=0.,prox=.15,coef_prox=1e-5,
          outer=5,iterations=10,closure_rounds=2,seconds=300.,
          sep_seconds=90.,master_seconds=300.,threads=1,workers=15,
           master_threads=None,
          improvement=1e-4,sep_batch=96,ridge=0.,reference=0.,
          reference_w=None,line_fractions=(1.,.5,.25,.125),
          terminal_separation=False,max_terminal_restarts=1,trust_growth=1.,
          tight_surrogate=False):
    """Full-cohort training with finite-budget outer search.

    Claims: fixed-library majorization and actual empirical-cost improvement
    for accepted steps. Does NOT claim optimality of original bilevel problem.
    """
    if (ridge<0 or reference<0 or trust_growth<1. or
            not line_fractions or any(x<=0 for x in line_fractions) or
            max_terminal_restarts<0):
        raise ValueError('Invalid regularization or line-search configuration')
    if not isinstance(sep_batch, int) or sep_batch < 0:
        raise ValueError('sep_batch must be a nonnegative integer (0 means all)')
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
        'library':{},'complete':False,'initial_w':list(map(float,initial)),
        'terminal_restarts':0,'trust_scale':1.}
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
        size_before_round=sum(len(v) for v in state['library'].values())
        incumbent=verify_policy(output,f'current_{k}',weeks,index,X,booked,w,alpha,h,
                                seconds,threads,workers)
        old_total=sum(v['realized_cost'] for v in incumbent.values())
        old_regret=(old_total-sum(refs[j]['oracle_cost'] for j in incumbent))/n
        if old_regret < -.02:
            raise AssertionError('Deployed training cost below certified hindsight optimum')
        _,current_duration=display_and_duration(X@w,booked,alpha,h)
        blocks=prepare_components(weeks,index,refs,state['library'],
                                  current=incumbent,current_duration=current_duration,
                                  tight_surrogate=tight_surrogate)
        blocks=[b for b in blocks if b['key'] not in excluded]
        if not blocks:raise RuntimeError('All components analytically locked: no learnable schedule changes')
        proposals=[]
        candidate=w.copy()
        inner=[]
        for cut_round in range(closure_rounds):
            blocks=prepare_components(weeks,index,refs,state['library'],
                                      current=incumbent,current_duration=current_duration,
                                      tight_surrogate=tight_surrogate)
            blocks=[b for b in blocks if b['key'] not in excluded]
            outcome=pdca(blocks,X,booked,candidate,alpha,h,gamma,lam,
                         iterations,prox,coef_prox,master_seconds,
                          master_threads if master_threads is not None else threads,
                          ridge=ridge,reference=reference,reference_w=reference_w)
            candidate=np.asarray(outcome['w'],float)
            _,dur=display_and_duration(X@candidate,booked,alpha,h)
            candidate_id=hashlib.sha256(np.ascontiguousarray(dur,dtype=np.float64).tobytes()).hexdigest()[:12]
            close=enrich_cuts(output,f'{k}_{cut_round}_{candidate_id}',blocks,dur,gamma,
                              state['library'],sep_seconds,threads,workers,
                              prior_duration=current_duration,batch=sep_batch,
                              round_id=k*closure_rounds+cut_round,tight_surrogate=tight_surrogate)
            inner.append({'pdca':outcome['inner'],'adaptive_exact_tangent_separation':close,
                          'restricted_objective':outcome['restricted_objective']})
            save_json(state_path,state)
            if close['all_closed'] and close['added']==0:
                break
        trials=[];accepted=False
        separation_added=sum(x['adaptive_exact_tangent_separation']['added'] for x in inner)
        separation_untested=max(x['adaptive_exact_tangent_separation']['untested'] for x in inner)
        improved_references=0
        for j,base_fraction in enumerate(line_fractions):
            fraction=float(base_fraction)*float(state.get('trust_scale',1.))
            trial=w+fraction*(candidate-w)
            if np.any(np.abs(trial)>150.):
                trials.append({'fraction':fraction,'verified':False,'accepted':False,
                               'reason':'coefficient_box_violation'})
                continue
            trial_id=hashlib.sha256(np.ascontiguousarray(trial,dtype=np.float64).tobytes()).hexdigest()[:12]
            plan=verify_policy(output,f'trial_{k}_{j}_{trial_id}',weeks,index,X,booked,trial,
                               alpha,h,seconds,threads,workers,
                               allow_incomplete=True)
            pending=[ck for ck,v in plan.items() if not v.get('complete')]
            if pending:
                trials.append({'fraction':fraction,'verified':False,
                               'accepted':False,
                               'reason':'unresolved_deployed_solver',
                               'pending_components':pending[:10],
                               'pending_count':len(pending)})
                continue
            total=sum(v['realized_cost'] for v in plan.values())
            regret=(total-sum(refs[key]['oracle_cost'] for key in plan))/n
            if regret < -.02:
                raise AssertionError('Candidate plan beat certified hindsight optimum')
            _,pred=display_and_duration(X@trial,booked,alpha,h)
            feature_effect=float(np.sqrt(np.mean((X@(trial-w))**2)))
            score={'fraction':fraction,'verified':True,'regret_per_case':regret,'cost_per_case':total/n,
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
                                          'fraction':fraction,'w':trial.tolist(),
                                          'prediction_mae':score['prediction_mae'],
                                          'rms_recommendation_change':feature_effect,
                                          'max_abs_coefficient':float(np.max(np.abs(trial))),
                                          'separation_added':separation_added})
                state['terminal_restarts']=0
                state['trust_scale']=min(2.,float(state.get('trust_scale',1.))*trust_growth)
                break
        state.setdefault('history',[]).append({'outer':k+1,'before':old_regret,
                            'inner':inner,'candidates':trials,'accepted':accepted,
                            'separation_added':separation_added,
                            'separation_untested':separation_untested,
                            'library_size_before':size_before_round,
                            'library_size_after':sum(len(v) for v in state['library'].values())})
        state['outer']=k+1
        save_json(state_path,state)
        if not accepted:
            if any(x.get('reason')=='unresolved_deployed_solver' for x in trials):
                state['stopping']='unresolved_candidate_deployment'
                state['outer']=k
                state['complete']=False
                save_json(state_path,state)
                return state
            terminal=None
            if terminal_separation:
                _,probe_duration=display_and_duration(X@candidate,booked,alpha,h)
                terminal=enrich_cuts(output,f'terminal_{k}_'+hashlib.sha256(
                    np.ascontiguousarray(probe_duration,dtype=np.float64).tobytes()
                ).hexdigest()[:12],blocks,probe_duration,gamma,state['library'],
                    sep_seconds,threads,workers,prior_duration=current_duration,
                    batch=0,round_id=k,tight_surrogate=tight_surrogate)
                state['history'][-1]['terminal_separation']=terminal
            gained=(sum(len(v) for v in state['library'].values())>size_before_round
                    or improved_references>0)
            restarts=int(state.get('terminal_restarts',0))
            if not terminal_separation and gained and k+1<outer:
                # Preserve historical behavior when terminal verification
                # is disabled: additional cuts may improve the next direction.
                print('New competitor/reference information; continue within outer budget',
                      flush=True)
            elif (terminal_separation and gained
                    and restarts<max_terminal_restarts and k+1<outer):
                state['terminal_restarts']=restarts+1
                state['trust_scale']=max(.25,float(state.get('trust_scale',1.))/trust_growth)
                print('New cuts or references: bounded restart from incumbent',flush=True)
            else:
                if terminal is not None and not terminal['all_closed']:
                    state['stopping']='no_improvement_terminal_separation_incomplete'
                else:
                    state['stopping']='no_verified_improvement_at_tested_steps'
                break
    else:
        state['stopping']='outer_budget'
    state['complete']=True
    state['method']='ANCHORED_REGRET_DC'
    state['feature_dimensions']=int(X.shape[1])
    state['training_cases']=int(n)
    state['training_site_weeks']=len(weeks)
    state['model_claim']='monotone verified empirical regret, not global bilevel optimality'
    state['training_finished']=True
    state['final_policy_frozen']=True
    state['separation_closed']=bool(
        state.get('history') and
        (state['history'][-1].get('terminal_separation') or
         state['history'][-1]['inner'][-1]['adaptive_exact_tangent_separation']).get('all_closed',False))
    state['deployment_verified']=True
    state['tight_surrogate']=bool(tight_surrogate)
    save_json(state_path,state)
    return state
