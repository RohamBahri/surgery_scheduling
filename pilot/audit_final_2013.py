"""Strict training-only cost, congestion, tie-break and time-budget audit.

No test cohorts or 2013 outcomes are accessed by this module.
"""
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from pilot import run_final_2013 as run
from surgery.anchored_core import key
from surgery.anchored_oracles import deployed
from surgery.anchored_training import verify_policy
from surgery.planner import display_and_duration
from surgery.weekly_planner import loads


def attainable_cap(component,cap=960.,seconds=180.):
    """Can a clairvoyant reassignment avoid every room exceeding cap?

    Uses exactly the existing surgeon-to-slot eligibility and required-days
    constraints; the duration is ACTUAL and is not available to the planner.
    A feasible result diagnoses avoidable overload, not a deployable policy.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from surgery.weekly_planner import _model
    m,x,used,rooms=_model(component,seconds,1)
    try:
        for j,cases in enumerate(rooms):
            duration=gp.quicksum((float(component.actual[i])+30.)*x[i,j] for i in cases)
            m.addConstr(duration-30.*used[j] <= cap*used[j])
        m.setObjective(0.,GRB.MINIMIZE)
        m.optimize()
        if m.SolCount:
            return {'classification':'feasible_below_cap',
                    'certified':True,
                    'max_load':float(max(loads(component,[next(k for k in
                    component.allowed[int(component.surgeon[i])]
                    if x[i,k].X > .5) for i in range(component.n)],
                    component.actual)))}
        if m.Status == GRB.INFEASIBLE:
            return {'classification':'cannot_fit_under_cap','certified':True}
        return {'classification':'unresolved','certified':False,'gurobi_status':int(m.Status)}
    finally:
        m.dispose()


def audit_training(args):
    run.settings(args)
    if (args.root/'FINAL_FREEZE.json').exists():
        raise RuntimeError('Training audit is immutable after test freeze')
    paths=run.complete_policies(args)
    data,weeks,index=run.training(args,'cohort')
    out=args.root/'train_audit';out.mkdir(parents=True,exist_ok=True)
    refroot=args.root/'cohort'/'train_refs'/'anchored_references.json'
    if not refroot.is_file():raise RuntimeError('Training hindsight references missing')
    oracle=json.loads(refroot.read_text())
    expected={key(c) for w in weeks for c in w.components()}
    if set(oracle)!=expected or any(not oracle[k].get('complete') for k in expected):
        raise RuntimeError('Uncertified training oracle; stop before 2013 scoring')
    models={'BOOKED':np.zeros(data['X'].shape[1]),
            'CASE_COHORT':np.asarray(json.loads(paths['CASE_COHORT'].read_text())['w'],float),
            'ANCHORED_0p001':np.asarray(json.loads(paths['ANCHORED_0p001'].read_text())['w'],float)}
    cache={}
    for name,w in models.items():
        if name=='ANCHORED_0p001':
            folder=args.root/'cohort'/'sensitivity'/run.label(.001)/'anchored'
        else:folder=out/'verified'
        cache[name]=verify_policy(folder,name,weeks,index,data['X'],data['booked'],w,
              run.ALPHA,run.H,run.BUDGET['evaluation_seconds'],1,run.BUDGET['workers'])
        if set(cache[name])!=expected:raise RuntimeError('Missing audit components '+name)
        for k,v in cache[name].items():
            p=v.get('planner',{})
            if not v.get('complete') or not p.get('primary_complete') or not p.get('tie_complete'):
                raise RuntimeError('Incomplete exact planner certification in audit: '+name+' '+k)
    saved=out/'audit_extreme_cap_results.json'
    caprecord=json.loads(saved.read_text()) if saved.is_file() else {}
    listing=[]
    for week in weeks:
        for c in week.components():
            k=key(c)
            b=cache['CASE_COHORT'][k];a=cache['ANCHORED_0p001'][k]
            v=cache['BOOKED'][k]
            delta=float(a['realized_cost']-b['realized_cost'])
            full=loads(c,a['assignment'],c.actual)
            summary={'component':k,'monday':week.monday,'site':week.group,
                     'cases':int(c.n),'delta_anchored_minus_case':delta,
                     'booked_max_load':float(v['max_realized_room_load']),
                     'case_max_load':float(b['max_realized_room_load']),
                     'anchored_max_load':float(a['max_realized_room_load']),
                     'anchored_extreme_rooms':int((full>960).sum()),
                     'case_changed':bool(a['assignment']!=b['assignment']),
                     'anchored_assignment':a['assignment'],
                     'case_assignment':b['assignment']}
            listing.append((summary,c,full))
    ordered=sorted(listing,key=lambda x:x[0]['delta_anchored_minus_case'])
    topgain=[x[0] for x in ordered[:25]]
    largest_losses=[x[0] for x in ordered[-15:]]
    extreme=sorted([x for x in listing if max(x[0]['anchored_max_load'],
              x[0]['case_max_load'])>960],
              key=lambda x:-max(x[0]['anchored_max_load'],x[0]['case_max_load']))
    unresolved=0
    for item,c,load in extreme[:40]:
        k=item['component']
        if k not in caprecord or not caprecord[k].get('certified'):
            caprecord[k]=attainable_cap(c,cap=960.,seconds=180.)
            run.save(saved,caprecord)
        if not caprecord[k].get('certified'):unresolved+=1
        item['clairvoyant_reallocation']=caprecord[k]
    # Larger extreme set still reported, not silently claimed resolved.
    unchecked=max(0,len(extreme)-40)
    # Small effective-duration perturbations on a fixed, deterministic subset
    # of training components. These are independent from surgeon-response
    # display edits; we are testing the deployed planner's discontinuities.
    sampler=np.random.default_rng(20261019)
    choices=[x for x in listing if x[1].n>1 and any(
             len(x[1].allowed[int(s)])>1 for s in x[1].surgeon)]
    indices=sampler.choice(len(choices),size=min(30,len(choices)),replace=False)
    stability=[]
    _,reviewed=display_and_duration(data['X']@models['ANCHORED_0p001'],
                                     data['booked'],run.ALPHA,run.H)
    for j in indices:
        item,c,_=choices[int(j)]
        before=cache['ANCHORED_0p001'][item['component']]
        pos=np.array([index[int(cid)] for cid in c.case_ids],int)
        for eps in (.5,2.):
            sign=np.where(np.arange(c.n)%2==0,1.,-1.)
            revised=np.maximum(1.,reviewed[pos]+eps*sign)
            planned=deployed(c,revised,seconds=run.BUDGET['evaluation_seconds'])
            if not planned.get('complete'):
                unresolved+=1
                stability.append(dict(component=item['component'],eps=eps,
                                      complete=False,status=planned.get('status')))
                continue
            stability.append(dict(component=item['component'],eps=eps,complete=True,
                     changed_assignment=bool(planned['assignment']!=before['assignment']),
                     regret_change=float(planned['realized_cost']-before['realized_cost'])))
    limits={}
    for method,p in paths.items():
        record=json.loads(p.read_text())
        if method.startswith('ANCHORED'):
            candidates=[a for h in record.get('history',[]) for a in h.get('candidates',[])]
            incomplete=[a for a in candidates if a.get('reason')=='unresolved_deployed_solver']
            masters=[row for h in record.get('history',[])
                      for inner in h.get('inner',[]) for row in inner.get('pdca',[])]
            failed_master=[row for row in masters if row.get('complete') is False]
            limits[method]=dict(training_outer=int(record.get('outer',0)),
               stopping=record.get('stopping'),unresolved_training_candidates=len(incomplete),
               incomplete_master_qp=len(failed_master),
               master_qp_statuses=dict(Counter(str(x.get('status')) for x in masters)),
               terminal_separation_closed=bool(record.get('separation_closed',False)),
               accepted_count=len(record.get('accepted',[])))
            if incomplete:unresolved+=len(incomplete)
            if failed_master:unresolved+=len(failed_master)
        elif method=='VF_COHORT':
            steps=record.get('iterations',[])
            hitlimit=sum(any(p.get('iterations',0)>=run.BUDGET['vf_inner'] for p in s.get('fit_passes',[])) for s in steps)
            unsuccessful=sum(any(p.get('optimizer_success') is False
                        for p in step.get('fit_passes',[])) for step in steps)
            limits[method]=dict(stopping=record.get('stopping'),outer=len(steps),
                                inner_iteration_limit_rounds=hitlimit,
                                inner_not_converged_rounds=unsuccessful,
                                accepted=sum(bool(x.get('accepted')) for x in steps))
        elif method=='GAP_COHORT':
            steps=record.get('history',[])
            exact_last=steps[-1].get('certified_after',{}).get('all_weekly_oracles_exact',False) if steps else False
            final_width=steps[-1].get('certified_after',{}).get('regularized_gap_interval_width_per_case') if steps else None
            limits[method]=dict(stopping=record.get('stopping'),outer=len(steps),
                                overlapping_intervals=sum(a.get('comparison')=='OVERLAPPING_INTERVALS'
                                  for h in steps for a in h.get('candidates',[])),
                                all_final_weekly_oracles_exact=exact_last,
                                final_certificate_interval_width=final_width,
                                paused=bool(record.get('paused')))
            if record.get('paused'):unresolved+=1
            if not exact_last:unresolved+=1
        elif method.startswith('CASE'):
            bic=json.loads((p.parent/'case_bic_path.json').read_text())
            selected=min(bic,key=lambda x:x['bic_proxy'])
            passes=selected['optimization']
            limits[method]=dict(lambda_l1=selected['lambda_l1'],
                  selected_optimizer_success=[bool(x.get('optimizer_success')) for x in passes],
                  selected_optimizer_iterations=[x.get('iterations') for x in passes],
                  selected_optimizer_messages=[x.get('message') for x in passes])
    risk=[]
    if extreme:risk.append('realized_room_days_above_960_minutes')
    if topgain and sum(x['delta_anchored_minus_case'] for x in topgain)<0:
        risk.append('concentrated_realized_cost_gains')
    if unchecked:risk.append('more_than_40_extreme_components_not_checked_for_reallocations')
    if any(x.get('inner_iteration_limit_rounds',0) for x in limits.values()):
        risk.append('vf_inner_optimizer_reached_iteration_cap')
    if any(x.get('inner_not_converged_rounds',0) for x in limits.values()):
        risk.append('vf_lbfgs_not_converged')
    if any(not all(x.get('selected_optimizer_success',[True])) for x in limits.values()):
        risk.append('case_optimizer_nonconvergence')
    stats={name:dict(regret_per_case=sum(v['realized_cost']-oracle[k]['realized_cost']
                    for k,v in cache[name].items())/len(data['booked']),
               max_load=max(v['max_realized_room_load'] for v in cache[name].values()),
               rooms_over_720=sum(v['room_days_over_720'] for v in cache[name].values()),
               rooms_over_960=sum(v.get('room_days_over_960',0) for v in cache[name].values()))
               for name in cache}
    report=dict(complete=True,protocol='pre-2013 training only',
        cases=len(data['booked']),group_weeks=len(weeks),components=len(expected),
        method_regret_and_extremes=stats,solver_limits=limits,
        top25_beneficial_components=topgain,top15_harmful_components=largest_losses,
        extreme_components=len(extreme),extreme_reallocation_checks=caprecord,
        extreme_components_not_checked=unchecked,
        perturbation_stability=stability,
        unresolved_certifications=unresolved,
        operational_risk_flags=risk,
        interpretation='Retrospective eligibility and true-duration reallocation are diagnostics, NOT deployable constraints.',
        no_test_outcomes_used=True)
    run.save(out/'audit_summary.json',report)
    print('TRAIN AUDIT:',json.dumps({
        'components':len(expected),'unresolved':unresolved,
        'extreme_components':len(extreme),'flags':risk,
        'regret':{k:v['regret_per_case'] for k,v in stats.items()}},indent=2),flush=True)
    if unresolved:raise RuntimeError('Unresolved audit solves; inspect train_audit/audit_summary.json and rerun audit')