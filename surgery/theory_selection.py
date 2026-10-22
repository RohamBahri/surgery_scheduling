"""Small, honest candidate generators for the 12-hour paper selection run.

A GAP or ANCHORED proposal is never called trained on the full cohort until
the SAME direct SHARED objective has been checked on every training component.
These are bounded proposal engines, not exact optimization of the original
full-population GAP or anchored DC objectives.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize

from surgery.anchored_core import anchored_score, normalize_tangent, slopes_from_reference
from surgery.weekly_planner import evaluate


class Expired(Exception):
    pass


def _sample(items, limit=28):
    rng=np.random.default_rng(20261022)
    flexible=[item for item in items if any(len(item[1].allowed[int(s)])>1
                      for s in item[1].surgeon)]
    source=flexible or list(items)
    if len(source)<=limit:return source
    # Preserve large cases; small cases alone conceal all tie/complexity issues.
    ordered=sorted(source,key=lambda x:-x[1].n)
    large=ordered[:min(6,limit//3)]
    chosen={x[0] for x in large}
    remaining=[x for x in source if x[0] not in chosen]
    idx=rng.choice(len(remaining),size=limit-len(large),replace=False)
    return large+[remaining[int(j)] for j in idx]


def _confirm(ctx, start, proposed, output, seconds, cores):
    """Attempt full-training, same-objective confirmation; never use partial sums."""
    from pilot import run_theory_full as runner
    deadline=time.monotonic()+max(.1,seconds)
    cands=[np.asarray(start,float).tolist(),np.asarray(proposed,float).tolist()]
    score,valid=runner.parallel_score(ctx['train'],ctx['fit'],ctx['H'],
             cands,'SHARED',ctx['pools'],min(120.,seconds),cores,190,
             output/'full_direct_A',task_seconds=min(480.,seconds),deadline=deadline)
    record=dict(completed=bool(valid.all()),validated_components=len(ctx['train']),
                baseline_score=float(score[0]) if valid[0] else None,
                proposed_score=float(score[1]) if valid[1] else None)
    record['accepted']=bool(valid.all() and score[1]<score[0]-1e-7)
    return (np.asarray(proposed if record['accepted'] else start,float),record)


def anchored_proposal(ctx,start,output,*,seconds=1800.,cores=15):
    """Anchored schedule-specific tangent majorization in the five-feature class.

    Uses incumbent/hash and original feasible assignments as *aspirations*.
    No optimal hindsight value is required or silently manufactured.
    """
    from pilot import run_theory_full as runner
    from surgery.theory_cache import connect
    output.mkdir(parents=True,exist_ok=True)
    clock=time.monotonic();deadline=clock+seconds
    baseline=ctx['fit']['X']@ctx['fit']['case_w']
    H=ctx['H'];booked=ctx['fit']['booked']
    old_state=runner._SCORE_STATE
    cache=connect(str(ctx['root']/'exact_deployments.sqlite3'))
    runner._SCORE_STATE=dict(planner_conn=cache,planner_stats={})
    blocks=[];missing=0
    try:
        for ck,c,rows in _sample(ctx['train']):
            if deadline-time.monotonic()<30:break
            d0=runner.reviewed(c.booked,baseline[rows])
            advised=runner.reviewed_policy(c,rows,H,d0,start)
            candidates=[tuple(map(int,c.original))]
            try:
                plans=runner.deploy(c,advised,min(60.,deadline-time.monotonic()),
                                    deadline=deadline)
                candidates+=list(plans.values())
            except RuntimeError:
                missing+=1
            assignments=list(dict.fromkeys(candidates))
            best=min(assignments,key=lambda z:evaluate(c,z,c.actual)['cost'])
            anchor=dict(assignment=list(best),beta=advised.copy(),
                        slopes=slopes_from_reference(c,best,advised))
            cuts=[normalize_tangent(c,z,advised) for z in assignments]
            blocks.append((c,rows,d0,anchor,cuts))
    finally:
        runner._SCORE_STATE=old_state
        cache.close()
    if not blocks:
        return np.asarray(start,float),dict(
            status='NO_ANCHOR_BLOCKS',confirmed=False,missing=missing)
    def objective(coeff):
        if time.monotonic()>deadline:raise Expired()
        total=0.
        for c,rows,d0,anchor,cuts in blocks:
            d=runner.reviewed_policy(c,rows,H,d0,coeff)
            total+=anchored_score(c,anchor,cuts,d,0.,gamma=2.,exact=True)
            total+=runner.LAM/2*np.square(d-d0).sum()
        return float(total/sum(c.n for c,*_ in blocks))
    start=np.asarray(start,float)
    radius=32.
    lo=ctx['beta']-radius;hi=ctx['beta']+radius
    try:
        fit=minimize(objective,np.clip(start,lo,hi),method='Powell',
                     bounds=list(zip(lo,hi)),
                     options=dict(maxiter=10,maxfev=70,xtol=1e-3,ftol=1e-5))
        proposed=np.clip(fit.x,lo,hi)
        surrogate=dict(initial=float(objective(start)),candidate=float(objective(proposed)),
                       scipy_success=bool(fit.success),scipy_message=str(fit.message))
    except Expired:
        proposed=start.copy()
        surrogate=dict(status='ANCHOR_PROPOSAL_TIME_LIMIT')
    left=deadline-time.monotonic()
    if left<30 or np.max(abs(proposed-start))<1e-8:
        return start,dict(status='NO_CONFIRMED_IMPROVEMENT',
                          anchored_components=len(blocks),missing=missing,
                          surrogate=surrogate,confirmed=False)
    final,certificate=_confirm(ctx,start,proposed,output,left,cores)
    return final,dict(status='ACCEPTED' if certificate['accepted'] else
                      'FULL_DIRECT_CONFIRMATION_INCOMPLETE_OR_NOT_BETTER',
                      anchored_components=len(blocks),missing=missing,
                      surrogate=surrogate,confirmed=certificate['completed'],
                      direct_confirmation=certificate,
                      proposal=proposed.tolist())


def _gap_one(task):
    """Certified lower/upper interval for a SINGLE EG adverse maximization."""
    from surgery.theory_exact import expected_gap_mip
    c,d,d0,bank,seconds=task
    from surgery.advice_comparison import LAM
    ans=expected_gap_mip(c,d,bank,seconds=seconds,gamma=1.)
    penalty=LAM/2*np.square(d-d0).sum()
    lb=(None if ans.value is None else float(ans.value+penalty))
    ub=(None if ans.bound is None else float(ans.bound+penalty))
    return lb,ub,bool(ans.complete)


def gap_proposal(ctx,start,vf,output,*,seconds=1800.,cores=15):
    """EG-interval screening; direct-cost verification is a separate gate.

    An unclosed maximization returns a proper interval, never a fabricated
    exact objective. This is GAP-guided candidate generation, not an EG
    full-cohort optimum.
    """
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    from pilot import run_theory_full as runner
    output.mkdir(parents=True,exist_ok=True)
    clock=time.monotonic();deadline=clock+seconds
    start=np.asarray(start,float)
    proposals=[start]
    if vf is not None:
        other=np.asarray(vf,float)
        if np.max(abs(other-start))>1e-8:proposals.append(other)
    for j in range(len(start)):
        for sign in (-1.,1.):
            theta=start.copy()
            theta[j]=np.clip(theta[j]+sign*4.,ctx['beta'][j]-32.,
                                      ctx['beta'][j]+32.)
            proposals.append(theta)
    banked=[]
    base=ctx['fit']['X']@ctx['fit']['case_w']
    for ck,c,rows in _sample(ctx['train'],limit=18):
        d0=runner.reviewed(c.booked,base[rows])
        bank=runner.scenarios(c,d0,ctx['pools'],20261020,count=50)
        banked.append((c,rows,d0,bank))
    score=[dict(lower=0.,upper=0.,resolved=0,unresolved=0) for _ in proposals]
    tasks=[]
    for j,co in enumerate(proposals):
        for c,rows,d0,bank in banked:
            d=runner.reviewed_policy(c,rows,ctx['H'],d0,co)
            tasks.append((j,(c,d,d0,bank,min(20.,seconds/6))))
    # Limit queue to core count, so stage overrun doesn't launch unused work.
    with ProcessPoolExecutor(max_workers=min(cores,len(tasks)),
                             mp_context=get_context('spawn')) as pool:
        source=iter(tasks);active={}
        def send():
            if time.monotonic()+20>deadline:return
            try:j,task=next(source)
            except StopIteration:return
            future=pool.submit(_gap_one,task);active[future]=j
        for _ in range(min(cores,len(tasks))):send()
        while active and time.monotonic()<deadline-5:
            from concurrent.futures import wait,FIRST_COMPLETED
            done,_=wait(active,timeout=min(15.,deadline-time.monotonic()),
                         return_when=FIRST_COMPLETED)
            if not done:continue
            for future in done:
                j=active.pop(future)
                try:
                    lb,ub,cert=future.result()
                    if lb is not None:score[j]['lower']+=lb
                    if ub is not None:score[j]['upper']+=ub
                    if lb is None or ub is None:score[j]['unresolved']+=1
                    if cert:score[j]['resolved']+=1
                except Exception:score[j]['unresolved']+=1
                send()
        for future in active:future.cancel()
    eligible=[j for j,v in enumerate(score) if v['unresolved']==0 and
              v['resolved']+v['unresolved']==len(banked)]
    # A finite EG upper bound is a valid *pessimistic* comparison. Using it
    # to propose does not imply a full-cohort EG optimum.
    chosen=min(eligible,key=lambda j:score[j]['upper'],default=0)
    selected=proposals[chosen]
    left=deadline-time.monotonic()
    if chosen==0 or left<30:
        return start,dict(status='NO_EG_VERIFIED_PROPOSAL',EG_intervals=score,
                          screened_components=len(banked),confirmed=False)
    final,confirmation=_confirm(ctx,start,selected,output,left,cores)
    return final,dict(status='ACCEPTED' if confirmation['accepted'] else
                      'NOT_FULLY_CONFIRMED_OR_INFERIOR',
                      EG_intervals=score,screened_components=len(banked),
                      candidate_source='bounded_expected_gap_maximization',
                      candidate=selected.tolist(),confirmed=confirmation['completed'],
                      direct_confirmation=confirmation)
