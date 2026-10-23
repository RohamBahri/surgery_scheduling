"""Nominal SHARED: direct realized schedule-cost training, not a scenario bank.

The historical SHARED objective used expected costs under duration banks.
Here the TRAINING outcome for each week is its historically realized case
durations, with the same non-monotone response and hash-0 planner as CASE/VF.

The direct value is discontinuous in coefficients. Thus pDCA is used for
VF's continuous upper-bound surrogate, while SHARED trains using bounded
multidirectional search. All SHARED accepted updates have a certified
full-training direct objective. Frozen libraries screen but never decide
acceptance. No test outcomes enter SHARED selection.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from surgery.nonmonotone_learning import planned_cost_fast


def tag(theta):
    w=np.ascontiguousarray(theta,dtype="<f8")
    return hashlib.sha256(w.tobytes()).hexdigest()[:18]


def score(items, results, theta, lam):
    """Cost per training component; None for any uncertified deployment."""
    total=0.
    for key,week,rows in items:
        entry=results.get(key)
        if not entry or not entry["complete"]:
            return None
        total+=planned_cost_fast(week,entry["assignment"],week.actual)
    return float(total/len(items)+lam*np.abs(theta).sum())


def proxy(items, libraries, durations, costs, theta, lam):
    """Low-cost screening ONLY; never treated as a SHARED training result."""
    total=0.
    for key,week,rows in items:
        d=durations[rows]
        options=libraries[key]
        j=min(range(len(options)),key=lambda j:
                 (planned_cost_fast(week,options[j],d),tuple(options[j])))
        total+=costs[key][j]
    return float(total/len(items)+lam*np.abs(theta).sum())


def subset(items, size=128):
    """Deterministic mixed-size, both-site screening cohort, train only."""
    if len(items)<=size:
        return list(items)
    rng=np.random.default_rng(20261022)
    selected=set()
    sites=sorted({c.group for _,c,_ in items})
    for site in sites:
        index=[i for i,(_,c,_) in enumerate(items) if c.group==site]
        largest=sorted(index,key=lambda i:(-items[i][1].n,items[i][0]))
        selected.update(largest[:min(12,len(largest))])
    remaining=[i for i in range(len(items)) if i not in selected]
    extra=rng.choice(remaining,size=size-len(selected),replace=False)
    selected.update(map(int,extra))
    return [items[i] for i in sorted(selected)]


def directions(p):
    eye=np.eye(p)
    out=[sgn*eye[j] for j in range(p) for sgn in (-1.,1.)]
    # Mixed updates can cross a threshold when single axes cannot.
    anchor=2 if p>2 else 0
    for j in range(p):
        if j==anchor:
            continue
        for sign in (-1.,1.):
            out.extend([eye[anchor]+sign*eye[j],
                        -eye[anchor]-sign*eye[j]])
    return [v/np.max(np.abs(v)) for v in out]


def train_shared(items, booked, H, case_raw, response_map, cache, *,
                 root, vf_thetas, lam, libraries, radius=64.,
                 max_rounds=8, screen_size=96, shortlist=8):
    """Train one SHARED policy with exact full-training acceptance.

    All candidates are generated without knowing 2013 outcomes. Library
    scores only shortlist; small real-planner solves rank the shortlist;
    a candidate is accepted ONLY after complete hash-0 solves on EVERY
    training component, and only if its full-training direct objective
    improves. Failures to certify are never labeled optima.
    """
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    file=root/"shared_progress.json"
    if file.exists():
        state=json.loads(file.read_text())
        if state.get("status")=="COMPLETE":
            return state
    else:
        state=dict(status="RUNNING",history=[],theta=None,score=None)
    cands=[np.zeros(H.shape[1])]+[np.asarray(v,float) for v in vf_thetas]
    screening=subset(items,screen_size)
    # Validate every previously generated library assignment exactly once.
    # Library costing may then use the fast, mathematically identical cost.
    from surgery.weekly_planner import loads
    realized={}
    for key,week,rows in items:
        opts=libraries[key]
        for z in opts:
            loads(week,z,week.booked)
        realized[key]=[planned_cost_fast(week,z,week.actual) for z in opts]

    def reviewed(theta):
        return response_map.reviewed(case_raw+H@theta)

    def solve(candidate, selected, prefix):
        label=prefix+"_"+tag(candidate)
        return cache.solve(selected,reviewed(candidate),label,tie=True)

    if state.get("theta") is None:
        seeds=[]
        seeded=set()
        for candidate in cands:
            if tag(candidate) in seeded:
                continue
            seeded.add(tag(candidate))
            results=solve(candidate,items,"shared_seed")
            exact=score(items,results,candidate,lam)
            seeds.append(dict(theta=candidate.tolist(),score=exact,
                              certified=exact is not None,
                              certified_components=sum(x["complete"]
                                  for x in results.values())))
        valid=[x for x in seeds if x["certified"]]
        if not valid:
            state.update(status="NO_CERTIFIED_INITIALIZATION",seeds=seeds)
            file.write_text(json.dumps(state,indent=2))
            raise RuntimeError("SHARED: cannot certify either full-training start")
        best=min(valid,key=lambda x:x["score"])
        state.update(theta=best["theta"],score=best["score"],
                     starts=seeds,step=16.)
        file.write_text(json.dumps(state,indent=2))
    incumbent=np.asarray(state["theta"],float)
    best=float(state["score"])
    step=float(state.get("step",16.))
    start=time.perf_counter()
    for iteration in range(len(state["history"]),max_rounds):
        trial=[]
        seen={tag(incumbent)}
        for direction in directions(H.shape[1]):
            x=np.clip(incumbent+step*direction,-radius,radius)
            key=tag(x)
            if key in seen:
                continue
            seen.add(key)
            objective=proxy(items,libraries,reviewed(x),realized,x,lam)
            trial.append((objective,x))
        trial.sort(key=lambda pair:pair[0])
        # Proxy never accepts a solution; actual hash-0 schedules screen.
        baseline_screen=solve(incumbent,screening,"shared_screen")
        base_screen=score(screening,baseline_screen,incumbent,lam)
        candidates=[]
        for approx,x in trial[:shortlist]:
            partial=solve(x,screening,"shared_screen")
            screened=score(screening,partial,x,lam)
            candidates.append(dict(theta=x.tolist(),proxy=float(approx),
                                   screen=screened,
                                   certified_screen=screened is not None))
        eligible=[r for r in candidates if r["screen"] is not None
                  and base_screen is not None
                  and r["screen"]<base_screen-1e-8*max(1.,abs(base_screen))]
        eligible.sort(key=lambda x:x["screen"])
        accepted=False
        confirmed=[]
        # Confirm the best two screened directions on ALL training components.
        for candidate in eligible[:2]:
            x=np.asarray(candidate["theta"],float)
            full=solve(x,items,"shared_full")
            direct=score(items,full,x,lam)
            confirmed.append(dict(theta=x.tolist(),score=direct,
                                  certified=direct is not None,
                                  complete_components=sum(v["complete"]
                                                       for v in full.values())))
            if direct is not None and direct<best-1e-8*max(1.,abs(best)):
                incumbent=x
                best=direct
                accepted=True
                break
        row=dict(iteration=iteration+1,step=step,accepted=accepted,
                 score=float(best),candidates=candidates,
                 confirmed=confirmed,
                 elapsed_seconds=round(time.perf_counter()-start,3))
        state["history"].append(row)
        if not accepted:
            step/=2.
        state.update(theta=incumbent.tolist(),score=float(best),step=step)
        file.write_text(json.dumps(state,indent=2,allow_nan=False))
        print(f"SHARED {iteration+1}/{max_rounds}: "
              f"direct training={best:.6f}; accepted={accepted}; "
              f"step={row['step']:.2f}; elapsed={row['elapsed_seconds']:.1f}s",
              flush=True)
        if step<1.:
            break
    state["status"]="COMPLETE"
    file.write_text(json.dumps(state,indent=2,allow_nan=False))
    return state
