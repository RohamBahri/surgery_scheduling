"""Outcome-blind scheduling context for the joint VF recommendation policy.

The original feature schemas use a booked-duration provisional schedule.
The new 'structural' schema depends ONLY on booked durations, the eligibility
graph and historical case information. In particular it never uses a plan,
realized durations, future observations, or the scheduler's tie-breaking rule.
All schemas use training-fitted standardization at evaluation time.
"""
from __future__ import annotations
import numpy as np
from .weekly_planner import loads

BASE_NAMES=(
    "surgeon_spread",
    "procedure_spread",
    "draft_congestion",
    "surgeon_spread_x_congestion",
    "procedure_spread_x_congestion",
)
INTERACTION_NAMES=BASE_NAMES+("centered_booked_x_congestion",)
CLEAN_NAMES=("draft_congestion","surgeon_spread_x_congestion",
             "procedure_spread_x_congestion")
STRUCTURAL_NAMES=(
    "component_capacity_pressure",
    "eligibility_bottleneck_pressure",
    "surgeon_capacity_pressure",
    "log_eligible_slots",
    "surgeon_spread_x_eligibility_pressure",
    "procedure_spread_x_eligibility_pressure",
)
CONTEXT_NAMES=BASE_NAMES+(
    "absolute_congestion",
    "reachable_capacity_proximity",
    "log_room_case_count",
    "booking_share_of_room",
    "log_surgeon_alternative_slots",
    "congestion_x_booking_share",
    "congestion_x_alternative_slots",
)
# Keep ASCII identifiers in data artifacts and avoid names depending on locale.
CONTEXT_NAMES=tuple(x.replace("sur­geon","surgeon") for x in CONTEXT_NAMES)


def feature_names(schema="baseline"):
    if schema=="case_only":
        return ()
    if schema=="clean":
        return CLEAN_NAMES
    if schema=="structural":
        return STRUCTURAL_NAMES
    if schema in ("baseline","congestion_clip3"):
        return BASE_NAMES
    if schema in ("congestion_booked","congestion_booked_clip3"):
        return INTERACTION_NAMES
    if schema=="context":
        return CONTEXT_NAMES
    raise ValueError("Unknown VF feature schema: "+str(schema))



def structural_features(items, booked, history, *, capacity=480.,
                        turnover=30., pressure_scale=120.):
    """Six predeclared plan-free scheduling-context features.

    A set E of eligible slots must accommodate all cases whose eligibility
    sets lie inside E. Thus the necessary load lower bound for E is
       sum(booked[M(E)]) + turnover*max(|M(E)|-|E|,0).
    This does NOT assign cases to rooms. Overall and surgeon pressure use
    the same conservative turnover lower bound. The asinh transform avoids
    hard clipping while remaining finite and monotone.
    """
    b=np.asarray(booked,float)
    h=np.asarray(history,float)
    if (b.ndim!=1 or not np.isfinite(b).all() or np.any(b<=0) or
            h.ndim!=2 or h.shape[0]!=len(b) or h.shape[1]<2 or
            not np.isfinite(h[:,:2]).all()):
        raise ValueError("Invalid booked durations or historical spread array")
    if not (capacity>0 and turnover>=0 and pressure_scale>0):
        raise ValueError("Invalid room-load normalization")
    out=np.empty((len(b),len(STRUCTURAL_NAMES)),float)
    seen=np.zeros(len(b),bool)

    def pressure(load,n_cases,n_slots):
        if n_slots<=0:
            raise ValueError("No eligible scheduling slots")
        t=turnover*max(0,n_cases-n_slots)
        return float(np.arcsinh((load+t-capacity*n_slots)/
                                (pressure_scale*n_slots)))

    for key,c,rows in items:
        rows=np.asarray(rows,int)
        if (len(rows)!=c.n or len(set(map(int,rows)))!=len(rows)
                or (rows<0).any() or (rows>=len(b)).any()
                or seen[rows].any() or c.k<=0):
            raise ValueError("Overlapping or invalid component rows")
        seen[rows]=True
        local=b[rows]
        n_s=len(c.surgeon_ids)
        if (n_s!=len(c.allowed) or len(c.surgeon)!=c.n):
            raise ValueError("Malformed surgeon eligibility")
        eligible=[frozenset(map(int,arc)) for arc in c.allowed]
        if any(not e or min(e)<0 or max(e)>=c.k for e in eligible):
            raise ValueError("Invalid room-day eligibility")
        surgeon=np.asarray(c.surgeon,int)
        if np.any(surgeon<0) or np.any(surgeon>=n_s):
            raise ValueError("Invalid surgeon indices")
        totals=np.bincount(surgeon,weights=local,minlength=n_s)
        counts=np.bincount(surgeon,minlength=n_s)
        component_p=pressure(float(local.sum()),c.n,c.k)
        # Eligibility is shared by all cases of the same surgeon.
        # Cache one bottleneck calculation per distinct eligible set.
        bottleneck={}
        for e in set(eligible):
            forced=np.array([bool(q.issubset(e)) for q in eligible],bool)
            nforced=int(counts[forced].sum())
            forcedload=float(totals[forced].sum())
            bottleneck[e]=pressure(forcedload,nforced,len(e))
        own=np.array([pressure(float(totals[s]),int(counts[s]),
                               len(eligible[s])) for s in range(n_s)])
        eligibility_pressure=np.array(
            [bottleneck[eligible[s]] for s in surgeon],float)
        own_pressure=own[surgeon]
        flexibility=np.log1p(np.array(
            [len(eligible[s]) for s in surgeon],float))
        hist_s, hist_p=h[rows,0],h[rows,1]
        out[rows]=np.column_stack([
            np.full(c.n,component_p),
            eligibility_pressure,
            own_pressure,
            flexibility,
            eligibility_pressure*hist_s,
            eligibility_pressure*hist_p,
        ])
    if not seen.all() or not np.isfinite(out).all():
        raise ValueError("Incomplete or nonfinite structural features")
    return out


def booked_plan_features(items, booked, plans, history, schema="baseline",
                         capacity=480., turnover=30., reach=24.):
    """Return unstandardized context features in the case-row order.

    The provisional assignment is solved using booked durations and is fixed
    when the recommendation policy is optimized. Never access actual outcomes.
    """
    if schema=="structural":
        return structural_features(items,booked,history,capacity=capacity,
                                   turnover=turnover)
    booked=np.asarray(booked,float)
    history=np.asarray(history,float)
    if history.ndim!=2 or history.shape[0]!=len(booked) or history.shape[1]<2:
        raise ValueError("Historical spread matrix must have at least 2 columns")
    n=len(booked)
    out=np.empty((n,len(feature_names(schema))),float)
    seen=np.zeros(n,dtype=bool)
    for key,c,rows in items:
        rows=np.asarray(rows,int)
        if seen[rows].any() or len(rows)!=c.n:
            raise ValueError("Overlapping or incomplete component row indices")
        seen[rows]=True
        assignment=plans[key]
        if isinstance(assignment,dict):
            assignment=assignment["assignment"]
        z=np.asarray(assignment,int)
        b=booked[rows]
        room_load=loads(c,z,b)
        count=np.bincount(z,minlength=c.k)[z].astype(float)
        clip=3. if schema in ("congestion_clip3","congestion_booked_clip3") else 2.
        p=np.clip((room_load[z]-capacity)/120.,-clip,clip)
        surgeon,procedure=history[rows,0],history[rows,1]
        cols=([p,surgeon*p,procedure*p] if schema=="clean" else
              ([] if schema=="case_only" else
               [surgeon,procedure,p,surgeon*p,procedure*p]))
        if schema in ("congestion_booked","congestion_booked_clip3"):
            # Main congestion term is the effect at a 120-minute booking.
            cols.append(((b-120.)/120.)*p)
        if schema=="context":
            room_booked=np.bincount(z,weights=b,minlength=c.k)[z]
            share=b/room_booked
            flexibility=np.array([
                max(0,len(c.allowed[int(c.surgeon[i])])-1)
                for i in range(c.n)],float)
            log_alt=np.log1p(flexibility)
            cols.extend((
                np.abs(p),
                np.maximum(0.,1.-np.abs(room_load[z]-capacity)/(reach*count)),
                np.log1p(count),
                share,
                log_alt,
                p*share,
                p*log_alt,
            ))
        if cols:
            out[rows]=np.column_stack(cols)
    if not seen.all() or not np.isfinite(out).all():
        raise ValueError("Missing or nonfinite VF feature values")
    return out


def standardize_features(raw, scaler=None):
    """Training-fitted scaling only; do not learn scaling on evaluation cases."""
    raw=np.asarray(raw,float)
    if raw.ndim!=2:
        raise ValueError("Plan features must be a matrix")
    if raw.shape[1]==0:
        if scaler is not None and any(len(np.asarray(x)) for x in scaler):
            raise ValueError("Nonempty scaler for case-only feature schema")
        return raw.copy(),(np.empty(0),np.empty(0))
    if scaler is None:
        mu=raw.mean(axis=0)
        measured=raw.std(axis=0)
        sd=np.where(measured>1e-8,measured,1.)
    else:
        mu,sd=map(lambda x:np.asarray(x,float),scaler)
        if mu.shape!=(raw.shape[1],) or sd.shape!=(raw.shape[1],):
            raise ValueError("Feature schema does not match the fitted scaler")
        if np.any(sd<=0) or not np.isfinite(sd).all():
            raise ValueError("Invalid VF scaler")
    return (raw-mu)/sd,(mu,sd)
