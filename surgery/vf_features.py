"""Schedule-independent structural features for the single joint-VF policy.

Features use only bookings, eligibility, rostered capacity and historical
training-derived spread scores. No provisional plan, actual duration,
tie-breaking seed or future outcome enters feature construction.
"""
from __future__ import annotations
import numpy as np

STRUCTURAL_NAMES=(
    "component_capacity_pressure",
    "eligibility_bottleneck_pressure",
    "surgeon_capacity_pressure",
    "log_eligible_slots",
    "surgeon_spread_x_eligibility_pressure",
    "procedure_spread_x_eligibility_pressure",
)


def feature_names(schema="structural"):
    if schema!="structural":
        raise ValueError("Only the frozen structural VF feature schema is supported")
    return STRUCTURAL_NAMES


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


def booked_plan_features(items,booked,plans,history,schema="structural",
                         capacity=480.,turnover=30.,reach=24.):
    """Compatibility entry point; plans are deliberately ignored.

    Kept only for existing structural feature tests and reproducibility.
    """
    feature_names(schema)
    return structural_features(items,booked,history,capacity=capacity,
                               turnover=turnover)


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
