"""Outcome-blind, booked-plan context for the joint VF recommendation policy.

All features depend only on bookings, the fixed roster/eligibility graph,
historical spread summaries, and a provisional planner solution for bookings.
The same deterministic feature recipe is applied during fit and deployment.
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
    if schema=="baseline":
        return BASE_NAMES
    if schema=="context":
        return CONTEXT_NAMES
    raise ValueError("Unknown VF feature schema: "+str(schema))


def booked_plan_features(items, booked, plans, history, schema="baseline",
                         capacity=480., turnover=30., reach=24.):
    """Return unstandardized context features in the case-row order.

    The provisional assignment is solved using booked durations and is fixed
    when the recommendation policy is optimized. Never access actual outcomes.
    """
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
        p=np.clip((room_load[z]-capacity)/120.,-2.,2.)
        surgeon,procedure=history[rows,0],history[rows,1]
        cols=[surgeon,procedure,p,surgeon*p,procedure*p]
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
        out[rows]=np.column_stack(cols)
    if not seen.all() or not np.isfinite(out).all():
        raise ValueError("Missing or nonfinite VF feature values")
    return out


def standardize_features(raw, scaler=None):
    """Training-fitted scaling only; do not learn scaling on evaluation cases."""
    raw=np.asarray(raw,float)
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
