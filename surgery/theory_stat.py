"""Strong matched statistical recommendation, fitted on training outcomes only."""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from surgery.advice_comparison import LAM, reviewed


class TrainingBudget(Exception):
    pass


def fit_stat_plus(data, H, beta, *, radius=32., seconds=1800., out=None):
    """Multistart bounded optimization of the SAME five-feature advice class.

    STAT+ optimizes asymmetric case-duration loss, not scheduling cost.
    Every returned point is feasible, and its training objective cannot be
    worse than the initial STAT policy. No claim of global optimality.
    """
    if out is not None:
        path=Path(out)
        if path.exists():
            saved=json.loads(path.read_text())
            return np.asarray(saved["coeff"],float)
    else:
        path=None
    tic=time.monotonic()
    base=np.asarray(data["X"]@data["case_w"],float)
    d0=reviewed(data["booked"],base)
    booked=np.asarray(data["booked"],float)
    actual=np.asarray(data["actual"],float)
    beta=np.asarray(beta,float)
    H=np.asarray(H,float)
    radius=float(radius)
    n=len(booked)
    if n!=H.shape[0] or H.shape[1]!=len(beta):
        raise ValueError("STAT+ case/feature dimensions differ")
    def value(delta):
        if time.monotonic()-tic>=seconds:raise TrainingBudget()
        d=reviewed(booked,d0-booked+H@(beta+delta))
        err=actual-d
        return float((1.75*np.maximum(err,0.).sum()
                      +np.maximum(-err,0.).sum()
                      +LAM/2*np.square(d-d0).sum())/n)
    best=np.zeros(len(beta),float)
    best_value=value(best)
    rng=np.random.default_rng(20261022)
    starts=[best.copy()]
    for j in range(len(beta)):
        for sign in (-1.,1.):
            x=np.zeros(len(beta));x[j]=sign*radius/2
            starts.append(x)
    starts.extend(rng.uniform(-radius,radius,len(beta)) for _ in range(8))
    history=[];status="COMPLETE_STARTS"
    bounds=[(-radius,radius)]*len(beta)
    for i,start in enumerate(starts):
        if time.monotonic()-tic>=seconds:
            status="BUDGET_EXHAUSTED"
            break
        try:
            ans=minimize(value,start,method="Powell",bounds=bounds,
                         options=dict(maxiter=180,xtol=5e-4,ftol=1e-8))
            candidate=np.clip(np.asarray(ans.x,float),-radius,radius)
            objective=value(candidate)
            if np.isfinite(objective) and objective<best_value-1e-10:
                best,best_value=candidate.copy(),objective
        except TrainingBudget:
            status="BUDGET_EXHAUSTED"
        history.append(dict(start=i,best_objective=best_value,
                            elapsed_seconds=time.monotonic()-tic))
        if path is not None:
            tmp=path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(coeff=(beta+best).tolist(),
               theta=best.tolist(),objective=best_value,status=status,
               completed_starts=len(history),history=history,
               wall_seconds=time.monotonic()-tic),indent=2))
            tmp.replace(path)
        if status!="COMPLETE_STARTS":break
    return beta+best
