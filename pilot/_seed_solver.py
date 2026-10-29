#!/usr/bin/env python3
"""Explicit-seed scheduling helper for the single structural VF method.

IMPORTANT: never use PlannerCache.solve for seed 1 or 2. Its worker historically
hardcoded hash_seed=0. This independent worker passes each seed explicitly to
the solver, checks provenance, and caches with the seed in the signature.
The solver receives the seed explicitly; no environment-variable workaround.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from pilot._joint_vf_engine import save


def explicit_job(job):
    key,c,d,seconds,seed=job
    from surgery.anchored_weekly_solver import solve_week
    result=solve_week(c,d,seconds=seconds,threads=1,tie=True,hash_seed=seed)
    z=(list(map(int,result.assignment))
       if result.assignment is not None else list(map(int,c.original)))
    return key,dict(assignment=z,complete=bool(result.complete),
                    status=str(result.status),hash_seed=seed,
                    balance_complete=bool(getattr(result,"balance_complete",False)),
                    balance_objective=getattr(result,"balance_objective",None),
                    balance_seconds=float(getattr(result,"balance_seconds",0.)))


def run_seed(items,duration,folder,label,seed,cores,seconds):
    """Per-component resumability with a duration- and seed-specific signature."""
    cache=folder/"explicit_plans"/("seed"+str(seed))/label
    cache.mkdir(parents=True,exist_ok=True)
    answers={};jobs=[]
    for key,c,rows in items:
        d=np.ascontiguousarray(np.asarray(duration[rows],float),dtype="<f8")
        signature=hashlib.sha256(
            str(key).encode()+d.tobytes()+
            repr((float(seconds),int(seed),"tie","zero_turnover_sqload_v1")).encode()).hexdigest()
        path=cache/(hashlib.sha256(str(key).encode()).hexdigest()+".json")
        if path.exists():
            record=json.loads(path.read_text())
            if (record.get("signature")==signature and
                    record.get("result",{}).get("hash_seed")==seed):
                answers[key]=record["result"]
                continue
        jobs.append(((key,c,d,seconds,seed),path,signature))
    print(label,"seed",seed,":",len(answers),"cached;",len(jobs),"solves",flush=True)
    def store(reply,path,signature):
        key,data=reply
        if data["hash_seed"]!=seed:
            raise AssertionError("Solver received incorrect hash seed")
        save(path,dict(signature=signature,result=data))
        answers[key]=data
    if cores<=1:
        for job,path,sig in jobs:
            store(explicit_job(job),path,sig)
    elif jobs:
        with ProcessPoolExecutor(max_workers=cores,
                                 mp_context=get_context("spawn")) as pool:
            pending={pool.submit(explicit_job,job):(path,sig)
                     for job,path,sig in jobs}
            for i,future in enumerate(as_completed(pending),1):
                path,sig=pending[future]
                store(future.result(),path,sig)
                if i%100==0 or i==len(pending):
                    print(label,"seed",seed,":",i,"/",len(pending),flush=True)
    return answers

