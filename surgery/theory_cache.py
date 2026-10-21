"""Exact deployed-planner cache for repeated candidate evaluations.

Caches only certified deterministic hash-selected assignments, never costs,
scenario samples, incomplete solves or approximate assignments. The calling
experiment's manifest fingerprints the source and data; its sqlite file must
live within that output root and is not portable across specifications.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import numpy as np


def connect(path):
    conn=sqlite3.connect(str(path),timeout=120.,isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("CREATE TABLE IF NOT EXISTS deployments "
                 "(signature TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    return conn


def signature(component, durations, seeds):
    d=np.asarray(durations,dtype=np.float64)
    if d.ndim!=1 or d.size!=component.n or not np.isfinite(d).all():
        raise ValueError("Invalid duration vector for cached deployment")
    content=pickle.dumps((1,component,d.tobytes(),tuple(seeds)),protocol=4)
    return hashlib.sha256(content).hexdigest()


def cached_deploy(component, durations, seeds, certify, conn=None, stats=None):
    """Return only fully verified assignments; cache hits are checked exactly."""
    d=np.asarray(durations,dtype=np.float64)
    sig=signature(component,d,seeds)
    if conn is not None:
        row=conn.execute("SELECT payload FROM deployments WHERE signature=?",
                         (sig,)).fetchone()
        if row is not None:
            value=json.loads(row[0])
            if value["seeds"]!=list(seeds) or not np.array_equal(
                    np.asarray(value["durations"],float),d):
                raise AssertionError("Planner cache identity mismatch")
            if stats is not None:stats["hits"]=stats.get("hits",0)+len(seeds)
            return {int(seed):tuple(map(int,value["assignments"][str(seed)]))
                    for seed in seeds}
    assignments={}
    for seed in seeds:
        assignments[int(seed)]=tuple(map(int,certify(component,d,int(seed))))
        if stats is not None:stats["solves"]=stats.get("solves",0)+1
    if conn is not None:
        payload=json.dumps(dict(seeds=list(seeds),durations=d.tolist(),
                assignments={str(k):list(v) for k,v in assignments.items()}),
                allow_nan=False,separators=(",",":"))
        conn.execute("INSERT OR IGNORE INTO deployments VALUES (?,?)",(sig,payload))
    return assignments
