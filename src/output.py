import json
import numpy as np
from datetime import date
from typing import Dict, List, Any

def init_output(saa_scenarios: int, num_horizons: int) -> Dict[str, Any]:
    return {
        "config": {
            "saa_scenarios": saa_scenarios,
            "NUM_HORIZONS":  num_horizons
        },
        "horizons": []
    }

def append_horizon(
    out: Dict[str, Any],
    horizon_idx: int,
    start_date: date,
    results: Dict[str, Dict[str, Any]]
) -> None:
    entry: Dict[str, Any] = {
        "horizon": horizon_idx,
        "start":   start_date.isoformat()
    }
    for tag, d in results.items():
        r, k = d["res"], d["kpi"]
        entry[tag] = {
            "status":    r["model"].Status,
            "planned":   r["obj"],
            "actual":    k["total_actual_cost"],
            "overtime":  k["overtime_min_total"],
            "idle":      k["idle_min_total"],
            "scheduled": k["scheduled"],
            "rejected":  k["rejected"],
            "runtime":   r["model"].Runtime,
        }
    out["horizons"].append(entry)

def print_console_summary(tags: List[str], out: Dict[str, Any]) -> None:
    def stats(xs):
        return f"mean={np.mean(xs):.0f}, median={np.median(xs):.0f}, min={np.min(xs):.0f}, max={np.max(xs):.0f}"
    print("\n=== Summary over horizons ===")
    for tag in tags:
        vals = out["horizons"]
        planned = [h[tag]["planned"] for h in vals]
        actual  = [h[tag]["actual"]  for h in vals]
        idle    = [h[tag]["idle"]    for h in vals]
        ot      = [h[tag]["overtime"] for h in vals]
        rt      = [h[tag]["runtime"] for h in vals]

        print(f"{tag} planned obj : {stats(planned)}")
        print(f"{tag} actual obj  : {stats(actual)}")
        print(f"{tag} idle        : {stats(idle)}")
        print(f"{tag} overtime    : {stats(ot)}")
        print(f"{tag} runtime     : {stats(rt)}")

def save_detailed(out: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Detailed results written to {path}")

def save_aggregate(
    out: Dict[str, Any],
    path: str,
    methods: List[str]
) -> None:
    agg = {
        "config":    out["config"],
        "aggregate": {}
    }
    n = len(out["horizons"])
    for tag in methods:
        def vals(field): return [h[tag][field] for h in out["horizons"]]
        agg_vals = {
            "total_planned":   sum(vals("planned")),
            "avg_planned":     sum(vals("planned")) / n,
            "total_actual":    sum(vals("actual")),
            "avg_actual":      sum(vals("actual"))  / n,
            "total_overtime":  sum(vals("overtime")),
            "avg_overtime":    sum(vals("overtime"))/ n,
            "total_idle":      sum(vals("idle")),
            "avg_idle":        sum(vals("idle"))    / n,
            "avg_runtime":     sum(vals("runtime")) / n,
        }
        agg["aggregate"][tag] = agg_vals

    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[INFO] Aggregated results written to {path}")
