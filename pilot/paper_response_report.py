#!/usr/bin/env python3
"""Resumable evaluation of fitted joint VF under response misspecification.

Training-set scenario costs choose the minimax-regret policy; 2013 is used
only for exploratory reporting. Advice is frozen under its training scenario.
No surgeon response observations exist: heterogeneity is a stated stress test.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np

from pilot.run_nonmonotone_learning import five_features,save
from pilot.run_response_scenario_grid import (
    ContentCache,freeze_displays)
from surgery.response_grid import (
    TRAIN_SCENARIOS,EVAL_SCENARIOS,implemented_from_display,
    verify_same_scenario,mapping)
from surgery.weekly_planner import evaluate

POLICIES=tuple("VF__"+s.name for s in TRAIN_SCENARIOS)
NOMINAL="VF__SAT_A80_H30"


def csv_rows(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    temp.replace(path)


def case_displays(ctx,root,cache,policies):
    """Produce one fixed training display vector per fitted VF policy."""
    data=ctx["fit"];items=ctx["train"]
    b=np.asarray(data["booked"],float)
    base=np.asarray(data["X"],float)
    plans=cache.solve(items,b,"paper_train_booked",tie=True)
    answers={}
    for scenario in TRAIN_SCENARIOS:
        entry=policies[scenario.name]
        scaler=json.loads((root/"training"/scenario.name/"vf_features.json").read_text())
        H,_,_=five_features(ctx,items,cache,"train_paper_matrix",
            scaler=(np.asarray(scaler["mean"]),np.asarray(scaler["std"])),
            plans=plans,schema=scaler.get("schema","baseline"))
        theta=np.asarray(entry["vf_theta"],float)
        X=np.column_stack([base,H])
        if X.shape[1]!=len(theta):
            raise ValueError("Joint VF dimension mismatch in training matrix")
        answers["VF__"+scenario.name]=verify_same_scenario(
            X@theta,b,scenario)
    return answers


def evaluate_training_matrix(ctx,root,cache,policies):
    """All four homogeneous scenarios x four VF policies on train components.

    The certified-case set is intersected across all cells before selecting
    a policy. Keep each cell on disk; this stage can be safely resumed.
    """
    items=ctx["train"];b=np.asarray(ctx["fit"]["booked"],float)
    displays=case_displays(ctx,root,cache,policies)
    folder=root/"paper_training_cells"
    for scenario in TRAIN_SCENARIOS:
        for label in ("BOOKED",*POLICIES):
            target=folder/scenario.name/(label+".json")
            if target.exists():
                continue
            duration=(b if label=="BOOKED" else
                      implemented_from_display(b,displays[label],scenario))
            results=cache.solve(items,duration,"paper_train_"+scenario.name+"_"+label,tie=True)
            entries=[]
            for key,week,rows in items:
                rec=results[key]
                score=evaluate(week,rec["assignment"],week.actual)
                entries.append(dict(component=key,week=week.monday,cases=week.n,
                    cost=float(score["cost"]),complete=bool(rec["complete"]),
                    planner_status=rec["status"]))
            save(target,dict(scenario=scenario.name,policy=label,entries=entries))
    return matrix_from_cells(folder,TRAIN_SCENARIOS,("BOOKED",*POLICIES))


def matrix_from_cells(folder,scenarios,methods):
    cells={};common=None
    for s in scenarios:
        for label in methods:
            rows=json.loads((folder/s.name/(label+".json")).read_text())["entries"]
            idx={r["component"]:r for r in rows}
            cells[s.name,label]=idx
            good={k for k,v in idx.items() if v["complete"]}
            common=(good if common is None else common&good)
    if not common:
        raise RuntimeError("No common fully certified components")
    keys=sorted(common)
    reference=cells[scenarios[0].name,methods[0]]
    cases=sum(reference[k]["cases"] for k in keys)
    if cases<=0:
        raise AssertionError("Empty common-case set")
    result={s.name:{m:sum(cells[s.name,m][k]["cost"] for k in keys)/cases
                      for m in methods} for s in scenarios}
    return dict(costs=result,certified_components=len(keys),
                certified_cases=cases,complete=(len(keys)==len(reference)))


def candidate_regret(matrix):
    return {s:{p:matrix[s][p]-min(matrix[s][q] for q in POLICIES)
               for p in POLICIES} for s in matrix}


def selected_policy(matrix,mode):
    regrets=candidate_regret(matrix)
    if mode=="minimax":
        return min(POLICIES,key=lambda p:(max(regrets[s][p] for s in matrix),
                                          sum(regrets[s][p] for s in matrix),p))
    if mode=="average":
        return min(POLICIES,key=lambda p:(sum(matrix[s][p] for s in matrix),
                                          p))
    raise ValueError(mode)


def stable_uniform(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8],"big")/2**64


def heterogeneous_duration(items,booked,display,profile):
    """Fixed surgeon-specific response types; NOT estimated from the UHN data."""
    ans=np.empty_like(booked)
    for key,week,rows in items:
        rows=np.asarray(rows,int)
        labels=[]
        for i in range(week.n):
            surgeon_id=week.surgeon_ids[int(week.surgeon[i])]
            u=stable_uniform((str(week.group),str(surgeon_id)))
            if profile=="equal_four":
                ix=min(int(4*u),3)
            elif profile=="nominal_majority":
                ix=2 if u<.75 else (0 if u<.8333333333333333 else
                                    (1 if u<.9166666666666666 else 3))
            else:
                raise ValueError(profile)
            labels.append(ix)
        z=np.asarray(labels,int)
        for ix,sc in enumerate(TRAIN_SCENARIOS):
            mask=z==ix
            if mask.any():
                ans[rows[mask]]=implemented_from_display(
                    booked[rows[mask]],display[rows[mask]],sc)
    if not np.isfinite(ans).all():
        raise AssertionError("Heterogeneous adoption has nonfinite durations")
    return ans


def heterogeneity(ctx,root,cache,displays):
    items=ctx["eval"];booked=np.asarray(ctx["held"]["booked"],float)
    profiles=("equal_four","nominal_majority")
    folder=root/"paper_heterogeneity_cells"
    for profile in profiles:
        for label in ("BOOKED",*POLICIES):
            file=folder/profile/(label+".json")
            if file.exists():
                continue
            durations=(booked if label=="BOOKED" else
                heterogeneous_duration(items,booked,displays[label],profile))
            plans=cache.solve(items,durations,"paper_hetero_"+profile+"_"+label,tie=True)
            records=[]
            for key,c,rows in items:
                p=plans[key];sc=evaluate(c,p["assignment"],c.actual)
                records.append(dict(component=key,week=c.monday,cases=c.n,
                    cost=float(sc["cost"]),complete=bool(p["complete"]),
                    planner_status=p["status"]))
            save(file,dict(profile=profile,policy=label,entries=records))
    class Label:
        def __init__(self,name):self.name=name
    return matrix_from_cells(folder,[Label(p) for p in profiles],
                             ("BOOKED",*POLICIES))


def eval_2013_matrix(root):
    """The existing scenario grid already uses globally common certified cases."""
    data=json.loads((root/"summary.json").read_text())
    if "global_common_grid" not in data:
        raise RuntimeError("No globally matched 2013 scenario matrix")
    return {s:{p:float(rec[p]["cost_per_case"])
               for p in ("BOOKED",*POLICIES)}
            for s,rec in data["global_common_grid"].items()},data


def summarize(root,train,test,hetero,grid):
    rc=train["costs"]
    minimax=selected_policy(rc,"minimax")
    average=selected_policy(rc,"average")
    gr=candidate_regret(rc)
    rtest=candidate_regret(test)
    rhetero=candidate_regret(hetero["costs"])
    candidate=[NOMINAL,minimax,average]
    rules={"nominal":NOMINAL,"training_minimax_regret":minimax,
           "training_equal_weight_average":average}
    table=[]
    for s in EVAL_SCENARIOS:
        name=s.name
        row=[name]
        for p in POLICIES:
            row.append(test[name][p])
        table.append(row)
    csv_rows(root/"paper_2013_vf_cost_matrix.csv",
             ["evaluation_scenario",*POLICIES],table)
    csv_rows(root/"paper_2013_vf_regret_matrix.csv",
             ["evaluation_scenario",*POLICIES],
             [[s,*[rtest[s][p] for p in POLICIES]]
              for s in test])
    csv_rows(root/"paper_train_vf_cost_matrix.csv",
             ["evaluation_scenario",*POLICIES],
             [[s,*[rc[s][p] for p in POLICIES]] for s in rc])
    csv_rows(root/"paper_train_vf_regret_matrix.csv",
             ["evaluation_scenario",*POLICIES],
             [[s,*[gr[s][p] for p in POLICIES]] for s in rc])
    rows=[]
    for rule,p in rules.items():
        rows.append(dict(selection=rule,trained_policy=p,
            train_max_regret=max(gr[s][p] for s in rc),
            train_mean_cost=np.mean([rc[s][p] for s in rc]),
            test_mean_cost=np.mean([test[s][p] for s in test]),
            test_max_candidate_regret=max(rtest[s][p] for s in test),
            test_worst_minus_booked=max(test[s][p]-test[s]["BOOKED"]
                                        for s in test),
            test_scenarios_better_than_booked=sum(test[s][p]<test[s]["BOOKED"]
                                                  for s in test)))
    csv_rows(root/"paper_policy_selection.csv",list(rows[0]),
             [[x[k] for k in rows[0]] for x in rows])
    hr=[]
    for profile in hetero["costs"]:
        costs=hetero["costs"][profile]
        hr.append(dict(profile=profile,
            booked=costs["BOOKED"],
            nominal_vf=costs[NOMINAL],
            minimax_vf=costs[minimax],
            best_of_four=min(costs[p] for p in POLICIES),
            nominal_regret=rhetero[profile][NOMINAL],
            minimax_regret=rhetero[profile][minimax]))
    csv_rows(root/"paper_heterogeneity.csv",list(hr[0]),
             [[x[k] for k in hr[0]] for x in hr])
    report=dict(selection=rules,selection_table=rows,
        reference_population="Four VF candidate policies only",
        selection_based_on="training-period 2011-2012 scenario costs",
        uncertainty_probabilities="None estimated; equal weights are illustrative",
        evaluation="Exploratory 2013, repeatedly examined during method development",
        train_certified=train["certified_components"],
        train_cases=train["certified_cases"],
        training_matrix_complete=train["complete"],
        test_certified=grid.get("all_scenarios_common_certified_components"),
        test_cases=grid.get("all_scenarios_common_certified_cases"),
        heterogeneity_certified=hetero["certified_components"],
        heterogeneity_cases=hetero["certified_cases"],
        heterogeneity=hr)
    save(root/"paper_response_summary.json",report)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context",type=Path,required=True)
    ap.add_argument("--grid",type=Path,required=True)
    ap.add_argument("--cores",type=int,default=15)
    ap.add_argument("--plan-seconds",type=float,default=180.)
    args=ap.parse_args()
    root=args.grid
    with args.context.open("rb") as f:ctx=pickle.load(f)
    fitted=json.loads((root/"selected_policies.json").read_text())["policies"]
    for s in TRAIN_SCENARIOS:
        x=fitted[s.name]
        if x.get("vf_features")!="clean":
            raise ValueError("Unclean VF schema in response matrix "+s.name)
    cache=ContentCache(root,args.cores,args.plan_seconds)
    train=evaluate_training_matrix(ctx,root,cache,fitted)
    test,grid=eval_2013_matrix(root)
    naive=json.loads((root/"naive_policy.json").read_text())["weights"]
    displays=freeze_displays(ctx,root,cache,np.asarray(naive,float),fitted)
    hetero=heterogeneity(ctx,root,cache,displays)
    result=summarize(root,train,test,hetero,grid)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
