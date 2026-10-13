"""Certifiably scored VF-GAP learning with restricted-library *proposals*.

B_gamma(w) = gamma V(pred(w)) + max_z{J(z,actual)-gamma J(z,pred(w))}
             - V(actual).
V(actual) is constant and omitted during training.

The schedule-library objective is neither an upper nor lower bound on B_gamma.
It is used ONLY to propose updates. Every accepted update must strictly
decrease the FULL exact-oracle B_gamma over every training week.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp, expit

from .planner import display_and_duration
from .weekly_planner import solve_week, solve_adversary, evaluate
from .weekly_training import response_with_derivative, save_json


def array_hash(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def week_key(week):
    return week.group + "|" + week.monday


def assignment_physical(component, positions):
    return [[int(cid), int(component.slots[int(pos)][0]),
             str(component.slots[int(pos)][1])]
            for cid, pos in zip(component.case_ids, positions)]


def oracle_week(task):
    """Both exact oracles use one predicted duration vector and same feasible set."""
    week, duration, digest, gamma, seconds, threads = task
    values = {int(cid): float(d) for cid, d in zip(week.case_ids, duration)}
    planned, adversarial = [], []
    pred_total, adv_total, realized_total, statuses = 0., 0., 0., []
    for j, comp in enumerate(week.components()):
        d = np.asarray([values[int(i)] for i in comp.case_ids], float)
        p = solve_week(comp, d, seconds=seconds, threads=threads, tie=False)
        if not p.complete or p.assignment is None:
            return {"key": week_key(week), "input_sha256": digest,
                    "complete": False,
                    "failure": f"component {j}: planner status={p.status} "
                               f"seconds={p.seconds:.1f} bound={p.bound} value={p.value}"}
        a = solve_adversary(comp, d, gamma=gamma,
                            seconds=seconds, threads=threads)
        if not a.complete or a.assignment is None:
            return {"key": week_key(week), "input_sha256": digest,
                    "complete": False,
                    "failure": f"component {j}: adversary status={a.status} "
                               f"seconds={a.seconds:.1f} bound={a.bound} value={a.value}"}
        pv = evaluate(comp, p.assignment, d)["cost"]
        ac = evaluate(comp, a.assignment, comp.actual)["cost"]
        ap = evaluate(comp, a.assignment, d)["cost"]
        if abs(pv - p.value) > .1 or abs((ac - gamma * ap) - a.value) > .1:
            raise AssertionError("Exact oracle's solution cost reconstruction differs")
        realized = evaluate(comp, p.assignment, comp.actual)["cost"]
        theta = float((1.75*np.maximum(comp.actual-d,0.)
                       + np.maximum(d-comp.actual,0.)).sum())
        gap_term = gamma*pv + ac-gamma*ap
        if gap_term < realized-.02 or gap_term > pv+theta+.02:
            raise AssertionError(
                "VF-GAP bound chain violated: realized <= gap <= original VF")
        pred_total += pv
        adv_total += ac - gamma * ap
        realized_total += realized
        planned.extend(assignment_physical(comp, p.assignment))
        adversarial.extend(assignment_physical(comp, a.assignment))
        statuses.append([p.status, a.status])
    if set(int(t[0]) for t in planned) != set(map(int, week.case_ids)):
        raise AssertionError("Incomplete week planner assignment")
    return {"key": week_key(week), "input_sha256": digest,
            "complete": True, "planner_assignment": planned,
            "adversary_assignment": adversarial,
            "predicted_plan_cost": float(pred_total),
            "adversary_value": float(adv_total),
            "gap_without_hindsight": float(gamma*pred_total + adv_total),
            "realized_plan_cost": float(realized_total),
            "statuses": statuses}


def certify(root, filename, weeks, index, X, booked, w, alpha, h,
            gamma, seconds, threads, workers):
    """Incremental, provenance-bound exact evaluation; no partial-score acceptance."""
    path = Path(root) / filename
    _, durations = display_and_duration(X @ np.asarray(w, float), booked, alpha, h)
    identity = {"weights_sha256": array_hash(w), "gamma": float(gamma)}
    data = json.loads(path.read_text()) if path.exists() else {
        "identity": identity, "weeks": {}}
    if data.get("identity") != identity:
        raise ValueError(f"Checkpoint {path.name} belongs to a different policy/gamma")
    state = data["weeks"]
    pending = []
    for week in weeks:
        loc = np.array([index[int(cid)] for cid in week.case_ids], dtype=int)
        d = np.asarray(durations[loc], float)
        digest = array_hash(d)
        row = state.get(week_key(week))
        if row and row.get("complete") and row.get("input_sha256") == digest:
            continue
        pending.append((week, d, digest, gamma, seconds, threads))
    print(f"{filename}: {len(pending)} of {len(weeks)} group-weeks pending, "
          f"{workers} worker(s) x {threads} solver threads", flush=True)

    def persist(row):
        state[row["key"]] = row
        save_json(path, data)

    if workers == 1:
        for task in pending:
            persist(oracle_week(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers,len(pending)),
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(oracle_week, task) for task in pending]
            for future in as_completed(futures):
                persist(future.result())
    bad = [week_key(w) for w in weeks if not
           state.get(week_key(w), {}).get("complete")]
    if bad:
        print("Examples of failures:", [(k, state.get(k, {}).get("failure"))
                                      for k in bad[:4]], flush=True)
        raise RuntimeError(f"{len(bad)} uncertified group-weeks in {filename}; "
                           "resume with higher --seconds")
    return data


def certified_score(result, weeks, w, lam, n):
    rows = result["weeks"]
    if len(rows) != len(weeks) or any(
            not rows[week_key(t)].get("complete") for t in weeks):
        raise ValueError("All training group-weeks require certified exact oracles")
    gap = sum(rows[week_key(t)]["gap_without_hindsight"] for t in weeks) / n
    realized = sum(rows[week_key(t)]["realized_plan_cost"] for t in weeks) / n
    regularizer = lam * float(np.abs(np.asarray(w)[1:]).sum())
    return {"gap_without_hindsight_per_case": float(gap),
            "realized_plan_cost_per_case": float(realized),
            "regularization_per_case": float(regularizer),
            "regularized_gap_without_hindsight_per_case": float(gap+regularizer)}


def _physical_matrix(week, physical, index, n):
    lookup = {int(cid): (int(day), str(room)) for cid, day, room in physical}
    if len(lookup) != week.n or set(lookup) != set(map(int, week.case_ids)):
        raise ValueError("Incomplete or duplicate physical assignment")
    slots = {key: j for j, key in enumerate(week.slots)}
    assigned = np.empty(week.n, dtype=int)
    for pos, cid in enumerate(week.case_ids):
        if lookup[int(cid)] not in slots:
            raise ValueError("Invalid physical slot in stored schedule")
        assigned[pos] = slots[lookup[int(cid)]]
    # Evaluate validates eligibility, required surgeon days, and room occupancy.
    evaluate(week, assigned, week.actual)
    rows = assigned
    cols = np.array([index[int(i)] for i in week.case_ids], dtype=int)
    mat = csr_matrix((np.ones(week.n), (rows, cols)), shape=(week.k, n))
    counts = np.bincount(rows, minlength=week.k)
    offset = 30. * np.maximum(counts - 1, 0)
    return mat, offset


def room_cost_and_derivative(load, eps):
    e = np.asarray(load) - 480.
    if eps == 0:
        values = 1.75 * np.maximum(e, 0.) + np.maximum(-e, 0.)
        slope = np.where(e > 0, 1.75, np.where(e < 0, -1., .375))
    else:
        values = eps * (1.75 * np.logaddexp(0.,e/eps)
                        + np.logaddexp(0.,-e/eps))
        slope = 1.75*expit(e/eps) - expit(-e/eps)
    return values, slope


class ScheduleLibrary:
    """Sparse cost library. Both min and max are restricted: NEVER certify with this."""

    def __init__(self, weeks, checkpoints, index, n, gamma):
        self.gamma, self.n = float(gamma), int(n)
        self.schedule_types = {}
        self.all_keys = [week_key(w) for w in weeks]
        matrices, offsets, week_number, types, actual_cost = [], [], [], [], []
        seen = set()
        for week_i, week in enumerate(weeks):
            for checkpoint in checkpoints:
                record = checkpoint["weeks"][week_key(week)]
                for typ, field in ((0, "planner_assignment"),
                                   (1, "adversary_assignment")):
                    physical = record[field]
                    key = (week_key(week), typ,
                           tuple(sorted((int(c),int(d),str(r)) for c,d,r in physical)))
                    if key in seen:
                        continue
                    seen.add(key)
                    mat, offset = _physical_matrix(week, physical, index, n)
                    matrices.append(mat)
                    offsets.extend(offset.tolist())
                    week_number.append(week_i)
                    types.append(typ)
                    actual_cost.append(float(
                        (room_cost_and_derivative(mat @ week_actual_vector(
                            week,index,n) + offset, 0.)[0]).sum()))
        from scipy.sparse import vstack
        if not matrices:
            raise ValueError("Empty schedule library")
        self.M = vstack(matrices, format="csr")
        self.offset = np.asarray(offsets, float)
        self.actual = np.asarray(actual_cost, float)
        self.types = np.asarray(types, int)
        self.week_number = np.asarray(week_number, int)
        self.schedule_room_slices = np.cumsum([0] + [m.shape[0] for m in matrices])
        self.room_schedule = np.repeat(
            np.arange(len(matrices)),
            np.diff(self.schedule_room_slices))
        self.candidates = []
        for i in range(len(weeks)):
            p = np.flatnonzero((self.week_number == i) & (self.types==0))
            a = np.flatnonzero((self.week_number == i) & (self.types==1))
            if len(p)==0 or len(a)==0:
                raise AssertionError(f"No planner/adversary schedule for {week_key(weeks[i])}")
            self.candidates.append((p,a))
        self.schedule_count = len(matrices)

    def objective_gradient(self, w, X, booked, alpha, h, lam,
                           temp=1.5, room_eps=1.5, l1_eps=.02):
        w=np.asarray(w,float)
        duration, dr = response_with_derivative(X@w,booked,alpha,h)
        load=self.M@duration+self.offset
        room_value, room_grad=room_cost_and_derivative(load,room_eps)
        per_schedule=np.bincount(self.room_schedule,weights=room_value,
                                 minlength=self.schedule_count)
        schedule_weight=np.zeros(self.schedule_count)
        total=0.
        for p,a in self.candidates:
            # Restricted library is ONLY a proposal model, not an upper bound.
            xp=-per_schedule[p]/temp
            xa=(self.actual[a]-self.gamma*per_schedule[a])/temp
            zp=logsumexp(xp)
            za=logsumexp(xa)
            total+=self.gamma*(-temp*zp)+temp*za
            schedule_weight[p]+=self.gamma*np.exp(xp-zp)
            schedule_weight[a]-=self.gamma*np.exp(xa-za)
        room_coeff=schedule_weight[self.room_schedule] * room_grad
        dldd=np.asarray(self.M.T@room_coeff,float)
        coef=w[1:]
        smooth=np.sqrt(coef**2+l1_eps**2)
        total=total/self.n+lam*float(smooth.sum())
        grad=X.T@(dldd*dr)/self.n
        grad[1:]+=lam*coef/smooth
        return float(total), np.asarray(grad,float)


def week_actual_vector(week, index, n):
    a=np.zeros(n)
    a[np.asarray([index[int(cid)] for cid in week.case_ids],dtype=int)]=week.actual
    return a


def propose(library, current, X, booked, alpha, h, lam,
            trust, iterations, temperature):
    current=np.asarray(current,float)
    bounds=[(max(-150.,float(v)-trust), min(150.,float(v)+trust))
            for v in current]
    result=minimize(library.objective_gradient,current,
                    args=(X,booked,alpha,h,lam,temperature,1.5),
                    method="L-BFGS-B",jac=True,bounds=bounds,
                    options={"maxiter":iterations,"ftol":1e-10,"maxls":35})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("Nonfinite schedule-library proposal")
    return np.asarray(result.x,float), {
        "proposal_objective": float(result.fun),
        "proposal_iterations": int(result.nit),
        "proposal_success": bool(result.success),
        "proposal_message": str(result.message),
        "proposal_max_coefficient_change": float(
            np.max(np.abs(result.x-current))),
        "library_schedules": library.schedule_count,
    }


def train_gap(root, weeks, index, X, booked, lam, start, settings,
              alpha, h, gamma, max_outer, inner_iterations,
              trust, temperature, seconds, threads, workers):
    root=Path(root)
    progress_file=root/"gap_progress.json"
    state=json.loads(progress_file.read_text()) if progress_file.exists() else {
        "settings":settings,"w":list(map(float,start)),"history":[],"complete":False}
    if state["settings"] != settings:
        raise ValueError("Different VF-GAP settings in existing output directory")
    if state.get("complete"):
        print("VF-GAP training already complete; see",progress_file,flush=True)
        return state
    n=len(booked)
    reason="maximum_outer_iterations"
    for outer in range(len(state["history"]),max_outer):
        current=np.asarray(state["w"],float)
        current_file=f"gap_current_{outer:02}.json"
        current_result=certify(root,current_file,weeks,index,X,booked,current,
                               alpha,h,gamma,seconds,threads,workers)
        old=certified_score(current_result,weeks,current,lam,n)
        checkpoints=[json.loads((root/f"gap_current_{k:02}.json").read_text())
                     for k in range(outer+1)]
        library=ScheduleLibrary(weeks,checkpoints,index,n,gamma)
        proposal,meta=propose(library,current,X,booked,alpha,h,lam,
                              trust,inner_iterations,temperature)
        print(f"VF-GAP round {outer+1}: certified start={old['regularized_gap_without_hindsight_per_case']:.6f}; "
              f"library has {meta['library_schedules']} schedules",flush=True)
        selected=None
        candidates=[]
        for j,factor in enumerate((1.,.5,.25)):
            w=current+factor*(proposal-current)
            if np.allclose(w,current,atol=1e-9):
                continue
            filename=f"gap_candidate_{outer:02}_{j}.json"
            cert=certify(root,filename,weeks,index,X,booked,w,
                         alpha,h,gamma,seconds,threads,workers)
            scored=certified_score(cert,weeks,w,lam,n)
            candidates.append({"fraction":factor,"certificate_file":filename,
                               "score":scored})
            print(f"  factor {factor:g}: full exact GAP={scored['regularized_gap_without_hindsight_per_case']:.6f}, "
                  f"realized plan cost/case={scored['realized_plan_cost_per_case']:.6f}",flush=True)
            if (scored["regularized_gap_without_hindsight_per_case"]
                    < old["regularized_gap_without_hindsight_per_case"]-1e-5):
                selected=(w,cert,scored,filename,factor)
                break  # first exact-certified improvement; do not waste extra full solves
        accepted=selected is not None
        if accepted:
            w,cert,score,filename,factor=selected
            state["w"]=w.tolist()
            # Preserves fully certified exact scores and schedules for the
            # next outer step; no duplicated oracle solves required.
            next_path=root/f"gap_current_{outer+1:02}.json"
            if next_path.exists():
                if json.loads(next_path.read_text()) != cert:
                    raise ValueError("Unexpected incompatible next-round certificate")
            else:
                save_json(next_path,cert)
        state["history"].append({
            "outer":outer+1,"certified_before":old,
            "candidates":candidates, "proposal":meta,
            "accepted":accepted,
            "accepted_fraction":float(selected[4]) if accepted else None,
            "certified_after":selected[2] if accepted else old,
            "train_weeks":len(weeks)})
        save_json(progress_file,state)
        if not accepted:
            reason="no_full_oracle_gap_improvement"
            break
        relative=(old["regularized_gap_without_hindsight_per_case"]-selected[2][
            "regularized_gap_without_hindsight_per_case"])/max(1.,abs(old[
            "regularized_gap_without_hindsight_per_case"]))
        if relative<1e-4:
            reason="small_full_oracle_gap_improvement"
            break
    state.update({"complete":True,"stopping":reason,
                  "method":"VF_GAP","gamma":float(gamma),
                  "objective_definition":"exact gamma*predicted planning optimum "
                      "+ exact max(actual-gamma*predicted) + lambda L1, "
                      "divided by training cases; realized oracle constant omitted",
                  "optimization_status":"locally improved certified VF-GAP; "
                      "no global optimality certificate"})
    save_json(progress_file,state)
    return state
