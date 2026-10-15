"""Exact physical-room MILP oracles for anchored-regret training.

Fast separation optimizes a globally valid room-fixed-slope underestimator;
exact separation optimizes schedule-specific supporting tangents at beta.
Both use *exact* realized overtime disjunctions. No approximated infeasible
assignment is ever admitted to the schedule library.
"""
import time
import numpy as np

from .anchored_core import OVER, TURN, CAP, fixed_slope_lower, phi
from .weekly_planner import evaluate, loads, _cost, _assignment
from .anchored_weekly_solver import solve_week, _model, solve_adversary


def reconcile_maximization_bounds(checked, solver_objective, solver_bound):
    """Audit floating-point incumbent reconstruction against a MAX dual bound.

    Gurobi's floating-point ObjBound can differ slightly from an independently
    reconstructed feasible cost. The same tolerance used for reconstruction
    must be used for the bound consistency check; otherwise valid solves may
    crash on differences of 1e-6. Never conceal material disagreement or
    label it certified: return an invalid audit and no usable upper bound.
    """
    checked=float(checked)
    incumbent=float(solver_objective)
    dual=float(solver_bound)
    values=(checked,incumbent,dual)
    if not all(np.isfinite(values)):
        return {'valid':False,'reason':'nonfinite_objective_or_bound',
                'checked':checked,'solver_objective':incumbent,
                'raw_dual_bound':dual,'upper':None}
    tolerance=max(0.02,2e-6*max(abs(checked),abs(incumbent),abs(dual)))
    reconstruction_error=checked-incumbent
    raw_bound_shortfall=max(0.,checked-dual)
    audit={'valid':True,'checked':checked,'solver_objective':incumbent,
           'raw_dual_bound':dual,'objective_reconstruction_error':reconstruction_error,
           'raw_dual_bound_shortfall':raw_bound_shortfall,
           'numerical_tolerance':tolerance}
    if abs(reconstruction_error)>tolerance:
        audit.update(valid=False,reason='objective_reconstruction_mismatch',
                     upper=None)
    elif raw_bound_shortfall>tolerance:
        audit.update(valid=False,reason='material_dual_bound_violation',
                     upper=None)
    else:
        upper=max(dual,checked,incumbent)
        audit.update(reason='numerically_reconciled' if
                     raw_bound_shortfall>1e-6 else 'consistent',
                     upper=upper,dual_adjustment=upper-dual)
    return audit


def separate(component, d, beta, gamma, *, slopes=None, exact=False,
             seconds=120., threads=1, min_realized_cost=None):
    """Max J(z;actual)-gamma*T(z;d), including primary room opening.

    For exact=False, slopes[j] must be given for each physical room (0..2.75).
    When exact=True, the oracle may select a valid kink subgradient, which is
    stored with the returned cut. Returns solver *maximization* primal and dual.
    """
    import gurobipy as gp
    from gurobipy import GRB
    d = np.asarray(d,float)
    beta = np.asarray(beta,float)
    if d.shape != (component.n,) or beta.shape != d.shape or not np.isfinite(d).all():
        raise ValueError('Incorrect duration dimension')
    if exact:
        slopes=None
    else:
        slopes=np.asarray(slopes,float)
        if slopes.shape!=(component.k,) or np.any(slopes < -1e-12) or np.any(slopes > OVER+1e-12):
            raise ValueError('Invalid fixed room slopes')
    if all(len(component.allowed[int(s)])==1 for s in component.surgeon):
        z=component.original.astype(int).tolist()
        if exact:
            from .anchored_core import exact_tangent, slopes_from_reference
            slope=slopes_from_reference(component,z,beta)
            lower=exact_tangent(component,z,beta,d)
        else:
            slope=slopes
            lower=fixed_slope_lower(component,z,d,slopes)
        value=evaluate(component,z,component.actual)['cost']-gamma*lower
        return {'complete':True,'status':'ANALYTIC','seconds':0.,'assignment':z,
                'slopes':slope.tolist(),'lower':value,'upper':value}
    start=time.monotonic()
    m,x,used,byroom=_model(component,seconds,threads)
    try:
        realized=_cost(m,x,used,byroom,component.actual,exact=True,name='anchor_actual')
        if min_realized_cost is not None:
            if not np.isfinite(min_realized_cost):
                raise ValueError('Nonfinite realized-cost restriction')
            m.addConstr(realized >= float(min_realized_cost)-1e-6,
                        name='non_dominating_competitor')
        if not exact:
            lower=TURN*used.sum()+gp.quicksum(
                float(slopes[j]) * (
                    gp.quicksum((float(d[i])+TURN)*x[i,j] for i in cases)
                    -(TURN+CAP)*used[j]
                ) for j,cases in enumerate(byroom) if slopes[j]>0)
        else:
            tangents=[]
            high=[]
            for j,cases in enumerate(byroom):
                h=m.addVar(vtype=GRB.BINARY,name=f'beta_high_{j}')
                m.addConstr(h<=used[j])
                excess_beta=(gp.quicksum((float(beta[i])+TURN)*x[i,j] for i in cases)
                             -(TURN+CAP)*used[j])
                excess_d=(gp.quicksum((float(d[i])+TURN)*x[i,j] for i in cases)
                          -(TURN+CAP)*used[j])
                m.addGenConstrIndicator(h,0,excess_beta<=0.)
                m.addGenConstrIndicator(h,1,excess_beta>=0.)
                ub=sum(float(d[i])+TURN for i in cases)+TURN+CAP
                v=m.addVar(lb=-ub,ub=ub,name=f'beta_tangent_{j}')
                m.addGenConstrIndicator(h,0,v==0.)
                m.addGenConstrIndicator(h,1,v==excess_d)
                tangents.append(v)
                high.append(h)
            lower=TURN*used.sum()+OVER*gp.quicksum(tangents)
        m.setObjective(realized-gamma*lower,GRB.MAXIMIZE)
        m.optimize()
        if not m.SolCount:
            return {'complete':False,'status':str(m.Status),'seconds':time.monotonic()-start,
                    'assignment':None,'lower':None,'upper':float(m.ObjBound) if m.Status==GRB.TIME_LIMIT else None}
        z=_assignment(component,x)
        slope=np.array([OVER*(v.X>.5) for v in high]) if exact else slopes
        checked=(evaluate(component,z,component.actual)['cost']
                 -gamma*fixed_slope_lower(component,z,d,slope))
        audit=reconcile_maximization_bounds(checked,m.ObjVal,m.ObjBound)
        if not audit['valid']:
            # Keep the independently reconstructed feasible assignment,
            # but NEVER issue a certificate or a usable upper bound.
            # Training records this result as unresolved and can resume.
            return {'complete':False,'status':'NUMERICAL_AUDIT_FAILED',
                    'seconds':time.monotonic()-start,'assignment':z,
                    'slopes':slope.tolist(),'lower':checked,'upper':None,
                    'numerical_audit':audit}
        return {'complete':bool(m.Status==GRB.OPTIMAL),
                'status':str(m.Status),
                'seconds':time.monotonic()-start,'assignment':z,
                'slopes':slope.tolist(),'lower':checked,
                'upper':audit['upper'],'numerical_audit':audit}
    finally:
        m.dispose()


def deployed(component, durations, *, seconds=300., threads=1,
             hindsight=False):
    """Exact deterministic deployment (tie=True); optional hindsight primary."""
    planned=solve_week(component,durations,seconds=seconds,threads=threads,tie=not hindsight)
    if planned.assignment is None:
        return {'complete':False,'status':planned.status,'assignment':None,
                'planner':planned.as_dict()}
    a=list(map(int,planned.assignment))
    cp=evaluate(component,a,durations)['cost']
    real=evaluate(component,a,component.actual)
    ca=real['cost']
    room_loads=loads(component,a,component.actual)
    return {'complete':bool(planned.complete),'status':planned.status,
            'planner':planned.as_dict(),'assignment':a,'predicted_cost':cp,'realized_cost':ca,
            'realized_overtime_minutes':real['overtime'],
            'realized_idle_minutes':real['idle'],
            'moved_cases':real['moved_cases'],
            'room_days_over_720':real['room_days_over_720'],
            'room_days_over_960':int(np.count_nonzero(room_loads>960.)),
            'room_days_over_1440':int(np.count_nonzero(room_loads>1440.)),
            'max_realized_room_load':real['max_room_load'],
            'primary_bound':planned.bound,'seconds':planned.seconds}


def certified_bound(component, anchor, duration, oracle_cost, deployed_cost,
                    gamma, *, seconds=300.,threads=1,exact=True,report_gap=False,
                    deployed_record=None,tight_surrogate=False):
    """A rigorous tie-safe anchored upper bound, even with an incomplete MAX.

    A finite MAX dual upper bound is required; otherwise no claim is made.
    Delta of deployed primary cost is bounded using an exact MIN lower bound.
    """
    if deployed_record is None:
        plan=solve_week(component,duration,seconds=seconds,threads=threads,tie=True)
    else:
        # A complete result from the EXACT SAME duration-keyed verification
        # cache is a deployment certificate, not just an assignment warm-start.
        import os
        from .anchored_weekly_solver import AnchoredResult, hash_score
        if not deployed_record.get('complete') or 'planner' not in deployed_record:
            raise ValueError('Cannot reuse an incomplete deployment record')
        payload=deployed_record['planner']
        if not (payload.get('complete') and payload.get('primary_complete')
                and payload.get('tie_complete') and payload.get('hash_unique')):
            raise ValueError('Missing primary/hash/uniqueness deployment certificate')
        expected_seed=int(os.getenv('SURGERY_ANCHORED_HASH_SEED','0'))
        if int(payload.get('hash_seed',-1)) != expected_seed:
            raise ValueError('Hash seed mismatch in reused deployment record')
        if payload.get('assignment') != deployed_record.get('assignment'):
            raise ValueError('Planner and deployment assignment mismatch')
        if hash_score(component,payload['assignment'],expected_seed) != payload.get('hash_score'):
            raise ValueError('Reused hash score mismatch')
        checked=evaluate(component,payload['assignment'],duration)['cost']
        if (abs(checked-float(payload['value']))>1e-4
                or abs(checked-float(deployed_record['predicted_cost']))>1e-4):
            raise ValueError('Reused predicted-cost reconstruction mismatch')
        checked_real=evaluate(component,payload['assignment'],component.actual)['cost']
        if abs(checked_real-float(deployed_record['realized_cost']))>1e-4:
            raise ValueError('Reused realized-cost reconstruction mismatch')
        if (payload.get('bound') is None or
                not np.isfinite(payload['bound']) or
                float(payload['bound'])>checked+1e-4):
            raise ValueError('Invalid reused primary lower bound')
        plan=AnchoredResult(**payload)
    if plan.assignment is None or plan.bound is None or not np.isfinite(plan.bound):
        return {'certified':False,'reason':'no planner primal+dual'}
    if not plan.complete:
        return {'certified':False,'reason':'planner hash deployment incomplete'}
    oracle=separate(component,duration,anchor['beta'],gamma,
                    slopes=anchor['slopes'],exact=exact,seconds=seconds,threads=threads,
                    min_realized_cost=(evaluate(component,anchor['assignment'],component.actual)['cost']
                                       if tight_surrogate else None))
    if oracle['upper'] is None or not np.isfinite(oracle['upper']):
        return {'certified':False,'reason':'no separation dual','oracle':oracle}
    phi_anchor=phi(component,anchor['assignment'],duration)
    constant=CAP*component.k-float(np.sum(duration))-TURN*component.n
    # Primary MIN lower bound includes the schedule-independent constant.
    phi_lb=float(plan.bound)-constant
    deployed_phi=phi(component,plan.assignment,duration)
    delta_upper=max(0.,deployed_phi-phi_lb)
    s_upper=gamma*phi_anchor+oracle['upper']-oracle_cost
    regret=float(evaluate(component,plan.assignment,component.actual)['cost']-oracle_cost)
    if s_upper+gamma*delta_upper+0.02<regret:
        raise AssertionError('Invalid anchored regret upper bound')
    result={'certified':True,'regret':regret,'anchored_upper':s_upper,
            'tie_delta_upper':delta_upper,
            'tie_safe_upper':s_upper+gamma*delta_upper,
            'separation_exact':oracle['complete'],'separation':oracle,
            'planner':plan.as_dict()}
    if report_gap:
        primary=solve_week(component,duration,seconds=seconds,threads=threads,tie=False)
        adversary=solve_adversary(component,duration,gamma=gamma,seconds=seconds,threads=threads)
        if (primary.value is not None and primary.bound is not None
                and adversary.value is not None and adversary.bound is not None):
            gap_lower=gamma*primary.bound+adversary.value-oracle_cost
            gap_upper=gamma*primary.value+adversary.bound-oracle_cost
            if gap_upper+gamma*delta_upper+.02 < regret:
                raise AssertionError('Certified primary GAP upper bound invalid')
            if s_upper+.02 < gap_lower:
                raise AssertionError('Anchored bound below primary GAP lower bound')
            result['vf_gap_interval']=[float(gap_lower),float(gap_upper)]
            result['vf_gap_exact']=bool(primary.complete and adversary.complete)
        else:
            result['vf_gap_interval']=None
            result['vf_gap_exact']=False
    return result
