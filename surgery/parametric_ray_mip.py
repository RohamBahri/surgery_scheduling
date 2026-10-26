"""Parametric separation for anchored OR planner; Gurobi numerical oracle.

P1: global primary-improvement MILP, with objective difference to fixed y.
P2: exact rational check of fixed z's tolerance eligibility.
P3: global better-hash entrant MILPs (h0, h1, lex) maximizing interior
    distance rho; endpoint-only feasibility does NOT alter open cells.

The output NEVER claims rational exactness from floating-point Gurobi bounds.
Witness schedules are checked with exact rational costs before admission.
"""
from __future__ import annotations

from fractions import Fraction as Q
import time
from surgery.parametric_ray import rational, hash_weight, CAP, TURN, OVER


class NumericOracle:
    def __init__(self,component,week,*,seconds=10.,mu=1e-6):
        if mu<0 or seconds<=0:raise ValueError('Invalid numerical tolerance/budget')
        self.c=component
        self.week=week
        self.seconds=float(seconds)
        self.mu=float(mu)

    def _problem(self,a,b,*,sense):
        import gurobipy as gp
        from gurobipy import GRB
        from surgery.anchored_weekly_solver import _model
        if not a<b or float(a)>=float(b):
            raise ArithmeticError('Exact interval collapses at float precision')
        m,x,used,by_room=_model(self.week,self.seconds,1)
        m.Params.OutputFlag=0
        m.Params.MIPGap=0
        m.Params.MIPGapAbs=0
        m.Params.TimeLimit=self.seconds
        m.Params.FeasibilityTol=1e-9
        m.Params.OptimalityTol=1e-9
        m.Params.IntFeasTol=1e-9
        s=m.addVar(lb=float(a),ub=float(b),name='ray_s')
        # Every duration is affine throughout the original primitive cell.
        da=self.c.durations(a)
        db=self.c.durations(b)
        slope=[(v-u)/(b-a) for u,v in zip(da,db)]
        offset=[u-a*p for u,p in zip(da,slope)]
        cost=[]
        for j,cases in enumerate(by_room):
            terms=[]
            for i in cases:
                xi=x[i,j]
                # v_ij = s*x_ij, exact binary-continuous McCormick.
                v=m.addVar(lb=min(0.,float(a)),ub=max(0.,float(b)),
                           name=f'prod_{i}_{j}')
                m.addConstr(v>=float(a)*xi)
                m.addConstr(v<=float(b)*xi)
                m.addConstr(v>=s-float(b)*(1-xi))
                m.addConstr(v<=s-float(a)*(1-xi))
                terms.append((float(offset[i])+float(TURN))*xi+
                             float(slope[i])*v)
            load=gp.quicksum(terms)-float(TURN)*used[j]
            idle=m.addVar(lb=0,name=f'idle_{j}')
            ot=m.addVar(lb=0,name=f'ot_{j}')
            m.addConstr(idle>=float(CAP)-load)
            m.addConstr(ot>=load-float(CAP))
            cost.append(idle+float(OVER)*ot)
        return m,x,s,gp.quicksum(cost)

    def _reference(self,a,b,y,s):
        # Cost of fixed y affine on certified restricted cell.
        ca=self.c.cost(y,self.c.durations(a))
        cb=self.c.cost(y,self.c.durations(b))
        slope=(cb-ca)/(b-a)
        return float(ca-a*slope)+float(slope)*s

    def _tau_reference(self,a,b,y,s):
        qa=self.c.cost(y,self.c.durations(a))
        qb=self.c.cost(y,self.c.durations(b))
        ta=self.c.tolerance(qa)
        tb=self.c.tolerance(qb)
        slope=(tb-ta)/(b-a)
        return float(ta-a*slope)+float(slope)*s

    def _winner(self,x):
        return tuple(next(j for j in self.c.allowed[i] if x[i,j].X>.5)
                     for i in range(self.c.n))

    def primary(self,a,b,y):
        """Return (ADD, schedule) or NUMERIC_PASS or UNRESOLVED.

        The min objective has a feasible zero candidate y. A float bound
        >= -mu is only a numerical pass, never an exact P1 certificate.
        """
        import gurobipy as gp
        from gurobipy import GRB
        m=None
        try:
            m,x,s,cost=self._problem(a,b,sense='min')
            m.setObjective(cost-self._reference(a,b,y,s),GRB.MINIMIZE)
            m.optimize()
            status=str(m.Status)
            if m.SolCount:
                trial=self._winner(x)
                step=Q.from_float(float(s.X))
                if a<=step<=b and self.c.feasible(trial):
                    improvement=(self.c.cost(trial,self.c.durations(step))-
                                 self.c.cost(y,self.c.durations(step)))
                    if improvement<0:
                        return ('ADD',trial,dict(kind='P1',status=status,
                                        exact_witness_delta=str(improvement)))
            bound=float(m.ObjBound) if m.SolCount or m.Status==GRB.TIME_LIMIT else None
            if m.Status==GRB.OPTIMAL and bound is not None and bound>=-self.mu:
                return ('NUMERIC_PASS',None,dict(kind='P1',bound=bound,
                           margin=self.mu,status=status,NOT_EXACT=True))
            return ('UNRESOLVED',None,dict(kind='P1',bound=bound,status=status))
        except (ArithmeticError,OverflowError) as exc:
            return ('UNRESOLVED',None,dict(kind='P1',error=str(exc)))
        finally:
            if m is not None:m.dispose()

    def eligibility(self,a,b,y,z):
        """P2: exact Fraction check on both endpoints (cost/tau affine)."""
        for s in (a,b):
            dur=self.c.durations(s)
            q=self.c.cost(y,dur)
            if self.c.cost(z,dur)>q+self.c.tolerance(q):
                return False
        return True

    def _preference_constraints(self,m,x,z,kind,k=None):
        import gurobipy as gp
        h0=gp.quicksum(hash_weight(0,self.c.case_ids[i],*self.c.slots[j])*x[i,j]
                   for i in range(self.c.n) for j in self.c.allowed[i])
        h1=gp.quicksum(hash_weight(1,self.c.case_ids[i],*self.c.slots[j])*x[i,j]
                   for i in range(self.c.n) for j in self.c.allowed[i])
        b0=self.c.hash(z,0)
        b1=self.c.hash(z,1)
        if kind=='h0':m.addConstr(h0<=b0-1)
        elif kind=='h1':
            m.addConstr(h0==b0)
            m.addConstr(h1<=b1-1)
        elif kind=='lex':
            m.addConstr(h0==b0)
            m.addConstr(h1==b1)
            for i in range(k):m.addConstr(x[i,z[i]]==1)
            preceding=[j for j in self.c.allowed[k] if j<z[k]]
            if not preceding:
                raise ValueError('No lexicographically smaller assigned slot')
            m.addConstr(gp.quicksum(x[k,j] for j in preceding)==1)
        else:raise ValueError(kind)

    def preferred(self,a,b,y,z):
        """P3: search ALL lex-preferred disjuncts, not just first hash.

        A strictly interior rationally rechecked witness adds a schedule.
        A Gurobi upper bound rho<=mu is only a numerical pass.
        """
        import gurobipy as gp
        from gurobipy import GRB
        variants=[('h0',None),('h1',None)]+[
            ('lex',k) for k in range(self.c.n)
            if any(j<z[k] for j in self.c.allowed[k])]
        reports=[]
        for kind,k in variants:
            m=None
            try:
                m,x,s,cost=self._problem(a,b,sense='max')
                margin=m.addVar(lb=0.,ub=float((b-a)/2),name='interior_rho')
                m.addConstr(s>=float(a)+margin)
                m.addConstr(s<=float(b)-margin)
                m.addConstr(cost<=self._reference(a,b,y,s)+
                            self._tau_reference(a,b,y,s))
                self._preference_constraints(m,x,z,kind,k)
                m.setObjective(margin,GRB.MAXIMIZE)
                m.optimize()
                if m.SolCount:
                    candidate=self._winner(x)
                    step=Q.from_float(float(s.X))
                    if a<step<b and self.c.feasible(candidate):
                        duration=self.c.durations(step)
                        q=self.c.cost(y,duration)
                        if (self.c.preference(candidate)<self.c.preference(z)
                           and self.c.cost(candidate,duration)<=q+self.c.tolerance(q)):
                            return ('ADD',candidate,dict(
                               kind='P3',stage=kind,index=k,
                               exact_witness_s=str(step),
                               exact_witness_cost=str(self.c.cost(candidate,duration))))
                if m.Status==GRB.INFEASIBLE:
                    # Numerical infeasibility is NOT a rational certificate.
                    reports.append(dict(stage=kind,index=k,
                                        status='GUROBI_INFEASIBLE_NOT_EXACT'))
                    continue
                ub=float(m.ObjBound) if m.SolCount or m.Status==GRB.TIME_LIMIT else None
                if m.Status==GRB.OPTIMAL and ub is not None and ub<=self.mu:
                    reports.append(dict(stage=kind,index=k,upper_rho=ub,
                                        status='NUMERIC_PASS'))
                    continue
                return ('UNRESOLVED',None,dict(kind='P3',stage=kind,index=k,
                                              bound=ub,status=str(m.Status)))
            except (ArithmeticError,OverflowError) as exc:
                return ('UNRESOLVED',None,dict(kind='P3',stage=kind,
                                             error=str(exc)))
            finally:
                if m is not None:m.dispose()
        return ('NUMERIC_PASS',None,dict(kind='P3',checks=reports,NOT_EXACT=True))


    def interval_lower_bound(self,a,b,reference_schedules,anchor=None):
        """Numerical MILP bound on deployed REALIZED cost over a full interval.

        For every s in [a,b], the deployed z is included in the anchor's
        planned-cost relaxation. Minimizing its ACTUAL cost is therefore a
        lower bound on the unknown deployed cost for every s in that interval.
        The validity of the mathematical relaxation is exact; the numeric
        Gurobi dual must still be verified before it can be claimed exact.
        """
        from gurobipy import GRB
        from surgery.anchored_weekly_solver import _model
        from surgery.weekly_planner import _cost
        from surgery.parametric_ray import OVER
        c=self.c
        if anchor is None:anchor=(a+b)/2
        d_a=c.durations(a);d_b=c.durations(b);d_0=c.durations(anchor)
        radius=OVER*max(sum((abs(u-v) for u,v in zip(d_a,d_0)),Q(0)),
                        sum((abs(u-v) for u,v in zip(d_b,d_0)),Q(0)))
        knots={a,b}
        for booked,raw,p in zip(c.booked,c.case_raw,c.projection):
            if p:
                for v in (-c.h,c.h,(Q(1)-booked)/c.alpha):
                    q=(v-raw)/p
                    if a<q<b:knots.add(q)
        refs=tuple(reference_schedules)
        if not refs:raise ValueError('Missing feasible planned-cost reference')
        upper=min(max(c.cost(z,c.durations(s)) for s in knots) for z in refs)
        threshold=upper+radius+c.tolerance(upper)
        m=None
        try:
            m,x,used,by_room=_model(self.week,self.seconds,1)
            m.Params.TimeLimit=self.seconds
            m.Params.OutputFlag=0
            predicted=_cost(m,x,used,by_room,[float(v) for v in d_0],
                            exact=False,name='ray_predicted_anchor')
            actual=_cost(m,x,used,by_room,[float(v) for v in c.actual],
                         exact=False,name='ray_realized')
            # Expand the floating RHS outward, rather than exclude a point
            # solely due to one representational rounding direction.
            import numpy as np
            safe_upper=float(np.nextafter(float(threshold),np.inf))
            m.addConstr(predicted<=safe_upper)
            m.setObjective(actual,GRB.MINIMIZE)
            m.optimize()
            dual=(float(m.ObjBound) if m.Status in (GRB.OPTIMAL,GRB.TIME_LIMIT)
                  else None)
            return dict(kind='INTERVAL_RELAXATION',a=str(a),b=str(b),
                        upper_predicted_exact=str(threshold),
                        bound_numerical=dual,status=str(m.Status),
                        mathematically_verified=False,
                        unconditional_lower_bound=0.)
        except Exception as ex:
            return dict(kind='INTERVAL_RELAXATION',a=str(a),b=str(b),
                        error=f'{type(ex).__name__}: {ex}',
                        unconditional_lower_bound=0.)
        finally:
            if m is not None:m.dispose()


def discover_component(component,week,initial_schedules,*,
                       seconds=180.,separation_seconds=5.,mu=1e-6,
                       max_rounds=20,max_cells=500,max_schedules=50,
                       exhaustive_limit=64,boundary_seconds=10.,
                       verify_boundaries=True):
    """Complete method for small components; bounded numerical MILP discovery for large.

    Exhaustive method gives a genuine exact-arithmetic reference certificate.
    Larger results are NUMERICALLY_CHECKED, never EXACT. Missing proofs are
    UNRESOLVED; no unverified intervals are silently discarded.
    """
    started=time.monotonic()
    # Do not enumerate an unsafe combinatorial explosion.
    combinations=1
    for row in component.allowed:
        combinations*=len(row)
        if combinations>exhaustive_limit:break
    if combinations<=exhaustive_limit:
        try:
            result=component.exact_exhaustive(max_schedules=exhaustive_limit,
                                               max_cells=max_cells)
            return result
        except RuntimeError:
            pass
    oracle=NumericOracle(component,week,seconds=separation_seconds,mu=mu)
    library={tuple(z) for z in initial_schedules if component.feasible(tuple(z))}
    if not library:
        return dict(status='UNRESOLVED',reason='No feasible initial schedules')
    discoveries=[]
    for rnd in range(max_rounds):
        if time.monotonic()-started>seconds:
            return dict(status='UNRESOLVED',reason='COMPONENT_WALL',
                       library_size=len(library),discoveries=discoveries)
        if len(library)>max_schedules:
            return dict(status='UNRESOLVED',reason='LIBRARY_LIMIT',
                       library_size=len(library),discoveries=discoveries)
        try:
            cuts=component.library_partition(tuple(sorted(library)),max_cells)
        except (RuntimeError,OverflowError) as ex:
            return dict(status='UNRESOLVED',reason='PARTITION_LIMIT',error=str(ex),
                       library_size=len(library),discoveries=discoveries)
        cells=[]
        added=False
        for a,b in zip(cuts,cuts[1:]):
            if time.monotonic()-started>seconds:
                return dict(status='UNRESOLVED',reason='COMPONENT_WALL',
                           library_size=len(library),discoveries=discoveries)
            try:
                y,z,j=component.cell(tuple(sorted(library)),a,b)
            except AssertionError as ex:
                return dict(status='UNRESOLVED',reason='AFFINE_PARTITION',error=str(ex))
            if not oracle.eligibility(a,b,y,z):
                return dict(status='UNRESOLVED',reason='P2_REFERENCE_FAILURE',
                           a=str(a),b=str(b))
            first,u,info=oracle.primary(a,b,y)
            if first=='ADD':
                if u in library:
                    return dict(status='UNRESOLVED',reason='P1_DUPLICATE_WITNESS')
                library.add(u)
                discoveries.append(info)
                added=True
                break
            if first!='NUMERIC_PASS':
                return dict(status='UNRESOLVED',reason='P1_NOT_CERTIFIED',
                           detail=info,a=str(a),b=str(b),
                           library_size=len(library))
            third,u,info3=oracle.preferred(a,b,y,z)
            if third=='ADD':
                if u in library:
                    return dict(status='UNRESOLVED',reason='P3_DUPLICATE_WITNESS')
                library.add(u)
                discoveries.append(info3)
                added=True
                break
            if third!='NUMERIC_PASS':
                return dict(status='UNRESOLVED',reason='P3_NOT_CERTIFIED',
                           detail=info3,a=str(a),b=str(b),
                           library_size=len(library))
            cells.append(dict(a=a,b=b,z=z,real=j,
                              status='NUMERICAL_REGION_ONLY'))
        if added:continue
        # Interior separation is complete to solver tolerance. Exact
        # boundary rays are handled distinctly, never via s+delta.
        boundaries=[]
        for s in cuts:
            proposed,q,actual=component.follower(s,tuple(library))
            row=dict(s=s,z=proposed,real=actual,
                     status='UNVERIFIED_FULL_PLANNER')
            if verify_boundaries:
                from surgery.anchored_weekly_solver import solve_week
                try:
                    if time.monotonic()-started>seconds:
                        return dict(status='UNRESOLVED',reason='BOUNDARY_WALL',
                                    library_size=len(library))
                    deployed=solve_week(week,[float(v) for v in
                        component.durations(s)],seconds=boundary_seconds,
                        threads=1,tie=True,hash_seed=0)
                    if not deployed.complete or deployed.assignment is None:
                        return dict(status='UNRESOLVED',
                                    reason='BOUNDARY_DEPLOYMENT_NOT_CERTIFIED',
                                    s=str(s),planner_status=deployed.status)
                    observed=tuple(deployed.assignment)
                    if observed!=proposed:
                        if observed not in library and component.feasible(observed):
                            library.add(observed)
                            discoveries.append(dict(kind='BOUNDARY_ADDED',
                                                    s=str(s)))
                            added=True
                            break
                        return dict(status='UNRESOLVED',
                                    reason='BOUNDARY_NUMERICAL_MISMATCH',
                                    s=str(s),library=proposed,planner=observed)
                    row['status']='NUMERICALLY_CHECKED_ANCHORED'
                except (ArithmeticError,OverflowError) as exc:
                    return dict(status='UNRESOLVED',reason='BOUNDARY_FLOAT',
                                s=str(s),error=str(exc))
            boundaries.append(row)
        if added:continue
        state=('NUMERIC_PARTITION' if verify_boundaries else
               'NUMERIC_PARTITION_BOUNDARIES_UNVERIFIED')
        return dict(status=state,cells=cells,boundaries=boundaries,
                    library_size=len(library),discoveries=discoveries,
                    rounds=rnd+1)
    return dict(status='UNRESOLVED',reason='ROUND_LIMIT',
                library_size=len(library),discoveries=discoveries)
