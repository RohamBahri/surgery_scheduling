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
        elif kind=='collision':
            m.addConstr(h0==b0)
            m.addConstr(h1==b1)
        elif kind=='lex':
            m.addConstr(h0==b0)
            m.addConstr(h1==b1)
            for i in range(k):m.addConstr(x[i,z[i]]==1)
            preceding=[j for j in self.c.allowed[k] if j<z[k]]
            if not preceding:
                raise ValueError('No lexicographically smaller assigned slot')
            m.addConstr(gp.quicksum(x[k,j] for j in preceding)==1)
        else:raise ValueError(kind)

    def _same_hash_collision(self,a,b,y,z,*,point=None):
        """One MILP to rule out all lexicographic fallback variants.

        If numerically infeasible, no distinct competitor can have BOTH
        hashes equal. This is only a floating-point screening result, never
        an exact-arithmetic infeasibility certificate.
        """
        import gurobipy as gp
        from gurobipy import GRB
        m=None
        try:
            if point is None:
                m,x,param,cost=self._problem(a,b,sense='min')
                rhs=self._reference(a,b,y,param)+self._tau_reference(a,b,y,param)
            else:
                m,x,cost=self._point_problem(point)
                q=self.c.cost(y,self.c.durations(point))
                rhs=float(q+self.c.tolerance(q))
            m.addConstr(cost<=rhs+self.mu)
            self._preference_constraints(m,x,z,'collision')
            m.addConstr(gp.quicksum(x[i,z[i]] for i in range(self.c.n))<=self.c.n-1)
            m.setObjective(0.,GRB.MINIMIZE)
            m.optimize()
            if m.Status==GRB.INFEASIBLE:
                return ('NUMERIC_NONE',None,dict(status='COLLISION_NUMERIC_INFEASIBLE'))
            if m.SolCount:
                trial=self._winner(x)
                probe=Q.from_float(float(param.X)) if point is None else point
                if (self.c.feasible(trial) and
                        (a<probe<b if point is None else True)):
                    duration=self.c.durations(probe)
                    q=self.c.cost(y,duration)
                    if (self.c.hash(trial,0)==self.c.hash(z,0) and
                        self.c.hash(trial,1)==self.c.hash(z,1) and
                        self.c.cost(trial,duration)<=q+self.c.tolerance(q)):
                        if self.c.preference(trial)<self.c.preference(z):
                            return ('ADD',trial,dict(kind='COLLISION',s=str(probe)))
                        return ('POSSIBLE',None,dict(status='COLLISION_EXISTS'))
            # Unknown or numerically spurious: MUST NOT skip lex tests.
            return ('POSSIBLE',None,dict(status='COLLISION_NOT_RULED_OUT',
                                        gurobi_status=str(m.Status)))
        except (ArithmeticError,OverflowError) as exc:
            return ('POSSIBLE',None,dict(error=str(exc)))
        finally:
            if m is not None:m.dispose()

    def preferred(self,a,b,y,z):
        """Parametric P3 with a single same-hash precheck before lex search.

        Skipping lex after a numeric-infeasible collision test is only a
        NUMERIC_PASS, never an exact-arithmetic certificate.
        """
        from gurobipy import GRB
        reports=[]
        # Hash comparisons are cheap and typically settle the entire order.
        for kind,k in [('h0',None),('h1',None)]:
            state,trial,info=self._preferred_variant(a,b,y,z,kind,k)
            if state=='ADD':return (state,trial,info)
            if state!='NUMERIC_PASS':return (state,trial,info)
            reports.append(info)
        if any(any(j<z[k] for j in self.c.allowed[k]) for k in range(self.c.n)):
            state,trial,check=self._same_hash_collision(a,b,y,z)
            reports.append(check)
            if state=='ADD':return ('ADD',trial,check)
            if state!='NUMERIC_NONE':
                for k in range(self.c.n):
                    if not any(j<z[k] for j in self.c.allowed[k]):continue
                    state,trial,info=self._preferred_variant(a,b,y,z,'lex',k)
                    if state=='ADD':return (state,trial,info)
                    if state!='NUMERIC_PASS':return (state,trial,info)
                    reports.append(info)
        return ('NUMERIC_PASS',None,dict(kind='P3',checks=reports,NOT_EXACT=True))

    def _preferred_variant(self,a,b,y,z,kind,k):
        from gurobipy import GRB
        m=None
        try:
            m,x,s,cost=self._problem(a,b,sense='max')
            margin=m.addVar(lb=0.,ub=float((b-a)/2),name='interior_rho')
            m.addConstr(s>=float(a)+margin)
            m.addConstr(s<=float(b)-margin)
            m.addConstr(cost<=self._reference(a,b,y,s)+
                        self._tau_reference(a,b,y,s)+self.mu)
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
                            kind='P3',stage=kind,index=k,exact_witness_s=str(step)))
            if m.Status==GRB.INFEASIBLE:
                return ('NUMERIC_PASS',None,dict(
                    stage=kind,index=k,status='GUROBI_INFEASIBLE_NOT_EXACT'))
            ub=float(m.ObjBound) if m.SolCount or m.Status==GRB.TIME_LIMIT else None
            if m.Status==GRB.OPTIMAL and ub is not None and ub<=self.mu:
                return ('NUMERIC_PASS',None,dict(
                    stage=kind,index=k,upper_rho=ub,status='NUMERIC_PASS'))
            return ('UNRESOLVED',None,dict(
                kind='P3',stage=kind,index=k,bound=ub,status=str(m.Status)))
        except (ArithmeticError,OverflowError) as exc:
            return ('UNRESOLVED',None,dict(kind='P3',stage=kind,error=str(exc)))
        finally:
            if m is not None:m.dispose()

    def _point_problem(self,step):
        """Fixed-rational-point scheduling MILP; NO s*x products."""
        from surgery.anchored_weekly_solver import _model
        from surgery.weekly_planner import _cost
        m,x,used,by_room=_model(self.week,self.seconds,1)
        m.Params.OutputFlag=0
        m.Params.MIPGap=0
        m.Params.MIPGapAbs=0
        m.Params.TimeLimit=self.seconds
        m.Params.FeasibilityTol=1e-9
        m.Params.IntFeasTol=1e-9
        m.Params.OptimalityTol=1e-9
        d=[float(v) for v in self.c.durations(step)]
        cost=_cost(m,x,used,by_room,d,exact=False,name='point_cost')
        return m,x,cost

    def point_primary(self,step,y):
        """Point P1: verified rational witness or numerical primal/dual check."""
        from gurobipy import GRB
        m=None
        try:
            m,x,cost=self._point_problem(step)
            m.setObjective(cost,GRB.MINIMIZE)
            m.optimize()
            d=self.c.durations(step)
            q=self.c.cost(y,d)
            if m.SolCount:
                challenger=self._winner(x)
                if self.c.feasible(challenger):
                    improvement=self.c.cost(challenger,d)-q
                    if improvement<0:
                        return ('ADD',challenger,dict(
                            kind='POINT_P1',s=str(step),delta=str(improvement)))
            lb=float(m.ObjBound) if m.SolCount or m.Status==GRB.TIME_LIMIT else None
            if m.Status==GRB.OPTIMAL and lb is not None and lb>=float(q)-self.mu:
                return ('NUMERIC_PASS',None,dict(
                    kind='POINT_P1',bound=lb,NOT_EXACT=True))
            return ('UNRESOLVED',None,dict(
                kind='POINT_P1',status=str(m.Status),bound=lb))
        except (ArithmeticError,OverflowError) as exc:
            return ('UNRESOLVED',None,dict(kind='POINT_P1',error=str(exc)))
        finally:
            if m is not None:m.dispose()

    def point_preferred(self,step,y,z):
        """Point P3: exact-input comparison, numerically solved fixed MIPs.

        At a cost/tolerance crossing, a verified rational challenger is
        admitted; a numerical infeasibility is only NUMERIC_PASS.
        """
        from gurobipy import GRB
        import numpy as np
        reports=[]
        for kind,k in [('h0',None),('h1',None)]:
            state,trial,info=self._point_variant(step,y,z,kind,k)
            if state=='ADD':return (state,trial,info)
            if state!='NUMERIC_PASS':return (state,trial,info)
            reports.append(info)
        if any(any(j<z[k] for j in self.c.allowed[k]) for k in range(self.c.n)):
            state,trial,info=self._same_hash_collision(Q(0),self.c.upper,y,z,
                                                       point=step)
            reports.append(info)
            if state=='ADD':return ('ADD',trial,info)
            if state!='NUMERIC_NONE':
                for k in range(self.c.n):
                    if not any(j<z[k] for j in self.c.allowed[k]):continue
                    state,trial,info=self._point_variant(step,y,z,'lex',k)
                    if state=='ADD':return (state,trial,info)
                    if state!='NUMERIC_PASS':return (state,trial,info)
                    reports.append(info)
        return ('NUMERIC_PASS',None,dict(kind='POINT_P3',checks=reports,NOT_EXACT=True))

    def _point_variant(self,step,y,z,kind,k):
        from gurobipy import GRB
        import numpy as np
        m=None
        try:
            m,x,cost=self._point_problem(step)
            d=self.c.durations(step)
            q=self.c.cost(y,d)
            bound=q+self.c.tolerance(q)
            # Numeric RHS is outward-rounded. This is not an exact proof.
            safe=float(np.nextafter(float(bound),np.inf))+self.mu
            m.addConstr(cost<=safe)
            self._preference_constraints(m,x,z,kind,k)
            m.setObjective(0.,GRB.MINIMIZE)
            m.optimize()
            if m.SolCount:
                challenger=self._winner(x)
                if (self.c.feasible(challenger)
                    and self.c.preference(challenger)<self.c.preference(z)
                    and self.c.cost(challenger,d)<=bound):
                    return ('ADD',challenger,dict(
                        kind='POINT_P3',stage=kind,index=k,s=str(step)))
                return ('UNRESOLVED',None,dict(
                    kind='POINT_P3',stage=kind,index=k,
                    reason='NUMERIC_FALSE_FEASIBLE_AT_BOUNDARY'))
            if m.Status==GRB.INFEASIBLE:
                return ('NUMERIC_PASS',None,dict(
                    kind='POINT_P3',stage=kind,index=k,NOT_EXACT=True))
            return ('UNRESOLVED',None,dict(
                kind='POINT_P3',stage=kind,index=k,status=str(m.Status)))
        except (ArithmeticError,OverflowError) as exc:
            return ('UNRESOLVED',None,dict(
                kind='POINT_P3',stage=kind,error=str(exc)))
        finally:
            if m is not None:m.dispose()

    def point_verify(self,step,y,z):
        p1,u,info=self.point_primary(step,y)
        if p1!='NUMERIC_PASS':return (p1,u,info)
        d=self.c.durations(step)
        q=self.c.cost(y,d)
        if self.c.cost(z,d)>q+self.c.tolerance(q):
            return ('UNRESOLVED',None,dict(
                kind='POINT_P2',reason='LIBRARY_INELIGIBLE',s=str(step)))
        return self.point_preferred(step,y,z)

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
                       verify_boundaries=True,resume=None):
    """Finite schedule discovery with numerical MILP separation.

    Add ALL rationally verified witnesses in a pass, instead of restarting
    at the first. Retain completed P1/P2/P3 regions over their subintervals
    if their reference and selected schedules are unchanged. Incomplete
    attempts carry resumable libraries and certificates. Numeric Gurobi
    passes are NEVER promoted to rational exact certificates.
    """
    started=time.monotonic()
    combinations=1
    for row in component.allowed:
        combinations*=len(row)
        if combinations>exhaustive_limit:break
    if combinations<=exhaustive_limit:
        try:
            return component.exact_exhaustive(max_schedules=exhaustive_limit,
                                               max_cells=max_cells)
        except RuntimeError:
            pass
    oracle=NumericOracle(component,week,seconds=separation_seconds,mu=mu)
    library={tuple(z) for z in initial_schedules if component.feasible(tuple(z))}
    records=[]
    certificates=[]
    if isinstance(resume,dict):
        state=resume.get('resume_state',{})
        # The calling runner rejects checkpoints from incompatible models.
        for assignment in state.get('library',[]):
            z=tuple(map(int,assignment))
            if component.feasible(z):library.add(z)
        for cell in state.get('certificates',[]):
            try:
                a,b=Q(cell['a']),Q(cell['b'])
                y,z=tuple(cell['y']),tuple(cell['z'])
                if a<b and 0<=a and b<=component.upper and component.feasible(y) and component.feasible(z):
                    certificates.append(dict(a=str(a),b=str(b),y=list(y),z=list(z)))
            except (KeyError,TypeError,ValueError,ZeroDivisionError):
                continue
        records=list(state.get('discoveries',[]))
    if not library:
        return dict(status='UNRESOLVED',reason='No feasible initial schedules')
    def finished(reason,**extra):
        return dict(status='UNRESOLVED',reason=reason,library_size=len(library),
                    discoveries=records,
                    resume_state=dict(library=[list(z) for z in sorted(library)],
                                      certificates=certificates,discoveries=records),
                    **extra)
    def remaining():
        return time.monotonic()-started < seconds
    def covering(a,b,y,z):
        return any(Q(row['a'])<=a and b<=Q(row['b'])
                   and tuple(row['y'])==tuple(y)
                   and tuple(row['z'])==tuple(z)
                   for row in certificates)
    for rnd in range(max_rounds):
        if not remaining():return finished('COMPONENT_WALL',rounds=rnd)
        if len(library)>max_schedules:return finished('LIBRARY_LIMIT',rounds=rnd)
        try:
            cuts=component.library_partition(tuple(sorted(library)),max_cells)
        except (RuntimeError,OverflowError) as ex:
            return finished('PARTITION_LIMIT',error=str(ex),rounds=rnd)
        cells=[]
        added=set()
        uncertainty=None
        for a,b in zip(cuts,cuts[1:]):
            if not remaining():
                return finished('COMPONENT_WALL',rounds=rnd)
            try:
                y,z,j=component.cell(tuple(sorted(library)),a,b)
            except AssertionError as ex:
                return finished('AFFINE_PARTITION',error=str(ex))
            if covering(a,b,y,z):
                cells.append(dict(a=a,b=b,z=z,real=j,
                                  status='NUMERICAL_REGION_REUSED'))
                continue
            if not oracle.eligibility(a,b,y,z):
                uncertainty=dict(reason='P2_REFERENCE_FAILURE',a=str(a),b=str(b))
                continue
            first,u,info=oracle.primary(a,b,y)
            if first=='ADD':
                if u in library:
                    return finished('P1_DUPLICATE_WITNESS',a=str(a),b=str(b))
                if u not in added:
                    added.add(tuple(u))
                    records.append(info)
                continue
            if first!='NUMERIC_PASS':
                if uncertainty is None:
                    uncertainty=dict(reason='P1_NOT_CERTIFIED',detail=info,
                                     a=str(a),b=str(b))
                continue
            third,u,info3=oracle.preferred(a,b,y,z)
            if third=='ADD':
                if u in library:
                    return finished('P3_DUPLICATE_WITNESS',a=str(a),b=str(b))
                if u not in added:
                    added.add(tuple(u))
                    records.append(info3)
                continue
            if third!='NUMERIC_PASS':
                if uncertainty is None:
                    uncertainty=dict(reason='P3_NOT_CERTIFIED',detail=info3,
                                     a=str(a),b=str(b))
                continue
            certificates.append(dict(a=str(a),b=str(b),
                                     y=list(y),z=list(z)))
            cells.append(dict(a=a,b=b,z=z,real=j,
                              status='NUMERICAL_REGION_ONLY'))
        if added:
            # All rationally checked witnesses in this pass are valid. The
            # changed restricted partition is rebuilt in the next round.
            library.update(added)
            if len(library)>max_schedules:
                return finished('LIBRARY_LIMIT',rounds=rnd+1)
            continue
        if uncertainty is not None:
            return finished(uncertainty.pop('reason'),**uncertainty,rounds=rnd+1)
        # Interior separation is complete to solver tolerance.
        boundaries=[]
        for step in cuts:
            if not remaining():
                return finished('BOUNDARY_WALL',rounds=rnd+1)
            proposed,q,actual=component.follower(step,tuple(library))
            row=dict(s=step,z=proposed,real=actual,
                     status='UNVERIFIED_FULL_PLANNER')
            if verify_boundaries:
                y=min(library,key=lambda v:(component.cost(v,component.durations(step)),v))
                state,u,info=oracle.point_verify(step,y,proposed)
                if state=='ADD':
                    if u in library:
                        return finished('POINT_DUPLICATE_WITNESS',s=str(step))
                    if u not in added:
                        added.add(tuple(u))
                        records.append(info)
                    continue
                if state!='NUMERIC_PASS':
                    return finished('BOUNDARY_POINT_NOT_CERTIFIED',s=str(step),
                                    detail=info,rounds=rnd+1)
                row['status']='NUMERICALLY_CHECKED_POINT_NOT_EXACT'
            boundaries.append(row)
        if added:
            library.update(added)
            if len(library)>max_schedules:
                return finished('LIBRARY_LIMIT',rounds=rnd+1)
            continue
        state=('NUMERIC_PARTITION' if verify_boundaries else
               'NUMERIC_PARTITION_BOUNDARIES_UNVERIFIED')
        return dict(status=state,cells=cells,boundaries=boundaries,
                    library_size=len(library),discoveries=records,
                    rounds=rnd+1,verified_cell_count=len(certificates))
    return finished('ROUND_LIMIT',rounds=max_rounds)
