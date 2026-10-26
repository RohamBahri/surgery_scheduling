"""Parametric operating-room ray: rational partition plus MILP separation.

This module intentionally distinguishes three levels of evidence:
  EXACT_ENUMERATION: every feasible schedule is explicitly enumerated, and
      the complete piecewise-constant function is evaluated with Fractions.
  NUMERICAL_GUROBI: all parametric separation MILPs received numerical
      solver certificates; NOT an exact-arithmetic global proof.
  UNRESOLVED: time limit, floating ambiguity, region budget, or missing proof.

Never infer exact global optimality from Gurobi MIPGap=0, a rounded result,
checking a witness, or the width of a parameter interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from collections import Counter
import hashlib
import math
import time
import numpy as np

CAP=Q(480)
TURN=Q(30)
OVER=Q(7,4)
TAU_FLOOR=Q(1,500)
TAU_CEIL=Q(1,10)
TAU_SLOPE=Q(1,500000)


def rational(value):
    """Exactly represent the stored IEEE-754 input, never Q(str(float))."""
    if isinstance(value,Q):
        return value
    if isinstance(value,(float,np.floating)):
        return Q.from_float(float(value))
    return Q(int(value)) if isinstance(value,(int,np.integer)) else Q(value)


def saturation_endpoint(raw,projection,h):
    """Intrinsic domain, retains s=0 and s=1 regardless of early saturation.

    For p>0 scores eventually remain >= h; for p<0 scores remain <= -h.
    p=0 is already constant. A floor may saturate sooner, never later.
    """
    h=rational(h)
    result=Q(1)
    for a,p in zip(raw,projection):
        a,p=rational(a),rational(p)
        if p>0:
            result=max(result,Q(0),(h-a)/p)
        elif p<0:
            result=max(result,Q(0),(-h-a)/p)
    return result


def affine_root(a,b):
    return -a/b if b else None


def add_if_between(cuts,x,left,right):
    if x is not None and left<x<right:
        cuts.add(x)


def hash_weight(seed,case_id,weekday,room):
    msg=f"{int(seed)}|{int(case_id)}|{int(weekday)}|{room}".encode()
    return 1+int.from_bytes(hashlib.sha256(msg).digest()[:8],"big")%(1<<20)


@dataclass(frozen=True)
class RationalComponent:
    """Exact fixed-input idealization of one anchored solver component."""
    allowed:tuple
    original:tuple
    surgeon:tuple
    required_days:tuple
    slots:tuple
    case_ids:tuple
    booked:tuple
    actual:tuple
    case_raw:tuple
    projection:tuple
    alpha:Q
    h:Q
    upper:Q

    @classmethod
    def from_week(cls,week,case_raw,projection,alpha,h,upper):
        return cls(
            tuple(tuple(int(j) for j in week.allowed[int(week.surgeon[i])])
                  for i in range(week.n)),
            tuple(int(x) for x in week.original),
            tuple(int(x) for x in week.surgeon),
            tuple(tuple(int(v) for v in days) for days in week.required_days),
            tuple((int(day),str(room)) for day,room in week.slots),
            tuple(int(v) for v in week.case_ids),
            tuple(map(rational,week.booked)),
            tuple(map(rational,week.actual)),
            tuple(map(rational,case_raw)),
            tuple(map(rational,projection)),
            rational(alpha),rational(h),rational(upper))

    @property
    def n(self): return len(self.booked)

    def feasible(self,z):
        if len(z)!=self.n or any(v not in self.allowed[i] for i,v in enumerate(z)):
            return False
        return all(any(self.surgeon[i]==sur and self.slots[z[i]][0]==day
                       for i in range(self.n))
                   for sur,days in enumerate(self.required_days) for day in days)

    def schedules(self,maximum=100000):
        combinations=math.prod(len(row) for row in self.allowed)
        if combinations>maximum:
            raise ValueError(f"Exhaustive schedule count may exceed {maximum}: {combinations}")
        return tuple(z for z in product(*self.allowed) if self.feasible(z))

    def durations(self,s):
        s=rational(s)
        return tuple(max(Q(1),b+max(-self.alpha*self.h,
                         min(self.alpha*self.h,self.alpha*(a+s*p))))
                     for b,a,p in zip(self.booked,self.case_raw,self.projection))

    def cost(self,z,d):
        count=Counter(z)
        loads=[Q(0) for _ in self.slots]
        for i,j in enumerate(z):
            loads[j]+=d[i]+TURN
        for j,n in count.items():
            if n:loads[j]-=TURN
        return sum((max(CAP-l,Q(0))+OVER*max(l-CAP,Q(0)) for l in loads),Q(0))

    def hash(self,z,seed=0):
        return sum(hash_weight(seed,self.case_ids[i],*self.slots[z[i]])
                   for i in range(self.n))

    def preference(self,z):
        return (self.hash(z,0),self.hash(z,1),tuple(z))

    @staticmethod
    def tolerance(q):
        return min(TAU_CEIL,max(TAU_FLOOR,TAU_SLOPE*abs(q)))

    def follower(self,s,schedules):
        if not schedules:raise ValueError("Empty schedule set")
        d=self.durations(s)
        costs={z:self.cost(z,d) for z in schedules}
        q=min(costs.values())
        eligible=(z for z,c in costs.items() if c<=q+self.tolerance(q))
        z=min(eligible,key=self.preference)
        return z,q,self.cost(z,self.actual)

    def primitive_cuts(self,schedules):
        """Cuts at response/floor kinks and candidate room-load hinges."""
        cuts={Q(0),self.upper}
        for b,a,p in zip(self.booked,self.case_raw,self.projection):
            if not p:continue
            for score in (-self.h,self.h,(Q(1)-b)/self.alpha):
                add_if_between(cuts,(score-a)/p,Q(0),self.upper)
        primitive=sorted(cuts)
        for left,right in zip(primitive,primitive[1:]):
            mid=(left+right)/2
            d=self.durations(mid)
            slopes=tuple(self.alpha*p if -self.h<a+mid*p<self.h
                    and b+self.alpha*(a+mid*p)>1 else Q(0)
                    for b,a,p in zip(self.booked,self.case_raw,self.projection))
            for z in schedules:
                counts=Counter(z)
                for j,n in counts.items():
                    if not n:continue
                    indices=[i for i,k in enumerate(z) if k==j]
                    load=sum((d[i]+TURN for i in indices),Q(0))-TURN
                    slope=sum((slopes[i] for i in indices),Q(0))
                    add_if_between(cuts,affine_root(load-slope*mid-CAP,slope),
                                   left,right)
        return sorted(cuts)

    def cost_lines(self,schedules,left,right):
        mid=(left+right)/2
        d=self.durations(mid)
        slopes=tuple(self.alpha*p if -self.h<a+mid*p<self.h
                     and b+self.alpha*(a+mid*p)>1 else Q(0)
                     for b,a,p in zip(self.booked,self.case_raw,self.projection))
        result={}
        for z in schedules:
            slope=Q(0)
            for j in range(len(self.slots)):
                ix=[i for i,k in enumerate(z) if k==j]
                if not ix:continue
                load=sum((d[i]+TURN for i in ix),Q(0))-TURN
                dl=sum((slopes[i] for i in ix),Q(0))
                slope+=(-dl if load<CAP else OVER*dl)
            result[z]=(self.cost(z,d)-slope*mid,slope)
        return result

    def library_partition(self,schedules,max_cells=50000):
        """Exact partition of all *library* switches and tolerance equalities.

        Additional cuts beyond true switches are harmless. P1-P3 decide
        whether schedules omitted from the library can ever be deployed.
        """
        schedules=tuple(dict.fromkeys(tuple(z) for z in schedules))
        primitive=self.primitive_cuts(schedules)
        cuts=set(primitive)
        for a,b in zip(primitive,primitive[1:]):
            lines=self.cost_lines(schedules,a,b)
            for z in schedules:
                cz,dz=lines[z]
                for target in (Q(1000),Q(50000)):
                    add_if_between(cuts,affine_root(cz-target,dz),a,b)
                for y in schedules:
                    cy,dy=lines[y]
                    add_if_between(cuts,affine_root(cz-cy,dz-dy),a,b)
                    # q >= 0; tau(q)=.002, 2e-6*q, or .1
                    for c,mult in ((TAU_FLOOR,Q(1)),(TAU_CEIL,Q(1)),
                                   (Q(0),Q(1)+TAU_SLOPE)):
                        add_if_between(cuts,affine_root(cz-mult*cy-c,
                                                  dz-mult*dy),a,b)
                if len(cuts)>max_cells:
                    raise RuntimeError("Restricted partition exceeds cell budget")
        cuts=sorted(cuts)
        if len(cuts)>max_cells:
            raise RuntimeError("Restricted partition exceeds cell budget")
        return cuts

    def cell(self,schedules,a,b):
        mid=(a+b)/2
        z,q,j=self.follower(mid,schedules)
        y=min(schedules,key=lambda x:(self.cost(x,self.durations(mid)),x))
        # Must check fixed y cost and tau are affine throughout this cell.
        c_left=self.cost(y,self.durations(a))
        c_mid=self.cost(y,self.durations(mid))
        c_right=self.cost(y,self.durations(b))
        if 2*c_mid!=c_left+c_right:
            raise AssertionError("Missing primary reference cost kink")
        tol_l=self.tolerance(c_left)
        tol_r=self.tolerance(c_right)
        if 2*self.tolerance(c_mid)!=tol_l+tol_r:
            raise AssertionError("Missing primary-tolerance kink")
        return y,z,j

    def exact_exhaustive(self,max_schedules=20000,max_cells=100000):
        """TRUE exact global result on small components, with SHA hash order."""
        schedule_set=self.schedules(maximum=max_schedules)
        if not schedule_set:raise ValueError("No feasible schedules")
        cuts=self.library_partition(schedule_set,max_cells=max_cells)
        atoms=[]
        for j,a in enumerate(cuts):
            z,q,v=self.follower(a,schedule_set)
            atoms.append(dict(type="point",s=a,value=v,schedule=z))
            if j+1<len(cuts):
                b=cuts[j+1]
                mid=(a+b)/2
                z,q,v=self.follower(mid,schedule_set)
                atoms.append(dict(type="open",left=a,right=b,s=mid,
                                  value=v,schedule=z))
        best=min(x["value"] for x in atoms)
        stable=min((x["value"] for x in atoms if x["type"]=="open"),default=best)
        choices=[x for x in atoms if x["value"]==best]
        selected=min(choices,key=lambda x:x.get("left",x["s"]))
        return dict(s=selected["s"],value=best,stable_value=stable,
                    schedule_count=len(schedule_set),boundary_count=len(cuts),
                    atoms=atoms,status="EXACT_ENUMERATION")


def to_json_number(value):
    if isinstance(value,Q):return dict(num=value.numerator,den=value.denominator,
                                        approx=float(value))
    return value


def exact_interval_relaxation(component,reference_schedules,all_schedules,
                              left,right,anchor=None):
    """Exact-rational interval lower bound, enumerative validation oracle.

    This evaluates the *same* certified relaxed-hindsight model proposed
    for the large-scale Gurobi implementation, but enumerates schedules on
    tiny real/test components. It never uses a sample grid.
    """
    c=component
    if not (0<=left<right<=c.upper):raise ValueError('Invalid interval')
    if anchor is None:anchor=(left+right)/2
    if not left<=anchor<=right:raise ValueError('Anchor outside interval')
    refs=tuple(reference_schedules)
    schedules=tuple(all_schedules)
    if not refs or not schedules:raise ValueError('Need reference/all feasible schedules')
    # Every case is monotone along s, so the L1 distance to the anchor
    # achieves its maximum at one endpoint of the scalar interval.
    da=c.durations(left)
    db=c.durations(right)
    d0=c.durations(anchor)
    R=OVER*max(sum((abs(x-y) for x,y in zip(da,d0)),Q(0)),
               sum((abs(x-y) for x,y in zip(db,d0)),Q(0)))
    # A fixed schedule's cost is convex between consecutive response/floor
    # kinks. Its interval maximum is attained at a response-kink endpoint.
    response_knots={left,right}
    for booked,a,p in zip(c.booked,c.case_raw,c.projection):
        if not p:continue
        for score in (-c.h,c.h,(Q(1)-booked)/c.alpha):
            kink=(score-a)/p
            if left<kink<right:response_knots.add(kink)
    Pbar=min(max(c.cost(z,c.durations(s)) for s in response_knots)
             for z in refs)
    bound=Pbar+R+c.tolerance(Pbar)
    maybe=[z for z in schedules if c.cost(z,d0)<=bound]
    if not maybe:raise AssertionError('Relaxation excluded all feasible deployed plans')
    return min(c.cost(z,c.actual) for z in maybe)
