"""Coupled full-feature convex pDCA master for the fixed-slope library loss.

No schedule assignment or behavioral-regime binaries enter this master.
Each case's *complete clipped* response is modeled with convex hinge epigraphs.
At an inner iteration the DC upper/lower duration envelopes touch the exact
reviewed duration and give a globally valid convex objective majorizer.
"""
import time
import numpy as np
from .anchored_core import (hinge_response, anchored_score, OVER, TURN, CAP, key)
from .weekly_planner import evaluate


def envelopes(terms):
    """Affine pieces whose pointwise maximum equals a convex hinge sum."""
    result=[(0.,0.)]
    intercept,slope=0.,0.
    for knot,weight in terms:
        slope+=weight
        intercept-=weight*knot
        result.append((intercept,slope))
    return result


def prepare_components(weeks,index,refs,library,*,current=None,current_duration=None):
    """Select lowest-realized-cost references; prefer current on cost ties."""
    if (current is None)!=(current_duration is None):
        raise ValueError('Supply both incumbent schedules and durations')
    out=[]
    for week in weeks:
        for c in week.components():
            k=key(c); record=refs.get(k)
            if not record or not record.get('complete') or not record.get('references'):
                raise ValueError(f'Uncertified reference component {k}')
            rows=np.array([index[int(i)] for i in c.case_ids],int)
            anchors=list(record['references'])
            incumbent=None
            if current is not None:
                rec=current.get(k)
                if rec is None or not rec.get('complete'):
                    raise ValueError(f'Uncertified incumbent schedule for {k}')
                incumbent=list(map(int,rec['assignment']))
                beta=np.asarray(current_duration,float)[rows]
                if abs(evaluate(c,incumbent,c.actual)['cost']-rec['realized_cost'])>1e-5:
                    raise ValueError(f'Incumbent cost mismatch for {k}')
                anchors.append(dict(source='CURRENT_POLICY',type='shared-policy',
                    assignment=incumbent,beta=beta.tolist(),
                    realized_cost=float(rec['realized_cost']),
                    predicted_cost=float(rec['predicted_cost'])))
            best_cost=min(float(r['realized_cost']) for r in anchors)
            ties=[r for r in anchors if float(r['realized_cost'])<=best_cost+1e-7]
            best=next((r for r in ties if r['source']=='CURRENT_POLICY'),
                      min(ties,key=lambda r:r['source']))
            y=list(map(int,best['assignment']))
            beta=np.asarray(best['beta'],float)
            if len(y)!=c.n or beta.shape!=(c.n,) or not np.isfinite(beta).all():
                raise ValueError(f'Invalid anchor {k}')
            from .anchored_core import slopes_from_reference
            slope_schedule=incumbent if incumbent is not None else y
            anchor=dict(assignment=y,beta=beta,
                slopes=slopes_from_reference(c,slope_schedule,beta),
                source=best['source'])
            schedules=[];seen=set()
            for r in anchors:
                z=tuple(map(int,r['assignment']))
                if z not in seen:seen.add(z);schedules.append(list(z))
            for z in library.get(k,[]):
                z=tuple(map(int,z))
                if z not in seen:seen.add(z);schedules.append(list(z))
            for z in schedules:evaluate(c,z,c.actual)
            out.append(dict(component=c,key=k,rows=rows,anchor=anchor,
                schedules=schedules,oracle_cost=float(record['oracle_cost'])))
    return out


def restricted_value(blocks, X, booked, w, alpha, h, gamma, lam=0.):
    from .planner import display_and_duration
    _,pred=display_and_duration(X @ np.asarray(w,float), booked, alpha, h)
    loss=sum(anchored_score(b['component'],b['anchor'],b['schedules'],
                            pred[b['rows']],b['oracle_cost'],gamma,exact=False)
             for b in blocks)
    return loss/len(booked)+lam*float(np.abs(np.asarray(w)[1:]).sum())


def solve_majorizer(blocks,X,booked,at,alpha,h,gamma,lam,
                    prox=0.15,coef_prox=1e-5,seconds=300.,threads=4):
    """Solve one convex touching-majorizer QP with all original rich X columns.

    Optimizer epigraphs use the full (unclipped) score X@w, and therefore do
    not accidentally remove attenuation/rejection/clipping regimes.
    """
    import gurobipy as gp
    from gurobipy import GRB
    X=np.asarray(X,float);booked=np.asarray(booked,float);at=np.asarray(at,float)
    n,p=X.shape
    if n!=len(booked) or p!=len(at) or not np.isfinite(X).all():
        raise ValueError('Feature dimension or coefficients invalid')
    if gamma<=0 or lam<0 or prox<=0 or coef_prox<=0:
        raise ValueError('Invalid optimization parameters')
    if not np.allclose(X[:,0],1.):
        raise ValueError('First rich feature must be the intercept')
    raw0=X@at
    dc=[hinge_response(v,alpha,h) for v in booked]
    tangent=[f.affine_tangents(s) for f,s in zip(dc,raw0)]
    start=time.monotonic()
    with gp.Model('anchored_response_pdca') as m:
        m.Params.OutputFlag=0
        m.Params.Threads=threads
        m.Params.TimeLimit=seconds
        m.Params.Seed=0
        m.Params.NumericFocus=1
        w=m.addVars(p,lb=-150.,ub=150.,name='w')
        score=m.addVars(n,lb=-GRB.INFINITY,name='raw_score')
        P=m.addVars(n,lb=0.,name='positive_hinges')
        Q=m.addVars(n,lb=0.,name='negative_hinges')
        for i in range(n):
            row=np.flatnonzero(X[i])
            m.addConstr(score[i]==gp.quicksum(float(X[i,j])*w[j] for j in row),name=f'feature_{i}')
            for intercept,slope in envelopes(dc[i].positive):
                m.addConstr(P[i]>=intercept+slope*score[i])
            for intercept,slope in envelopes(dc[i].negative):
                m.addConstr(Q[i]>=intercept+slope*score[i])
        upper={}
        lower={}
        for i in range(n):
            pc,ps,qc,qs=tangent[i]
            upper[i]=booked[i]+dc[i].constant+P[i]-qc-qs*score[i]
            lower[i]=booked[i]+dc[i].constant+pc+ps*score[i]-Q[i]
        total=gp.LinExpr()
        for block in blocks:
            c=block['component']; rows=block['rows']; a=block['anchor']
            ref=np.asarray(a['assignment'],int)
            counts=np.bincount(ref,minlength=c.k)
            score_ref=TURN*int(np.count_nonzero(counts))
            for j,count in enumerate(counts):
                if count:
                    ot=m.addVar(lb=0.,name='reference_ot')
                    selected=np.flatnonzero(ref==j)
                    m.addConstr(ot>=gp.quicksum(upper[int(rows[i])] for i in selected)
                                +TURN*(int(count)-1)-480.)
                    score_ref+=OVER*ot
            theta=m.addVar(lb=-GRB.INFINITY,name='component_max')
            for assignment in block['schedules']:
                z=np.asarray(assignment,int)
                counts_z=np.bincount(z,minlength=c.k)
                opened=counts_z>0
                slope=a['slopes']  # Match routine fixed-slope separation exactly.
                # q_s(z,d)=30*U + sum_j s_j [sum_i(d_i+30)x_ij-510*u_j]
                affine=TURN*int(opened.sum())
                affine+=float(np.dot(slope,TURN*counts_z-(TURN+CAP)*opened))
                affine+=gp.quicksum(float(slope[j])*lower[int(rows[i])]
                                     for i,j in enumerate(z) if slope[j]>0)
                real=float(evaluate(c,z,c.actual)['cost'])
                m.addConstr(theta>=real-gamma*affine)
            total+=gamma*score_ref+theta-float(block['oracle_cost'])
        l1=gp.LinExpr()
        if lam>0:
            for j in range(1,p):
                v=m.addVar(lb=0.,name='l1_abs')
                m.addConstr(v>=w[j]);m.addConstr(v>=-w[j]);l1+=v
        prox_score=gp.quicksum((score[i]-float(raw0[i]))**2 for i in range(n))
        prox_coef=gp.quicksum((w[j]-float(at[j]))**2 for j in range(p))
        m.setObjective(total/n+lam*l1+prox/(2*n)*prox_score
                       +coef_prox/2*prox_coef,GRB.MINIMIZE)
        m.optimize()
        if not m.SolCount:
            return {'complete':False,'status':str(m.Status),'seconds':time.monotonic()-start,
                    'w':None,'bound':None}
        new=np.array([w[j].X for j in range(p)],float)
        result={'complete':bool(m.Status==GRB.OPTIMAL),
                'status':str(m.Status),'seconds':time.monotonic()-start,
                'w':new.tolist(),'objective':float(m.ObjVal),
                'bound':float(m.ObjVal) if m.Status==GRB.OPTIMAL else None, 'variables':m.NumVars,
                'constraints':m.NumConstrs,'used_full_feature_matrix':True,
                'feature_coefficient_changes':int(np.count_nonzero(abs(new[1:]-at[1:])>1e-6)),
                'mean_abs_score_change':float(np.mean(abs(X@(new-at))))}
        return result


def pdca(blocks,X,booked,start,alpha,h,gamma,lam,
         iterations=10,prox=.15,coef_prox=1e-5,
         seconds=300.,threads=4,tolerance=1e-5):
    current=np.asarray(start,float).copy()
    log=[]
    previous=restricted_value(blocks,X,booked,current,alpha,h,gamma,lam)
    for k in range(iterations):
        answer=solve_majorizer(blocks,X,booked,current,alpha,h,gamma,lam,
                               prox,coef_prox,seconds,threads)
        if answer['w'] is None:
            log.append({'iteration':k+1,**answer,'accepted':False})
            break
        candidate=np.asarray(answer['w'],float)
        objective=restricted_value(blocks,X,booked,candidate,alpha,h,gamma,lam)
        delta=float(np.sqrt(np.mean((X@(candidate-current))**2)))
        success=bool(objective<=previous+tolerance)
        log.append({'iteration':k+1,**{a:b for a,b in answer.items() if a!='w'},
                    'actual_restricted_score':objective,'before':previous,
                    'rms_recommendation_change':delta,'accepted':success})
        if not success:
            break
        gain=previous-objective
        current=candidate;previous=objective
        if delta<1e-3 or gain<1e-5:
            break
    return {'w':current.tolist(),'restricted_objective':previous,'inner':log}
