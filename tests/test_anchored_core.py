import numpy as np
import pytest
from surgery.anchored_core import (hinge_response, phi, cost_identity,
    fixed_slope_lower, exact_tangent, anchored_score, slopes_from_reference)
from surgery.planner import display_and_duration
from surgery.weekly_data import Week
from surgery.weekly_planner import evaluate


def component():
    return Week('TGH','2012-01-02',((0,'OR1'),(0,'OR2')),
        np.array([1,2,3,4]),('S1','S2'),np.array([0,0,1,1]),
        np.array([0,0,1,1]),((0,1),(0,1)),((0,),(0,)),
        np.array([150.,160.,200.,170.]),np.array([215.,170.,130.,180.]),()).validate()

@pytest.mark.parametrize('b',[1.,5.,15.,30.,75.,120.,250.])
@pytest.mark.parametrize('alpha,h',[(.8,30.),(1.,30.),(.5,20.)])
def test_response_hinge_exact(b,alpha,h):
    model=hinge_response(b,alpha,h)
    xx=np.linspace(-250,260,1121)
    _,ref=display_and_duration(xx,np.full_like(xx,b),alpha,h)
    np.testing.assert_allclose(model.correction(xx),ref-b,atol=1e-8)
    for at in (-180.,-90.,-20.,0.,8.,30.,75.,175.):
        lo,hi=model.duration_bounds(xx,at)
        assert np.all(lo <= model.correction(xx)+1e-8)
        assert np.all(model.correction(xx) <= hi+1e-8)
        idx=np.argmin(abs(xx-at))
        lo_at,hi_at=model.duration_bounds(at,at)
        assert lo_at == pytest.approx(model.correction(at),abs=1e-8)
        assert hi_at == pytest.approx(model.correction(at),abs=1e-8)


def test_room_cost_identity_and_global_tangents():
    c=component()
    rng=np.random.default_rng(61)
    schedules=[[0,0,1,1],[0,1,0,1],[1,1,0,0]]
    for y in schedules:
        b=c.booked
        slopes=slopes_from_reference(c,y,b)
        for z in schedules:
            for trial in range(30):
                d=rng.uniform(10,380,4)
                assert cost_identity(c,z,d)==pytest.approx(evaluate(c,z,d)['cost'])
                assert fixed_slope_lower(c,z,d,slopes)<=phi(c,z,d)+1e-7
                assert exact_tangent(c,z,b,d)<=phi(c,z,d)+1e-7
            assert exact_tangent(c,z,b,b)==pytest.approx(phi(c,z,b))


def test_anchored_bound_against_bruteforce():
    c=component(); schedules=[[i,j,k,l] for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1)]
    actual_best=min(evaluate(c,z,c.actual)['cost'] for z in schedules)
    anchor={'beta':c.booked,'assignment':schedules[2],
            'slopes':slopes_from_reference(c,schedules[2],c.booked)}
    for d in [c.booked,c.actual,c.booked+20,c.booked-30]:
        opt=min(phi(c,z,d) for z in schedules)
        dep=next(z for z in schedules if abs(phi(c,z,d)-opt)<1e-7)
        regret=evaluate(c,dep,c.actual)['cost']-actual_best
        score=anchored_score(c,anchor,schedules,d,actual_best,gamma=2.)
        assert score>=regret-1e-7


def test_actual_feature_nonintercept_effect_and_reachable_reference():
    c=component()
    X=np.array([[1.,0.,-2.,1.],[1.,1.,1.,.1],[1.,2.,-1.,.3],[1.,3.,0.2,-1.]])
    w=np.array([3.,-2.,.5,.1]); d0=c.booked
    _,reviewed=display_and_duration(X@w,d0,.8,30.)
    _,intercept_only=display_and_duration(np.full(4,w[0]),d0,.8,30.)
    assert np.max(np.abs(reviewed-intercept_only))>1.
    from surgery.planner import reachable
    lower,upper=reachable(d0,.8,30.)
    assert np.all(reviewed >= lower-1e-8)
    assert np.all(reviewed <= upper+1e-8)


@pytest.mark.parametrize('kind',[False,True])
def test_gurobi_separation_matches_enumeration(kind):
    pytest.importorskip('gurobipy')
    from surgery.anchored_oracles import separate
    c=component(); d=c.booked+np.array([4.,-15.,20.,-5.]);beta=c.booked
    s=slopes_from_reference(c,[0,0,1,1],beta)
    res=separate(c,d,beta,2.,slopes=s,exact=kind,seconds=30.,threads=1)
    assert res['complete'],res
    schedules=[[i,j,k,l] for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1)]
    score=max(evaluate(c,z,c.actual)['cost']-2.*(
          exact_tangent(c,z,beta,d) if kind else fixed_slope_lower(c,z,d,s))
              for z in schedules)
    assert res['lower']==pytest.approx(score,abs=1e-4)


def test_gurobi_master_uses_every_feature_and_is_majorizing():
    pytest.importorskip('gurobipy')
    from surgery.anchored_master import solve_majorizer,restricted_value
    c=component()
    X=np.array([[1.,0.,-2.,1.],[1.,1.,1.,.1],[1.,2.,-1.,.3],[1.,3.,0.2,-1.]])
    w=np.array([6.,-.1,.4,.1]);cases=[[0,0,1,1],[0,1,0,1],[1,1,0,0]]
    anchor={'assignment':cases[0],'beta':c.booked,
            'slopes':slopes_from_reference(c,cases[0],c.booked)}
    block={'component':c,'key':'test','rows':np.arange(4),
           'anchor':anchor,'schedules':cases,
           'oracle_cost':min(evaluate(c,z,c.actual)['cost'] for z in cases)}
    ans=solve_majorizer([block],X,c.booked,w,.8,30.,2.,0.,
                        prox=.15,seconds=60.,threads=1)
    assert ans['w'] is not None
    assert ans['used_full_feature_matrix']
    before=restricted_value([block],X,c.booked,w,.8,30.,2.)
    after=restricted_value([block],X,c.booked,np.array(ans['w']),.8,30.,2.)
    assert after<=before+1e-3


def test_touching_convex_majorizer_many_random_shared_policies():
    from surgery.anchored_core import response_decomposition
    rng=np.random.default_rng(2026)
    c=component()
    X=np.column_stack([np.ones(c.n),rng.normal(size=(c.n,3))])
    candidates=[[0,0,1,1],[0,1,0,1],[1,1,0,0],[1,0,1,0]]
    anchor={'assignment':candidates[1],'beta':c.booked.copy(),
            'slopes':slopes_from_reference(c,candidates[1],c.booked)}
    oracle=min(evaluate(c,z,c.actual)['cost'] for z in candidates)
    models=response_decomposition(c.booked,.8,30.)
    for _ in range(35):
        w0=rng.uniform(-60,60,4)
        w1=rng.uniform(-65,65,4)
        s0=X@w0;s1=X@w1
        actual=np.array([b+v.correction(s) for b,v,s in zip(c.booked,models,s1)])
        low=[];up=[]
        for b,v,a,z in zip(c.booked,models,s0,s1):
            lower,upper=v.duration_bounds(z,a)
            low.append(b+lower);up.append(b+upper)
        true_score=anchored_score(c,anchor,candidates,actual,oracle,2.,exact=True)
        major=2.*phi(c,anchor['assignment'],up)+max(
            evaluate(c,z,c.actual)['cost']-2.*exact_tangent(c,z,anchor['beta'],low)
            for z in candidates)-oracle
        assert major+1e-7>=true_score
        # Exact touching at the current score is the critical pDCA condition.
        before=c.booked+np.array([v.correction(s) for v,s in zip(models,s0)])
        base=anchored_score(c,anchor,candidates,before,oracle,2.,exact=True)
        low0=[];up0=[]
        for b,v,a in zip(c.booked,models,s0):
            l,u=v.duration_bounds(a,a);low0.append(b+l);up0.append(b+u)
        base_major=2.*phi(c,anchor['assignment'],up0)+max(
            evaluate(c,z,c.actual)['cost']-2.*exact_tangent(c,z,anchor['beta'],low0)
            for z in candidates)-oracle
        assert base_major==pytest.approx(base,abs=1e-7)


def test_convex_epigraph_envelope_is_exact_for_all_response_regions():
    from surgery.anchored_master import envelopes
    for b in (1.,10.,110.,250.):
        v=hinge_response(b,.8,30.)
        s=np.linspace(-300,300,310)
        p,q=v.parts(s)
        p_hat=np.maximum.reduce([a+k*s for a,k in envelopes(v.positive)])
        q_hat=np.maximum.reduce([a+k*s for a,k in envelopes(v.negative)])
        np.testing.assert_allclose(p_hat,p,atol=1e-8)
        np.testing.assert_allclose(q_hat,q,atol=1e-8)


def test_component_preparation_resolves_key_and_retains_all_case_rows():
    from surgery.anchored_master import prepare_components
    from surgery.anchored_core import key
    c = component()
    z = list(map(int, c.original))
    indices = {int(cid): i for i, cid in enumerate(c.case_ids)}
    refs = {key(c): {
        'complete': True,
        'oracle_cost': evaluate(c, z, c.actual)['cost'],
        'references': [{
            'source': 'BOOKED', 'type': 'shared-policy',
            'beta': c.booked.tolist(), 'assignment': z,
            'realized_cost': evaluate(c, z, c.actual)['cost'],
            'predicted_cost': evaluate(c, z, c.booked)['cost']
        }]
    }}
    blocks = prepare_components([c], indices, refs, {})
    assert len(blocks) == 1
    assert blocks[0]['key'] == key(c)
    np.testing.assert_array_equal(blocks[0]['rows'], np.arange(c.n))
    assert blocks[0]['schedules'] == [z]
