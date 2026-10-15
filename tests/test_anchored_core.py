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
    from surgery.anchored_core import normalize_tangent
    block={'component':c,'key':'test','rows':np.arange(4),
           'anchor':anchor,'schedules':cases,
           'cuts':[normalize_tangent(c,z,anchor['beta']) for z in cases],
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


def test_single_feasible_schedule_lock_even_with_high_overtime():
    from surgery.anchored_training import locked, unique_schedule
    c=component()
    c.allowed=((0,),(1,))
    c.booked[:]=600.
    assert unique_schedule(c)
    assert locked(c,.8,30.)
    c.allowed=((0,1),(1,))
    assert not unique_schedule(c)


def test_single_case_required_day_forces_one_slot():
    from surgery.anchored_training import unique_schedule
    c=Week('TGH','2012-01-02',((0,'A'),(1,'B'),(0,'C')),
        np.array([10]),('S1',),np.array([0]),np.array([0]),
        ((0,1),),((0,),),np.array([700.]),np.array([680.]),()).validate()
    assert unique_schedule(c)
    c.allowed=((0,1,2),)
    assert not unique_schedule(c)


def test_current_policy_wins_equal_cost_reference_ties():
    from surgery.anchored_master import prepare_components, restricted_value
    from surgery.anchored_core import key, anchored_score, slopes_from_reference
    c=component();z=c.original.astype(int).tolist()
    recorded=evaluate(c,z,c.actual)['cost']
    past={'source':'BOOKED','type':'shared-policy','assignment':z,
          'beta':c.booked.tolist(),'realized_cost':recorded,
          'predicted_cost':evaluate(c,z,c.booked)['cost']}
    refs={key(c):{'complete':True,'references':[past],'oracle_cost':recorded}}
    current_beta=c.booked+np.array([5.,-7.,12.,0.])
    incumbent={key(c):{'complete':True,'assignment':z,
        'realized_cost':recorded,'predicted_cost':evaluate(c,z,current_beta)['cost']}}
    idx={int(cid):i for i,cid in enumerate(c.case_ids)}
    blocks=prepare_components([c],idx,refs,{},current=incumbent,
                               current_duration=current_beta)
    b=blocks[0]
    assert b['anchor']['source']=='CURRENT_POLICY'
    np.testing.assert_allclose(b['anchor']['beta'],current_beta)
    assert z in b['schedules']
    np.testing.assert_allclose(b['anchor']['slopes'],
                               slopes_from_reference(c,z,current_beta))
    X=np.column_stack([np.ones(c.n),np.arange(c.n)])
    w=np.array([10.,2.])
    from surgery.planner import display_and_duration
    _,d=display_and_duration(X@w,c.booked,.8,30.)
    expected=anchored_score(c,b['anchor'],b['cuts'],d,recorded,2.,exact=True)
    assert restricted_value(blocks,X,c.booked,w,.8,30.,2.)==pytest.approx(expected/c.n)


def test_strictly_better_reference_beats_current_and_uses_incumbent_slopes():
    from surgery.anchored_master import prepare_components
    from surgery.anchored_core import key, slopes_from_reference
    c=component()
    schedules=[[i,j,k,l] for i in (0,1) for j in (0,1)
               for k in (0,1) for l in (0,1)]
    options=sorted((evaluate(c,z,c.actual)['cost'],z) for z in schedules)
    best=options[0]; worst=options[-1]
    assert worst[0]>best[0]
    beta=c.booked+np.array([20.,-10.,4.,10.])
    refs={key(c):{'complete':True,'oracle_cost':best[0],
       'references':[{'source':'BOX_CLIP','beta':beta.tolist(),
          'assignment':best[1],'realized_cost':best[0],
          'predicted_cost':evaluate(c,best[1],beta)['cost']}]}}
    curr={key(c):{'complete':True,'assignment':worst[1],
         'realized_cost':worst[0],
         'predicted_cost':evaluate(c,worst[1],c.booked)['cost']}}
    idx={int(cid):i for i,cid in enumerate(c.case_ids)}
    b=prepare_components([c],idx,refs,{},current=curr,
                         current_duration=c.booked)[0]
    assert b['anchor']['source']=='BOX_CLIP'
    assert b['anchor']['assignment']==best[1]
    np.testing.assert_allclose(b['anchor']['slopes'],
            slopes_from_reference(c,worst[1],beta))


def test_deployed_planner_cache_is_duration_keyed(tmp_path,monkeypatch):
    import surgery.anchored_training as training
    c=component()
    count=[0]
    def fake_deployed(week,d,**kwargs):
        count[0]+=1
        z=week.original.astype(int).tolist()
        return {'complete':True,'assignment':z,'realized_cost':
                evaluate(week,z,week.actual)['cost'],
                'predicted_cost':evaluate(week,z,d)['cost']}
    monkeypatch.setattr(training,'deployed',fake_deployed)
    X=np.column_stack([np.ones(c.n),np.arange(c.n)])
    idx={int(cid):i for i,cid in enumerate(c.case_ids)}
    for name in ('current_0','trial_0','current_1'):
        ans=training.verify_policy(tmp_path,name,[c],idx,X,c.booked,
                                   np.array([2.,1.]),.8,30.,workers=1)
        assert all(v['complete'] for v in ans.values())
    assert count[0]==1
    assert len(list(tmp_path.glob('anchored_verify_*.json')))==1


def test_exact_tangent_separation_closes_its_own_learning_cuts(tmp_path):
    pytest.importorskip('gurobipy')
    from surgery.anchored_training import enrich_cuts
    from surgery.anchored_core import (key, slopes_from_reference,
                                       normalize_tangent)
    c=component();a=c.original.astype(int).tolist()
    anchor={'assignment':a,'beta':c.booked,
            'slopes':slopes_from_reference(c,a,c.booked)}
    b={'component':c,'key':key(c),'rows':np.arange(c.n),
       'anchor':anchor,'schedules':[a],
       'cuts':[normalize_tangent(c,a,c.booked)],
       'oracle_cost':evaluate(c,a,c.actual)['cost']}
    lib={}
    out=enrich_cuts(tmp_path,'check',[b],c.booked,2.,lib,
                    seconds=30.,threads=1,workers=1,batch=0)
    assert out['closed']==1 and out['unresolved']==0
    assert out['untested']==0 and out['tested']==1
    assert out['max_dual_cut_gap_tested']<=1e-4
    assert out['all_closed']
    assert lib[key(c)]


def test_tangent_cut_slope_preserved_and_reanchored():
    from surgery.anchored_core import (normalize_tangent, tangent_key,
            reference_fingerprint, anchored_cut)
    c=component()
    z=[0,0,1,1]
    beta=np.array([220.,230.,210.,240.])
    changed=beta+np.array([10.,-5.,0.,2.])
    # Both used room loads equal 480, allowing two supporting slopes.
    custom={'assignment':z,'slopes':[2.75,0.],
            'anchor_sha256':reference_fingerprint(beta)}
    cut=normalize_tangent(c,custom,beta)
    assert cut['slopes']==[2.75,0.]
    alternate=normalize_tangent(c,z,beta)
    assert tangent_key(cut)!=tangent_key(alternate)
    anchor={'assignment':z,'beta':beta,'slopes':np.array([0.,0.])}
    assert anchored_cut(c,anchor,cut,changed,2.,exact=True)!=pytest.approx(
        anchored_cut(c,anchor,alternate,changed,2.,exact=True))
    # A slope valid only at the old kink is NOT a tangent at the new beta.
    new_beta=beta-np.array([20.,20.,0.,0.])
    retangent=normalize_tangent(c,cut,new_beta)
    assert retangent['slopes'][0]==0.


def test_exact_oracle_at_capacity_kink_matches_all_valid_tangents():
    pytest.importorskip('gurobipy')
    from surgery.anchored_oracles import separate
    from surgery.anchored_core import fixed_slope_lower, anchored_cut
    from surgery.weekly_planner import loads
    c=component()
    beta=np.array([220.,230.,210.,240.])
    d=beta+np.array([8.,-2.,-5.,12.])
    res=separate(c,d,beta,2.,exact=True,seconds=60.,threads=1)
    assert res['complete'],res
    brute=-float('inf')
    from itertools import product
    for z in product((0,1),repeat=c.n):
        ld=loads(c,z,beta)
        choices=[]
        for load in ld:
            if load<480-1e-7:choices.append((0.,))
            elif load>480+1e-7:choices.append((2.75,))
            else:choices.append((0.,2.75))
        for slope in product(*choices):
            value=evaluate(c,z,c.actual)['cost']-2.*fixed_slope_lower(c,z,d,slope)
            brute=max(brute,value)
    assert res['lower']==pytest.approx(brute,abs=1e-4)
    assert res['upper']==pytest.approx(brute,abs=1e-4)
    assert len(res['slopes'])==c.k


def test_adaptive_batch_reports_unchecked_components(tmp_path):
    pytest.importorskip('gurobipy')
    from surgery.anchored_training import enrich_cuts
    from surgery.anchored_core import (key, normalize_tangent,
                                        slopes_from_reference)
    c=component()
    z=c.original.astype(int).tolist()
    anchor={'assignment':z,'beta':c.booked,
            'slopes':slopes_from_reference(c,z,c.booked)}
    b={'component':c,'key':key(c),'rows':np.arange(c.n),
       'anchor':anchor,'schedules':[z],
       'cuts':[normalize_tangent(c,z,c.booked)],
       'oracle_cost':evaluate(c,z,c.actual)['cost']}
    # Use an alias key to simulate an independent component in the batch.
    b2=dict(b,key='second')
    out=enrich_cuts(tmp_path,'batch_check',[b,b2],c.booked,2.,{},
                    seconds=30.,threads=1,workers=1,batch=1)
    assert out['tested']==1 and out['untested']==1
    assert out['all_closed'] is False


def test_anchored_schedule_specific_cuts_align_with_master_value():
    from surgery.anchored_core import (normalize_tangent, anchored_score)
    from surgery.anchored_master import restricted_value
    c=component()
    z=c.original.astype(int).tolist()
    beta=c.booked
    cuts=[normalize_tangent(c,zz,beta) for zz in
          (z,[0,1,0,1],[1,1,0,0])]
    anchor={'assignment':z,'beta':beta,'slopes':np.array([0.,0.])}
    X=np.column_stack([np.ones(c.n),np.arange(c.n)])
    w=np.array([6.,2.])
    _,d=display_and_duration(X@w,c.booked,.8,30.)
    oracle=min(evaluate(c,zz,c.actual)['cost'] for zz in
               ([0,0,1,1],[0,1,0,1],[1,1,0,0]))
    block=dict(component=c,key='unit',rows=np.arange(c.n),anchor=anchor,
               cuts=cuts,schedules=[v['assignment'] for v in cuts],
               oracle_cost=oracle)
    value=restricted_value([block],X,c.booked,w,.8,30.,2.)
    expected=anchored_score(c,anchor,cuts,d,oracle,2.,exact=True)/c.n
    assert value==pytest.approx(expected)



def test_learn_accepts_exact_sep_batch_and_completes_outer_loop(tmp_path, monkeypatch):
    """Regression: CLI supplied sep_batch used to raise TypeError immediately."""
    from surgery import anchored_training as training
    from surgery.anchored_core import key
    c=component()
    pieces=c.components()
    X=np.zeros((c.n,74), dtype=float);X[:,0]=1.
    indexed={int(cid):i for i,cid in enumerate(c.case_ids)}
    refs={}
    for part in pieces:
        z=list(map(int,part.original))
        book=part.booked.tolist()
        cost=evaluate(part,z,part.actual)['cost']
        refs[key(part)]={'complete':True,'oracle_cost':0.,
            'references':[dict(source='BOOKED',type='shared-policy',
                assignment=z,beta=book,realized_cost=cost,
                predicted_cost=evaluate(part,z,part.booked)['cost'])]}
    data={'X':X,'booked':c.booked.copy(),'actual':c.actual.copy(),
          'case_id':c.case_ids.copy()}
    recorded=[]
    def verified(output,name,weeks,index,X,booked,w,alpha,h,*args,**kwargs):
        _,d=display_and_duration(X@w,booked,alpha,h)
        result={}
        for part in pieces:
            rows=[index[int(cid)] for cid in part.case_ids]
            z=list(map(int,part.original))
            result[key(part)]=dict(complete=True,assignment=z,
                realized_cost=evaluate(part,z,part.actual)['cost'],
                predicted_cost=evaluate(part,z,d[rows])['cost'],
                realized_overtime_minutes=0.,room_days_over_720=0,
                max_realized_room_load=480.,moved_cases=0)
        return result
    def proposed(blocks,X,booked,w,*args,**kwargs):
        return {'w':np.asarray(w,float).tolist(),'inner':[],
                'restricted_objective':0.}
    def separated(*args,**kwargs):
        recorded.append(kwargs['batch'])
        return {'all_closed':True,'added':0,'closed':len(args[2]),
                'unresolved':0,'tested':len(args[2]),'untested':0}
    monkeypatch.setattr(training,'verify_policy',verified)
    monkeypatch.setattr(training,'pdca',proposed)
    monkeypatch.setattr(training,'enrich_cuts',separated)
    result=training.learn(tmp_path,[c],indexed,data,refs,np.zeros(74),.8,30.,
                          outer=1,iterations=1,closure_rounds=1,
                          workers=1,sep_batch=17)
    assert recorded==[17]
    assert result['complete'] and result['outer']==1
    assert result['feature_dimensions']==74
    assert result['training_cases']==c.n
    assert result['stopping']=='outer_budget'


def test_certify_anchors_at_final_deployed_policy_on_realized_ties(
        tmp_path,monkeypatch):
    """Regression: certify used BOOKED anchors instead of the final policy."""
    import json
    from types import SimpleNamespace
    import run_weekly_anchored as runner
    from surgery.anchored_core import key
    c=component()
    index={int(cid):i for i,cid in enumerate(c.case_ids)}
    X=np.zeros((c.n,74));X[:,0]=1.
    data={'X':X,'booked':c.booked.copy(),'actual':c.actual.copy(),
          'case_id':c.case_ids.copy()}
    refs={}
    for part in c.components():
        z=list(map(int,part.original))
        real=evaluate(part,z,part.actual)['cost']
        refs[key(part)]={'complete':True,'oracle_cost':0.,
             'references':[dict(source='BOOKED',type='shared-policy',
                  assignment=z,beta=part.booked.tolist(),realized_cost=real,
                  predicted_cost=evaluate(part,z,part.booked)['cost'])]}
    (tmp_path/'anchored_references.json').write_text(json.dumps(refs))
    w=[0.]*74;w[0]=20.
    (tmp_path/'anchored_progress.json').write_text(json.dumps(dict(
        complete=True,w=w,library={})))
    monkeypatch.setattr(runner,'initialize',
        lambda args:(None,None,[c],index,data,None,None))
    hits=[]
    def verified(output,name,weeks,index,X,booked,w,alpha,h,**kwargs):
        hits.append(name)
        _,d=display_and_duration(X@w,booked,alpha,h)
        result={}
        for part in c.components():
            rows=[index[int(cid)] for cid in part.case_ids]
            z=list(map(int,part.original))
            result[key(part)]=dict(complete=True,assignment=z,
                realized_cost=evaluate(part,z,part.actual)['cost'],
                predicted_cost=evaluate(part,z,d[rows])['cost'])
        return result
    monkeypatch.setattr(runner,'verify_policy',verified)
    anchors=[]
    def certificate(task):
        blocks, *_ =task
        anchors.extend((b['anchor']['source'],
                        b['anchor']['beta'].tolist()) for b in blocks)
        return {b['key']:{'certified':True,'separation_exact':True,
                          'regret':1.,'tie_safe_upper':5.} for b in blocks}
    monkeypatch.setattr(runner,'_certificate_week',certificate)
    args=SimpleNamespace(output=tmp_path,report_gap=False,
         workers=1,seconds=10.,threads=1,alpha=.8,h=30.,gamma=2.)
    runner.certify(args)
    assert hits==['certification_incumbent']
    assert anchors and all(src=='CURRENT_POLICY' for src,_ in anchors)
    assert any(not np.allclose(beta,c.booked[:len(beta)])
               for _,beta in anchors)
    assert (tmp_path/'anchored_bound_summary.json').exists()


def test_oracle_max_bound_audit_accepts_roundoff_but_not_material_violations():
    from surgery.anchored_oracles import reconcile_maximization_bounds
    # Gurobi's MAX ObjBound can be marginally below a reconstructed feasible
    # incumbent without indicating a mathematically invalid model.
    a=reconcile_maximization_bounds(1020.000004, 1020.0, 1020.0)
    assert a['valid'] and a['reason']=='numerically_reconciled'
    assert a['upper'] >= a['checked']
    assert 0. < a['dual_adjustment'] < a['numerical_tolerance']
    b=reconcile_maximization_bounds(1020.0, 1020.0, 1020.25)
    assert b['valid'] and b['upper']==1020.25
    # Large disagreements are explicitly unresolved, never converted into
    # a reported dual certificate or a silently inflated upper bound.
    c=reconcile_maximization_bounds(1020.10, 1020.0, 1020.0)
    assert not c['valid'] and c['upper'] is None
    assert c['reason']=='objective_reconstruction_mismatch'
    d=reconcile_maximization_bounds(1020.10, 1020.1, 1020.0)
    assert not d['valid'] and d['upper'] is None
    assert d['reason']=='material_dual_bound_violation'
    e=reconcile_maximization_bounds(1020.0, float('nan'), 1020.1)
    assert not e['valid'] and e['upper'] is None


def test_fresh_experiment_rejects_mismatched_source_without_migration(
        tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import run_weekly_anchored as runner
    artifacts=tmp_path/'baselines';artifacts.mkdir()
    workbook=tmp_path/'data.xlsx';workbook.write_bytes(b'fixed fake workbook')
    names=['intercept']+[f'feature_{i}' for i in range(73)]
    meta={'feature_names':names,
          'matrix_sha256':{'train':'frozen-train','validation':'frozen-validation'}}
    origin={'workbook_sha256':runner.sha(workbook),
            'matrix_sha256':meta['matrix_sha256'],
            'feature_code_sha256':'fixed-features','alpha':.8,'h':30.}
    (artifacts/'learning_settings.json').write_text(json.dumps(origin))
    for fname,model in (
            ('model_case.json',{'settings':origin,'w':[0.]*74}),
            ('vf_progress.json',{'settings':origin,'w':[0.]*74,'complete':True}),
            ('lasso_policy.json',{'features':names,'weights':[0.]*74})):
        (artifacts/fname).write_text(json.dumps(model))
    train={'case_id':np.arange(18705,dtype=int),
           'X':np.ones((18705,74),dtype=float),
           'booked':np.full(18705,100.,dtype=float),
           'actual':np.full(18705,105.,dtype=float)}
    frames={'train':train,'validation':{'case_id':np.arange(5779,dtype=int)}}
    weeks=[SimpleNamespace(group='TGH',case_ids=np.array([i]),n=1)
           for i in range(60)]
    monkeypatch.setattr(runner,'load_matrices',lambda root:(meta,frames))
    monkeypatch.setattr(runner,'load_weeks',lambda *args:weeks)
    monkeypatch.setattr(runner,'source_fingerprint',lambda :'fixed-features')
    identity={'trainer_source_sha256':'new-reviewed-implementation',
              'feature_source_sha256':'fixed-features',
              'matrix_sha256':meta['matrix_sha256'],
              'site':'TGH','gamma':2.,'max_outer':15}
    monkeypatch.setattr(runner,'settings',lambda *args:dict(identity))
    output=tmp_path/'clean';output.mkdir()
    previous=dict(identity,trainer_source_sha256='exploratory-code')
    (output/'anchored_settings.json').write_text(json.dumps(previous))
    args=SimpleNamespace(output=output,artifacts=artifacts,workbook=workbook,
                         gap_model=None,initial_policy=None,site='TGH',alpha=.8,h=30.)
    with pytest.raises(ValueError,match='NEW output directory'):
        runner.initialize(args)
    assert json.loads((output/'anchored_settings.json').read_text())==previous
    assert not (output/'anchored_source_migration.json').exists()
    (output/'anchored_settings.json').unlink()
    result=runner.initialize(args)
    assert len(result[2])==60
    assert json.loads((output/'anchored_settings.json').read_text())==identity
