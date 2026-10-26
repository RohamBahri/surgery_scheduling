"""Exact SHA-hash anchored reference tests; Gurobi not required."""
from fractions import Fraction as Q
import random
from surgery.parametric_ray import (
    RationalComponent,saturation_endpoint,hash_weight)


def test_saturation_domain_contains_case_and_vf():
    assert saturation_endpoint([0,20,-90],[0,2,-3],30)==Q(5)
    assert saturation_endpoint([100,-100],[1,-1],30)==Q(1)
    assert saturation_endpoint([0],[0],30)==Q(1)
    # Intrinsic plateau beyond S, even with mixed signed projections.
    c=fixture()
    S=saturation_endpoint(c.case_raw,c.projection,c.h)
    assert S==c.upper
    assert c.durations(S)==c.durations(5*S)


def fixture():
    raw=tuple(map(Q,(-42,15,18,-25)))
    proj=tuple(map(Q,(31,-27,26,-19)))
    S=saturation_endpoint(raw,proj,Q(30))
    return RationalComponent(
        allowed=((0,1),)*4,original=(0,0,1,1),
        surgeon=(0,0,1,1),required_days=((),()),
        slots=((0,'OR1'),(0,'OR2')),
        case_ids=(101,102,103,104),
        booked=tuple(map(Q,(250,230,210,150))),
        actual=tuple(map(Q,(230,280,200,190))),
        case_raw=raw,projection=proj,alpha=Q(4,5),h=Q(30),upper=S)


def test_anchor_hash_matches_exact_implementation():
    import hashlib
    c=fixture()
    for j in (0,1):
        for seed in (0,1):
            msg=f'{seed}|101|0|OR{j+1}'.encode()
            expected=1+int.from_bytes(hashlib.sha256(msg).digest()[:8],
                                      'big')%(1<<20)
            assert hash_weight(seed,101,0,f'OR{j+1}')==expected
    for z in c.schedules():
        score=sum(hash_weight(0,c.case_ids[i],*c.slots[z[i]])
                  for i in range(c.n))
        assert c.hash(z,0)==score


def test_exhaustive_closed_ray_and_stable_cells():
    c=fixture()
    result=c.exact_exhaustive()
    assert result['status']=='EXACT_ENUMERATION'
    assert result['value']<=c.follower(Q(0),c.schedules())[2]
    assert result['value']<=c.follower(Q(1),c.schedules())[2]
    assert result['s']<=c.upper
    for atom in result['atoms']:
        if atom['type']=='point':
            assert c.follower(atom['s'],c.schedules())[0]==atom['schedule']
        else:
            a,b=atom['left'],atom['right']
            for k in range(1,10):
                s=a+(b-a)*Q(k,10)
                assert c.follower(s,c.schedules())[0]==atom['schedule']
    assert all(atom['value']%Q(1,4)==0 for atom in result['atoms'])


def test_random_small_components_exact_open_and_boundary():
    rng=random.Random(518)
    for rep in range(10):
        n=rng.randint(2,4)
        raw=tuple(Q(rng.randint(-50,50)) for _ in range(n))
        p=tuple(Q(rng.choice([-1,1])*rng.randint(5,50)) for _ in range(n))
        S=saturation_endpoint(raw,p,Q(30))
        c=RationalComponent(
            allowed=tuple((0,1) for _ in range(n)),
            original=tuple(i%2 for i in range(n)),
            surgeon=tuple(0 for _ in range(n)),
            required_days=((),),slots=((0,'A'),(0,'B')),
            case_ids=tuple(100+i for i in range(n)),
            booked=tuple(Q(rng.randint(80,300)) for _ in range(n)),
            actual=tuple(Q(rng.randint(80,300)) for _ in range(n)),
            case_raw=raw,projection=p,alpha=Q(1,2),h=Q(30),upper=S)
        result=c.exact_exhaustive()
        schedules=c.schedules()
        assert result['value']<=c.follower(0,schedules)[2]
        assert result['value']<=c.follower(1,schedules)[2]
        for atom in result['atoms']:
            if atom['type']=='open':
                for q in (Q(1,17),Q(1,2),Q(16,17)):
                    s=atom['left']+q*(atom['right']-atom['left'])
                    assert c.follower(s,schedules)[0]==atom['schedule']
            else:
                assert c.follower(atom['s'],schedules)[0]==atom['schedule']


def test_floats_are_binary_rationals_not_decimal_rounding():
    from surgery.parametric_ray import rational
    assert rational(.1)==Q.from_float(.1)
    assert rational(.1)!=Q(1,10)


def test_exact_interval_lower_bound_is_valid_on_all_atoms():
    from surgery.parametric_ray import exact_interval_relaxation
    c=fixture()
    schedules=c.schedules()
    for a,b in ((Q(0),c.upper),(Q(0),Q(1)),
                (Q(1),c.upper)):
        lb=exact_interval_relaxation(c,(schedules[0],schedules[-1]),
                                     schedules,a,b)
        solved=c.exact_exhaustive()
        true_min=min(atom['value'] for atom in solved['atoms']
                     if (a<=atom['s']<=b if atom['type']=='point'
                         else atom['left']<b and atom['right']>a))
        assert lb<=true_min


def test_global_sweep_matches_exact_component_enumeration():
    import pytest
    runner=pytest.importorskip('pilot.run_exact_ray_calibration')
    from dataclasses import replace
    c1=fixture()
    c2=replace(c1,actual=tuple(v+Q(10) for v in c1.actual))
    a=c1.exact_exhaustive()
    b=c2.exact_exhaustive()
    out=runner.merge_global({'first':a,'second':b},c1.upper)
    assert out['exact_arithmetic_certificate']
    assert out['best_total_cost'] == (
        c1.follower(out['best_s'],c1.schedules())[2]+
        c2.follower(out['best_s'],c2.schedules())[2])
    for atom in a['atoms']+b['atoms']:
        val=(c1.follower(atom['s'],c1.schedules())[2]+
             c2.follower(atom['s'],c2.schedules())[2])
        assert out['best_total_cost']<=val


def test_exhaustive_respects_surgeday_retention():
    raw=(Q(0),Q(0),Q(0))
    p=(Q(5),Q(-7),Q(3))
    c=RationalComponent(
      allowed=((0,1,2),(0,1,2),(0,1,2)),
      original=(0,1,2),surgeon=(0,0,0),required_days=((0,1),),
      slots=((0,'A'),(0,'B'),(1,'A')),case_ids=(1,2,3),
      booked=(Q(150),Q(220),Q(180)),actual=(Q(200),Q(150),Q(210)),
      case_raw=raw,projection=p,alpha=Q(4,5),h=Q(30),
      upper=saturation_endpoint(raw,p,Q(30)))
    schedules=c.schedules()
    assert schedules
    assert all(any(c.slots[z[i]][0]==0 for i in range(c.n)) and
               any(c.slots[z[i]][0]==1 for i in range(c.n))
               for z in schedules)
    result=c.exact_exhaustive()
    assert result['value']<=c.follower(0,schedules)[2]
    assert result['value']<=c.follower(1,schedules)[2]


def test_same_hash_precheck_skips_costly_lex_variants(monkeypatch):
    from surgery.parametric_ray_mip import NumericOracle
    c=fixture();z=c.schedules()[-1];calls=[]
    oracle=NumericOracle(c,None,seconds=1.)
    def variant(a,b,y,z,kind,k):
        calls.append((kind,k))
        return ('NUMERIC_PASS',None,dict(kind=kind))
    def collision(a,b,y,z,*,point=None):
        calls.append(('collision',None))
        return ('NUMERIC_NONE',None,dict(status='COLLISION_NUMERIC_INFEASIBLE'))
    monkeypatch.setattr(oracle,'_preferred_variant',variant)
    monkeypatch.setattr(oracle,'_same_hash_collision',collision)
    assert oracle.preferred(Q(0),c.upper,z,z)[0]=='NUMERIC_PASS'
    assert calls==[('h0',None),('h1',None),('collision',None)]


def test_lex_fallback_still_checked_when_collision_possible(monkeypatch):
    from surgery.parametric_ray_mip import NumericOracle
    c=fixture();z=c.schedules()[-1];calls=[]
    oracle=NumericOracle(c,None,seconds=1.)
    def variant(a,b,y,z,kind,k):
        calls.append((kind,k))
        return ('NUMERIC_PASS',None,{})
    monkeypatch.setattr(oracle,'_preferred_variant',variant)
    monkeypatch.setattr(oracle,'_same_hash_collision',
        lambda *a,**kw:('POSSIBLE',None,dict(status='COLLISION_EXISTS')))
    assert oracle.preferred(Q(0),c.upper,z,z)[0]=='NUMERIC_PASS'
    assert calls[:2]==[('h0',None),('h1',None)]
    assert any(kind=='lex' for kind,_ in calls)


def test_fixed_rational_boundary_checks(monkeypatch):
    from surgery.parametric_ray_mip import NumericOracle
    c=fixture();z=c.schedules()[0];trace=[]
    oracle=NumericOracle(c,None,seconds=1.)
    def primary(step,y):
        trace.append(('P1',step));return ('NUMERIC_PASS',None,{})
    def preferred(step,y,z):
        trace.append(('P3',step));return ('NUMERIC_PASS',None,{})
    monkeypatch.setattr(oracle,'point_primary',primary)
    monkeypatch.setattr(oracle,'point_preferred',preferred)
    assert oracle.point_verify(Q(1,3),z,z)[0]=='NUMERIC_PASS'
    assert trace==[('P1',Q(1,3)),('P3',Q(1,3))]

def test_full_runner_does_not_shadow_hashlib():
    """Regression for Python UnboundLocalError after data/cache preparation."""
    import inspect
    import symtable
    from pilot import run_exact_ray_calibration as runner

    source=inspect.getsource(runner)
    scope=symtable.symtable(source,runner.__file__,"exec")
    run_scope=next(block for block in scope.get_children()
                   if block.get_name()=="run")
    assert run_scope.lookup("hashlib").is_global()
    assert not run_scope.lookup("hashlib").is_local()
