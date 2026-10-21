"""Pure-math verification; licensed-Gurobi integration tests are run on user's host."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import pytest

from surgery.theory_exact import (CAP, TURN, KAPPA, coefficient_box_images,
                                   stop_loss, reduced, expected_reduced)


@pytest.fixture
def fake_planner(monkeypatch):
    from surgery import theory_exact
    wp=ModuleType('surgery.weekly_planner')
    def loads(c,z,d):
        z=np.asarray(z,dtype=int)
        count=np.bincount(z,minlength=c.k)
        return np.bincount(z,weights=np.asarray(d)+TURN,minlength=c.k)-TURN*(count>0)
    wp.loads=loads
    fake_asdbs=ModuleType('surgery.asdbs')
    def mean_cost(c,z,bank):
        vals=[]
        for d in bank:
            v=loads(c,z,d)
            vals.append(np.maximum(CAP-v,0).sum()+1.75*np.maximum(v-CAP,0).sum())
        return float(np.mean(vals))
    fake_asdbs.mean_cost=mean_cost
    monkeypatch.setitem(sys.modules,'surgery.weekly_planner',wp)
    monkeypatch.setitem(sys.modules,'surgery.asdbs',fake_asdbs)
    return loads,mean_cost


def test_expected_score_exact_reduction(fake_planner):
    loads,mean_cost=fake_planner
    c=SimpleNamespace(n=4,k=3)
    z=[0,0,1,2]
    bank=np.array([[400.,150.,220.,460.],[500.,180.,300.,380.]],float)
    score=expected_reduced(c,z,bank)
    assert score==pytest.approx(np.mean([reduced(c,z,a) for a in bank]),abs=1e-8)
    assert score==pytest.approx(mean_cost(c,z,bank)-CAP*c.k+bank.mean(axis=0).sum()+TURN*c.n)


def test_schedule_comparison_lipschitz(fake_planner):
    rng=np.random.default_rng(42)
    c=SimpleNamespace(n=6,k=3)
    z=np.array([0,0,1,1,2,2]); zz=np.array([0,1,1,2,2,0])
    for _ in range(1000):
        x=rng.uniform(20,600,size=6);y=rng.uniform(20,600,size=6)
        difference=(reduced(c,z,x)-reduced(c,zz,x))-(reduced(c,z,y)-reduced(c,zz,y))
        assert abs(difference)<=KAPPA*np.abs(x-y).sum()+1e-8


def test_box_images_cover_all_coefficients():
    rng=np.random.default_rng(17)
    n,p=60,5
    H=rng.normal(size=(n,p))
    base=rng.uniform(100,200,size=n)
    low=np.array([-5,-9,-2,-11,1.]);high=np.array([2,3,6,-1,10.])
    dlo=base-12;dhi=base+15
    lower,upper=coefficient_box_images(base,H,low,high,dlo,dhi)
    for _ in range(1000):
        theta=rng.uniform(low,high)
        d=np.clip(base+H@theta,dlo,dhi)
        assert (lower<=d+1e-8).all()
        assert (d<=upper+1e-8).all()


def test_finite_support_stoploss_is_convex_pwl():
    rng=np.random.default_rng(0)
    bank=rng.uniform(10,600,size=(50,4))
    for i in range(4):
        samples=bank[:,i];support=np.unique(samples)
        for d in np.linspace(1,800,100):
            truth=np.maximum(samples-d,0).mean()
            cuts=[0.,samples.mean()-d]
            cuts += [np.maximum(samples-a,0).mean() - (samples>a).mean()*(d-a) for a in support]
            assert max(cuts)==pytest.approx(truth,abs=1e-8)


def test_sharp_distributional_regret_reduced(fake_planner):
    c=SimpleNamespace(n=4,k=2)
    z=np.array([0,0,1,1]);zz=np.array([0,1,0,1])
    rng=np.random.default_rng(3)
    X=rng.uniform(50,500,size=(100,4))
    d=np.mean(X,axis=0)
    gap=np.mean([reduced(c,z,a)-reduced(c,zz,a) for a in X])
    assert gap==pytest.approx(expected_reduced(c,z,X)-expected_reduced(c,zz,X))
    # Upper regret bound from the stop-loss form, for the minimum-phi plan.
    value=min(reduced(c,z,d),reduced(c,zz,d))
    vf=value+KAPPA*stop_loss(d,X)
    expected=min(expected_reduced(c,z,X),expected_reduced(c,zz,X))
    assert vf>=expected-1e-8


def test_equal_room_count_imbalance_identity(fake_planner):
    loads,_=fake_planner
    c=SimpleNamespace(n=6,k=3)
    z=(0,0,1,1,2,2);z2=(0,1,1,2,0,2)
    rng=np.random.default_rng(91)
    for _ in range(1000):
        a=rng.uniform(20,700,size=c.n)
        def imbalance(z):
            l=loads(c,z,a)-CAP
            return min(np.maximum(l,0).sum(),np.maximum(-l,0).sum())
        lhs=reduced(c,z,a)-reduced(c,z2,a)
        rhs=KAPPA*(imbalance(z)-imbalance(z2))
        assert lhs==pytest.approx(rhs,abs=1e-7)