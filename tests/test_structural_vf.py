"""Structural VF: preassigned roster, no draft schedule or realized outcomes."""
import subprocess
import sys

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.vf_features import (structural_features,booked_plan_features,
                                 feature_names,standardize_features)
from pilot.run_nonmonotone_learning import five_features


def sample():
    # S2 and S3 compete for slot 1; S1 is flexible across slots 0 and 1.
    c=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103,104]),("S1","S2","S3"),
        np.array([0,0,1,2]),np.array([0,0,1,1]),
        ((0,1),(1,),(1,)),((0,),(0,),(0,)),
        np.array([200.,250.,180.,210.]),
        np.array([190.,240.,195.,205.]),()).validate()
    items=[("toy",c,np.arange(c.n))]
    history=np.array([[.3,.2],[.4,.5],[.2,.7],[.9,.1]])
    return c,items,history


def test_structural_eligibility_bottleneck_uses_competing_surgeons():
    c,items,h=sample()
    z=structural_features(items,c.booked,h)
    assert z.shape==(c.n,6)
    # Component: 840 minutes booked, 2*(?) turnover minimal 60.
    overall=np.arcsinh((840.+60.-960.)/(120.*2.))
    np.testing.assert_allclose(z[:,0],overall)
    # E={1} must host BOTH S2 and S3; its minimum turnover is 30.
    bottleneck=np.arcsinh((180.+210.+30.-480.)/120.)
    assert z[2,1]==pytest.approx(bottleneck)
    assert z[3,1]==pytest.approx(bottleneck)
    assert z[0,1]==pytest.approx(overall)
    assert z[2,2]==pytest.approx(np.arcsinh((180.-480.)/120.))
    assert z[3,2]==pytest.approx(np.arcsinh((210.-480.)/120.))
    np.testing.assert_allclose(z[:,3],
                              np.log1p([2.,2.,1.,1.]))
    np.testing.assert_allclose(z[:,4],z[:,1]*h[:,0])
    np.testing.assert_allclose(z[:,5],z[:,1]*h[:,1])


def test_features_ignore_provisional_assignment_and_actual_durations():
    c,items,h=sample()
    original=booked_plan_features(items,c.booked,None,h,"structural")
    impossible={"toy":{"assignment":[-99]*c.n}}
    changed=booked_plan_features(items,c.booked,impossible,h,"structural")
    np.testing.assert_array_equal(original,changed)
    c.actual[:]=[999.,1.,500.,8.]
    other=booked_plan_features(items,c.booked,impossible,h,"structural")
    np.testing.assert_array_equal(original,other)
    assert len(feature_names("structural"))==6


def test_runner_bypasses_planner_for_features_fit_and_evaluation():
    c,items,h=sample()
    class Forbidden:
        def solve(self,*args,**kwargs):
            raise AssertionError("Structural features must not solve a plan")
    data=dict(booked=c.booked,actual=c.actual,X=np.ones((c.n,77)))
    ctx=dict(fit=data,held=data,H=h,H_eval=h)
    train,scaler,draft=five_features(ctx,items,Forbidden(),"train",
                                    schema="structural")
    assert draft is None
    ev,_,draft=five_features(ctx,items,Forbidden(),"eval",
                              scaler=scaler,schema="structural")
    assert draft is None
    np.testing.assert_array_equal(train,ev)
    np.testing.assert_allclose(train.mean(axis=0),0,atol=1e-11)


def test_scaler_is_fit_on_training_only():
    c,items,h=sample()
    raw=structural_features(items,c.booked,h)
    normalized,(mu,sd)=standardize_features(raw)
    assert (sd>0).all()
    new=raw.copy()
    new[:,0]+=3.
    test,_=standardize_features(new,(mu,sd))
    np.testing.assert_allclose(test[:,0]-normalized[:,0],3./sd[0])
    with pytest.raises(ValueError,match="schema"):
        standardize_features(new,(mu[:4],sd[:4]))


def test_missing_or_overlapping_rows_fail_closed():
    c,items,h=sample()
    with pytest.raises(ValueError,match="Incomplete"):
        structural_features([],c.booked,h)
    repeated=[("a",c,np.arange(c.n)),("b",c,np.arange(c.n))]
    with pytest.raises(ValueError,match="Overlapping"):
        structural_features(repeated,c.booked,h)


def test_structural_cli_available():
    p=subprocess.run([sys.executable,"-m","pilot.run_structural_vf","--help"],
                     check=True,capture_output=True,text=True)
    assert "--plan-seconds" in p.stdout
    assert "--clean" in p.stdout


def test_frozen_lambda025_legacy_policy_metadata_is_accepted():
    # The October 27 lambda025 artifact has no vf_features field.
    from pilot.run_structural_vf import LAMBDA,validate_frozen_baseline
    legacy=dict(
        method="JOINT_VF",link="saturating",alpha=.8,h=30.,cap=180.,
        vf_loading="uniform",vf_plan_penalty=1.,vf_lambda_scale=.25,
        lambda_vf=LAMBDA,case_coeff=[0.]*77,vf_theta=[0.]*82)
    assert "vf_features" not in legacy
    assert validate_frozen_baseline(legacy)
    legacy["vf_features"]="baseline"
    assert validate_frozen_baseline(legacy)


@pytest.mark.parametrize("key,value",[
    ("vf_features","clean"),
    ("lambda_vf",.5),
    ("link","nonmonotone"),
    ("vf_loading","score"),
    ("vf_lambda_scale",1.),
    ("vf_theta",[0.]*83),
    ("case_coeff",[0.]*76),
])
def test_frozen_lambda025_scientific_mismatches_rejected(key,value):
    from pilot.run_structural_vf import LAMBDA,validate_frozen_baseline
    baseline=dict(
        method="JOINT_VF",link="saturating",alpha=.8,h=30.,cap=180.,
        vf_loading="uniform",vf_plan_penalty=1.,vf_lambda_scale=.25,
        lambda_vf=LAMBDA,case_coeff=[0.]*77,vf_theta=[0.]*82)
    baseline[key]=value
    with pytest.raises(ValueError,match="Frozen baseline mismatch"):
        validate_frozen_baseline(baseline)
