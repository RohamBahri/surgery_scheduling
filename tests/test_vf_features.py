"""Outcome-blind feature design and exact baseline compatibility."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from surgery.weekly_data import Week
from surgery.weekly_planner import loads
from surgery.vf_features import (
    feature_names, booked_plan_features, standardize_features)


def example():
    c=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([101,102,103]),("S1","S2"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,1),(1,)),((0,),(0,)),
        np.array([250.,200.,460.]),np.array([10.,30.,40.]),()).validate()
    items=[("toy",c,np.arange(c.n))]
    plans={"toy":{"assignment":[0,0,1]}}
    history=np.array([[.1,.3],[.2,.4],[-.1,.5]])
    return c,items,plans,history


def test_baseline_features_exactly_match_old_implementation():
    c,items,plans,history=example()
    old_p=np.clip((loads(c,[0,0,1],c.booked)[[0,0,1]]-480.)/120.,-2.,2.)
    old=np.column_stack((history[:,0],history[:,1],old_p,
                         history[:,0]*old_p,history[:,1]*old_p))
    got=booked_plan_features(items,c.booked,plans,history,"baseline")
    assert got.shape==(3,5)
    np.testing.assert_array_equal(got,old)
    assert feature_names("baseline")[2]=="draft_congestion"


def test_context_features_are_valid_and_have_expected_interactions():
    c,items,plans,history=example()
    v=booked_plan_features(items,c.booked,plans,history,"context")
    assert v.shape==(3,12)
    assert len(feature_names("context"))==12
    assert v[0,2]==pytest.approx(0.)
    assert v[2,2]==pytest.approx(-20./120.)
    assert v[0,6]==pytest.approx(1.)
    assert v[2,6]==pytest.approx(1.-20./24.)
    assert v[0,8]==pytest.approx(250./450.)
    assert v[1,8]==pytest.approx(200./450.)
    assert v[2,8]==pytest.approx(1.)
    assert v[0,9]==pytest.approx(np.log(2.))
    assert v[2,9]==pytest.approx(0.)
    assert v[2,10]==pytest.approx(v[2,2]*v[2,8])
    assert v[0,11]==pytest.approx(v[0,2]*v[0,9])
    # Realized duration changes must never affect any recommendation feature.
    c.actual[:]=[999.,1.,500.]
    np.testing.assert_array_equal(
        v,booked_plan_features(items,c.booked,plans,history,"context"))


def test_context_standardization_uses_training_only_and_rejects_bad_shape():
    c,items,plans,history=example()
    raw=booked_plan_features(items,c.booked,plans,history,"context")
    train,stats=standardize_features(raw)
    np.testing.assert_allclose(train.mean(axis=0),0.,atol=1e-10)
    again,_=standardize_features(raw,stats)
    np.testing.assert_allclose(again,train)
    with pytest.raises(ValueError,match="schema"):
        standardize_features(raw,stats[:1]+(np.ones(5),))


def test_feature_expansion_cli_options_available():
    runner=Path(__file__).parents[1]/"pilot"/"run_nonmonotone_learning.py"
    p=subprocess.run([sys.executable,str(runner),"--help"],check=True,
                     capture_output=True,text=True)
    assert "--vf-features" in p.stdout
    assert "--vf-lambda-absolute" in p.stdout


def test_feature_comparison_requires_completed_reference(tmp_path):
    entry=Path(__file__).parents[1]/"pilot"/"compare_vf_features.py"
    p=subprocess.run([sys.executable,str(entry),"--base",str(tmp_path)],
                     capture_output=True,text=True)
    assert p.returncode!=0
    assert "selected_policy.json" in p.stderr
