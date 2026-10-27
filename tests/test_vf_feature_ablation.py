"""Nested congestion-feature ablation, retaining the joint VF objective."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from surgery.vf_features import booked_plan_features, feature_names
from surgery.weekly_data import Week


def example():
    comp=Week("TGH","2012-01-02",((0,"OR1"),(0,"OR2")),
        np.array([1,2,3]),("S1","S2"),
        np.array([0,0,1]),np.array([0,0,1]),
        ((0,1),(1,)),((0,),(0,)),
        np.array([900.,200.,460.]),np.array([901.,202.,470.]),()).validate()
    items=[("c",comp,np.arange(comp.n))]
    plans={"c":{"assignment":[0,0,1]}}
    history=np.array([[.1,.2],[-.3,.4],[.5,.6]])
    return comp,items,plans,history


def test_nested_congestion_variants_are_purely_predictive():
    comp,items,plans,history=example()
    b=booked_plan_features(items,comp.booked,plans,history,"baseline")
    i=booked_plan_features(items,comp.booked,plans,history,"congestion_booked")
    c=booked_plan_features(items,comp.booked,plans,history,"congestion_clip3")
    both=booked_plan_features(items,comp.booked,plans,history,
                              "congestion_booked_clip3")
    assert [b.shape,i.shape,c.shape,both.shape]==[(3,5),(3,6),(3,5),(3,6)]
    np.testing.assert_array_equal(i[:,:5],b)
    np.testing.assert_array_equal(both[:,:5],c)
    assert b[0,2]==pytest.approx(2.)
    assert c[0,2]==pytest.approx(3.)
    assert b[2,2]==pytest.approx(-1./6.)
    assert i[0,5]==pytest.approx((900.-120.)/120.*2.)
    assert both[1,5]==pytest.approx((200.-120.)/120.*3.)
    assert feature_names("congestion_booked")[-1]=="centered_booked_x_congestion"
    comp.actual[:]=[1.,3000.,5.]
    np.testing.assert_array_equal(i,booked_plan_features(
        items,comp.booked,plans,history,"congestion_booked"))


def test_feature_compare_options_with_reference_fixture(tmp_path):
    base=tmp_path/"baseline";base.mkdir()
    (base/"case_policy.json").write_text("{}")
    (base/"selected_policy.json").write_text(json.dumps({
        "link":"saturating","vf_features":"baseline",
        "lambda_vf":.109242}))
    compare=Path(__file__).parents[1]/"pilot"/"compare_vf_features.py"
    for schema in ("congestion_booked","congestion_clip3",
                   "congestion_booked_clip3"):
        out=tmp_path/schema
        p=subprocess.run([sys.executable,str(compare),
            "--base",str(base),"--out",str(out),
            "--vf-features",schema,"--dry-run"],
            capture_output=True,text=True,check=True)
        assert p.stdout.count("--stage all")==1
        assert p.stdout.count("--stage evaluate")==1
        assert p.stdout.count("--stage probe")==1
        assert f"--vf-features {schema}" in p.stdout
        assert "--vf-lambda-absolute 0.109242" in p.stdout
        assert "--link saturating" in p.stdout
