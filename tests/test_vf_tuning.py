"""Focused VF-tuning regressions without patient data or Gurobi."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from surgery.nonmonotone_learning import (
    DCObjective, ResponseDC, mm_vf, response, score_loadings)


def setup_objective():
    booked=np.array([140.,100.,180.,125.])
    actual=np.array([150.,110.,155.,129.])
    X=np.column_stack((np.ones(4),[-1.,1.,.5,-.5],[.2,-1.,2.,.3]))
    rooms=sparse.csr_matrix(
        (np.ones(4),([0,0,1,1],[0,1,2,3])),shape=(2,4))
    link=ResponseDC(booked,.8,30.,180.,1.,link="saturating")
    obj=DCObjective(booked,X,np.zeros(4),link,mode="VF",
        actual=actual,components=2,rooms=rooms,lam=.4,
        penalize=np.array([0.,1.,1.]))
    items=[("a",None,np.array([0,1])),("b",None,np.array([2,3]))]
    return obj,items


def test_score_loadings_are_positive_normalized_and_clustered():
    obj,items=setup_objective()
    weights=score_loadings(obj,np.array([3.,2.,-4.]),items)
    assert weights.shape==(3,)
    assert weights[0]==0.
    assert (weights[1:]>0).all()
    assert weights[1:].mean()==pytest.approx(1.)


def test_score_loadings_are_rescaling_equivariant():
    obj,items=setup_objective()
    start=np.zeros(3)
    weights=score_loadings(obj,start,items)
    X=obj.X.copy();X[:,1]*=3.
    rescaled=DCObjective(obj.booked,X,obj.base,obj.map,
        mode="VF",actual=obj.actual,components=obj.components,
        rooms=obj.rooms,lam=obj.lam,penalize=obj.penalize)
    new=score_loadings(rescaled,start,items)
    assert new[1]/new[2]==pytest.approx(3.*weights[1]/weights[2])


def test_mm_vf_rejects_invalid_weight_vectors(tmp_path):
    obj,items=setup_objective()
    for penalty in (np.array([0.,-1.,1.]),np.ones(2),
                    np.array([1.,1.,1.])):
        with pytest.raises(ValueError,match="penalty"):
            mm_vf(items,obj.booked,obj.X,obj.map,obj.actual,
                start=np.zeros(3),root=tmp_path,lam=.1,
                initial_libraries={},solve_plans=lambda *a: {},
                penalize=penalty)


def test_saturating_training_matches_frozen_three_regime_deployment():
    booked=np.array([7.,100.,160.])
    c=ResponseDC(booked,.8,30.,180.,1.,link="saturating")
    raw=np.array([-150.,12.,80.])
    d=c.reviewed(raw)
    deployed=np.maximum(1.,booked+response(c.display(raw),.8,30.))
    assert np.allclose(deployed,d,rtol=0,atol=1e-9)


def test_only_public_runner_exposes_structural_configuration():
    script=Path(__file__).parents[1]/"pilot"/"run_joint_vf.py"
    p=subprocess.run([sys.executable,"-m","pilot.run_joint_vf","--help"],
                     cwd=script.parents[1],capture_output=True,text=True,check=True)
    for flag in ("--stage","--alpha","--h","--link","--lambda-vf",
                 "--true-alpha","--true-h","--true-link","--seeds"):
        assert flag in p.stdout
    assert "--vf-features" not in p.stdout
    assert "--vf-loading" not in p.stdout
