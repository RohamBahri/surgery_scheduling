"""Research-v3 runner, feature audit, and regularization contract tests."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

from surgery.rich_features import RichCorrectionEncoder, REMOVED_SERVICE_FEATURES
from surgery.anchored_master import restricted_value
from run_anchored_extras import _describe
from run_anchored_research_diagnostics import audit_features, response_regions
from test_rich_features import sample

ROOT=Path(__file__).resolve().parents[1]

def test_research_shell_syntax_and_variant_parameters(tmp_path):
    script=ROOT/'run_research_v3.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    mock=tmp_path/'git'
    mock.write_text('#!/bin/sh\necho anchored-research-v3\n')
    mock.chmod(0o755)
    env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
             OUT='artifacts/nonstandard_experiment',INITIAL='LASSO',
             RIDGE='0.001',REFERENCE_PENALTY='0.002',
             LINE_FRACTIONS='1.5,1,0.5',TERMINAL_SEPARATION='1',
             TIGHT_SURROGATE='1')
    run=subprocess.run(['bash',str(script),'train','--dry-run'],env=env,
                       text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    text=run.stdout
    assert '--initial-method LASSO' in text
    assert '--ridge 0.001' in text
    assert '--reference-penalty 0.002' in text
    assert '--terminal-separation' in text
    assert '--tight-surrogate' in text
    assert 'artifacts/nonstandard_experiment' in text
    before=subprocess.run(['bash',str(script),'features','--dry-run'],env=env,
                          text=True,capture_output=True)
    assert before.returncode==0
    assert '--stage features' in before.stdout

def test_offline_service_columns_absent_and_remaining_effects_auditable():
    data=sample(days=8)
    enc=RichCorrectionEncoder()
    fit=enc.fit_transform(data)
    full,statistics=audit_features(enc,data,fit)
    assert full.shape==fit.shape
    assert REMOVED_SERVICE_FEATURES.isdisjoint(enc.names)
    assert 'hist_procedure_mean_abs_error' in statistics
    assert all(np.isfinite(row['lodo_vs_full_rms']) for row in statistics.values())
    assert 'procedure_bias' in enc.names and 'surgeon_bias' in enc.names

def test_response_audit_counts_adoption_without_postoperative_outcomes():
    X=np.array([[1.,0.],[1.,1.],[1.,4.5]])
    w=np.array([0.,20.])
    b=np.array([120.,120.,120.])
    result=response_regions(X,w,b,.8,30.)
    assert result['full_adoption_fraction']==pytest.approx(2/3)
    assert result['attenuation_fraction']==pytest.approx(1/3)
    assert result['rejection_fraction']==0.
    assert result['mean_absolute_effective_correction']>=0.

def test_ridge_and_reference_score_penalties_match_closed_form():
    X=np.array([[1.,0.],[1.,2.]],float)
    b=np.array([120.,120.])
    w=np.array([1.,3.])
    ref=np.array([0.,1.])
    baseline=restricted_value([],X,b,w,.8,30.,2.)
    ridge=restricted_value([],X,b,w,.8,30.,2.,ridge=.2)
    anchored=restricted_value([],X,b,w,.8,30.,2.,
                              reference=.1,reference_w=ref)
    assert ridge-baseline==pytest.approx(.1*9.)
    expected=.1/2*np.mean((X@(w-ref))**2)
    assert anchored-baseline==pytest.approx(expected)
    with pytest.raises(ValueError,match='Reference policy'):
        restricted_value([],X,b,w,.8,30.,2.,reference=.1)


def test_full_offline_history_preserves_informative_features_without_lodo_contrast():
    data=sample(days=10)
    enc=RichCorrectionEncoder(history_mode='full_offline')
    training=enc.fit_transform(data)
    deployment_like=enc.transform(data)
    assert REMOVED_SERVICE_FEATURES.isdisjoint(enc.names)
    for name in ('hist_procedure_mean_abs_error',
                 'hist_surgeon_under_rate',
                 'hist_pair_residual_interaction',
                 'proportional_booked_x_procedure'):
        assert name in enc.names
        k=enc.names.index(name)
        np.testing.assert_allclose(training[:,k],deployment_like[:,k],atol=1e-9)
    # The preserved legacy encoder is LODO, so no assertion that all
    # train/deployment columns are identical or completely label-free.


def test_boosted_raw_preoperative_features_exclude_all_actual_outcomes():
    from run_anchored_extras import _raw_preoperative
    cases=sample(days=7)
    raw=_raw_preoperative(cases)
    altered=cases.copy()
    altered['actual']=999999.
    import pandas as pd
    pd.testing.assert_frame_equal(raw,_raw_preoperative(altered))
    assert all('error' not in s and 'hist_' not in s
               for s in raw.columns)
