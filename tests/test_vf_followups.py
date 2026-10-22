"""Non-solver mathematical and launch-contract tests for frozen VF follow-ups."""
import inspect
import numpy as np
import pytest
from surgery.weekly_data import Week

def component():
    return Week(group="TGH",monday="2012-01-02",
        slots=((0,"OR1"),(0,"OR2")),
        case_ids=np.array([101,102,103]),surgeon_ids=("S1","S2","S3"),
        surgeon=np.array([0,1,2]),original=np.array([0,1,0]),
        allowed=((0,1),)*3,required_days=((0,),)*3,
        booked=np.array([160.,200.,270.]),actual=np.array([150.,260.,230.]),
        added_days=()).validate()

def test_followups_budget_separate_from_selection_and_policies():
    from pilot import run_vf_followups as r
    assert sum(limit for _,limit in r.STAGES)<11.5*3600
    assert len(set(name for name,_ in r.STAGES))==len(r.STAGES)
    assert set(("diagnose","response","tie","train","evaluate_variants",
                "duration","report"))==set(name for name,_ in r.STAGES)
    assert r.VARIANTS["congestion_only"]["active_features"]==(2,)
    assert 2 not in r.VARIANTS["without_congestion"]["active_features"]
    assert r.VARIANTS["wider_32"]["coefficient_radius"]==32.
    assert r.VARIANTS["observed_outcomes"]["scenario_mode"]=="observed"
    assert "leave_out_TGH" in r.VARIANTS
    assert "scenario_mode" in inspect.signature(r.train_objective).parameters

def test_fixed_display_response_changes_but_does_not_refit_advice():
    from pilot.run_vf_followups import response_case
    from surgery.advice_comparison import display
    c=component()
    target=np.array([175.,185.,277.])
    nominal=response_case(c,c.booked,target,("nominal",.8,30.))
    np.testing.assert_allclose(nominal,target,atol=1e-7)
    weak=response_case(c,c.booked,target,("weak",.4,30.))
    assert np.abs(weak-c.booked).sum()<np.abs(nominal-c.booked).sum()
    assert np.all(np.abs(display(c.booked,target))<=30.00001)
    het=response_case(c,c.booked,target,("heterogeneous",None,None))
    np.testing.assert_allclose(het,response_case(c,c.booked,target,
                              ("heterogeneous",None,None)))
    assert np.all(het>=1.)

def test_slack_decomposition_nonnegative_and_exact_identity():
    from pilot.run_vf_followups import criterion_slack
    c=component()
    bank=np.array([[145.,235.,250.],[155.,300.,220.],
                   [170.,210.,255.],[120.,290.,245.]])
    x=criterion_slack((c,[0,1,0]),np.array([165.,220.,270.]),bank)
    assert x["dependence_gap"]>=-1.e-6
    assert x["split_gap"]>=-1.e-6
    assert x["upper_slack"]==pytest.approx(
             x["dependence_gap"]+x["split_gap"],abs=1e-7)

def test_historical_heterogeneity_is_not_mislabeled_adoption():
    from pilot import run_vf_followups as r
    assert r.RESPONSE_SCENARIOS[-1][0]=="heterogeneous"
    assert r.RESPONSE_SCENARIOS[-1][1:]==(None,None)
    assert 0<r.stable_fraction("surgeon-123")<1
