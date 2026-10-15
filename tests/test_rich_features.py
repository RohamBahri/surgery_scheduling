"""Rich correction features: fixed booked anchor, LODO, and cold-start behavior."""
import numpy as np
import pandas as pd

from surgery.rich_features import RichCorrectionEncoder, _counts, _lookup, REMOVED_SERVICE_FEATURES
from run_weekly_rich import windows


def sample(days=5, per_day=16):
    rng = np.random.default_rng(4871)
    n = days * per_day
    date = np.repeat(pd.bdate_range("2012-04-02", periods=days), per_day)
    procedure = rng.choice(["P1", "P2", "P3", "P4"], n)
    service = np.where(np.isin(procedure, ["P1", "P2"]), "GEN", "CARD")
    booked = rng.choice([60., 120., 180., 240.], n)
    actual = booked + np.where(service == "GEN", -12., 8.) + rng.normal(0, 16, n)
    return pd.DataFrame({
        "case_id": np.arange(2, n + 2), "date": date,
        "group": np.where(service == "GEN", "TWH-main", "TGH"),
        "service": service, "procedure": procedure,
        "surgeon": rng.choice(["S1", "S2", "S3", "S4"], n),
        "booked": booked, "actual": np.maximum(actual, 1.),
        "patient_type": np.where(booked < 150, "Same Day", "Inpatient"),
        "dtt": date - pd.to_timedelta(10, unit="D"),
        "consult": date - pd.to_timedelta(40, unit="D"),
    })


def test_historical_subtraction_excludes_all_own_day_cases():
    frame = sample(days=4)
    full = _counts(frame, ("service",))
    dated = _counts(frame, ("service", "date"))
    found = _lookup(frame, full, ("service",), dated=dated)
    for i, case in frame.iterrows():
        others = frame[(frame.service == case.service) & (frame.date != case.date)]
        assert found.iloc[i].n == len(others)
        assert np.isclose(found.iloc[i].err_sum,
                          (others.actual - others.booked).sum())


def test_fixed_anchor_and_no_test_outcome_leakage():
    data = sample()
    training = data[data.date < pd.Timestamp("2012-04-06")]
    test = data[data.date >= pd.Timestamp("2012-04-06")]
    enc = RichCorrectionEncoder()
    X = enc.fit_transform(training)
    Xt = enc.transform(test)
    assert X.shape[1] == Xt.shape[1] == len(enc.names)
    assert enc.names[0] == "intercept"
    assert "booked" not in enc.names
    assert any(n.startswith("interaction_booked_x_service:") for n in enc.names)
    assert np.isfinite(X).all() and np.isfinite(Xt).all()
    changed = test.copy()
    changed["actual"] = 100000.
    np.testing.assert_allclose(enc.transform(changed), Xt)
    np.testing.assert_array_equal(training.booked + X @ np.zeros(X.shape[1]),
                                  training.booked)


def test_own_date_outcome_changes_do_not_alter_historical_features():
    data = sample()
    enc = RichCorrectionEncoder()
    enc.fit_transform(data)
    f = enc._prepare(data)
    before = enc._history(f, train_mode=True)
    chosen = f.date.min()
    changed = f.copy()
    changed.loc[changed.date.eq(chosen), "actual"] += 1000
    enc._summaries(changed)
    after = enc._history(f, train_mode=True)
    selected = f.date.eq(chosen).to_numpy()
    for key in before:
        np.testing.assert_allclose(before[key][selected], after[key][selected])


def test_unseen_service_procedure_surgeon_finite():
    training = sample()
    unknown = training.iloc[:3].copy()
    unknown["case_id"] += 5000
    unknown["service"] = "NEVER_SEEN"
    unknown["procedure"] = "NEVER_SEEN"
    unknown["surgeon"] = "NEVER_SEEN"
    enc = RichCorrectionEncoder()
    enc.fit_transform(training)
    assert np.isfinite(enc.transform(unknown)).all()


def test_window_selection_uses_no_2013_test_rows():
    groups = ("TGH", "TWH-main", "TWH-day")
    rows = []
    for group in groups:
        for k in range(7):
            rows.append(dict(case_id=len(rows)+2, group=group,
                             date=pd.Timestamp("2012-01-02") + pd.Timedelta(days=7*k),
                             split="train"))
        rows.append(dict(case_id=len(rows)+2, group=group,
                         date=pd.Timestamp("2013-01-02"), split="test"))
    tr, val, tkeys, vkeys = windows(pd.DataFrame(rows), 0, 4, 3)
    assert len(tr) == 12 and len(val) == 9
    assert set(tr.case_id).isdisjoint(val.case_id)
    assert len(tkeys) == 12 and len(vkeys) == 9
    assert tr.date.max() < val.date.min()


def test_service_history_artifact_columns_removed_without_changing_base_encoder():
    full=sample(days=8)
    enc=RichCorrectionEncoder()
    X=enc.fit_transform(full)
    Xv=enc.transform(full.iloc[:8].copy())
    assert REMOVED_SERVICE_FEATURES.isdisjoint(enc.names)
    assert X.shape[1]==Xv.shape[1]==len(enc.names)
    assert 'procedure_bias' in enc.names
    assert 'surgeon_bias' in enc.names
    assert 'proportional_booked_x_service' in enc.names
    assert len(enc.names)>=10
